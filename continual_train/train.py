"""Continual learning orchestration, training loop and track routing."""
from __future__ import annotations
import random
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from dataclasses import replace
import ray
from tqdm.auto import tqdm
from config import Config, parse_config
from data_load import DataLoader, PromptFormatter
from distributed import launch_ray, shutdown_ray, BatchRouter, await_workers_progress
from continual_train.storage import (choose_baseline, evaluation_fingerprint,
                                     restore_from_logged_counts, PrivateTaskCheckpoints)
from continual_train.evaluate import load_domain_data, evaluate_domains
from continual_train import evaluate as benchmark
import visualization
import diagnostics as trace


SIGNATURE_FIELDS = (
    "parallel_mode",
    "hf_continual_repo",
    "hf_model_repo",
    "model_name",
    "model_revision",
    "model_dtype",
    "train_algorithm",
    "train_domains",
    "train_epochs",
    "train_batch_size",
    "train_max_steps",
    "train_seed",
    "train_shuffle",
    "train_lr",
    "train_weight_decay",
    "train_grad_clip",
    "ray_num_workers",
    "grpo_group_size",
    "grpo_max_prompt_length",
    "grpo_max_completion_length",
    "grpo_updates_per_rollout",
    "grpo_temperature",
    "grpo_top_p",
    "grpo_clip_range",
    "continual_eval_samples_per_worker",
    "continual_eval_limit",
)


class ContinualTrainer:
    """One experiment; all per-task operations share the same configuration."""

    def __init__(self, config: Config):
        self.config = config

    def train_domain(
        self,
        workers,
        domain: str,
        records: list[dict],
        task_id: int,
    ) -> int:
        config = self.config
        batch_size = (config.train_finance_batch_size if domain == "finance"
                      else config.train_batch_size)
        accumulation = (config.train_finance_gradient_accumulation if domain == "finance"
                        else 1)
        if accumulation > 1 and config.grpo_updates_per_rollout != 1:
            raise ValueError("Finance gradient accumulation requires RL_GRPO_UPDATES_PER_ROLLOUT=1")
        router = BatchRouter(config, len(workers), batch_size)
        samples_per_step = router.global_size * accumulation
        print(f"[TRAIN CONFIG] {domain}: batch={batch_size}, accumulation={accumulation}, "
              f"prompts/optimizer_step={samples_per_step}", flush=True)

        if len(records) < samples_per_step:
            raise ValueError(
                f"{domain}: insufficient training samples"
            )

        sampler = None
        if config.train_sampler != "random":
            from sampling import StepDistanceSampler, embed_entire_task
            ray.get([w.cleanup_gpu.remote() for w in workers])
            features = embed_entire_task(workers, config, records, domain)
            sampler = StepDistanceSampler(
                features, config.train_sampler,
                seed=config.train_seed + task_id * 1000,
                k=config.sampling_knn_k,
            )
            if sampler.explained_variance_ratio is not None:
                print(f"[PCA KNN] explained_variance_ratio="
                      f"{sampler.explained_variance_ratio}", flush=True)
            del features
            ray.get([w.cleanup_gpu.remote() for w in workers])

        steps = 0
        full_steps_per_epoch = len(records) // samples_per_step
        planned_steps = full_steps_per_epoch * config.train_epochs
        if config.train_max_steps is not None:
            planned_steps = min(planned_steps, config.train_max_steps)
        planned_samples = planned_steps * samples_per_step

        with tqdm(
            total=planned_samples,
            desc=f"Task {task_id}/{len(config.train_domains)} TRAIN {domain}",
            unit="sample", dynamic_ncols=True, position=0, leave=True,
            disable=not config.progress_bar,
        ) as bar:
            for epoch in range(config.train_epochs):
                if sampler is not None:
                    sampler.reset()
                indices = list(range(len(records)))

                if config.train_shuffle:
                    random.Random(
                        config.train_seed
                        + task_id * 1000
                        + epoch
                    ).shuffle(indices)

                usable = (
                    len(indices) // samples_per_step
                ) * samples_per_step
                if config.trace_mode and epoch == 0 and sampler is None:
                    trace.choose(config, records, "train", domain, indices[:min(5, usable)])

                for start in range(
                    0, usable, samples_per_step
                ):
                    selected = (sampler.sample(samples_per_step) if sampler is not None
                                else indices[start:start + samples_per_step])
                    if config.trace_mode and epoch == 0 and steps == 0 and sampler is not None:
                        trace.choose(config, records, "train", domain, selected[:5])

                    if config.trace_mode:
                        step_start = time.perf_counter()
                    for micro in range(accumulation):
                        local_indices = selected[
                            micro * router.global_size:(micro + 1) * router.global_size
                        ]
                        selected_batch = [records[i] for i in local_indices]
                        samples = router.split(selected_batch)
                        if config.trace_mode:
                            trace.batch(config, "train.dispatched", selected_batch,
                                        epoch=epoch + 1, step=steps + 1,
                                        microbatch=micro + 1)
                        requests = [
                            worker.rollout_batch.remote(domain, part, micro, accumulation)
                            for worker, part in zip(workers, samples)
                        ]
                        results = await_workers_progress(
                            config, requests, "train.rollout", bar=bar,
                            status=f"step {steps + 1}/{planned_steps} micro {micro + 1}/{accumulation}",
                            no_timeout=True, domain=domain, step=steps + 1,
                        )
                    if config.trace_mode:
                        step_seconds = time.perf_counter() - step_start
                        trace.log(
                            config, "train.step.duration",
                            domain=domain,
                            epoch=epoch + 1,
                            step=steps + 1,
                            seconds=round(step_seconds, 3),
                            workers=len(workers),
                            samples=samples_per_step,
                        )
                        for rank, (part, result) in enumerate(zip(samples, results)):
                            if config.parallel_mode != "tp" or rank == 0:
                                trace.batch(config, "train.finished", part, rank=rank, metrics=result)

                    steps += 1
                    bar.update(samples_per_step)
                    bar.set_postfix_str(
                        f"step={steps}/{planned_steps} epoch={epoch + 1}/{config.train_epochs} "
                        f"dataset={len(records)} group={config.grpo_group_size}",
                        refresh=True,
                    )

                    if not config.progress_bar and (steps == 1 or steps % 10 == 0):
                        print(
                            f"[TRAIN] {domain} "
                            f"epoch={epoch + 1} "
                            f"step={steps} "
                            f"{results[0]}"
                        )

                    if (
                        config.train_max_steps is not None
                        and steps >= config.train_max_steps
                    ):
                        return steps

        return steps


    @staticmethod
    def save_domain_eval(progress, root, hub, entry, domain, score):
        entry["accuracy"] = {**(entry.get("accuracy") or {}), domain: score}
        hub.publish_progress(root, progress)

    @staticmethod
    def restore_logged_eval(config, progress, evaluation, hub, root):
        """Import known complete domain scores only for the exact HF checkpoint."""
        if not config.eval_restore_counts or not progress.get("history"):
            return
        entry = progress["history"][-1]
        if entry.get("checkpoint_revision") != config.eval_restore_revision:
            print("[EVAL RESTORE] revision differs; not importing log counts", flush=True)
            return
        fingerprint = evaluation_fingerprint(evaluation)
        if (progress.get("baseline_fingerprint") and
                progress["baseline_fingerprint"] != fingerprint):
            raise ValueError("Eval restore fingerprint differs from saved baseline")
        counts = json.loads(config.eval_restore_counts)
        if not isinstance(counts, dict):
            raise ValueError("RL_EVAL_RESTORE_COUNTS must be a JSON object")
        scores = dict(entry.get("accuracy") or {})
        for domain, pair in counts.items():
            if (domain not in evaluation or not isinstance(pair, list) or len(pair) != 2):
                raise ValueError(f"Invalid restored eval domain: {domain}")
            correct, total = pair
            if (type(correct) is not int or type(total) is not int or
                    total != len(evaluation[domain]) or not 0 <= correct <= total):
                raise ValueError(f"Invalid restored eval count for {domain}")
            result = {"correct": correct, "total": total, "accuracy": correct / total}
            if domain in scores and scores[domain] != result:
                raise ValueError(f"HF score conflicts with restored {domain}")
            scores[domain] = result
        if entry.get("accuracy") != scores:
            entry["accuracy"] = scores
            hub.publish_progress(root, progress)
            print(f"[EVAL RESTORED TO HF] {entry['domain']}: {list(counts)}", flush=True)

    def run(self):
        config = self.config
        # Each call uses an independently instantiated model and optimizer.
        config.resolve_hf_repos()
        if config.trace_mode:
            trace.start(config)

        if (
            config.train_algorithm == "grpo"
            and "coding" in config.train_domains
        ):
            raise NotImplementedError(
                "Coding GRPO requires a verified reward"
            )

        root = Path(config.continual_checkpoint_dir).resolve()
        hub = PrivateTaskCheckpoints(config)  # Creates / verifies a PRIVATE model repo.
        signature = {name: getattr(config, name) for name in SIGNATURE_FIELDS}
        if config.train_sampler != "random":
            # Original GRPO signature remains byte-for-byte backward compatible.
            for key in ("train_sampler", "sampling_projection_dim",
                        "sampling_embed_batch_size", "sampling_max_length",
                        "sampling_knn_k", "sampling_projector_seed",
                        "train_finance_batch_size", "train_finance_gradient_accumulation"):
                signature[key] = getattr(config, key)

        # Only HF progress.json is authoritative. No local baseline fallback.
        remote_progress = hub.load_progress()
        progress = remote_progress if remote_progress is not None else {
            "signature": signature, "history": [], "metrics": [],
        }

        # Task-0-only baseline is independent of the number of future training steps.
        # Permit changing just train_max_steps before *any* completed task; all
        # evaluation/model/data settings remain protected by signature + fingerprint.
        signature_changed_for_baseline_only = False
        saved_signature = progress.get("signature", {})
        if saved_signature != signature:
            changed = {
                field: (saved_signature.get(field), signature.get(field))
                for field in sorted(set(saved_signature) | set(signature))
                if saved_signature.get(field) != signature.get(field)
            }
            can_retarget_untrained_baseline = (
                not progress.get("history")
                and progress.get("baseline_accuracy") is not None
                and config.baseline_mode == "reuse"
                and set(changed) == {"train_max_steps"}
            )
            if can_retarget_untrained_baseline:
                print("[BASELINE-ONLY RESUME] train_max_steps changed "
                      f"{changed['train_max_steps'][0]} -> {changed['train_max_steps'][1]}; "
                      "no completed tasks. Original baseline will be revalidated.", flush=True)
                progress["signature"] = signature
                signature_changed_for_baseline_only = True
            else:
                differences = ", ".join(
                    f"{key}: {old_value!r} -> {new_value!r}"
                    for key, (old_value, new_value) in changed.items()
                )
                raise ValueError(
                    "Resume configuration differs from the private Hub run: "
                    + differences
                    + ". Keep the original settings, or start a new Hub repo + checkpoint."
                )
        if config.baseline_mode == "restore" and progress["history"]:
            raise ValueError("Cannot restore baseline onto a trained HF checkpoint history")
        if progress["history"] and not config.continual_resume:
            raise ValueError("Private Hub run found; enable resume or choose a new model repo")

        loader = DataLoader(config)
        formatter = PromptFormatter()

        # Load test data once from the private Hub.
        evaluation = {}

        for domain in config.train_domains:
            if config.trace_mode:
                trace.log(config, "hub.test.start", domain=domain)
            records = load_domain_data(loader, formatter, domain, "test")
            if config.trace_mode:
                trace.log(config, "hub.test.finished", domain=domain, records=len(records))

            if config.continual_eval_limit is not None:
                records = records[
                    :config.continual_eval_limit
                ]

            if not records:
                raise ValueError(
                    f"Empty evaluation domain: {domain}"
                )

            evaluation[domain] = records
            if config.trace_mode:
                trace.choose(config, records, "eval", domain)

        # One-time HF-only import of finished Task-0 counts from the terminal log.
        # No GPU, Ray, or new model inference is started in restore mode.
        if config.baseline_mode == "restore":
            if not config.benchmark_forward_transfer:
                raise ValueError("restore requires RL_BENCHMARK_FORWARD_TRANSFER=true")
            counts = restore_from_logged_counts(config, evaluation)
            previous = progress.get("baseline_accuracy")
            if previous is not None and previous != counts:
                raise ValueError("HF baseline already differs; refusing overwrite")
            fingerprint = evaluation_fingerprint(evaluation)
            if progress.get("baseline_fingerprint") not in (None, fingerprint):
                raise ValueError("HF baseline fingerprint differs; refusing overwrite")
            progress["baseline_accuracy"] = counts
            progress["baseline_fingerprint"] = fingerprint
            progress["baseline_origin"] = "restored_from_completed_terminal_counts"
            hub.publish_progress(root, progress)
            print("[BASELINE RESTORED TO HF] Task 0: "
                  f"{sum(v['correct'] for v in counts.values())}/"
                  f"{sum(v['total'] for v in counts.values())}; "
                  "set RL_BASELINE_MODE=reuse to begin training", flush=True)
            return

        # Fixed sample panel: visualization never modifies training batches.
        anchors = None
        if config.visualization_mode:
            source = (evaluation if config.visualization_source == "test" else {
                domain: load_domain_data(loader, formatter, domain, "train")
                for domain in config.train_domains
            })
            anchors = visualization.select_samples(config, source)

        # Resolve an existing HF baseline before allocating GPU workers.
        # 'reuse' never silently falls back to hours of baseline evaluation.
        baseline, baseline_fp, baseline_source = choose_baseline(
            config, progress, hub, evaluation, signature
        )
        if baseline is not None:
            progress["baseline_accuracy"] = baseline
            progress["baseline_fingerprint"] = baseline_fp
            print(f"[BASELINE REUSED] {baseline_source}", flush=True)
            # External baseline needs one-time copying into this track's HF repo.
            if signature_changed_for_baseline_only or (
                not progress.get("history") and (
                    remote_progress is None or remote_progress.get("baseline_accuracy") is None
                )
            ):
                # Publish only AFTER choose_baseline validates the data fingerprint.
                hub.publish_progress(root, progress)
        elif config.benchmark_forward_transfer and not progress.get("history"):
            print(f"[BASELINE EVALUATE] {baseline_source}", flush=True)

        if config.eval_skip_completed:
            self.restore_logged_eval(config, progress, evaluation, hub, root)

        # Initialize persistent Ray + FSDP workers.
        workers, group = launch_ray(config)

        try:
            history = progress["history"]

            # The baseline has already been validated and selected before Ray starts.
            if config.benchmark_forward_transfer and history and not baseline:
                raise ValueError("Forward transfer on resume requires the original baseline_accuracy")

            # Resume from the last completed domain.
            if history:
                checkpoint = hub.download_checkpoint(history[-1], root)
                ray.get([worker.load_checkpoint.remote(checkpoint) for worker in workers])
                shutil.rmtree(root / "hub_downloads" / Path(history[-1]["checkpoint"]).name)
                print(f"[RESUME from private Hub] {history[-1]['checkpoint']}")

                # Resume only missing domains; a completed domain was already saved to HF.
                domains = (config.train_domains if config.benchmark_forward_transfer else
                           [entry["domain"] for entry in history])
                saved = history[-1].get("accuracy") or {}
                if set(saved) != set(domains):
                    history[-1]["accuracy"] = evaluate_domains(
                        workers, config,
                        {domain: evaluation[domain] for domain in domains},
                        task_id=len(history), previous=saved,
                        on_domain=lambda domain, score: self.save_domain_eval(
                            progress, root, hub, history[-1], domain, score),
                    )
                    progress["metrics"] = benchmark.update(history, config, baseline)
                    hub.publish_progress(root, progress)

            # Optional zero-shot baseline before any training, required for FWT.
            if config.benchmark_forward_transfer and not history and baseline is None:
                partial = progress.get("baseline_partial_accuracy") or {}
                def save_baseline_domain(domain, score):
                    progress["baseline_partial_accuracy"] = {
                        **(progress.get("baseline_partial_accuracy") or {}), domain: score,
                    }
                    hub.publish_progress(root, progress)
                baseline = evaluate_domains(
                    workers, config, evaluation, task_id=0,
                    previous=partial, on_domain=save_baseline_domain,
                )
                progress.pop("baseline_partial_accuracy", None)
                progress["baseline_accuracy"] = baseline
                progress["baseline_fingerprint"] = evaluation_fingerprint(evaluation)
                hub.publish_progress(root, progress)

            if history and config.benchmark_mode:
                progress["metrics"] = benchmark.update(history, config, baseline)

            if config.visualization_mode:
                initial_stage = (f"after_{history[-1]['domain']}" if history else "before_training")
                visualization.capture(workers, config, anchors, initial_stage)
            ray.get([worker.cleanup_gpu.remote() for worker in workers])

            # Train the remaining domains.
            for task_id in range(
                len(history) + 1,
                len(config.train_domains) + 1,
            ):
                domain = config.train_domains[
                    task_id - 1
                ]

                print(
                    f"\n[TASK {task_id}] {domain}"
                )

                if config.trace_mode:
                    trace.log(config, "hub.train.start", domain=domain, task=task_id)
                records = load_domain_data(loader, formatter, domain, "train")
                if config.trace_mode:
                    trace.log(config, "hub.train.finished", domain=domain,
                              task=task_id, records=len(records))

                # Set seeds across all FSDP workers.
                ray.get(
                    [
                        worker.seed_domain.remote(
                            task_id
                        )
                        for worker in workers
                    ],
                    timeout=config.ray_dist_timeout_s,
                )

                # Reuse the existing training engine.
                steps = self.train_domain(
                    workers,
                    domain,
                    records,
                    task_id,
                )
                ray.get([worker.cleanup_gpu.remote() for worker in workers])

                # Save the full sharded model + AdamW state, then synchronously
                # upload to the PRIVATE Hugging Face MODEL repository.
                local_checkpoint = root / (
                    f"task_{task_id:02d}_{domain}_{uuid.uuid4().hex[:8]}"
                )
                if config.trace_mode:
                    trace.log(config, "checkpoint.save.start", task=task_id,
                              path=str(local_checkpoint))
                requests = [
                    worker.save_checkpoint.remote(str(local_checkpoint))
                    for worker in workers
                ]
                (await_workers_progress(config, requests, "checkpoint.save", no_timeout=True,
                                     task=task_id)
                 if config.trace_mode else ray.get(requests))

                uploaded = hub.upload_checkpoint(local_checkpoint)
                history.append({
                    "domain": domain,
                    "steps": steps,
                    "train_batch_size": (config.train_finance_batch_size if domain == "finance"
                                         else config.train_batch_size),
                    "gradient_accumulation": (config.train_finance_gradient_accumulation
                                              if domain == "finance" else 1),
                    **uploaded,
                    "accuracy": None,
                })
                hub.publish_progress(root, progress)
                # The verified private Hub version is now authoritative; avoid
                # accumulating multi-GB checkpoints on the container disk.
                shutil.rmtree(local_checkpoint)

                # Mandatory round-trip: obtain an immutable Hub revision, download
                # into a distinct folder, and RESTORE model + optimizer on ALL ranks.
                # If the upload/download/load fails, no next task is permitted.
                downloaded = hub.download_checkpoint(history[-1], root)
                if config.trace_mode:
                    trace.log(config, "checkpoint.hub.reload.start", task=task_id,
                              repo=hub.repo_id, path=history[-1]["checkpoint"])
                ray.get([worker.load_checkpoint.remote(downloaded) for worker in workers])
                shutil.rmtree(root / "hub_downloads" / Path(history[-1]["checkpoint"]).name)
                if config.trace_mode:
                    trace.log(config, "checkpoint.hub.reload.finished", task=task_id)
                print(f"[HUB] Task {task_id} reloaded from PRIVATE repo", flush=True)

                # By default evaluate learned tasks; optional FWT also evaluates unseen tasks.
                domains = (config.train_domains if config.benchmark_forward_transfer
                           else config.train_domains[:task_id])

                history[-1]["accuracy"] = evaluate_domains(
                    workers,
                    config,
                    {domain: evaluation[domain] for domain in domains},
                    task_id=task_id,
                    on_domain=lambda domain, score: self.save_domain_eval(
                        progress, root, hub, history[-1], domain, score),
                )

                # Benchmark owns all continual metrics and chart rendering.
                progress["metrics"] = benchmark.update(history, config, baseline)
                if config.trace_mode:
                    trace.log(config, "task.eval.finished", task=task_id,
                              accuracy=history[-1]["accuracy"],
                              metrics=progress["metrics"][-1])
                hub.publish_progress(root, progress)

                print(f"[METRICS] {progress['metrics'][-1]}")

                if config.visualization_mode:
                    visualization.capture(workers, config, anchors, f"after_{domain}")
                ray.get([worker.cleanup_gpu.remote() for worker in workers])

        finally:
            # Do not hide the actual Hub failure behind a secondary Ray shutdown timeout.
            pending_error = sys.exc_info()[0] is not None
            try:
                shutdown_ray(workers, group, config)
            except Exception as exc:
                if not pending_error:
                    raise
                print(f"[WARN] Ray cleanup failed after an earlier error: {exc}", flush=True)



def build_track_config(base: Config, algorithm: str) -> Config:
    """Identical schedule/seed/data with independent state and result locations.

    Preserve existing GRPO paths/repo for backward-compatible resume. The
    self-improve track always uses a distinct local/remote checkpoint namespace.
    """
    if algorithm not in {"grpo", "self_improve"}:
        raise ValueError(f"Unknown track: {algorithm}")
    cfg = replace(base, train_algorithm=algorithm)
    # Config.__post_init__ reads the shell environment on dataclass cloning;
    # force the requested algorithm even if a legacy variable is still set.
    cfg.train_algorithm = algorithm
    if algorithm == "self_improve":
        cfg.continual_checkpoint_dir = Path(str(base.continual_checkpoint_dir) + "_self_improve")
        cfg.benchmark_dir = Path(str(base.benchmark_dir) + "_self_improve")
        cfg.visualization_dir = Path(str(base.visualization_dir) + "_self_improve")
        cfg.trace_dir = Path(str(base.trace_dir) + "_self_improve")
        # If the user specified a model repo, derive a distinct private repo.
        # If None, Config.resolve_hf_model_repo() picks the independent default.
        if base.hf_model_repo:
            owner, repo = base.hf_model_repo.split("/", 1)
            cfg.hf_model_repo = f"{owner}/{repo}-self-improve"
    if cfg.train_sampler != "random":
        if algorithm != "grpo":
            raise ValueError("Embedding sampling is supported only for GRPO")
        # Derive independent repo names and local directories per sampler.
        # The original GRPO baseline is reused read-only (fingerprint checked).
        original_repo = base.resolve_hf_model_repo()
        cfg.hf_model_repo = original_repo + "-" + cfg.train_sampler.replace("_", "-")
        cfg.baseline_source_repo = cfg.baseline_source_repo or original_repo
        cfg.continual_checkpoint_dir = Path(str(cfg.continual_checkpoint_dir) + "_" + cfg.train_sampler)
        cfg.benchmark_dir = Path(str(cfg.benchmark_dir) + "_" + cfg.train_sampler)
        cfg.visualization_dir = Path(str(cfg.visualization_dir) + "_" + cfg.train_sampler)
        cfg.trace_dir = Path(str(cfg.trace_dir) + "_" + cfg.train_sampler)
        # Old Task-1 recovery data must never enter a different training policy.
        cfg.eval_restore_counts = None
        cfg.eval_restore_revision = None
    cfg.validate()
    return cfg


def main() -> None:
    base = parse_config()
    requested = (["grpo", "self_improve"] if base.run_track == "both"
                 else [base.run_track])

    # Avoid even launching GPUs/uploading checkpoints when an experiment would
    # silently be invalid. Users implement their policy before choosing it.
    if "self_improve" in requested:
        from policy import SelfImprovePolicy
        if not SelfImprovePolicy.IMPLEMENTED:
            raise NotImplementedError(
                "RL_RUN_TRACK requests self_improve but the custom algorithm "
                "is not yet implemented. Implement SelfImprovePolicy in "
                "train/self_improve_policy.py, set IMPLEMENTED=True, then rerun. "
                "No GRPO fallback will be used."
            )

    if base.baseline_mode == "restore" and len(requested) > 1:
        raise ValueError("For one-time baseline restore select RL_RUN_TRACK=grpo or self_improve")

    for index, algorithm in enumerate(requested, 1):
        cfg = build_track_config(base, algorithm)
        print(f"\n[EXPERIMENT {index}/{len(requested)}] {algorithm} "
              f"tasks={cfg.train_domains} max_steps={cfg.train_max_steps} "
              f"seed={cfg.train_seed} checkpoint={cfg.continual_checkpoint_dir}",
              flush=True)
        ContinualTrainer(cfg).run()


if __name__ == "__main__":
    main()
