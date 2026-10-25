"""Shared evaluation, continual metrics, and Matplotlib metric charts."""
from __future__ import annotations
import csv
import os
from pathlib import Path
import torch
from tqdm.auto import tqdm
from distributed import BatchRouter, await_workers_progress
from policy import reward_breakdown
from continual_train.storage import atomic_write_json
import diagnostics as trace


def load_domain_data(loader, formatter, domain: str, mode: str) -> list[dict]:
    """Map normalized Hub records into the common model input/label shape."""
    rows = loader.read_hub(domain, mode=mode, continual=True)
    return [
        {
            "messages": formatter.build_messages(row, domain),
            "answer": row["answer"],
            "answer_letter": row["answer_letter"],
            "reference": row["reference"],
            **({"raw_record": row} if mode == "test" and i < 5 else {}),
        }
        for i, row in enumerate(rows)
    ]


@torch.no_grad()
def evaluate_worker(
    rollout,
    domain: str,
    samples: list[dict],
) -> dict:
    # TP ranks receive the same samples and must split them identically.
    correct = total = 0
    size = (rollout.config.eval_finance_microbatch_size if domain == "finance"
            else rollout.config.eval_microbatch_size)
    examples = []
    for start in range(0, len(samples), size):
        batch = samples[start:start + size]
        output = rollout.generate(batch, group_size=1, greedy=True)
        if rollout.config.trace_mode:
            trace.evaluated(rollout.config, batch, output, domain,
                            lambda d, response, row: reward_breakdown(d, response, row)["correctness_reward"])
        for response, reason, row in zip(output["responses"], output["finish_reasons"], batch):
            if row.get("_eval_valid", True):
                reward = reward_breakdown(domain, response, row)
                matched = bool(reward["correctness_reward"])
                correct += int(matched)
                total += 1
                if row.get("_eval_index", 1_000_000) < rollout.config.eval_example_count:
                    examples.append({
                        "sample_index": row["_eval_index"],
                        "raw_record": row.get("raw_record"),
                        "messages": row["messages"],
                        "model_response": response,
                        "predicted_final_answer": reward["extracted_final_answer"],
                        "answer": row["answer"],
                        "answer_letter": row.get("answer_letter"),
                        "reference": row.get("reference"),
                        "correct": matched,
                        "finish_reason": reason,
                    })
    return {"correct": int(correct), "total": total, "examples": examples}


def evaluate_domains(
    workers,
    config,
    datasets: dict[str, list[dict]],
    task_id: int | None = None,
    *,
    previous: dict | None = None,
    on_domain=None,
) -> dict:
    router = BatchRouter(config, len(workers), config.continual_eval_samples_per_worker)
    samples_per_step = router.global_size
    scores = dict(previous or {}) if config.eval_skip_completed else {}
    task_count = len(config.train_domains)
    task_number = task_id if task_id is not None else "?"
    phase = "BASELINE" if task_id == 0 else "EVAL"
    eval_total = sum(len(records) for records in datasets.values())
    if not eval_total:
        raise ValueError("No evaluation samples")

    with tqdm(
        total=eval_total, desc=f"Task {task_number}/{task_count} {phase} total",
        unit="sample", position=0, leave=True, dynamic_ncols=True,
        disable=not config.progress_bar,
    ) as overall_bar:
        for domain_index, (domain, records) in enumerate(datasets.items(), start=1):
            if config.trace_mode:
                trace_task = task_id if task_id is not None else len(datasets)
                for row in records[:5]:
                    if "_trace_id" in row:
                        row["_trace_task"] = trace_task
                trace.log(config, "eval.domain.start", domain=domain,
                          task=trace_task, count=len(records))
            if not records:
                raise ValueError(f"Empty evaluation domain: {domain}")

            if domain in scores:
                result = scores[domain]
                if (result.get("total") != len(records) or
                    type(result.get("correct")) is not int or
                    not 0 <= result["correct"] <= len(records) or
                    abs(result.get("accuracy", -1) - result["correct"] / len(records)) > 1e-8):
                    raise ValueError(f"Corrupt saved evaluation for {domain}")
                print(f"[EVAL SKIP HF] Task {task_number} {domain}: "
                      f"{result['correct']}/{result['total']}", flush=True)
                overall_bar.update(len(records))
                continue

            correct = 0
            total = 0
            examples = []
            audit_path = (Path("outputs/eval_samples") /
                          f"task_{task_number:02d}" / f"{domain}.json")
            sample_saved = False
            with tqdm(
                total=len(records),
                desc=(f"  domain {domain_index}/{len(datasets)} {domain}"),
                unit="sample", position=1, leave=False,
                dynamic_ncols=True, disable=not config.progress_bar,
            ) as domain_bar:
                for start in range(0, len(records), samples_per_step):
                    # Count real prompts only. Padding repeats are excluded.
                    samples = [
                        dict(record, _eval_valid=True, _eval_index=start + i)
                        for i, record in enumerate(records[start:start + samples_per_step])
                    ]
                    valid_count = len(samples)
                    samples.extend(
                        dict(samples[-1], _eval_valid=False)
                        for _ in range(samples_per_step - valid_count)
                    )

                    batches = router.split(samples)
                    requests = [
                        worker.evaluate.remote(domain, batch)
                        for worker, batch in zip(workers, batches)
                    ]
                    batch_index = start // samples_per_step + 1
                    batch_count = (len(records) + samples_per_step - 1) // samples_per_step
                    results = await_workers_progress(
                        config, requests, "eval.rollout", bar=domain_bar,
                        status=f"batch {batch_index}/{batch_count} ({valid_count} in-flight)",
                        domain=domain, batch_start=start,
                    )

                    # TP ranks process exactly the SAME prompts; count only one.
                    for result in router.unique(results):
                        correct += result["correct"]
                        total += result["total"]
                        examples.extend(result.get("examples", []))
                    if not sample_saved and len(examples) >= min(config.eval_example_count, len(records)):
                        atomic_write_json(audit_path, {
                            "task_id": task_id, "domain": domain,
                            "examples": sorted(examples, key=lambda item: item["sample_index"])[
                                :config.eval_example_count],
                        }, indent=2)
                        print(f"[EVAL SAMPLES] {audit_path}", flush=True)
                        sample_saved = True
                    if total != start + valid_count:
                        raise RuntimeError(
                            f"{domain}: expected {start + valid_count} evaluated prompts, "
                            f"got {total}"
                        )
                    domain_bar.update(valid_count)
                    overall_bar.update(valid_count)
                    overall_bar.set_postfix_str(
                        f"domain={domain} ({domain_index}/{len(datasets)})",
                        refresh=True,
                    )

            scores[domain] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            }
            if on_domain is not None:
                on_domain(domain, scores[domain])
            if config.progress_bar:
                tqdm.write(f"[EVAL] Task {task_number}/{task_count} "
                           f"{domain}: {correct}/{total} = {correct / total:.4%}")
            else:
                print(f"[EVAL] {domain}: {correct}/{total} = {correct / total:.4%}")
            if config.trace_mode:
                trace.log(config, "eval.domain.finished", domain=domain,
                          task=trace_task, correct=correct, total=total,
                          accuracy=correct / total)

    return scores


def summarize(history: list[dict], domains: list[str], baseline: dict | None = None) -> list[dict]:
    """Compute AA, learning accuracy, forgetting, BWT and optional FWT.

    FWT is only available when baseline (before any task) and scores for
    not-yet-trained domains are provided. Missing values are never invented.
    """
    metrics = []
    for t, entry in enumerate(history):
        scores = entry.get("accuracy")
        if scores is None:
            break
        learned = domains[:t + 1]
        if any(domain not in scores for domain in learned):
            raise ValueError(f"Missing learned-domain score after task {t + 1}")

        current = {domain: float(scores[domain]["accuracy"]) for domain in learned}
        forgetting = {}
        backward = {}
        for j, domain in enumerate(learned[:-1]):
            first_score = float(history[j]["accuracy"][domain]["accuracy"])
            past_max = max(
                float(history[k]["accuracy"][domain]["accuracy"])
                for k in range(j, t)
            )
            forgetting[domain] = max(0.0, past_max - current[domain])
            backward[domain] = current[domain] - first_score

        fwt_values = []
        if baseline is not None:
            for j in range(1, t + 1):
                domain = domains[j]
                earlier = history[j - 1]["accuracy"].get(domain)
                reference = baseline.get(domain)
                if earlier is not None and reference is not None:
                    fwt_values.append(
                        float(earlier["accuracy"]) - float(reference["accuracy"])
                    )

        metrics.append({
            "after_domain": entry["domain"],
            "task_index": t + 1,
            "average_accuracy": sum(current.values()) / len(learned),
            "learning_accuracy": current[entry["domain"]],
            "average_forgetting": sum(forgetting.values()) / len(forgetting) if forgetting else 0.0,
            "backward_transfer": sum(backward.values()) / len(backward) if backward else 0.0,
            "forward_transfer": sum(fwt_values) / len(fwt_values) if fwt_values else None,
            "per_domain_forgetting": forgetting,
            "accuracy": current,
        })
    return metrics


def update(history: list[dict], config, baseline: dict | None = None) -> list[dict]:
    """Called after every completed task; metrics are returned even if charts are off."""
    metrics = summarize(history, config.train_domains, baseline)
    if not config.benchmark_mode or not metrics:
        return metrics

    root = Path(config.benchmark_dir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "metrics.json", {"domains": config.train_domains, "history": metrics,
                                         "baseline": baseline})

    keys = ("task_index", "after_domain", "average_accuracy", "learning_accuracy",
            "average_forgetting", "backward_transfer", "forward_transfer")
    with (root / "metrics.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: item[key] for key in keys} for item in metrics)
    os.replace(root / "metrics.csv.tmp", root / "metrics.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(metrics)
    labels = [f"{i+1}: {m['after_domain']}" for i, m in enumerate(metrics)]
    indices = range(1, n + 1)
    fig, ax = plt.subplots(figsize=(max(7, n * 1.4), 4.5))
    for key, label in (
        ("average_accuracy", "Average accuracy"),
        ("learning_accuracy", "Learning accuracy"),
        ("average_forgetting", "Average forgetting"),
        ("backward_transfer", "Backward transfer"),
        ("forward_transfer", "Forward transfer"),
    ):
        series = [float(m[key]) if m[key] is not None else float("nan") for m in metrics]
        if not all(np.isnan(value) for value in series):
            ax.plot(indices, series, marker="o", label=label)
    ax.axhline(0, linewidth=0.8, color="gray")
    ax.set(xticks=list(indices), xticklabels=labels, ylabel="Score (fraction)",
           title="Continual-learning metrics after each task")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, root / "metrics_trends.png")

    grid = np.full((n, len(config.train_domains)), np.nan)
    for i, entry in enumerate(history[:n]):
        for j, domain in enumerate(config.train_domains):
            score = entry["accuracy"].get(domain)
            if score is not None:
                grid[i, j] = float(score["accuracy"])
    fig, ax = plt.subplots(figsize=(max(7, len(config.train_domains) * 1.45), max(3, n * 0.7 + 1.5)))
    image = ax.imshow(np.ma.masked_invalid(grid), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set(xticks=range(len(config.train_domains)), xticklabels=config.train_domains,
           yticks=range(n), yticklabels=labels, xlabel="Evaluation domain",
           ylabel="After learning", title="Task accuracy matrix (A[t, j])")
    ax.tick_params(axis="x", rotation=25)
    for i in range(n):
        for j in range(len(config.train_domains)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{100 * grid[i, j]:.1f}%", ha="center", va="center",
                        color="white" if grid[i, j] < .65 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Accuracy")
    fig.tight_layout()
    _save(fig, root / "accuracy_matrix.png")

    fig, ax = plt.subplots(figsize=(max(7, n * 1.4), 4.5))
    for domain in config.train_domains:
        values = [m["accuracy"].get(domain, float("nan")) for m in metrics]
        if not all(np.isnan(value) for value in values):
            ax.plot(indices, values, marker="o", label=domain)
    ax.set(xticks=list(indices), xticklabels=labels, ylabel="Accuracy", ylim=(-0.05, 1.05),
           title="Accuracy by domain")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, root / "domain_accuracy.png")

    # A self-refreshing dashboard: images are regenerated after each evaluation.
    from html import escape
    stamp = os.stat(root / "metrics_trends.png").st_mtime_ns
    images = ("metrics_trends.png", "accuracy_matrix.png", "domain_accuracy.png")
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta http-equiv='refresh' content='15'><meta http-equiv='Cache-Control' content='no-store'>"
            "<title>Continual benchmark</title><style>body{font:16px system-ui;"
            "max-width:1200px;margin:2em auto;padding:0 1em}img{width:100%;max-width:1000px;}"
            "</style></head><body><h1>Continual benchmark</h1>"
            "<p>Updated after each completed task; auto-refresh every 15 seconds. "
            "Missing future-domain scores are not estimated.</p>"
            + "".join(f"<img alt='{escape(file)}' src='{file}?v={stamp}'>" for file in images)
            + "</body></html>")
    (root / "index.html").write_text(html, encoding="utf-8")
    return metrics


def _save(fig, path: Path):
    import matplotlib.pyplot as plt
    temporary = path.with_suffix(".tmp")
    fig.savefig(temporary, format="png", dpi=120)
    plt.close(fig)
    os.replace(temporary, path)
