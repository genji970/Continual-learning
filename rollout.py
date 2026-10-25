"""KV-cached rollout/generation and optional NCCL profile."""
from __future__ import annotations
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function
from config import Config
from policy import PolicyTrainer, completed_final_answer_line
from continual_train.storage import atomic_write_json
import diagnostics as trace


PROFILE_DECODE_STEPS = 16


_BENCHMARK_INPUT_MIB = (4, 32)  # test latency and throughput regimes


_BENCHMARK_ITERS = 10


def _category(name: str) -> str:
    normalized = name.lower().replace("_", "").replace("-", "")
    if "allgather" in normalized:
        return "all_gather"
    if "reducescatter" in normalized:
        return "reduce_scatter"
    if "allreduce" in normalized:
        return "all_reduce"
    if "broadcast" in normalized:
        return "broadcast"
    return "other_nccl"


def _bytes_to_mib(num_bytes: float) -> float:
    return round(num_bytes / 1024**2, 3)


def benchmark_collectives(device: torch.device) -> dict:
    """Measure short, in-process NCCL collective tests (all ranks participate)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size < 2:
        return {"world_size": world_size, "note": "single rank: no inter-GPU NCCL benchmark"}
    if device.type != "cuda" or dist.get_backend() != "nccl":
        return {"world_size": world_size, "note": "NCCL CUDA benchmark unavailable"}

    results = {}
    for input_mib in _BENCHMARK_INPUT_MIB:
        count = (input_mib * 1024**2) // 2  # BF16 elements
        shard = torch.ones(count, device=device, dtype=torch.bfloat16)
        gathered = torch.empty(count * world_size, device=device, dtype=shard.dtype)
        scatter_in = torch.ones(count * world_size, device=device, dtype=shard.dtype)
        scattered = torch.empty_like(shard)
        allreduce = torch.ones_like(shard)

        def all_gather():
            dist.all_gather_into_tensor(gathered, shard)

        def reduce_scatter():
            dist.reduce_scatter_tensor(scattered, scatter_in)

        def all_reduce():
            dist.all_reduce(allreduce)

        by_op = {}
        # All workers use identical benchmark sizes, order and iteration count.
        for label, fn in (
            ("all_gather", all_gather),
            ("reduce_scatter", reduce_scatter),
            ("all_reduce", all_reduce),
        ):
            for _ in range(3):
                fn()
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            start_event.record()
            for _ in range(_BENCHMARK_ITERS):
                fn()
            stop_event.record()
            stop_event.synchronize()
            wall_ms = (time.perf_counter() - started) * 1000.0 / _BENCHMARK_ITERS
            gpu_ms = start_event.elapsed_time(stop_event) / _BENCHMARK_ITERS
            item = {
                "gpu_ms_per_call": round(gpu_ms, 4),
                "host_wall_ms_per_call": round(wall_ms, 4),
                "iterations": _BENCHMARK_ITERS,
                "input_shard_mib": input_mib,
            }
            if label == "all_gather":
                logical_receive_bytes = (world_size - 1) * shard.numel() * shard.element_size()
                item["logical_received_mib_per_rank"] = _bytes_to_mib(logical_receive_bytes)
                item["effective_received_gib_s"] = round(
                    logical_receive_bytes / (1024**3) / (gpu_ms / 1000), 4
                ) if gpu_ms > 0 else None
            by_op[label] = item

        results[f"{input_mib}_mib"] = by_op
        del shard, gathered, scatter_in, scattered, allreduce
    return {
        "rank": rank,
        "world_size": world_size,
        "tensor_dtype": "bfloat16",
        "note": (
            "Synthetic collective microbenchmark on the current NCCL topology. "
            "Effective all_gather receive rate is logical per-rank bytes / GPU event time; "
            "it is not a direct NIC/SHM bandwidth or FSDP layer bandwidth measurement."
        ),
        "collectives_by_input_size": results,
    }


class RolloutCommunicationProfile:
    """Profile one prefill + up to 16 decode steps without per-token sync."""

    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.path = Path(config.trace_dir) / "profiler"
        self.path.mkdir(parents=True, exist_ok=True)
        self.profiler = None
        self.active = False
        self.finished = False
        self.forward_steps = 0
        self.benchmark = None

    def run_bandwidth_benchmark(self):
        # Every rank runs this before its FIRST sampled GRPO rollout.
        self.benchmark = benchmark_collectives(self.device)
        atomic_write_json(self.path / f"rank{self.rank}_nccl_bandwidth.json", self.benchmark)

    def start(self):
        if self.active or self.finished:
            raise RuntimeError("Communication profiler can only start once")
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self.profiler.start()
        self.active = True

    def stop(self, steps: int):
        if not self.active:
            return
        self.active = False
        self.finished = True
        self.forward_steps = steps
        self.profiler.stop()  # synchronization only at the end of this short window

        chrome_path = self.path / f"rank{self.rank}_rollout_cuda_trace.json"
        self.profiler.export_chrome_trace(str(chrome_path))

        categories = defaultdict(lambda: {"kernels": 0, "summed_cuda_kernel_ms": 0.0})
        nccl_names = defaultdict(lambda: {"kernels": 0, "summed_cuda_kernel_ms": 0.0})
        all_cuda_kernel_us = 0.0
        for event in self.profiler.events():
            if event.device_type != torch.autograd.DeviceType.CUDA:
                continue
            microseconds = float(getattr(event, "device_time_total", 0) or 0)
            if not microseconds:
                microseconds = float(getattr(event, "cuda_time_total", 0) or 0)
            all_cuda_kernel_us += microseconds
            if "nccl" not in event.name.lower():
                continue
            category = _category(event.name)
            categories[category]["kernels"] += 1
            categories[category]["summed_cuda_kernel_ms"] += microseconds / 1000
            nccl_names[event.name]["kernels"] += 1
            nccl_names[event.name]["summed_cuda_kernel_ms"] += microseconds / 1000

        def rounded(d):
            return {
                k: {"kernels": v["kernels"],
                    "summed_cuda_kernel_ms": round(v["summed_cuda_kernel_ms"], 4)}
                for k, v in sorted(d.items())
            }

        report = {
            "rank": self.rank,
            "profiled_forward_steps": steps,
            "prefill_forwards": 1 if steps else 0,
            "decoding_forwards": max(0, steps - 1),
            "cuda_kernels_summed_ms": round(all_cuda_kernel_us / 1000, 4),
            "nccl_cuda_kernels": rounded(categories),
            "nccl_kernel_names": rounded(nccl_names),
            "chrome_trace": str(chrome_path),
            "benchmark_json": str(self.path / f"rank{self.rank}_nccl_bandwidth.json"),
            "note": (
                "Summed CUDA kernel durations are NOT critical-path wall times. "
                "Communication kernels can overlap with computation and with other streams. "
                "Use Chrome trace to locate waits, gaps, and overlap. "
                "Only the first prefill + 16 decoding forwards are profiled."
            ),
        }
        path = self.path / f"rank{self.rank}_rollout_communication.json"
        atomic_write_json(path, report)
        if not self.config.progress_bar:
            print(
                f"[NCCL PROFILE] rank={self.rank} steps={steps} report={path}",
                flush=True,
            )
        return report


class Rollout:
    """The training entry point used by Ray workers."""

    def __init__(self, model, tokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.policy = PolicyTrainer(model, tokenizer, config)
        self.device = next(model.parameters()).device
        self._communication_profile_taken = False

    def _stop_communication_profile(self, profiler, steps: int) -> None:
        report = profiler.stop(steps)
        trace.log(
            self.config, "rollout.nccl_profile",
            rank=profiler.rank,
            profiled_forward_steps=report["profiled_forward_steps"],
            file=str(profiler.path / f"rank{profiler.rank}_rollout_communication.json"),
        )

    @torch.no_grad()
    def generate(self, batch: list[dict], *, group_size: int | None = None, greedy: bool = False) -> dict:
        c = self.config
        g = group_size if group_size is not None else c.grpo_group_size
        pad = self.tokenizer.pad_token_id
        eos_ids = set()
        for source in (self.tokenizer, getattr(self.model, "generation_config", None),
                       getattr(self.model, "config", None)):
            value = getattr(source, "eos_token_id", None)
            if value is not None:
                eos_ids.update(value if isinstance(value, (list, tuple, set)) else [value])
        if pad is None or not eos_ids:
            raise ValueError("Tokenizer/model requires pad_token_id and eos_token_id")
        eos_values = torch.tensor(sorted(eos_ids), device=self.device, dtype=torch.long)
        if not batch:
            raise ValueError("Empty rollout batch")

        if c.trace_mode:
            torch.cuda.synchronize(self.device)
            rollout_start = time.perf_counter()

        prompts = [
            self.tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True
            )
            for row in batch
        ]
        if any(not p or len(p) > c.grpo_max_prompt_length for p in prompts):
            raise ValueError("Empty or too-long rollout prompt")

        if c.trace_mode:
            trace.batch(c, "rollout.prompt", batch, prompt_lengths=[len(p) for p in prompts],
                        group_size=g, greedy=greedy)
        prompts = [p for p in prompts for _ in range(g)]
        width = max(len(p) for p in prompts)
        n = len(prompts)
        ids = torch.full((n, width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(prompts):
            ids[i, -len(p):] = torch.tensor(p, device=self.device)
            mask[i, -len(p):] = 1

        self.model.eval()
        finished = torch.zeros(n, dtype=torch.bool, device=self.device)
        completions = []
        completion_masks = []
        generated_ids = [[] for _ in range(n)]
        finish_reasons = [None] * n
        cache = None
        # Amortize CPU decoding and cross-rank completion checks. EOS is still
        # masked on the GPU in the very step it is sampled. A completed final-
        # answer line may be recognized up to 7 steps later; those speculative
        # trailing tokens are retrospectively removed from the training mask.
        sync_interval = 8
        block_start = 0

        # Diagnostic mode: first rollout (evaluation OR training) on every rank.
        # Eval uses greedy=True, so gating this on 'not greedy' would postpone
        # bandwidth measurements until the first training rollout.
        # All ranks must execute these collectives in the same order.
        communication_profile = None
        if c.trace_mode and not c.progress_bar and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[ROLLOUT START] {'eval' if greedy else 'train'} "
                f"batch={len(batch)} completions={n} "
                f"prompt_width={width} max_new_tokens={c.grpo_max_completion_length}",
                flush=True,
            )
        if c.trace_mode and not self._communication_profile_taken:
            self._communication_profile_taken = True
            communication_profile = RolloutCommunicationProfile(c, self.device)
            benchmark_started = time.perf_counter()
            communication_profile.run_bandwidth_benchmark()
            # Exclude the diagnostic microbenchmark from actual rollout latency.
            rollout_start += time.perf_counter() - benchmark_started
            if communication_profile.rank == 0 and not c.progress_bar:
                results = communication_profile.benchmark.get("collectives_by_input_size", {})
                for size, measurements in results.items():
                    gather = measurements["all_gather"]
                    scatter = measurements["reduce_scatter"]
                    reduce = measurements["all_reduce"]
                    print(
                        f"[NCCL BENCH] size={size} "
                        f"all_gather={gather['gpu_ms_per_call']:.3f}ms "
                        f"logical_receive={gather.get('effective_received_gib_s')}GiB/s "
                        f"reduce_scatter={scatter['gpu_ms_per_call']:.3f}ms "
                        f"all_reduce={reduce['gpu_ms_per_call']:.3f}ms "
                        f"(synthetic benchmark, not measured rollout NVLink traffic)",
                        flush=True,
                    )
                print(
                    f"[NCCL BENCH] reports={communication_profile.path} "
                    f"(one-time measurement on first rollout)",
                    flush=True,
                )
            communication_profile.start()

        if c.trace_mode:
            prefill_start = torch.cuda.Event(enable_timing=True)
            prefill_end = torch.cuda.Event(enable_timing=True)
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            decode_started = False

        # All TP/FSDP ranks must execute the same number of forward passes.
        for step in range(c.grpo_max_completion_length):
            if c.trace_mode and (step == 0 or step % 16 == 0):
                trace.batch(c, "rollout.tokens", batch, generated=step)
            # CUDA event timing records the first Forward (prefill) separately
            # from subsequent auto-regressive decoding. No per-token sync.
            if c.trace_mode and step == 0:
                prefill_start.record()
            elif c.trace_mode and step == 1:
                decode_start.record()
                decode_started = True
            profiling = communication_profile is not None and communication_profile.active
            label = ("rollout.prefill.forward" if step == 0
                     else "rollout.decoding.forward")
            scope = record_function(label) if profiling else nullcontext()
            with scope:
                output = self.model(
                    input_ids=ids if cache is None else ids[:, -1:],
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,  # Only next-token logits; preserves full CoT decoding.
                )
            if c.trace_mode and step == 0:
                prefill_end.record()
            cache = output.past_key_values
            logits = output.logits[:, -1, :].float() / c.grpo_temperature

            if c.grpo_top_p < 1:
                ordered, index = logits.sort(dim=-1, descending=True)
                probs = ordered.softmax(dim=-1)
                remove = (probs.cumsum(-1) - probs) > c.grpo_top_p
                ordered = ordered.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, index, ordered
                )

            if c.parallel_mode == "tp" and dist.is_initialized():
                # Sample only on rank 0. TP ranks must run identical KV caches.
                if dist.get_rank() == 0:
                    sampled = (logits.argmax(dim=-1, keepdim=True) if greedy
                               else torch.multinomial(logits.softmax(dim=-1), 1))
                else:
                    sampled = torch.empty((n, 1), device=self.device, dtype=torch.long)
                dist.broadcast(sampled, src=0)
            else:
                sampled = (logits.argmax(dim=-1, keepdim=True) if greedy
                           else torch.multinomial(logits.softmax(dim=-1), 1))
            active = ~finished
            token = torch.where(active[:, None], sampled, pad)
            completions.append(token.squeeze(-1))
            completion_masks.append(active)

            # EOS is final for this individual response, even if other FSDP
            # ranks/responses still need more forwards. Never generate another
            # valid token after EOS.
            hit_eos = active & torch.isin(token.squeeze(-1), eos_values)
            finished |= hit_eos

            ids = torch.cat((ids, token), dim=1)
            mask = torch.cat((mask, active[:, None].long()), dim=1)

            if (communication_profile is not None and communication_profile.active
                    and step == PROFILE_DECODE_STEPS):
                self._stop_communication_profile(communication_profile, step + 1)

            # All ranks check completion at identical, deterministic steps.
            # Avoid .tolist()/.item() and CPU-GPU synchronization per token.
            if ((step + 1) % sync_interval != 0
                    and step + 1 != c.grpo_max_completion_length):
                continue

            # One GPU->CPU copy per block, not per generated token. Preserve
            # the active mask: pad_token_id is allowed to equal eos_token_id.
            chunk = torch.stack(
                (
                    torch.stack(completions[block_start:step + 1], dim=1),
                    torch.stack(completion_masks[block_start:step + 1], dim=1).long(),
                ),
                dim=-1,
            ).cpu().tolist()

            final_answer_rows = []
            for i, sequence in enumerate(chunk):
                if finish_reasons[i] is not None:
                    continue

                for offset, (tok, is_active) in enumerate(sequence):
                    if not is_active:
                        break

                    generated_ids[i].append(tok)
                    if tok in eos_ids:
                        finish_reasons[i] = "eos"
                        break

                    # This is CPU-only tokenization after the batched copy;
                    # it never synchronizes CUDA on every decoding step.
                    piece = self.tokenizer.decode([tok], skip_special_tokens=False)
                    if "\n" not in piece and "\r" not in piece:
                        continue

                    text_so_far = self.tokenizer.decode(
                        generated_ids[i], skip_special_tokens=True
                    )
                    # Some tokenizer tokens contain a newline *and* text after
                    # it. Inspect the last completed line rather than requiring
                    # the entire decoded prefix to end in a newline.
                    line_end = max(text_so_far.rfind("\n"), text_so_far.rfind("\r"))
                    completed_text = text_so_far[:line_end + 1]
                    if not completed_final_answer_line(completed_text):
                        continue

                    finish_reasons[i] = "final_answer"
                    final_answer_rows.append(i)
                    end_at = block_start + offset
                    # The GPU may have speculatively generated up to seven
                    # tokens after the final answer. Exclude all of them from
                    # GRPO log-probabilities and from future attention.
                    for k in range(end_at + 1, step + 1):
                        completion_masks[k][i] = False
                        completions[k][i] = pad
                        ids[i, width + k] = pad
                        mask[i, width + k] = 0
                    break

            if final_answer_rows:
                finished[final_answer_rows] = True

            all_finished = finished.all().to(dtype=torch.int32)
            if dist.is_initialized():
                dist.all_reduce(all_finished, op=dist.ReduceOp.MIN)
            globally_done = bool(all_finished.item())

            # Live progress printed AFTER the existing 8-token GPU/CPU sync.
            # This adds no per-token device sync or NCCL collective.
            rank = dist.get_rank() if dist.is_initialized() else 0
            if c.trace_mode and rank == 0 and (
                (step + 1) % 16 == 0 or step == 7 or globally_done
            ):
                elapsed = max(time.perf_counter() - rollout_start, 1e-9)
                generated = int(torch.stack(completion_masks, dim=0).sum().item())
                values = dict(
                    phase="eval" if greedy else "train",
                    rank=rank,
                    step=step + 1,
                    max_steps=c.grpo_max_completion_length,
                    wall_s=round(elapsed, 2),
                    generated_tokens=generated,
                    tokens_per_s=round(generated / elapsed, 2),
                    gpu_allocated_gib=round(torch.cuda.memory_allocated(self.device) / 1024**3, 2),
                    gpu_peak_gib=round(torch.cuda.max_memory_allocated(self.device) / 1024**3, 2),
                )
                if not c.progress_bar:
                    print(
                        f"[ROLLOUT LIVE] {values['phase']} "
                        f"step={values['step']}/{values['max_steps']} "
                        f"wall={values['wall_s']:.2f}s "
                        f"tokens={values['generated_tokens']} "
                        f"rate={values['tokens_per_s']:.2f} tokens/s "
                        f"gpu_mem={values['gpu_allocated_gib']:.2f}GiB "
                        f"peak={values['gpu_peak_gib']:.2f}GiB",
                        flush=True,
                    )
                trace.log(c, "rollout.live", **values)

            if globally_done:
                break

            block_start = step + 1

        if c.trace_mode and decode_started:
            decode_end.record()
        if communication_profile is not None and communication_profile.active:
            # Also handle a rollout which ended before 16 decoding steps.
            self._stop_communication_profile(communication_profile, len(completions))

        del cache, output
        completion_ids = torch.stack(completions, dim=1)
        completion_mask = torch.stack(completion_masks, dim=1)

        responses = [
            self.tokenizer.decode(tokens[valid].tolist(), skip_special_tokens=True)
            for tokens, valid in zip(completion_ids, completion_mask)
        ]
        finish_reasons = [reason or "max_tokens" for reason in finish_reasons]
        result = {
            "finish_reasons": finish_reasons,
            "input_ids": ids,
            "attention_mask": mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "prompt_width": width,
            "responses": responses,
        }
        if c.trace_mode:
            torch.cuda.synchronize(self.device)
            rollout_seconds = time.perf_counter() - rollout_start
            prefill_ms = prefill_start.elapsed_time(prefill_end)
            decode_ms = (decode_start.elapsed_time(decode_end)
                         if decode_started else 0.0)
            trace.log(
                c, "rollout.phases",
                rank=dist.get_rank() if dist.is_initialized() else 0,
                prefill_forward_gpu_ms=round(prefill_ms, 3),
                decoding_gpu_timeline_ms=round(decode_ms, 3),
                decoding_steps=max(0, len(completions) - 1),
                decoding_ms_per_step=round(decode_ms / (len(completions) - 1), 3)
                if len(completions) > 1 else None,
                note="GPU timeline includes gaps/host delays; not pure kernel time",
            )
            result["rollout_seconds"] = rollout_seconds
            generated_token_count = int(completion_mask.sum().item())
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0 and not c.progress_bar:
                print(
                    f"[ROLLOUT END] {'eval' if greedy else 'train'} "
                    f"wall={rollout_seconds:.3f}s "
                    f"prefill_forward_gpu={prefill_ms:.3f}ms "
                    f"decode_timeline={decode_ms:.3f}ms "
                    f"steps={len(completions)} "
                    f"tokens={generated_token_count} "
                    f"rate={generated_token_count / max(rollout_seconds, 1e-9):.2f} tokens/s",
                    flush=True,
                )
            trace.log(
                c, "rollout.duration",
                rank=rank,
                seconds=round(rollout_seconds, 3),
                prompts=len(batch),
                completions=len(responses),
                generated_tokens=generated_token_count,
                group_size=g,
                greedy=greedy,
            )
            # Prompts/completions run in parallel: this is the shared batch
            # wall time, not an independent latency for each response.
            trace.batch(
                c, "rollout.sample.duration", batch,
                seconds=round(rollout_seconds, 3),
                group_size=g,
                concurrent_prompts=len(batch),
            )
            trace.generated(c, batch, result, g, greedy)
            trace.batch(
                c, "rollout.stop", batch,
                finish_reasons=finish_reasons,
                generated_tokens=completion_mask.sum(dim=1).tolist(),
            )
        return result

    def train_batch(self, domain: str, batch: list[dict],
                    accumulation_index: int = 0, accumulation_steps: int = 1) -> dict:
        """Delegate one rollout/backward; optimizer step happens on final accumulation."""
        return self.policy.algorithm.train_batch(
            domain, batch, self, accumulation_index, accumulation_steps,
        )
