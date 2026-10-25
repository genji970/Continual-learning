"""Opt-in distributed training trace. Core training remains unchanged.

Writes JSONL with one atomic append per event. For multi-node Ray, trace_dir
must point to shared storage; local output works for a single-node cluster.
"""
from __future__ import annotations

import json
from itertools import islice
import os
from datetime import datetime, timezone
from pathlib import Path


def log(cfg, stage: str, row: dict | None = None, **data):
    """Append event to timeline and, if tagged, to its per-sample history."""
    if not cfg.trace_mode:
        return
    # Tensor-parallel workers operate on the same prompts. Keep one full
    # per-sample JSONL trace (rank 0); untagged rank-wise profiler metrics
    # and driver-side traces are retained for every participating rank.
    if row is not None and getattr(cfg, "parallel_mode", "fsdp") == "tp":
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    entry = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "pid": os.getpid(),
        **data,
    }
    paths = [Path(cfg.trace_dir) / "timeline.jsonl"]
    if row is not None and "_trace_id" in row:
        entry.update(
            sample_id=row["_trace_id"],
            domain=row["_trace_domain"],
            phase=row["_trace_phase"],
        )
        if "_trace_task" in row:
            entry["task"] = row["_trace_task"]
        paths.append(Path(cfg.trace_dir) / "samples" / f"{row['_trace_id']}.jsonl")
    payload = (json.dumps(entry, ensure_ascii=False, default=str) + "\n").encode()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)


def start(cfg):
    if not cfg.trace_mode:
        return
    run = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    cfg.trace_dir = (Path(cfg.trace_dir) / f"{run}_{os.getpid()}").resolve()
    log(cfg, "run.start", model=cfg.model_name, workers=cfg.ray_num_workers,
        domains=cfg.train_domains, max_completion=cfg.grpo_max_completion_length)
    print(f"[TRACE] {cfg.trace_dir}", flush=True)


def batch(cfg, stage: str, rows: list[dict], **data):
    for row in rows:
        if "_trace_id" in row:
            log(cfg, stage, row, **data)


def choose(cfg, rows: list[dict], phase: str, domain: str,
           indices: list[int] | range | None = None):
    """Tag up to five genuine, scheduled examples; no additional sampling."""
    for order, index in enumerate(islice(indices if indices is not None else range(len(rows)), 5), 1):
        row = rows[index]
        row.update(_trace_id=f"{phase}_{domain}_{order:02d}",
                   _trace_phase=phase, _trace_domain=domain)
        log(cfg, "data.sample", row, source_index=index,
            messages=row["messages"], answer=row["answer"],
            answer_letter=row["answer_letter"], reference=row["reference"])


def generated(cfg, rows: list[dict], output: dict, group_size: int, greedy: bool):
    """Log all completions belonging to each traced prompt."""
    for i, row in enumerate(rows):
        if "_trace_id" not in row:
            continue
        begin = i * group_size
        responses = output["responses"][begin:begin + group_size]
        lengths = output["completion_mask"][begin:begin + group_size].sum(-1).tolist()
        log(cfg, "rollout.finished", row, greedy=greedy,
            responses=responses, generated_tokens=lengths,
            max_tokens=cfg.grpo_max_completion_length)


def rewards(cfg, rows: list[dict], output: dict):
    """Explain missing reward without modifying the reward calculation."""
    g = cfg.grpo_group_size
    values = output["rewards"].tolist()
    advantages = output["advantages"].tolist()
    for i, row in enumerate(rows):
        if "_trace_id" not in row:
            continue
        start = i * g
        responses = output["responses"][start:start + g]
        # Keep trace extraction identical to the real GRPO reward logic.
        from policy import extract_final_answer
        extracted = [extract_final_answer(text) for text in responses]
        scores = output.get("reward_breakdowns", [])[start:start + g]
        log(cfg, "reward.checked", row,
            expected=row.get("answer_letter") or row.get("answer"),
            found_final_answer=[x is not None for x in extracted],
            extracted_final_answer=extracted,
            reward=values[start:start + g],
            advantage=advantages[start:start + g],
            correctness_reward=[x["correctness_reward"] for x in scores],
            format_reward=[x["format_reward"] for x in scores],
            cleanliness_reward=[x["cleanliness_reward"] for x in scores],
            post_answer_chars=[x["post_answer_chars"] for x in scores],
            finish_reasons=output.get("finish_reasons", [])[start:start + g])



def update(cfg, rows: list[dict], stage: str, **data):
    batch(cfg, f"policy.{stage}", rows, **data)


def evaluated(cfg, rows: list[dict], output: dict, domain: str, reward_fn):
    for row, response in zip(rows, output["responses"]):
        if "_trace_id" in row and row.get("_eval_valid", True):
            log(cfg, "eval.sample", row,
                expected=row.get("answer_letter") or row["answer"],
                response=response,
                correct=reward_fn(domain, response, row))

