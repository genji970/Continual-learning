"""Reward verification, GRPO objective and policy optimizer.

SelfImprovePolicy is intentionally NOT implemented; selecting it raises.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import re
import time
from decimal import Decimal, InvalidOperation
import torch
import torch.distributed as dist
from config import Config
import diagnostics as trace


_FINAL_ANSWER = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:[-*][ \t]+)?"
    r"(?:\*\*|__)?(?:(?:thus|therefore)[,:]?[ \t]+)?"
    r"(?:the[ \t]+)?(?:final[ \t_-]+)?answer\b"
    r"[ \t]*(?:\*\*|__)?[ \t]*"
    r"(?:(?:is[ \t]+)?[:=\-–—][ \t]*|is[ \t]+)?"
    r"(?P<answer>[^\r\n]*)"
)


_REPEATED_ANSWER = re.compile(
    r"(?i)\s+(?:(?:the[ \t]+)?final[ \t_-]+)?answer[ \t]*(?::|=|is\b)"
)


_PLACEHOLDERS = {"<answer>", "<your answer>", "<final answer>", "<result>"}


_CANONICAL_FORMAT = re.compile(r"Final Answer ?: ")


_NEW_TASK = re.compile(
    r"(?im)(?:^|\n)\s*(?:system|user|assistant|question|problem)\s*[:#]"
    r"|\b(?:next|another|new)\s+(?:question|problem)\b"
)


def _first_final_answer(response: str):
    """Return (answer, label_match, post_answer_text), using the FIRST answer.

    Separate a repeated Final Answer on the same line into post-answer text.
    No fallback to numbers appearing in reasoning or another question.
    """
    for match in _FINAL_ANSWER.finditer(response):
        raw = match.group("answer")
        repeated = _REPEATED_ANSWER.search(raw)
        before_repeat = raw[:repeated.start()] if repeated else raw
        candidate = before_repeat.strip().strip("*_").strip()
        answer_end = match.start("answer") + len(before_repeat)

        if not candidate:
            # Permit Final Answer:\n4 without accepting an arbitrary paragraph.
            tail = response[match.end():]
            stripped = tail.lstrip(" \t\r\n")
            line = stripped.splitlines()[0].strip() if stripped else ""
            if len(line) <= 80 and len(line.split()) <= 10:
                candidate = line.strip("*_").strip()
                answer_end = len(response) - len(stripped) + len(line)

        if candidate and candidate.casefold() not in _PLACEHOLDERS:
            return candidate, match, response[answer_end:]
    return None


def extract_final_answer(response: str) -> str | None:
    """First explicitly labeled, non-placeholder answer, not the last one."""
    parsed = _first_final_answer(response)
    return parsed[0] if parsed else None


def completed_final_answer_line(response: str) -> bool:
    """Stop after an answer line is closed by a newline (not mid-number)."""
    return response.endswith(("\n", "\r")) and extract_final_answer(response) is not None


def _normalize_answer(x: str) -> str:
    x = str(x).strip().strip("*_").strip()
    if x.startswith("$") and x.endswith("$"):
        x = x[1:-1].strip()
    if x.startswith(r"\(") and x.endswith(r"\)"):
        x = x[2:-2].strip()
    if x.startswith(r"\boxed{") and x.endswith("}"):
        x = x[7:-1]
    x = x.rstrip(".,;:").strip()
    return re.sub(r"\s+", " ", x).casefold()


def _is_correct(extracted: str, row: dict) -> bool:
    guess = _normalize_answer(extracted)
    truth = _normalize_answer(row.get("answer_letter") or row["answer"])
    if guess == truth:
        return True
    try:
        numeric_guess = re.sub(r"\s+(?:digits?|units?)$", "", guess)
        return Decimal(numeric_guess.replace(",", "")) == Decimal(truth.replace(",", ""))
    except InvalidOperation:
        return False


def reward_breakdown(domain: str, response: str, row: dict, *,
                     finish_reason: str | None = None) -> dict:
    """Separate correctness, canonical final-answer format and clean stopping.

    Correctness dominates both style components combined, so a wrong but
    perfectly formatted answer never outranks a correct answer with poor style.
    Total: 3 * correct (0/1) + format (0/0.5/1) + cleanliness (0..1).
    """
    if domain == "coding":
        raise NotImplementedError(
            "Coding reward needs an isolated, trusted unit-test verifier"
        )
    parsed = _first_final_answer(response)
    if parsed is None:
        return {
            "extracted_final_answer": None,
            "correctness_reward": 0.0,
            "format_reward": 0.0,
            "cleanliness_reward": 0.0,
            "post_answer_chars": 0,
            "total_reward": 0.0,
        }

    extracted, match, tail = parsed
    repeated = bool(_REPEATED_ANSWER.search(match.group("answer"))) or bool(
        _FINAL_ANSWER.search(tail)
    )
    header = response[match.start():match.start("answer")]
    # Canonical accepts both 'Final Answer: 4' and 'Final Answer : 4'.
    format_reward = 1.0 if _CANONICAL_FORMAT.fullmatch(header) and not repeated else 0.5
    correctness = float(_is_correct(extracted, row))

    # Whitespace/newline after a final answer is fine; *generated content* is not.
    extra = tail.strip()
    if extra:
        cleanliness = 1.0 / (1.0 + len(extra) / 48.0)
        if repeated:
            cleanliness *= 0.5
        if _NEW_TASK.search(extra):
            cleanliness *= 0.25
    else:
        cleanliness = 1.0

    # A max-length truncation is not a clean semantic stopping event.
    if finish_reason == "max_tokens":
        cleanliness *= 0.5

    return {
        "extracted_final_answer": extracted,
        "correctness_reward": correctness,
        "format_reward": format_reward,
        "cleanliness_reward": round(cleanliness, 6),
        "post_answer_chars": len(extra),
        "total_reward": round(3.0 * correctness + format_reward + cleanliness, 6),
    }


def evaluation_correctness(domain: str, response: str, row: dict, *,
                           finish_reason: str | None = None) -> float:
    """Shared, algorithm-independent accuracy for the continual benchmark."""
    return reward_breakdown(
        domain, response, row, finish_reason=finish_reason
    )["correctness_reward"]

class TrainingAlgorithm(ABC):
    """Shared contract for interchangeable policy algorithms."""

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def train_batch(self, domain: str, batch: list[dict], engine) -> dict:
        """Generate, score and update for one scheduled batch."""


class GRPOPolicy(TrainingAlgorithm):
    """Reward, group advantage and clipped policy objective."""

    def advantages(self, rewards: torch.Tensor, group_size: int):
        grouped = rewards.reshape(-1, group_size)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True, unbiased=False)
        return ((grouped - mean) / (std + 1e-4)).flatten()

    def loss(self, logps: torch.Tensor, rollout: dict):
        ratio = (logps - rollout["old_logps"]).clamp(-20, 20).exp()
        clipped = ratio.clamp(
            1 - self.config.grpo_clip_range,
            1 + self.config.grpo_clip_range,
        )
        advantage = rollout["advantages"][:, None]
        objective = torch.minimum(ratio * advantage, clipped * advantage)
        mask = rollout["completion_mask"].to(objective.dtype)
        return -((objective * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()

    def train_batch(self, domain: str, batch: list[dict], engine,
                    accumulation_index: int = 0, accumulation_steps: int = 1) -> dict:
        """GRPO rollout and backward; step only at the last accumulation index."""
        if domain == "coding":
            raise NotImplementedError("Coding reward verifier is not yet available")

        if engine.config.trace_mode:
            torch.cuda.synchronize(engine.device)
            batch_start = time.perf_counter()

        rollout = engine.generate(batch)
        breakdowns = [
            reward_breakdown(
                domain, response, batch[i // engine.config.grpo_group_size],
                finish_reason=rollout["finish_reasons"][i],
            )
            for i, response in enumerate(rollout["responses"])
        ]
        rewards = torch.tensor(
            [item["total_reward"] for item in breakdowns],
            dtype=torch.float32,
            device=engine.device,
        )
        rollout["reward_breakdowns"] = breakdowns
        rollout["rewards"] = rewards
        rollout["advantages"] = self.advantages(
            rewards, engine.config.grpo_group_size
        )
        if engine.config.trace_mode:
            trace.rewards(engine.config, batch, rollout)
            rollout["_trace_rows"] = [r for r in batch if "_trace_id" in r]
            trace.batch(engine.config, "policy.old_logps.start", batch)
        with torch.no_grad():
            rollout["old_logps"] = engine.policy.token_logps(rollout).detach()
        if engine.config.trace_mode:
            trace.batch(engine.config, "policy.old_logps.finished", batch)

        metrics = engine.policy.update(
            rollout, accumulation_index=accumulation_index,
            accumulation_steps=accumulation_steps,
        )
        if engine.config.trace_mode:
            trace.update(engine.config, rollout["_trace_rows"], "result", metrics=metrics)
        metrics.update({
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "completions": int(rewards.numel()),
            "active_groups": float(
                rollout["advantages"].abs()
                .reshape(len(batch), -1).sum(dim=1).gt(0)
                .float().mean().item()
            ),
        })
        if engine.config.trace_mode:
            torch.cuda.synchronize(engine.device)
            batch_seconds = time.perf_counter() - batch_start
            metrics["batch_seconds"] = round(batch_seconds, 3)
            metrics["rollout_seconds"] = round(rollout["rollout_seconds"], 3)
            trace.log(
                engine.config, "train.batch.duration",
                rank=dist.get_rank() if dist.is_initialized() else 0,
                domain=domain,
                seconds=round(batch_seconds, 3),
                rollout_seconds=round(rollout["rollout_seconds"], 3),
                prompts=len(batch),
                completions=len(rollout["responses"]),
                skipped=bool(metrics.get("skipped", False)),
            )
            # If train_batch_size > 1, all prompts share the same batch time.
            trace.batch(
                engine.config, "train.sample.duration", batch,
                seconds=round(batch_seconds, 3),
                rollout_seconds=round(rollout["rollout_seconds"], 3),
                concurrent_prompts=len(batch),
            )
        return metrics

class SelfImprovePolicy(TrainingAlgorithm):
    IMPLEMENTED = False

    def train_batch(self, domain: str, batch: list[dict], rollout):
        """One real optimizer step for the current domain and batch.

        Return a dict of numeric metrics compatible with the shared trainer.
        The step budget and task order are enforced by continual_train.train.
        """
        raise NotImplementedError(
            "SelfImprovePolicy.train_batch is not implemented. "
            "Implement your own self-improvement algorithm in "
            "train/self_improve_policy.py; it must not silently call GRPO."
        )

class DistanceSamplingGRPOPolicy(GRPOPolicy):
    """Same reward, group advantage, objective, optimizer as GRPO.

    Only the driver-side selection of training prompts changes.
    """


class PolicyTrainer:
    def __init__(self, model, tokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        grpo_cls = (GRPOPolicy if config.train_sampler == "random"
                    else DistanceSamplingGRPOPolicy)
        algorithms = {"grpo": grpo_cls, "self_improve": SelfImprovePolicy}
        try:
            self.algorithm = algorithms[config.train_algorithm](config)
        except KeyError as exc:
            raise ValueError(f"Unknown algorithm: {config.train_algorithm}") from exc
        # FSDP and TP both expose sharded model parameters to AdamW.
        # foreach=False avoids a large transient tensor-list allocation.
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train_lr,
            weight_decay=config.train_weight_decay,
            foreach=False if config.parallel_mode == "tp" else None,
        )

    def _sync_tp_attention_norm_gradients(self) -> None:
        """Correct Qwen3 TP q_norm/k_norm gradients on transformers 4.56.

        These RMSNorm weights are shared by attention heads, but the 4.56
        native TP plan shards q/k projections without the newer
        replicated_with_grad_allreduce norm strategy. Each rank sees only
        some heads, so the per-head contributions must be summed across TP.
        """
        from torch.distributed.tensor import DTensor

        plan = getattr(self.model.config, "base_model_tp_plan", {}) or {}
        if plan.get("layers.*.self_attn.q_norm") == "replicated_with_grad_allreduce":
            return  # Future Transformers release already synchronizes these.
        for name, parameter in self.model.named_parameters():
            if (name.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight"))
                    and parameter.grad is not None):
                if isinstance(parameter.grad, DTensor):
                    raise RuntimeError("Unexpected sharded Qwen3 attention norm gradient")
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)

    def _clip_tp_grad_norm(self) -> torch.Tensor:
        """Global L2 norm of TP shards; count replicated parameters only once.

        A local clip_grad_norm_ would incorrectly ignore gradient shards
        on other ranks, potentially causing a different scale per rank.
        """
        from torch.distributed.tensor import DTensor, Replicate

        device = next(self.model.parameters()).device
        sq = torch.zeros((), dtype=torch.float32, device=device)
        rank = dist.get_rank()
        for parameter in self.model.parameters():
            grad = parameter.grad
            if grad is None:
                continue
            if isinstance(grad, DTensor):
                local = grad.to_local()
                replicated = all(isinstance(p, Replicate) for p in grad.placements)
            else:
                local = grad
                replicated = True
            if not replicated or rank == 0:
                sq += local.float().square().sum()
        dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        norm = sq.sqrt()
        if not torch.isfinite(norm).all():
            raise RuntimeError("Nonfinite tensor-parallel gradient norm")
        coefficient = (self.config.train_grad_clip / (norm + 1e-6)).clamp(max=1)
        for parameter in self.model.parameters():
            grad = parameter.grad
            if grad is None:
                continue
            if isinstance(grad, DTensor):
                # Work on local shards; avoid mixing a local scalar Tensor with
                # a DTensor in elementwise tensor-parallel operations.
                grad.to_local().mul_(coefficient)
            else:
                grad.mul_(coefficient)
        return norm

    def token_logps(self, rollout: dict) -> torch.Tensor:
        start = rollout["prompt_width"] - 1
        length = rollout["completion_ids"].shape[1]
        logits = self.model(
            input_ids=rollout["input_ids"],
            attention_mask=rollout["attention_mask"],
            use_cache=False,
            logits_to_keep=length + 1,  # prompt-last + all completion positions
        ).logits[:, :length, :].float()
        return (
            logits.gather(-1, rollout["completion_ids"].unsqueeze(-1))
            .squeeze(-1)
            - logits.logsumexp(dim=-1)
        )

    def update(self, rollout: dict, *, accumulation_index: int = 0,
               accumulation_steps: int = 1) -> dict:
        """Accumulate scaled gradients; clip and AdamW-step once per full group.

        TP ranks must enter exactly the same collectives at every microbatch.
        An all-zero-advantage microbatch contributes zero without erasing
        gradients from earlier microbatches.
        """
        if self.config.train_algorithm != "grpo":
            raise RuntimeError("Custom algorithm manages its own updates")
        if not 0 <= accumulation_index < accumulation_steps:
            raise ValueError("Invalid gradient accumulation index")
        if accumulation_steps > 1 and self.config.grpo_updates_per_rollout != 1:
            raise ValueError("Gradient accumulation requires grpo_updates_per_rollout=1")

        if accumulation_index == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_active = False

        active = torch.tensor(
            [int(rollout["advantages"].abs().sum().item() > 0)],
            device=rollout["advantages"].device,
        )
        if dist.is_initialized():
            dist.all_reduce(active, op=dist.ReduceOp.SUM)
        is_active = bool(active.item())

        metrics = {"loss": 0.0, "grad_norm": 0.0, "skipped": not is_active}
        for iteration in range(self.config.grpo_updates_per_rollout):
            if iteration and accumulation_steps == 1:
                self.optimizer.zero_grad(set_to_none=True)
                self._accum_active = False
            if is_active:
                self.model.train()
                if self.config.trace_mode:
                    trace.update(self.config, rollout.get("_trace_rows", []), "forward.start")
                logps = self.token_logps(rollout)
                loss = self.algorithm.loss(logps, rollout)
                if self.config.trace_mode:
                    trace.update(self.config, rollout.get("_trace_rows", []),
                                 "backward.start", loss=float(loss.detach().item()))
                (loss / accumulation_steps).backward()
                self._accum_active = True
                metrics["loss"] = float(loss.detach().item())
                if self.config.trace_mode:
                    trace.update(self.config, rollout.get("_trace_rows", []), "backward.finished")

            if accumulation_index == accumulation_steps - 1 and self._accum_active:
                if self.config.parallel_mode == "tp":
                    self._sync_tp_attention_norm_gradients()
                norm = (self._clip_tp_grad_norm()
                        if self.config.parallel_mode == "tp"
                        else torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.config.train_grad_clip,
                            error_if_nonfinite=True,
                        ))
                if self.config.trace_mode:
                    trace.update(self.config, rollout.get("_trace_rows", []),
                                 "optimizer.start", grad_norm=float(norm.detach().item()))
                self.optimizer.step()
                if self.config.trace_mode:
                    trace.update(self.config, rollout.get("_trace_rows", []), "optimizer.finished")
                metrics["grad_norm"] = float(norm.detach().item())
                metrics["skipped"] = False
                metrics["optimizer_step"] = True
            else:
                metrics["optimizer_step"] = False
        return metrics
