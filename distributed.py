"""Ray actor cluster, model TP/FSDP2, and distributed batching."""
from __future__ import annotations
import faulthandler
import os
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
import ray
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from config import Config, parse_config
from rollout import Rollout
from continual_train.storage import atomic_write_json
import diagnostics as trace


def load_model(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, revision=config.model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # autoregressive rollout

    tp = config.parallel_mode == "tp"
    kwargs = {}
    if tp:
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        if not dist.is_initialized():
            raise RuntimeError("TP requires NCCL process group initialized by Ray")
        architecture = AutoConfig.from_pretrained(
            config.model_name, revision=config.model_revision
        )
        if not getattr(architecture, "base_model_tp_plan", None):
            raise RuntimeError(f"{config.model_name}: no Hugging Face native TP plan")
        degree = dist.get_world_size()
        if (architecture.num_attention_heads % degree != 0 or
                architecture.num_key_value_heads % degree != 0):
            raise ValueError(
                f"TP={degree} must divide attention heads "
                f"({architecture.num_attention_heads}) and KV heads "
                f"({architecture.num_key_value_heads}). For Qwen3-8B use TP=2 or 4."
            )
        mesh = init_device_mesh("cuda", (degree,), mesh_dim_names=("tp",))
        kwargs.update(tp_plan="auto", device_mesh=mesh)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        dtype=getattr(torch, config.model_dtype),
        attn_implementation=config.model_attn,
        low_cpu_mem_usage=True,
        **kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # overridden explicitly for no-grad rollout

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model, tokenizer

def wrap_fsdp(model: torch.nn.Module, config: Config):
    if not dist.is_initialized() or not torch.cuda.is_available():
        raise RuntimeError("Initialize CUDA and torch.distributed first")

    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Expected Transformer blocks in model.model.layers")
    if not config.fsdp_reshard_after_forward:
        raise ValueError(
            "Memory-conservative FSDP2 requires fsdp_reshard_after_forward=True"
        )

    policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, config.model_dtype),
        reduce_dtype=getattr(torch, config.fsdp_reduce_dtype),
    )

    # Split the large MLP from attention. Keep the rest of each decoder layer
    # in one group to avoid excessive NCCL all-gather calls during generation.
    # fully_shard is intentionally applied bottom-up.
    for layer in layers:
        fully_shard(layer.mlp, mp_policy=policy, reshard_after_forward=True)
        fully_shard(layer, mp_policy=policy, reshard_after_forward=True)

    # Avoid all-gathering the large vocabulary embedding and output head as
    # one root group. Qwen3-8B-Base uses separate (untied) weights.
    embedding = core.embed_tokens
    head = model.lm_head
    if embedding.weight is not head.weight:
        fully_shard(embedding, mp_policy=policy, reshard_after_forward=True)
        fully_shard(head, mp_policy=policy, reshard_after_forward=True)
    # When weights are tied, leave both to the root group so their shared
    # parameter is sharded only once and remains shared.

    fully_shard(model, mp_policy=policy, reshard_after_forward=True)
    return model

class BatchRouter:
    """One batch semantics for train, eval and PCA across TP or FSDP.

    TP workers process the same examples; FSDP workers process disjoint
    slices. Ignore duplicate TP results when computing metrics/embeddings.
    """
    def __init__(self, config, num_workers: int, per_worker: int):
        if min(num_workers, per_worker) < 1:
            raise ValueError("Worker and batch counts must be positive")
        self.num_workers = num_workers
        self.per_worker = per_worker
        self.tp = config.parallel_mode == "tp"
        self.global_size = per_worker if self.tp else per_worker * num_workers

    def split(self, batch: list) -> list[list]:
        if len(batch) != self.global_size:
            raise ValueError(f"Expected {self.global_size} batch items; got {len(batch)}")
        if self.tp:
            return [batch] * self.num_workers
        return [batch[i:i + self.per_worker]
                for i in range(0, len(batch), self.per_worker)]

    def unique(self, results: list) -> list:
        return results[:1] if self.tp else results


def await_workers_progress(
    config,
    requests: list,
    stage: str,
    *,
    bar=None,
    status: str = "",
    no_timeout: bool = False,
    **data,
):
    """Return worker results in submission order; surface exceptions promptly.

    Only completed batches advance sample counters. An active tqdm bar shows
    elapsed time during a long Ray call without claiming samples are finished.
    """
    progress = bool(getattr(config, "progress_bar", True))
    if not progress and not config.trace_mode:
        if no_timeout:
            return ray.get(requests)
        return ray.get(requests, timeout=config.ray_dist_timeout_s)

    pending = list(requests)
    started = time.monotonic()
    last_refresh = -2.0
    last_heartbeat = 0.0

    if config.trace_mode:
        trace.log(config, stage + ".started", workers=len(pending), **data)

    while pending:
        elapsed = time.monotonic() - started
        if not no_timeout and elapsed >= config.ray_dist_timeout_s:
            if config.trace_mode:
                trace.log(config, stage + ".timeout", pending=len(pending),
                          elapsed=round(elapsed), **data)
            raise TimeoutError(f"{stage} timeout: {len(pending)} workers pending")

        wait_seconds = 1.0 if progress else 20.0
        if not no_timeout:
            wait_seconds = min(wait_seconds, max(0.001, config.ray_dist_timeout_s - elapsed))
        ready, pending = ray.wait(pending, num_returns=1, timeout=wait_seconds)
        if ready:
            ray.get(ready)
        elapsed = time.monotonic() - started

        if progress and bar is not None and (elapsed - last_refresh >= 2.0 or not pending):
            phase = f"running {int(elapsed)}s" if pending else f"done {int(elapsed)}s"
            bar.set_postfix_str(f"{status} | {phase}", refresh=True)
            last_refresh = elapsed

        if config.trace_mode and pending and elapsed - last_heartbeat >= 20.0:
            trace.log(config, stage + ".waiting", pending=len(pending),
                      elapsed=round(elapsed), **data)
            if not progress:
                print(f"[TRACE] {stage}: {len(pending)} worker(s) pending", flush=True)
            last_heartbeat = elapsed

    result = ray.get(requests)
    if config.trace_mode:
        trace.log(config, stage + ".completed", workers=len(requests), **data)
    return result

@ray.remote(max_restarts=0)
class PolicyWorker:
    def __init__(self, config: Config):
        self.config = config
        self.store = None
        self.model = None
        self.tokenizer = None
        self.rollout = None

        # Ray node identity is shared by actors on the same Ray node.
        # NCCL uses NCCL_HOSTID to form its host hash, not Ray's node ID.
        self.ray_node_id = str(ray.get_runtime_context().get_node_id())
        os.environ["NCCL_HOSTID"] = f"ray-node-{self.ray_node_id}"

    def nccl_identity(self):
        # Called by the driver *before* initializing any NCCL communicators.
        try:
            with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as f:
                boot_id = f.read().strip()
        except OSError:
            boot_id = None
        return {
            "ray_node_id": self.ray_node_id,
            "nccl_hostid": os.environ.get("NCCL_HOSTID"),
            "hostname": socket.gethostname(),
            "boot_id": boot_id,
            "shm_dev": os.stat("/dev/shm").st_dev,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def start_rendezvous(self, world_size: int):
        address = ray.util.get_node_ip_address()
        self.store = dist.TCPStore(
            host_name=address,
            port=0,
            world_size=world_size,
            is_master=True,
            timeout=timedelta(seconds=self.config.ray_dist_timeout_s),
            wait_for_workers=False,
        )
        return address, self.store.port

    def initialize(self, rank: int, world_size: int, address: str, port: int):
        if self.config.trace_mode:
            # Dump the Python stacks of this worker every 60s while initializing.
            # Ray captures stderr in /tmp/ray/session_latest/logs/worker-*.err.
            faulthandler.enable(file=sys.stderr, all_threads=True)
            faulthandler.dump_traceback_later(
                60, repeat=True, file=sys.stderr
            )

        try:
            if self.config.trace_mode:
                trace.log(
                    self.config, "worker.initialize.start", rank=rank,
                    visible_gpus=os.getenv("CUDA_VISIBLE_DEVICES"),
                )

            torch.cuda.set_device(0)
            if self.config.ray_gpu_memory_fraction is not None:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.ray_gpu_memory_fraction, device=0
                )
            torch.set_num_threads(max(1, int(self.config.ray_cpus_per_worker)))
            torch.manual_seed(self.config.train_seed + rank)
            torch.cuda.manual_seed_all(self.config.train_seed + rank)

            if rank != 0:
                self.store = dist.TCPStore(
                    host_name=address,
                    port=port,
                    world_size=world_size,
                    is_master=False,
                    timeout=timedelta(seconds=self.config.ray_dist_timeout_s),
                )

            dist.init_process_group(
                backend="nccl",
                store=self.store,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=self.config.ray_dist_timeout_s),
            )

            if self.config.trace_mode:
                trace.log(self.config, "worker.model.load.start", rank=rank)
            if self.config.parallel_mode == "tp":
                # HF 4.56 native tensor-parallel loader reads LOCAL_RANK even
                # when torch.distributed has already been initialized.
                # Ray exposes one assigned GPU per actor as local cuda:0.
                os.environ["LOCAL_RANK"] = "0"
            self.model, self.tokenizer = load_model(self.config)

            # DTensor.numel() reports logical/global parameter elements;
            # do not sum local shards across TP ranks.
            total_params = sum(
                p.numel() for p in self.model.parameters()
            )

            trainable_params = sum(
                p.numel()
                for p in self.model.parameters()
                if p.requires_grad
            )

            if self.config.parallel_mode == "fsdp":
                if self.config.trace_mode:
                    trace.log(self.config, "worker.fsdp.start", rank=rank)
                self.model = wrap_fsdp(self.model, self.config)
                if self.config.trace_mode:
                    trace.log(self.config, "worker.fsdp.done", rank=rank)
            else:
                # Ensure native TP actually sharded some weights. Otherwise the
                # run could silently train full copies with no gradient sync.
                from torch.distributed.tensor import DTensor
                if not any(isinstance(p, DTensor) for p in self.model.parameters()):
                    raise RuntimeError("TP requested but model has no DTensor parameters")
                if self.config.trace_mode:
                    trace.log(self.config, "worker.tp.ready", rank=rank,
                              tp_degree=world_size)

            self.rollout = Rollout(self.model, self.tokenizer, self.config)
            if self.config.trace_mode:
                trace.log(self.config, "worker.barrier.start", rank=rank)
            dist.barrier(device_ids=[0])
            if self.config.trace_mode:
                trace.log(self.config, "worker.barrier.done", rank=rank)
                trace.log(self.config, "worker.ready", rank=rank)

            return {
                "rank": rank,
                "world_size": world_size,
                "visible_gpus": os.getenv("CUDA_VISIBLE_DEVICES"),
                "algorithm": self.config.train_algorithm,
                "fsdp_ready": self.config.parallel_mode == "fsdp",
                "parallel_mode": self.config.parallel_mode,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            }
        finally:
            if self.config.trace_mode:
                faulthandler.cancel_dump_traceback_later()

    def rollout_batch(self, domain: str, batch: list[dict],
                      accumulation_index: int = 0, accumulation_steps: int = 1):
        if self.rollout is None:
            raise RuntimeError("Rollout not initialized")
        return self.rollout.train_batch(domain, batch, accumulation_index, accumulation_steps)

    def cleanup_gpu(self):
        import gc
        if self.rollout is not None:
            self.rollout.policy.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    def embed_samples(self, samples: list[dict]):
        # Visualization owns embedding extraction; Ray only dispatches the call.
        from visualization import embed_worker
        return embed_worker(self.rollout, samples)

    @torch.no_grad()
    def begin_sampling_projection(self, seed: int):
        """One frozen GPU projector per Task; identical weights on both TP ranks."""
        import math

        if getattr(self, "_sampling_projector", None) is not None:
            raise RuntimeError("Previous Task's sampling projector was not released")
        hidden_size = int(self.model.config.hidden_size)
        dim = self.config.sampling_projection_dim
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # F.linear uses [output_dim, input_dim]; fixed weight is GPU-only during embedding.
        # No nn.Linear.reset_parameters(): GRPO's CUDA RNG state stays unchanged.
        weights = torch.randn((hidden_size, dim), generator=generator,
                              dtype=torch.float32) / math.sqrt(dim)
        self._sampling_projector = weights.T.contiguous().to(self.rollout.device)
        self._sampling_projector_seed = seed

    @torch.no_grad()
    def embed_sampling_batch(self, samples: list[dict], seed: int):
        """Run Qwen backbone + 256-dim Linear entirely on GPU; return CPU vectors."""
        projector = getattr(self, "_sampling_projector", None)
        if projector is None or seed != self._sampling_projector_seed:
            raise RuntimeError("Call begin_sampling_projection() before embedding")

        tokenizer = self.tokenizer
        max_len = self.config.sampling_max_length
        tokens = [tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=True,
        ) for row in samples]
        tokens = [(t[:max_len // 2] + t[-(max_len - max_len // 2):]
                   if len(t) > max_len else t) for t in tokens]
        if not tokens or any(not t for t in tokens):
            raise ValueError("Empty embedding input")
        width = max(map(len, tokens))
        ids = torch.full((len(tokens), width), tokenizer.pad_token_id,
                         dtype=torch.long, device=self.rollout.device)
        mask = torch.zeros_like(ids)
        for i, values in enumerate(tokens):
            ids[i, -len(values):] = torch.tensor(values, dtype=torch.long,
                                                 device=self.rollout.device)
            mask[i, -len(values):] = 1
        was_training = self.model.training
        self.model.eval()
        try:
            output = self.model.model(input_ids=ids, attention_mask=mask,
                                      use_cache=False, return_dict=True)
            hidden = output.last_hidden_state
            pooled = (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
            return torch.nn.functional.linear(pooled, projector).cpu().tolist()
        finally:
            self.model.train(was_training)

    def end_sampling_projection(self):
        """Free the auxiliary GPU module before any GRPO backward/optimizer step."""
        import gc
        if hasattr(self, "_sampling_projector"):
            del self._sampling_projector
            del self._sampling_projector_seed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def seed_domain(self, task_id: int):
        seed = (self.config.train_seed + task_id * 10000 +
                (0 if self.config.parallel_mode == "tp" else dist.get_rank()))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def save_checkpoint(self, path: str):
        from continual_train.storage import save_checkpoint
        save_checkpoint(self.model, self.rollout.policy.optimizer, path)

    def load_checkpoint(self, path: str):
        from continual_train.storage import load_checkpoint
        load_checkpoint(self.model, self.rollout.policy.optimizer, path)

    def evaluate(self, domain: str, samples: list[dict]):
        from continual_train.evaluate import evaluate_worker
        return evaluate_worker(self.rollout,domain,samples,)
    
    def shutdown(self):
        self.rollout = None
        self.model = None
        self.tokenizer = None
        if dist.is_initialized():
            dist.destroy_process_group()
        self.store = None
        torch.cuda.empty_cache()


def launch_ray(config: Config):
    root = Path(__file__).resolve().parent  # distributed.py lives in project root
    started_ray = not ray.is_initialized()
    if started_ray:
        options = {
            "address": config.ray_address,
            "runtime_env": {
                "working_dir": str(root),
                "excludes": [".git/", "data/", "outputs/", ".venv/", ".env", "config.sh"],
            },
        }
        if config.ray_address is None and not os.getenv("RAY_ADDRESS"):
            if config.ray_init_cpus is not None:
                options["num_cpus"] = config.ray_init_cpus
            if config.ray_init_gpus is not None:
                options["num_gpus"] = config.ray_init_gpus
            if config.ray_object_store_memory_gb is not None:
                options["object_store_memory"] = int(
                    config.ray_object_store_memory_gb * 1024**3
                )
        ray.init(**options)

    bundle = {
        "CPU": config.ray_cpus_per_worker,
        "GPU": config.ray_gpus_per_worker,
    }
    actor_options = {
        "num_cpus": config.ray_cpus_per_worker,
        "num_gpus": config.ray_gpus_per_worker,
    }
    if config.ray_worker_memory_gb is not None:
        memory = int(config.ray_worker_memory_gb * 1024**3)
        bundle["memory"] = memory
        actor_options["memory"] = memory

    group = placement_group(
        [bundle.copy() for _ in range(config.ray_num_workers)],
        strategy=config.ray_placement_strategy,
    )
    workers = []
    try:
        ray.get(group.ready(), timeout=config.ray_pg_timeout_s)
        for rank in range(config.ray_num_workers):
            workers.append(
                PolicyWorker.options(
                    **actor_options,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=group,
                        placement_group_bundle_index=rank,
                    ),
                ).remote(config)
            )

        # Fail fast before NCCL init if actors aren't on the intended host.
        identities = ray.get(
            [worker.nccl_identity.remote() for worker in workers],
            timeout=config.ray_dist_timeout_s,
        )
        for rank, identity in enumerate(identities):
            print(f"[NCCL IDENTITY] rank={rank}: {identity}", flush=True)

        if config.ray_placement_strategy == "STRICT_PACK":
            if len({item["ray_node_id"] for item in identities}) != 1:
                raise RuntimeError("PolicyWorkers are on different Ray nodes")
            if len({item["shm_dev"] for item in identities}) != 1:
                raise RuntimeError(
                    "PolicyWorkers have different /dev/shm devices; "
                    "NCCL local P2P/SHM transport may be unavailable"
                )

        address, port = ray.get(
            workers[0].start_rendezvous.remote(config.ray_num_workers),
            timeout=config.ray_dist_timeout_s,
        )
        status = ray.get(
            [
                worker.initialize.remote(rank, config.ray_num_workers, address, port)
                for rank, worker in enumerate(workers)
            ],
            timeout=config.ray_dist_timeout_s,
        )
        
        counts = {
            (item["total_parameters"], item["trainable_parameters"])
            for item in status
        }

        if len(counts) != 1:
            raise RuntimeError("Parameter counts differ across Ray workers")

        total, trainable = counts.pop()
        if total <= 0:
            raise RuntimeError("Model has no parameters")

        summary = {
            "model": config.model_name,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_percent": round(100 * trainable / total, 4),
            "fsdp_workers": len(workers) if config.parallel_mode == "fsdp" else 0,
            "tp_degree": len(workers) if config.parallel_mode == "tp" else 1,
            "parallel_mode": config.parallel_mode,
            "algorithm": config.train_algorithm,
        }

        output = root / "outputs" / f"summary_{config.train_algorithm}.json"
        output.parent.mkdir(parents=True, exist_ok=True)

        atomic_write_json(output, summary, indent=2)

        print(
            f"[PARAMETERS] {trainable:,}/{total:,} "
            f"({summary['trainable_percent']:.2f}%) trainable",
            flush=True,
        )

        for item in status:
            print(item)

        return workers, group

    except Exception:
        for worker in workers:
            ray.kill(worker, no_restart=True)
        remove_placement_group(group)
        if started_ray:
            ray.shutdown()
        raise


def shutdown_ray(workers, group, config: Config):
    try:
        ray.get(
            [worker.shutdown.remote() for worker in workers],
            timeout=config.ray_dist_timeout_s,
        )
    finally:
        for worker in workers:
            ray.kill(worker, no_restart=True)
        remove_placement_group(group)
        ray.shutdown()
