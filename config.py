
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path


MATH_CONFIGS = (
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory",
    "prealgebra", "precalculus",
)

DATASET_REPOS = {
    "amc": "MilaWang/amc-test-23",
    "minerva": "zwhe99/simplerl-minerva-math",
    "math": "EleutherAI/hendrycks_math",
    "gsm8k": "openai/gsm8k",
    "olympiad": "ScaleFrontierData/olympiadbench",
    "aime25": "MathArena/aime_2025",
    "aime24": "HuggingFaceH4/aime_2024",
    "supergpqa": "m-a-p/SuperGPQA",
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "bbeh": "BBEH/bbeh",
    "math500": "HuggingFaceH4/MATH-500",
    "sciq": "allenai/sciq",
    "scienceqa_social": "derek-thomas/ScienceQA",
    "finqa": "bevaya/FinQA",
    "mbpp": "huggy-1/mbpp",
}

DATASET_SUBSETS = {
    "gsm8k": "main",
    "mbpp": "full",
}

DATASET_SPLITS = {
    "amc": {"train": None, "test": "test"},
    "minerva": {"train": None, "test": "test"},
    "math": {"train": "train", "test": "test"},
    "gsm8k": {"train": "train", "test": "test"},
    "olympiad": {"train": "train", "test": "train"},
    "aime25": {"train": "train", "test": "train"},
    "aime24": {"train": "train", "test": "train"},
    "supergpqa": {"train": "train", "test": "train"},
    "mmlu_pro": {"train": None, "test": "test"},
    "bbeh": {"train": "train", "test": "train"},
    "math500": {"train": None, "test": "test"},
    "sciq": {"train": "train", "test": "test"},
    "scienceqa_social": {"train": "train", "test": "test"},
    "finqa": {"train": "train", "test": "test"},
    "mbpp": {"train": "train", "test": "test"},
}

DOMAIN_TASKS = {
    "mathematics": ["gsm8k", "math"],
    "science": ["sciq"],
    "social_science": ["scienceqa_social"],
    "finance": ["finqa"],
    "coding": ["mbpp"],
}


_BOOLEANS = {
    "plain", "strict", "hf_upload",
    "gradient_checkpointing", "fsdp_reshard_after_forward",
    "train_shuffle", "continual_resume", "trace_mode", "progress_bar",
    "benchmark_mode", "benchmark_forward_transfer", "visualization_mode",
    "eval_skip_completed",
    }
_INTEGERS = {
    "limit", "debug_samples", "ray_num_workers",
    "ray_gpus_per_worker", "ray_init_cpus", "ray_init_gpus",
    "ray_pg_timeout_s", "ray_dist_timeout_s",
    "train_epochs", "train_batch_size", "train_finance_batch_size",
    "train_finance_gradient_accumulation", "train_max_steps",
    "train_seed", "grpo_group_size", "grpo_max_prompt_length",
    "grpo_max_completion_length", "grpo_updates_per_rollout",
    "continual_eval_samples_per_worker", "continual_eval_limit",
    "eval_microbatch_size", "eval_finance_microbatch_size", "eval_example_count",
    "visualization_samples_per_domain", "visualization_max_length",
    "visualization_batch_size",
    "sampling_projection_dim", "sampling_embed_batch_size", "sampling_max_length",
    "sampling_knn_k", "sampling_projector_seed",
    }
_FLOATS = {
    "ray_cpus_per_worker", "ray_worker_memory_gb",
    "ray_gpu_memory_fraction", "ray_object_store_memory_gb",
    "train_lr", "train_weight_decay", "train_grad_clip",
    "grpo_temperature", "grpo_top_p", "grpo_clip_range",
    }


def _env_value(name: str, value: str):
    if name in _BOOLEANS:
        if value.lower() in {"1", "true", "yes", "on"}:
            return True
        if value.lower() in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean: {name}={value}")

    if name in _INTEGERS:
        return int(value)

    if name in _FLOATS:
        return float(value)

    if name in {"output_dir", "continual_dir", "continual_checkpoint_dir", "trace_dir",
                "benchmark_dir", "visualization_dir"}:
        return Path(value)

    if name in {"datasets", "train_domains"}:
        return re.split(r"[,\s]+", value.strip())

    return value


@dataclass(slots=True)
class Config:
    # Data
    mode: str = "train"
    datasets: list[str] = field(
        default_factory=lambda: ["gsm8k"]
    )
    output_dir: Path = Path("data")
    continual_dir: Path = Path("data/domain_continual")
    limit: int | None = None
    strict: bool = False
    debug_samples: int = 6
    plain: bool = False

    # Hugging Face
    hf_upload: bool = True
    hf_data_repo: str | None = None
    hf_continual_repo: str | None = None
    hf_model_repo: str | None = None
    hf_token_env: str = "HF_TOKEN"

    # Stable names: upload and later download will resolve to these same
    # account-scoped dataset repositories unless explicitly overridden.
    HF_NORMALIZED_REPO_NAME = "recursive-rl-normalized-data"
    HF_CONTINUAL_REPO_NAME = "recursive-rl-continual-data"
    HF_MODEL_REPO_NAME = "recursive-rl-continual-model"

    def resolve_hf_repos(self) -> tuple[str, str]:
        """Resolve stable private dataset repo IDs from the active HF login.

        Lazy: ordinary training/config parsing never requires a Hub request.
        Explicit `RL_HF_*_REPO` values remain supported as overrides.
        """
        if not self.hf_data_repo or not self.hf_continual_repo:
            from huggingface_hub import HfApi

            api = HfApi(token=os.getenv(self.hf_token_env))
            try:
                identity = api.whoami()
            except Exception as exc:
                raise RuntimeError(
                    "Could not identify the Hugging Face account. "
                    "Set HF_TOKEN to a valid write-enabled token or "
                    "run `hf auth login` before uploading/reading private data."
                ) from exc

            # Prefer the authenticated user namespace; organizations are not
            # selected implicitly because membership does not ensure write access.
            username = identity.get("name")
            if not username or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", username):
                raise ValueError("Hugging Face returned an invalid account name")

            self.hf_data_repo = self.hf_data_repo or (
                f"{username}/{self.HF_NORMALIZED_REPO_NAME}"
            )
            self.hf_continual_repo = self.hf_continual_repo or (
                f"{username}/{self.HF_CONTINUAL_REPO_NAME}"
            )

        for repo in (self.hf_data_repo, self.hf_continual_repo):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo or ""):
                raise ValueError(f"Invalid HF dataset repo ID: {repo!r}")
        if self.hf_data_repo == self.hf_continual_repo:
            raise ValueError("Raw and continual HF dataset repos must differ")
        return self.hf_data_repo, self.hf_continual_repo

    def hf_repo(self, continual: bool = True) -> str:
        """Return the exact upload/download repo ID for the requested data."""
        source, domain = self.resolve_hf_repos()
        return domain if continual else source

    def resolve_hf_model_repo(self) -> str:
        """Use one account-owned model repo; no public fallback."""
        if not self.hf_model_repo:
            from huggingface_hub import HfApi
            api = HfApi(token=os.getenv(self.hf_token_env) or None)
            try:
                username = api.whoami()["name"]
            except Exception as exc:
                raise RuntimeError(
                    "Cannot identify Hugging Face account for private model checkpoints; "
                    "set HF_TOKEN or run `hf auth login`."
                ) from exc
            suffix = "-tp" if self.parallel_mode == "tp" else ""
            track_suffix = "-self-improve" if self.train_algorithm == "self_improve" else ""
            self.hf_model_repo = f"{username}/{self.HF_MODEL_REPO_NAME}{suffix}{track_suffix}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.hf_model_repo):
            raise ValueError(f"Invalid HF model repo ID: {self.hf_model_repo!r}")
        return self.hf_model_repo

    # Model
    model_name: str = "Qwen/Qwen3-8B-Base"
    model_dtype: str = "bfloat16"
    model_attn: str = "sdpa"
    model_revision: str | None = None
    gradient_checkpointing: bool = True

    # 'tp' replaces FSDP across the same Ray worker group. All TP ranks
    # must process identical prompts and sampled tokens.
    parallel_mode: str = "fsdp"  # fsdp | tp

    # FSDP
    fsdp_reshard_after_forward: bool = True
    fsdp_reduce_dtype: str = "bfloat16"

    # Ray
    ray_address: str | None = None
    ray_num_workers: int = 2
    ray_cpus_per_worker: float = 2.0
    ray_gpus_per_worker: int = 1
    ray_worker_memory_gb: float | None = None
    ray_gpu_memory_fraction: float | None = None
    ray_placement_strategy: str = "PACK"
    ray_pg_timeout_s: int = 300
    ray_dist_timeout_s: int = 600

    # Local Ray initialization only
    ray_init_cpus: int | None = None
    ray_init_gpus: int | None = None
    ray_object_store_memory_gb: float | None = None

    # RL training
    # Which independent experiment to launch. The two tracks share data/task/eval.
    run_track: str = "grpo"  # grpo | self_improve | both
    train_algorithm: str = "grpo"  # set per-track by continual_train.train
    train_domains: list[str] = field(default_factory=lambda: [
        "mathematics", "science", "social_science", "finance"
    ])
    train_epochs: int = 1
    train_batch_size: int = 1        # prompts per logical batch (TP duplicates batches)
    train_finance_batch_size: int = 2
    train_finance_gradient_accumulation: int = 2
    train_max_steps: int | None = None  # per domain
    train_seed: int = 42
    train_shuffle: bool = True
    train_sampler: str = "random"  # random | euclidean | cosine | pca_knn
    sampling_projection_dim: int = 256
    sampling_embed_batch_size: int = 1
    sampling_max_length: int = 256
    sampling_knn_k: int = 2
    sampling_projector_seed: int = 2026
    train_lr: float = 1e-6
    train_weight_decay: float = 0.01
    train_grad_clip: float = 1.0

    # GRPO
    grpo_group_size: int = 4
    grpo_max_prompt_length: int = 512
    grpo_max_completion_length: int = 1024
    grpo_updates_per_rollout: int = 1
    grpo_temperature: float = 1.0
    grpo_top_p: float = 1.0
    grpo_clip_range: float = 0.2

    # Continual-learning orchestration
    continual_checkpoint_dir: Path = Path("outputs/continual")
    continual_resume: bool = True
    continual_eval_samples_per_worker: int = 1
    continual_eval_limit: int | None = None
    eval_microbatch_size: int = 32
    eval_finance_microbatch_size: int = 2
    eval_example_count: int = 5
    eval_skip_completed: bool = True
    eval_restore_counts: str | None = None
    eval_restore_revision: str | None = None
    baseline_mode: str = "auto"  # auto | reuse | evaluate | restore (one-time HF import)
    baseline_restore_counts: str | None = None  # JSON from completed terminal evaluation
    baseline_source_repo: str | None = None  # optional other track's private model repo

    # Optional diagnostics (never changes model training behavior)
    trace_mode: bool = False
    progress_bar: bool = True  # Live terminal tqdm bars. RL_PROGRESS_BAR=false restores log output.
    trace_dir: Path = Path("outputs/debug")

    # Benchmark plots are updated at each completed task.
    benchmark_mode: bool = True
    benchmark_dir: Path = Path("outputs/benchmark")
    benchmark_forward_transfer: bool = False  # requires zero-shot + unseen-domain evaluation

    # Optional representation PCA; GPU forward passes add evaluation overhead.
    visualization_mode: bool = False
    visualization_dir: Path = Path("outputs/visualization")
    visualization_samples_per_domain: int = 8
    visualization_max_length: int = 128
    visualization_batch_size: int = 1
    visualization_pooling: str = "mean"  # mean | last
    visualization_source: str = "test"  # test | train

    def __post_init__(self):
        for item in fields(self):
            value = os.getenv(f"RL_{item.name.upper()}")

            if value is not None and value != "":
                setattr(
                    self,
                    item.name,
                    _env_value(item.name, value),
                )

        self.validate()

    def validate(self):
        if self.mode not in {"train", "test"}:
            raise ValueError("Invalid mode")

        if self.limit is not None and self.limit < 1:
            raise ValueError("Invalid limit")

        if self.debug_samples < 0:
            raise ValueError("Invalid debug_samples")

        if self.parallel_mode not in {"fsdp", "tp"}:
            raise ValueError("parallel_mode must be fsdp or tp")
        if self.parallel_mode == "tp" and self.ray_num_workers < 2:
            raise ValueError("TP requires at least two Ray GPU workers")
        if self.model_dtype not in {"bfloat16", "float16"}:
            raise ValueError("Invalid model_dtype")

        if self.model_attn not in {"flash_attention_2", "sdpa"}:
            raise ValueError("Invalid model_attn")

        if self.fsdp_reduce_dtype not in {"bfloat16", "float32"}:
            raise ValueError("Invalid fsdp_reduce_dtype")

        if self.ray_num_workers < 1 or self.ray_cpus_per_worker <= 0:
            raise ValueError(
                "Ray workers and CPUs/worker must be positive"
            )

        if self.ray_gpus_per_worker != 1:
            raise ValueError(
                "This implementation requires exactly one GPU per PolicyWorker; "
                "change ray_num_workers"
            )

        if self.ray_placement_strategy not in {
            "PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"
        }:
            raise ValueError("Invalid ray_placement_strategy")

        if self.ray_pg_timeout_s < 1 or self.ray_dist_timeout_s < 1:
            raise ValueError("Ray timeouts must be positive")

        if (
            self.ray_worker_memory_gb is not None
            and self.ray_worker_memory_gb <= 0
        ):
            raise ValueError("ray_worker_memory_gb must be positive")

        if (
            self.ray_gpu_memory_fraction is not None
            and not 0 < self.ray_gpu_memory_fraction <= 1
        ):
            raise ValueError(
                "ray_gpu_memory_fraction must be in (0, 1]"
            )

        for name in ("ray_init_cpus", "ray_init_gpus"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")

        if (
            self.ray_object_store_memory_gb is not None
            and self.ray_object_store_memory_gb < 0.1
        ):
            raise ValueError(
                "ray_object_store_memory_gb must be >= 0.1"
            )

        if "all" in self.datasets:
            self.datasets = list(DATASET_REPOS)

        if set(self.datasets) - DATASET_REPOS.keys():
            raise ValueError("Unknown dataset")

        if not self.train_domains or len(self.train_domains) != len(set(self.train_domains)):
            raise ValueError("train_domains must be unique and nonempty")
        if any(name not in DOMAIN_TASKS for name in self.train_domains):
            raise ValueError("Unknown train domain")

        if self.run_track not in {"grpo", "self_improve", "both"}:
            raise ValueError("run_track must be grpo, self_improve or both")
        if self.train_algorithm not in {"grpo", "self_improve"}:
            raise ValueError("Invalid train_algorithm (use RL_RUN_TRACK in config.sh)")
        if min(self.train_epochs, self.train_batch_size,
               self.grpo_group_size, self.grpo_max_prompt_length,
               self.grpo_max_completion_length, self.grpo_updates_per_rollout) < 1:
            raise ValueError("Training counts must be positive")
        if self.train_sampler not in {"random", "euclidean", "cosine", "pca_knn"}:
            raise ValueError("Invalid RL_TRAIN_SAMPLER")
        if self.train_sampler != "random" and self.parallel_mode != "tp":
            raise ValueError("Policy-embedding sampling currently supports TP, not FSDP")
        if min(self.sampling_projection_dim, self.sampling_embed_batch_size,
               self.sampling_max_length, self.sampling_knn_k) < 1:
            raise ValueError("Sampling settings must be positive")
        if self.train_sampler == "pca_knn" and self.sampling_knn_k > (
            self.train_finance_batch_size * self.train_finance_gradient_accumulation
            if "finance" in self.train_domains else self.train_batch_size
        ):
            raise ValueError("PCA-kNN k cannot exceed prior step's sample count")
        if self.train_max_steps is not None and self.train_max_steps < 1:
            raise ValueError("train_max_steps must be positive")
        if self.grpo_group_size < 2:
            raise ValueError("GRPO requires grpo_group_size >= 2")
        if self.train_lr <= 0 or self.train_weight_decay < 0 or self.train_grad_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if self.grpo_temperature <= 0 or not 0 < self.grpo_top_p <= 1:
            raise ValueError("Invalid GRPO sampling settings")
        if not 0 < self.grpo_clip_range < 1:
            raise ValueError("Invalid GRPO clip range")

        if self.baseline_mode not in {"auto", "reuse", "evaluate", "restore"}:
            raise ValueError("baseline_mode must be auto, reuse, evaluate or restore")
        if self.baseline_mode == "restore" and not self.baseline_restore_counts:
            raise ValueError("RL_BASELINE_RESTORE_COUNTS is required in restore mode")
        if self.baseline_source_repo is not None and not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.baseline_source_repo
        ):
            raise ValueError("Invalid baseline_source_repo")
        if min(self.eval_microbatch_size, self.eval_finance_microbatch_size) < 1:
            raise ValueError("Evaluation microbatch sizes must be >= 1")
        if self.eval_example_count < 0:
            raise ValueError("RL_EVAL_EXAMPLE_COUNT must be >= 0")
        if min(self.train_finance_batch_size, self.train_finance_gradient_accumulation) < 1:
            raise ValueError("Finance batch and gradient accumulation must be >= 1")
        if bool(self.eval_restore_counts) != bool(self.eval_restore_revision):
            raise ValueError("RL_EVAL_RESTORE_COUNTS and RL_EVAL_RESTORE_REVISION must be set together")
        if self.continual_eval_samples_per_worker < 1:
            raise ValueError("Invalid continual_eval_samples_per_worker")
        if self.continual_eval_limit is not None and self.continual_eval_limit < 1:
            raise ValueError("Invalid continual_eval_limit")
        self.continual_checkpoint_dir = Path(self.continual_checkpoint_dir)
        if (self.parallel_mode == "tp" and
                self.continual_checkpoint_dir == Path("outputs/continual") and
                not os.getenv("RL_CONTINUAL_CHECKPOINT_DIR")):
            # Do not reuse historical FSDP checkpoints/progress with TP.
            self.continual_checkpoint_dir = Path("outputs/continual_tp")
        self.trace_dir = Path(self.trace_dir)
        self.benchmark_dir = Path(self.benchmark_dir)
        self.visualization_dir = Path(self.visualization_dir)
        if min(self.visualization_samples_per_domain, self.visualization_max_length,
               self.visualization_batch_size) < 1:
            raise ValueError("Visualization sizes must be positive")
        if self.visualization_pooling not in {"mean", "last"}:
            raise ValueError("visualization_pooling must be mean or last")
        if self.visualization_source not in {"test", "train"}:
            raise ValueError("visualization_source must be test or train")

        self.output_dir = Path(self.output_dir)
        self.continual_dir = Path(self.continual_dir)


LoaderConfig = Config


def parse_config() -> Config:
    """CLI mirrors the dataclass fields; field type tables also parse RL_* env."""
    parser = argparse.ArgumentParser()
    argument_groups = {
        "Data": "mode datasets output_dir continual_dir limit debug_samples plain",
        "Hugging Face": "hf_upload hf_data_repo hf_continual_repo hf_model_repo hf_token_env",
        "Model": "model_name model_dtype model_attn model_revision gradient_checkpointing",
        "Parallelism": "parallel_mode",
        "FSDP": "fsdp_reshard_after_forward fsdp_reduce_dtype",
        "Ray": ("ray_address ray_num_workers ray_cpus_per_worker ray_gpus_per_worker "
                "ray_worker_memory_gb ray_gpu_memory_fraction ray_placement_strategy "
                "ray_pg_timeout_s ray_dist_timeout_s ray_init_cpus ray_init_gpus "
                "ray_object_store_memory_gb"),
        "RL training": ("run_track train_algorithm train_domains train_epochs train_batch_size "
                        "train_finance_batch_size train_finance_gradient_accumulation "
                        "train_max_steps train_seed train_shuffle train_lr train_weight_decay train_grad_clip "
                        "train_sampler sampling_projection_dim sampling_embed_batch_size "
                        "sampling_max_length sampling_knn_k sampling_projector_seed"),
        "GRPO": ("grpo_group_size grpo_max_prompt_length grpo_max_completion_length "
                 "grpo_updates_per_rollout grpo_temperature grpo_top_p grpo_clip_range"),
        "Continual Learning": ("continual_checkpoint_dir continual_resume "
                              "continual_eval_samples_per_worker continual_eval_limit "
                              "eval_microbatch_size eval_finance_microbatch_size eval_example_count "
                              "eval_skip_completed eval_restore_counts eval_restore_revision"),
        "Trace": "trace_mode trace_dir",
        "Continual benchmark": ("benchmark_mode benchmark_dir benchmark_forward_transfer baseline_mode "
                                "baseline_restore_counts baseline_source_repo"),
        "Representation PCA": ("visualization_mode visualization_dir visualization_samples_per_domain "
                               "visualization_max_length visualization_batch_size "
                               "visualization_pooling visualization_source"),
    }
    choices = {
        "mode": ("train", "test"), "model_dtype": ("bfloat16", "float16"),
        "model_attn": ("flash_attention_2", "sdpa"),
        "parallel_mode": ("fsdp", "tp"),
        "fsdp_reduce_dtype": ("bfloat16", "float32"),
        "ray_placement_strategy": ("PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"),
        "run_track": ("grpo", "self_improve", "both"),
        "train_algorithm": ("grpo", "self_improve"),
        "baseline_mode": ("auto", "reuse", "evaluate", "restore"),
        "train_sampler": ("random", "euclidean", "cosine", "pca_knn"),
        "visualization_pooling": ("mean", "last"),
        "visualization_source": ("train", "test"),
    }
    path_fields = {"output_dir", "continual_dir", "continual_checkpoint_dir",
                   "trace_dir", "benchmark_dir", "visualization_dir"}
    for group_name, names in argument_groups.items():
        group = parser.add_argument_group(group_name)
        for name in names.split():
            option = "--dataset" if name == "datasets" else "--" + name.replace("_", "-")
            kwargs = {"dest": "datasets", "nargs": "+"} if name == "datasets" else {}
            if name == "train_domains":
                kwargs["nargs"] = "+"
            if name in _BOOLEANS:
                kwargs["action"] = argparse.BooleanOptionalAction
            elif name in _INTEGERS:
                kwargs["type"] = int
            elif name in _FLOATS:
                kwargs["type"] = float
            elif name in path_fields:
                kwargs["type"] = Path
            if name in choices:
                kwargs["choices"] = choices[name]
            group.add_argument(option, **kwargs)
    parser.add_argument("--skip-invalid", action="store_true")

    config = Config()
    for key, value in vars(parser.parse_args()).items():
        if key == "skip_invalid":
            if value:
                config.strict = False
        elif value is not None:
            setattr(config, key, value)
    config.validate()
    return config
