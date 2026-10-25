#!/usr/bin/env bash
# Source this file: source config.sh
# Set HF_TOKEN in the shell or use `hf auth login`.
# Do NOT store HF tokens in this file or commit them to git.
# Authenticate with `hf auth login` or export HF_TOKEN in your shell (never commit a token).
export HF_TOKEN=""
#!/usr/bin/env bash
# Source this file: source config.sh
# Set HF_TOKEN in the shell or use `hf auth login`.
# Do NOT store HF tokens in this file or commit them to git.
# Authenticate with `hf auth login` or export HF_TOKEN in your shell (never commit a token).
# Baseline 평가: auto | reuse | evaluate
# restore: upload past baseline counts; reuse: load existing HF baseline
export RL_BASELINE_MODE="reuse"

# 다른 Hugging Face 저장소의 Baseline 재사용 시에만 설정
# export RL_BASELINE_SOURCE_REPO="changgyu/recursive-rl-continual-model-tp"

# -----------------------------------------------------------------------------
# Common controls: change values here; dependent settings below follow them.
# -----------------------------------------------------------------------------
export RL_PARALLEL_MODE="tp"                       # tp | fsdp
export RL_RAY_NUM_WORKERS="2"                      # one Ray worker per GPU
export RL_CONTINUAL_EVAL_SAMPLES_PER_WORKER="32" # TP: global eval batch; FSDP: per-worker
export RL_TRAIN_BATCH_SIZE="4"                     # prompts per training batch (non-Finance)
# Sampling-policy ablation (random retains the original GRPO and Hub repo).
# Choose one of: random | euclidean | cosine | pca_knn
export RL_TRAIN_SAMPLER="random"
export RL_SAMPLING_PROJECTION_DIM="256"  # Projected CPU features (not trainable params)
export RL_SAMPLING_EMBED_BATCH_SIZE="1"  # Added GPU forward batch; keep at 1 for OOM safety
export RL_SAMPLING_MAX_LENGTH="256"      # Input token cap for representation extraction
export RL_SAMPLING_KNN_K="2"            # Only used by PCA-kNN
export RL_SAMPLING_PROJECTOR_SEED="2026"
export RL_TRAIN_FINANCE_BATCH_SIZE="2"             # Finance only: prompts per rollout microbatch
export RL_TRAIN_FINANCE_GRADIENT_ACCUMULATION="2"  # Finance only: 2 microbatches per optimizer step
export RL_GRPO_GROUP_SIZE="4"                      # completions per prompt (independent)
export RL_GRPO_MAX_COMPLETION_LENGTH="512"

# -----------------------------------------------------------------------------
# Data loading / Hugging Face
# -----------------------------------------------------------------------------
export RL_MODE="train"
export RL_DATASETS="gsm8k math"
export RL_OUTPUT_DIR="data"
export RL_CONTINUAL_DIR="data/domain_continual"
export RL_STRICT="false"
export RL_DEBUG_SAMPLES="6"
export RL_PLAIN="false"
export RL_HF_TOKEN_ENV="HF_TOKEN"
export RL_HF_UPLOAD="true"

# -----------------------------------------------------------------------------
# Model / optional FSDP2 settings (only effective for FSDP mode)
# -----------------------------------------------------------------------------
export RL_MODEL_NAME="Qwen/Qwen3-8B-Base"
export RL_MODEL_DTYPE="bfloat16"
export RL_MODEL_ATTN="sdpa"
export RL_GRADIENT_CHECKPOINTING="true"
export RL_FSDP_RESHARD_AFTER_FORWARD="true"
export RL_FSDP_REDUCE_DTYPE="bfloat16"

# -----------------------------------------------------------------------------
# Ray: one GPU per worker. World size follows RL_RAY_NUM_WORKERS above.
# -----------------------------------------------------------------------------
export RL_RAY_GPUS_PER_WORKER="1"
export RL_RAY_CPUS_PER_WORKER="8"
export RL_RAY_PLACEMENT_STRATEGY="PACK"
export RL_RAY_PG_TIMEOUT_S="3000"
export RL_RAY_DIST_TIMEOUT_S="6000"

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
# Single experiment switch: grpo | self_improve | both (runs sequentially)
export RL_RUN_TRACK="grpo"
unset RL_TRAIN_ALGORITHM  # legacy selector; avoid overriding per-track routing
export RL_TRAIN_DOMAINS="mathematics science social_science finance"
export RL_TRAIN_EPOCHS="1"
export RL_TRAIN_MAX_STEPS="100"
export RL_TRAIN_SEED="42"
export RL_TRAIN_SHUFFLE="true"
export RL_TRAIN_LR="1e-6"
export RL_TRAIN_WEIGHT_DECAY="0.01"
export RL_TRAIN_GRAD_CLIP="1.0"

# GRPO batch: RL_TRAIN_BATCH_SIZE * RL_GRPO_GROUP_SIZE = 4 * 4 = 16 completions
export RL_GRPO_MAX_PROMPT_LENGTH="32768"
export RL_GRPO_UPDATES_PER_ROLLOUT="1"
export RL_GRPO_TEMPERATURE="1.0"
export RL_GRPO_TOP_P="1.0"
export RL_GRPO_CLIP_RANGE="0.2"

# -----------------------------------------------------------------------------
# Continual learning: separate TP and FSDP checkpoints to avoid mixing formats.
# Preserve the existing FSDP directory; TP uses a distinct directory.
# -----------------------------------------------------------------------------
export RL_CONTINUAL_RESUME="true"
case "$RL_PARALLEL_MODE" in
  tp)   export RL_CONTINUAL_CHECKPOINT_DIR="outputs/continual_tp" ;;
  fsdp) export RL_CONTINUAL_CHECKPOINT_DIR="outputs/continual" ;;
  *) echo "Invalid RL_PARALLEL_MODE: $RL_PARALLEL_MODE" >&2; return 1 2>/dev/null || exit 1 ;;
esac

# Evaluation batch is already set once in Common controls. Do not reassign it.

# Evaluation restart and GPU memory controls (require eval patch installed).
# Dispatch remains at 32 so the saved experiment signature is unchanged.
export RL_EVAL_SKIP_COMPLETED="true"
export RL_EVAL_MICROBATCH_SIZE="32"        # Mathematics, Science, Social Science
export RL_EVAL_FINANCE_MICROBATCH_SIZE="2" # Finance only
export RL_EVAL_EXAMPLE_COUNT="5"           # outputs/eval_samples/task_XX_*/<domain>.json

# One-time recovery of completed Task 1 domain evaluations.
# Accepted only if the HF checkpoint revision exactly matches below.
export RL_EVAL_RESTORE_REVISION="72227063e5095349fb8a4dacd1569b0dcf8c3526"
export RL_EVAL_RESTORE_COUNTS='{"mathematics":[2508,6318],"science":[390,993],"social_science":[57,110]}'

# -----------------------------------------------------------------------------
# Diagnostics / benchmark
# -----------------------------------------------------------------------------
export RL_BENCHMARK_MODE="true"
export RL_BENCHMARK_DIR="outputs/benchmark"
export RL_BENCHMARK_FORWARD_TRANSFER="true"

# Visualization batch follows the evaluation batch automatically.
# Number of samples per domain and max input length are independent concepts.
export RL_VISUALIZATION_MODE="true"
export RL_VISUALIZATION_DIR="outputs/visualization"
export RL_VISUALIZATION_SAMPLES_PER_DOMAIN="8"
export RL_VISUALIZATION_MAX_LENGTH="1024"
export RL_VISUALIZATION_BATCH_SIZE="$RL_CONTINUAL_EVAL_SAMPLES_PER_WORKER"
export RL_VISUALIZATION_POOLING="mean"
export RL_VISUALIZATION_SOURCE="test"

export RL_TRACE_MODE="true"
export RL_TRACE_DIR="outputs/debug"

