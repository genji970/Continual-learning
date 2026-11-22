# Recursive Learning

**On-policy GRPO for continual learning and recursive self-improvement in LLMs.**

self-improvement is not yet ready but am planned to do it using data distribution 

## Architecture

```text
Domain datasets → Sampling → Ray GPU workers TP
                → Rollout → Reward verification → GRPO update
                → Cross-domain evaluation → Checkpoint → Next domain
```

### Project Structure

```text
recursive/
├── config.py                # Model and training configuration
├── config.sh                # Experiment settings
├── data_load.py             # Dataset loading and prompt formatting
├── distributed.py           # Ray actors, NCCL, TP / FSDP2(not yet done)
├── rollout.py               # KV-cache generation
├── policy.py                # Reward verification and GRPO
├── sampling.py              # Random / Euclidean / Cosine / PCA-kNN
├── visualization.py         # PCA representation trajectories
├── diagnostics.py           # Training traces and profiling
├── continual_train/
│   ├── train.py             # Sequential domain training
│   ├── evaluate.py          # Cross-domain evaluation
│   └── storage.py           # Checkpoints and Hugging Face Hub
├── requirements.txt
├── README.md
└── LICENSE
```

## Training

| Component | Setting |
| --- | --- |
| Model | Qwen3-8B-Base (configurable) |
| Optimization | On-policy GRPO; no reference-policy KL penalty |
| Parallelism | Ray orchestration + TP or FSDP2 / NCCL |
| Reward | `3 × correctness + format + cleanliness` |
| Sampling | Random, Euclidean, Cosine, PCA-kNN |
| Domains | Mathematics, Science, Social Science, Finance|
| Evaluation | Accuracy, forgetting, backward/forward transfer |

- **Reward:** correct final answer, `Final Answer:` formatting, and clean termination (no unrelated continuation).
- **Continual learning:** **stability** (retain old skills), **plasticity** (learn new skills), **generalization** (unseen domains).
- **Current vs. planned:** rewards are currently rule-based; a *single-model policy + semantic judge*, question generation, and self-distillation are research directions.

## Run

```bash
pip install -r requirements.txt
source config.sh
python -u -m continual_train.train --ray-num-workers 2
```

> Keep Hugging Face tokens out of tracked configuration files; use environment secrets or `hf auth login`.

## Related Papers

### Self-rewarding and self-play

- **[Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020):** One LLM as generator and judge → **shared policy/reward model** (planned).
- **[SPELL](https://proceedings.iclr.cc/paper_files/paper/2026/hash/61af4e55bbc1e29b4f8e2669a829a683-Abstract-Conference.html):** Questioner–responder–verifier self-play and difficulty adaptation → **multi-role self-improvement** (planned).
- **[R-Zero](https://arxiv.org/abs/2508.05004):** Self-generated questions and majority-vote pseudo-labels → **reduced dependence on labeled reasoning data** (planned).

### Continual learning and forgetting

- **[Reinforcement Fine-Tuning Naturally Mitigates Forgetting](https://arxiv.org/abs/2507.05386):** RFT vs. SFT forgetting; hard-sample effects → **continual RL and sampling ablations**.
- **[Retaining by Doing](https://proceedings.mlr.press/v306/chen26do.html):** On-policy experience helps retain prior abilities → **motivation for on-policy GRPO**.
- **[Self-Distillation Enables Continual Learning](https://proceedings.mlr.press/v306/shenfeld26a.html):** Self-distillation for retaining old capabilities → **future retention strategy**.

## Visualization
**ACC Matrix**
<img width="420" height="258" alt="Image" src="https://github.com/user-attachments/assets/8bada889-65d2-4421-8693-57faffeac820" />

Two GIFs show **task-wise representation drift under different training/sampling methods**. Each frame tracks the **same fixed samples**, projected onto the PCA basis fitted before training; colors indicate domains, and movement shows how representations change after each task.

**Experiment 1**

<img width="450" height="340" alt="Task-wise PCA trajectory - experiment 1" src="https://github.com/user-attachments/assets/cf3a6d5b-d179-4001-ac20-803e734042cb" />

**Experiment 2**

<img width="450" height="340" alt="Task-wise PCA trajectory - experiment 2" src="https://github.com/user-attachments/assets/c596d932-6a88-4e24-b120-c11d4a2893ab" />

## License

See [LICENSE](LICENSE).
