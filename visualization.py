"""Optional FSDP-safe, task-by-task PCA of fixed held-out/train prompts.

A single fixed PCA basis is fitted once, before the first tracked task.
Subsequent frames project the *same prompts* using updated model embeddings.
This visualizes model representation drift, not a change in raw data distribution.
"""
from __future__ import annotations
from continual_train.storage import atomic_write_json

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def select_samples(config, datasets: dict[str, list[dict]]) -> list[dict]:
    """Select a small, deterministic anchor panel without changing training data."""
    anchors = []
    for i, domain in enumerate(config.train_domains):
        rows = datasets[domain]
        if not rows:
            raise ValueError(f"No visualization samples for domain {domain}")
        rng = random.Random(config.train_seed + i)
        chosen = sorted(rng.sample(range(len(rows)), min(config.visualization_samples_per_domain, len(rows))))
        anchors.extend({"id": f"{domain}:{j}", "domain": domain,
                        "messages": rows[j]["messages"]} for j in chosen)
    if len(anchors) < 2:
        raise ValueError("PCA requires at least two visualization samples")
    return anchors


@torch.no_grad()
def embed_worker(rollout, rows: list[dict]) -> list[list[float]]:
    """Runs inside one Ray/FSDP rank. All ranks must invoke it equally often."""
    config = rollout.config
    tokenizer = rollout.tokenizer
    inputs = [tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=True
    )[:config.visualization_max_length] for row in rows]
    if any(not tokens for tokens in inputs):
        raise ValueError("Empty tokenized visualization prompt")

    width = max(map(len, inputs))
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("Tokenizer requires a pad_token_id")
    ids = torch.full((len(inputs), width), pad, device=rollout.device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, tokens in enumerate(inputs):
        ids[i, -len(tokens):] = torch.tensor(tokens, device=rollout.device)
        mask[i, -len(tokens):] = 1

    rollout.model.eval()
    outputs = rollout.model(input_ids=ids, attention_mask=mask,
                            use_cache=False, output_hidden_states=True,
                            logits_to_keep=1)
    hidden = outputs.hidden_states[-1].float()
    if config.visualization_pooling == "last":
        feature = hidden[:, -1, :]
    else:
        feature = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return feature.cpu().tolist()


def _gather(workers, config, anchors: list[dict]) -> np.ndarray:
    import ray

    # Pad EVERY batch so every FSDP rank enters the same number of collectives.
    from distributed import BatchRouter
    router = BatchRouter(config, len(workers), config.visualization_batch_size)
    width = router.global_size
    vectors = []
    for start in range(0, len(anchors), width):
        batch = anchors[start:start + width]
        valid = len(batch)
        batch = batch + [batch[-1]] * (width - valid)
        parts = router.split(batch)
        refs = [worker.embed_samples.remote(part) for worker, part in zip(workers, parts)]
        result = ray.get(refs, timeout=config.ray_dist_timeout_s)
        vectors.extend(vector for rank_result in router.unique(result)
                       for vector in rank_result)
        vectors = vectors[:start + valid]
    return np.asarray(vectors, dtype=np.float32)


def capture(workers, config, anchors: list[dict], stage: str):
    """Append one model checkpoint to an animated, fixed-coordinate PCA timeline."""
    root = Path(config.visualization_dir)
    root.mkdir(parents=True, exist_ok=True)
    panel = [{"id": item["id"], "domain": item["domain"]} for item in anchors]
    state_file, frames_file = root / "pca_basis.npz", root / "frames.json"
    run_name = str(Path(config.continual_checkpoint_dir).resolve())

    if frames_file.exists() != state_file.exists():
        raise RuntimeError("Incomplete PCA state; use a new visualization_dir")

    previous = json.loads(frames_file.read_text()) if frames_file.exists() else None
    if previous is not None:
        if previous["samples"] != panel or previous["checkpoint_dir"] != run_name:
            raise ValueError("Old PCA run differs; choose another visualization_dir")
        if previous["frames"] and previous["frames"][-1]["stage"] == stage:
            # Restarts should still render PNGs from existing saved coordinates.
            if not (root / "pca_latest.png").exists():
                render_saved(root)
            return

    vectors = _gather(workers, config, anchors)
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite PCA input embeddings")
    if previous is None:
        center = vectors.mean(axis=0)
        _, _, vh = np.linalg.svd(vectors - center, full_matrices=False)
        components = vh[:2].T
        if components.shape[1] < 2:
            components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])))
        # Store basis without pickled Python objects.
        temp = state_file.with_suffix(".npz.tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, center=center, components=components)
        os.replace(temp, state_file)
        previous = {"checkpoint_dir": run_name, "samples": panel,
                    "axes": "PCA from initial captured model", "frames": []}
    else:
        with np.load(state_file, allow_pickle=False) as state:
            center, components = state["center"], state["components"]
        if vectors.shape[1] != components.shape[0]:
            raise ValueError("Model hidden dimension changed between tasks")

    coords = (vectors - center) @ components
    previous["frames"].append({"stage": stage, "points": coords.round(5).tolist()})
    atomic_write_json(frames_file, previous)
    render_saved(root)
    print(f"[PCA] {stage}: {len(anchors)} samples -> {root / 'pca_latest.png'}", flush=True)


def _matplotlib():
    # Headless backend works over SSH and inside RunPod containers.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_figure(fig, filename: Path):
    tmp = filename.with_name(filename.stem + ".tmp.png")
    fig.savefig(tmp, dpi=160, bbox_inches="tight")
    os.replace(tmp, filename)


def _safe_stage(stage: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("._") or "stage"


def _plot_limits(frames: list[dict]):
    all_points = np.concatenate([np.asarray(f["points"], dtype=float)
                                 for f in frames], axis=0)
    mn = np.min(all_points, axis=0)
    mx = np.max(all_points, axis=0)
    mid = (mn + mx) * 0.5
    half = max(float(np.max(mx - mn)) * 0.6, 1.0)
    return ((mid[0] - half, mid[0] + half),
            (mid[1] - half, mid[1] + half))


def _plot_frame(ax, points, samples, domains, colors, bounds, title: str):
    for domain in domains:
        indices = [i for i, s in enumerate(samples) if s["domain"] == domain]
        ax.scatter(points[indices, 0], points[indices, 1],
                   label=domain, color=colors[domain], s=40, alpha=0.82,
                   linewidth=0.4, edgecolors="white")
    ax.set(xlim=bounds[0], ylim=bounds[1], xlabel="PC1 (fixed initial basis)",
           ylabel="PC2 (fixed initial basis)", title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.16)
    ax.legend(loc="best", fontsize=8)


def render_saved(root: str | Path) -> list[Path]:
    """Rebuild Matplotlib images from existing frames.json without model/GPU.

    Saves pca_<stage>.png, pca_latest.png, pca_trajectory.png,
    and (with 2+ frames and Pillow) pca_animation.gif.
    The .npz PCA basis and frames.json are retained for reproducibility.
    """
    root = Path(root)
    data_path = root / "frames.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. A pca_basis.npz alone stores only "
            "the PCA basis; coordinates require a completed capture."
        )
    data = json.loads(data_path.read_text(encoding="utf-8"))
    samples, frames = data["samples"], data["frames"]
    if not frames:
        raise ValueError(f"No PCA frames in {data_path}")
    for frame in frames:
        if np.asarray(frame["points"]).shape != (len(samples), 2):
            raise ValueError(f"Invalid PCA point shape in stage {frame['stage']}")

    plt = _matplotlib()
    domains = list(dict.fromkeys(s["domain"] for s in samples))
    palette = plt.get_cmap("tab10")
    colors = {domain: palette(i % 10) for i, domain in enumerate(domains)}
    bounds = _plot_limits(frames)
    outputs = []

    for frame in frames:
        fig, ax = plt.subplots(figsize=(9, 6.8), constrained_layout=True)
        pts = np.asarray(frame["points"], dtype=float)
        _plot_frame(ax, pts, samples, domains, colors, bounds,
                    f"Representation PCA - {frame['stage']}")
        path = root / f"pca_{_safe_stage(frame['stage'])}.png"
        _save_figure(fig, path)
        if frame is frames[-1]:
            _save_figure(fig, root / "pca_latest.png")
        plt.close(fig)
        outputs.append(path)

    # All samples are the same at every checkpoint, so thin paths represent
    # actual movement of a fixed anchor's embedding in the same PCA basis.
    if len(frames) > 1:
        fig, ax = plt.subplots(figsize=(9, 6.8), constrained_layout=True)
        for i, sample in enumerate(samples):
            route = np.asarray([f["points"][i] for f in frames], dtype=float)
            ax.plot(route[:, 0], route[:, 1], color=colors[sample["domain"]],
                    alpha=0.30, linewidth=1)
            ax.annotate("", xy=route[-1], xytext=route[-2],
                        arrowprops=dict(arrowstyle="->", color=colors[sample["domain"]],
                                        lw=1, alpha=0.6))
        first = np.asarray(frames[0]["points"], dtype=float)
        last = np.asarray(frames[-1]["points"], dtype=float)
        ax.scatter(first[:, 0], first[:, 1], s=18, c="gray", alpha=0.4,
                   label="initial anchors")
        _plot_frame(ax, last, samples, domains, colors, bounds,
                    f"Fixed-anchor PCA drift: {frames[0]['stage']} → {frames[-1]['stage']}")
        path = root / "pca_trajectory.png"
        _save_figure(fig, path)
        plt.close(fig)
        outputs.append(path)

        # Optional Matplotlib GIF; static PNGs always remain available.
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
            fig, ax = plt.subplots(figsize=(9, 6.8), constrained_layout=True)

            def draw(index):
                ax.clear()
                frame = frames[index]
                _plot_frame(ax, np.asarray(frame["points"], dtype=float),
                            samples, domains, colors, bounds,
                            f"Representation PCA - {frame['stage']} "
                            f"({index + 1}/{len(frames)})")

            animation = FuncAnimation(fig, draw, frames=len(frames),
                                      interval=1100, repeat=True)
            tmp_gif = root / "pca_animation.tmp.gif"
            animation.save(str(tmp_gif), writer=PillowWriter(fps=1))
            os.replace(tmp_gif, root / "pca_animation.gif")
            plt.close(fig)
            outputs.append(root / "pca_animation.gif")
        except (ImportError, OSError, RuntimeError) as exc:
            plt.close("all")
            print(f"[PCA WARN] GIF not saved ({exc}); PNGs saved successfully", flush=True)
    return outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render saved PCA frames as Matplotlib figures")
    parser.add_argument("--visualization-dir", default="outputs/visualization")
    args = parser.parse_args()
    for path in render_saved(args.visualization_dir):
        print(f"[PCA IMAGE] {path}")
