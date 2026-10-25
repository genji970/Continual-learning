"""Continual-learning metrics and automatically refreshed charts.

All accuracies/transfer values use fractions (0..1), not percent points.
A[t, j] is accuracy on domain j after completing domain t.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from utils.io import atomic_write_json


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
