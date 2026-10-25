"""Task-wise frozen embedding and distance-aware sampling for GRPO.

GPU memory stays bounded: embed one small batch at a time using the current
sharded policy; GPU projector is destroyed after each Task embedding pass.
Only the projected representation matrix stays in driver CPU RAM.
"""
from __future__ import annotations

import math
import numpy as np


class StepDistanceSampler:
    """Greedy farthest-next-step sampling with no repeated index per epoch."""

    def __init__(self, embeddings, method: str, seed: int, k: int = 2):
        if method not in {"euclidean", "cosine", "pca_knn"}:
            raise ValueError("Unknown embedding sampler")
        z = np.asarray(embeddings, dtype=np.float32)
        if z.ndim != 2 or len(z) == 0 or not np.isfinite(z).all():
            raise ValueError("Invalid embeddings")
        self.method, self.k = method, k
        self.explained_variance_ratio = None
        if method == "pca_knn":
            z, self.explained_variance_ratio = pca_two(z, seed)
        elif method == "cosine":
            z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        self.z = z
        self.squares = (z * z).sum(axis=1)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.available = np.ones(len(self.z), dtype=bool)
        self.previous = []

    def _scores(self, candidates: np.ndarray, references: list[int]):
        a, b = self.z[candidates], self.z[references]
        if self.method == "cosine":
            return (1 - a @ b.T).min(axis=1)
        d2 = np.maximum(self.squares[candidates, None] +
                        self.squares[None, references] - 2 * a @ b.T, 0.0)
        if self.method == "pca_knn":
            if len(references) < self.k:
                raise ValueError("k exceeds available reference count")
            return np.partition(d2, self.k - 1, axis=1)[:, self.k - 1]
        return d2.min(axis=1)  # squared Euclidean gives same ordering

    def sample(self, count: int) -> list[int]:
        pool = np.flatnonzero(self.available)
        if len(pool) < count:
            raise ValueError("Insufficient unused training samples")
        if not self.previous:
            selected = self.rng.choice(pool, count, replace=False).tolist()
        else:
            refs, selected = list(self.previous), []
            for _ in range(count):
                candidates = pool[~np.isin(pool, selected)]
                scores = self._scores(candidates, refs)
                picked = int(candidates[int(np.argmax(scores))])
                selected.append(picked)
                refs.append(picked)
        self.available[selected] = False
        self.previous = selected
        return selected


def pca_two(embeddings: np.ndarray, seed: int):
    """Low-rank randomized PCA on CPU; never creates an NxN distance matrix."""
    import torch
    if embeddings.shape[0] < 3 or embeddings.shape[1] < 2:
        raise ValueError("PCA requires >= 3 samples and >= 2 features")
    x = torch.from_numpy(embeddings.copy())
    center = x.mean(0, keepdim=True)
    xc = x - center
    total_variance = float(xc.square().sum())
    if total_variance < 1e-12:
        raise ValueError("PCA cannot project identical embeddings")
    # Fork RNG: randomized PCA must not change GRPO seed/rollout randomness.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, singular, components = torch.pca_lowrank(xc, q=min(10, *xc.shape),
                                                    center=False, niter=3)
    axes = components[:, :2]
    projected = (xc @ axes).numpy()
    # Fix arbitrary component sign for deterministic visual comparison.
    for j in range(2):
        pivot = np.argmax(np.abs(axes[:, j].numpy()))
        if axes[pivot, j] < 0:
            projected[:, j] *= -1
    ratio = (singular[:2].square().numpy() / total_variance).tolist()
    return projected, ratio


def embed_entire_task(workers, config, records: list[dict], domain: str):
    """Frozen representation for this Task; all TP ranks enter identical forwards."""
    import ray
    from tqdm.auto import tqdm
    from distributed import BatchRouter

    router = BatchRouter(config, len(workers), config.sampling_embed_batch_size)
    width = router.global_size
    vectors = []
    projector_seed = config.sampling_projector_seed
    print(f"[SAMPLING EMBED] {domain}: {len(records)} prompts; "
          f"batch={width}, max_tokens={config.sampling_max_length}, "
          f"projection={config.sampling_projection_dim} (GPU, temporary)", flush=True)
    # Both TP ranks must create, use, and release the same projector together.
    try:
        ray.get([w.begin_sampling_projection.remote(projector_seed) for w in workers],
                timeout=config.ray_dist_timeout_s)
        for start in tqdm(range(0, len(records), width),
                          desc=f"Embed {domain}", disable=not config.progress_bar):
            chunk = records[start:start + width]
            valid = len(chunk)
            if valid < width:
                chunk = chunk + [chunk[-1]] * (width - valid)
            batches = router.split(chunk)
            results = ray.get([w.embed_sampling_batch.remote(part, projector_seed)
                               for w, part in zip(workers, batches)],
                              timeout=config.ray_dist_timeout_s)
            rows = [v for part in router.unique(results) for v in part]
            vectors.extend(rows[:valid])
    finally:
        # Cleanup also runs when one of the embedding forwards raises/OOMs.
        ray.get([w.end_sampling_projection.remote() for w in workers],
                timeout=config.ray_dist_timeout_s)
    z = np.asarray(vectors, dtype=np.float32)
    if z.shape != (len(records), config.sampling_projection_dim):
        raise ValueError(f"Sampling embedding shape mismatch: {z.shape}")
    return z
