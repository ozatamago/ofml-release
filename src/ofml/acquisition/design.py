"""Lightweight design-based acquisition: closed-form ALC and feature-space k-center.

Both avoid the per-candidate model refitting used by fantasy sampling. They operate in a
feature space provided by ``model.embed(rows)`` when available (e.g. the deep-kernel feature
embedding of an OFML row model); otherwise they fall back to the raw row inputs ``x1..xN``.

- ``alc_sampling``: expected variance reduction (Active Learning Cohn). With an RBF-GP surrogate
  over the embeddings, the reduction of posterior variance on the reference rows from observing a
  candidate has the closed form ``sum_e k(e,c)^2 / (k(c,c) + noise)`` and needs no observed labels.
  Batches are selected greedily with exact rank-1 GP conditioning between picks.
- ``kcenter_sampling``: greedy farthest-point (k-center) selection in the embedding space, covering
  the input/feature space relative to the already-observed rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row

from .base import AcquisitionResult, PredictiveModel


def embed_rows(model: object, rows: Sequence[Row]) -> np.ndarray:
    """Return a feature matrix for rows via ``model.embed`` or a raw-input fallback."""

    if not rows:
        return np.zeros((0, 1), dtype=float)
    if hasattr(model, "embed"):
        features = np.asarray(model.embed(rows), dtype=float)
        return features.reshape(len(rows), -1)
    matrix = []
    for row in rows:
        values = []
        index = 1
        while f"x{index}" in row:
            values.append(float(row[f"x{index}"]))
            index += 1
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def _median_lengthscale(features: np.ndarray) -> float:
    if features.shape[0] < 2:
        return 1.0
    diffs = features[:, None, :] - features[None, :, :]
    distances = np.sqrt(np.maximum(np.sum(diffs**2, axis=2), 0.0))
    upper = distances[np.triu_indices(features.shape[0], k=1)]
    median = float(np.median(upper)) if upper.size else 1.0
    return median if median > 1e-9 else 1.0


def _rbf(a: np.ndarray, b: np.ndarray, lengthscale: float) -> np.ndarray:
    sq = np.sum(a**2, axis=1)[:, None] + np.sum(b**2, axis=1)[None, :] - 2.0 * a @ b.T
    return np.exp(-0.5 * np.maximum(sq, 0.0) / (lengthscale**2))


def alc_sampling(
    model: PredictiveModel,
    train_rows: Sequence[Row],
    candidates: Sequence[Row],
    evaluation_rows: Sequence[Row],
    n: int,
    *,
    noise: float = 1e-3,
    lengthscale: float | None = None,
) -> AcquisitionResult:
    """Closed-form expected variance reduction (ALC) over reference rows; no refitting."""

    if not candidates:
        raise ValueError("candidates must not be empty.")
    if not evaluation_rows:
        raise ValueError("evaluation_rows must not be empty.")
    if n < 1 or n > len(candidates):
        raise ValueError("n must be between 1 and the number of candidates.")

    cand = embed_rows(model, candidates)
    ref = embed_rows(model, evaluation_rows)
    lengthscale = _median_lengthscale(np.vstack([cand, ref])) if lengthscale is None else lengthscale

    k_ec = _rbf(ref, cand, lengthscale)  # (n_ref, n_cand)
    k_cc = _rbf(cand, cand, lengthscale)  # (n_cand, n_cand)

    chosen: list[int] = []
    scores: list[float] = []
    available = set(range(len(candidates)))
    for _ in range(n):
        self_var = np.diag(k_cc) + noise
        candidate_scores = np.sum(k_ec**2, axis=0) / np.maximum(self_var, 1e-12)
        best = max(available, key=lambda index: candidate_scores[index])
        chosen.append(best)
        scores.append(float(candidate_scores[best]))
        available.discard(best)
        if not available:
            break
        # Condition the surrogate GP on the selected candidate (rank-1 downdate).
        factor = 1.0 / (k_cc[best, best] + noise)
        k_ec = k_ec - np.outer(k_ec[:, best], k_cc[best, :]) * factor
        k_cc = k_cc - np.outer(k_cc[:, best], k_cc[best, :]) * factor

    return AcquisitionResult(
        selected_indices=tuple(chosen),
        selected_rows=tuple(candidates[index] for index in chosen),
        scores=tuple(scores),
    )


def kcenter_sampling(
    model: PredictiveModel,
    train_rows: Sequence[Row],
    candidates: Sequence[Row],
    n: int,
    *,
    observed_rows: Sequence[Row] | None = None,
) -> AcquisitionResult:
    """Greedy farthest-point (k-center) selection in the embedding space."""

    if not candidates:
        raise ValueError("candidates must not be empty.")
    if n < 1 or n > len(candidates):
        raise ValueError("n must be between 1 and the number of candidates.")

    cand = embed_rows(model, candidates)
    seed_rows = list(observed_rows) if observed_rows is not None else list(train_rows)
    seed = embed_rows(model, seed_rows) if seed_rows else np.zeros((0, cand.shape[1]), dtype=float)

    if seed.shape[0] > 0:
        min_distance = np.min(np.linalg.norm(cand[:, None, :] - seed[None, :, :], axis=2), axis=1)
    else:
        min_distance = np.full(cand.shape[0], np.inf, dtype=float)

    chosen: list[int] = []
    scores: list[float] = []
    available = set(range(len(candidates)))
    for _ in range(n):
        best = max(available, key=lambda index: min_distance[index])
        chosen.append(best)
        scores.append(float(min_distance[best] if np.isfinite(min_distance[best]) else 0.0))
        available.discard(best)
        if not available:
            break
        new_distance = np.linalg.norm(cand - cand[best][None, :], axis=1)
        min_distance = np.minimum(min_distance, new_distance)

    return AcquisitionResult(
        selected_indices=tuple(chosen),
        selected_rows=tuple(candidates[index] for index in chosen),
        scores=tuple(scores),
    )
