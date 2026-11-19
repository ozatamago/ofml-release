"""Shared acquisition types and helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row


class PredictiveModel(Protocol):
    """Protocol for models used by acquisition strategies."""

    def fit(self, rows: Sequence[Row]) -> "PredictiveModel":
        """Fit the model from benchmark rows."""

    def predict_distribution(self, rows: Sequence[Row]) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance for benchmark rows."""


@dataclasses.dataclass(frozen=True)
class AcquisitionResult:
    """Selected candidates and their acquisition scores."""

    selected_indices: tuple[int, ...]
    selected_rows: tuple[Row, ...]
    scores: tuple[float, ...]


def take_top(scores: Sequence[float], candidates: Sequence[Row], n: int) -> AcquisitionResult:
    """Return the top-scoring candidate rows."""

    if n < 1:
        raise ValueError("n must be at least 1.")
    if n > len(candidates):
        raise ValueError("n must not exceed the number of candidates.")
    ordered = np.argsort(np.asarray(scores, dtype=float))[::-1][:n]
    selected_indices = tuple(int(index) for index in ordered)
    return AcquisitionResult(
        selected_indices=selected_indices,
        selected_rows=tuple(candidates[index] for index in selected_indices),
        scores=tuple(float(scores[index]) for index in selected_indices),
    )


def with_observed_y(row: Row, y: float) -> dict[str, str]:
    """Return a row copy with a substituted observed target."""

    updated = dict(row)
    updated["y"] = f"{float(y):.10f}"
    return updated


def variance_score(variance: np.ndarray) -> np.ndarray:
    """Reduce predictive variance to one acquisition score per row."""

    array = np.asarray(variance, dtype=float)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return np.mean(array, axis=1)
    raise ValueError(f"variance must be 1D or 2D. Got shape {array.shape}.")
