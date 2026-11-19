"""Uncertainty acquisition strategy."""

from __future__ import annotations

from collections.abc import Sequence

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row

from .base import AcquisitionResult, PredictiveModel, take_top, variance_score


def uncertainty_sampling(model: PredictiveModel, candidates: Sequence[Row], n: int) -> AcquisitionResult:
    """Select candidates with the largest predictive variance."""

    if not candidates:
        raise ValueError("candidates must not be empty.")
    _, variance = model.predict_distribution(candidates)
    return take_top(variance_score(variance), candidates, n)
