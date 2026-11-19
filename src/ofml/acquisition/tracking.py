"""Active-learning tracking metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row, target_vector


def simple_regret(observed_rows: Sequence[Row], candidate_rows: Sequence[Row], maximize: bool = True) -> float:
    """Return simple regret against the best value available in the benchmark pool."""

    if not observed_rows:
        raise ValueError("observed_rows must not be empty.")
    if not candidate_rows:
        raise ValueError("candidate_rows must not be empty.")
    observed = target_vector(observed_rows)
    available = target_vector(candidate_rows)
    if maximize:
        return float(np.max(available) - np.max(observed))
    return float(np.min(observed) - np.min(available))


def sample_efficiency(metric_history: Sequence[float], threshold: float) -> int | None:
    """Return the first 1-based step at which a metric reaches a threshold."""

    for index, value in enumerate(metric_history, start=1):
        if value <= threshold:
            return index
    return None
