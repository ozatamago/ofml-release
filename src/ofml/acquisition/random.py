"""Random acquisition strategy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row

from .base import AcquisitionResult


def random_sampling(candidates: Sequence[Row], n: int, seed: int | None = None) -> AcquisitionResult:
    """Select candidates uniformly at random without replacement."""

    if n < 1:
        raise ValueError("n must be at least 1.")
    if n > len(candidates):
        raise ValueError("n must not exceed the number of candidates.")
    rng = np.random.default_rng(seed)
    selected = tuple(int(index) for index in rng.choice(len(candidates), size=n, replace=False))
    return AcquisitionResult(
        selected_indices=selected,
        selected_rows=tuple(candidates[index] for index in selected),
        scores=tuple(1.0 for _ in selected),
    )
