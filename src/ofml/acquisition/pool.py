"""Candidate pool helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row


@dataclasses.dataclass
class CandidatePool:
    """Mutable active-learning pool."""

    train_rows: list[dict[str, str]]
    candidate_rows: list[dict[str, str]]
    evaluation_rows: list[dict[str, str]]

    def observe(self, indices: Sequence[int]) -> list[dict[str, str]]:
        """Move selected candidates into the training set with their true targets."""

        selected = [self.candidate_rows[index] for index in indices]
        selected_set = set(indices)
        self.train_rows.extend(dict(row) for row in selected)
        self.candidate_rows = [row for index, row in enumerate(self.candidate_rows) if index not in selected_set]
        return [dict(row) for row in selected]


def build_candidate_pool(
    rows: Sequence[Row],
    initial_size: int,
    seed: int = 0,
    evaluation_rows: Sequence[Row] | None = None,
) -> CandidatePool:
    """Split rows into an initial train set, candidate pool, and evaluation set."""

    if initial_size < 1:
        raise ValueError("initial_size must be at least 1.")
    if initial_size >= len(rows):
        raise ValueError("initial_size must be smaller than the number of rows.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(rows))
    train_indices = set(int(index) for index in indices[:initial_size])
    train_rows = [dict(row) for index, row in enumerate(rows) if index in train_indices]
    candidate_rows = [dict(row) for index, row in enumerate(rows) if index not in train_indices]
    evaluation = [dict(row) for row in (evaluation_rows if evaluation_rows is not None else rows)]
    return CandidatePool(train_rows=train_rows, candidate_rows=candidate_rows, evaluation_rows=evaluation)
