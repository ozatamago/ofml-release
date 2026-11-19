"""Simple baseline models for OFML benchmarks."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row, protocol_labels, target_vector


@dataclasses.dataclass
class ProtocolMeanBaseline:
    """Predict the mean training target for each protocol."""

    global_mean_: float | None = None
    protocol_means_: dict[str, float] = dataclasses.field(default_factory=dict)

    def fit(self, rows: Sequence[Row]) -> "ProtocolMeanBaseline":
        """Fit protocol-level and global target means."""

        if not rows:
            raise ValueError("Cannot fit ProtocolMeanBaseline with no rows.")

        targets = target_vector(rows)
        labels = protocol_labels(rows)
        self.global_mean_ = float(np.mean(targets))
        self.protocol_means_ = {
            label: float(np.mean([target for row_label, target in zip(labels, targets) if row_label == label]))
            for label in sorted(set(labels))
        }
        return self

    def predict(self, rows: Sequence[Row]) -> np.ndarray:
        """Predict targets for benchmark rows."""

        if self.global_mean_ is None:
            raise ValueError("ProtocolMeanBaseline must be fit before predict.")
        return np.array(
            [self.protocol_means_.get(row["protocol"], self.global_mean_) for row in rows],
            dtype=float,
        )
