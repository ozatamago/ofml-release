"""Feature encoders for black-box benchmark models."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row


DEFAULT_OPERATION_VOCABULARY = ("A", "B", "C", "D", "M", "O")


@dataclasses.dataclass
class RowFeatureEncoder:
    """Encode benchmark rows into fixed-length numeric features."""

    include_protocol_id: bool = False
    include_step_counts: bool = True
    include_semantic_variant: bool = True
    operation_vocabulary: tuple[str, ...] = DEFAULT_OPERATION_VOCABULARY
    protocols_: tuple[str, ...] = ()
    semantic_variants_: tuple[str, ...] = ("default", "shifted")

    def fit(self, rows: Sequence[Row]) -> "RowFeatureEncoder":
        """Fit protocol and semantic variant vocabularies."""

        if not rows:
            raise ValueError("Cannot fit RowFeatureEncoder with no rows.")
        self.protocols_ = tuple(sorted({row["protocol"] for row in rows}))
        variants = {row.get("semantic_variant", "default") for row in rows}
        variants.update(self.semantic_variants_)
        self.semantic_variants_ = tuple(sorted(variants))
        return self

    def transform(self, rows: Sequence[Row]) -> np.ndarray:
        """Transform rows into a feature matrix."""

        if not rows:
            return np.zeros((0, self.n_features), dtype=float)

        features = []
        for row in rows:
            values = [float(row["x1"]), float(row["x2"])]
            if self.include_protocol_id:
                values.extend(1.0 if row["protocol"] == protocol else 0.0 for protocol in self.protocols_)
            if self.include_step_counts:
                steps = str(row["steps"]).split()
                values.extend(float(steps.count(operation)) for operation in self.operation_vocabulary)
            if self.include_semantic_variant:
                variant = row.get("semantic_variant", "default")
                values.extend(1.0 if variant == known else 0.0 for known in self.semantic_variants_)
            features.append(values)
        return np.asarray(features, dtype=float)

    def fit_transform(self, rows: Sequence[Row]) -> np.ndarray:
        """Fit and transform rows."""

        return self.fit(rows).transform(rows)

    @property
    def n_features(self) -> int:
        """Return the encoded feature dimension."""

        return (
            2
            + (len(self.protocols_) if self.include_protocol_id else 0)
            + (len(self.operation_vocabulary) if self.include_step_counts else 0)
            + (len(self.semantic_variants_) if self.include_semantic_variant else 0)
        )
