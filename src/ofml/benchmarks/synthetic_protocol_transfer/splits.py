"""Dataset split and row conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .scenarios import ProtocolRole, parse_protocol_role


Row = Mapping[str, str]
Dataset = dict[str, list[dict[str, str]]]


def split_rows(rows: Sequence[dict[str, str]], train_fraction: float, rng: np.random.Generator) -> Dataset:
    """Split rows into train and test subsets with a seeded RNG."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    train = [row for row in rows if _row_role(row) == ProtocolRole.SOURCE]
    test = [row for row in rows if _row_role(row) == ProtocolRole.TARGET]
    both = [row for row in rows if _row_role(row) == ProtocolRole.BOTH]

    indices = rng.permutation(len(both))
    train_size = int(round(len(both) * train_fraction))
    train_indices = set(int(index) for index in indices[:train_size])

    train.extend(row for index, row in enumerate(both) if index in train_indices)
    test.extend(row for index, row in enumerate(both) if index not in train_indices)
    return {"train": train, "test": test}


def rows_for_protocol(rows: Sequence[Row], protocol: str) -> list[Row]:
    """Return rows belonging to one protocol."""

    return [row for row in rows if row["protocol"] == protocol]


def feature_matrix(rows: Sequence[Row]) -> np.ndarray:
    """Convert benchmark rows into a numeric feature matrix."""

    return np.array([[float(row["x1"]), float(row["x2"])] for row in rows], dtype=float)


def target_vector(rows: Sequence[Row]) -> np.ndarray:
    """Convert benchmark rows into a numeric target vector."""

    return np.array([float(row["y"]) for row in rows], dtype=float)


def protocol_labels(rows: Sequence[Row]) -> tuple[str, ...]:
    """Return protocol labels for rows."""

    return tuple(row["protocol"] for row in rows)


def _row_role(row: Row) -> ProtocolRole:
    return parse_protocol_role(row.get("protocol_role", ProtocolRole.BOTH.value))
