"""Metrics for synthetic protocol-transfer benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .splits import Row, rows_for_protocol, target_vector


def rmse(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Return root mean squared error."""

    true, pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Return mean absolute error."""

    true, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def target_protocol_rmse(rows: Sequence[Row], predictions: Sequence[float] | np.ndarray, protocol: str) -> float:
    """Return RMSE on a selected target protocol."""

    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length.")
    selected_indices = [index for index, row in enumerate(rows) if row["protocol"] == protocol]
    if not selected_indices:
        raise ValueError(f"No rows found for protocol '{protocol}'.")
    y_true = target_vector(rows_for_protocol(rows, protocol))
    y_pred = np.asarray(predictions, dtype=float)[selected_indices]
    return rmse(y_true, y_pred)


def transfer_ratio(model_rmse: float, reference_rmse: float) -> float:
    """Return improvement ratio over a reference baseline."""

    if model_rmse < 0.0 or reference_rmse < 0.0:
        raise ValueError("RMSE values must be non-negative.")
    if model_rmse == 0.0:
        return float("inf")
    return float(reference_rmse / model_rmse)


def evaluate_predictions(
    rows: Sequence[Row],
    predictions: Sequence[float] | np.ndarray,
    reference_predictions: Sequence[float] | np.ndarray | None = None,
    target_protocol: str | None = None,
) -> dict[str, float]:
    """Evaluate predictions on benchmark rows."""

    y_true = target_vector(rows)
    results = {
        "rmse": rmse(y_true, predictions),
        "mae": mae(y_true, predictions),
    }
    if target_protocol is not None:
        results["target_protocol_rmse"] = target_protocol_rmse(rows, predictions, target_protocol)
    if reference_predictions is not None:
        reference_rmse = rmse(y_true, reference_predictions)
        results["reference_rmse"] = reference_rmse
        results["transfer_ratio"] = transfer_ratio(results["rmse"], reference_rmse)
    return results


def _as_arrays(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if true.size == 0:
        raise ValueError("Metric inputs must not be empty.")
    return true, pred
