"""Active-learning loop."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.metrics import evaluate_predictions
from ofml.benchmarks.synthetic_protocol_transfer.splits import Row

from .base import AcquisitionResult, PredictiveModel
from .pool import CandidatePool
from .tracking import simple_regret


Selector = Callable[[PredictiveModel, Sequence[Row], Sequence[Row], Sequence[Row], int], AcquisitionResult]


@dataclasses.dataclass
class ActiveLearningHistory:
    """History collected during active learning."""

    metrics: list[dict[str, float]]
    selected_rows: list[dict[str, str]]
    regrets: list[float]


def run_active_learning(
    model_factory: Callable[[], PredictiveModel],
    pool: CandidatePool,
    selector: Selector,
    rounds: int,
    batch_size: int,
    target_protocol: str | None = None,
    maximize_regret: bool = True,
) -> ActiveLearningHistory:
    """Run an active-learning loop over a candidate pool."""

    if rounds < 1:
        raise ValueError("rounds must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    history = ActiveLearningHistory(metrics=[], selected_rows=[], regrets=[])
    all_rows = [*pool.train_rows, *pool.candidate_rows, *pool.evaluation_rows]

    for _ in range(rounds):
        if not pool.candidate_rows:
            break
        model = model_factory()
        model.fit(pool.train_rows)
        predictions, _ = model.predict_distribution(pool.evaluation_rows)
        history.metrics.append(
            evaluate_predictions(pool.evaluation_rows, predictions, target_protocol=target_protocol)
        )
        history.regrets.append(simple_regret(pool.train_rows, all_rows, maximize=maximize_regret))
        result = selector(
            model,
            pool.train_rows,
            pool.candidate_rows,
            pool.evaluation_rows,
            min(batch_size, len(pool.candidate_rows)),
        )
        selected = pool.observe(result.selected_indices)
        history.selected_rows.extend(selected)

    if pool.evaluation_rows:
        model = model_factory()
        model.fit(pool.train_rows)
        predictions, _ = model.predict_distribution(pool.evaluation_rows)
        history.metrics.append(
            evaluate_predictions(pool.evaluation_rows, predictions, target_protocol=target_protocol)
        )
        history.regrets.append(simple_regret(pool.train_rows, all_rows, maximize=maximize_regret))

    return history
