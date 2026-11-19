"""Fantasy acquisition strategies with explicit model refitting."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Sequence

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row, target_vector

from .base import AcquisitionResult, PredictiveModel, take_top, variance_score, with_observed_y


@dataclasses.dataclass(frozen=True)
class FantasyConfig:
    """Configuration for fantasy sampling."""

    n_fantasies: int = 8
    seed: int = 0
    score: str = "variance_reduction"
    loss_metric: str = "mse"
    variance_weight: float = 1.0
    loss_weight: float = 1.0
    nll_weight: float = 1.0
    model_change_weight: float = 1.0
    nll_variance_floor: float = 1e-6


def deterministic_fantasy_sampling(
    model: PredictiveModel,
    train_rows: Sequence[Row],
    candidates: Sequence[Row],
    evaluation_rows: Sequence[Row],
    n: int,
) -> AcquisitionResult:
    """Greedy fantasy sampling using posterior means as deterministic fantasies."""

    return fantasy_sampling(
        model=model,
        train_rows=train_rows,
        candidates=candidates,
        evaluation_rows=evaluation_rows,
        n=n,
        config=FantasyConfig(n_fantasies=1, seed=0),
        deterministic=True,
    )


def fantasy_sampling(
    model: PredictiveModel,
    train_rows: Sequence[Row],
    candidates: Sequence[Row],
    evaluation_rows: Sequence[Row],
    n: int,
    config: FantasyConfig | None = None,
    deterministic: bool = False,
) -> AcquisitionResult:
    """Greedy fantasy sampling by refitting on fantasized observations.

    Each candidate is scored by expected reduction in predictive variance on
    `evaluation_rows` after adding the candidate with fantasized observations.
    Batch selection is greedy: selected candidates are added to the working
    training set with posterior-mean fantasies before the next selection.
    """

    if not candidates:
        raise ValueError("candidates must not be empty.")
    if not evaluation_rows:
        raise ValueError("evaluation_rows must not be empty.")
    if n < 1:
        raise ValueError("n must be at least 1.")
    if n > len(candidates):
        raise ValueError("n must not exceed the number of candidates.")

    config = config or FantasyConfig()
    rng = np.random.default_rng(config.seed)
    working_model = copy.deepcopy(model)
    working_train = [dict(row) for row in train_rows]
    remaining = [dict(row) for row in candidates]
    selected_indices = []
    selected_rows = []
    selected_scores = []
    original_index = list(range(len(candidates)))

    for _ in range(n):
        scores = _fantasy_scores(
            model=working_model,
            train_rows=working_train,
            candidates=remaining,
            evaluation_rows=evaluation_rows,
            rng=rng,
            config=config,
            deterministic=deterministic,
        )
        result = take_top(scores, remaining, 1)
        local_index = result.selected_indices[0]
        selected = dict(remaining.pop(local_index))
        selected_indices.append(original_index.pop(local_index))
        selected_rows.append(selected)
        selected_scores.append(result.scores[0])

        mean, _ = working_model.predict_distribution([selected])
        working_train.append(with_observed_y(selected, float(mean[0])))
        working_model = copy.deepcopy(model)
        working_model.fit(working_train)

    return AcquisitionResult(
        selected_indices=tuple(selected_indices),
        selected_rows=tuple(selected_rows),
        scores=tuple(selected_scores),
    )


def _fantasy_scores(
    model: PredictiveModel,
    train_rows: Sequence[Row],
    candidates: Sequence[Row],
    evaluation_rows: Sequence[Row],
    rng: np.random.Generator,
    config: FantasyConfig,
    deterministic: bool,
) -> list[float]:
    base_mean, base_variance = model.predict_distribution(evaluation_rows)
    base_mean = np.asarray(base_mean, dtype=float)
    base_variance_score = float(np.mean(variance_score(base_variance)))
    y_eval = target_vector(evaluation_rows)
    base_loss = _loss(y_eval, base_mean, metric=config.loss_metric)
    base_nll = _gaussian_nll(
        y_eval,
        base_mean,
        variance_score(base_variance),
        variance_floor=config.nll_variance_floor,
    )
    candidate_mean, candidate_variance = model.predict_distribution(candidates)
    candidate_std = np.sqrt(np.maximum(variance_score(candidate_variance), 0.0))
    components: list[dict[str, float]] = []

    for index, candidate in enumerate(candidates):
        if deterministic:
            fantasy_values = np.array([candidate_mean[index]], dtype=float)
        else:
            fantasy_values = rng.normal(
                loc=float(candidate_mean[index]),
                scale=float(candidate_std[index]),
                size=config.n_fantasies,
            )
        utility = []
        for fantasy_y in fantasy_values:
            fantasy_model = copy.deepcopy(model)
            fantasy_train = [dict(row) for row in train_rows]
            fantasy_train.append(with_observed_y(candidate, float(fantasy_y)))
            fantasy_model.fit(fantasy_train)
            fantasy_mean, fantasy_variance = fantasy_model.predict_distribution(evaluation_rows)
            fantasy_mean = np.asarray(fantasy_mean, dtype=float)
            fantasy_variance_score = float(np.mean(variance_score(fantasy_variance)))
            fantasy_loss = _loss(y_eval, fantasy_mean, metric=config.loss_metric)
            fantasy_nll = _gaussian_nll(
                y_eval,
                fantasy_mean,
                variance_score(fantasy_variance),
                variance_floor=config.nll_variance_floor,
            )
            utility.append(
                {
                    "variance_reduction": base_variance_score - fantasy_variance_score,
                    "oracle_loss_reduction": base_loss - fantasy_loss,
                    "calibration_nll_reduction": base_nll - fantasy_nll,
                    "model_change": float(np.mean((fantasy_mean - base_mean) ** 2)),
                }
            )
        components.append(_mean_components(utility))
    return _score_components(components, config)


def _mean_components(components: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = components[0].keys()
    return {key: float(np.mean([component[key] for component in components])) for key in keys}


def _score_components(components: Sequence[dict[str, float]], config: FantasyConfig) -> list[float]:
    if config.score == "variance_reduction":
        return [component["variance_reduction"] for component in components]
    if config.score == "oracle_loss_reduction":
        return [component["oracle_loss_reduction"] for component in components]
    if config.score == "calibration_nll_reduction":
        return [component["calibration_nll_reduction"] for component in components]
    if config.score == "model_change":
        return [component["model_change"] for component in components]
    if config.score == "hybrid":
        return _hybrid_scores(components, config)
    raise ValueError(f"Unsupported fantasy score '{config.score}'.")


def _hybrid_scores(components: Sequence[dict[str, float]], config: FantasyConfig) -> list[float]:
    variance = _normalize([component["variance_reduction"] for component in components])
    loss = _normalize([component["oracle_loss_reduction"] for component in components])
    nll = _normalize([component["calibration_nll_reduction"] for component in components])
    model_change = _normalize([component["model_change"] for component in components])
    return [
        (
            config.variance_weight * variance[index]
            + config.loss_weight * loss[index]
            + config.nll_weight * nll[index]
            + config.model_change_weight * model_change[index]
        )
        for index in range(len(components))
    ]


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    low = float(np.min(array))
    high = float(np.max(array))
    if np.isclose(high, low):
        return np.zeros_like(array)
    return (array - low) / (high - low)


def _loss(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "mse":
        return float(np.mean((y_true - y_pred) ** 2))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    raise ValueError(f"Unsupported fantasy loss metric '{metric}'.")


def _gaussian_nll(
    y_true: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    variance_floor: float,
) -> float:
    safe_variance = np.maximum(np.asarray(variance, dtype=float), variance_floor)
    residual = y_true - np.asarray(mean, dtype=float)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * safe_variance) + (residual**2) / (2.0 * safe_variance)))
