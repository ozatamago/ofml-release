"""OFML composition wrapper."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row


OperationExample = Mapping[str, str]


class OperationModel(Protocol):
    """Interface for a learned operation model."""

    def fit(self, examples: Sequence[OperationExample]) -> "OperationModel":
        """Fit the operation model from operation-level examples."""
        ...

    def predict_state(self, states: np.ndarray) -> np.ndarray:
        """Predict output state or scalar output for input states."""
        ...

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict output mean and variance for input states."""
        ...


@dataclasses.dataclass
class OFMLModel:
    """Compose assigned operation models according to protocol step sequences."""

    operation_models: Mapping[str, OperationModel]
    n_composition_samples: int = 64
    seed: int = 0
    initial_state_dim: int = 2

    def fit(self, operation_examples: Sequence[OperationExample], split: str = "train") -> "OFMLModel":
        """Fit every assigned operation model using matching operation examples."""

        for operation, model in self.operation_models.items():
            examples = [
                example
                for example in operation_examples
                if example["operation"] == operation and example.get("split", split) == split
            ]
            if examples:
                model.fit(examples)
        return self

    def predict(self, rows: Sequence[Row]) -> np.ndarray:
        """Predict scalar protocol outputs."""

        mean, _ = self.predict_distribution(rows)
        return mean

    def predict_distribution(
        self,
        rows: Sequence[Row],
        propagation: str = "mean",
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict scalar output means and variances by composing operation models."""

        if propagation == "sampling":
            samples = self.predict_samples(rows, n_samples=n_samples, seed=seed)
            return np.mean(samples, axis=0), np.var(samples, axis=0)
        if propagation != "mean":
            raise ValueError("propagation must be either 'mean' or 'sampling'.")

        means = []
        variances = []
        for row in rows:
            state_mean = _initial_state(row, self.initial_state_dim)
            state_variance = np.zeros_like(state_mean)
            steps = str(row["steps"])
            for operation in steps.split():
                model = self._operation_model(operation)
                output_mean, output_variance = model.predict_distribution(
                    self._augment_input(state_mean, steps)
                )
                state_mean = _as_2d(output_mean)
                state_variance = _as_2d(output_variance)
            if state_mean.shape[1] != 1:
                raise ValueError(
                    "The final operation must return one scalar output. "
                    f"Got shape {state_mean.shape} for protocol '{row['protocol']}'."
                )
            means.append(float(state_mean[0, 0]))
            variances.append(float(state_variance[0, 0]))
        return np.asarray(means, dtype=float), np.asarray(variances, dtype=float)

    def predict_samples(
        self,
        rows: Sequence[Row],
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Predict scalar output samples by Monte Carlo composition."""

        sample_count = n_samples or self.n_composition_samples
        if sample_count < 1:
            raise ValueError("n_samples must be at least 1.")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        output_samples = np.zeros((sample_count, len(rows)), dtype=float)
        for row_index, row in enumerate(rows):
            states = np.repeat(
                _initial_state(row, self.initial_state_dim),
                sample_count,
                axis=0,
            )
            steps = str(row["steps"])
            for operation in steps.split():
                model = self._operation_model(operation)
                mean, variance = model.predict_distribution(self._augment_input(states, steps))
                mean = _as_2d(mean)
                variance = _as_2d(variance)
                states = rng.normal(loc=mean, scale=np.sqrt(np.maximum(variance, 0.0)))
            if states.shape[1] != 1:
                raise ValueError(
                    "The final operation must return one scalar output. "
                    f"Got shape {states.shape} for protocol '{row['protocol']}'."
                )
            output_samples[:, row_index] = states[:, 0]
        return output_samples

    def _augment_input(self, state: np.ndarray, steps: str) -> np.ndarray:
        """Hook: optionally append a per-protocol context vector to an operation
        input. Default is identity (context-free); :class:`ContextOFMLModel`
        overrides this to append a presence/counts descriptor so shared operation
        models can condition on protocol context and resolve hidden-interaction
        misspecification (e.g. s7's set-gate, s8's dose-count)."""

        return state

    def _operation_model(self, operation: str) -> OperationModel:
        try:
            return self.operation_models[operation]
        except KeyError as error:
            raise ValueError(f"No operation model assigned for operation '{operation}'.") from error


def _as_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Operation model output must be 1D or 2D. Got shape {array.shape}.")


def _initial_state(row: Row, state_dim: int) -> np.ndarray:
    if state_dim < 1:
        raise ValueError("initial_state_dim must be at least 1.")
    values = [float(row[f"x{index}"]) for index in range(1, min(state_dim, 2) + 1)]
    for index in range(3, state_dim + 1):
        values.append(float(row.get(f"x{index}", 0.0)))
    return np.asarray([values], dtype=float)
