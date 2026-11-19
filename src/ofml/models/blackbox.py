"""Black-box GP models for protocol benchmarks."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row, feature_matrix, protocol_labels, target_vector

from .features import RowFeatureEncoder


def _default_kernel():
    return ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
        length_scale=1.0,
        length_scale_bounds="fixed",
    ) + WhiteKernel(
        noise_level=1e-4,
        noise_level_bounds="fixed",
    )


@dataclasses.dataclass
class ProtocolSpecificGP:
    """Fit one black-box GP per protocol."""

    alpha: float = 1e-6
    normalize_y: bool = True
    random_state: int = 0
    models_: dict[str, GaussianProcessRegressor] = dataclasses.field(default_factory=dict)

    def fit(self, rows: Sequence[Row]) -> "ProtocolSpecificGP":
        """Fit a separate GP for every protocol in rows."""

        if not rows:
            raise ValueError("Cannot fit ProtocolSpecificGP with no rows.")

        labels = protocol_labels(rows)
        self.models_ = {}
        for protocol in sorted(set(labels)):
            protocol_rows = [row for row in rows if row["protocol"] == protocol]
            model = GaussianProcessRegressor(
                kernel=_default_kernel(),
                alpha=self.alpha,
                normalize_y=self.normalize_y,
                random_state=self.random_state,
            )
            model.fit(feature_matrix(protocol_rows), target_vector(protocol_rows))
            self.models_[protocol] = model
        return self

    def predict(self, rows: Sequence[Row]) -> np.ndarray:
        """Predict means for rows."""

        mean, _ = self.predict_distribution(rows)
        return mean

    def predict_distribution(self, rows: Sequence[Row]) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance for rows."""

        if not self.models_:
            raise ValueError("ProtocolSpecificGP must be fit before predict.")
        means = np.zeros(len(rows), dtype=float)
        variances = np.zeros(len(rows), dtype=float)
        for index, row in enumerate(rows):
            protocol = row["protocol"]
            if protocol not in self.models_:
                raise ValueError(f"No protocol-specific GP was fit for protocol '{protocol}'.")
            mean, std = self.models_[protocol].predict(feature_matrix([row]), return_std=True)
            means[index] = float(mean[0])
            variances[index] = float(std[0] ** 2)
        return means, variances


@dataclasses.dataclass
class PooledBlackBoxGP:
    """Fit one GP over experiment conditions and protocol features."""

    encoder: RowFeatureEncoder = dataclasses.field(
        default_factory=lambda: RowFeatureEncoder(include_protocol_id=True, include_step_counts=True)
    )
    alpha: float = 1e-6
    normalize_y: bool = True
    random_state: int = 0
    model_: GaussianProcessRegressor | None = None

    def fit(self, rows: Sequence[Row]) -> "PooledBlackBoxGP":
        """Fit a pooled GP across all protocols."""

        if not rows:
            raise ValueError("Cannot fit PooledBlackBoxGP with no rows.")
        x = self.encoder.fit_transform(rows)
        y = target_vector(rows)
        self.model_ = GaussianProcessRegressor(
            kernel=_default_kernel(),
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
        )
        self.model_.fit(x, y)
        return self

    def predict(self, rows: Sequence[Row]) -> np.ndarray:
        """Predict means for rows."""

        mean, _ = self.predict_distribution(rows)
        return mean

    def predict_distribution(self, rows: Sequence[Row]) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance for rows."""

        if self.model_ is None:
            raise ValueError("PooledBlackBoxGP must be fit before predict.")
        mean, std = self.model_.predict(self.encoder.transform(rows), return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float) ** 2
