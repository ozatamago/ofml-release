"""Shared black-box MLP models for protocol benchmarks."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row, target_vector

from .features import RowFeatureEncoder


@dataclasses.dataclass(frozen=True)
class SharedMLPConfig:
    """Training configuration for shared MLP models."""

    hidden_dim: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    max_epochs: int = 300
    dropout: float = 0.0
    seed: int = 0


class _MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclasses.dataclass
class SharedMLP:
    """Fit one deterministic MLP across all protocols."""

    config: SharedMLPConfig = dataclasses.field(default_factory=SharedMLPConfig)
    encoder: RowFeatureEncoder = dataclasses.field(default_factory=RowFeatureEncoder)
    model_: _MLP | None = None

    def fit(self, rows: Sequence[Row]) -> "SharedMLP":
        """Fit the shared MLP."""

        if not rows:
            raise ValueError("Cannot fit SharedMLP with no rows.")
        _set_torch_seed(self.config.seed)
        x = torch.as_tensor(self.encoder.fit_transform(rows), dtype=torch.float32)
        y = torch.as_tensor(target_vector(rows), dtype=torch.float32)
        self.model_ = _MLP(x.shape[1], self.config.hidden_dim, self.config.dropout)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.model_.train()
        for _ in range(self.config.max_epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(self.model_(x), y)
            loss.backward()
            optimizer.step()
        self.model_.eval()
        return self

    def predict(self, rows: Sequence[Row]) -> np.ndarray:
        """Predict deterministic means for rows."""

        return self.predict_samples(rows, n_samples=1)[0]

    def predict_samples(self, rows: Sequence[Row], n_samples: int = 1) -> np.ndarray:
        """Return repeated deterministic predictions."""

        if self.model_ is None:
            raise ValueError("SharedMLP must be fit before predict.")
        x = torch.as_tensor(self.encoder.transform(rows), dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            prediction = self.model_(x).detach().cpu().numpy()
        return np.repeat(prediction.reshape(1, -1), n_samples, axis=0)

    def predict_distribution(self, rows: Sequence[Row]) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance for rows."""

        mean = self.predict(rows)
        return mean, np.zeros_like(mean)


@dataclasses.dataclass
class SharedMLPMCD(SharedMLP):
    """Shared MLP with MC dropout predictive samples."""

    config: SharedMLPConfig = dataclasses.field(
        default_factory=lambda: SharedMLPConfig(dropout=0.1, max_epochs=300)
    )

    def predict_samples(self, rows: Sequence[Row], n_samples: int = 30) -> np.ndarray:
        """Draw MC dropout prediction samples."""

        if self.model_ is None:
            raise ValueError("SharedMLPMCD must be fit before predict.")
        x = torch.as_tensor(self.encoder.transform(rows), dtype=torch.float32)
        samples = []
        self.model_.train()
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.model_(x).detach().cpu().numpy())
        self.model_.eval()
        return np.asarray(samples, dtype=float)

    def predict_distribution(self, rows: Sequence[Row]) -> tuple[np.ndarray, np.ndarray]:
        """Predict MC dropout mean and variance."""

        samples = self.predict_samples(rows)
        return np.mean(samples, axis=0), np.var(samples, axis=0)


@dataclasses.dataclass
class SharedMLPEnsemble:
    """Deep ensemble of shared MLP models."""

    n_members: int = 5
    config: SharedMLPConfig = dataclasses.field(default_factory=SharedMLPConfig)
    bootstrap: bool = True
    models_: list[SharedMLP] = dataclasses.field(default_factory=list)

    def fit(self, rows: Sequence[Row]) -> "SharedMLPEnsemble":
        """Fit ensemble members with different seeds and optional bootstrap rows."""

        if not rows:
            raise ValueError("Cannot fit SharedMLPEnsemble with no rows.")
        rng = np.random.default_rng(self.config.seed)
        self.models_ = []
        for member_index in range(self.n_members):
            if self.bootstrap:
                indices = rng.integers(0, len(rows), size=len(rows))
                member_rows = [rows[int(index)] for index in indices]
            else:
                member_rows = list(rows)
            member_config = dataclasses.replace(self.config, seed=self.config.seed + member_index)
            model = SharedMLP(config=member_config).fit(member_rows)
            self.models_.append(model)
        return self

    def predict(self, rows: Sequence[Row]) -> np.ndarray:
        """Predict ensemble mean for rows."""

        mean, _ = self.predict_distribution(rows)
        return mean

    def predict_samples(self, rows: Sequence[Row]) -> np.ndarray:
        """Return one prediction vector per ensemble member."""

        if not self.models_:
            raise ValueError("SharedMLPEnsemble must be fit before predict.")
        return np.asarray([model.predict(rows) for model in self.models_], dtype=float)

    def predict_distribution(self, rows: Sequence[Row]) -> tuple[np.ndarray, np.ndarray]:
        """Predict ensemble mean and variance."""

        samples = self.predict_samples(rows)
        return np.mean(samples, axis=0), np.var(samples, axis=0)


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
