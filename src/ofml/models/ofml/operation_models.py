"""Basic operation-level model implementations."""

from __future__ import annotations

import csv
import dataclasses
import pathlib
from collections.abc import Sequence

import gpytorch
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from .base import OperationExample


def read_operation_examples(path: str | pathlib.Path) -> list[dict[str, str]]:
    """Read operation-level examples from CSV."""

    with pathlib.Path(path).open() as handle:
        return list(csv.DictReader(handle))


@dataclasses.dataclass
class LinearOperationModel:
    """Simple linear operation model for OFML interface smoke tests."""

    model_: LinearRegression | None = None
    output_dim_: int | None = None

    def fit(self, examples: Sequence[OperationExample]) -> "LinearOperationModel":
        """Fit from operation-level input/output examples."""

        if not examples:
            raise ValueError("Cannot fit LinearOperationModel with no examples.")
        output_dim = int(examples[0]["output_dim"])
        x = np.array([[float(example["input_1"]), float(example["input_2"])] for example in examples], dtype=float)
        y = np.array([_example_output(example, output_dim) for example in examples], dtype=float)
        self.model_ = LinearRegression()
        self.model_.fit(x, y)
        self.output_dim_ = output_dim
        return self

    def predict_state(self, states: np.ndarray) -> np.ndarray:
        """Predict output state or scalar output."""

        mean, _ = self.predict_distribution(states)
        return mean

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict output mean with zero variance."""

        if self.model_ is None:
            raise ValueError("LinearOperationModel must be fit before predict.")
        mean = np.asarray(self.model_.predict(np.asarray(states, dtype=float)), dtype=float)
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
        return mean, np.zeros_like(mean)


def _example_output(example: OperationExample, output_dim: int) -> list[float]:
    return [float(example[f"output_{index}"]) for index in range(1, output_dim + 1)]


@dataclasses.dataclass(frozen=True)
class OperationMLPConfig:
    """Training configuration for operation-level MLP models."""

    hidden_dim: int = 32
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    max_epochs: int = 300
    dropout: float = 0.0
    seed: int = 0
    spectral_norm: bool = False


class _OperationMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float,
                 n_hidden: int = 2) -> None:
        super().__init__()
        layers = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        for _ in range(max(1, n_hidden) - 1):
            layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        layers += [torch.nn.Linear(hidden_dim, output_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _linear(input_dim: int, output_dim: int, spectral_norm: bool) -> torch.nn.Module:
    layer = torch.nn.Linear(input_dim, output_dim)
    if spectral_norm:
        return torch.nn.utils.parametrizations.spectral_norm(layer)
    return layer


class _DKLFeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, spectral_norm: bool = False) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            _linear(input_dim, hidden_dim, spectral_norm),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            _linear(hidden_dim, hidden_dim, spectral_norm),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class _ExactDKLGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        feature_extractor: _DKLFeatureExtractor,
        feature_dim: int,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@dataclasses.dataclass
class MLPOperationModel:
    """Deterministic MLP operation model for OFML composition."""

    config: OperationMLPConfig = dataclasses.field(default_factory=OperationMLPConfig)
    model_: _OperationMLP | None = None
    output_dim_: int | None = None
    input_dim_: int | None = None

    def fit(self, examples: Sequence[OperationExample]) -> "MLPOperationModel":
        """Fit an operation-level MLP from input/output examples."""

        if not examples:
            raise ValueError("Cannot fit MLPOperationModel with no examples.")
        input_dim = _consistent_input_dim(examples)
        output_dim = _consistent_output_dim(examples)
        _set_torch_seed(self.config.seed)
        x = torch.as_tensor([_example_input(example, input_dim) for example in examples], dtype=torch.float32)
        y = torch.as_tensor([_example_output(example, output_dim) for example in examples], dtype=torch.float32)
        self.model_ = _OperationMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )
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
        self.output_dim_ = output_dim
        self.input_dim_ = input_dim
        return self

    def predict_state(self, states: np.ndarray) -> np.ndarray:
        """Predict output state or scalar output."""

        mean, _ = self.predict_distribution(states)
        return mean

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict output mean with zero variance."""

        if self.model_ is None:
            raise ValueError("MLPOperationModel must be fit before predict.")
        x = torch.as_tensor(np.asarray(states, dtype=float), dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            mean = self.model_(x).detach().cpu().numpy()
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
        return mean, np.zeros_like(mean)


@dataclasses.dataclass
class MCDOperationModel(MLPOperationModel):
    """MC-dropout operation model for OFML composition."""

    config: OperationMLPConfig = dataclasses.field(
        default_factory=lambda: OperationMLPConfig(dropout=0.1, max_epochs=300)
    )
    n_samples: int = 30

    def predict_samples(self, states: np.ndarray, n_samples: int | None = None) -> np.ndarray:
        """Draw MC-dropout output samples."""

        if self.model_ is None:
            raise ValueError("MCDOperationModel must be fit before predict.")
        sample_count = n_samples or self.n_samples
        x = torch.as_tensor(np.asarray(states, dtype=float), dtype=torch.float32)
        samples = []
        self.model_.train()
        with torch.no_grad():
            for _ in range(sample_count):
                samples.append(self.model_(x).detach().cpu().numpy())
        self.model_.eval()
        return np.asarray(samples, dtype=float)

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict MC-dropout mean and variance."""

        samples = self.predict_samples(states)
        return np.mean(samples, axis=0), np.var(samples, axis=0)


@dataclasses.dataclass
class EnsembleOperationModel:
    """Deep ensemble operation model for OFML composition."""

    n_members: int = 5
    config: OperationMLPConfig = dataclasses.field(default_factory=OperationMLPConfig)
    bootstrap: bool = True
    models_: list[MLPOperationModel] = dataclasses.field(default_factory=list)
    output_dim_: int | None = None

    def fit(self, examples: Sequence[OperationExample]) -> "EnsembleOperationModel":
        """Fit operation-level ensemble members."""

        if not examples:
            raise ValueError("Cannot fit EnsembleOperationModel with no examples.")
        self.output_dim_ = _consistent_output_dim(examples)
        rng = np.random.default_rng(self.config.seed)
        self.models_ = []
        for member_index in range(self.n_members):
            if self.bootstrap:
                indices = rng.integers(0, len(examples), size=len(examples))
                member_examples = [examples[int(index)] for index in indices]
            else:
                member_examples = list(examples)
            member_config = dataclasses.replace(self.config, seed=self.config.seed + member_index)
            self.models_.append(MLPOperationModel(config=member_config).fit(member_examples))
        return self

    def predict_state(self, states: np.ndarray) -> np.ndarray:
        """Predict ensemble mean state."""

        mean, _ = self.predict_distribution(states)
        return mean

    def predict_samples(self, states: np.ndarray) -> np.ndarray:
        """Return one output sample per ensemble member."""

        if not self.models_:
            raise ValueError("EnsembleOperationModel must be fit before predict.")
        return np.asarray([model.predict_state(states) for model in self.models_], dtype=float)

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict ensemble mean and variance."""

        samples = self.predict_samples(states)
        return np.mean(samples, axis=0), np.var(samples, axis=0)


@dataclasses.dataclass
class DKLOperationModel:
    """End-to-end deep-kernel operation model trained with gpytorch."""

    config: OperationMLPConfig = dataclasses.field(default_factory=OperationMLPConfig)
    alpha: float = 1e-6
    normalize_y: bool = True
    predictive_std_floor: float = 0.0
    gp_models_: list[_ExactDKLGP] = dataclasses.field(default_factory=list)
    likelihoods_: list[gpytorch.likelihoods.GaussianLikelihood] = dataclasses.field(default_factory=list)
    target_means_: list[float] = dataclasses.field(default_factory=list)
    target_stds_: list[float] = dataclasses.field(default_factory=list)
    input_mean_: np.ndarray | None = None
    input_std_: np.ndarray | None = None
    output_dim_: int | None = None
    input_dim_: int | None = None

    def hyperparameters(self) -> list[dict[str, float]]:
        """Return learned (target_std, outputscale, noise, mean lengthscale) per output GP."""

        out = []
        for index, (model, likelihood) in enumerate(zip(self.gp_models_, self.likelihoods_)):
            lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
            out.append({
                "target_std": float(self.target_stds_[index]) if self.target_stds_ else 1.0,
                "outputscale": float(model.covar_module.outputscale.detach()),
                "noise": float(likelihood.noise.detach().cpu().numpy().ravel()[0]),
                "lengthscale_mean": float(np.mean(lengthscale)),
            })
        return out

    def fit(self, examples: Sequence[OperationExample]) -> "DKLOperationModel":
        """Fit neural features and GP hyperparameters by marginal likelihood."""

        if not examples:
            raise ValueError("Cannot fit DKLOperationModel with no examples.")
        input_dim = _consistent_input_dim(examples)
        output_dim = _consistent_output_dim(examples)
        _set_torch_seed(self.config.seed)
        x_raw = np.asarray([_example_input(example, input_dim) for example in examples], dtype=float)
        self.input_mean_ = np.mean(x_raw, axis=0)
        self.input_std_ = np.std(x_raw, axis=0)
        self.input_std_[self.input_std_ < 1e-6] = 1.0
        x = torch.as_tensor((x_raw - self.input_mean_) / self.input_std_, dtype=torch.float32)
        y = torch.as_tensor([_example_output(example, output_dim) for example in examples], dtype=torch.float32)
        self.gp_models_ = []
        self.likelihoods_ = []
        self.target_means_ = []
        self.target_stds_ = []
        for output_index in range(output_dim):
            target = y[:, output_index]
            target_mean = float(torch.mean(target)) if self.normalize_y else 0.0
            target_std = float(torch.std(target)) if self.normalize_y else 1.0
            if target_std < 1e-6:
                target_std = 1.0
            train_y = (target - target_mean) / target_std
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
            )
            likelihood.noise = max(float(self.alpha), 1e-3)
            feature_extractor = _DKLFeatureExtractor(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
                spectral_norm=self.config.spectral_norm,
            )
            model = _ExactDKLGP(
                train_x=x,
                train_y=train_y,
                likelihood=likelihood,
                feature_extractor=feature_extractor,
                feature_dim=self.config.hidden_dim,
            )
            self._fit_gp_head(model, likelihood, x, train_y)
            self.gp_models_.append(model)
            self.likelihoods_.append(likelihood)
            self.target_means_.append(target_mean)
            self.target_stds_.append(target_std)
        self.output_dim_ = output_dim
        self.input_dim_ = input_dim
        return self

    def fine_tune(self, examples: Sequence[OperationExample], max_epochs: int) -> "DKLOperationModel":
        """Continue training from the current DKL parameters on updated examples."""

        if max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")
        if not self.gp_models_ or not self.likelihoods_ or self.input_dim_ is None or self.output_dim_ is None:
            return self.fit(examples)
        if not examples:
            raise ValueError("Cannot fine-tune DKLOperationModel with no examples.")

        input_dim = _consistent_input_dim(examples)
        output_dim = _consistent_output_dim(examples)
        if input_dim != self.input_dim_ or output_dim != self.output_dim_:
            return self.fit(examples)

        x_raw = np.asarray([_example_input(example, input_dim) for example in examples], dtype=float)
        self.input_mean_ = np.mean(x_raw, axis=0)
        self.input_std_ = np.std(x_raw, axis=0)
        self.input_std_[self.input_std_ < 1e-6] = 1.0
        x = torch.as_tensor((x_raw - self.input_mean_) / self.input_std_, dtype=torch.float32)
        y = torch.as_tensor([_example_output(example, output_dim) for example in examples], dtype=torch.float32)

        for output_index, (model, likelihood) in enumerate(zip(self.gp_models_, self.likelihoods_)):
            target = y[:, output_index]
            target_mean = float(torch.mean(target)) if self.normalize_y else 0.0
            target_std = float(torch.std(target)) if self.normalize_y else 1.0
            if target_std < 1e-6:
                target_std = 1.0
            train_y = (target - target_mean) / target_std
            model.set_train_data(inputs=x, targets=train_y, strict=False)
            self._fit_gp_head(model, likelihood, x, train_y, max_epochs=max_epochs)
            self.target_means_[output_index] = target_mean
            self.target_stds_[output_index] = target_std
        return self

    def predict_state(self, states: np.ndarray) -> np.ndarray:
        """Predict GP posterior mean."""

        mean, _ = self.predict_distribution(states)
        return mean

    def transform(self, states: np.ndarray) -> np.ndarray:
        """Return the learned deep-kernel feature embedding for input states.

        Useful for feature-space acquisition (k-center, ALC surrogate) without refitting.
        """

        if not self.gp_models_ or self.input_mean_ is None or self.input_std_ is None:
            raise ValueError("DKLOperationModel must be fit before transform.")
        x_raw = np.asarray(states, dtype=float)
        x = torch.as_tensor((x_raw - self.input_mean_) / self.input_std_, dtype=torch.float32)
        with torch.no_grad():
            features = self.gp_models_[0].feature_extractor(x).detach().cpu().numpy()
        return np.asarray(features, dtype=float)

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict GP posterior mean and variance with learned features."""

        if not self.gp_models_ or not self.likelihoods_:
            raise ValueError("DKLOperationModel must be fit before predict.")
        if self.input_mean_ is None or self.input_std_ is None:
            raise ValueError("DKLOperationModel input normalization is not initialized.")
        x_raw = np.asarray(states, dtype=float)
        x = torch.as_tensor((x_raw - self.input_mean_) / self.input_std_, dtype=torch.float32)
        means = []
        variances = []
        for gp_index, (model, likelihood) in enumerate(zip(self.gp_models_, self.likelihoods_)):
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = likelihood(model(x))
            target_std = self.target_stds_[gp_index]
            target_mean = self.target_means_[gp_index]
            mean = posterior.mean.detach().cpu().numpy() * target_std + target_mean
            variance = posterior.variance.detach().cpu().numpy() * (target_std**2)
            means.append(mean)
            variances.append(np.maximum(variance, 0.0))
        variance_out = np.stack(variances, axis=1)
        if self.predictive_std_floor > 0.0:
            variance_out = np.maximum(variance_out, self.predictive_std_floor**2)
        return np.stack(means, axis=1), variance_out

    def _fit_gp_head(
        self,
        model: _ExactDKLGP,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        x: torch.Tensor,
        y: torch.Tensor,
        max_epochs: int | None = None,
    ) -> None:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(self.config.max_epochs if max_epochs is None else max_epochs):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        model.eval()
        likelihood.eval()


class _StandardExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim: int) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


@dataclasses.dataclass
class GPOperationModel:
    """Plain exact GP operation model: ARD-RBF kernel on standardized inputs, no neural features.

    Calibration baseline against DKL: a standard GP's posterior variance grows away from the
    training inputs, so comparing it with DKL isolates whether neural features drive OOD overconfidence.
    """

    config: OperationMLPConfig = dataclasses.field(default_factory=OperationMLPConfig)
    alpha: float = 1e-6
    normalize_y: bool = True
    predictive_std_floor: float = 0.0
    gp_models_: list = dataclasses.field(default_factory=list)
    likelihoods_: list = dataclasses.field(default_factory=list)
    target_means_: list[float] = dataclasses.field(default_factory=list)
    target_stds_: list[float] = dataclasses.field(default_factory=list)
    input_mean_: np.ndarray | None = None
    input_std_: np.ndarray | None = None
    output_dim_: int | None = None
    input_dim_: int | None = None

    def hyperparameters(self) -> list[dict[str, float]]:
        """Return learned (target_std, outputscale, noise, mean lengthscale) per output GP."""

        out = []
        for index, (model, likelihood) in enumerate(zip(self.gp_models_, self.likelihoods_)):
            lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
            out.append({
                "target_std": float(self.target_stds_[index]) if self.target_stds_ else 1.0,
                "outputscale": float(model.covar_module.outputscale.detach()),
                "noise": float(likelihood.noise.detach().cpu().numpy().ravel()[0]),
                "lengthscale_mean": float(np.mean(lengthscale)),
            })
        return out

    def fit(self, examples: Sequence[OperationExample]) -> "GPOperationModel":
        if not examples:
            raise ValueError("Cannot fit GPOperationModel with no examples.")
        input_dim = _consistent_input_dim(examples)
        output_dim = _consistent_output_dim(examples)
        _set_torch_seed(self.config.seed)
        x_raw = np.asarray([_example_input(example, input_dim) for example in examples], dtype=float)
        self.input_mean_ = np.mean(x_raw, axis=0)
        self.input_std_ = np.std(x_raw, axis=0)
        self.input_std_[self.input_std_ < 1e-6] = 1.0
        x = torch.as_tensor((x_raw - self.input_mean_) / self.input_std_, dtype=torch.float32)
        y = torch.as_tensor([_example_output(example, output_dim) for example in examples], dtype=torch.float32)
        self.gp_models_, self.likelihoods_, self.target_means_, self.target_stds_ = [], [], [], []
        for output_index in range(output_dim):
            target = y[:, output_index]
            target_mean = float(torch.mean(target)) if self.normalize_y else 0.0
            target_std = float(torch.std(target)) if self.normalize_y else 1.0
            if target_std < 1e-6:
                target_std = 1.0
            train_y = (target - target_mean) / target_std
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
            likelihood.noise = max(float(self.alpha), 1e-3)
            model = _StandardExactGP(x, train_y, likelihood, input_dim)
            self._fit_gp(model, likelihood, x, train_y)
            self.gp_models_.append(model)
            self.likelihoods_.append(likelihood)
            self.target_means_.append(target_mean)
            self.target_stds_.append(target_std)
        self.output_dim_, self.input_dim_ = output_dim, input_dim
        return self

    def predict_state(self, states: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_distribution(states)
        return mean

    def transform(self, states: np.ndarray) -> np.ndarray:
        if self.input_mean_ is None or self.input_std_ is None:
            raise ValueError("GPOperationModel must be fit before transform.")
        return (np.asarray(states, dtype=float) - self.input_mean_) / self.input_std_

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.gp_models_ or self.input_mean_ is None:
            raise ValueError("GPOperationModel must be fit before predict.")
        x = torch.as_tensor((np.asarray(states, dtype=float) - self.input_mean_) / self.input_std_, dtype=torch.float32)
        means, variances = [], []
        for gp_index, (model, likelihood) in enumerate(zip(self.gp_models_, self.likelihoods_)):
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = likelihood(model(x))
            target_std, target_mean = self.target_stds_[gp_index], self.target_means_[gp_index]
            means.append(posterior.mean.detach().cpu().numpy() * target_std + target_mean)
            variances.append(np.maximum(posterior.variance.detach().cpu().numpy() * (target_std**2), 0.0))
        variance_out = np.stack(variances, axis=1)
        if self.predictive_std_floor > 0.0:
            variance_out = np.maximum(variance_out, self.predictive_std_floor**2)
        return np.stack(means, axis=1), variance_out

    def _fit_gp(self, model, likelihood, x: torch.Tensor, y: torch.Tensor) -> None:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(self.config.max_epochs):
            optimizer.zero_grad()
            loss = -mll(model(x), y)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        model.eval()
        likelihood.eval()


@dataclasses.dataclass
class UnknownOperationModel:
    """Untrained fallback operation model for unknown operations."""

    backend: str = "mlp"
    input_dim: int = 2
    output_dim: int = 2
    config: OperationMLPConfig = dataclasses.field(default_factory=OperationMLPConfig)
    prior_variance: float = 1e3
    model_: _OperationMLP | None = None
    feature_extractor_: _DKLFeatureExtractor | None = None
    dkl_head_: torch.nn.Linear | None = None

    def __post_init__(self) -> None:
        if self.output_dim < 1:
            raise ValueError("output_dim must be at least 1.")
        _set_torch_seed(self.config.seed)
        if self.backend == "mlp":
            self.model_ = _OperationMLP(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
            )
            self.model_.eval()
        elif self.backend == "dkl":
            self.feature_extractor_ = _DKLFeatureExtractor(
                input_dim=self.input_dim,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
            )
            self.dkl_head_ = torch.nn.Linear(self.config.hidden_dim, self.output_dim)
            self.feature_extractor_.eval()
            self.dkl_head_.eval()
        else:
            raise ValueError("backend must be either 'mlp' or 'dkl'.")

    def fit(self, examples: Sequence[OperationExample]) -> "UnknownOperationModel":
        """Keep the unknown operation intentionally untrained."""

        return self

    def predict_state(self, states: np.ndarray) -> np.ndarray:
        """Predict untrained output state or scalar output."""

        mean, _ = self.predict_distribution(states)
        return mean

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with random initialized parameters and large prior variance."""

        x = torch.as_tensor(np.asarray(states, dtype=float), dtype=torch.float32)
        with torch.no_grad():
            if self.backend == "mlp":
                if self.model_ is None:
                    raise ValueError("UnknownOperationModel MLP backend is not initialized.")
                mean = self.model_(x).detach().cpu().numpy()
            else:
                if self.feature_extractor_ is None or self.dkl_head_ is None:
                    raise ValueError("UnknownOperationModel DKL backend is not initialized.")
                features = self.feature_extractor_(x)
                mean = self.dkl_head_(features).detach().cpu().numpy()
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
        variance = np.full_like(mean, fill_value=float(self.prior_variance), dtype=float)
        return mean, variance


def _consistent_output_dim(examples: Sequence[OperationExample]) -> int:
    output_dims = {int(example["output_dim"]) for example in examples}
    if len(output_dims) != 1:
        raise ValueError(f"Operation examples must have one output_dim. Got {sorted(output_dims)}.")
    return output_dims.pop()


def _consistent_input_dim(examples: Sequence[OperationExample]) -> int:
    input_dims = {int(example.get("input_dim", 2)) for example in examples}
    if len(input_dims) != 1:
        raise ValueError(f"Operation examples must have one input_dim. Got {sorted(input_dims)}.")
    input_dim = input_dims.pop()
    for example in examples:
        for index in range(1, input_dim + 1):
            if f"input_{index}" not in example:
                raise ValueError(f"Missing input_{index} for input_dim={input_dim}.")
    return input_dim


def _example_input(example: OperationExample, input_dim: int) -> list[float]:
    return [float(example[f"input_{index}"]) for index in range(1, input_dim + 1)]


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
