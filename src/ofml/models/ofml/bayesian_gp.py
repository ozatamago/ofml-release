"""Fully-Bayesian (NUTS) GP-head operation models: M2 (Matern/ARD), M3 (SAAS).

Each output coordinate gets a botorch fully-Bayesian single-task GP fit by NUTS. The MCMC draws
are treated as posterior "members": `predict_distribution` returns the Bayesian model-average
(total predictive mean and variance = within-sample + between-sample), and `draw_predictor(s)`
returns a single-draw predictor so the OFML composition can be threaded draw-by-draw (giving
row-level posterior samples for epistemic / BALD / IVR acquisition, like the deep ensemble).

Variants (set at construction):
  * prior="matern" : standard ARD Matern fully-Bayesian GP  (M2 / plain-GP-NUTS)
  * prior="saas"   : SAAS sparse-lengthscale prior          (M3)
  * features="identity" : GP on standardized raw inputs     (plain GP NUTS)
  * features="dkl"      : MAP-train DKL features, freeze them, NUTS GP head on the features
                          (M2 frozen-feature head; also stage 1 of fully-Bayesian DKL / M4)
"""

from __future__ import annotations

import dataclasses
import gc

import numpy as np
import torch
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import MaternPyroModel, SaasFullyBayesianSingleTaskGP

from .base import OperationExample
from .operation_models import (
    DKLOperationModel,
    OperationMLPConfig,
    _consistent_input_dim,
    _consistent_output_dim,
    _example_input,
    _example_output,
    _set_torch_seed,
)


@dataclasses.dataclass
class FullyBayesianGPOperationModel:
    """NUTS GP-head operation model (one fully-Bayesian GP per output coordinate)."""

    config: OperationMLPConfig = dataclasses.field(default_factory=OperationMLPConfig)
    prior: str = "matern"  # "matern" (ARD, M2) or "saas" (M3)
    features: str = "identity"  # "identity" (plain GP NUTS) or "dkl" (frozen DKL features)
    warmup_steps: int = 64
    num_samples: int = 128
    thinning: int = 4
    gps_: list = dataclasses.field(default_factory=list)
    y_means_: list[float] = dataclasses.field(default_factory=list)
    y_stds_: list[float] = dataclasses.field(default_factory=list)
    input_mean_: np.ndarray | None = None
    input_std_: np.ndarray | None = None
    dkl_: DKLOperationModel | None = None
    output_dim_: int | None = None
    input_dim_: int | None = None
    n_samples_: int = 0

    # ------------------------------------------------------------------ features

    def _standardize(self, x_raw: np.ndarray) -> np.ndarray:
        return (np.asarray(x_raw, dtype=float) - self.input_mean_) / self.input_std_

    def _features_for_dim(self, x_raw: np.ndarray, output_index: int) -> torch.Tensor:
        x_std = torch.as_tensor(self._standardize(x_raw), dtype=torch.float32)
        if self.features == "dkl":
            extractor = self.dkl_.gp_models_[output_index].feature_extractor
            with torch.no_grad():
                feats = extractor(x_std).detach().cpu().numpy()
            return torch.as_tensor(feats, dtype=torch.double)
        return torch.as_tensor(x_std.numpy(), dtype=torch.double)

    # ------------------------------------------------------------------ fit

    def fit(self, examples) -> "FullyBayesianGPOperationModel":
        if not examples:
            raise ValueError("Cannot fit FullyBayesianGPOperationModel with no examples.")
        input_dim = _consistent_input_dim(examples)
        output_dim = _consistent_output_dim(examples)
        _set_torch_seed(self.config.seed)
        x_raw = np.asarray([_example_input(e, input_dim) for e in examples], dtype=float)
        y_raw = np.asarray([_example_output(e, output_dim) for e in examples], dtype=float)
        self.input_mean_ = np.mean(x_raw, axis=0)
        self.input_std_ = np.std(x_raw, axis=0)
        self.input_std_[self.input_std_ < 1e-6] = 1.0

        if self.features == "dkl":
            self.dkl_ = DKLOperationModel(config=self.config).fit(examples)

        self.gps_, self.y_means_, self.y_stds_ = [], [], []
        for j in range(output_dim):
            phi = self._features_for_dim(x_raw, j)
            y = y_raw[:, j]
            y_mean, y_std = float(np.mean(y)), float(np.std(y))
            if y_std < 1e-6:
                y_std = 1.0
            train_y = torch.as_tensor(((y - y_mean) / y_std).reshape(-1, 1), dtype=torch.double)
            gp = SaasFullyBayesianSingleTaskGP(train_X=phi, train_Y=train_y)
            if self.prior == "matern":
                gp.pyro_model = MaternPyroModel()
                gp.pyro_model.set_inputs(phi, train_y, None)
            elif self.prior != "saas":
                raise ValueError(f"Unknown prior '{self.prior}'.")
            fit_fully_bayesian_model_nuts(
                gp, warmup_steps=self.warmup_steps, num_samples=self.num_samples,
                thinning=self.thinning, disable_progbar=True,
            )
            self.gps_.append(gp)
            self.y_means_.append(y_mean)
            self.y_stds_.append(y_std)
            gc.collect()
        self.output_dim_, self.input_dim_ = output_dim, input_dim
        self.n_samples_ = int(self.gps_[0].num_mcmc_samples)
        return self

    # ------------------------------------------------------------------ predict

    def _sample_predictions(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-MCMC-sample means and variances: (S, n, output_dim)."""
        x_raw = np.asarray(states, dtype=float)
        means, varis = [], []
        for j, gp in enumerate(self.gps_):
            phi = self._features_for_dim(x_raw, j)
            gp.eval()
            with torch.no_grad():
                post = gp.posterior(phi)
                m = post.mean.detach().cpu().numpy()  # (S, n, 1)
                v = post.variance.detach().cpu().numpy()
            m = m[..., 0] * self.y_stds_[j] + self.y_means_[j]  # (S, n)
            v = np.maximum(v[..., 0], 0.0) * (self.y_stds_[j] ** 2)
            means.append(m)
            varis.append(v)
        means = np.stack(means, axis=2)  # (S, n, O)
        varis = np.stack(varis, axis=2)
        return means, varis

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bayesian model-average: total mean and total predictive variance over MCMC draws."""
        means, varis = self._sample_predictions(states)
        total_mean = means.mean(0)
        total_var = varis.mean(0) + means.var(0)  # E_s[Var] + Var_s[E]
        return total_mean, total_var

    def predict_state(self, states: np.ndarray) -> np.ndarray:
        return self.predict_distribution(states)[0]

    def draw_predictor(self, sample_index: int) -> "_DrawPredictor":
        return _DrawPredictor(self, sample_index)


@dataclasses.dataclass
class _DrawPredictor:
    """Single-MCMC-draw view of a FullyBayesianGPOperationModel (for draw-by-draw composition)."""

    parent: FullyBayesianGPOperationModel
    sample_index: int

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, varis = self.parent._sample_predictions(states)
        s = self.sample_index
        return means[s], varis[s]

    def predict_state(self, states: np.ndarray) -> np.ndarray:
        return self.predict_distribution(states)[0]
