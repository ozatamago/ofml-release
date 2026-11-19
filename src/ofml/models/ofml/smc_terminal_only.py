"""GPU-vectorized tempered SMC for terminal-only OFML (OFML-BNN / OFML-GPN).

Motivation. With HMC the composed BNN/GPN posterior is highly multimodal; poor
mode exploration makes the pushforward predictive average over unexplored modes,
so the deep composition's output *flattens* toward the prior (the well-known
"deep network prediction reverts to the mean" pathology; in this project's
cx_bayes study the dominant error was exactly a multimodal-MCMC ESS shortfall,
fixed by SMC/tempering). This module replaces HMC with an adaptive tempered
Sequential Monte Carlo sampler and pushes the particle count as high as the GPU
allows, so the effective sample size is large and the predictive stays sharp.

Everything is vectorized over particles with ``torch.func.vmap`` + ``functional_call``
and runs on CUDA: N particles are a single ``[N, dim]`` tensor, the composed
forward / per-particle gradient are one batched kernel each. One V100 handles
several thousand particles at once.

Sampler (Del Moral–Doucet–Jasra adaptive SMC sampler):
  * particles ~ prior N(0, prior_std^2)
  * anneal beta: 0 -> 1, each step chosen by bisection so the incremental-weight
    ESS hits ess_frac*N
  * systematic resampling every step (weights reset to uniform)
  * MALA rejuvenation (n_rejuv steps) targeting the tempered posterior p_beta,
    with per-particle step-size adaptation toward the MALA-optimal ~0.574 accept.
Final particles at beta=1 are the posterior sample; predictive mean/var is their
pushforward spread.
"""
from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from .bayesian_terminal_only import _ComposedNet, BayesianTerminalOnlyConfig, _group


@dataclasses.dataclass(frozen=True)
class SMCConfig:
    model: str = "bnn"            # "bnn" or "gpn"
    latent_dim: int = 5
    hidden_dim: int = 16
    rff_features: int = 24
    rff_lengthscale: float = 1.0
    prior_std: float = 0.4
    sigma_obs: float = 0.05
    # SMC
    n_particles: int = 4096
    ess_frac: float = 0.5         # target incremental-weight ESS = ess_frac * N
    n_rejuv: int = 12             # MALA rejuvenation steps per tempering stage
    mala_step0: float = 0.02      # initial MALA step (adapted online)
    max_stages: int = 200
    device: str = "cuda"
    seed: int = 0


def _base_config(cfg: SMCConfig) -> BayesianTerminalOnlyConfig:
    return BayesianTerminalOnlyConfig(
        model=cfg.model, latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim,
        rff_features=cfg.rff_features, rff_lengthscale=cfg.rff_lengthscale,
        prior_std=cfg.prior_std, sigma_obs=cfg.sigma_obs, seed=cfg.seed)


def _systematic_resample(w: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Return resampled indices [N] for normalized weights ``w`` [N]."""
    n = w.shape[0]
    u0 = torch.rand((), generator=gen, device=w.device) / n
    positions = u0 + torch.arange(n, device=w.device) / n
    cum = torch.cumsum(w, 0)
    cum[-1] = 1.0
    return torch.searchsorted(cum, positions).clamp_(0, n - 1)


@dataclasses.dataclass
class SMCTerminalOnlyOFML:
    config: SMCConfig = dataclasses.field(default_factory=SMCConfig)
    samples_: torch.Tensor | None = None      # [N, dim] on cpu
    diagnostics_: dict | None = None
    y_mean_: float = 0.0
    y_std_: float = 1.0

    def _build(self):
        net = _ComposedNet(_base_config(self.config))
        names = [n for n, _ in net.named_parameters()]
        shapes = [tuple(p.shape) for _, p in net.named_parameters()]
        numels = [int(np.prod(s)) for s in shapes]
        return net, names, shapes, numels

    def fit(self, rows: Sequence[Mapping[str, str]]) -> "SMCTerminalOnlyOFML":
        cfg = self.config
        dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        gen = torch.Generator(device=dev).manual_seed(cfg.seed * 7 + 1)

        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        y = np.array([float(r["y"]) for r in rows], dtype=float)
        self.y_mean_ = float(y.mean()); self.y_std_ = float(y.std()) or 1.0

        net, names, shapes, numels = self._build()
        net = net.to(dev)
        buffers = {k: v.to(dev) for k, v in net.named_buffers()}
        dim = int(sum(numels))
        prior_var = cfg.prior_std ** 2
        obs_var = cfg.sigma_obs ** 2

        batches = []
        for steps, idx in _group(rows).items():
            xb = torch.as_tensor(x[idx], dtype=torch.float32, device=dev)
            yb = torch.as_tensor((y[idx] - self.y_mean_) / self.y_std_, dtype=torch.float32, device=dev)
            batches.append((steps.split(), xb, yb))

        offs, off = [], 0
        for n in numels:
            offs.append((off, off + n)); off += n

        def unflatten(theta):  # single particle [dim] -> params dict
            return {names[i]: theta[a:b].reshape(shapes[i]) for i, (a, b) in enumerate(offs)}

        def loglik_single(theta):
            p = {**unflatten(theta), **buffers}
            tot = theta.new_zeros(())
            for steps, xb, yb in batches:
                pred = functional_call(net, p, (xb, steps))
                tot = tot - 0.5 * torch.sum((pred - yb) ** 2) / obs_var
            return tot

        def logprior_single(theta):
            return -0.5 * torch.sum(theta ** 2) / prior_var

        def logpost_single(theta, beta):
            return beta * loglik_single(theta) + logprior_single(theta)

        loglik_v = vmap(loglik_single)
        logprior_v = vmap(logprior_single)
        gradpost_v = vmap(grad(logpost_single, argnums=0), in_dims=(0, None))

        N = cfg.n_particles
        theta = torch.randn(N, dim, generator=gen, device=dev) * cfg.prior_std
        loglik = loglik_v(theta)                          # [N]
        beta = 0.0
        eps = cfg.mala_step0
        target_ess = cfg.ess_frac * N
        stage = 0
        accepts = []

        def ess_of_delta(delta):
            lw = delta * (loglik - loglik.max())
            w = torch.softmax(lw, 0)
            return float(1.0 / torch.sum(w ** 2))

        while beta < 1.0 and stage < cfg.max_stages:
            stage += 1
            # --- adaptive temperature: bisection for delta s.t. ESS == target ---
            hi = 1.0 - beta
            if ess_of_delta(hi) >= target_ess:
                delta = hi
            else:
                lo, d = 0.0, hi
                for _ in range(40):
                    mid = 0.5 * (lo + d)
                    if ess_of_delta(mid) >= target_ess:
                        lo = mid
                    else:
                        d = mid
                delta = lo
            delta = max(delta, 1e-4)
            beta_new = min(1.0, beta + delta)

            # --- reweight + systematic resample ---
            lw = (beta_new - beta) * (loglik - loglik.max())
            w = torch.softmax(lw, 0)
            idx = _systematic_resample(w, gen)
            theta = theta[idx].contiguous()
            beta = beta_new

            # --- MALA rejuvenation at beta ---
            lp = beta * loglik_v(theta) + logprior_v(theta)
            g = gradpost_v(theta, beta)
            for _ in range(cfg.n_rejuv):
                noise = torch.randn(N, dim, generator=gen, device=dev)
                prop = theta + 0.5 * eps ** 2 * g + eps * noise
                lp_prop = beta * loglik_v(prop) + logprior_v(prop)
                gp = gradpost_v(prop, beta)
                # MALA proposal log-densities
                mu_fwd = theta + 0.5 * eps ** 2 * g
                mu_bwd = prop + 0.5 * eps ** 2 * gp
                logq_fwd = -torch.sum((prop - mu_fwd) ** 2, 1) / (2 * eps ** 2)
                logq_bwd = -torch.sum((theta - mu_bwd) ** 2, 1) / (2 * eps ** 2)
                log_acc = lp_prop - lp + logq_bwd - logq_fwd
                u = torch.rand(N, generator=gen, device=dev)
                acc = torch.log(u + 1e-38) < log_acc
                a = acc.float().mean().item(); accepts.append(a)
                theta = torch.where(acc[:, None], prop, theta)
                lp = torch.where(acc, lp_prop, lp)
                g = torch.where(acc[:, None], gp, g)
                eps *= math.exp(0.35 * (a - 0.574))       # adapt toward MALA-optimal accept
                eps = min(max(eps, 1e-4), 0.5)
            loglik = loglik_v(theta)                       # refresh for next stage

        self.samples_ = theta.detach().to("cpu")
        self._net = net.to("cpu")
        self._buffers = {k: v.to("cpu") for k, v in buffers.items()}
        self._unflatten_cpu = lambda th: {names[i]: th[a:b].reshape(shapes[i]) for i, (a, b) in enumerate(offs)}
        self._offs, self._names, self._shapes = offs, names, shapes
        self.diagnostics_ = {
            "n_particles": N, "stages": stage, "final_beta": beta,
            "mean_accept": float(np.mean(accepts)) if accepts else float("nan"),
            "final_eps": eps, "dim": dim, "device": str(dev),
        }
        return self

    def predict_distribution(self, rows: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        if self.samples_ is None:
            raise ValueError("SMCTerminalOnlyOFML must be fit before predict.")
        dev = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        net = self._net.to(dev)
        buffers = {k: v.to(dev) for k, v in self._buffers.items()}
        theta = self.samples_.to(dev)
        names, shapes, offs = self._names, self._shapes, self._offs

        def unflatten(th):
            return {names[i]: th[a:b].reshape(shapes[i]) for i, (a, b) in enumerate(offs)}

        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        mean = np.zeros(len(rows)); var = np.zeros(len(rows))
        with torch.no_grad():
            for steps, idx in _group(rows).items():
                xb = torch.as_tensor(x[idx], dtype=torch.float32, device=dev)
                sl = steps.split()

                def one(th):
                    return functional_call(net, {**unflatten(th), **buffers}, (xb, sl))

                preds = vmap(one)(theta)                   # [N, n_group]
                p = preds.to("cpu").numpy() * self.y_std_ + self.y_mean_
                m = p.mean(0); v = p.var(0)
                for k, i in enumerate(idx):
                    mean[i] = m[k]; var[i] = v[k]
        net.to("cpu")
        return mean, var

    def predict_samples(self, rows: Sequence[Mapping[str, str]], n_samples: int = 512, seed: int = 0) -> np.ndarray:
        """Per-particle pushforward predictions -> [S, n_rows] (a subsample of the SMC
        particles). Feeds the covariance-based acquisitions (IVR / MaxVar); with SMC the
        effective sample size is large so the acquisition covariance is well-resolved."""
        if self.samples_ is None:
            raise ValueError("SMCTerminalOnlyOFML must be fit before predict_samples.")
        dev = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        net = self._net.to(dev)
        buffers = {k: v.to(dev) for k, v in self._buffers.items()}
        theta = self.samples_
        N = theta.shape[0]
        if n_samples < N:
            g = torch.Generator().manual_seed(seed)
            theta = theta[torch.randperm(N, generator=g)[:n_samples]]
        theta = theta.to(dev)
        names, shapes, offs = self._names, self._shapes, self._offs

        def unflatten(th):
            return {names[i]: th[a:b].reshape(shapes[i]) for i, (a, b) in enumerate(offs)}

        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        out = np.zeros((theta.shape[0], len(rows)))
        with torch.no_grad():
            for steps, idx in _group(rows).items():
                xb = torch.as_tensor(x[idx], dtype=torch.float32, device=dev); sl = steps.split()

                def one(th):
                    return functional_call(net, {**unflatten(th), **buffers}, (xb, sl))

                preds = vmap(one)(theta).to("cpu").numpy() * self.y_std_ + self.y_mean_  # [S, n_group]
                for k, i in enumerate(idx):
                    out[:, i] = preds[:, k]
        net.to("cpu")
        return out

    def predict(self, rows):
        return self.predict_distribution(rows)[0]
