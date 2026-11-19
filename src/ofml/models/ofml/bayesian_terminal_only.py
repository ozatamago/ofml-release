"""Bayesian terminal-observation-only OFML (OFML-BNN / OFML-GPN).

This is the *principled* companion to :class:`TerminalOnlyOFML` (point / deep
ensemble / MC-dropout). It ports the cx_bayes common-terminal-observer function
network to the OFML setting:

    G(protocol, x) = f_o( u_p(x) ),

where ``u_p`` is the protocol-specific composition of *shared* per-operation
modules and ``f_o`` is the shared terminal observer. Only the terminal scalar
``y`` is measured per protocol run -- intermediate operation states are latent.
A full posterior over all (shared) operation parameters is inferred from the
terminal scalar alone by Hamiltonian Monte Carlo, and predictive uncertainty is
the pushforward spread of that posterior through the composition.

Two module families (cx_bayes ``bnn2.py`` / ``models.py`` ChainGPN):

* ``model="bnn"`` -- each operation is a small tanh MLP; a Gaussian prior on all
  weights. Composition is NON-residual ``s <- f_op(s)`` (general module map, per the
  generalized OFML setting); an operation never seen in training maps via its prior,
  not the identity.
* ``model="gpn"`` -- each operation is a random-feature GP: a *frozen* random
  Fourier basis ``phi(s) = sqrt(2/L) cos(Omega s + b)`` (``Omega ~ N(0, 1/l^2)``)
  with *Bayesian linear* readout coefficients (Gaussian prior). This is a
  finite-basis GP per operation; the posterior over coefficients is exactly the
  weight-space view of a GP, so ``model="gpn"`` is the compositional terminal-only
  Gaussian-process network.

Inference. Rather than the hand-coded analytic Jacobian of cx_bayes, gradients of
the log-posterior w.r.t. the flattened parameter vector are obtained by torch
autograd through the composition (``torch.func.functional_call``), which is exact
and composition-agnostic. The sampler is standard HMC with dual-averaging
step-size adaptation (Hoffman & Gelman 2014) and a diagonal identity mass matrix,
run over several chains started from independent prior draws. Convergence is
monitored in *function space* (split-R-hat / ESS on a probe-set mean prediction),
following the cx_bayes finding that parameter-space diagnostics are unreliable for
these multimodal weight-space posteriors.

Row-level interface: ``fit(rows)`` / ``predict_distribution(rows)`` on
``(x1, x2, steps, y)`` only -- no operation examples / trace.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch.func import functional_call

INTERMEDIATE_OPS = ("A", "B", "C", "D", "M")


@dataclasses.dataclass(frozen=True)
class BayesianTerminalOnlyConfig:
    model: str = "bnn"              # "bnn" (tanh MLP ops) or "gpn" (RFF-GP ops)
    latent_dim: int = 5             # headroom ~2.5x the true 2-D state
    hidden_dim: int = 16            # BNN op hidden width
    rff_features: int = 24          # GPN random-feature count L per op
    rff_lengthscale: float = 1.0    # GPN RFF frequency scale (Omega ~ N(0, 1/l^2))
    prior_std: float = 0.4          # Gaussian prior std on all sampled parameters
    sigma_obs: float = 0.05         # observation-noise std (standardized y space)
    trace_weight: float = 1.0       # weight of the optional intermediate-observation likelihood
    # HMC
    num_warmup: int = 300
    num_samples: int = 200
    num_chains: int = 4
    n_leapfrog: int = 15
    target_accept: float = 0.75
    thin: int = 2
    seed: int = 0


# --------------------------------------------------------------------------
# Composition modules (forward(x, steps) -> terminal scalar; differentiable
# w.r.t. parameters via functional_call).


class _OperationMLP(torch.nn.Module):
    """Non-residual tanh MLP operation module (BNN family): s <- f_op(s)."""

    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(d, hidden)
        self.l2 = torch.nn.Linear(hidden, d)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.tanh(self.l1(s)))


class _ObserverMLP(torch.nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(d, hidden)
        self.l2 = torch.nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.tanh(self.l1(s)))[:, 0]


class _OperationRFF(torch.nn.Module):
    """Non-residual random-feature GP operation module (GPN family): s <- f_op(s).

    ``Omega``/``b`` are *frozen* random features (buffers); only ``coef`` is a
    Gaussian-prior parameter, so this is a finite-basis GP on the operation input.
    """

    def __init__(self, d: int, n_feat: int, lengthscale: float, gen: torch.Generator) -> None:
        super().__init__()
        omega = torch.randn(n_feat, d, generator=gen) / lengthscale
        b = torch.rand(n_feat, generator=gen) * (2.0 * math.pi)
        self.register_buffer("omega", omega)
        self.register_buffer("bias", b)
        self.scale = math.sqrt(2.0 / n_feat)
        self.coef = torch.nn.Parameter(torch.zeros(d, n_feat))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        phi = self.scale * torch.cos(s @ self.omega.T + self.bias)  # [N, L]
        return phi @ self.coef.T


class _ObserverRFF(torch.nn.Module):
    def __init__(self, d: int, n_feat: int, lengthscale: float, gen: torch.Generator) -> None:
        super().__init__()
        omega = torch.randn(n_feat, d, generator=gen) / lengthscale
        b = torch.rand(n_feat, generator=gen) * (2.0 * math.pi)
        self.register_buffer("omega", omega)
        self.register_buffer("bias", b)
        self.scale = math.sqrt(2.0 / n_feat)
        self.coef = torch.nn.Parameter(torch.zeros(1, n_feat))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        phi = self.scale * torch.cos(s @ self.omega.T + self.bias)
        return (phi @ self.coef.T)[:, 0]


class _ComposedNet(torch.nn.Module):
    """Shared latent operation modules composed along a protocol step string."""

    def __init__(self, cfg: BayesianTerminalOnlyConfig) -> None:
        super().__init__()
        d = cfg.latent_dim
        self.embed = torch.nn.Linear(2, d)
        gen = torch.Generator().manual_seed(cfg.seed + 12345)
        if cfg.model == "bnn":
            self.ops = torch.nn.ModuleDict(
                {op: _OperationMLP(d, cfg.hidden_dim) for op in INTERMEDIATE_OPS}
            )
            self.obs = _ObserverMLP(d, cfg.hidden_dim)
        elif cfg.model == "gpn":
            self.ops = torch.nn.ModuleDict(
                {op: _OperationRFF(d, cfg.rff_features, cfg.rff_lengthscale, gen)
                 for op in INTERMEDIATE_OPS}
            )
            self.obs = _ObserverRFF(d, cfg.rff_features, cfg.rff_lengthscale, gen)
        else:
            raise ValueError(f"Unknown model '{cfg.model}'. Expected 'bnn' or 'gpn'.")
        # latent -> observed 2-D alignment for optional intermediate (trace) likelihood
        self.align = torch.nn.Linear(d, 2)

    def forward(self, x: torch.Tensor, steps: Sequence[str], obs_pos=None):
        """Terminal scalar; if ``obs_pos`` (positions into steps[:-1]) is given, also
        return {t: aligned latent} for the intermediate-observation likelihood."""
        obs_set = set(obs_pos) if obs_pos is not None else set()
        s = self.embed(x)
        captured = {}
        for t, op in enumerate(steps[:-1]):  # last step is the terminal observer
            if op in self.ops:
                s = self.ops[op](s)
            if t in obs_set:
                captured[t] = self.align(s)
        if obs_pos is None:
            return self.obs(s)
        return self.obs(s), captured


def _group(rows: Sequence[Mapping[str, str]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[str(r["steps"])].append(i)
    return groups


# --------------------------------------------------------------------------
# HMC with dual-averaging step-size adaptation.


def _split_rhat(chains: np.ndarray) -> float:
    """Split-R-hat on a 1-D per-chain scalar series ``chains`` [C, S]."""
    c, s = chains.shape
    if s < 4 or c < 2:
        return float("nan")
    half = s // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    m = split.shape[1]
    means = split.mean(axis=1)
    b = m * means.var(ddof=1)
    w = split.var(axis=1, ddof=1).mean()
    if w <= 0:
        return float("nan")
    var_hat = (m - 1) / m * w + b / m
    return float(np.sqrt(var_hat / w))


def _ess_1d(chains: np.ndarray) -> float:
    """Multi-chain effective sample size on a 1-D scalar series ``chains`` [C, S]."""
    c, s = chains.shape
    if s < 4:
        return float(c * s)
    means = chains.mean(axis=1, keepdims=True)
    var = chains.var(axis=1, ddof=1).mean()
    if var <= 0:
        return float(c * s)
    centered = chains - means
    rho = np.zeros(s)
    for t in range(s):
        acov = np.mean([np.mean(centered[k, : s - t] * centered[k, t:]) for k in range(c)])
        rho[t] = acov / var
    tau = 1.0
    for t in range(1, s - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        tau += 2.0 * pair
    return float(c * s / max(tau, 1.0))


class _HMC:
    def __init__(self, potential_and_grad, dim: int, cfg: BayesianTerminalOnlyConfig):
        self._pot = potential_and_grad
        self.dim = dim
        self.cfg = cfg

    def _leapfrog(self, theta, p, eps, U0, g0):
        p = p - 0.5 * eps * g0
        theta = theta + eps * p
        U, g = self._pot(theta)
        for _ in range(self.cfg.n_leapfrog - 1):
            p = p - eps * g
            theta = theta + eps * p
            U, g = self._pot(theta)
        p = p - 0.5 * eps * g
        return theta, p, U, g

    def run_chain(self, theta0: torch.Tensor, rng: torch.Generator, probe):
        cfg = self.cfg
        theta = theta0.clone()
        U, g = self._pot(theta)
        # dual-averaging state
        eps = self._find_reasonable_eps(theta, U, g, rng)
        mu = math.log(10 * eps)
        log_eps_bar = 0.0
        h_bar = 0.0
        gamma, t0, kappa = 0.05, 10.0, 0.75
        samples, probes = [], []
        total = cfg.num_warmup + cfg.num_samples * cfg.thin
        for it in range(total):
            p0 = torch.randn(self.dim, generator=rng)
            H0 = U + 0.5 * (p0 @ p0)
            theta_new, p_new, U_new, g_new = self._leapfrog(theta, p0, eps, U, g)
            if torch.isfinite(U_new):
                H_new = U_new + 0.5 * (p_new @ p_new)
                log_alpha = float(H0 - H_new)
            else:
                log_alpha = -float("inf")
            accept_prob = min(1.0, math.exp(min(0.0, log_alpha)))
            if math.log(torch.rand(1, generator=rng).item() + 1e-38) < log_alpha:
                theta, U, g = theta_new, U_new, g_new
            if it < cfg.num_warmup:  # dual averaging toward target_accept
                m1 = it + 1
                h_bar = (1 - 1 / (m1 + t0)) * h_bar + (cfg.target_accept - accept_prob) / (m1 + t0)
                log_eps = mu - math.sqrt(m1) / gamma * h_bar
                eta = m1 ** (-kappa)
                log_eps_bar = eta * log_eps + (1 - eta) * log_eps_bar
                eps = math.exp(log_eps)
            else:
                eps = math.exp(log_eps_bar)
                if (it - cfg.num_warmup) % cfg.thin == 0:
                    samples.append(theta.clone())
                    probes.append(probe(theta))
        return samples, np.array(probes)

    def _find_reasonable_eps(self, theta, U, g, rng):
        eps = 0.1
        p = torch.randn(self.dim, generator=rng)
        H0 = U + 0.5 * (p @ p)
        theta1, p1, U1, _ = self._leapfrog(theta, p, eps, U, g)
        if not torch.isfinite(U1):
            return 1e-3
        H1 = U1 + 0.5 * (p1 @ p1)
        a = 1.0 if float(H0 - H1) > math.log(0.5) else -1.0
        for _ in range(20):
            eps *= 2.0 ** a
            theta1, p1, U1, _ = self._leapfrog(theta, p, eps, U, g)
            if not torch.isfinite(U1):
                eps *= 2.0 ** (-a)
                break
            H1 = U1 + 0.5 * (p1 @ p1)
            if a * (float(H0 - H1) - math.log(0.5)) <= 0:
                break
        return max(min(eps, 1.0), 1e-4)


@dataclasses.dataclass
class BayesianTerminalOnlyOFML:
    """Terminal-only OFML with a full HMC posterior over shared operation params.

    ``config.model="bnn"`` -> compositional Bayesian neural network.
    ``config.model="gpn"`` -> compositional random-feature Gaussian-process network.
    Predictive variance is the pushforward spread of the posterior.
    """

    config: BayesianTerminalOnlyConfig = dataclasses.field(default_factory=BayesianTerminalOnlyConfig)
    samples_: list | None = None
    diagnostics_: dict | None = None
    y_mean_: float = 0.0
    y_std_: float = 1.0

    def _build(self):
        net = _ComposedNet(self.config)
        names = [n for n, _ in net.named_parameters()]
        shapes = [tuple(p.shape) for _, p in net.named_parameters()]
        numels = [int(np.prod(s)) for s in shapes]
        return net, names, shapes, numels

    def fit(self, rows: Sequence[Mapping[str, str]], trace=None) -> "BayesianTerminalOnlyOFML":
        cfg = self.config
        torch.manual_seed(cfg.seed)
        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        y = np.array([float(r["y"]) for r in rows], dtype=float)
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0

        n = len(rows)
        use_trace = trace is not None and any(len(trace[i]) > 0 for i in range(n))
        if use_trace:
            allt = np.array([v for i in range(n) for (_, v) in trace[i]], dtype=float)
            self.trace_mean_ = allt.mean(0); self.trace_std_ = np.where(allt.std(0) < 1e-6, 1.0, allt.std(0))
        else:
            self.trace_mean_, self.trace_std_ = np.zeros(2), np.ones(2)

        batches = []  # (steps_list, X_tensor, Y_tensor_standardized, obs_pos, tb)
        for steps, idx in _group(rows).items():
            xb = torch.as_tensor(x[idx], dtype=torch.float32)
            yb = torch.as_tensor((y[idx] - self.y_mean_) / self.y_std_, dtype=torch.float32)
            obs_pos, tb = [], None
            if use_trace and len(trace[idx[0]]) > 0:
                obs_pos = [t for (t, _) in trace[idx[0]]]
                tb = {}
                for pos_k, t in enumerate(obs_pos):
                    tgt = np.array([trace[i][pos_k][1] for i in idx], dtype=float)
                    tgt = (tgt - self.trace_mean_) / self.trace_std_
                    tb[t] = torch.as_tensor(tgt, dtype=torch.float32)
            batches.append((steps.split(), xb, yb, obs_pos, tb))

        net, names, shapes, numels = self._build()
        buffers = {k: v for k, v in net.named_buffers()}
        dim = int(sum(numels))
        prior_var = cfg.prior_std ** 2
        obs_var = cfg.sigma_obs ** 2

        def unflatten(theta: torch.Tensor) -> dict:
            out, off = {}, 0
            for name, shape, n in zip(names, shapes, numels):
                out[name] = theta[off:off + n].reshape(shape)
                off += n
            return out

        trace_var = obs_var / max(cfg.trace_weight, 1e-8)   # smaller var => stronger trace pull
        def potential_and_grad(theta: torch.Tensor):
            th = theta.detach().clone().requires_grad_(True)
            params = unflatten(th)
            merged = {**params, **buffers}
            nll = th.new_zeros(())
            for steps, xb, yb, obs_pos, tb in batches:
                if tb:
                    pred, captured = functional_call(net, merged, (xb, steps, obs_pos))
                    nll = nll + 0.5 * torch.sum((pred - yb) ** 2) / obs_var
                    for t in obs_pos:
                        nll = nll + 0.5 * torch.sum((captured[t] - tb[t]) ** 2) / trace_var
                else:
                    pred = functional_call(net, merged, (xb, steps))
                    nll = nll + 0.5 * torch.sum((pred - yb) ** 2) / obs_var
            logprior = 0.5 * torch.sum(th ** 2) / prior_var
            U = nll + logprior
            (grad,) = torch.autograd.grad(U, th)
            return U.detach(), grad.detach()

        # function-space probe: mean prediction over a fixed grid on the first group
        probe_steps, probe_x = batches[0][0], batches[0][1]

        def probe(theta: torch.Tensor) -> float:
            with torch.no_grad():
                merged = {**unflatten(theta), **buffers}
                return float(functional_call(net, merged, (probe_x, probe_steps)).mean())

        hmc = _HMC(potential_and_grad, dim, cfg)
        all_samples: list[torch.Tensor] = []
        probe_chains = []
        for c in range(cfg.num_chains):
            gen = torch.Generator().manual_seed(cfg.seed * 1000 + c)
            theta0 = torch.randn(dim, generator=gen) * cfg.prior_std
            samples, probes = hmc.run_chain(theta0, gen, probe)
            all_samples.extend(samples)
            probe_chains.append(probes)

        self.samples_ = all_samples
        self._net = net
        self._buffers = buffers
        self._unflatten = unflatten
        pc = np.array(probe_chains)  # [C, S]
        self.diagnostics_ = {
            "rhat": _split_rhat(pc),
            "ess": _ess_1d(pc),
            "n_samples": len(all_samples),
            "dim": dim,
        }
        return self

    def predict_distribution(self, rows: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        if not self.samples_:
            raise ValueError("BayesianTerminalOnlyOFML must be fit before predict.")
        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        mean = np.zeros(len(rows)); var = np.zeros(len(rows))
        with torch.no_grad():
            for steps, idx in _group(rows).items():
                xb = torch.as_tensor(x[idx], dtype=torch.float32)
                step_list = steps.split()
                preds = []
                for theta in self.samples_:
                    merged = {**self._unflatten(theta), **self._buffers}
                    preds.append(functional_call(self._net, merged, (xb, step_list)).numpy())
                p = np.stack(preds, axis=0)  # [S, n]
                m = p.mean(0) * self.y_std_ + self.y_mean_
                v = p.var(0) * (self.y_std_ ** 2)
                for k, i in enumerate(idx):
                    mean[i] = m[k]; var[i] = v[k]
        return mean, var

    def predict(self, rows: Sequence[Mapping[str, str]]) -> np.ndarray:
        return self.predict_distribution(rows)[0]
