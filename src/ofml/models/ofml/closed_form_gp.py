"""Closed-form GP operation model (OFML-GPN).

A calibrated, closed-form (pure NumPy/SciPy) Gaussian-process operation model:
one closed-form ARD-RBF GP per output coordinate, with input/target
standardization, a noise floor, and jitter escalation in the Cholesky. Composed
by ``OFMLModel`` this yields the **OFML-GPN** model.

Motivation (vs OFML-DKL): the DKL operation model becomes overconfident on the
extrapolated operation states that protocol recomposition induces (feature
collapse), which the uncertainty-calibration diagnostic identifies as the reason
variance-based active learning stalls. A stationary-kernel closed-form GP keeps
posterior variance growing away from the data and needs no gradient-descent MLL
loop, so it is both calibrated and numerically stable. This module has NO torch
dependency, so OFML-GPN runs in a numpy-only environment.

Ported from the standalone gp_network package's closed-form GP.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from .base import OperationExample

try:
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

_JITTERS = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
_2PI = 2.0 * np.pi
_2PI2 = 2.0 * np.pi**2


def _rbf(a: np.ndarray, b: np.ndarray, lengthscales: np.ndarray, signal_var: float) -> np.ndarray:
    a_s = a / lengthscales
    b_s = b / lengthscales
    sq = (
        np.sum(a_s**2, axis=1)[:, None]
        + np.sum(b_s**2, axis=1)[None, :]
        - 2.0 * a_s @ b_s.T
    )
    np.maximum(sq, 0.0, out=sq)
    return signal_var * np.exp(-0.5 * sq)


def _sm(a: np.ndarray, b: np.ndarray, w: np.ndarray, v: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Spectral-mixture kernel (Wilson & Adams). ``w`` [Q], ``v`` [Q,D], ``mu`` [Q,D]."""

    tau = a[:, None, :] - b[None, :, :]
    tau2 = tau**2
    out = np.zeros((a.shape[0], b.shape[0]))
    for q in range(w.shape[0]):
        expo = np.exp(-_2PI2 * np.sum(tau2 * v[q], axis=2))
        cosp = np.prod(np.cos(_2PI * tau * mu[q]), axis=2)
        out += w[q] * expo * cosp
    return out


class _RBFKernel:
    """ARD-RBF: lengthscales (D) + signal variance (1)."""

    def __init__(self, d: int):
        self.d = d
        self.n_params = d + 1

    def init_theta(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, d = x.shape
        if n > 1:
            diffs = x[:, None, :] - x[None, :, :]
            sq = np.sum(diffs**2, axis=2)
            median_sq = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
            ell0 = np.sqrt(max(median_sq, 1e-6) / max(d, 1))
        else:
            ell0 = 1.0
        return np.concatenate([np.log(np.full(d, ell0)), [np.log(np.var(y) + 1e-6)]])

    def bounds(self) -> list:
        return [(np.log(1e-2), np.log(1e2))] * self.d + [(np.log(1e-4), np.log(1e2))]

    def matrix(self, a: np.ndarray, b: np.ndarray, kt: np.ndarray) -> np.ndarray:
        return _rbf(a, b, np.exp(kt[: self.d]), float(np.exp(kt[self.d])))

    def kxx(self, kt: np.ndarray) -> float:
        return float(np.exp(kt[self.d]))

    def grads(self, x: np.ndarray, kt: np.ndarray) -> list:
        ell = np.exp(kt[: self.d]); sf2 = float(np.exp(kt[self.d]))
        k = _rbf(x, x, ell, sf2)
        gl = []
        for dd in range(self.d):
            tau_d = x[:, dd][:, None] - x[:, dd][None, :]
            gl.append(k * (tau_d**2 / ell[dd] ** 2))
        gl.append(k.copy())
        return gl


class _SMKernel:
    """Spectral-mixture: w [Q], v [Q,D], mu [Q,D]; component 0 inits at freq 0 (RBF)."""

    def __init__(self, d: int, q: int = 4):
        self.d = d
        self.q = q
        self.n_params = q + q * d + q * d

    def _unpack(self, kt: np.ndarray):
        q, d = self.q, self.d
        w = np.exp(kt[:q])
        v = np.exp(kt[q : q + q * d]).reshape(q, d)
        mu = kt[q + q * d :].reshape(q, d)
        return w, v, mu

    def init_theta(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, d = x.shape
        if n > 1:
            diffs = x[:, None, :] - x[None, :, :]
            sq = np.sum(diffs**2, axis=2)
            median_sq = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
            ell0 = np.sqrt(max(median_sq, 1e-6) / max(d, 1))
        else:
            ell0 = 1.0
        sf2 = float(np.var(y) + 1e-6)
        v0 = 1.0 / (4.0 * np.pi**2 * ell0**2)
        w = np.log(np.full(self.q, sf2 / self.q))
        v = np.log(np.full((self.q, d), v0)).reshape(-1)
        f0 = 0.5 / max(ell0, 1e-3)
        mu = np.array([[min(qi, self.q) * f0 for _ in range(d)] for qi in range(self.q)], dtype=float)
        mu[0, :] = 0.0
        return np.concatenate([w, v, mu.reshape(-1)])

    def bounds(self) -> list:
        q, d = self.q, self.d
        return (
            [(np.log(1e-6), np.log(1e2))] * q
            + [(np.log(1e-6), np.log(1e4))] * (q * d)
            + [(0.0, 10.0)] * (q * d)
        )

    def matrix(self, a: np.ndarray, b: np.ndarray, kt: np.ndarray) -> np.ndarray:
        w, v, mu = self._unpack(kt)
        return _sm(a, b, w, v, mu)

    def kxx(self, kt: np.ndarray) -> float:
        return float(np.sum(np.exp(kt[: self.q])))

    def grads(self, x: np.ndarray, kt: np.ndarray) -> list:
        w, v, mu = self._unpack(kt)
        tau = x[:, None, :] - x[None, :, :]
        tau2 = tau**2
        gw, gv, gmu = [], [], []
        for qi in range(self.q):
            expo = np.exp(-_2PI2 * np.sum(tau2 * v[qi], axis=2))
            cos_d = np.cos(_2PI * tau * mu[qi])
            cprod = np.prod(cos_d, axis=2)
            term = w[qi] * expo * cprod
            gw.append(term)
            for dd in range(self.d):
                gv.append(term * (-_2PI2 * tau2[:, :, dd]) * v[qi, dd])
            for dd in range(self.d):
                other = cprod / np.where(np.abs(cos_d[:, :, dd]) < 1e-12, 1e-12, cos_d[:, :, dd])
                dmu = -_2PI * tau[:, :, dd] * np.sin(_2PI * tau[:, :, dd] * mu[qi, dd])
                gmu.append(w[qi] * expo * other * dmu)
        return gw + gv + gmu


def _cholesky(matrix: np.ndarray):
    n = matrix.shape[0]
    last_error = None
    for jitter in _JITTERS:
        try:
            reg = matrix + jitter * np.eye(n)
            if _HAVE_SCIPY:
                return cho_factor(reg, lower=True)
            return (np.linalg.cholesky(reg), True)
        except Exception as error:  # pragma: no cover
            last_error = error
    raise np.linalg.LinAlgError(f"Cholesky failed even with max jitter: {last_error}")


def _cho_solve(factor, rhs: np.ndarray) -> np.ndarray:
    if _HAVE_SCIPY:
        return cho_solve(factor, rhs)
    chol, _ = factor
    return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))


class _ClosedFormGP:
    """Single-output closed-form GP with ARD-RBF or Spectral-Mixture kernel."""

    def __init__(self, optimize_hypers: bool = True, noise_floor: float = 1e-4, max_train: int = 400,
                 seed: int = 0, kernel: str = "rbf", sm_q: int = 4):
        self.optimize_hypers = optimize_hypers
        self.noise_floor = noise_floor
        self.max_train = max_train
        self.seed = seed
        self.kernel = kernel
        self.sm_q = sm_q

    def _make_kernel(self, d: int):
        if self.kernel == "rbf":
            return _RBFKernel(d)
        if self.kernel == "sm":
            return _SMKernel(d, self.sm_q)
        raise ValueError(f"Unknown kernel '{self.kernel}'. Expected 'rbf' or 'sm'.")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_ClosedFormGP":
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.shape[0] > self.max_train:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(x.shape[0], size=self.max_train, replace=False)
            x, y = x[idx], y[idx]
        self._x_mean = x.mean(axis=0)
        self._x_std = x.std(axis=0)
        self._x_std[self._x_std < 1e-9] = 1.0
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        self._x = (x - self._x_mean) / self._x_std
        self._y = (y - self._y_mean) / self._y_std
        n, d = self._x.shape
        self._kernel = self._make_kernel(d)
        theta0 = np.concatenate([self._kernel.init_theta(self._x, self._y), [np.log(max(0.01, self.noise_floor))]])
        if self.optimize_hypers and _HAVE_SCIPY and n >= 4:
            bounds = self._kernel.bounds() + [(np.log(self.noise_floor), np.log(1.0))]
            try:
                res = minimize(self._nll_and_grad, theta0, method="L-BFGS-B", jac=True, bounds=bounds,
                               options={"maxiter": 40 if self.kernel == "sm" else 60})
                theta = res.x if np.all(np.isfinite(res.x)) else theta0
            except Exception:
                theta = theta0
        else:
            theta = theta0
        self._set_theta(theta)
        self._factorise()
        return self

    def _set_theta(self, theta: np.ndarray) -> None:
        self._ktheta = theta[:-1]
        self._noise = float(max(np.exp(theta[-1]), self.noise_floor))

    def _nll(self, theta: np.ndarray) -> float:
        kt = theta[:-1]
        noise = float(max(np.exp(theta[-1]), self.noise_floor))
        n = self._x.shape[0]
        k = self._kernel.matrix(self._x, self._x, kt) + noise * np.eye(n)
        try:
            factor = _cholesky(k)
        except np.linalg.LinAlgError:
            return 1e18
        alpha = _cho_solve(factor, self._y)
        chol = factor[0]
        log_det = 2.0 * np.sum(np.log(np.abs(np.diag(chol))))
        return float(0.5 * self._y @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi))

    def _nll_and_grad(self, theta: np.ndarray):
        """Marginal NLL + analytic gradient (jac=True) -- ~20x faster SM fits."""
        kt = theta[:-1]
        noise = float(max(np.exp(theta[-1]), self.noise_floor))
        n = self._x.shape[0]
        k = self._kernel.matrix(self._x, self._x, kt) + noise * np.eye(n)
        try:
            factor = _cholesky(k)
        except np.linalg.LinAlgError:
            return 1e18, np.zeros_like(theta)
        alpha = _cho_solve(factor, self._y)
        chol = factor[0]
        log_det = 2.0 * np.sum(np.log(np.abs(np.diag(chol))))
        nll = float(0.5 * self._y @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi))
        k_inv = _cho_solve(factor, np.eye(n))
        outer = k_inv - np.outer(alpha, alpha)
        dks = self._kernel.grads(self._x, kt)
        dks.append(noise * np.eye(n))
        grad = np.array([0.5 * float(np.sum(outer * dk)) for dk in dks])
        return nll, grad

    def _factorise(self) -> None:
        n = self._x.shape[0]
        k = self._kernel.matrix(self._x, self._x, self._ktheta) + self._noise * np.eye(n)
        self._factor = _cholesky(k)
        self._alpha = _cho_solve(self._factor, self._y)

    def predict(self, x_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = (np.atleast_2d(np.asarray(x_star, dtype=float)) - self._x_mean) / self._x_std
        ks = self._kernel.matrix(xs, self._x, self._ktheta)
        mean = ks @ self._alpha
        v = _cho_solve(self._factor, ks.T)
        var = self._kernel.kxx(self._ktheta) - np.sum(ks.T * v, axis=0)
        np.maximum(var, 0.0, out=var)
        return mean * self._y_std + self._y_mean, var * (self._y_std**2)

    @property
    def noise_orig(self) -> float:
        """Observation-noise variance in the original (unstandardized) units."""
        return self._noise * (self._y_std**2)

    def posterior_cov(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Posterior covariance cov_n(a, b) in original output units."""
        a = np.atleast_2d(np.asarray(a, dtype=float))
        b = np.atleast_2d(np.asarray(b, dtype=float))
        a_s = (a - self._x_mean) / self._x_std
        b_s = (b - self._x_mean) / self._x_std
        k_ab = self._kernel.matrix(a_s, b_s, self._ktheta)
        ks_a = self._kernel.matrix(a_s, self._x, self._ktheta)
        ks_b = self._kernel.matrix(b_s, self._x, self._ktheta)
        cov = k_ab - ks_a @ _cho_solve(self._factor, ks_b.T)
        return cov * (self._y_std**2)

    def alc_scores(self, refs: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Exact rank-1 ALC (Cohn) variance reduction at ``refs`` per candidate
        (fantasy-free): sum_e cov_n(e,c)^2 / (var_n(c) + noise)."""
        candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
        refs = np.atleast_2d(np.asarray(refs, dtype=float))
        cov_ec = self.posterior_cov(refs, candidates)  # [R, C]
        _, var_c = self.predict(candidates)  # [C]
        denom = np.maximum(var_c + self.noise_orig, 1e-12)
        return np.sum(cov_ec**2, axis=0) / denom


@dataclasses.dataclass
class ClosedFormGPOperationModel:
    """OFML-GPN operation model: one closed-form GP per output coordinate."""

    optimize_hypers: bool = True
    noise_floor: float = 1e-4
    seed: int = 0
    kernel: str = "rbf"
    sm_q: int = 4
    max_train: int = 400
    gps_: list | None = None
    output_dim_: int | None = None

    def fit(self, examples: Sequence[OperationExample]) -> "ClosedFormGPOperationModel":
        if not examples:
            raise ValueError("Cannot fit ClosedFormGPOperationModel with no examples.")
        output_dim = int(examples[0]["output_dim"])
        x = np.array([[float(e["input_1"]), float(e["input_2"])] for e in examples], dtype=float)
        # Optional per-protocol context descriptor (see models/ofml/context.py):
        # ContextOFMLModel attaches ``_ctx`` (a list of floats) to each example so
        # the shared GP is fit in the same augmented input space the composition
        # feeds at predict time. ARD ignores context dims for ops that do not use them.
        if examples[0].get("_ctx"):
            ctx = np.array([list(map(float, e["_ctx"])) for e in examples], dtype=float)
            if ctx.ndim == 2 and ctx.shape[1] > 0:
                x = np.concatenate([x, ctx], axis=1)
        y = np.array([[float(e[f"output_{j}"]) for j in range(1, output_dim + 1)] for e in examples], dtype=float)
        self.gps_ = []
        for j in range(output_dim):
            gp = _ClosedFormGP(
                optimize_hypers=self.optimize_hypers, noise_floor=self.noise_floor,
                seed=self.seed + j, kernel=self.kernel, sm_q=self.sm_q, max_train=self.max_train,
            )
            gp.fit(x, y[:, j])
            self.gps_.append(gp)
        self.output_dim_ = output_dim
        return self

    def predict_state(self, states: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_distribution(states)
        return mean

    def predict_distribution(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.gps_ is None:
            raise ValueError("ClosedFormGPOperationModel must be fit before predict.")
        states = np.atleast_2d(np.asarray(states, dtype=float))
        means, variances = [], []
        for gp in self.gps_:
            m, v = gp.predict(states)
            means.append(m)
            variances.append(v)
        return np.stack(means, axis=1), np.stack(variances, axis=1)
