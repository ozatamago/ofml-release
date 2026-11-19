"""gpn_smc.quantile_acq -- posterior-quantile acquisition for the U15 GPN arms.

NEW module (2026-11-01q); nothing in the existing code imports it.

The posterior predictive of f(x) is the particle mixture
    sum_i w_i N(m_i(x), s_i(x)^2)
with (m_i, s_i^2) the per-particle conditional mean / variance of the terminal GP.
The acquisition is its upper a-quantile
    U(x; a) = the u with  T(u) := sum_i w_i * Phibar((u - m_i(x)) / s_i(x)) = a,   Phibar = 1 - Phi,
with level a set by one of two schedules (n = len(observed) at selection time):
    "matched":  a = Phibar(c_n),  c_n = sqrt(log(n + 2))  (the c_n of ``numerics.c_schedule``);
                if all particles agree, U = m + c_n s, i.e. the current UCB;
    "bayes":    a = Phibar(z_scale * z_t),  z_t = Phibar^{-1}(6 delta / (pi^2 N t^2)),  t = n + 1,
                N = number of candidates (finite-domain Bayesian GP-UCB level, Srinivas et al. 2010
                Thm 1 analogue; z_scale = 1/sqrt(5) is their Sec. 6 "beta / 5" convention).

Root finding (float64).  Bracket [lo, hi] with
    lo = min_i (m_i + z s_i),  hi = max_i (m_i + z s_i),   z = Phibar^{-1}(a),
over particles with positive weight (global over devices via pmin/pmax): at lo every particle has
Phibar >= a, at hi every particle has Phibar <= a, so T(lo) >= a >= T(hi) and the root is inside.
Optionally ``f32_steps`` float32 bisection steps first (bracket re-checked in float64, see the
kernel docstring).  Then ``steps`` bisection steps, then ``newton`` safeguarded Newton steps (a Newton iterate that leaves
the current bracket is replaced by the bracket midpoint; the bracket is updated with the sign of
T(u) - a at every iterate).  Sums over particles use psum over axis "device", so the pure kernels
must run inside ``pmap(..., axis_name="device")``.  s_i is floored at sqrt(1e-15) (same floor as
``terminal_rb.summary_extended_from_h``).

Public API
    c_matched(n), alpha_matched(n), z_bayes(n, N, delta), alpha_bayes(n, N, delta, z_scale),
    level_for(n, schedule, N, delta, z_scale)
    mixture_moments_and_quantile(cm, cv, w, level, steps, newton, f32_steps=)   (pure, in-pmap)
    SharedQuantileAcq(backend, steps, newton, f32_steps)       .posterior_moments_and_quantile(...)
    FactorisedQuantileAcq(backend, steps, newton, f32_steps)   .posterior_moments_and_quantile(...)
    get_quantile_acq(backend, steps, newton, f32_steps)        builder cached on the backend object

Two pmaps per call so the two costs can be timed separately: stage 1 computes the per-particle
(cm, cv) on the device (pushforward + terminal GP: the existing ``moments_from_h`` /
low-rank ``conditional_mean_variance`` functions, unchanged); stage 2 computes the weighted moments
and the quantile from (cm, cv).  ``last_seconds`` = {"moments": stage 1, "quantile": stage 2}.
"""

from __future__ import annotations

import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri

DEFAULT_STEPS = 16  # 2026-11-01 GPU benchmarks: 16 bisection + 2 Newton steps (see QUANTILE_ACQ_DEV_2026-11-01.md)
DEFAULT_NEWTON = 2
DEFAULT_F32_STEPS = 0
VARIANCE_FLOOR = 1e-15  # same floor as terminal_rb.summary_extended_from_h
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- level schedule
def c_matched(n_observations: int) -> float:
    """c_n = sqrt(log(n + 2)), identical to ``numerics.c_schedule``."""
    return float(math.sqrt(math.log(n_observations + 2.0)))


def alpha_matched(n_observations: int) -> float:
    """a_n = Phibar(c_n) = 0.5 erfc(c_n / sqrt 2)."""
    return float(0.5 * math.erfc(c_matched(n_observations) / math.sqrt(2.0)))


def z_bayes(n_observations: int, n_candidates: int, delta: float = 0.1) -> float:
    """z_t = Phibar^{-1}(6 delta / (pi^2 N t^2)), t = n + 1 (finite-domain Bayesian GP-UCB level,
    Srinivas et al. 2010 Thm 1 analogue; the initial design counts as rounds)."""
    from scipy.special import ndtri as _ndtri

    t = float(n_observations) + 1.0
    p = 6.0 * float(delta) / (math.pi**2 * float(n_candidates) * t * t)
    return float(-_ndtri(p))


def alpha_bayes(n_observations: int, n_candidates: int, delta: float = 0.1, z_scale: float = 1.0) -> float:
    """a = Phibar(z_scale * z_t).  z_scale = 1: theoretical level; z_scale = 1/sqrt(5): the
    "beta / 5" convention of Srinivas et al. 2010 Sec. 6."""
    z = float(z_scale) * z_bayes(n_observations, n_candidates, delta)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


LEVEL_SCHEDULES = ("matched", "bayes")


def level_for(
    n_observations: int, schedule: str = "matched", n_candidates: int | None = None,
    delta: float = 0.1, z_scale: float = 1.0,
) -> float:
    """Level a used at selection time with ``n_observations = len(observed)``."""
    if schedule == "matched":
        return alpha_matched(n_observations)
    if schedule == "bayes":
        if n_candidates is None:
            raise ValueError("schedule 'bayes' needs n_candidates")
        return alpha_bayes(n_observations, n_candidates, delta, z_scale)
    raise ValueError(f"schedule={schedule!r} must be one of {LEVEL_SCHEDULES}")


# --------------------------------------------------------------------------- pure kernel
def mixture_moments_and_quantile(
    cm: jax.Array,
    cv: jax.Array,
    w: jax.Array,
    level,
    steps: int = DEFAULT_STEPS,
    newton: int = DEFAULT_NEWTON,
    axis_name: str | None = "device",
    f32_steps: int = 0,
):
    """Weighted moments and upper ``level``-quantile of the particle mixture.

    cm, cv: (P_local, *B, K) per-particle conditional mean / variance; w: (P_local, *B) local
    weights (one weight vector per batch index b, e.g. per protocol).  Particle sums run over
    axis 0.  Returns ``(weighted_mean, weighted_second, weighted_within, quantile)``, each (*B, K):
    the first three are this device's PARTIAL sums (exactly what ``terminal_rb.moments_from_h``
    returns; the host sums them over devices), the quantile is global (identical on every device).
    ``axis_name=None`` = single-device use outside pmap.

    ``f32_steps > 0``: first that many bisection steps in float32 on offsets (cm - lo) relative to
    the initial bracket (one float32 tail pass costs ~1/4 of a float64 one on a V100); the
    resulting bracket, widened by 16 of its final widths on each side, is then checked in float64
    (T(lo) >= level, T(hi) <= level) and an end that fails the check is reset to its initial
    value, so the float64 bracket always holds the root; then ``steps`` float64 bisection steps
    and ``newton`` Newton steps as above."""

    def psum(x):
        return x if axis_name is None else jax.lax.psum(x, axis_name=axis_name)

    def pmin(x):
        return x if axis_name is None else jax.lax.pmin(x, axis_name=axis_name)

    def pmax(x):
        return x if axis_name is None else jax.lax.pmax(x, axis_name=axis_name)

    wk = w[..., None]
    weighted_mean = jnp.sum(wk * cm, axis=0)
    weighted_second = jnp.sum(wk * (cv + cm * cm), axis=0)
    weighted_within = jnp.sum(wk * cv, axis=0)

    sd = jnp.sqrt(jnp.maximum(cv, VARIANCE_FLOOR))
    inv_sd = 1.0 / sd
    level = jnp.asarray(level, dtype=jnp.float64)
    z = -ndtri(level)  # Phibar(z) = level
    score = cm + z * sd
    positive = wk > 0.0
    lo = pmin(jnp.min(jnp.where(positive, score, jnp.inf), axis=0))
    hi = pmax(jnp.max(jnp.where(positive, score, -jnp.inf), axis=0))

    def tail(u):
        # Phibar((u - m)/s) = Phi((m - u)/s)
        return psum(jnp.sum(wk * ndtr((cm - u[None]) * inv_sd), axis=0))

    def bisect_body(_, bounds):
        a_lo, a_hi = bounds
        mid = 0.5 * (a_lo + a_hi)
        above = tail(mid) > level  # T is decreasing: the root is right of mid
        return jnp.where(above, mid, a_lo), jnp.where(above, a_hi, mid)

    if f32_steps > 0:
        f32 = jnp.float32
        d32 = (cm - lo[None]).astype(f32)
        isd32 = inv_sd.astype(f32)
        w32 = wk.astype(f32)
        level32 = level.astype(f32)

        def bisect32_body(_, bounds):
            a_lo, a_hi = bounds
            mid = 0.5 * (a_lo + a_hi)
            t = psum(jnp.sum(w32 * ndtr((d32 - mid[None]) * isd32), axis=0))
            above = t > level32
            return jnp.where(above, mid, a_lo), jnp.where(above, a_hi, mid)

        o_lo, o_hi = jax.lax.fori_loop(
            0, int(f32_steps), bisect32_body, (jnp.zeros_like(lo, dtype=f32), (hi - lo).astype(f32))
        )
        # A float32 decision is wrong when the midpoint lies within the float32 error of the tail
        # (relative ~1e-6, i.e. ~1e-7 s in u) of the root, so the float32 bracket can miss the
        # root by that much: widen it by 16 final widths (= W 2^(4 - f32_steps)) on each side
        # before the float64 check.
        margin = 16.0 * (o_hi - o_lo).astype(jnp.float64)
        lo32 = jnp.maximum(lo + o_lo.astype(jnp.float64) - margin, lo)
        hi32 = jnp.minimum(lo + o_hi.astype(jnp.float64) + margin, hi)
        lo = jnp.where(tail(lo32) >= level, lo32, lo)
        hi = jnp.where(tail(hi32) <= level, hi32, hi)

    if steps > 0:
        lo, hi = jax.lax.fori_loop(0, int(steps), bisect_body, (lo, hi))
    u = 0.5 * (lo + hi)

    if newton > 0:

        def newton_body(_, carry):
            u, a_lo, a_hi = carry
            t = (cm - u[None]) * inv_sd
            f = psum(jnp.sum(wk * ndtr(t), axis=0)) - level
            dens = psum(jnp.sum(wk * jnp.exp(-0.5 * t * t) * inv_sd, axis=0)) * _INV_SQRT_2PI
            # f > 0: root right of u.  f'(u) = -dens.
            a_lo = jnp.where(f > 0, u, a_lo)
            a_hi = jnp.where(f > 0, a_hi, u)
            a_lo = jnp.where(f == 0, u, a_lo)
            a_hi = jnp.where(f == 0, u, a_hi)
            step = jnp.where(dens > 0, f / jnp.where(dens > 0, dens, 1.0), jnp.inf)
            cand = u + step
            ok = (cand >= a_lo) & (cand <= a_hi) & jnp.isfinite(cand)
            u_new = jnp.where(ok, cand, 0.5 * (a_lo + a_hi))
            return u_new, a_lo, a_hi

        u, lo, hi = jax.lax.fori_loop(0, int(newton), newton_body, (u, lo, hi))

    return weighted_mean, weighted_second, weighted_within, u


def _host_moments(out) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Device outputs of the stage-2 pmap -> numpy (mean, variance, within, quantile), the same
    host reduction as ``SharedProtocolGpnBackend.posterior_moments``."""
    mean_sum, second_sum, within_sum = [np.sum(np.asarray(v), axis=0) for v in out[:3]]
    variance = np.maximum(second_sum - mean_sum * mean_sum, 0.0)
    return mean_sum, variance, within_sum, np.asarray(out[3])[0]


def _make_stage2_pmap(devices, steps: int, newton: int, n_batch: int | None = None, f32_steps: int = 0):
    """Stage-2 pmap.  ``n_batch=None``: inputs cm, cv (P_local, K), w (P_local,).
    ``n_batch=Q``: inputs are tuples of Q arrays (P_local, G) / (P_local,), stacked on axis 1."""
    if n_batch is None:

        def local(cm, cv, w, level):
            return mixture_moments_and_quantile(cm, cv, w, level, steps, newton, f32_steps=f32_steps)

    else:

        def local(cms, cvs, ws, level):
            cm = jnp.stack(cms, axis=1)
            cv = jnp.stack(cvs, axis=1)
            w = jnp.stack(ws, axis=1)
            return mixture_moments_and_quantile(cm, cv, w, level, steps, newton, f32_steps=f32_steps)

    return jax.pmap(local, in_axes=(0, 0, 0, None), axis_name="device", devices=devices)


# --------------------------------------------------------------------------- shared backend
class SharedQuantileAcq:
    """Quantile acquisition for ``SharedProtocolGpnBackend`` (exact or S6-LR low-rank moments,
    same dispatch rule as ``posterior_moments``: low rank iff ``backend.lowrank_moments and not
    exact``).  Stage-1 pmaps are built lazily per (n_max bucket, low-rank flag)."""

    def __init__(
        self, backend: Any, steps: int = DEFAULT_STEPS, newton: int = DEFAULT_NEWTON,
        f32_steps: int = DEFAULT_F32_STEPS,
    ) -> None:
        self.backend = backend
        self.steps = int(steps)
        self.newton = int(newton)
        self.f32_steps = int(f32_steps)
        self._stage1: dict[tuple[int, bool], Any] = {}
        self._stage2 = None
        self.last_seconds = {"moments": 0.0, "quantile": 0.0}

    def _stage1_pmap(self, n_max: int, use_lowrank: bool):
        key = (n_max, use_lowrank)
        if key not in self._stage1:
            b = self.backend
            bundle = b._bundle(n_max)
            fn = bundle["moments_lr_from_h"] if use_lowrank else bundle["moments_from_h"]
            cmv = fn.conditional_mean_variance
            push = b._pushforward

            def local(state, obs_indices, y_padded, active):
                return cmv(push(state), obs_indices, y_padded, active)

            self._stage1[key] = jax.pmap(
                local, in_axes=(0, None, None, None), axis_name="device", devices=b.devices
            )
        return self._stage1[key]

    def particle_moments(self, latent, obs_indices, y_padded, active, exact: bool = False):
        """Device arrays ``(cm, cv)``, each ``(n_devices, local_particles, n_candidates)``."""
        use_lowrank = bool(self.backend.lowrank_moments and not exact)
        f = self._stage1_pmap(int(y_padded.shape[0]), use_lowrank)
        cm, cv = f(latent, obs_indices, y_padded, active)
        cv.block_until_ready()
        return cm, cv

    def posterior_moments_and_quantile(
        self, latent, weights, obs_indices, y_padded, active, level, exact: bool = False
    ):
        """numpy ``(mean, variance, within, quantile)``, each ``(n_candidates,)``; the first three
        follow ``SharedProtocolGpnBackend.posterior_moments`` exactly."""
        b = self.backend
        t0 = time.perf_counter()
        cm, cv = self.particle_moments(latent, obs_indices, y_padded, active, exact)
        t1 = time.perf_counter()
        if self._stage2 is None:
            self._stage2 = _make_stage2_pmap(b.devices, self.steps, self.newton, f32_steps=self.f32_steps)
        local_weights = np.asarray(weights, dtype=np.float64).reshape(b.n_devices, b.local_particles)
        out = self._stage2(cm, cv, local_weights, np.float64(level))
        host = _host_moments(out)
        t2 = time.perf_counter()
        self.last_seconds = {"moments": t1 - t0, "quantile": t2 - t1}
        return host

    __call__ = posterior_moments_and_quantile


# --------------------------------------------------------------------------- factorised backend
class FactorisedQuantileAcq:
    """Quantile acquisition for ``FactorisedIndependentGpnBackend`` / ``FactorisedState``.

    Per protocol p the per-particle (cm, cv) device arrays and the device copy of the weights are
    cached on the state object (attribute ``_quantile_cache``).  An entry is recomputed iff the
    particle array ``state.latents[p]`` or ``state.logweights[p]`` is a different object, or p's
    own observation triple (index, y, active) changed.  ``assimilate_observation`` replaces both
    arrays' objects and appends to p's triple whenever it touches p, i.e. this covers the
    invalidation of the existing ``state._moments_cache[p] = None`` (and is checked by value, not
    by that flag).  The quantile is recomputed for ALL protocols every call, in ONE stage-2 pmap
    over the stacked (P_local, Q, G) arrays, since the level changes every round."""

    def __init__(
        self, backend: Any, steps: int = DEFAULT_STEPS, newton: int = DEFAULT_NEWTON,
        f32_steps: int = DEFAULT_F32_STEPS,
    ) -> None:
        self.backend = backend
        self.steps = int(steps)
        self.newton = int(newton)
        self.f32_steps = int(f32_steps)
        self._stage1: dict[int, Any] = {}
        self._stage2 = None
        self.last_seconds = {"moments": 0.0, "quantile": 0.0}
        self.last_recomputed: list[int] = []

    def _stage1_pmap(self, n_max: int):
        if n_max not in self._stage1:
            b = self.backend
            bundle = b._bundle(n_max)
            cmv = bundle["moments_from_h"].conditional_mean_variance
            push = bundle["push"]

            def local(state, obs_idx, y, active, chol):
                return cmv(push(state, chol), obs_idx, y, active)

            self._stage1[n_max] = jax.pmap(
                local, in_axes=(0, None, None, None, None), axis_name="device", devices=b.devices
            )
        return self._stage1[n_max]

    def particle_moments(self, p, latent_p, idx, y, act):
        """Device arrays ``(cm, cv)`` of protocol p, each ``(n_devices, local_particles, G)``."""
        f = self._stage1_pmap(int(y.shape[0]))
        cm, cv = f(latent_p, idx, y, act, self.backend._protocol_layer1_cholesky[p])
        cv.block_until_ready()
        return cm, cv

    def posterior_moments_and_quantile(
        self, state, observed_flat, noisy_truth_flat, level, *, observed_y=None
    ):
        """numpy ``(mean, variance, within, quantile)``, each ``(Q*G,)`` protocol-major, the first
        three following ``state_posterior_moments``."""
        b = self.backend
        Q = b.n_protocols
        cache = state.__dict__.setdefault("_quantile_cache", [None] * Q)
        if self._stage2 is None:
            self._stage2 = _make_stage2_pmap(b.devices, self.steps, self.newton, n_batch=Q, f32_steps=self.f32_steps)
        t0 = time.perf_counter()
        recomputed: list[int] = []
        for p in range(Q):
            idx, y, act, _ = b.protocol_observations(
                p, observed_flat, noisy_truth_flat, observed_y=observed_y
            )
            e = cache[p]
            if (
                e is None
                or e["latent"] is not state.latents[p]
                or e["logw"] is not state.logweights[p]
                or not (
                    np.array_equal(e["idx"], idx)
                    and np.array_equal(e["y"], y)
                    and np.array_equal(e["act"], act)
                )
            ):
                cm, cv = self.particle_moments(p, state.latents[p], idx, y, act)
                w_dev = jax.device_put(
                    np.asarray(state.weights(p), dtype=np.float64).reshape(b.n_devices, b.local_particles),
                    b.particle_sharding,
                )
                # the entry holds references to the arrays, so `is` cannot be fooled by id reuse
                cache[p] = {
                    "latent": state.latents[p], "logw": state.logweights[p],
                    "idx": idx, "y": y, "act": act, "cm": cm, "cv": cv, "w": w_dev,
                }
                recomputed.append(p)
        t1 = time.perf_counter()
        out = self._stage2(
            tuple(cache[p]["cm"] for p in range(Q)),
            tuple(cache[p]["cv"] for p in range(Q)),
            tuple(cache[p]["w"] for p in range(Q)),
            np.float64(level),
        )
        m, v, w, q = _host_moments(out)  # each (Q, G)
        t2 = time.perf_counter()
        self.last_seconds = {"moments": t1 - t0, "quantile": t2 - t1}
        self.last_recomputed = recomputed
        return m.reshape(-1), v.reshape(-1), w.reshape(-1), q.reshape(-1)

    __call__ = posterior_moments_and_quantile


# --------------------------------------------------------------------------- factory
def get_quantile_acq(
    backend: Any, steps: int = DEFAULT_STEPS, newton: int = DEFAULT_NEWTON, f32_steps: int = DEFAULT_F32_STEPS
):
    """Quantile-acquisition object for ``backend``, built once per (steps, newton, f32_steps) and cached
    on the backend object (its lazily compiled pmaps survive across replicates)."""
    cache = backend.__dict__.setdefault("_quantile_acq_cache", {})
    key = (int(steps), int(newton), int(f32_steps))
    if key not in cache:
        name = type(backend).__name__
        if name == "SharedProtocolGpnBackend":
            cache[key] = SharedQuantileAcq(backend, *key)
        elif name == "FactorisedIndependentGpnBackend":
            cache[key] = FactorisedQuantileAcq(backend, *key)
        else:
            raise NotImplementedError(f"quantile acquisition is not implemented for {name}")
    return cache[key]
