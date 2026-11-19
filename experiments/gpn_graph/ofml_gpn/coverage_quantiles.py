"""gpn_smc.coverage_quantiles -- posterior quantiles of f for coverage tests (BENCH9 phase 2, 2026-11-02).

NEW module; nothing in the existing code imports it unless ``--save-quantiles-rounds`` is given.

Definition (identical to ``terminal_rb.build_terminal_rb_functions.summary_extended_from_h``, the
implementation behind the earlier U15 calibration/coverage work, ``u15_protocol_bandits`` +
``u13_aggregate.compute_coverage``): the posterior predictive of f(x) is the particle mixture
``sum_i w_i N(m_i(x), s_i(x)^2)`` with ``(m_i, s_i^2)`` the Rao-Blackwellised exact terminal-GP
conditional mean / variance of particle i (the exact terminal, never the low-rank one); its quantile at
level ``q`` is found by 42 bisection steps of the mixture CDF on the bracket
``[min_i(m_i - 12 s_i), max_i(m_i + 12 s_i)]`` (over positive-weight particles, global over devices), with
``s_i^2`` floored at 1e-15.  Levels = ``EXTENDED_QUANTILE_LEVELS`` = (.005, .025, .05, .10, .25, .75, .90,
.95, .975, .995), i.e. the 99/95/90/80/50 % central bands.

Differences from the original function (numerically none): the per-particle ``(m, s^2)`` are computed once
(stage 1) and the bisection runs over blocks of candidates (stage 2), so the (levels x particles x
candidates) work arrays stay small for 8192-candidate libraries.  The independent arm uses each protocol's
own particle system (its own weights).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

from .numerics import EXTENDED_QUANTILE_LEVELS

COVERAGE_LEVELS = tuple(float(v) for v in EXTENDED_QUANTILE_LEVELS)
BAND_LEVELS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95), 95: (0.025, 0.975), 99: (0.005, 0.995)}
VARIANCE_FLOOR = 1e-15
BISECTION_STEPS = 42
_WORK_BYTES = 400e6


def _block_width(n_levels: int, p_local: int, n_cand: int) -> int:
    best = None
    w = 64
    while w <= n_cand:
        if n_cand % w == 0 and n_levels * p_local * w * 8 <= _WORK_BYTES:
            best = w
        w += 64
    if best is None:
        best = 64 if n_cand % 64 == 0 else n_cand
    return best


def _make_stage2(devices):
    levels = jnp.asarray(COVERAGE_LEVELS, dtype=jnp.float64)

    def stage2(cm, cv, w):
        sd = jnp.sqrt(jnp.maximum(cv, VARIANCE_FLOOR))
        positive = w[:, None] > 0.0
        lower = jax.lax.pmin(jnp.min(jnp.where(positive, cm - 12.0 * sd, jnp.inf), axis=0), axis_name="device")
        upper = jax.lax.pmax(jnp.max(jnp.where(positive, cm + 12.0 * sd, -jnp.inf), axis=0), axis_name="device")

        def cdf_batched(threshold):
            standardized = (threshold[:, None, :] - cm[None, :, :]) / sd[None, :, :]
            local = jnp.sum(w[None, :, None] * ndtr(standardized), axis=1)
            return jax.lax.psum(local, axis_name="device")

        n_levels = levels.shape[0]
        low0 = jnp.broadcast_to(lower[None, :], (n_levels, lower.shape[0]))
        high0 = jnp.broadcast_to(upper[None, :], (n_levels, upper.shape[0]))

        def body(_, bounds):
            low, high = bounds
            mid = 0.5 * (low + high)
            below = cdf_batched(mid) < levels[:, None]
            return jnp.where(below, mid, low), jnp.where(below, high, mid)

        low, high = jax.lax.fori_loop(0, BISECTION_STEPS, body, (low0, high0))
        q = 0.5 * (low + high)
        err = jnp.max(jnp.abs(cdf_batched(q) - levels[:, None]), axis=0)
        return q, err

    return jax.pmap(stage2, axis_name="device", devices=devices)


def _stage2_blocks(backend: Any, cm: jax.Array, cv: jax.Array, weights_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cm, cv: (n_dev, P_loc, C) device arrays; weights_local (n_dev, P_loc). Returns ((L, C), (C,))."""
    cache = backend.__dict__.setdefault("_cov_quantile_cache", {})
    if "stage2" not in cache:
        cache["stage2"] = _make_stage2(backend.devices)
    stage2 = cache["stage2"]
    n_dev, p_loc, c = cm.shape
    bw = _block_width(len(COVERAGE_LEVELS), p_loc, c)
    w = jnp.asarray(weights_local, dtype=jnp.float64)
    qs, errs = [], []
    for a in range(0, c, bw):
        q, e = stage2(cm[:, :, a:a + bw], cv[:, :, a:a + bw], w)
        qs.append(np.asarray(q)[0])
        errs.append(np.asarray(e)[0])
    return np.concatenate(qs, axis=1), np.concatenate(errs)


# ----------------------------------------------------------------------------- shared-model backends
def shared_quantiles(backend: Any, latent: jax.Array, weights: np.ndarray, obs_indices: np.ndarray,
                     y_padded: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantiles ``(L, n_candidates)`` and max-CDF-error ``(n_candidates,)`` of a shared-model backend
    (``SharedProtocolGpnBackend`` or ``GraphSharedGpnBackend``)."""
    n_max = int(y_padded.shape[0])
    bundle = backend._bundle(n_max)
    cache = backend.__dict__.setdefault("_cov_quantile_cache", {})
    key = ("stage1", n_max)
    if key not in cache:
        cmv = bundle["moments_from_h"].conditional_mean_variance
        push = backend._pushforward

        def stage1(state, obs, y, act):
            return cmv(push(state), obs, y, act)

        cache[key] = jax.pmap(stage1, in_axes=(0, None, None, None), devices=backend.devices)
    cm, cv = cache[key](latent, obs_indices, y_padded, active)
    cm.block_until_ready()
    wl = np.asarray(weights, dtype=np.float64).reshape(backend.n_devices, backend.local_particles)
    return _stage2_blocks(backend, cm, cv, wl)


# ----------------------------------------------------------------------------- factorised independent backends
def indep_protocol_quantiles(backend: Any, p: int, latent_p: jax.Array, weights_p: np.ndarray,
                             obs_idx: np.ndarray, y: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantiles ``(L, G)`` of protocol p (its own particle system) for ``FactorisedIndependentGpnBackend`` or
    ``GraphFactorisedIndependentGpnBackend``."""
    n_max = int(y.shape[0])
    cache = backend.__dict__.setdefault("_cov_quantile_cache", {})
    graph = hasattr(backend, "_gbundle")
    if graph:
        topo = backend._topology_of_protocol[p]
        bundle = backend._gbundle(topo, n_max)
        key = ("stage1", n_max, topo)
    else:
        bundle = backend._bundle(n_max)
        key = ("stage1", n_max)
    if key not in cache:
        cmv = bundle["moments_from_h"].conditional_mean_variance
        push = bundle["push"]
        if graph:
            def stage1(state, obs, yy, act):
                return cmv(push(state), obs, yy, act)

            cache[key] = jax.pmap(stage1, in_axes=(0, None, None, None), devices=backend.devices)
        else:
            def stage1(state, obs, yy, act, chol):
                return cmv(push(state, chol), obs, yy, act)

            cache[key] = jax.pmap(stage1, in_axes=(0, None, None, None, None), devices=backend.devices)
    if graph:
        cm, cv = cache[key](latent_p, obs_idx, y, active)
    else:
        cm, cv = cache[key](latent_p, obs_idx, y, active, backend._protocol_layer1_cholesky[p])
    cm.block_until_ready()
    wl = np.asarray(weights_p, dtype=np.float64).reshape(backend.n_devices, backend.local_particles)
    return _stage2_blocks(backend, cm, cv, wl)


def indep_quantiles(backend: Any, state: Any, observed_flat: list[int], noisy_truth_flat: np.ndarray,
                    *, observed_y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Quantiles ``(L, Q*G)`` over all protocols (protocol-major) for a ``FactorisedState``."""
    qs, errs = [], []
    for p in range(backend.n_protocols):
        idx, y, act, _ = backend.protocol_observations(p, observed_flat, noisy_truth_flat, observed_y=observed_y)
        q, e = indep_protocol_quantiles(backend, p, state.latents[p], state.weights(p), idx, y, act)
        qs.append(q)
        errs.append(e)
    return np.concatenate(qs, axis=1), np.concatenate(errs)
