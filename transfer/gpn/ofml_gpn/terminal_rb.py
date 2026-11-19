"""gpn_smc.terminal_rb -- build_terminal_rb_functions, moved verbatim from
u15_protocol_backend.py, per K2_DESIGN.md K2a.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import LOG2PI
from .numerics import EXTENDED_QUANTILE_LEVELS


def build_terminal_rb_functions(
    n_candidates: int,
    n_max: int,
    amplitude: float,
    lengthscale: float,
    noise_variance: float,
    jitter: float,
    chunk_size: int | None = None,
    loglik_chunk_size: int | None = None,
):
    """Return ``(loglik_from_h, moments_from_h, summary_extended_from_h)``.

    Each is a pure function of ``(h_last, obs_indices, y_padded, active[,
    truth_values])`` where ``h_last`` has shape ``(P, n_candidates)``.
    ``loglik_from_h`` and ``moments_from_h`` have no ``axis_name``
    dependency (safe to call directly, single- or multi-device).
    ``summary_extended_from_h`` uses ``jax.lax.pmin/pmax/psum`` with
    ``axis_name="device"`` for the global 42-step bisection exactly as
    ``gpn_deep_backend.DeepGpnGpuBackend``'s does, so it must be called
    from inside a ``pmap(..., axis_name="device")`` (a single-device pmap
    is a valid, if trivial, way to satisfy that when testing standalone).

    ``chunk_size`` sizes the ``jax.lax.map`` chunking of
    ``_conditional_mean_variance``'s ``(P, n_candidates, n_max)``-shaped
    cross-covariance/whitened-cross tensors (used by both
    ``moments_from_h`` and ``summary_extended_from_h``) -- the same
    mandatory pattern ``pushforward_shared_h``/``gpn_deep_backend.pushforward_h``
    already apply to their own ``(P, d, d)`` tensors. At U15's joint scale
    (``n_candidates=256``, ``n_max=40``) these buffers are ~8x larger per
    particle than anything U11/U14 exercised (``n_candidates<=64``,
    ``n_max<=20``, never chunked there because the un-chunked footprint
    was small enough): ~5 GiB per buffer at 65,536 local particles with
    multiple such buffers live at once, an OOM risk on a 16 GiB V100.
    ``chunk_size=None`` (or ``>= local particle count``) disables chunking
    (single call, matching the previous unchunked behaviour) -- callers
    should size it via ``compute_chunk_size`` as the other tensors here do.
    """
    sf2_sq = float(amplitude) ** 2
    ell2 = float(lengthscale)
    noise_var = float(noise_variance)
    jitter_const = float(jitter)
    eye_max = jnp.eye(n_max, dtype=jnp.float64)
    levels = jnp.asarray(EXTENDED_QUANTILE_LEVELS, dtype=jnp.float64)
    n_levels = levels.shape[0]
    rb_chunk_size = int(chunk_size) if chunk_size is not None else None
    # M10f: ``loglik_from_h`` only ever holds ``(P, n_max, n_max)`` tensors
    # (never the ``(P, n_candidates, n_max)`` cross-covariance the moments
    # path builds), so it gets its OWN chunk size sized on ``n_max`` alone.
    # Reusing ``chunk_size`` (sized on ``max(n_candidates, n_max)``=1024 at
    # U15's joint scale) here gave 32-particle chunks -- 512 sequential tiny
    # Cholesky launches per call, which made every pCN block update ~40 ms
    # slower than the unchunked pre-M10f call. ``None`` = unchunked.
    ll_chunk_size = int(loglik_chunk_size) if loglik_chunk_size is not None else None

    def covariance_from_difference(difference: jax.Array) -> jax.Array:
        return sf2_sq * jnp.exp(-0.5 * (difference / ell2) ** 2)

    def _loglik_from_h_chunk(
        h_chunk: jax.Array, obs_indices: jax.Array, y_padded: jax.Array, active: jax.Array
    ) -> jax.Array:
        z_obs = h_chunk[:, obs_indices]
        difference = z_obs[:, :, None] - z_obs[:, None, :]
        covariance = covariance_from_difference(difference)
        pair_mask = active[None, :, None] * active[None, None, :]
        covariance = covariance * pair_mask
        diagonal = (noise_var + jitter_const) * active + (1.0 - active)
        covariance = covariance + eye_max[None, :, :] * diagonal[None, :, None]
        cholesky = jnp.linalg.cholesky(covariance)
        rhs = jnp.broadcast_to(y_padded[None, :, None], (h_chunk.shape[0], n_max, 1))
        solved = jax.scipy.linalg.solve_triangular(cholesky, rhs, lower=True)[..., 0]
        quadratic = jnp.sum(solved * solved, axis=1)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(cholesky, axis1=1, axis2=2)), axis=1)
        return -0.5 * (quadratic + logdet + jnp.sum(active) * LOG2PI)

    def loglik_from_h(
        h_last: jax.Array, obs_indices: jax.Array, y_padded: jax.Array, active: jax.Array
    ) -> jax.Array:
        """Chunked over particles via ``jax.lax.map`` (M10f additive -- same
        mandatory pattern ``_conditional_mean_variance`` below already
        applies to its own ``(P, n_max, n_max)`` tensors): at ``n_max=100``
        the unchunked ``(P, 100, 100)`` covariance/Cholesky here is large
        enough to OOM a 16 GiB V100 at full particle count, whereas the
        pre-M10f ``n_max<=40`` footprint never needed chunking.
        ``loglik_chunk_size=None`` (or ``>=`` the particle count) disables
        chunking, reproducing the pre-M10f single-call behaviour exactly.
        """
        total_particles = h_last.shape[0]

        def chunk_fn(h_chunk: jax.Array) -> jax.Array:
            return _loglik_from_h_chunk(h_chunk, obs_indices, y_padded, active)

        if ll_chunk_size is None or ll_chunk_size >= total_particles:
            return chunk_fn(h_last)
        n_chunks = total_particles // ll_chunk_size
        chunks = h_last.reshape(n_chunks, ll_chunk_size, h_last.shape[-1])
        loglik_chunks = jax.lax.map(chunk_fn, chunks)
        return loglik_chunks.reshape(total_particles)

    def _conditional_mean_variance_chunk(
        h_chunk: jax.Array, obs_indices: jax.Array, y_padded: jax.Array, active: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        z_obs = h_chunk[:, obs_indices]
        difference_oo = z_obs[:, :, None] - z_obs[:, None, :]
        covariance = covariance_from_difference(difference_oo)
        pair_mask = active[None, :, None] * active[None, None, :]
        covariance = covariance * pair_mask
        diagonal = (noise_var + jitter_const) * active + (1.0 - active)
        covariance = covariance + eye_max[None, :, :] * diagonal[None, :, None]
        cholesky = jnp.linalg.cholesky(covariance)

        difference_xo = h_chunk[:, :, None] - z_obs[:, None, :]
        cross_covariance = covariance_from_difference(difference_xo)
        cross_covariance = cross_covariance * active[None, None, :]
        rhs = jnp.broadcast_to(y_padded[None, :, None], (h_chunk.shape[0], n_max, 1))
        intermediate = jax.scipy.linalg.solve_triangular(cholesky, rhs, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(
            jnp.swapaxes(cholesky, 1, 2), intermediate, lower=False
        )[..., 0]
        conditional_mean = jnp.einsum("pmn,pn->pm", cross_covariance, alpha)
        whitened_cross = jax.scipy.linalg.solve_triangular(
            cholesky, jnp.swapaxes(cross_covariance, 1, 2), lower=True
        )
        conditional_variance = jnp.maximum(
            sf2_sq - jnp.sum(whitened_cross * whitened_cross, axis=1), 0.0
        )
        return conditional_mean, conditional_variance

    def _conditional_mean_variance(
        h_last: jax.Array, obs_indices: jax.Array, y_padded: jax.Array, active: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Chunked over particles via ``jax.lax.map`` (mandatory pattern for
        any ``(P, d, d)``-or-larger tensor here -- see ``build_terminal_rb_
        functions``'s ``chunk_size`` docstring paragraph). ``cholesky`` is
        NOT returned: both callers below (``moments_from_h``,
        ``summary_extended_from_h``) already discarded it, so keeping it
        out of the chunked/reassembled outputs avoids stacking a needless
        ``(P, n_max, n_max)`` tensor across chunks.
        """
        total_particles = h_last.shape[0]

        def chunk_fn(h_chunk: jax.Array) -> tuple[jax.Array, jax.Array]:
            return _conditional_mean_variance_chunk(h_chunk, obs_indices, y_padded, active)

        if rb_chunk_size is None or rb_chunk_size >= total_particles:
            return chunk_fn(h_last)
        n_chunks = total_particles // rb_chunk_size
        chunks = h_last.reshape(n_chunks, rb_chunk_size, h_last.shape[-1])
        mean_chunks, variance_chunks = jax.lax.map(chunk_fn, chunks)
        conditional_mean = mean_chunks.reshape(total_particles, n_candidates)
        conditional_variance = variance_chunks.reshape(total_particles, n_candidates)
        return conditional_mean, conditional_variance

    def moments_from_h(
        h_last: jax.Array,
        local_weights: jax.Array,
        obs_indices: jax.Array,
        y_padded: jax.Array,
        active: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        conditional_mean, conditional_variance = _conditional_mean_variance(
            h_last, obs_indices, y_padded, active
        )
        weighted_mean = jnp.sum(local_weights[:, None] * conditional_mean, axis=0)
        weighted_second = jnp.sum(
            local_weights[:, None] * (conditional_variance + conditional_mean * conditional_mean),
            axis=0,
        )
        weighted_within = jnp.sum(local_weights[:, None] * conditional_variance, axis=0)
        return weighted_mean, weighted_second, weighted_within

    # 2026-11-01q additive hook (quantile acquisition): expose the per-particle conditional moments.
    moments_from_h.conditional_mean_variance = _conditional_mean_variance

    def summary_extended_from_h(
        h_last: jax.Array,
        local_weights: jax.Array,
        obs_indices: jax.Array,
        y_padded: jax.Array,
        active: jax.Array,
        truth_values: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        conditional_mean, conditional_variance = _conditional_mean_variance(
            h_last, obs_indices, y_padded, active
        )
        conditional_sd = jnp.sqrt(jnp.maximum(conditional_variance, 1e-15))

        weighted_mean = jnp.sum(local_weights[:, None] * conditional_mean, axis=0)
        weighted_second = jnp.sum(
            local_weights[:, None] * (conditional_variance + conditional_mean * conditional_mean),
            axis=0,
        )
        weighted_within = jnp.sum(local_weights[:, None] * conditional_variance, axis=0)

        positive = local_weights[:, None] > 0.0
        lower_bound = jax.lax.pmin(
            jnp.min(
                jnp.where(positive, conditional_mean - 12.0 * conditional_sd, jnp.inf), axis=0
            ),
            axis_name="device",
        )
        upper_bound = jax.lax.pmax(
            jnp.max(
                jnp.where(positive, conditional_mean + 12.0 * conditional_sd, -jnp.inf), axis=0
            ),
            axis_name="device",
        )

        def mixture_cdf_single(threshold: jax.Array) -> jax.Array:
            standardized = (threshold[None, :] - conditional_mean) / conditional_sd
            local_cdf = jnp.sum(
                local_weights[:, None] * jax.scipy.special.ndtr(standardized), axis=0
            )
            return jax.lax.psum(local_cdf, axis_name="device")

        def mixture_cdf_batched(threshold: jax.Array) -> jax.Array:
            standardized = (
                threshold[:, None, :] - conditional_mean[None, :, :]
            ) / conditional_sd[None, :, :]
            local_cdf = jnp.sum(
                local_weights[None, :, None] * jax.scipy.special.ndtr(standardized), axis=1
            )
            return jax.lax.psum(local_cdf, axis_name="device")

        def mixture_quantiles_all_levels() -> jax.Array:
            low0 = jnp.broadcast_to(lower_bound[None, :], (n_levels, lower_bound.shape[0]))
            high0 = jnp.broadcast_to(upper_bound[None, :], (n_levels, upper_bound.shape[0]))

            def bisect_body(_, bounds):
                low, high = bounds
                midpoint = 0.5 * (low + high)
                global_cdf = mixture_cdf_batched(midpoint)
                below = global_cdf < levels[:, None]
                return (
                    jnp.where(below, midpoint, low),
                    jnp.where(below, high, midpoint),
                )

            low, high = jax.lax.fori_loop(0, 42, bisect_body, (low0, high0))
            return 0.5 * (low + high)

        quantiles = mixture_quantiles_all_levels()
        cdf_at_quantiles = mixture_cdf_batched(quantiles)
        pit = mixture_cdf_single(truth_values)
        return (
            weighted_mean,
            weighted_second,
            weighted_within,
            quantiles,
            cdf_at_quantiles,
            pit,
        )

    return loglik_from_h, moments_from_h, summary_extended_from_h


def build_loglik_from_hobs(
    n_max: int,
    amplitude: float,
    lengthscale: float,
    noise_variance: float,
    jitter: float,
    loglik_chunk_size: int | None = None,
):
    """S2 additive sibling of ``build_terminal_rb_functions``' ``loglik_from_h``.

    Returns ``loglik_from_hobs(z_obs, y_padded, active)`` where ``z_obs`` is the
    ALREADY-GATHERED ``(P, n_max)`` array ``h_last[:, obs_indices]``. The
    covariance / mask / Cholesky / quadratic-form code is a line-for-line copy
    of ``_loglik_from_h_chunk`` minus the gather, so padding (``active``) is
    handled identically. Chunked over particles with ``jax.lax.map`` exactly as
    ``loglik_from_h`` is (``loglik_chunk_size=None`` or ``>= P`` = unchunked).
    """
    sf2_sq = float(amplitude) ** 2
    ell2 = float(lengthscale)
    noise_var = float(noise_variance)
    jitter_const = float(jitter)
    eye_max = jnp.eye(n_max, dtype=jnp.float64)
    ll_chunk_size = int(loglik_chunk_size) if loglik_chunk_size is not None else None

    def _chunk(z_obs: jax.Array, y_padded: jax.Array, active: jax.Array) -> jax.Array:
        difference = z_obs[:, :, None] - z_obs[:, None, :]
        covariance = sf2_sq * jnp.exp(-0.5 * (difference / ell2) ** 2)
        pair_mask = active[None, :, None] * active[None, None, :]
        covariance = covariance * pair_mask
        diagonal = (noise_var + jitter_const) * active + (1.0 - active)
        covariance = covariance + eye_max[None, :, :] * diagonal[None, :, None]
        cholesky = jnp.linalg.cholesky(covariance)
        rhs = jnp.broadcast_to(y_padded[None, :, None], (z_obs.shape[0], n_max, 1))
        solved = jax.scipy.linalg.solve_triangular(cholesky, rhs, lower=True)[..., 0]
        quadratic = jnp.sum(solved * solved, axis=1)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(cholesky, axis1=1, axis2=2)), axis=1)
        return -0.5 * (quadratic + logdet + jnp.sum(active) * LOG2PI)

    def loglik_from_hobs(z_obs: jax.Array, y_padded: jax.Array, active: jax.Array) -> jax.Array:
        total_particles = z_obs.shape[0]
        if ll_chunk_size is None or ll_chunk_size >= total_particles:
            return _chunk(z_obs, y_padded, active)
        n_chunks = total_particles // ll_chunk_size
        chunks = z_obs.reshape(n_chunks, ll_chunk_size, z_obs.shape[-1])
        out = jax.lax.map(lambda c: _chunk(c, y_padded, active), chunks)
        return out.reshape(total_particles)

    return loglik_from_hobs


# ---------------------------------------------------------------------------
# S6-LR additive: fixed-grid Nystrom (DTC) low-rank form of the terminal GP.
# The terminal GP has ONE scalar input per observation (the penultimate value
# h), an SE kernel and fixed amplitude/lengthscale/noise, so the n x n kernel
# matrix has numerical rank 20-35 at n=500 whatever n is.  With m inducing
# points u on [-H, H] and the eigen-truncated factor C = V diag(w)^(-1/2) of
# K_uu (top r eigenpairs), K_xx ~ B B^T with B = K_xu C (n x r).  Woodbury:
#   y^T (B B^T + s2 I)^-1 y = (y^T y - c^T c) / s2,   c = L^-1 B^T y,
#   L L^T = s2 I + B^T B,
#   logdet(B B^T + s2 I) = (n - r) log s2 + 2 sum log diag L.
# Numerical rule (measured): form K_xu * mask, multiply by C FIRST, then form
# the r x r Gram B^T B (the order C^T (K_xu^T K_xu) C loses 0.145 nats).
# ---------------------------------------------------------------------------
LOWRANK_H_MIN = 4.5
LOWRANK_H_MAX = 8.0
LOWRANK_H_MARGIN = 1.1
LOWRANK_EIG_REL_TOL = 1e-13


def lowrank_rank(m: int, amplitude: float, lengthscale: float, h: float = LOWRANK_H_MIN) -> int:
    """Number of eigenvalues of K_uu (m points on [-h, h]) above 1e-13 of the largest."""
    import numpy as _np

    u = _np.linspace(-h, h, int(m))
    kuu = float(amplitude) ** 2 * _np.exp(-0.5 * ((u[:, None] - u[None, :]) / float(lengthscale)) ** 2)
    w = _np.linalg.eigvalsh(kuu)
    return int(_np.sum(w > LOWRANK_EIG_REL_TOL * w.max()))


class LowRankTerminal:
    """Pure-jnp helpers of the Nystrom terminal likelihood; ``m`` inducing points,
    rank ``r`` fixed at construction (value at H = LOWRANK_H_MIN)."""

    def __init__(self, m, amplitude, lengthscale, noise_variance, jitter):
        self.m = int(m)
        self.sf2 = float(amplitude) ** 2
        self.ell = float(lengthscale)
        self.s2 = float(noise_variance) + float(jitter)
        self.r = lowrank_rank(self.m, amplitude, lengthscale)

    def grid(self, h_half):
        """(u (m,), C (m, r)) for the grid [-h_half, h_half] (traceable)."""
        u = h_half * jnp.linspace(-1.0, 1.0, self.m, dtype=jnp.float64)
        kuu = self.sf2 * jnp.exp(-0.5 * ((u[:, None] - u[None, :]) / self.ell) ** 2)
        w, v = jnp.linalg.eigh(kuu)  # ascending
        w_r = w[self.m - self.r :]
        v_r = v[:, self.m - self.r :]
        return u, v_r / jnp.sqrt(w_r)[None, :]

    def features(self, h, u, c_mat, mask=None):
        """B = (K(h, u) * mask) @ C, shape h.shape + (r,); mask applied BEFORE the C product."""
        kxu = self.sf2 * jnp.exp(-0.5 * ((h[..., None] - u) / self.ell) ** 2)
        if mask is not None:
            kxu = kxu * mask[..., None]
        return kxu @ c_mat

    def contrib(self, h_t, valid_t, y_t, u, c_mat):
        """Per-type Gram contributions. h_t (c, T, cap), valid_t/y_t (T, cap)
        -> G (c, T, r, r), b (c, T, r)."""
        b_mat = self.features(h_t, u, c_mat, valid_t[None])
        g = jnp.einsum("ptci,ptcj->ptij", b_mat, b_mat)
        b = jnp.einsum("ptci,tc->pti", b_mat, y_t)
        return g, b

    def loglik_from_gram(self, g, b, yty, n_active):
        """g (c, r, r), b (c, r) -> loglik (c,) of the same observations as the exact GP."""
        r = self.r
        m_mat = g + self.s2 * jnp.eye(r, dtype=jnp.float64)[None]
        chol = jnp.linalg.cholesky(m_mat)
        c = jax.scipy.linalg.solve_triangular(chol, b[..., None], lower=True)[..., 0]
        quad = (yty - jnp.sum(c * c, axis=1)) / self.s2
        logdet = (n_active - r) * jnp.log(self.s2) + 2.0 * jnp.sum(
            jnp.log(jnp.diagonal(chol, axis1=1, axis2=2)), axis=1
        )
        return -0.5 * (quad + logdet + n_active * LOG2PI)


def build_lowrank_moments(
    lr: LowRankTerminal, n_candidates: int, n_max: int, chunk_size: int | None = None
):
    """Low-rank counterpart of ``build_terminal_rb_functions``' ``moments_from_h``.

    Returns ``moments_lr(h_last, local_weights, obs_indices, y_padded, active)`` ->
    ``(weighted_mean, weighted_second, weighted_within)`` over the candidates.  The
    grid half-width is chosen inside the call from the largest |h| over ALL particles
    and candidates (pmax over axis_name "device": call from a pmap with that axis).
    Posterior predictive of the Nystrom GP:
      mean = B_* M^-1 b,   var = sf2 - |B_*|^2 + s2 |L^-1 B_*^T|^2  (>= 0),
    with M = s2 I + B^T B = L L^T.
    """
    sf2 = lr.sf2
    s2 = lr.s2
    rb_chunk = int(chunk_size) if chunk_size is not None else None

    def _chunk(h_chunk, obs_indices, y_padded, active, u, c_mat):
        z_obs = h_chunk[:, obs_indices]
        b_obs = lr.features(z_obs, u, c_mat, active[None, :])
        g = jnp.einsum("pni,pnj->pij", b_obs, b_obs)
        b = jnp.einsum("pni,n->pi", b_obs, y_padded * active)
        chol = jnp.linalg.cholesky(g + s2 * jnp.eye(lr.r, dtype=jnp.float64)[None])
        alpha = jax.scipy.linalg.cho_solve((chol, True), b[..., None])[..., 0]
        b_star = lr.features(h_chunk, u, c_mat)  # (p, n_cand, r)
        mean = jnp.einsum("pmi,pi->pm", b_star, alpha)
        z = jax.scipy.linalg.solve_triangular(chol, jnp.swapaxes(b_star, 1, 2), lower=True)
        var = sf2 - jnp.sum(b_star * b_star, axis=2) + s2 * jnp.sum(z * z, axis=1)
        return mean, jnp.maximum(var, 0.0)

    def conditional_mean_variance(h_last, obs_indices, y_padded, active):
        hmax = jax.lax.pmax(jnp.max(jnp.abs(h_last)), axis_name="device")
        half = jnp.clip(LOWRANK_H_MARGIN * hmax, LOWRANK_H_MIN, LOWRANK_H_MAX)
        u, c_mat = lr.grid(half)
        total = h_last.shape[0]
        if rb_chunk is None or rb_chunk >= total:
            return _chunk(h_last, obs_indices, y_padded, active, u, c_mat)
        nc = total // rb_chunk
        chunks = h_last.reshape(nc, rb_chunk, h_last.shape[-1])
        mean_c, var_c = jax.lax.map(lambda c: _chunk(c, obs_indices, y_padded, active, u, c_mat), chunks)
        return mean_c.reshape(total, n_candidates), var_c.reshape(total, n_candidates)

    def moments_lr(h_last, local_weights, obs_indices, y_padded, active):
        cm, cv = conditional_mean_variance(h_last, obs_indices, y_padded, active)
        weighted_mean = jnp.sum(local_weights[:, None] * cm, axis=0)
        weighted_second = jnp.sum(local_weights[:, None] * (cv + cm * cm), axis=0)
        weighted_within = jnp.sum(local_weights[:, None] * cv, axis=0)
        return weighted_mean, weighted_second, weighted_within

    moments_lr.conditional_mean_variance = conditional_mean_variance
    return moments_lr
