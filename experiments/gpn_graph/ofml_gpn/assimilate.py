"""gpn_smc.assimilate -- padded_observations_u15, assimilate_u15,
assimilate_batches_u15, moved verbatim from u15_protocol_backend.py, per
K2_DESIGN.md K2a. Bodies are character-identical to the source except these
import lines and this module docstring.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import numpy as np

from .constants import N_MAX_U15
from .numerics import derive_stream_seed, normalize_logweights, systematic_resample


def padded_observations_u15(
    observed_flat: list[int], noisy_truth_flat: np.ndarray, n_max: int = N_MAX_U15,
    *, observed_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K2c additive ``n_max`` parameter (default ``N_MAX_U15``=40,
    byte-identical to every pre-K2c caller, which never passes this):
    the padded-array capacity. Before K2c, ``len(observed_flat) > n_max``
    fell through to a numpy broadcast error (``could not broadcast input
    array from shape (n,) into shape (40,)``) at the ``indices[:n] = ...``
    assignment below -- a correct but opaque failure. This now raises a
    clear ``ValueError`` up front instead.

    2026-10-31 additive keyword ``observed_y`` (default ``None`` = old
    behaviour, ``values[:n] = noisy_truth_flat[observed_flat]``): when given,
    the y of the i-th observation is ``observed_y[i]`` (needed once an index
    may be observed more than once with a fresh noise draw per visit, where
    ``noisy_truth_flat[index]`` no longer determines y). Duplicated entries
    in ``observed_flat`` are otherwise fine here (plain copy, no scatter-by-
    index, no uniqueness assumption).
    """
    n = len(observed_flat)
    if n > n_max:
        raise ValueError(
            f"len(observed_flat)={n} exceeds n_max={n_max}: this backend/"
            "call cannot pad this many observations"
        )
    indices = np.zeros(n_max, dtype=np.int32)
    values = np.zeros(n_max, dtype=np.float64)
    active = np.zeros(n_max, dtype=np.float64)
    indices[:n] = np.asarray(observed_flat, dtype=np.int32)
    if observed_y is None:
        values[:n] = noisy_truth_flat[np.asarray(observed_flat, dtype=int)]
    else:
        observed_y = np.asarray(observed_y, dtype=np.float64)
        if observed_y.shape != (n,):
            raise ValueError(f"observed_y has shape {observed_y.shape}, expected ({n},)")
        values[:n] = observed_y
    active[:n] = 1.0
    return indices, values, active

def assimilate_u15(
    backend: Any,
    latent: jax.Array,
    old_loglik: np.ndarray,
    logweights: np.ndarray,
    observed_flat: list[int],
    noisy_truth_flat: np.ndarray,
    ess_fraction: float,
    pcn_beta: float | np.ndarray,
    inference_seed: int,
    n_initial: int,
    resampling_count: int,
    *,
    force_rejuvenate: bool = False,
    observed_y: np.ndarray | None = None,
) -> tuple[jax.Array, np.ndarray, np.ndarray, float, Any, int, bool]:
    """Unchanged by the M7b remedy except this signature's widened type:
    ``pcn_beta`` is forwarded verbatim to ``backend.rejuvenate`` (only
    consumed if a resampling event actually triggers this call), so it is
    already agnostic to whether it is a scalar (``pcn_mode in
    {"fixed","adaptive"}``) or a length-``N_PCN_BLOCKS`` array
    (``pcn_mode="adaptive-blocked"``) -- the caller (the driver's
    ``PcnBetaState.current()``) decides which shape to pass, and the
    returned ``acceptance`` mirrors that same shape (``None`` when no
    resampling happened this round, exactly as before); in
    ``pcn_mode in {"segment","segment-blocked"}`` (M7c) it is instead a
    ``SegmentRejuvenationInfo``, again forwarded verbatim from
    ``backend.rejuvenate``.

    M7c ADDITIVE parameter ``force_rejuvenate`` (default ``False``, so the
    pre-M7c ESS-trigger-only behaviour is exactly preserved when omitted):
    when ``True`` AND the ESS trigger does NOT fire this round, rejuvenate
    IN PLACE anyway -- a resampling-free move step -- rather than doing
    nothing. This is a valid SMC move step even though the particles are
    NOT resampled to equal weights first: a pCN kernel is, by construction,
    an exact Markov kernel invariant to the CURRENT full posterior (exactly
    reversible w.r.t. the whitened prior, MH-corrected on the likelihood of
    ALL current observations); applying a pi-invariant kernel to each
    particle's STATE while leaving its IMPORTANCE WEIGHT untouched leaves
    the weighted empirical measure sum_i w_i * delta_{x_i} an equally valid
    (unbiased) approximation of that same posterior in distribution --
    the weights need not be uniform for that argument, unlike the
    ESS-triggered branch below (which resamples FIRST only because
    ``systematic_resample`` itself needs uniform incoming weights to
    produce an equal-count particle set, not because pCN itself requires
    it). When the ESS trigger DOES fire, behaviour is unchanged (resample
    then rejuvenate) regardless of ``force_rejuvenate`` -- per the task's
    precommitment, resampling always takes priority over the force path.
    Returns an additional ``rejuvenated`` bool (``True`` iff EITHER branch
    actually ran a rejuvenation this call), the M7c additive npz key.

    M10f additive: the padding capacity is read off
    ``backend.n_max_for(len(observed_flat))`` -- the SMALLEST bucket that
    can hold this many observations (see
    ``gpn_smc.backends.SharedProtocolGpnBackend.n_max_for``'s docstring) --
    rather than the single fixed ``backend.n_max`` (K2c's scheme, which is
    now the LARGEST bucket, and would defeat the point of bucketing by
    always padding/compiling for the biggest bucket from round 1). Falls
    back to ``N_MAX_U15`` via ``getattr``/``AttributeError`` for a
    duck-typed backend that defines neither method nor attribute.

    2026-10-31 additive keyword ``observed_y`` (default ``None``): per-
    observation y values aligned with ``observed_flat``; forwarded to
    ``padded_observations_u15`` (see there). ``None`` = old behaviour.
    """
    if hasattr(backend, "n_max_for"):
        n_max = backend.n_max_for(len(observed_flat))
    else:
        n_max = getattr(backend, "n_max", N_MAX_U15)
    obs_indices, y_padded, active = padded_observations_u15(
        observed_flat, noisy_truth_flat, n_max=n_max, observed_y=observed_y
    )
    loglik_device = backend.loglik(latent, obs_indices, y_padded, active)
    new_loglik = np.asarray(loglik_device).reshape(backend.n_particles)
    _lw_prev = logweights
    logweights = logweights + new_loglik - old_loglik
    weights, ess = normalize_logweights(logweights)
    acceptance = None
    rejuvenated = False
    # 2026-11-04 additive (mixing diagnostics; read-only, nothing set unless the backend opts in):
    # pre-resampling ESS, resampling flag and number of distinct ancestors of this assimilation.
    _mixdiag = bool(getattr(backend, "mixing_diagnostics", False))
    if _mixdiag:
        backend.last_assim_diag = {"ess_pre": float(ess), "resampled": False, "n_unique_ancestors": -1}
        # 2026-11-06 additive: log of the SMC evidence increment log p(y_n | y_1..n-1) (weighted mean of the
        # per-particle predictive densities) and, below, the resampling indices (for lineage counts).
        _m0, _m1 = np.max(_lw_prev), np.max(logweights)
        backend.last_assim_diag["log_evidence_increment"] = float(
            (_m1 + math.log(np.sum(np.exp(logweights - _m1)))) - (_m0 + math.log(np.sum(np.exp(_lw_prev - _m0)))))
        backend.last_assim_diag["indices"] = None
    if ess < ess_fraction * backend.n_particles:
        indices = systematic_resample(
            weights,
            derive_stream_seed(inference_seed, n_initial, len(observed_flat), purpose=0),
        )
        if _mixdiag:
            backend.last_assim_diag["resampled"] = True
            backend.last_assim_diag["n_unique_ancestors"] = int(np.unique(np.asarray(indices)).size)
            backend.last_assim_diag["indices"] = np.asarray(indices)
        particles = backend.host_particles(latent)[indices]
        latent = backend.shard_particles(particles)
        logweights = np.full(backend.n_particles, -math.log(backend.n_particles))
        weights = np.full(backend.n_particles, 1.0 / backend.n_particles)
        loglik_device = backend.loglik(latent, obs_indices, y_padded, active)
        latent, loglik_device, acceptance = backend.rejuvenate(
            latent,
            loglik_device,
            derive_stream_seed(inference_seed, n_initial, len(observed_flat), purpose=1),
            obs_indices,
            y_padded,
            active,
            pcn_beta,
        )
        new_loglik = np.asarray(loglik_device).reshape(backend.n_particles)
        ess = float(backend.n_particles)
        # S6-LR additive: low-rank moves return exact importance corrections
        # exp(ll_exact - ll_lr) (and exp(ll_lr0 - ll_exact0) at the start); None otherwise.
        corr = getattr(backend, "last_lowrank_logw_corr", None)
        if corr is not None:
            backend.last_lowrank_logw_corr = None
            logweights = logweights + corr
            weights, ess = normalize_logweights(logweights)
        resampling_count += 1
        rejuvenated = True
    elif force_rejuvenate:
        # M7c "--rejuvenate-every-round" branch: move step only, weights
        # and ess are left exactly as computed above (no resampling), per
        # the docstring's invariance argument.
        latent, loglik_device, acceptance = backend.rejuvenate(
            latent,
            loglik_device,
            derive_stream_seed(inference_seed, n_initial, len(observed_flat), purpose=1),
            obs_indices,
            y_padded,
            active,
            pcn_beta,
        )
        new_loglik = np.asarray(loglik_device).reshape(backend.n_particles)
        corr = getattr(backend, "last_lowrank_logw_corr", None)  # S6-LR additive
        if corr is not None:
            backend.last_lowrank_logw_corr = None
            logweights = logweights + corr
            weights, ess = normalize_logweights(logweights)
        rejuvenated = True
    return latent, new_loglik, logweights, ess, acceptance, resampling_count, rejuvenated

def assimilate_batches_u15(
    backend: Any,
    latent: jax.Array,
    old_loglik: np.ndarray,
    logweights: np.ndarray,
    observed: list[int],
    obs_batches: list[list[int]],
    noisy_truth_flat: np.ndarray,
    ess_fraction: float,
    pcn_state: Any,
    inference_seed: int,
    n_initial: int,
    resampling_count: int,
    is_segment_mode: bool,
    *,
    y_fn: Any = None,
) -> tuple[jax.Array, np.ndarray, np.ndarray, int, float | None, float | None]:
    """M13 additive: BATCH conditioning helper -- ``observed`` (grown IN
    PLACE, same list object the caller keeps) is extended by each batch in
    ``obs_batches`` (in order) before ONE ``assimilate_u15`` call per batch,
    exactly the M13a posterior-screen's batch-conditioning path
    (``u15_posterior_screen.py``'s ``run_replicate`` -- see its module
    docstring for why calling ``assimilate_u15`` once per batch of B new
    observations, instead of once per single observation, is already the
    valid SMC multi-observation-jump update: ``assimilate_u15`` always
    recomputes ``new_loglik`` from the CURRENT COMPLETE ``observed`` list, so
    the weight update ``new_loglik - old_loglik`` is exactly the loglik
    DIFFERENCE of the full observed list before vs. after the call,
    regardless of how many observations were added since the last call).
    Factored out here (rather than inlined separately in both driver
    scripts) so ``u15_posterior_screen.py`` and ``u15_compare.py`` share ONE
    copy of this batch-conditioning logic -- see each script's own docstring
    for how it is wired in.

    ``n_initial`` is forwarded verbatim to every ``assimilate_u15`` call as
    its own ``n_initial`` argument (the ``derive_stream_seed`` input that
    stays FIXED across a whole trajectory, not the batch's own size) --
    callers must pass the SAME value they use for every other
    ``assimilate_u15`` call in the same trajectory (the screen passes
    ``len(obs_order)`` (or 1 if empty); ``u15_compare.py``'s ``run_gpn_arm``
    passes its own ``n0``), so that switching an existing per-item
    assimilation loop over to this batched helper does not change the
    ``derive_stream_seed`` inputs consumed at any given ``len(observed)``.

    Returns ``(latent, old_loglik, logweights, resampling_count,
    last_acceptance, last_pcn_beta)`` -- the last two ``None`` if no batch
    ever triggered a rejuvenation (mirrors the per-item callers' own
    ``last_acceptance``/``last_pcn_beta`` convention, pre-reduced by
    ``np.mean`` exactly as they do, so callers can assign straight into
    their own trajectory-level state without re-deriving the segment/
    non-segment branching here).

    2026-10-31 additive keyword ``y_fn`` (default ``None`` = old behaviour):
    a callable ``observed -> np.ndarray`` returning the per-observation y
    values for the CURRENT ``observed`` list; when given, its result is
    forwarded to every ``assimilate_u15`` call as ``observed_y``.
    """
    last_acceptance: float | None = None
    last_pcn_beta: float | None = None
    for batch in obs_batches:
        observed.extend(int(i) for i in batch)
        effective_beta = pcn_state.current(len(observed))
        (
            latent, old_loglik, logweights, _ess, acceptance, resampling_count, _rejuvenated,
        ) = assimilate_u15(
            backend, latent, old_loglik, logweights, observed, noisy_truth_flat,
            ess_fraction, effective_beta, inference_seed, n_initial, resampling_count,
            observed_y=(None if y_fn is None else y_fn(observed)),
        )
        if acceptance is not None:
            if is_segment_mode:
                last_acceptance = float(np.mean(np.asarray(acceptance.last_segment_acceptance)))
                last_pcn_beta = float(np.mean(np.asarray(acceptance.beta_used_final_segment)))
            else:
                last_acceptance = float(np.mean(np.asarray(acceptance)))
                last_pcn_beta = float(np.mean(np.asarray(effective_beta)))
            pcn_state.update(acceptance)
    return latent, old_loglik, logweights, resampling_count, last_acceptance, last_pcn_beta
