"""gpn_smc.pcn -- pCN rejuvenation step-size machinery moved verbatim from
u15_protocol_backend.py, per K2_DESIGN.md K2a: PCN_MODES, adapt_pcn_beta,
_segment_beta_update, PcnBetaState, SegmentRejuvenationInfo. Bodies are
character-identical to the source except these import lines and this module
docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .constants import (
    ADAPTIVE_BETA_INITIAL,
    ADAPTIVE_BETA_MAX,
    ADAPTIVE_BETA_MIN,
    ADAPTIVE_BETA_TARGET_ACCEPTANCE,
    N_PCN_BLOCKS,
    SEGMENT_BETA_MIN,
)


PCN_MODES = (
    "fixed",
    "adaptive",
    "adaptive-blocked",
    "segment",
    "segment-blocked",
    # S2 (GPN_IMPROVEMENT_PROGRAM_2026-10-02): same kernel as "segment-blocked"
    # but the move loop carries only the observed rows of h. Shared backend only.
    "segment-blocked-obs",
)

def _segment_beta_update(beta: jax.Array, acceptance: jax.Array) -> jax.Array:
    """Pure-JAX (traced) analogue of ``adapt_pcn_beta``, called ONLY inside
    the "segment"/"segment-blocked" pCN kernels' ``jax.lax.scan`` over
    segments (never host-side). Same formula as ``adapt_pcn_beta``, floor
    lowered to ``SEGMENT_BETA_MIN`` (see its comment). ``acceptance`` here
    is always the POPULATION-mean acceptance of the just-finished segment
    (already psum'd across every device before this is called), which is
    what makes every device compute the SAME beta for the next segment --
    see the module docstring's "M7c remedy" paragraph for why that
    determinism is what keeps each segment's kernel a valid pi-invariant
    Metropolis step for the beta it actually used.
    """
    updated = beta * jnp.exp(0.5 * (acceptance - ADAPTIVE_BETA_TARGET_ACCEPTANCE))
    return jnp.clip(updated, SEGMENT_BETA_MIN, ADAPTIVE_BETA_MAX)

def adapt_pcn_beta(beta: Any, acceptance: Any) -> np.ndarray:
    """Host-side (never jitted) pCN step-size update, precommitted exactly::

        beta <- clip(beta * exp(0.5 * (acceptance - 0.23)), 0.01, 0.5)

    ``beta``/``acceptance`` are elementwise -- pass Python floats for the
    ``"adaptive"`` mode's single running beta, or length-``N_PCN_BLOCKS``
    arrays for ``"adaptive-blocked"``'s per-block betas (identical formula,
    applied independently per block, no cross-block coupling). Pure numpy,
    called only between rejuvenations (beta is frozen for the 48 pCN steps
    of any one rejuvenation, so pCN invariance per round is untouched --
    adaptation only ever affects the NEXT rejuvenation's step size). Always
    returns an ``np.ndarray`` (0-d for scalar input); callers needing a
    Python float wrap with ``float(...)``.
    """
    beta_arr = np.asarray(beta, dtype=np.float64)
    acceptance_arr = np.asarray(acceptance, dtype=np.float64)
    updated = beta_arr * np.exp(0.5 * (acceptance_arr - ADAPTIVE_BETA_TARGET_ACCEPTANCE))
    return np.clip(updated, ADAPTIVE_BETA_MIN, ADAPTIVE_BETA_MAX)

@dataclass
class SegmentRejuvenationInfo:
    """Returned by ``rejuvenate`` INSTEAD of a plain float/array acceptance
    when ``pcn_mode in {"segment", "segment-blocked"}`` (M7c). Every field
    is a Python float for ``"segment"`` and a length-``N_PCN_BLOCKS`` numpy
    array for ``"segment-blocked"`` -- callers that only need a scalar npz
    field reduce with ``np.mean`` themselves (exactly as the pre-existing
    blocked modes' plain array acceptance is already reduced for its own
    scalar npz fields); nothing here requires branching on which mode
    produced it.

    * ``last_segment_acceptance`` / ``beta_used_final_segment``: the
      population-mean acceptance measured during, and the beta actually
      used by, THIS round's LAST segment -- a matched pair (this round's
      diagnostic "what happened"), persisted as ``"pcn_acceptance"`` /
      ``"pcn_beta"``.
    * ``first_segment_acceptance``: the population-mean acceptance of
      THIS round's FIRST segment (using the beta warm-started from the
      previous round) -- persisted as the additive key
      ``"pcn_acceptance_first_segment"``, diagnosing how far off the
      warm start was.
    * ``beta_next``: the ADAPTED beta after the last segment's update --
      NOT the same value as ``beta_used_final_segment`` -- this is what
      ``PcnBetaState.update`` stores to warm-start the FIRST segment of
      the NEXT round (this round's freshest information, never actually
      applied to any pCN step in this round).
    * ``beta_trace`` / ``acceptance_trace``: the full per-segment
      ``(n_segments,)`` (or ``(n_segments, N_PCN_BLOCKS)``) sequence of
      betas USED and population-mean acceptances MEASURED, one entry per
      segment, in round order -- self-test/diagnostic use only (e.g. M7c's
      self-test 2), never persisted to the production npz schema.
    """

    last_segment_acceptance: Any
    first_segment_acceptance: Any
    beta_used_final_segment: Any
    beta_next: Any
    beta_trace: np.ndarray
    acceptance_trace: np.ndarray

@dataclass
class PcnBetaState:
    """Host-side (never jitted) running pCN step-size state for ONE replicate
    trajectory -- lives in the driver/assimilate layer, passed into
    ``rejuvenate`` as a plain array/float argument every call (never baked
    into a jitted closure). Reset fresh at the start of every replicate: a
    single compiled backend instance is reused across many replicates by
    the driver's ``backend_cache``, but each replicate is its own
    independent SMC trajectory/inference stream, so the running beta must
    not leak across replicates.

    * ``mode="fixed"``: ``current(n)`` reproduces the ORIGINAL M7 schedule
      exactly, ``fixed_beta * min(1, sqrt(4/n))``; ``update`` is a no-op
      (the schedule depends only on ``n``, never on observed acceptance) --
      this is the path the M7b invariance self-test checks is unchanged
      from the pre-M7b code.
    * ``mode="adaptive"``: one scalar ``beta``, seeded at
      ``ADAPTIVE_BETA_INITIAL`` (0.30); ``update(acceptance)`` applies
      ``adapt_pcn_beta`` using ONLY that rejuvenation's own observed mean
      acceptance.
    * ``mode="adaptive-blocked"``: a length-``N_PCN_BLOCKS`` ``beta_blocks``
      array, one entry per block, each adapted independently from that same
      block's own observed acceptance.
    * ``mode="segment"`` (M7c): one scalar ``beta``, seeded and warm-started
      exactly like ``"adaptive"``'s -- the difference is entirely inside
      the backend's kernel (within-round segmented adaptation; see the
      module docstring's "M7c remedy" paragraph). ``update`` here receives
      a ``SegmentRejuvenationInfo`` (not a plain acceptance float) and
      stores its ``beta_next`` verbatim -- no formula applied at this
      layer, since the backend's kernel already ran the full per-segment
      adaptation in-graph.
    * ``mode="segment-blocked"`` (M7c): a length-``N_PCN_BLOCKS``
      ``beta_blocks`` array, warm-started exactly like
      ``"adaptive-blocked"``'s; ``update`` stores the
      ``SegmentRejuvenationInfo.beta_next`` array verbatim, same rationale
      as ``"segment"``.
    """

    mode: str
    fixed_beta: float
    beta: float = ADAPTIVE_BETA_INITIAL
    beta_blocks: np.ndarray | None = None
    # M9 additive field: the number of pCN blocks the PAIRED backend
    # actually uses (``backend.n_pcn_blocks``) -- defaults to the module
    # constant (4), so every pre-M9 caller (which never passes this) is
    # completely unaffected. A caller driving an M9-library backend
    # (SharedProtocolGpnBackend with n_l2_types=4, or
    # IndependentProtocolGpnBackend with n_protocols=8) must pass its
    # backend's own ``n_pcn_blocks`` here so ``beta_blocks``'s default shape
    # (used only in "adaptive-blocked"/"segment-blocked" mode) matches what
    # the backend's blocked pCN kernel actually expects.
    n_blocks: int = N_PCN_BLOCKS

    def __post_init__(self) -> None:
        if self.mode not in PCN_MODES:
            raise ValueError(f"unknown pcn_mode {self.mode!r}; choices are {PCN_MODES}")
        if self.beta_blocks is None:
            self.beta_blocks = np.full(self.n_blocks, ADAPTIVE_BETA_INITIAL, dtype=np.float64)

    def current(self, n_observed: int) -> float | np.ndarray:
        """The beta value to pass into THIS round's ``assimilate_u15`` call
        (only actually consumed by ``rejuvenate`` if resampling triggers
        this round; otherwise computed but unused, exactly as the original
        fixed-schedule driver code already did every round). For the two
        segment modes this is the STARTING beta for the round's first
        segment; the round's kernel adapts it further internally."""
        if self.mode == "fixed":
            return self.fixed_beta * min(1.0, math.sqrt(4.0 / n_observed))
        if self.mode in ("adaptive", "segment"):
            return self.beta
        return np.array(self.beta_blocks, copy=True, dtype=np.float64)

    def update(self, acceptance: Any) -> None:
        """Called ONLY when a rejuvenation actually happened this round
        (i.e. the caller's ``acceptance is not None``); updates the running
        state consumed by FUTURE rounds' ``current()`` calls. No-op in
        fixed mode. In ``"segment"``/``"segment-blocked"`` mode,
        ``acceptance`` is actually a ``SegmentRejuvenationInfo`` (not a
        plain float/array) -- see that dataclass's ``beta_next`` field."""
        if self.mode == "fixed":
            return
        if self.mode == "adaptive":
            self.beta = float(adapt_pcn_beta(self.beta, acceptance))
        elif self.mode == "adaptive-blocked":
            self.beta_blocks = np.asarray(
                adapt_pcn_beta(self.beta_blocks, acceptance), dtype=np.float64
            )
        elif self.mode == "segment":
            self.beta = float(np.asarray(acceptance.beta_next))
        else:  # "segment-blocked"
            self.beta_blocks = np.asarray(acceptance.beta_next, dtype=np.float64)
