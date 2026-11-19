"""gpn_smc.backends -- SharedProtocolGpnBackend, IndependentProtocolGpnBackend,
split_flat_observations_by_protocol, moved verbatim from
u15_protocol_backend.py, per K2_DESIGN.md K2a. Bodies are character-identical
to the source except these import lines and this module docstring; the only
NEW code in this file is ``penultimate_outputs`` on both classes (K2_DESIGN.md
K2a's "New method" section), added after the moved class bodies below.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from .constants import (
    GRID_SIZE_PER_PROTOCOL,
    N_CANDIDATES_M9,
    N_MAX_U15,
    N_PCN_BLOCKS,
    N_PROTOCOLS,
    PCN_SEGMENT_LENGTH,
    RAMP_W_LIB_B,
)
from .numerics import compute_chunk_size, rbf_covariance, stable_cholesky
from .pcn import PCN_MODES, SegmentRejuvenationInfo, _segment_beta_update
from .library import ProtocolLibrarySpec
from .pushforward import (
    pushforward_h,
    pushforward_shared_h,
    pushforward_shared_h_battery,
    pushforward_shared_h_general,
    pushforward_shared_h_general_nl1,
    pushforward_shared_h_lib_s,
    pushforward_shared_h_m12,
    shared_pcn_block_bounds,
    shared_pcn_block_bounds_m12,
)
from .terminal_rb import (
    LOWRANK_H_MARGIN,
    LOWRANK_H_MAX,
    LOWRANK_H_MIN,
    LowRankTerminal,
    build_lowrank_moments,
    build_loglik_from_hobs,
    build_terminal_rb_functions,
)


REFINE_STEPS = 4  # S3: iterative-refinement steps in the null-space refresh


def da_subset_rows(
    obs_indices: np.ndarray, active: np.ndarray, m: int, per_protocol: int
) -> tuple[np.ndarray, np.ndarray]:
    """S5 additive: fixed subset of ``m`` observation rows for the delayed-acceptance
    stage-1 screen. State independent (depends on the observed history only).
    Returns ``(idx int32 (m,), mask float64 (m,))``. If ``m >= n_active`` every active
    row is included (mask 1) and the remaining slots are inactive rows (mask 0).
    Otherwise: first one row per observed protocol (the median-position row of that
    protocol, if the number of protocols <= m; else protocols evenly spread), then
    the remaining slots evenly spaced over the other active rows."""
    act_rows = np.flatnonzero(np.asarray(active) > 0.5)
    n_act = act_rows.size
    n_max = int(np.asarray(active).shape[0])
    if m >= n_act:
        inact = np.flatnonzero(np.asarray(active) <= 0.5)[: m - n_act]
        idx = np.concatenate([act_rows, inact]).astype(np.int32)
        mask = np.concatenate([np.ones(n_act), np.zeros(inact.size)]).astype(np.float64)
        return idx, mask
    protos = np.asarray(obs_indices)[act_rows] // per_protocol
    chosen: list[int] = []
    uniq = np.unique(protos)
    reps = [int(act_rows[np.flatnonzero(protos == u)][np.flatnonzero(protos == u).size // 2]) for u in uniq]
    if len(reps) > m:
        pick = np.floor(np.linspace(0, len(reps) - 1, m) + 0.5).astype(int)
        reps = [reps[i] for i in pick]
    chosen = list(reps)
    k = m - len(chosen)
    if k > 0:
        rem = np.setdiff1d(act_rows, np.asarray(chosen, dtype=np.int64))
        pick = np.floor(np.linspace(0, rem.size - 1, k) + 0.5).astype(int)
        chosen += [int(r) for r in rem[pick]]
    idx = np.sort(np.asarray(chosen, dtype=np.int32))
    assert idx.size == m and np.unique(idx).size == m and idx.max() < n_max
    return idx, np.ones(m, dtype=np.float64)


class SharedProtocolGpnBackend:
    """Protocol-bandits shared learner: one whitened state per TYPE.

    Under the M8 default library (``n_l2_types=2``): whitened per-particle
    state, width ``6*G = 384`` (``G = 64``): ``[xi_A(G), xi_D(G), xi_B(2G),
    xi_C(2G)]``. The terminal type R is Rao-Blackwellised jointly over the
    pooled 256-point candidate space.

    Under the M9 8-protocol library (``n_l2_types=4``, additive --
    U15_PROTOCOL_BANDITS_PLAN.md's 2026-08-21 section): whitened state width
    ``10*G = 640``: ``[xi_A(G), xi_D(G), xi_B(2G), xi_C(2G), xi_G(2G),
    xi_H(2G)]``; the pooled candidate space is 512-wide, in the M9 L2-major
    protocol order (see ``pushforward_shared_h_general``). Both libraries
    share the identical class/kernel scaffolding below -- only the
    pushforward function, state width, candidate count, and blocked-pCN
    block layout depend on ``n_l2_types``; the ``n_l2_types=2`` path calls
    the ORIGINAL, byte-unchanged ``pushforward_shared_h`` and computes
    exactly the pre-M9 literal block bounds, so M8's self-test 1 regression
    is unaffected.
    """

    def __init__(
        self,
        warped_x: np.ndarray,
        library_spec: ProtocolLibrarySpec,
        noise_variance: float,
        kernel_jitter: float,
        n_particles: int,
        pcn_steps: int,
        *,
        pcn_mode: str = "adaptive",
        chunk_target_bytes: int = 256 * 1024 * 1024,
        pcn_segment_length: int | None = None,
        n_l2_types: int = 2,
        protocol_order: str = "l2_major",
        n_l1_types: int = 2,
        n_l3_types: int = 0,
        ramp_w: np.ndarray | None = None,
        n_max_buckets: tuple[int, ...] | None = None,
        n_max: int | None = None,
        l2_chol_reuse: bool = True,
        obs_chunk_target_bytes: int | None = None,
        l2_null_refresh: int = 0,
        adapt_beta: bool = True,
        da_subset: int = 0,
        terminal_lowrank: int = 0,
        lowrank_moments: bool | None = None,
        l2_refresh_per_type: bool = False,
        lr_chunk_target_bytes: int = 1024 * 1024 * 1024,
        terminal_lowrank_min_n: int = 0,
        allow_repeat_observations: bool = False,
    ) -> None:
        # M10f backward-compat shim: gpn_smc is also imported by
        # kg/cost_aware_protocol_kg (the K2c ``n_max=<int>`` single-bucket
        # kwarg its tests/test_gpn_smc_nmax.py still constructs backends
        # with) -- ``n_max`` is accepted as a deprecated alias for
        # ``n_max_buckets=(n_max,)`` so that suite keeps working unchanged.
        # Passing BOTH is an error (ambiguous).
        if n_max is not None:
            if n_max_buckets is not None:
                raise ValueError("pass only one of n_max (deprecated) or n_max_buckets")
            n_max_buckets = (int(n_max),)
        devices = jax.devices("gpu")
        if not devices:
            raise RuntimeError("No JAX GPU devices are visible")
        if n_particles % len(devices):
            raise ValueError(
                f"particles={n_particles} must be divisible by devices={len(devices)}"
            )
        if pcn_mode not in PCN_MODES:
            raise ValueError(f"pcn_mode={pcn_mode!r} must be one of {PCN_MODES}")
        if protocol_order not in ("l2_major", "l1_major"):
            raise ValueError(
                f"protocol_order={protocol_order!r} must be 'l2_major' or 'l1_major'"
            )
        # M10f additive: ``n_max`` is no longer one static int -- it is now a
        # tuple of BUCKETS, each of which gets its OWN lazily-built terminal
        # RB / pCN kernel set (see ``self._bundle`` below), so a run started
        # at n_obs=8 can grow past a small early bucket's capacity without
        # ever recompiling more than each bucket's kernel exactly once.
        # Default ``(N_MAX_U15,)`` reproduces every pre-M10f caller's single-
        # bucket behaviour exactly (byte-identical bucket contents, byte-
        # identical kernel code inside that one bucket). ``self.n_max``
        # (backward-compat attribute -- gpn_smc.assimilate reads this) is the
        # MAX bucket, i.e. the largest observation count this backend can
        # ever pad to; callers that need the bucket for a SPECIFIC observed
        # count must call ``self.n_max_for(n_obs)`` instead (see its
        # docstring) rather than assuming a single fixed n_max.
        self.n_max_buckets = tuple(
            sorted(int(b) for b in (n_max_buckets if n_max_buckets is not None else (N_MAX_U15,)))
        )
        if not self.n_max_buckets:
            raise ValueError("n_max_buckets must not be empty")
        if len(set(self.n_max_buckets)) != len(self.n_max_buckets):
            raise ValueError(f"n_max_buckets={self.n_max_buckets} has duplicate entries")
        if any(b <= 0 for b in self.n_max_buckets):
            raise ValueError(f"n_max_buckets={self.n_max_buckets}: every bucket must be > 0")
        self.n_max = self.n_max_buckets[-1]
        # M10f additive: lazily-built per-bucket kernel bundles (dict keyed
        # by bucket n_max -> the dict ``_build_bundle`` below returns), plus
        # the Change A layer-2-Cholesky-reuse toggle. ``l2_chol_reuse=False``
        # keeps every kernel byte-for-byte identical to the pre-M10f code
        # (needed by the equivalence self-test); ``True`` (the default)
        # restructures ONLY the "segment-blocked" kernel's layer-2 block
        # updates at ``n_l3_types==0 and n_l1_types==2 and n_l2_types!=2``
        # (the ``pushforward_shared_h_general`` dispatch -- see this
        # constructor's pushforward-selection ``if``/``elif`` chain below)
        # to reuse ONE layer-2 Cholesky factor across all ``n_l2_types``
        # block proposals in a sweep instead of recomputing it inside every
        # block's full pushforward; every OTHER dispatch (M8's
        # ``n_l2_types==2``, M11's ``n_l1_types!=2``, M12/M13's
        # ``n_l3_types>=1``, and the "adaptive-blocked"/"fixed"/"adaptive"/
        # "segment" kernels) is UNCHANGED regardless of this flag -- see the
        # "segment-blocked" branch below for exactly which combination gets
        # the new kernel.
        self._bundles: dict[int, dict] = {}
        self.l2_chol_reuse = bool(l2_chol_reuse)
        self.devices = devices
        self.n_devices = len(devices)
        self.n_particles = n_particles
        self.local_particles = n_particles // self.n_devices
        self.library_spec = library_spec
        G = int(warped_x.shape[0])
        if G != GRID_SIZE_PER_PROTOCOL:
            raise ValueError(f"expected a {GRID_SIZE_PER_PROTOCOL}-point grid, got {G}")
        if n_l2_types < 1:
            raise ValueError(f"n_l2_types={n_l2_types} must be >= 1")
        if n_l1_types < 1:
            raise ValueError(f"n_l1_types={n_l1_types} must be >= 1")
        if n_l3_types not in (0, 1, 2, 3):
            raise ValueError(f"n_l3_types={n_l3_types} must be 0, 1, 2, or 3")
        if n_l3_types in (1, 2, 3) and n_l2_types != 1:
            raise ValueError(
                "n_l3_types in (1, 2, 3) (M12 mixed-depth / M13a LIB-S / "
                "M13a LIB-B) requires n_l2_types=1 (ONE shared B type and "
                f"ONE shared C type), got n_l2_types={n_l2_types}"
            )
        # M9 additive generalisation: n_l2_types=2 (default) reproduces the
        # M8 four-protocol library exactly (grid_size=6*G=384,
        # n_candidates=N_CANDIDATES=256, byte-identical values below);
        # n_l2_types=4 gives the M9 eight-protocol library
        # (grid_size=10*G=640, n_candidates=N_CANDIDATES_M9=512). See the
        # class docstring. M10 additive: n_l2_types=8, protocol_order=
        # "l1_major" gives M10's 16-protocol library (n_candidates=1024) --
        # see pushforward_shared_h_general's docstring for the ordering
        # convention. protocol_order only matters when n_l2_types != 2 (the
        # n_l2_types==2 path below always calls the ORIGINAL
        # pushforward_shared_h, which is hardcoded to the M8 L1-major
        # ordering already, so a caller passing protocol_order="l2_major"
        # -- the default -- at n_l2_types==2 gets the untouched M8 path
        # regardless; only a HYPOTHETICAL n_l2_types==2, protocol_order=
        # "l2_major" combination on the *general* function would actually
        # differ from pushforward_shared_h, and no caller here ever
        # constructs that combination since n_l2_types==2 always takes the
        # dedicated fast path).
        # M11 additive: n_l1_types (default 2, reproducing every pre-M11
        # caller's behaviour byte-for-byte -- the n_l1_types==2 branches
        # below are UNCHANGED from pre-M11 code) generalises the number of
        # layer-1 types (previously hardcoded to 2, "A" and "D"). At
        # n_l1_types!=2 the joint space is n_l1_types*n_l2_types protocols
        # in the l1-major order p = i*n_l2_types + j (see
        # ``pushforward_shared_h_general_nl1``'s docstring) -- used by the
        # M11 PB-L2W (n_l1_types=4, n_l2_types=2) and PB-L2N (n_l1_types=2,
        # n_l2_types=4) families.
        self.n_l2_types = int(n_l2_types)
        self.n_l1_types = int(n_l1_types)
        self.n_l3_types = int(n_l3_types)
        # M12 additive: n_l3_types=1 selects the mixed-depth "cross-depth
        # warm-up" library (PB-D24) -- SIMPLE protocols 0..n_l1_types-1 at
        # depth 2 (penultimate = h_A_i) and TARGET protocols at depth 4
        # (penultimate = h_C_i, after ONE shared B and ONE shared C layer),
        # 2*n_l1_types protocols total.
        # M13a additive: n_l3_types=2 selects LIB-S (PB-SIB) -- n_l1_types
        # (6) UNIFORM depth-4 protocols, no simple set, candidate space
        # n_l1_types*G; n_l3_types=3 selects LIB-B (PB-BAT) -- M12's own
        # mixed-depth library PLUS two ramp protocols evaluating the shared
        # B/C types at the known ``ramp_w``, candidate space
        # (2*n_l1_types+2)*G. All three override the generic
        # n_protocols/grid_size/n_candidates/n_pcn_blocks formulas below
        # (which assume n_l2_types independent layer-2 TYPES, not a single
        # B followed by a single C) -- see pushforward_shared_h_m12/
        # pushforward_shared_h_lib_s/pushforward_shared_h_battery's
        # docstrings / this module's "M12"/"M13a posterior screen" constants
        # blocks.
        self.mixed_depth = self.n_l3_types in (1, 3)
        self.uniform_deep = self.n_l3_types == 2
        self.battery = self.n_l3_types == 3
        self.protocol_order = protocol_order
        self.n_protocols = self.n_l1_types * self.n_l2_types
        self.grid_size = self.n_l1_types * G * (1 + self.n_l2_types)  # whitened state width
        self.n_candidates = self.n_protocols * G
        # Blocked-pCN block count (n_l1_types layer-1 blocks, then one block
        # per L2 type) -- exposed unconditionally (regardless of pcn_mode)
        # so a driver can size PcnBetaState's beta_blocks correctly before
        # choosing a mode.
        self.n_pcn_blocks = self.n_l1_types + self.n_l2_types
        if self.n_l3_types == 1:  # M12 PB-D24
            self.n_protocols = 2 * self.n_l1_types
            self.grid_size = 3 * self.n_l1_types * G
            self.n_candidates = self.n_protocols * G
            self.n_pcn_blocks = self.n_l1_types + 2
        elif self.n_l3_types == 2:  # M13a LIB-S (PB-SIB)
            self.n_protocols = self.n_l1_types
            self.grid_size = 3 * self.n_l1_types * G
            self.n_candidates = self.n_protocols * G
            self.n_pcn_blocks = self.n_l1_types + 2
        elif self.n_l3_types == 3:  # M13a LIB-B (PB-BAT)
            self.n_protocols = 2 * self.n_l1_types + 2
            self.grid_size = 3 * self.n_l1_types * G
            self.n_candidates = self.n_protocols * G
            self.n_pcn_blocks = self.n_l1_types + 2
            self.ramp_w = jnp.asarray(
                RAMP_W_LIB_B if ramp_w is None else ramp_w, dtype=jnp.float64
            )
            if self.ramp_w.shape[0] != G:
                raise ValueError(
                    f"ramp_w has {self.ramp_w.shape[0]} points, expected {G}"
                )
        self.pcn_steps = pcn_steps
        # M7b remedy: "fixed" and "adaptive" both use the SAME joint pCN
        # kernel below (self.blocked=False) -- the two modes differ ONLY in
        # what beta VALUE the driver computes and passes into rejuvenate()
        # (a fixed sqrt(4/n) schedule vs a running adapted value), never in
        # this backend's kernel structure. Only "adaptive-blocked" changes
        # the kernel itself (sequential per-type-block sweeps below).
        self.pcn_mode = pcn_mode
        # S0c additive: adapt_beta=False replaces every per-segment
        # _segment_beta_update(...) by the identity (beta carried unchanged;
        # traces still record beta/acceptance). Python-level switch at trace
        # time, so the default True compiles the identical kernel.
        self.adapt_beta = bool(adapt_beta)
        # S3 additive: exact conditional-prior refresh of the null-space part
        # of every layer-2 block every ``l2_null_refresh`` sweeps (0 = off).
        self.l2_null_refresh = int(l2_null_refresh)
        if self.l2_null_refresh < 0:
            raise ValueError("l2_null_refresh must be >= 0")
        if self.l2_null_refresh > 0 and pcn_mode != "segment-blocked-obs":
            raise ValueError("l2_null_refresh > 0 requires pcn_mode='segment-blocked-obs'")
        self._refresh_once_cache: dict = {}
        # S5 additive: delayed acceptance (Christen & Fox 2005) with an
        # m-row subset terminal likelihood as the stage-1 screen (0 = off).
        self.da_subset = int(da_subset)
        if self.da_subset < 0:
            raise ValueError("da_subset must be >= 0")
        if self.da_subset > 0 and pcn_mode != "segment-blocked-obs":
            raise ValueError("da_subset > 0 requires pcn_mode='segment-blocked-obs'")
        self.last_da_stats = None
        # S6-LR additive (all default off = old behaviour).
        # terminal_lowrank m > 0: rejuvenation moves use a fixed-grid Nystrom
        # (m inducing points) terminal likelihood; one exact evaluation per call
        # corrects the particle weights (exact SMC). lowrank_moments: per-particle
        # predictive mean/variance at the candidates from the same features
        # (default: on iff terminal_lowrank > 0). l2_refresh_per_type: the S3
        # null-space refresh solves one (cap x cap) block per L2 type instead of the
        # n_max x n_max masked Gram (forced on under terminal_lowrank > 0).
        self.terminal_lowrank = int(terminal_lowrank)
        if self.terminal_lowrank < 0 or 0 < self.terminal_lowrank < 8:
            raise ValueError("terminal_lowrank must be 0 or >= 8")
        if self.terminal_lowrank > 0 and pcn_mode != "segment-blocked-obs":
            raise ValueError("terminal_lowrank > 0 requires pcn_mode='segment-blocked-obs'")
        if self.terminal_lowrank > 0 and self.da_subset > 0:
            raise ValueError("terminal_lowrank > 0 requires da_subset == 0")
        self.lowrank_moments = (
            self.terminal_lowrank > 0 if lowrank_moments is None else bool(lowrank_moments)
        )
        if self.lowrank_moments and pcn_mode != "segment-blocked-obs":
            raise ValueError("lowrank_moments requires pcn_mode='segment-blocked-obs'")
        self.l2_refresh_per_type = bool(l2_refresh_per_type) or self.terminal_lowrank > 0
        if self.l2_refresh_per_type and self.da_subset > 0:
            raise ValueError("per-type refresh is not implemented for da_subset > 0")
        if self.l2_refresh_per_type and pcn_mode != "segment-blocked-obs":
            raise ValueError("per-type refresh requires pcn_mode='segment-blocked-obs'")
        self.lr_chunk_target_bytes = int(lr_chunk_target_bytes)
        # buckets with n_max < terminal_lowrank_min_n use the exact kernel (with the
        # per-type refresh); 0 = low-rank moves in every bucket.
        self.terminal_lowrank_min_n = int(terminal_lowrank_min_n)
        self.last_lowrank_logw_corr = None
        self.last_lowrank_stats = None
        self._refresh_typed_once_cache: dict = {}
        # 2026-11-01 additive: the same cell may be observed repeatedly (the driver's
        # --allow-repeat-observations). The per-type typed layout of the S6-LR path then holds
        # one slot per OBSERVATION, not per distinct cell, so the per-type capacity is n_max
        # (not min(n_max, 2G) = the number of distinct cells of a type); the null-space refresh
        # still needs only the distinct cells (<= 2G per type) and gets its own deduplicated
        # tables when the typed capacity exceeds 2G. False (default) = old behaviour exactly.
        self.allow_repeat_observations = bool(allow_repeat_observations)
        self.blocked = pcn_mode in ("adaptive-blocked", "segment-blocked", "segment-blocked-obs")
        # M7c remedy: "segment"/"segment-blocked" additionally split each
        # rejuvenation's pcn_steps into within-round segments (module
        # docstring's "M7c remedy" paragraph). pcn_segment_length defaults
        # to the module constant; a self-test may override it (see the
        # module docstring's closing paragraph).
        self.segment_mode = pcn_mode in ("segment", "segment-blocked", "segment-blocked-obs")
        if self.segment_mode:
            segment_length = (
                PCN_SEGMENT_LENGTH if pcn_segment_length is None else int(pcn_segment_length)
            )
            if segment_length <= 0 or pcn_steps % segment_length != 0:
                raise ValueError(
                    f"pcn_steps={pcn_steps} must be a positive multiple of "
                    f"pcn_segment_length={segment_length} in pcn_mode={pcn_mode!r}"
                )
            self.pcn_segment_length = segment_length
            self.n_pcn_segments = pcn_steps // segment_length
        self.particle_sharding = NamedSharding(Mesh(np.asarray(devices), ("device",)), P("device"))

        layer1_covariance = rbf_covariance(
            warped_x, library_spec.amplitude, library_spec.layer1_lengthscale
        )
        layer1_cholesky_np, layer1_jitter_used = stable_cholesky(layer1_covariance, kernel_jitter)
        self.layer1_jitter_used = layer1_jitter_used
        # ALL n_l1_types layer-1 types share amplitude/lengthscale by
        # precommitment on the SAME fixed grid -> ONE precomputed (G,G)
        # factor serves every one of them (named A/D for backward
        # compatibility with pre-M11 callers at n_l1_types==2).
        layer1_cholesky = jnp.asarray(layer1_cholesky_np, dtype=jnp.float64)
        self.layer1_cholesky_A = layer1_cholesky
        self.layer1_cholesky_D = layer1_cholesky
        self.layer1_cholesky = layer1_cholesky

        jitter_const = float(kernel_jitter)
        self._jitter_const = jitter_const
        self._layer2_amplitude = float(library_spec.amplitude)
        self._layer2_lengthscale = float(library_spec.layer2_lengthscale)

        # M9 additive: conservatively scale the memory target down as more
        # L2 types are added, so the chunk-size safety margin never shrinks
        # below what M9 already ran with. At n_l2_types=2 this reduces to
        # ``chunk_target_bytes`` exactly (int(chunk_target_bytes*2/2) ==
        # chunk_target_bytes), so compute_chunk_size is called with THE SAME
        # arguments as before -- byte-identical chunk_size_local for M8.
        # M10 note: the mandatory shared-Cholesky reuse (see
        # ``pushforward_shared_h_general``'s chunk_fn) means only ONE
        # (2G,2G) covariance/Cholesky pair is actually live per chunk
        # regardless of n_l2_types (down from n_l2_types pairs before the
        # reuse), so this formula is now MORE conservative than the true
        # footprint requires at n_l2_types>2 -- left unchanged anyway
        # (never loosened) both because it is still a safe, sufficient
        # margin and because loosening it would change ``chunk_size_local``
        # for the already-byte-reproducible M9 n_l2_types=4 path for no
        # correctness benefit (chunking never changes results, only
        # chunk_size_local's own value, which the M9 regression fixture
        # does not pin down independently of the run's output arrays, but
        # there is no reason to touch it).
        if self.n_l1_types == 2:
            # Pre-M11 chunk-sizing formula, left byte-identical at
            # n_l1_types==2 (its default) so M8/M9/M10 remain
            # byte-reproducible.
            layer2_chunk_target_bytes = int(chunk_target_bytes * 2 / max(2, self.n_l2_types))
            chunk_size_local = compute_chunk_size(
                self.local_particles, 2 * G, layer2_chunk_target_bytes
            )
        else:
            # M11 additive: the layer-2 covariance/Cholesky is now
            # (n_l1_types*G, n_l1_types*G) -- size the chunk off the ACTUAL
            # tensor dimension (compute_chunk_size's own contract), rather
            # than the pre-M11 formula's fixed 2*G assumption. Only one such
            # (n_l1_types*G)^2 buffer is live per chunk at a time (the
            # mandatory Cholesky-reuse precommitment), so chunk_target_bytes
            # is used directly (no extra conservative scaling needed).
            chunk_size_local = compute_chunk_size(
                self.local_particles, self.n_l1_types * G, chunk_target_bytes
            )
        self.chunk_size_local = chunk_size_local

        if self.n_l3_types >= 1:
            # M12/M13a: cross-depth warm-up / posterior-screen libraries --
            # see pushforward_shared_h_m12/pushforward_shared_h_lib_s/
            # pushforward_shared_h_battery's docstrings. Checked FIRST since
            # all three always have n_l1_types != 2 (4 or 6, precommitted)
            # and would otherwise fall into the n_l1_types!=2 ("else")
            # branch below, which calls the WRONG function
            # (pushforward_shared_h_general_nl1, which assumes n_l2_types
            # independent layer-2 TYPES rather than one shared B followed by
            # one shared C).
            if self.n_l3_types == 1:  # M12 PB-D24

                def pushforward(state: jax.Array) -> jax.Array:
                    return pushforward_shared_h_m12(
                        state,
                        layer1_cholesky,
                        self._layer2_amplitude,
                        self._layer2_lengthscale,
                        jitter_const,
                        G,
                        chunk_size_local,
                        self.n_l1_types,
                    )
            elif self.n_l3_types == 2:  # M13a LIB-S (PB-SIB)

                def pushforward(state: jax.Array) -> jax.Array:
                    return pushforward_shared_h_lib_s(
                        state,
                        layer1_cholesky,
                        self._layer2_amplitude,
                        self._layer2_lengthscale,
                        jitter_const,
                        G,
                        chunk_size_local,
                        self.n_l1_types,
                    )
            else:  # n_l3_types == 3, M13a LIB-B (PB-BAT)
                ramp_w_local = self.ramp_w

                def pushforward(state: jax.Array) -> jax.Array:
                    return pushforward_shared_h_battery(
                        state,
                        layer1_cholesky,
                        self._layer2_amplitude,
                        self._layer2_lengthscale,
                        jitter_const,
                        G,
                        chunk_size_local,
                        self.n_l1_types,
                        ramp_w_local,
                    )
        elif self.n_l1_types == 2 and self.n_l2_types == 2:
            # M8 byte-reproducible path: call the ORIGINAL, untouched
            # pushforward_shared_h exactly as before -- self-test 1's
            # regression check depends on this function never changing
            # behaviour or call signature.
            def pushforward(state: jax.Array) -> jax.Array:
                return pushforward_shared_h(
                    state,
                    layer1_cholesky,
                    self._layer2_amplitude,
                    self._layer2_lengthscale,
                    jitter_const,
                    G,
                    chunk_size_local,
                )
        elif self.n_l1_types == 2:
            # M9/M10 (and any future sharing-degree sweep): general
            # pushforward -- see pushforward_shared_h_general's docstring
            # for why this is a DIFFERENT function, not something reachable
            # from the n_l2_types==2 path above with a different argument.
            # ``self.protocol_order`` selects M9's "l2_major" (default) or
            # M10's "l1_major" convention.
            def pushforward(state: jax.Array) -> jax.Array:
                return pushforward_shared_h_general(
                    state,
                    layer1_cholesky,
                    self._layer2_amplitude,
                    self._layer2_lengthscale,
                    jitter_const,
                    G,
                    chunk_size_local,
                    self.n_l2_types,
                    self.protocol_order,
                )
        else:
            # M11: arbitrary n_l1_types (PB-L2W/PB-L2N). Protocol order is
            # always the l1-major convention generalised to n_l1_types (see
            # pushforward_shared_h_general_nl1's docstring); no other
            # protocol_order is meaningful here.
            if self.protocol_order != "l1_major":
                raise ValueError(
                    f"protocol_order={self.protocol_order!r} must be 'l1_major' "
                    f"when n_l1_types={self.n_l1_types} != 2"
                )

            def pushforward(state: jax.Array) -> jax.Array:
                return pushforward_shared_h_general_nl1(
                    state,
                    layer1_cholesky,
                    self._layer2_amplitude,
                    self._layer2_lengthscale,
                    jitter_const,
                    G,
                    chunk_size_local,
                    self.n_l1_types,
                    self.n_l2_types,
                )

        self._pushforward = pushforward

        # The pooled RB terminal tail's (P, 256, 40)/(P, 40, 256) tensors are
        # ~8x larger per particle than anything U11/U14 ever ran unchunked
        # (n_candidates<=64, n_max<=20 there); at the driver's default
        # particles=65536/gpus-per-worker=1 (local_particles=65536) a single
        # such buffer is ~5 GiB, an OOM risk on a 16 GiB V100 with several
        # live at once. Chunk it via the same jax.lax.map pattern as the
        # pushforward above -- sized conservatively using the LARGER of the
        # two tensor dimensions (n_candidates, n_max) as compute_chunk_size's
        # square-tensor assumption, which upper-bounds the true rectangular
        # footprint.
        def initialise_local(key: jax.Array) -> jax.Array:
            return jax.random.normal(
                key, (self.local_particles, self.grid_size), dtype=jnp.float64
            )

        # M10f Change A additive: whether the "segment-blocked" kernel below
        # (the only one restructured so far -- see this constructor's
        # ``l2_chol_reuse`` parameter docstring) actually gets the layer-2-
        # Cholesky-reuse fast path. Restricted to the EXACT dispatch that
        # calls ``pushforward_shared_h_general`` above (``n_l3_types==0``,
        # ``n_l1_types==2``, ``n_l2_types!=2``) -- the production PB-W16/
        # PB-WIND16 config (``n_l2_types=8``, ``protocol_order="l1_major"``)
        # is exactly this case. M8's ``n_l2_types==2`` (separate, not
        # shared, B/C Cholesky factors -- see ``pushforward_shared_h``'s
        # docstring), M11's ``n_l1_types!=2``, and M12/M13's
        # ``n_l3_types>=1`` all fall back to the byte-unchanged kernel below
        # regardless of ``l2_chol_reuse`` (documented in the final report,
        # not implemented here -- would need an analogous but separately
        # written reuse path for each dispatch).
        use_chol_reuse = (
            self.l2_chol_reuse
            and self.n_l3_types == 0
            and self.n_l1_types == 2
            and self.n_l2_types != 2
            and pcn_mode == "segment-blocked"
        )
        # S2 additive: "segment-blocked-obs" (observed-row move loop) needs the
        # same layer-2 Cholesky builder / candidate-slice table as the Change A
        # reuse path, and is only defined for the same dispatch.
        use_obs_kernel = pcn_mode == "segment-blocked-obs"
        if use_obs_kernel and not (
            self.n_l3_types == 0 and self.n_l1_types == 2 and self.n_l2_types != 2
        ):
            raise ValueError(
                "pcn_mode='segment-blocked-obs' requires n_l3_types==0, n_l1_types==2, "
                f"n_l2_types!=2; got ({self.n_l3_types}, {self.n_l1_types}, {self.n_l2_types})"
            )
        if use_chol_reuse or use_obs_kernel:
            # Layer-2 covariance/Cholesky builder, factored verbatim out of
            # ``pushforward_shared_h_general``'s ``chunk_fn`` (SAME
            # expression order, SAME chunking via ``jax.lax.map`` at
            # ``chunk_size_local`` -- see that function's docstring): depends
            # ONLY on ``state``'s layer-1 columns (``xi_A``, ``xi_D``, the
            # first ``2*G`` columns), which are FIXED once the sweep's two
            # layer-1 blocks have been proposed/accepted, so this is called
            # exactly ONCE per sweep (not once per layer-2 block) below.
            layer1_chol_T_l2 = layer1_cholesky.T
            eye2g_l2 = jnp.eye(2 * G, dtype=jnp.float64)
            sf2_l2 = self._layer2_amplitude**2
            ell_l2 = self._layer2_lengthscale

            def l2_cholesky_fn(state_full: jax.Array) -> jax.Array:
                def chunk_fn(state_chunk: jax.Array) -> jax.Array:
                    xi_A = state_chunk[:, 0:G]
                    xi_D = state_chunk[:, G : 2 * G]
                    h_A = xi_A @ layer1_chol_T_l2
                    h_D = xi_D @ layer1_chol_T_l2
                    induced = jnp.concatenate([h_A, h_D], axis=1)
                    difference = induced[:, :, None] - induced[:, None, :]
                    covariance = (
                        sf2_l2 * jnp.exp(-0.5 * (difference / ell_l2) ** 2)
                        + jitter_const * eye2g_l2[None, :, :]
                    )
                    return jnp.linalg.cholesky(covariance)

                total_particles = state_full.shape[0]
                if chunk_size_local >= total_particles:
                    return chunk_fn(state_full)
                n_chunks = total_particles // chunk_size_local
                chunks = state_full.reshape(n_chunks, chunk_size_local, state_full.shape[-1])
                chol_chunks = jax.lax.map(chunk_fn, chunks)
                return chol_chunks.reshape(total_particles, 2 * G, 2 * G)

            # Per-layer-2-type (j) candidate-column slices that
            # ``pushforward_shared_h_general``'s A-part/D-part concatenation
            # writes ``h_j``'s two G-wide halves into -- read off that
            # function's ``chunk_fn`` return statement for each
            # ``protocol_order`` branch (verified against it, not assumed):
            # ``"l1_major"``: ``concat(a_parts + d_parts)`` with
            # ``a_parts[j]=h_j[:,0:G]``, ``d_parts[j]=h_j[:,G:2G]`` ->
            # candidate columns ``j*G:(j+1)*G`` (A) and
            # ``(n_l2_types+j)*G:(n_l2_types+j+1)*G`` (D). ``"l2_major"``:
            # ``concat(parts)`` with ``parts=[h_0_A,h_0_D,h_1_A,h_1_D,...]``
            # -> candidate columns ``(2j)*G:(2j+1)*G`` (A) and
            # ``(2j+1)*G:(2j+2)*G`` (D).
            l2_candidate_slices = []
            for j in range(self.n_l2_types):
                if self.protocol_order == "l1_major":
                    a_start = j * G
                    d_start = (self.n_l2_types + j) * G
                else:
                    a_start = 2 * j * G
                    d_start = a_start + G
                l2_candidate_slices.append((a_start, a_start + G, d_start, d_start + G))
            l2_candidate_slices = tuple(l2_candidate_slices)

        if use_obs_kernel:
            # S2 additive: candidate -> (layer-2 type j, row in the (2G x 2G)
            # layer-2 Cholesky) tables, read off ``l2_candidate_slices`` above
            # (the A half of type j is chol rows 0:G, the D half rows G:2G).
            # Observation o at flat candidate index c has
            # h[:, c] = sum_k chol[:, row_tab[c], k] * xi_{jtype_tab[c]}[:, k].
            _jt = np.full(self.n_candidates, -1, dtype=np.int32)
            _rw = np.full(self.n_candidates, -1, dtype=np.int32)
            for _j, (_a0, _a1, _d0, _d1) in enumerate(l2_candidate_slices):
                _jt[_a0:_a1] = _j
                _rw[_a0:_a1] = np.arange(G)
                _jt[_d0:_d1] = _j
                _rw[_d0:_d1] = G + np.arange(G)
            if (_jt < 0).any():
                raise RuntimeError("l2_candidate_slices do not cover every candidate")
            jtype_table = jnp.asarray(_jt)
            self._jtype_np = _jt  # S6-LR: per-type counts on the host (cap selection)
            row_table = jnp.asarray(_rw)
            n_l2_obs = self.n_l2_types

            def l1_obs_fn(state_full: jax.Array, jt: jax.Array, rw: jax.Array) -> jax.Array:
                """h[:, obs] (P, n_max) from the layer-1 columns and ALL layer-2
                xi blocks: per chunk, factorise the (2G,2G) layer-2 covariance,
                form h_j = chol @ xi_j for every j in one batched einsum, then
                gather the observed (type, row) pairs. The chunk's Cholesky is
                never materialised for the full particle set."""

                def chunk_fn(state_chunk: jax.Array) -> jax.Array:
                    xi_A = state_chunk[:, 0:G]
                    xi_D = state_chunk[:, G : 2 * G]
                    h_A = xi_A @ layer1_chol_T_l2
                    h_D = xi_D @ layer1_chol_T_l2
                    induced = jnp.concatenate([h_A, h_D], axis=1)
                    difference = induced[:, :, None] - induced[:, None, :]
                    covariance = (
                        sf2_l2 * jnp.exp(-0.5 * (difference / ell_l2) ** 2)
                        + jitter_const * eye2g_l2[None, :, :]
                    )
                    chol_c = jnp.linalg.cholesky(covariance)
                    xi_l2 = state_chunk[:, 2 * G :].reshape(
                        state_chunk.shape[0], n_l2_obs, 2 * G
                    )
                    h_all = jnp.einsum("pik,pjk->pji", chol_c, xi_l2)
                    ar = jnp.arange(state_chunk.shape[0])[:, None]
                    return h_all[ar, jt[None, :], rw[None, :]]

                total_particles = state_full.shape[0]
                if chunk_size_local >= total_particles:
                    return chunk_fn(state_full)
                n_chunks = total_particles // chunk_size_local
                chunks = state_full.reshape(n_chunks, chunk_size_local, state_full.shape[-1])
                out = jax.lax.map(chunk_fn, chunks)
                return out.reshape(total_particles, out.shape[-1])

            # S6-LR additive: per-L2-type ("typed") observation layout. For each type
            # j the observed rows of that type are packed into ``cap`` static slots
            # (``type_tables``); ``l1_typed_fn`` returns h in that layout (P, T, cap).
            def type_tables(jt, rw, active, cap, unique=False):
                """(idx (T,cap) int32 positions into the padded obs, valid (T,cap) float,
                rw_t (T,cap) chol rows). Slots beyond a type's count have valid 0 and
                idx 0 (callers must mask with valid). ``unique=True`` (2026-11-01 additive,
                repeats only) keeps only the FIRST active observation of each (type, row) cell,
                so a type holds at most 2G rows whatever the number of repeats; used by the
                null-space refresh, which constrains each cell once (duplicates are masked
                there anyway)."""
                sel = (jt[None, :] == jnp.arange(n_l2_obs)[:, None]) & (active[None, :] > 0.5)
                if unique:
                    n_pad = jt.shape[0]
                    same_cell_o = (jt[:, None] == jt[None, :]) & (rw[:, None] == rw[None, :])
                    tril_o = jnp.tril(jnp.ones((n_pad, n_pad), dtype=bool), k=-1)
                    dup_o = jnp.any(same_cell_o & tril_o & (active[None, :] > 0.5), axis=1)
                    sel = sel & ~dup_o[None, :]
                idx = jax.vmap(lambda m_: jnp.nonzero(m_, size=cap, fill_value=0)[0])(sel).astype(
                    jnp.int32
                )
                cnt = jnp.sum(sel, axis=1)
                valid = (jnp.arange(cap)[None, :] < cnt[:, None]).astype(jnp.float64)
                return idx, valid, rw[idx]

            def l1_typed_fn(state_full: jax.Array, rw_t: jax.Array) -> jax.Array:
                """Same arithmetic as ``l1_obs_fn``; output in the typed layout (P, T, cap)."""

                def chunk_fn(state_chunk: jax.Array) -> jax.Array:
                    xi_A = state_chunk[:, 0:G]
                    xi_D = state_chunk[:, G : 2 * G]
                    h_A = xi_A @ layer1_chol_T_l2
                    h_D = xi_D @ layer1_chol_T_l2
                    induced = jnp.concatenate([h_A, h_D], axis=1)
                    difference = induced[:, :, None] - induced[:, None, :]
                    covariance = (
                        sf2_l2 * jnp.exp(-0.5 * (difference / ell_l2) ** 2)
                        + jitter_const * eye2g_l2[None, :, :]
                    )
                    chol_c = jnp.linalg.cholesky(covariance)
                    xi_l2 = state_chunk[:, 2 * G :].reshape(
                        state_chunk.shape[0], n_l2_obs, 2 * G
                    )
                    h_all = jnp.einsum("pik,pjk->pji", chol_c, xi_l2)  # (c, T, 2G)
                    return jnp.take_along_axis(h_all, rw_t[None, :, :], axis=2)

                total_particles = state_full.shape[0]
                if chunk_size_local >= total_particles:
                    return chunk_fn(state_full)
                n_chunks = total_particles // chunk_size_local
                chunks = state_full.reshape(n_chunks, chunk_size_local, state_full.shape[-1])
                out = jax.lax.map(chunk_fn, chunks)
                return out.reshape(total_particles, *out.shape[2:])

        # M10f Change B additive: per-``n_max``-bucket kernel bundle builder
        # -- everything ABOVE this point (pushforward, chunk sizing, block
        # bounds, the Change A reuse ingredients) is independent of
        # ``n_max`` and built exactly ONCE; everything the terminal RB tail
        # touches (``build_terminal_rb_functions``'s three closures and the
        # pCN/loglik/moments/summary_extended pmaps that call them) is
        # ``n_max``-shaped and therefore built per-bucket, lazily, the first
        # time that bucket is actually used (see ``self._bundle`` below).
        # Every closure/pmap defined inside is byte-identical in STRUCTURE
        # to the pre-M10f single-bucket code -- only ``n_max`` itself
        # (closed over via this function's own argument, replacing the old
        # ``self.n_max``) varies across buckets.
        def _build_bundle(n_max: int) -> dict:
            rb_chunk_size_local = compute_chunk_size(
                self.local_particles, max(self.n_candidates, n_max), chunk_target_bytes
            )
            # M10f: the pCN hot path's ``loglik_from_h`` holds only
            # ``(chunk, n_max, n_max)`` tensors, so size its chunks on
            # ``n_max`` alone: with the 256 MiB target and 16384 local
            # particles this is unchunked at n_max=40 (byte-identical to the
            # pre-M10f call) and 8 chunks of 2048 at n_max=100.
            loglik_chunk_size_local = compute_chunk_size(
                self.local_particles, n_max, chunk_target_bytes
            )

            loglik_from_h, moments_from_h, summary_extended_from_h = build_terminal_rb_functions(
                n_candidates=self.n_candidates,
                n_max=n_max,
                amplitude=library_spec.amplitude,
                lengthscale=library_spec.terminal_lengthscale,
                noise_variance=noise_variance,
                jitter=jitter_const,
                chunk_size=rb_chunk_size_local,
                loglik_chunk_size=loglik_chunk_size_local,
            )

            def loglik_local(state, obs_indices, y_padded, active):
                return loglik_from_h(pushforward(state), obs_indices, y_padded, active)

            lr_pcn_get = None  # S6-LR: cap -> pmapped low-rank pCN kernel (terminal_lowrank > 0)
            lr_r_value = 0

            if pcn_mode == "segment":
                # M7c "segment" kernel: same joint (whole-384-dim) proposal as
                # "fixed"/"adaptive" below, but pcn_steps is split into
                # n_pcn_segments consecutive segments of pcn_segment_length
                # steps each; beta is frozen within a segment and adapted
                # between segments from that segment's POPULATION-mean
                # acceptance (psum'd across devices -- module docstring's "M7c
                # remedy" paragraph). Implemented as an OUTER lax.scan over
                # segments wrapping an INNER lax.scan over the segment's steps
                # (rather than the fori_loop used by the pre-M7c kernels) so
                # both the per-segment beta USED and the per-segment acceptance
                # MEASURED can be collected as scan outputs for
                # self-test/diagnostic use (SegmentRejuvenationInfo.beta_trace/
                # acceptance_trace).
                n_segments = self.n_pcn_segments
                segment_length = self.pcn_segment_length

                def pcn_local(state, loglik, key, obs_indices, y_padded, active, beta0):
                    def pcn_step(carry, _):
                        current, current_loglik, state_key, beta = carry
                        shrink = jnp.sqrt(1.0 - beta * beta)
                        state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                        white = jax.random.normal(normal_key, current.shape, dtype=jnp.float64)
                        proposal = shrink * current + beta * white
                        proposal_loglik = loglik_local(proposal, obs_indices, y_padded, active)
                        accepted = jnp.log(
                            jax.random.uniform(uniform_key, (current.shape[0],), dtype=jnp.float64)
                        ) < (proposal_loglik - current_loglik)
                        current = jnp.where(accepted[:, None], proposal, current)
                        current_loglik = jnp.where(accepted, proposal_loglik, current_loglik)
                        return (current, current_loglik, state_key, beta), jnp.sum(accepted)

                    def segment_body(carry, _):
                        current, current_loglik, state_key, beta = carry
                        (current, current_loglik, state_key, _), accepted_per_step = jax.lax.scan(
                            pcn_step,
                            (current, current_loglik, state_key, beta),
                            None,
                            length=segment_length,
                        )
                        local_accepted = jnp.sum(accepted_per_step)
                        global_accepted = jax.lax.psum(local_accepted, axis_name="device")
                        seg_acceptance = global_accepted / (segment_length * self.n_particles)
                        new_beta = _segment_beta_update(beta, seg_acceptance) if self.adapt_beta else beta
                        return (current, current_loglik, state_key, new_beta), (beta, seg_acceptance)

                    (state, loglik, _, beta_final), (beta_trace, acceptance_trace) = jax.lax.scan(
                        segment_body, (state, loglik, key, beta0), None, length=n_segments
                    )
                    return state, loglik, beta_final, beta_trace, acceptance_trace
            elif pcn_mode == "segment-blocked":
                # M7c "segment-blocked" kernel: same per-type-block structure as
                # "adaptive-blocked" below, wrapped in the same outer/inner
                # lax.scan-over-segments structure as "segment" above, with a
                # length-N_PCN_BLOCKS beta_blocks state and per-block
                # population-mean segment acceptance.
                n_segments = self.n_pcn_segments
                segment_length = self.pcn_segment_length
                # M9 additive: generalised from the pre-M9 literal 4-tuple to
                # ``self.n_pcn_blocks`` blocks (identical values at
                # n_l2_types=2 -- see shared_pcn_block_bounds's docstring).
                block_bounds = (
                    shared_pcn_block_bounds_m12(G, self.n_l1_types)
                    if self.n_l3_types >= 1
                    else shared_pcn_block_bounds(G, self.n_l2_types, self.n_l1_types)
                )
                n_blocks = len(block_bounds)

                if use_chol_reuse:
                    # M10f Change A: layer-1 blocks (k < n_l1_types) are
                    # evaluated exactly as below (full pushforward+loglik per
                    # block, via ``pushforward``/``loglik_from_h`` directly
                    # rather than the ``loglik_local`` convenience wrapper, so
                    # the freshly-computed ``h_prop`` can be carried forward as
                    # ``current_h``). AFTER the layer-1 blocks, the layer-2
                    # Cholesky is factorised ONCE (``l2_cholesky_fn``, which
                    # depends only on the now-fixed layer-1 columns) and reused
                    # for every one of the ``n_l2_types`` layer-2 block
                    # proposals, each of which only needs one ``einsum`` (no
                    # Cholesky) plus a scatter into a copy of ``current_h`` at
                    # that type's own candidate columns
                    # (``l2_candidate_slices``) before calling
                    # ``loglik_from_h`` on the MODIFIED h. Block order and the
                    # per-block ``jax.random.split`` sequence are UNCHANGED
                    # (still ``k=0..n_blocks-1`` in ``block_bounds`` order), so
                    # this kernel draws the SAME proposals as the kernel below
                    # given the same key -- see this module's Change A
                    # equivalence self-test.
                    n_l1_types_local = self.n_l1_types

                    def pcn_local(state, loglik, key, obs_indices, y_padded, active, beta_blocks0):
                        h0 = pushforward(state)

                        def pcn_sweep(carry, _):
                            current, current_h, current_loglik, state_key, beta_blocks = carry
                            accepted_this_sweep = jnp.zeros(n_blocks, dtype=jnp.float64)
                            for k in range(n_l1_types_local):
                                start, end = block_bounds[k]
                                state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                                block_current = current[:, start:end]
                                white = jax.random.normal(
                                    normal_key, block_current.shape, dtype=jnp.float64
                                )
                                shrink_k = jnp.sqrt(1.0 - beta_blocks[k] * beta_blocks[k])
                                block_proposal = shrink_k * block_current + beta_blocks[k] * white
                                proposal = current.at[:, start:end].set(block_proposal)
                                h_prop = pushforward(proposal)
                                proposal_loglik = loglik_from_h(h_prop, obs_indices, y_padded, active)
                                accepted = jnp.log(
                                    jax.random.uniform(
                                        uniform_key, (current.shape[0],), dtype=jnp.float64
                                    )
                                ) < (proposal_loglik - current_loglik)
                                block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                                current = current.at[:, start:end].set(block_new)
                                current_h = jnp.where(accepted[:, None], h_prop, current_h)
                                current_loglik = jnp.where(accepted, proposal_loglik, current_loglik)
                                accepted_this_sweep = accepted_this_sweep.at[k].add(jnp.sum(accepted))

                            chol = l2_cholesky_fn(current)
                            for j in range(self.n_l2_types):
                                k = n_l1_types_local + j
                                start, end = block_bounds[k]
                                state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                                block_current = current[:, start:end]
                                white = jax.random.normal(
                                    normal_key, block_current.shape, dtype=jnp.float64
                                )
                                shrink_k = jnp.sqrt(1.0 - beta_blocks[k] * beta_blocks[k])
                                block_proposal = shrink_k * block_current + beta_blocks[k] * white
                                h_j = jnp.einsum("pij,pj->pi", chol, block_proposal)
                                a_start, a_end, d_start, d_end = l2_candidate_slices[j]
                                h_prop = current_h.at[:, a_start:a_end].set(h_j[:, 0:G])
                                h_prop = h_prop.at[:, d_start:d_end].set(h_j[:, G : 2 * G])
                                proposal_loglik = loglik_from_h(h_prop, obs_indices, y_padded, active)
                                accepted = jnp.log(
                                    jax.random.uniform(
                                        uniform_key, (current.shape[0],), dtype=jnp.float64
                                    )
                                ) < (proposal_loglik - current_loglik)
                                block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                                current = current.at[:, start:end].set(block_new)
                                current_h = jnp.where(accepted[:, None], h_prop, current_h)
                                current_loglik = jnp.where(accepted, proposal_loglik, current_loglik)
                                accepted_this_sweep = accepted_this_sweep.at[k].add(jnp.sum(accepted))
                            return (
                                current,
                                current_h,
                                current_loglik,
                                state_key,
                                beta_blocks,
                            ), accepted_this_sweep

                        def segment_body(carry, _):
                            current, current_h, current_loglik, state_key, beta_blocks = carry
                            (
                                (current, current_h, current_loglik, state_key, _),
                                accepted_per_sweep,
                            ) = jax.lax.scan(
                                pcn_sweep,
                                (current, current_h, current_loglik, state_key, beta_blocks),
                                None,
                                length=segment_length,
                            )
                            local_accepted_blocks = jnp.sum(accepted_per_sweep, axis=0)
                            global_accepted_blocks = jax.lax.psum(
                                local_accepted_blocks, axis_name="device"
                            )
                            seg_acceptance_blocks = global_accepted_blocks / (
                                segment_length * self.n_particles
                            )
                            new_beta_blocks = _segment_beta_update(beta_blocks, seg_acceptance_blocks) if self.adapt_beta else beta_blocks
                            return (
                                (current, current_h, current_loglik, state_key, new_beta_blocks),
                                (beta_blocks, seg_acceptance_blocks),
                            )

                        (
                            (state, _h_final, loglik, _, beta_blocks_final),
                            (beta_blocks_trace, acceptance_blocks_trace),
                        ) = jax.lax.scan(
                            segment_body,
                            (state, h0, loglik, key, beta_blocks0),
                            None,
                            length=n_segments,
                        )
                        return (
                            state,
                            loglik,
                            beta_blocks_final,
                            beta_blocks_trace,
                            acceptance_blocks_trace,
                        )
                else:

                    def pcn_local(state, loglik, key, obs_indices, y_padded, active, beta_blocks0):
                        def pcn_sweep(carry, _):
                            current, current_loglik, state_key, beta_blocks = carry
                            accepted_this_sweep = jnp.zeros(n_blocks, dtype=jnp.float64)
                            for k, (start, end) in enumerate(block_bounds):
                                state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                                block_current = current[:, start:end]
                                white = jax.random.normal(
                                    normal_key, block_current.shape, dtype=jnp.float64
                                )
                                shrink_k = jnp.sqrt(1.0 - beta_blocks[k] * beta_blocks[k])
                                block_proposal = shrink_k * block_current + beta_blocks[k] * white
                                proposal = current.at[:, start:end].set(block_proposal)
                                proposal_loglik = loglik_local(proposal, obs_indices, y_padded, active)
                                accepted = jnp.log(
                                    jax.random.uniform(
                                        uniform_key, (current.shape[0],), dtype=jnp.float64
                                    )
                                ) < (proposal_loglik - current_loglik)
                                block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                                current = current.at[:, start:end].set(block_new)
                                current_loglik = jnp.where(accepted, proposal_loglik, current_loglik)
                                accepted_this_sweep = accepted_this_sweep.at[k].add(jnp.sum(accepted))
                            return (current, current_loglik, state_key, beta_blocks), accepted_this_sweep

                        def segment_body(carry, _):
                            current, current_loglik, state_key, beta_blocks = carry
                            (
                                (current, current_loglik, state_key, _),
                                accepted_per_sweep,
                            ) = jax.lax.scan(
                                pcn_sweep,
                                (current, current_loglik, state_key, beta_blocks),
                                None,
                                length=segment_length,
                            )
                            local_accepted_blocks = jnp.sum(accepted_per_sweep, axis=0)
                            global_accepted_blocks = jax.lax.psum(
                                local_accepted_blocks, axis_name="device"
                            )
                            seg_acceptance_blocks = global_accepted_blocks / (
                                segment_length * self.n_particles
                            )
                            new_beta_blocks = _segment_beta_update(beta_blocks, seg_acceptance_blocks) if self.adapt_beta else beta_blocks
                            return (
                                (current, current_loglik, state_key, new_beta_blocks),
                                (beta_blocks, seg_acceptance_blocks),
                            )

                        (
                            (state, loglik, _, beta_blocks_final),
                            (beta_blocks_trace, acceptance_blocks_trace),
                        ) = jax.lax.scan(
                            segment_body, (state, loglik, key, beta_blocks0), None, length=n_segments
                        )
                        return state, loglik, beta_blocks_final, beta_blocks_trace, acceptance_blocks_trace
            elif pcn_mode == "segment-blocked-obs":
                # S2 "segment-blocked-obs" kernel: same blocks, same random-key
                # sequence and same acceptance rule as the Change A
                # "segment-blocked" kernel above, but the loop carries only
                # h_obs = h[:, obs_indices] (P, n_max) instead of the full
                # (P, n_candidates) h. Layer-1 block: proposal -> chol of the
                # layer-2 covariance -> observed rows only (l1_obs_fn).
                # Layer-2 block j: h_j = chol @ xi_j_proposal, written only into
                # the h_obs columns whose observation has layer-2 type j. A layer-2
                # type with no active observation leaves h_obs (hence the
                # likelihood) unchanged: its loglik evaluation is skipped via
                # lax.cond and the proposal is accepted unconditionally (same
                # outcome as the "segment-blocked" test log U < 0).
                n_segments = self.n_pcn_segments
                segment_length = self.pcn_segment_length
                block_bounds = shared_pcn_block_bounds(G, self.n_l2_types, self.n_l1_types)
                n_blocks = len(block_bounds)
                n_l1_types_local = self.n_l1_types
                obs_target = (
                    2 * chunk_target_bytes
                    if obs_chunk_target_bytes is None
                    else int(obs_chunk_target_bytes)
                )
                # covariance + Cholesky = two (chunk, n_max, n_max) fp64 buffers
                obs_ll_chunk = compute_chunk_size(self.local_particles, n_max, obs_target // 2)
                loglik_from_hobs = build_loglik_from_hobs(
                    n_max,
                    library_spec.amplitude,
                    library_spec.terminal_lengthscale,
                    noise_variance,
                    jitter_const,
                    obs_ll_chunk,
                )

                l2_null_k = self.l2_null_refresh
                n_l2_loc = self.n_l2_types
                eye_nmax = jnp.eye(n_max, dtype=jnp.float64)
                tril_strict = jnp.tril(jnp.ones((n_max, n_max), dtype=bool), k=-1)

                def refresh_prepare(jt, rw, active):
                    """Observation structure: duplicates of the same (type, row)
                    constrain once; inactive rows contribute an identity Gram
                    block and zero right-hand side."""
                    same_cell = (jt[:, None] == jt[None, :]) & (rw[:, None] == rw[None, :])
                    act_b = active > 0.5
                    dup_earlier = jnp.any(same_cell & tril_strict & act_b[None, :], axis=1)
                    uact = (act_b & ~dup_earlier).astype(jnp.float64)
                    same_type = (jt[:, None] == jt[None, :]).astype(jnp.float64)
                    same_row = (rw[:, None] == rw[None, :]).astype(jnp.float64)
                    type_onehot = (
                        jt[:, None] == jnp.arange(n_l2_loc)[None, :]
                    ).astype(jnp.float64)
                    return uact, same_type, same_row, type_onehot

                def refresh_chunk(chol_c, st_c, eta_c, rw, uact, same_type, same_row, type_onehot, jt):
                    # chol_c (p,2G,2G); st_c (p, dim) full state chunk; eta_c (p, n_l2, 2G)
                    p_c = chol_c.shape[0]
                    xi_c = st_c[:, 2 * G :].reshape(p_c, n_l2_loc, 2 * G)
                    c_rows = chol_c[:, rw, :]  # (p, n_max, 2G): observed rows of A
                    d = xi_c - eta_c
                    d_obs = d[:, jt, :]  # xi - eta of each obs's type
                    u = jnp.sum(c_rows * d_obs, axis=-1) * uact[None, :]
                    # Gram = A A^T = covariance[rows, rows] (+1e-10 on active rows),
                    # from the kernel formula (O(n^2) per particle).
                    h_A = st_c[:, 0:G] @ layer1_chol_T_l2
                    h_D = st_c[:, G : 2 * G] @ layer1_chol_T_l2
                    induced_o = jnp.concatenate([h_A, h_D], axis=1)[:, rw]  # (p, n_max)
                    diff = induced_o[:, :, None] - induced_o[:, None, :]
                    cov = sf2_l2 * jnp.exp(-0.5 * (diff / ell_l2) ** 2) + jitter_const * same_row[None]
                    pair = uact[:, None] * uact[None, :] * same_type
                    gram = cov * pair[None] + eye_nmax[None] * (
                        (1.0 - uact) + 1e-10 * uact
                    )[None, :, None]
                    lc = jnp.linalg.cholesky(gram)
                    w = jax.scipy.linalg.cho_solve((lc, True), u[..., None])[..., 0]
                    # The (jittered) Gram is ill-conditioned (smooth kernel): the
                    # iterative-refinement steps against the exact rows A bring
                    # A (xi' - xi) to roundoff. s = A^T w is O(|d|), so the
                    # residual u - A s is evaluated without cancellation.
                    for _it in range(REFINE_STEPS + 1):
                        s_t = jnp.einsum("ot,pok->ptk", type_onehot, w[:, :, None] * c_rows)
                        if _it == REFINE_STEPS:
                            break
                        s_obs = s_t[:, jt, :]
                        r = (u - jnp.sum(c_rows * s_obs, axis=-1)) * uact[None, :]
                        w = w + jax.scipy.linalg.cho_solve((lc, True), r[..., None])[..., 0]
                    return eta_c + s_t

                def l2_null_refresh_fn(current, chol, key, jt, rw, active):
                    uact, same_type, same_row, type_onehot = refresh_prepare(jt, rw, active)
                    n_p = current.shape[0]
                    eta = jax.random.normal(
                        key, (n_p, n_l2_loc, 2 * G), dtype=jnp.float64
                    )
                    if obs_ll_chunk >= n_p:
                        new_xi = refresh_chunk(chol, current, eta, rw, uact, same_type, same_row, type_onehot, jt)
                    else:
                        nc = n_p // obs_ll_chunk
                        resh = lambda a_: a_.reshape(nc, obs_ll_chunk, *a_.shape[1:])
                        new_xi = jax.lax.map(
                            lambda t: refresh_chunk(t[0], t[1], t[2], rw, uact, same_type, same_row, type_onehot, jt),
                            (resh(chol), resh(current), resh(eta)),
                        ).reshape(n_p, n_l2_loc, 2 * G)
                    return current.at[:, 2 * G :].set(new_xi.reshape(n_p, n_l2_loc * 2 * G))

                if l2_null_k > 0:
                    # test hook: one refresh on a given state (single device)
                    def _refresh_once(state, key, obs_indices, active):
                        jt_ = jtype_table[obs_indices]
                        rw_ = row_table[obs_indices]
                        return l2_null_refresh_fn(state, l2_cholesky_fn(state), key, jt_, rw_, active)

                    self._refresh_once_cache[n_max] = jax.jit(_refresh_once)

                # S6-LR additive: per-L2-type refresh. The Gram is block diagonal by type,
                # so each type's (cap x cap) block is solved separately (cap <= 2G = 128)
                # instead of one n_max x n_max masked Gram. Same formulas / same eta draw
                # as ``l2_null_refresh_fn``; equal up to roundoff.
                cap_ex = min(n_max, 2 * G)
                # repeats: with n_max > 2G a type can hold more observations than cap_ex while
                # still having <= 2G distinct cells -> the refresh uses deduplicated tables.
                uniq_ex = self.allow_repeat_observations and n_max > 2 * G

                def refresh_typed_prepare(rw_t, valid, cap):
                    tril_c = jnp.tril(jnp.ones((cap, cap), dtype=bool), k=-1)
                    same_row_t = rw_t[:, :, None] == rw_t[:, None, :]
                    dup_earlier = jnp.any(
                        same_row_t & tril_c[None] & (valid[:, None, :] > 0.5), axis=2
                    )
                    uact = (valid > 0.5) & ~dup_earlier
                    return same_row_t.astype(jnp.float64), uact.astype(jnp.float64)

                def refresh_chunk_typed(chol_c, st_c, eta_c, rw_t, same_row_t, uact, cap):
                    p_c = chol_c.shape[0]
                    xi_c = st_c[:, 2 * G :].reshape(p_c, n_l2_loc, 2 * G)
                    a_rows = chol_c[:, rw_t, :]  # (p, T, cap, 2G)
                    d = xi_c - eta_c
                    u = jnp.einsum("ptck,ptk->ptc", a_rows, d) * uact[None]
                    h_A = st_c[:, 0:G] @ layer1_chol_T_l2
                    h_D = st_c[:, G : 2 * G] @ layer1_chol_T_l2
                    induced_o = jnp.concatenate([h_A, h_D], axis=1)[:, rw_t]  # (p, T, cap)
                    diff = induced_o[..., :, None] - induced_o[..., None, :]
                    cov = sf2_l2 * jnp.exp(-0.5 * (diff / ell_l2) ** 2) + jitter_const * same_row_t[None]
                    pair = uact[:, :, None] * uact[:, None, :]
                    eye_c = jnp.eye(cap, dtype=jnp.float64)
                    gram = cov * pair[None] + eye_c[None, None] * (
                        (1.0 - uact) + 1e-10 * uact
                    )[None, :, :, None]
                    lc = jnp.linalg.cholesky(gram.reshape(p_c * n_l2_loc, cap, cap))

                    def solve(rhs):
                        out = jax.scipy.linalg.cho_solve(
                            (lc, True), rhs.reshape(p_c * n_l2_loc, cap, 1)
                        )[..., 0]
                        return out.reshape(p_c, n_l2_loc, cap)

                    w = solve(u)
                    for _it in range(REFINE_STEPS + 1):
                        s_t = jnp.einsum("ptc,ptck->ptk", w, a_rows)
                        if _it == REFINE_STEPS:
                            break
                        r = (u - jnp.einsum("ptck,ptk->ptc", a_rows, s_t)) * uact[None]
                        w = w + solve(r)
                    return eta_c + s_t

                def l2_null_refresh_typed_fn(current, chol, key, rw_t, valid, cap):
                    same_row_t, uact = refresh_typed_prepare(rw_t, valid, cap)
                    n_p = current.shape[0]
                    eta = jax.random.normal(key, (n_p, n_l2_loc, 2 * G), dtype=jnp.float64)
                    bytes_pp = 8 * n_l2_loc * cap * (2 * G + 3 * cap)
                    side = int(np.ceil(np.sqrt(bytes_pp / 8.0)))
                    ch = compute_chunk_size(n_p, side, self.lr_chunk_target_bytes)
                    if ch >= n_p:
                        new_xi = refresh_chunk_typed(chol, current, eta, rw_t, same_row_t, uact, cap)
                    else:
                        nc = n_p // ch
                        resh = lambda a_: a_.reshape(nc, ch, *a_.shape[1:])
                        new_xi = jax.lax.map(
                            lambda t: refresh_chunk_typed(t[0], t[1], t[2], rw_t, same_row_t, uact, cap),
                            (resh(chol), resh(current), resh(eta)),
                        ).reshape(n_p, n_l2_loc, 2 * G)
                    return current.at[:, 2 * G :].set(new_xi.reshape(n_p, n_l2_loc * 2 * G))

                if l2_null_k > 0:
                    # test hook: one per-type refresh on a given state (single device)
                    def _refresh_typed_once(state, key, obs_indices, active):
                        jt_ = jtype_table[obs_indices]
                        rw_ = row_table[obs_indices]
                        _, valid_, rw_t_ = type_tables(jt_, rw_, active, cap_ex, uniq_ex)
                        return l2_null_refresh_typed_fn(
                            state, l2_cholesky_fn(state), key, rw_t_, valid_, cap_ex
                        )

                    self._refresh_typed_once_cache[n_max] = jax.jit(_refresh_typed_once)

                def pcn_local(state, loglik, key, obs_indices, y_padded, active, beta_blocks0):
                    jt = jtype_table[obs_indices]
                    rw = row_table[obs_indices]
                    obs_counts = [
                        jnp.sum(active * (jt == j).astype(jnp.float64))
                        for j in range(self.n_l2_types)
                    ]
                    h_obs0 = l1_obs_fn(state, jt, rw)
                    if self.l2_refresh_per_type:  # S6-LR additive (trace-time switch)
                        _, valid_ex, rw_t_ex = type_tables(jt, rw, active, cap_ex, uniq_ex)

                    def pcn_sweep(carry, sweep_idx):
                        current, current_h, current_loglik, state_key, beta_blocks = carry
                        accepted_this_sweep = jnp.zeros(n_blocks, dtype=jnp.float64)
                        for k in range(n_l1_types_local):
                            start, end = block_bounds[k]
                            state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                            block_current = current[:, start:end]
                            white = jax.random.normal(
                                normal_key, block_current.shape, dtype=jnp.float64
                            )
                            shrink_k = jnp.sqrt(1.0 - beta_blocks[k] * beta_blocks[k])
                            block_proposal = shrink_k * block_current + beta_blocks[k] * white
                            proposal = current.at[:, start:end].set(block_proposal)
                            h_prop = l1_obs_fn(proposal, jt, rw)
                            proposal_loglik = loglik_from_hobs(h_prop, y_padded, active)
                            accepted = jnp.log(
                                jax.random.uniform(
                                    uniform_key, (current.shape[0],), dtype=jnp.float64
                                )
                            ) < (proposal_loglik - current_loglik)
                            block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                            current = current.at[:, start:end].set(block_new)
                            current_h = jnp.where(accepted[:, None], h_prop, current_h)
                            current_loglik = jnp.where(accepted, proposal_loglik, current_loglik)
                            accepted_this_sweep = accepted_this_sweep.at[k].add(jnp.sum(accepted))

                        chol = l2_cholesky_fn(current)
                        for j in range(self.n_l2_types):
                            k = n_l1_types_local + j
                            start, end = block_bounds[k]
                            state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                            block_current = current[:, start:end]
                            white = jax.random.normal(
                                normal_key, block_current.shape, dtype=jnp.float64
                            )
                            shrink_k = jnp.sqrt(1.0 - beta_blocks[k] * beta_blocks[k])
                            block_proposal = shrink_k * block_current + beta_blocks[k] * white
                            log_u = jnp.log(
                                jax.random.uniform(
                                    uniform_key, (current.shape[0],), dtype=jnp.float64
                                )
                            )

                            def unobserved_branch(h_in, ll_in, bp, lu, chol_in, j=j):
                                return h_in, ll_in, jnp.ones(h_in.shape[0], dtype=bool)

                            def observed_branch(h_in, ll_in, bp, lu, chol_in, j=j):
                                h_j = jnp.einsum("pij,pj->pi", chol_in, bp)
                                h_new = jnp.where((jt == j)[None, :], h_j[:, rw], h_in)
                                pl = loglik_from_hobs(h_new, y_padded, active)
                                acc = lu < (pl - ll_in)
                                return (
                                    jnp.where(acc[:, None], h_new, h_in),
                                    jnp.where(acc, pl, ll_in),
                                    acc,
                                )

                            current_h, current_loglik, accepted = jax.lax.cond(
                                obs_counts[j] == 0.0,
                                unobserved_branch,
                                observed_branch,
                                current_h,
                                current_loglik,
                                block_proposal,
                                log_u,
                                chol,
                            )
                            block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                            current = current.at[:, start:end].set(block_new)
                            accepted_this_sweep = accepted_this_sweep.at[k].add(jnp.sum(accepted))
                        if l2_null_k > 0:
                            # S3: exact conditional-prior refresh of the null space of
                            # each layer-2 block; chol (current L1) is unchanged by the
                            # L2 moves; h_obs / loglik are invariant. Key derived by
                            # fold_in so the main key stream is untouched.
                            current = jax.lax.cond(
                                (sweep_idx + 1) % l2_null_k == 0,
                                (lambda c: l2_null_refresh_typed_fn(
                                    c, chol, jax.random.fold_in(state_key, 0x53), rw_t_ex, valid_ex, cap_ex
                                )) if self.l2_refresh_per_type else (lambda c: l2_null_refresh_fn(
                                    c, chol, jax.random.fold_in(state_key, 0x53), jt, rw, active
                                )),
                                lambda c: c,
                                current,
                            )
                        return (
                            current,
                            current_h,
                            current_loglik,
                            state_key,
                            beta_blocks,
                        ), accepted_this_sweep

                    def segment_body(carry, _):
                        current, current_h, current_loglik, state_key, beta_blocks = carry
                        (
                            (current, current_h, current_loglik, state_key, _),
                            accepted_per_sweep,
                        ) = jax.lax.scan(
                            pcn_sweep,
                            (current, current_h, current_loglik, state_key, beta_blocks),
                            jnp.arange(segment_length) if l2_null_k > 0 else None,
                            length=segment_length,
                        )
                        local_accepted_blocks = jnp.sum(accepted_per_sweep, axis=0)
                        global_accepted_blocks = jax.lax.psum(
                            local_accepted_blocks, axis_name="device"
                        )
                        seg_acceptance_blocks = global_accepted_blocks / (
                            segment_length * self.n_particles
                        )
                        new_beta_blocks = _segment_beta_update(beta_blocks, seg_acceptance_blocks) if self.adapt_beta else beta_blocks
                        return (
                            (current, current_h, current_loglik, state_key, new_beta_blocks),
                            (beta_blocks, seg_acceptance_blocks),
                        )

                    (
                        (state, _h_final, loglik, _, beta_blocks_final),
                        (beta_blocks_trace, acceptance_blocks_trace),
                    ) = jax.lax.scan(
                        segment_body,
                        (state, h_obs0, loglik, key, beta_blocks0),
                        None,
                        length=n_segments,
                    )
                    return (
                        state,
                        loglik,
                        beta_blocks_final,
                        beta_blocks_trace,
                        acceptance_blocks_trace,
                    )

                if self.da_subset > 0:
                    # S5 additive: delayed-acceptance variant of the kernel above
                    # (pcn_local is REPLACED only when da_subset > 0; with
                    # da_subset == 0 none of this code is traced).
                    # Stage 1: log-accept = Delta1 = loglik_sub(prop) - loglik_sub(cur)
                    # on a fixed m-row subset (rows chosen on the host per call,
                    # state independent), uniform = the same stream as the plain
                    # kernel. Stage 2 (stage-1 survivors only): accept if
                    # log u2 < (Delta - Delta1), u2 from fold_in(uniform_key, 0xDA).
                    # Survivors are compacted (stable argsort) and the full loglik is
                    # evaluated by a lax.while_loop over only the particle chunks
                    # that contain survivors.
                    m_eff = min(self.da_subset, n_max)
                    sub_chunk = compute_chunk_size(self.local_particles, m_eff, obs_target // 2)
                    loglik_sub_fn = build_loglik_from_hobs(
                        m_eff,
                        library_spec.amplitude,
                        library_spec.terminal_lengthscale,
                        noise_variance,
                        jitter_const,
                        sub_chunk,
                    )
                    n_loc_p = self.local_particles
                    full_chunk = obs_ll_chunk

                    def _da_stage(h_prop, ll_cur, sub_cur, lu1, lu2, sub_idx, sub_mask, y_sub, y_padded, active):
                        sub_prop = loglik_sub_fn(h_prop[:, sub_idx], y_sub, sub_mask)
                        d1 = sub_prop - sub_cur
                        s1 = lu1 < d1
                        n_s = jnp.sum(s1.astype(jnp.int32))
                        order = jnp.argsort(jnp.logical_not(s1).astype(jnp.int32), stable=True)
                        h_sorted = h_prop[order]
                        n_need = (n_s + full_chunk - 1) // full_chunk

                        def w_cond(c):
                            return c[0] < n_need

                        def w_body(c):
                            i, buf = c
                            hc = jax.lax.dynamic_slice_in_dim(h_sorted, i * full_chunk, full_chunk, axis=0)
                            ll_c = loglik_from_hobs(hc, y_padded, active)
                            buf = jax.lax.dynamic_update_slice_in_dim(buf, ll_c, i * full_chunk, axis=0)
                            return i + 1, buf

                        _, buf = jax.lax.while_loop(
                            w_cond, w_body, (jnp.int32(0), jnp.zeros(n_loc_p, dtype=jnp.float64))
                        )
                        full_prop = jnp.zeros(n_loc_p, dtype=jnp.float64).at[order].set(buf)
                        acc2 = lu2 < ((full_prop - ll_cur) - d1)
                        return s1 & acc2, full_prop, sub_prop, s1

                    def pcn_local(state, loglik, key, obs_indices, y_padded, active, beta_blocks0, sub_idx, sub_mask):
                        jt = jtype_table[obs_indices]
                        rw = row_table[obs_indices]
                        obs_counts = [
                            jnp.sum(active * (jt == j).astype(jnp.float64))
                            for j in range(self.n_l2_types)
                        ]
                        y_sub = y_padded[sub_idx]
                        h_obs0 = l1_obs_fn(state, jt, rw)
                        sub0 = loglik_sub_fn(h_obs0[:, sub_idx], y_sub, sub_mask)
                        n_p_f = jnp.float64(n_loc_p)

                        def pcn_sweep(carry, sweep_idx):
                            current, current_h, current_loglik, current_sub, state_key, beta_blocks = carry
                            accepted_this_sweep = jnp.zeros(n_blocks, dtype=jnp.float64)
                            da_this_sweep = jnp.zeros((n_blocks, 3), dtype=jnp.float64)
                            for k in range(n_l1_types_local):
                                start, end = block_bounds[k]
                                state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                                block_current = current[:, start:end]
                                white = jax.random.normal(
                                    normal_key, block_current.shape, dtype=jnp.float64
                                )
                                shrink_k = jnp.sqrt(1.0 - beta_blocks[k] * beta_blocks[k])
                                block_proposal = shrink_k * block_current + beta_blocks[k] * white
                                proposal = current.at[:, start:end].set(block_proposal)
                                h_prop = l1_obs_fn(proposal, jt, rw)
                                lu1 = jnp.log(
                                    jax.random.uniform(uniform_key, (current.shape[0],), dtype=jnp.float64)
                                )
                                lu2 = jnp.log(
                                    jax.random.uniform(
                                        jax.random.fold_in(uniform_key, 0xDA),
                                        (current.shape[0],),
                                        dtype=jnp.float64,
                                    )
                                )
                                accepted, full_prop, sub_prop, s1 = _da_stage(
                                    h_prop, current_loglik, current_sub, lu1, lu2,
                                    sub_idx, sub_mask, y_sub, y_padded, active,
                                )
                                block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                                current = current.at[:, start:end].set(block_new)
                                current_h = jnp.where(accepted[:, None], h_prop, current_h)
                                current_loglik = jnp.where(accepted, full_prop, current_loglik)
                                current_sub = jnp.where(accepted, sub_prop, current_sub)
                                accepted_this_sweep = accepted_this_sweep.at[k].add(jnp.sum(accepted))
                                da_this_sweep = da_this_sweep.at[k].add(
                                    jnp.stack([n_p_f, jnp.sum(s1), jnp.sum(accepted)])
                                )

                            chol = l2_cholesky_fn(current)
                            for j in range(self.n_l2_types):
                                k = n_l1_types_local + j
                                start, end = block_bounds[k]
                                state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                                block_current = current[:, start:end]
                                white = jax.random.normal(
                                    normal_key, block_current.shape, dtype=jnp.float64
                                )
                                shrink_k = jnp.sqrt(1.0 - beta_blocks[k] * beta_blocks[k])
                                block_proposal = shrink_k * block_current + beta_blocks[k] * white
                                lu1 = jnp.log(
                                    jax.random.uniform(uniform_key, (current.shape[0],), dtype=jnp.float64)
                                )
                                lu2 = jnp.log(
                                    jax.random.uniform(
                                        jax.random.fold_in(uniform_key, 0xDA),
                                        (current.shape[0],),
                                        dtype=jnp.float64,
                                    )
                                )

                                def unobserved_branch(h_in, ll_in, sb_in, bp, lu1_, lu2_, chol_in, j=j):
                                    ones = jnp.ones(h_in.shape[0], dtype=bool)
                                    return h_in, ll_in, sb_in, ones, ones

                                def observed_branch(h_in, ll_in, sb_in, bp, lu1_, lu2_, chol_in, j=j):
                                    h_j = jnp.einsum("pij,pj->pi", chol_in, bp)
                                    h_new = jnp.where((jt == j)[None, :], h_j[:, rw], h_in)
                                    acc, full_prop, sub_prop, s1 = _da_stage(
                                        h_new, ll_in, sb_in, lu1_, lu2_,
                                        sub_idx, sub_mask, y_sub, y_padded, active,
                                    )
                                    return (
                                        jnp.where(acc[:, None], h_new, h_in),
                                        jnp.where(acc, full_prop, ll_in),
                                        jnp.where(acc, sub_prop, sb_in),
                                        acc,
                                        s1,
                                    )

                                current_h, current_loglik, current_sub, accepted, s1 = jax.lax.cond(
                                    obs_counts[j] == 0.0,
                                    unobserved_branch,
                                    observed_branch,
                                    current_h,
                                    current_loglik,
                                    current_sub,
                                    block_proposal,
                                    lu1,
                                    lu2,
                                    chol,
                                )
                                block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                                current = current.at[:, start:end].set(block_new)
                                accepted_this_sweep = accepted_this_sweep.at[k].add(jnp.sum(accepted))
                                ev = (obs_counts[j] > 0.0).astype(jnp.float64)
                                da_this_sweep = da_this_sweep.at[k].add(
                                    ev * jnp.stack([n_p_f, jnp.sum(s1), jnp.sum(accepted)])
                                )
                            if l2_null_k > 0:
                                current = jax.lax.cond(
                                    (sweep_idx + 1) % l2_null_k == 0,
                                    lambda c: l2_null_refresh_fn(
                                        c, chol, jax.random.fold_in(state_key, 0x53), jt, rw, active
                                    ),
                                    lambda c: c,
                                    current,
                                )
                            return (
                                current, current_h, current_loglik, current_sub, state_key, beta_blocks,
                            ), (accepted_this_sweep, da_this_sweep)

                        def segment_body(carry, _):
                            current, current_h, current_loglik, current_sub, state_key, beta_blocks = carry
                            (
                                (current, current_h, current_loglik, current_sub, state_key, _),
                                (accepted_per_sweep, da_per_sweep),
                            ) = jax.lax.scan(
                                pcn_sweep,
                                (current, current_h, current_loglik, current_sub, state_key, beta_blocks),
                                jnp.arange(segment_length) if l2_null_k > 0 else None,
                                length=segment_length,
                            )
                            local_accepted_blocks = jnp.sum(accepted_per_sweep, axis=0)
                            global_accepted_blocks = jax.lax.psum(
                                local_accepted_blocks, axis_name="device"
                            )
                            seg_acceptance_blocks = global_accepted_blocks / (
                                segment_length * self.n_particles
                            )
                            da_seg = jax.lax.psum(jnp.sum(da_per_sweep, axis=0), axis_name="device")
                            new_beta_blocks = _segment_beta_update(beta_blocks, seg_acceptance_blocks) if self.adapt_beta else beta_blocks
                            return (
                                (current, current_h, current_loglik, current_sub, state_key, new_beta_blocks),
                                (beta_blocks, seg_acceptance_blocks, da_seg),
                            )

                        (
                            (state, _h_final, loglik, _sub_final, _, beta_blocks_final),
                            (beta_blocks_trace, acceptance_blocks_trace, da_trace),
                        ) = jax.lax.scan(
                            segment_body,
                            (state, h_obs0, loglik, sub0, key, beta_blocks0),
                            None,
                            length=n_segments,
                        )
                        return (
                            state,
                            loglik,
                            beta_blocks_final,
                            beta_blocks_trace,
                            acceptance_blocks_trace,
                            da_trace,
                        )
                if self.terminal_lowrank > 0:
                    # S6-LR additive: low-rank (fixed-grid Nystrom) terminal likelihood
                    # for ALL pCN accept/reject steps of one rejuvenation call (exact SMC:
                    # the final exact loglik corrects the particle weights, see
                    # ``rejuvenate`` / ``assimilate_u15``). Same block structure and the
                    # same random-number stream as the exact kernel above. The per-type
                    # Gram contributions of the CURRENT state are recomputed from the
                    # carried typed inputs h_t (P, T, cap) instead of being stored
                    # (storing G_j = B_j^T B_j for all types needs P*T*r^2*8 B = 2.5 GB
                    # at P=16384, r=49).
                    lr_term = LowRankTerminal(
                        self.terminal_lowrank,
                        library_spec.amplitude,
                        library_spec.terminal_lengthscale,
                        noise_variance,
                        jitter_const,
                    )
                    lr_r_value = lr_term.r
                    lr_cache: dict = {}

                    def _lr_chunk(bytes_pp):
                        side = int(np.ceil(np.sqrt(bytes_pp / 8.0)))
                        return compute_chunk_size(self.local_particles, side, self.lr_chunk_target_bytes)

                    def _lr_map(fn, chunk, *arrays):
                        n_p = arrays[0].shape[0]
                        if chunk >= n_p:
                            return fn(*arrays)
                        nc = n_p // chunk
                        resh = lambda a_: a_.reshape(nc, chunk, *a_.shape[1:])
                        out = jax.lax.map(lambda t: fn(*t), tuple(resh(a_) for a_ in arrays))
                        return jax.tree_util.tree_map(lambda o: o.reshape(n_p, *o.shape[2:]), out)

                    def build_lr_pcn(cap):
                        T_l = n_l2_obs
                        chunk_full = _lr_chunk(
                            8 * T_l * cap * (lr_term.m + 2 * lr_term.r) * 3 + 8 * T_l * lr_term.r**2
                        )
                        chunk_one = _lr_chunk(8 * cap * (lr_term.m + 2 * lr_term.r) * 6)

                        def lr_pcn_local(state, loglik, key, obs_indices, y_padded, active, beta_blocks0):
                            jt = jtype_table[obs_indices]
                            rw = row_table[obs_indices]
                            obs_counts = [
                                jnp.sum(active * (jt == j).astype(jnp.float64))
                                for j in range(T_l)
                            ]
                            idx_t, valid, rw_t = type_tables(jt, rw, active, cap)
                            # repeats with cap > 2G: the refresh gets its own deduplicated
                            # (<= 2G per type) tables; otherwise the typed tables themselves
                            # (the old code path, unchanged).
                            if cap > 2 * G:
                                _, valid_rf, rw_t_rf = type_tables(jt, rw, active, 2 * G, True)
                                cap_rf = 2 * G
                            else:
                                valid_rf, rw_t_rf, cap_rf = valid, rw_t, cap
                            y_t = y_padded[idx_t] * valid
                            yty = jnp.sum(y_padded * y_padded * active)
                            n_act = jnp.sum(active)
                            h_t0 = l1_typed_fn(state, rw_t)
                            hmax = jax.lax.pmax(
                                jnp.max(jnp.abs(h_t0) * valid[None]), axis_name="device"
                            )
                            half = jnp.clip(LOWRANK_H_MARGIN * hmax, LOWRANK_H_MIN, LOWRANK_H_MAX)
                            u_g, c_g = lr_term.grid(half)

                            def full_fn(h_c):
                                g, b = lr_term.contrib(h_c, valid, y_t, u_g, c_g)
                                gs_c, bs_c = jnp.sum(g, axis=1), jnp.sum(b, axis=1)
                                return gs_c, bs_c, lr_term.loglik_from_gram(gs_c, bs_c, yty, n_act)

                            def one_fn(j):
                                def f(h_new_c, h_old_c, gs_c, bs_c):
                                    gn, bn = lr_term.contrib(
                                        h_new_c[:, None, :], valid[j][None], y_t[j][None], u_g, c_g
                                    )
                                    go, bo = lr_term.contrib(
                                        h_old_c[:, None, :], valid[j][None], y_t[j][None], u_g, c_g
                                    )
                                    g2 = gs_c - go[:, 0] + gn[:, 0]
                                    b2 = bs_c - bo[:, 0] + bn[:, 0]
                                    return g2, b2, lr_term.loglik_from_gram(g2, b2, yty, n_act)

                                return f

                            gs0, bs0, ll_lr0 = _lr_map(full_fn, chunk_full, h_t0)

                            def pcn_sweep(carry, sweep_idx):
                                current, h_t, gs, bs, ll, state_key, beta_blocks = carry
                                accepted_this_sweep = jnp.zeros(n_blocks, dtype=jnp.float64)
                                for k in range(n_l1_types_local):
                                    start, end = block_bounds[k]
                                    state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                                    block_current = current[:, start:end]
                                    white = jax.random.normal(
                                        normal_key, block_current.shape, dtype=jnp.float64
                                    )
                                    shrink_k = jnp.sqrt(1.0 - beta_blocks[k] * beta_blocks[k])
                                    block_proposal = shrink_k * block_current + beta_blocks[k] * white
                                    proposal = current.at[:, start:end].set(block_proposal)
                                    h_prop = l1_typed_fn(proposal, rw_t)
                                    gp, bp_, llp = _lr_map(full_fn, chunk_full, h_prop)
                                    accepted = jnp.log(
                                        jax.random.uniform(
                                            uniform_key, (current.shape[0],), dtype=jnp.float64
                                        )
                                    ) < (llp - ll)
                                    block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                                    current = current.at[:, start:end].set(block_new)
                                    h_t = jnp.where(accepted[:, None, None], h_prop, h_t)
                                    gs = jnp.where(accepted[:, None, None], gp, gs)
                                    bs = jnp.where(accepted[:, None], bp_, bs)
                                    ll = jnp.where(accepted, llp, ll)
                                    accepted_this_sweep = accepted_this_sweep.at[k].add(jnp.sum(accepted))

                                chol = l2_cholesky_fn(current)
                                for j in range(T_l):
                                    k = n_l1_types_local + j
                                    start, end = block_bounds[k]
                                    state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                                    block_current = current[:, start:end]
                                    white = jax.random.normal(
                                        normal_key, block_current.shape, dtype=jnp.float64
                                    )
                                    shrink_k = jnp.sqrt(1.0 - beta_blocks[k] * beta_blocks[k])
                                    block_proposal = shrink_k * block_current + beta_blocks[k] * white
                                    log_u = jnp.log(
                                        jax.random.uniform(
                                            uniform_key, (current.shape[0],), dtype=jnp.float64
                                        )
                                    )

                                    def unobserved_branch(h_in, gs_in, bs_in, ll_in, bp, lu, chol_in, j=j):
                                        return h_in, gs_in, bs_in, ll_in, jnp.ones(h_in.shape[0], dtype=bool)

                                    def observed_branch(h_in, gs_in, bs_in, ll_in, bp, lu, chol_in, j=j):
                                        h_j = jnp.einsum("pij,pj->pi", chol_in, bp)
                                        hj_rows = h_j[:, rw_t[j]]
                                        g2, b2, ll2 = _lr_map(
                                            one_fn(j), chunk_one, hj_rows, h_in[:, j], gs_in, bs_in
                                        )
                                        acc = lu < (ll2 - ll_in)
                                        h_out = h_in.at[:, j].set(
                                            jnp.where(acc[:, None], hj_rows, h_in[:, j])
                                        )
                                        return (
                                            h_out,
                                            jnp.where(acc[:, None, None], g2, gs_in),
                                            jnp.where(acc[:, None], b2, bs_in),
                                            jnp.where(acc, ll2, ll_in),
                                            acc,
                                        )

                                    h_t, gs, bs, ll, accepted = jax.lax.cond(
                                        obs_counts[j] == 0.0,
                                        unobserved_branch,
                                        observed_branch,
                                        h_t,
                                        gs,
                                        bs,
                                        ll,
                                        block_proposal,
                                        log_u,
                                        chol,
                                    )
                                    block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                                    current = current.at[:, start:end].set(block_new)
                                    accepted_this_sweep = accepted_this_sweep.at[k].add(jnp.sum(accepted))
                                if l2_null_k > 0:
                                    current = jax.lax.cond(
                                        (sweep_idx + 1) % l2_null_k == 0,
                                        lambda c: l2_null_refresh_typed_fn(
                                            c, chol, jax.random.fold_in(state_key, 0x53), rw_t_rf, valid_rf, cap_rf
                                        ),
                                        lambda c: c,
                                        current,
                                    )
                                return (
                                    current, h_t, gs, bs, ll, state_key, beta_blocks,
                                ), accepted_this_sweep

                            def segment_body(carry, _):
                                current, h_t, gs, bs, ll, state_key, beta_blocks = carry
                                (
                                    (current, h_t, gs, bs, ll, state_key, _),
                                    accepted_per_sweep,
                                ) = jax.lax.scan(
                                    pcn_sweep,
                                    (current, h_t, gs, bs, ll, state_key, beta_blocks),
                                    jnp.arange(segment_length) if l2_null_k > 0 else None,
                                    length=segment_length,
                                )
                                local_accepted_blocks = jnp.sum(accepted_per_sweep, axis=0)
                                global_accepted_blocks = jax.lax.psum(
                                    local_accepted_blocks, axis_name="device"
                                )
                                seg_acceptance_blocks = global_accepted_blocks / (
                                    segment_length * self.n_particles
                                )
                                new_beta_blocks = _segment_beta_update(beta_blocks, seg_acceptance_blocks) if self.adapt_beta else beta_blocks
                                return (
                                    (current, h_t, gs, bs, ll, state_key, new_beta_blocks),
                                    (beta_blocks, seg_acceptance_blocks),
                                )

                            (
                                (state_f, _h_t_f, _gs_f, _bs_f, ll_lr_f, _, beta_blocks_final),
                                (beta_blocks_trace, acceptance_blocks_trace),
                            ) = jax.lax.scan(
                                segment_body,
                                (state, h_t0, gs0, bs0, ll_lr0, key, beta_blocks0),
                                None,
                                length=n_segments,
                            )
                            # one exact evaluation per particle at the end of the call
                            h_obs_f = l1_obs_fn(state_f, jt, rw)
                            ll_exact = loglik_from_hobs(h_obs_f, y_padded, active)
                            logw_corr = (ll_lr0 - loglik) + (ll_exact - ll_lr_f)
                            outside = (jnp.abs(h_obs_f) > half) & (active[None, :] > 0.5)
                            stats = jnp.stack(
                                [
                                    half,
                                    jnp.max(jnp.abs(ll_exact - ll_lr_f)),
                                    jnp.max(jnp.abs(ll_lr0 - loglik)),
                                    jnp.sum(outside).astype(jnp.float64),
                                    n_act * h_obs_f.shape[0],
                                    jnp.sum(jnp.any(outside, axis=1)).astype(jnp.float64),
                                ]
                            )
                            return (
                                state_f,
                                ll_exact,
                                beta_blocks_final,
                                beta_blocks_trace,
                                acceptance_blocks_trace,
                                logw_corr,
                                stats,
                            )

                        return jax.pmap(
                            lr_pcn_local,
                            in_axes=(0, 0, 0, None, None, None, None),
                            axis_name="device",
                            devices=self.devices,
                        )

                    def _lr_get(cap):
                        if cap not in lr_cache:
                            lr_cache[cap] = build_lr_pcn(cap)
                        return lr_cache[cap]

                    lr_pcn_get = _lr_get
            elif self.blocked:
                # M7b "adaptive-blocked" kernel: one pCN step = a SEQUENTIAL scan
                # over the 4 type blocks (A, D, B, C, in whitened-state column
                # order), each proposing ONLY its own slice with its own beta_k
                # and accepting on the FULL likelihood ratio -- the full
                # pushforward is recomputed on every block update (touching all
                # 4 types' worth of covariance/Cholesky work even though only
                # one block's columns changed), a genuine 4x cost per sweep
                # versus the joint kernel below (which computes the pushforward
                # once per step). ``lax.fori_loop`` over ``pcn_steps`` SWEEPS,
                # each sweep a fixed (Python-unrolled, so `k` below is a static
                # int) sequence of 4 block updates, per the task's
                # implementation notes. NOT restructured by Change A (M10f) --
                # ``l2_chol_reuse`` has no effect here even at the dispatch
                # that would otherwise qualify, since this kernel's ``fori_loop``
                # (rather than the segment kernels' ``lax.scan``) makes threading
                # an extra ``current_h`` carry through it a separate, not
                # "straightforward", change; left exactly as before per the
                # task's own allowance.
                # M9 additive: generalised from the pre-M9 literal 4-tuple to
                # ``self.n_pcn_blocks`` blocks (identical values at
                # n_l2_types=2 -- see shared_pcn_block_bounds's docstring).
                block_bounds = (
                    shared_pcn_block_bounds_m12(G, self.n_l1_types)
                    if self.n_l3_types >= 1
                    else shared_pcn_block_bounds(G, self.n_l2_types, self.n_l1_types)
                )
                n_blocks = len(block_bounds)

                def pcn_local(state, loglik, key, obs_indices, y_padded, active, beta_blocks):
                    shrink_blocks = jnp.sqrt(1.0 - beta_blocks * beta_blocks)

                    def sweep_body(_, carry):
                        current, current_loglik, state_key, accepted_totals = carry
                        for k, (start, end) in enumerate(block_bounds):
                            state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                            block_current = current[:, start:end]
                            white = jax.random.normal(
                                normal_key, block_current.shape, dtype=jnp.float64
                            )
                            block_proposal = shrink_blocks[k] * block_current + beta_blocks[k] * white
                            proposal = current.at[:, start:end].set(block_proposal)
                            proposal_loglik = loglik_local(proposal, obs_indices, y_padded, active)
                            accepted = jnp.log(
                                jax.random.uniform(
                                    uniform_key, (current.shape[0],), dtype=jnp.float64
                                )
                            ) < (proposal_loglik - current_loglik)
                            block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                            current = current.at[:, start:end].set(block_new)
                            current_loglik = jnp.where(accepted, proposal_loglik, current_loglik)
                            accepted_totals = accepted_totals.at[k].add(jnp.sum(accepted))
                        return current, current_loglik, state_key, accepted_totals

                    accepted_totals0 = jnp.zeros(n_blocks, dtype=jnp.float64)
                    state, loglik, _, accepted_totals = jax.lax.fori_loop(
                        0,
                        self.pcn_steps,
                        sweep_body,
                        (state, loglik, key, accepted_totals0),
                    )
                    return state, loglik, accepted_totals / (self.pcn_steps * state.shape[0])
            else:

                def pcn_local(state, loglik, key, obs_indices, y_padded, active, beta):
                    shrink = jnp.sqrt(1.0 - beta * beta)

                    def body(_, carry):
                        current, current_loglik, state_key, accepted_total = carry
                        state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                        white = jax.random.normal(normal_key, current.shape, dtype=jnp.float64)
                        proposal = shrink * current + beta * white
                        proposal_loglik = loglik_local(proposal, obs_indices, y_padded, active)
                        accepted = jnp.log(
                            jax.random.uniform(uniform_key, (current.shape[0],), dtype=jnp.float64)
                        ) < (proposal_loglik - current_loglik)
                        current = jnp.where(accepted[:, None], proposal, current)
                        current_loglik = jnp.where(accepted, proposal_loglik, current_loglik)
                        return current, current_loglik, state_key, accepted_total + jnp.sum(accepted)

                    state, loglik, _, accepted_total = jax.lax.fori_loop(
                        0,
                        self.pcn_steps,
                        body,
                        (state, loglik, key, jnp.asarray(0.0, dtype=jnp.float64)),
                    )
                    return state, loglik, accepted_total / (self.pcn_steps * state.shape[0])

            def moments_local(state, local_weights, obs_indices, y_padded, active):
                return moments_from_h(pushforward(state), local_weights, obs_indices, y_padded, active)

            def summary_extended_local(state, local_weights, obs_indices, y_padded, active, truth):
                return summary_extended_from_h(
                    pushforward(state), local_weights, obs_indices, y_padded, active, truth
                )

            bundle_loglik = jax.pmap(
                loglik_local, in_axes=(0, None, None, None), devices=self.devices
            )
            if self.segment_mode:
                # M7c: the segment kernels' between-segment beta update needs a
                # cross-device psum (module docstring's "M7c remedy" paragraph),
                # so (unlike the pre-M7c kernels' _pcn pmap) this one needs
                # axis_name="device".
                bundle_pcn = jax.pmap(
                    pcn_local,
                    in_axes=(0, 0, 0, None, None, None, None) + ((None, None) if self.da_subset > 0 else ()),
                    axis_name="device",
                    devices=self.devices,
                )
            else:
                bundle_pcn = jax.pmap(
                    pcn_local, in_axes=(0, 0, 0, None, None, None, None), devices=self.devices
                )
            bundle_moments = jax.pmap(
                moments_local, in_axes=(0, 0, None, None, None), devices=self.devices
            )
            bundle_moments_lr = None
            moments_lr_fn = None  # 2026-11-01q additive hook (quantile acquisition)
            if self.lowrank_moments:
                # S6-LR additive: Nystrom predictive mean/variance at the candidates.
                lr_mom = LowRankTerminal(
                    self.terminal_lowrank if self.terminal_lowrank > 0 else 64,
                    library_spec.amplitude,
                    library_spec.terminal_lengthscale,
                    noise_variance,
                    jitter_const,
                )
                mom_side = int(
                    np.ceil(np.sqrt(self.n_candidates * (lr_mom.m + 3 * lr_mom.r) * 3.0))
                )
                mom_chunk = compute_chunk_size(self.local_particles, mom_side, chunk_target_bytes)
                moments_lr_fn = build_lowrank_moments(lr_mom, self.n_candidates, n_max, mom_chunk)

                def moments_lr_local(state, local_weights, obs_indices, y_padded, active):
                    return moments_lr_fn(
                        pushforward(state), local_weights, obs_indices, y_padded, active
                    )

                bundle_moments_lr = jax.pmap(
                    moments_lr_local,
                    in_axes=(0, 0, None, None, None),
                    axis_name="device",
                    devices=self.devices,
                )
            bundle_summary_extended = jax.pmap(
                summary_extended_local,
                in_axes=(0, 0, None, None, None, None),
                axis_name="device",
                devices=self.devices,
            )
            return {
                "rb_chunk_size_local": rb_chunk_size_local,
                "loglik_from_h": loglik_from_h,
                "moments_from_h": moments_from_h,
                "summary_extended_from_h": summary_extended_from_h,
                "_loglik": bundle_loglik,
                "_pcn": bundle_pcn,
                "_lr_pcn": lr_pcn_get,
                "_lr_r": lr_r_value,
                "_moments_lr": bundle_moments_lr,
                "moments_lr_from_h": moments_lr_fn,  # 2026-11-01q additive hook (quantile acquisition)
                "_moments": bundle_moments,
                "_summary_extended": bundle_summary_extended,
            }

        self._build_bundle = _build_bundle
        self._init = jax.pmap(initialise_local, devices=self.devices)
        # Eagerly build the LARGEST bucket now (matches the pre-M10f
        # constructor's eager-build-everything behaviour when there is only
        # one bucket, i.e. every caller that does not pass
        # ``n_max_buckets``); every OTHER bucket in ``self.n_max_buckets``
        # compiles lazily the first time ``self._bundle`` sees it.
        self._bundles[self.n_max] = self._build_bundle(self.n_max)

    def n_max_for(self, n_obs: int) -> int:
        """Smallest bucket in ``self.n_max_buckets`` that is ``>= n_obs``
        (M10f additive). Raises ``ValueError`` if every bucket is smaller
        than ``n_obs`` -- callers (``gpn_smc.assimilate``, the drivers) must
        pick their bucket list so the run's eventual observation count never
        exceeds ``self.n_max`` (the largest bucket)."""
        for bucket in self.n_max_buckets:
            if bucket >= n_obs:
                return bucket
        raise ValueError(
            f"n_obs={n_obs} exceeds every bucket in n_max_buckets={self.n_max_buckets}"
        )

    def _bundle(self, n_max: int) -> dict:
        """Lazily build (and cache) the kernel bundle for bucket ``n_max``
        (M10f additive) -- called by ``loglik``/``rejuvenate``/
        ``posterior_moments``/``posterior_summary_extended`` below, keyed off
        the ACTUAL padded-array width the caller passed in (so a call using a
        40-wide padding always reuses the 40 bucket's already-compiled
        kernels, regardless of how many OTHER buckets this backend has)."""
        if n_max not in self._bundles:
            if n_max not in self.n_max_buckets:
                raise ValueError(
                    f"n_max={n_max} is not one of this backend's n_max_buckets={self.n_max_buckets}"
                )
            self._bundles[n_max] = self._build_bundle(n_max)
        return self._bundles[n_max]

    def initialise(self, seed: int) -> jax.Array:
        keys = jax.random.split(jax.random.PRNGKey(seed), self.n_devices)
        result = self._init(keys)
        result.block_until_ready()
        return result

    def loglik(
        self, latent: jax.Array, obs_indices: np.ndarray, y_padded: np.ndarray, active: np.ndarray
    ) -> jax.Array:
        # M10f additive: dispatch to the bucket whose kernels were compiled
        # for THIS padded width (``len(y_padded)``) -- callers must already
        # pad to ``self.n_max_for(len(observed))`` (gpn_smc.assimilate does).
        bundle = self._bundle(int(y_padded.shape[0]))
        result = bundle["_loglik"](latent, obs_indices, y_padded, active)
        result.block_until_ready()
        return result

    LOWRANK_CAP_LADDER = (16, 32, 48, 64, 80, 96, 112, 128)
    # 2026-11-01 additive (repeats only): ladder values above 2G, in steps of 32.
    LOWRANK_CAP_LADDER_REPEAT_STEP = 32

    def lowrank_cap(
        self, obs_indices: np.ndarray, active: np.ndarray, n_max: int, check_only: bool = False
    ) -> int:
        """S6-LR additive: static per-type slot count for the typed layout. With
        ``check_only`` (exact kernel + per-type refresh) the cap is ``min(n_max, 2G)``;
        otherwise the smallest ladder value (multiples of 16 up to 128, capped by ``min(n_max, 2G)``)
        >= the largest number of active observations of one L2 type. Raises if one
        type holds more than ``min(n_max, 2G)`` active rows.

        2026-11-01 additive, ``allow_repeat_observations=True`` only: a type can hold up to
        ``n_max`` observations (repeats of the same cell), so the capacity of the typed layout is
        ``n_max`` and the ladder continues above 128 in steps of 32 (capped by ``n_max``); with
        ``check_only`` the returned cap stays ``min(n_max, 2G)`` (the refresh's slot count) and the
        check is on the number of DISTINCT cells per type when ``n_max > 2G`` (all rows when
        ``n_max <= 2G``, as before)."""
        G_ = self.grid_size // (self.n_l1_types * (1 + self.n_l2_types))
        rep_ = self.allow_repeat_observations
        top = int(n_max) if rep_ else min(int(n_max), 2 * G_)
        act = np.asarray(active) > 0.5
        obs_act = np.asarray(obs_indices)[act]
        if check_only and rep_ and int(n_max) > 2 * G_:
            obs_act = np.unique(obs_act)  # the refresh constrains each distinct cell once
            top = 2 * G_
        if rep_ and not check_only and top > 2 * G_:
            # the refresh's deduplicated tables hold 2G rows per type: guard the distinct-cell count
            n_dist = int(np.bincount(self._jtype_np[np.unique(obs_act)], minlength=self.n_l2_types).max())
            if n_dist > 2 * G_:
                raise ValueError(f"one layer-2 type has {n_dist} distinct observed cells > 2G={2 * G_}")
        counts = np.bincount(self._jtype_np[obs_act], minlength=self.n_l2_types)
        need = int(counts.max()) if counts.size else 0
        if need > top:
            raise ValueError(
                f"one layer-2 type has {need} active observations > per-type capacity {top}"
            )
        if check_only:
            return min(int(n_max), 2 * G_)
        ladder = self.LOWRANK_CAP_LADDER
        if rep_:
            ladder = ladder + tuple(range(2 * G_ + self.LOWRANK_CAP_LADDER_REPEAT_STEP, top + self.LOWRANK_CAP_LADDER_REPEAT_STEP, self.LOWRANK_CAP_LADDER_REPEAT_STEP))
        for c in ladder:
            if c >= need and c <= top:
                return c
        return top

    def rejuvenate(
        self,
        latent: jax.Array,
        loglik: jax.Array,
        seed: int,
        obs_indices: np.ndarray,
        y_padded: np.ndarray,
        active: np.ndarray,
        beta: float | np.ndarray,
    ) -> tuple[jax.Array, jax.Array, float | np.ndarray | SegmentRejuvenationInfo]:
        """``beta`` is a scalar for ``pcn_mode in {"fixed","adaptive",
        "segment"}`` and a length-``N_PCN_BLOCKS`` array for
        ``pcn_mode in {"adaptive-blocked","segment-blocked"}`` (the
        driver's ``PcnBetaState.current()`` already returns the right
        shape). The returned acceptance mirrors that shape for
        ``pcn_mode in {"fixed","adaptive","adaptive-blocked"}``: a scalar
        mean acceptance for the joint kernel, or a length-``N_PCN_BLOCKS``
        array of per-block mean acceptances for the blocked kernel (mean
        taken over devices only, in both cases). For
        ``pcn_mode in {"segment","segment-blocked"}`` the return is instead
        a ``SegmentRejuvenationInfo`` (see that dataclass) -- its fields
        are already reduced over devices (index ``[0]`` below), never
        needing the ``np.mean(..., axis=0)`` the other two return shapes
        use, because every device computes the identical (psum'd)
        per-segment beta/acceptance trace by construction.
        """
        keys = jax.random.split(jax.random.PRNGKey(seed), self.n_devices)
        beta_arg = np.asarray(beta, dtype=np.float64)
        bundle = self._bundle(int(y_padded.shape[0]))
        if self.segment_mode:
            if self.terminal_lowrank > 0 and int(y_padded.shape[0]) >= self.terminal_lowrank_min_n:
                cap = self.lowrank_cap(obs_indices, active, int(y_padded.shape[0]))
                lr_fn = bundle["_lr_pcn"](cap)
                (
                    latent, loglik, beta_final, beta_trace, acceptance_trace, logw_corr, lr_stats,
                ) = lr_fn(latent, loglik, keys, obs_indices, y_padded, active, beta_arg)
                logw_corr = np.asarray(logw_corr).reshape(self.n_particles)
                self.last_lowrank_logw_corr = logw_corr
                st = np.asarray(lr_stats)  # (n_devices, 6)
                wcorr = np.exp(logw_corr - np.max(logw_corr))
                self.last_lowrank_stats = {
                    "H": float(st[0, 0]),
                    "r": int(bundle["_lr_r"]),
                    "m": self.terminal_lowrank,
                    "cap": int(cap),
                    "n_obs": int(np.sum(active > 0.5)),
                    "max_abs_dll_final": float(np.max(st[:, 1])),
                    "max_abs_dll_init": float(np.max(st[:, 2])),
                    "ess_over_P": float(np.sum(wcorr) ** 2 / np.sum(wcorr * wcorr) / self.n_particles),
                    "outside_value_fraction": float(np.sum(st[:, 3]) / max(np.sum(st[:, 4]), 1.0)),
                    "outside_particle_fraction": float(np.sum(st[:, 5]) / self.n_particles),
                }
                print(
                    "[lowrank] n=%(n_obs)d m=%(m)d r=%(r)d cap=%(cap)d H=%(H).3f "
                    "max|dll|=%(max_abs_dll_final).3e (init %(max_abs_dll_init).3e) "
                    "ESS/P=%(ess_over_P).6f outside=%(outside_value_fraction).3e "
                    "(particles %(outside_particle_fraction).3e)" % self.last_lowrank_stats,
                    flush=True,
                )
            elif self.da_subset > 0:
                sub_idx, sub_mask = da_subset_rows(
                    obs_indices, active, min(self.da_subset, int(y_padded.shape[0])),
                    self.n_candidates // self.n_protocols,
                )
                latent, loglik, beta_final, beta_trace, acceptance_trace, da_trace = bundle["_pcn"](
                    latent, loglik, keys, obs_indices, y_padded, active, beta_arg, sub_idx, sub_mask
                )
                da_np = np.asarray(da_trace)[0]  # (n_seg, n_blocks, 3): n_prop, n_stage1_pass, n_accept
                self.last_da_stats = {
                    "n_prop": da_np[:, :, 0], "n_stage1": da_np[:, :, 1], "n_accept": da_np[:, :, 2],
                    "m_eff": int(sub_idx.shape[0]),
                }
            else:
                if self.l2_refresh_per_type:
                    self.lowrank_cap(obs_indices, active, int(y_padded.shape[0]), check_only=True)
                latent, loglik, beta_final, beta_trace, acceptance_trace = bundle["_pcn"](
                    latent, loglik, keys, obs_indices, y_padded, active, beta_arg
                )
            loglik.block_until_ready()
            # beta_final/beta_trace/acceptance_trace are bit-identical on
            # every device (psum'd inside the kernel -- see the module
            # docstring's "M7c remedy" paragraph), so device 0's copy is
            # the whole answer; no np.mean(..., axis=0) needed here.
            beta_trace_np = np.asarray(beta_trace)[0]
            acceptance_trace_np = np.asarray(acceptance_trace)[0]
            beta_final_np = np.asarray(beta_final)[0]
            info = SegmentRejuvenationInfo(
                last_segment_acceptance=acceptance_trace_np[-1],
                first_segment_acceptance=acceptance_trace_np[0],
                beta_used_final_segment=beta_trace_np[-1],
                beta_next=beta_final_np,
                beta_trace=beta_trace_np,
                acceptance_trace=acceptance_trace_np,
            )
            return latent, loglik, info
        latent, loglik, local_acceptance = bundle["_pcn"](
            latent, loglik, keys, obs_indices, y_padded, active, beta_arg
        )
        loglik.block_until_ready()
        if self.blocked:
            return latent, loglik, np.mean(np.asarray(local_acceptance), axis=0)
        return latent, loglik, float(np.mean(np.asarray(local_acceptance)))

    def posterior_moments(
        self,
        latent: jax.Array,
        weights: np.ndarray,
        obs_indices: np.ndarray,
        y_padded: np.ndarray,
        active: np.ndarray,
        exact: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bundle = self._bundle(int(y_padded.shape[0]))
        local_weights = weights.reshape(self.n_devices, self.local_particles)
        if self.lowrank_moments and not exact:  # S6-LR additive
            result = bundle["_moments_lr"](latent, local_weights, obs_indices, y_padded, active)
        else:
            result = bundle["_moments"](latent, local_weights, obs_indices, y_padded, active)
        mean_sum, second_sum, within_sum = [np.sum(np.asarray(v), axis=0) for v in result]
        variance = np.maximum(second_sum - mean_sum * mean_sum, 0.0)
        return mean_sum, variance, within_sum

    def posterior_summary_extended(
        self,
        latent: jax.Array,
        weights: np.ndarray,
        obs_indices: np.ndarray,
        y_padded: np.ndarray,
        active: np.ndarray,
        truth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(mean, variance, within_variance, quantiles, cdf_at_quantiles,
        pit)`` over all 256 joint candidates; ``quantiles``/``cdf_at_quantiles``
        have shape ``(10, 256)``, ``pit`` has shape ``(256,)``.
        """
        bundle = self._bundle(int(y_padded.shape[0]))
        local_weights = weights.reshape(self.n_devices, self.local_particles)
        result = bundle["_summary_extended"](
            latent, local_weights, obs_indices, y_padded, active, truth
        )
        mean_sum, second_sum, within_sum = [np.sum(np.asarray(result[i]), axis=0) for i in range(3)]
        quantiles = np.asarray(result[3])[0]
        cdf_at_quantiles = np.asarray(result[4])[0]
        pit = np.asarray(result[5])[0]
        variance = np.maximum(second_sum - mean_sum * mean_sum, 0.0)
        return mean_sum, variance, within_sum, quantiles, cdf_at_quantiles, pit

    def host_particles(self, latent: jax.Array) -> np.ndarray:
        return np.asarray(latent).reshape(self.n_particles, self.grid_size)

    def shard_particles(self, particles: np.ndarray) -> jax.Array:
        shaped = np.asarray(particles, dtype=np.float64).reshape(
            self.n_devices, self.local_particles, self.grid_size
        )
        return jax.device_put(shaped, self.particle_sharding)

    def sample_prior_h(self, key: jax.Array, n_samples: int | None = None) -> np.ndarray:
        """Test utility: push ``n_samples`` fresh whitened prior draws to the
        joint ``n_candidates``-dim PENULTIMATE (pre-terminal-R) values.
        Eager, runs on the default device (no pmap sharding), calling the
        SAME pure pushforward closure (``self._pushforward`` -- either
        ``pushforward_shared_h`` at ``n_l2_types=2`` or
        ``pushforward_shared_h_general`` otherwise) used inside the
        production pmap'd closures.  Mirrors
        ``gpn_deep_backend.DeepGpnGpuBackend.sample_prior_h``.  Uses its OWN
        eager chunk size (not ``self.chunk_size_local``, which is sized for
        ``self.local_particles``) since ``n_samples`` here is independent of
        the compiled backend's particle count.
        """
        n = self.n_particles if n_samples is None else int(n_samples)
        state = jax.random.normal(key, (n, self.grid_size), dtype=jnp.float64)
        chunk_size = compute_chunk_size(n, self.n_l1_types * GRID_SIZE_PER_PROTOCOL, 256 * 1024 * 1024)
        if self.n_l3_types == 1:
            h_last = pushforward_shared_h_m12(
                state,
                self.layer1_cholesky_A,
                self._layer2_amplitude,
                self._layer2_lengthscale,
                self._jitter_const,
                GRID_SIZE_PER_PROTOCOL,
                chunk_size,
                self.n_l1_types,
            )
        elif self.n_l3_types == 2:
            h_last = pushforward_shared_h_lib_s(
                state,
                self.layer1_cholesky_A,
                self._layer2_amplitude,
                self._layer2_lengthscale,
                self._jitter_const,
                GRID_SIZE_PER_PROTOCOL,
                chunk_size,
                self.n_l1_types,
            )
        elif self.n_l3_types == 3:
            h_last = pushforward_shared_h_battery(
                state,
                self.layer1_cholesky_A,
                self._layer2_amplitude,
                self._layer2_lengthscale,
                self._jitter_const,
                GRID_SIZE_PER_PROTOCOL,
                chunk_size,
                self.n_l1_types,
                self.ramp_w,
            )
        elif self.n_l1_types == 2 and self.n_l2_types == 2:
            h_last = pushforward_shared_h(
                state,
                self.layer1_cholesky_A,
                self._layer2_amplitude,
                self._layer2_lengthscale,
                self._jitter_const,
                GRID_SIZE_PER_PROTOCOL,
                chunk_size,
            )
        elif self.n_l1_types == 2:
            h_last = pushforward_shared_h_general(
                state,
                self.layer1_cholesky_A,
                self._layer2_amplitude,
                self._layer2_lengthscale,
                self._jitter_const,
                GRID_SIZE_PER_PROTOCOL,
                chunk_size,
                self.n_l2_types,
                self.protocol_order,
            )
        else:
            h_last = pushforward_shared_h_general_nl1(
                state,
                self.layer1_cholesky_A,
                self._layer2_amplitude,
                self._layer2_lengthscale,
                self._jitter_const,
                GRID_SIZE_PER_PROTOCOL,
                chunk_size,
                self.n_l1_types,
                self.n_l2_types,
            )
        h_last.block_until_ready()
        return np.asarray(h_last)

    def penultimate_outputs(self, latent: jax.Array) -> jax.Array:
        """K2a additive (K2_DESIGN.md's "New method" section; the only new
        code in this file): a pmap of the existing ``self._pushforward``
        closure over the device axis, returning shape ``(n_devices,
        local_particles, n_candidates)``. Built lazily on first call
        (cached as ``self._penultimate``) so no existing traced program
        changes. ``latent`` must already carry ``self.particle_sharding``'s
        ``(n_devices, local_particles, grid_size)`` layout (the same
        sharded state ``loglik``/``rejuvenate`` consume).
        """
        if getattr(self, "_penultimate", None) is None:
            self._penultimate = jax.pmap(self._pushforward, devices=self.devices)
        result = self._penultimate(latent)
        result.block_until_ready()
        return result

def split_flat_observations_by_protocol(
    obs_indices: np.ndarray,
    y_padded: np.ndarray,
    active: np.ndarray,
    n_max_per_protocol: int = N_MAX_U15,
    grid_size: int = GRID_SIZE_PER_PROTOCOL,
    n_protocols: int = N_PROTOCOLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host-side (numpy) split of a pooled flat-index padded observation
    triple into per-protocol padded arrays, each shape
    ``(n_protocols, n_max_per_protocol)``.  ``n_max_per_protocol`` (40) is a
    safe bound: no single protocol can receive more observations than the
    total joint budget (40), so it never overflows regardless of how
    adaptive acquisition concentrates across protocols.
    """
    per_indices = np.zeros((n_protocols, n_max_per_protocol), dtype=np.int32)
    per_y = np.zeros((n_protocols, n_max_per_protocol), dtype=np.float64)
    per_active = np.zeros((n_protocols, n_max_per_protocol), dtype=np.float64)
    counts = np.zeros(n_protocols, dtype=np.int64)
    for slot in range(len(obs_indices)):
        if active[slot] <= 0.0:
            continue
        flat_index = int(obs_indices[slot])
        protocol = flat_index // grid_size
        local_index = flat_index % grid_size
        slot_p = int(counts[protocol])
        per_indices[protocol, slot_p] = local_index
        per_y[protocol, slot_p] = y_padded[slot]
        per_active[protocol, slot_p] = 1.0
        counts[protocol] += 1
    return per_indices, per_y, per_active

class IndependentProtocolGpnBackend:
    """``n_protocols`` independent depth-3 whitened GPN stacks under one
    joint interface (M8 default ``n_protocols=4``; M9 additive
    ``n_protocols=8`` -- U15_PROTOCOL_BANDITS_PLAN.md's 2026-08-21 section).

    See the module docstring for why this reuses ``gpn_deep_backend``'s pure
    ``pushforward_h``/``compute_chunk_size`` helpers (not the
    ``DeepGpnGpuBackend`` class itself, whose hardcoded capacity-20 padding
    would silently overflow under U15's 40-observation adaptive budget).
    Whitened state width ``n_protocols * 2 * 64`` (each protocol a depth-3
    stack: one 64-dim layer-1 whitened state, one 64-dim layer-2 whitened
    state) -- ``512`` at the M8 default, ``1024`` at M9's 8 protocols.
    ``n_candidates = n_protocols * 64`` for interface parity with
    ``SharedProtocolGpnBackend`` (``256``/``512`` respectively), but every
    candidate's posterior is a function of that protocol's own state and
    observations only: the joint pCN step proposes all ``n_protocols``
    128-dim blocks at once (one vectorized isotropic whitened proposal
    covering the full state, cheaper than separate proposal draws) but then
    accepts/rejects EACH protocol block INDEPENDENTLY, using only that
    protocol's own log-likelihood delta -- genuinely equivalent to
    ``n_protocols`` independent pCN chains run inside one vectorized loop,
    since the protocols never interact statistically in this ablation (no
    shared types). This is deliberately NOT a single joint accept/reject
    decision built from the summed log-likelihood: that would still target
    the correct joint stationary distribution (a valid Metropolis kernel),
    but would couple the protocols' effective mixing/acceptance rates for
    reasons unrelated to the model, artificially depressing this arm's own
    inference quality relative to what truly separate chains achieve --
    exactly the failure mode this ablation exists to avoid, since
    ``indep_gpn`` is the no-sharing baseline shared_gpn is compared against.
    """

    def __init__(
        self,
        warped_x: np.ndarray,
        library_spec: ProtocolLibrarySpec,
        noise_variance: float,
        kernel_jitter: float,
        n_particles: int,
        pcn_steps: int,
        *,
        pcn_mode: str = "adaptive",
        chunk_target_bytes: int = 256 * 1024 * 1024,
        pcn_segment_length: int | None = None,
        n_protocols: int = N_PROTOCOLS,
        protocol_depths: tuple[int, ...] | None = None,
        protocol_layer1_cholesky: tuple[jax.Array, ...] | None = None,
        n_max_buckets: tuple[int, ...] | None = None,
        n_max: int | None = None,
        adapt_beta: bool = True,
    ) -> None:
        # M10f backward-compat shim -- see SharedProtocolGpnBackend's
        # analogous comment.
        if n_max is not None:
            if n_max_buckets is not None:
                raise ValueError("pass only one of n_max (deprecated) or n_max_buckets")
            n_max_buckets = (int(n_max),)
        devices = jax.devices("gpu")
        if not devices:
            raise RuntimeError("No JAX GPU devices are visible")
        if n_particles % len(devices):
            raise ValueError(
                f"particles={n_particles} must be divisible by devices={len(devices)}"
            )
        if pcn_mode not in PCN_MODES:
            raise ValueError(f"pcn_mode={pcn_mode!r} must be one of {PCN_MODES}")
        if pcn_mode == "segment-blocked-obs":
            raise NotImplementedError(
                "pcn_mode='segment-blocked-obs' is implemented for SharedProtocolGpnBackend only"
            )
        # M10f additive: see SharedProtocolGpnBackend's analogous comment --
        # ``n_max`` generalised from one static int to a tuple of lazily-
        # built buckets.
        self.n_max_buckets = tuple(
            sorted(int(b) for b in (n_max_buckets if n_max_buckets is not None else (N_MAX_U15,)))
        )
        if not self.n_max_buckets:
            raise ValueError("n_max_buckets must not be empty")
        if len(set(self.n_max_buckets)) != len(self.n_max_buckets):
            raise ValueError(f"n_max_buckets={self.n_max_buckets} has duplicate entries")
        if any(b <= 0 for b in self.n_max_buckets):
            raise ValueError(f"n_max_buckets={self.n_max_buckets}: every bucket must be > 0")
        self.n_max = self.n_max_buckets[-1]
        self._bundles: dict[int, dict] = {}
        self.devices = devices
        self.n_devices = len(devices)
        self.n_particles = n_particles
        self.local_particles = n_particles // self.n_devices
        self.library_spec = library_spec
        G = int(warped_x.shape[0])
        if G != GRID_SIZE_PER_PROTOCOL:
            raise ValueError(f"expected a {GRID_SIZE_PER_PROTOCOL}-point grid, got {G}")
        if n_protocols < 1:
            raise ValueError(f"n_protocols={n_protocols} must be >= 1")
        # M9 additive generalisation: n_protocols=4 (default) reproduces the
        # M8 four-protocol library exactly (grid_size=512, n_candidates=256,
        # byte-identical values below); n_protocols=8 gives M9's library
        # (grid_size=1024, n_candidates=512). See the class docstring.
        self.n_protocols = int(n_protocols)
        # M12 additive: ``protocol_depths`` (default None -> every protocol
        # depth 3, byte-identical to every pre-M12 caller) lets each
        # protocol have its OWN depth -- PB-D24's per-protocol-depth
        # independent learner: protocols 0..n_l1_types-1 (the SIMPLE set)
        # depth 2 (per-protocol state width G, a bare layer-1 whitened
        # state -- the terminal R is still Rao-Blackwellised, never
        # whitened), protocols n_l1_types..2*n_l1_types-1 (the TARGET set)
        # depth 4 (state width 3*G: layer-1, B, C whitened states), each on
        # its OWN 64 points -- no sharing of anything, R included (the
        # class docstring's "no-sharing ablation" contract, generalised to
        # mixed depth). ``self.depth`` (the old uniform-depth attribute) is
        # kept ONLY when every protocol shares one depth, for any external
        # caller that still reads it; mixed-depth construction leaves it
        # unset (``None``) since it is meaningless there.
        if protocol_depths is None:
            self.protocol_depths = tuple([3] * self.n_protocols)
        else:
            self.protocol_depths = tuple(int(d) for d in protocol_depths)
            if len(self.protocol_depths) != self.n_protocols:
                raise ValueError(
                    f"protocol_depths has {len(self.protocol_depths)} entries, "
                    f"expected n_protocols={self.n_protocols}"
                )
            if any(d < 2 for d in self.protocol_depths):
                raise ValueError(f"protocol_depths={self.protocol_depths}: every depth must be >= 2")
        self.depth = self.protocol_depths[0] if len(set(self.protocol_depths)) == 1 else None
        self.per_protocol_state_widths = tuple((d - 1) * G for d in self.protocol_depths)
        # Backward-compat scalar (meaningful only when every protocol shares
        # one depth/width, i.e. every pre-M12 caller); mixed-depth callers
        # must use ``self.per_protocol_state_widths``/``self._protocol_state_
        # bounds`` instead.
        self.per_protocol_state_width = (
            self.per_protocol_state_widths[0] if self.depth is not None else None
        )
        bounds: list[tuple[int, int]] = []
        start = 0
        for width_p in self.per_protocol_state_widths:
            bounds.append((start, start + width_p))
            start += width_p
        self._protocol_state_bounds = tuple(bounds)
        self.grid_size = start
        self.n_candidates = self.n_protocols * G  # joint interface parity
        # Blocked-pCN block count: one block per protocol (unconditional,
        # regardless of pcn_mode -- see SharedProtocolGpnBackend's analogous
        # attribute).
        self.n_pcn_blocks = self.n_protocols
        self.pcn_steps = pcn_steps
        # See SharedProtocolGpnBackend's analogous comment: "fixed" and
        # "adaptive" share the same joint kernel (self.blocked=False), only
        # "adaptive-blocked" switches to the sequential per-protocol-block
        # kernel below.
        self.pcn_mode = pcn_mode
        # S0c additive: adapt_beta=False replaces every per-segment
        # _segment_beta_update(...) by the identity (beta carried unchanged;
        # traces still record beta/acceptance). Python-level switch at trace
        # time, so the default True compiles the identical kernel.
        self.adapt_beta = bool(adapt_beta)
        self.blocked = pcn_mode in ("adaptive-blocked", "segment-blocked")
        # M7c remedy: see SharedProtocolGpnBackend's analogous comment.
        self.segment_mode = pcn_mode in ("segment", "segment-blocked")
        if self.segment_mode:
            segment_length = (
                PCN_SEGMENT_LENGTH if pcn_segment_length is None else int(pcn_segment_length)
            )
            if segment_length <= 0 or pcn_steps % segment_length != 0:
                raise ValueError(
                    f"pcn_steps={pcn_steps} must be a positive multiple of "
                    f"pcn_segment_length={segment_length} in pcn_mode={pcn_mode!r}"
                )
            self.pcn_segment_length = segment_length
            self.n_pcn_segments = pcn_steps // segment_length
        self.particle_sharding = NamedSharding(Mesh(np.asarray(devices), ("device",)), P("device"))

        layer1_covariance = rbf_covariance(
            warped_x, library_spec.amplitude, library_spec.layer1_lengthscale
        )
        layer1_cholesky_np, layer1_jitter_used = stable_cholesky(layer1_covariance, kernel_jitter)
        self.layer1_jitter_used = layer1_jitter_used
        layer1_cholesky = jnp.asarray(layer1_cholesky_np, dtype=jnp.float64)
        self._layer1_cholesky = layer1_cholesky
        # M13a additive: a PER-PROTOCOL override of which (G,G) Cholesky
        # factor serves as that protocol's OWN first whitened layer --
        # defaults to the standard grid-based ``layer1_cholesky`` for every
        # protocol (byte-identical to every pre-M13a caller, which never
        # passes this). Used by LIB-B's (PB-BAT's) independent-learner
        # counterpart to give its two ramp protocols (private B/C copies
        # "evaluated at w only") their OWN first-layer factor -- the (G,G)
        # RBF Cholesky at amplitude/lengthscale (0.70, 0.50) evaluated at the
        # known ``RAMP_W_LIB_B`` -- instead of the grid-based one, since
        # ``pushforward_h``'s ``first_cholesky`` argument is a fully generic
        # (G,G) factor, not hardwired to any particular domain.
        if protocol_layer1_cholesky is None:
            self._protocol_layer1_cholesky = tuple([layer1_cholesky] * self.n_protocols)
        else:
            self._protocol_layer1_cholesky = tuple(
                jnp.asarray(c, dtype=jnp.float64) for c in protocol_layer1_cholesky
            )
            if len(self._protocol_layer1_cholesky) != self.n_protocols:
                raise ValueError(
                    f"protocol_layer1_cholesky has {len(self._protocol_layer1_cholesky)} "
                    f"entries, expected n_protocols={self.n_protocols}"
                )

        jitter_const = float(kernel_jitter)
        self._jitter_const = jitter_const
        # Sized off the DEEPEST protocol's inner-layer count (M12 additive;
        # at the pre-M12 uniform depth=3 default this is exactly 1, so the
        # array is byte-identical to before). Every entry has the same
        # numeric value (amplitude 0.70 / lengthscale 0.50, precommitted) --
        # a shallower protocol's own ``pushforward_h`` call below simply
        # never indexes past its own ``depth_p - 2`` inner layers, so a
        # longer-than-needed array is always safe.
        max_inner_layers = max(d - 2 for d in self.protocol_depths)
        inner_amplitudes = jnp.asarray(
            [library_spec.amplitude] * max_inner_layers, dtype=jnp.float64
        )
        inner_lengthscales = jnp.asarray(
            [library_spec.layer2_lengthscale] * max_inner_layers, dtype=jnp.float64
        )
        self._inner_amplitudes = inner_amplitudes
        self._inner_lengthscales = inner_lengthscales

        chunk_size_local = compute_chunk_size(self.local_particles, G, chunk_target_bytes)
        self.chunk_size_local = chunk_size_local

        # M12 additive: ONE pushforward closure per protocol, each closing
        # over its OWN (static) depth -- byte-identical to the old single
        # ``pushforward_one`` closure when every protocol shares depth 3
        # (every pre-M12 caller), since ``pushforward_fns[p]`` then calls
        # ``pushforward_h`` with the same ``self.depth=3`` argument for
        # every ``p``.
        def _make_pushforward(depth_p: int, chol_p: jax.Array):
            def fn(state_p: jax.Array) -> jax.Array:
                return pushforward_h(
                    state_p,
                    chol_p,
                    inner_amplitudes,
                    inner_lengthscales,
                    jitter_const,
                    G,
                    depth_p,
                    chunk_size_local,
                )

            return fn

        pushforward_fns = [
            _make_pushforward(d, c)
            for d, c in zip(self.protocol_depths, self._protocol_layer1_cholesky)
        ]
        self._pushforward_fns = tuple(pushforward_fns)

        # See SharedProtocolGpnBackend's analogous comment: chunk the pooled
        # RB tail's (P, n_candidates, n_max) tensors via jax.lax.map, sized
        # conservatively off the larger of the two dimensions. Here
        # n_candidates=G=64 (one protocol's own space) and n_max=40, so the
        # per-protocol footprint is smaller than the shared backend's pooled
        # 256-candidate tail, but moments_local/summary_extended_local below
        # loop over all 4 protocols and concatenate -- chunking keeps each
        # protocol's own tensors bounded too.
        # M12 additive: per-protocol (start, end) state-column bounds
        # (uniform width ``W = (depth-1)*G`` for every protocol at the
        # pre-M12 default, so ``starts[p]:ends[p]`` below is exactly
        # ``p*W:(p+1)*W`` in that case -- byte-identical arithmetic).
        starts = [bound[0] for bound in self._protocol_state_bounds]
        ends = [bound[1] for bound in self._protocol_state_bounds]
        pushforward_fns = self._pushforward_fns
        n_protocols = self.n_protocols

        def initialise_local(key: jax.Array) -> jax.Array:
            return jax.random.normal(
                key, (self.local_particles, self.grid_size), dtype=jnp.float64
            )

        # M10f Change B additive: per-``n_max``-bucket kernel bundle builder
        # -- see ``SharedProtocolGpnBackend._build_bundle``'s analogous
        # comment. Everything ABOVE this point (per-protocol pushforward
        # closures, state-column bounds) is independent of ``n_max``.
        def _build_bundle(n_max: int) -> dict:
            # See SharedProtocolGpnBackend's analogous comment: chunk the pooled
            # RB tail's (P, n_candidates, n_max) tensors via jax.lax.map, sized
            # conservatively off the larger of the two dimensions. Here
            # n_candidates=G=64 (one protocol's own space), so the
            # per-protocol footprint is smaller than the shared backend's pooled
            # 256-candidate tail, but moments_local/summary_extended_local below
            # loop over all 4 protocols and concatenate -- chunking keeps each
            # protocol's own tensors bounded too.
            rb_chunk_size_local = compute_chunk_size(
                self.local_particles, max(G, n_max), chunk_target_bytes
            )
            # M10f: see the shared backend's loglik_chunk_size_local comment.
            loglik_chunk_size_local = compute_chunk_size(
                self.local_particles, n_max, chunk_target_bytes
            )

            loglik_from_h, moments_from_h, summary_extended_from_h = build_terminal_rb_functions(
                n_candidates=G,
                n_max=n_max,
                amplitude=library_spec.amplitude,
                lengthscale=library_spec.terminal_lengthscale,
                noise_variance=noise_variance,
                jitter=jitter_const,
                chunk_size=rb_chunk_size_local,
                loglik_chunk_size=loglik_chunk_size_local,
            )

            def per_protocol_loglik_local(state, obs_indices_pp, y_padded_pp, active_pp):
                """``(n_protocols, P)`` -- each protocol's OWN log-likelihood,
                kept unsummed so ``pcn_local`` can accept/reject each protocol
                block independently (see the class docstring). ``loglik_local``
                below is just this summed over axis 0 -- no extra pushforward
                cost versus the old sum-only version, since computing the sum
                already required computing all ``n_protocols`` terms."""
                per_protocol = []
                for p in range(n_protocols):
                    state_p = state[:, starts[p] : ends[p]]
                    h_last_p = pushforward_fns[p](state_p)
                    per_protocol.append(
                        loglik_from_h(h_last_p, obs_indices_pp[p], y_padded_pp[p], active_pp[p])
                    )
                return jnp.stack(per_protocol, axis=0)

            def loglik_local(state, obs_indices_pp, y_padded_pp, active_pp):
                return jnp.sum(
                    per_protocol_loglik_local(state, obs_indices_pp, y_padded_pp, active_pp),
                    axis=0,
                )

            if pcn_mode == "segment":
                # M7c "segment" kernel: same joint (whole-512-dim proposal,
                # per-protocol-block accept/reject) structure as "adaptive"
                # below, wrapped in the same outer/inner lax.scan-over-segments
                # structure as SharedProtocolGpnBackend's "segment" kernel (see
                # its comment / the module docstring's "M7c remedy" paragraph).
                # The population for the segment's acceptance denominator here
                # is (segment_length * n_protocols * self.n_particles), summing
                # over protocol blocks too, exactly mirroring the pre-existing
                # "adaptive" kernel's own denominator below.
                n_segments = self.n_pcn_segments
                segment_length = self.pcn_segment_length

                def pcn_local(state, loglik, key, obs_indices_pp, y_padded_pp, active_pp, beta0):
                    current_per_protocol_loglik0 = per_protocol_loglik_local(
                        state, obs_indices_pp, y_padded_pp, active_pp
                    )

                    def pcn_step(carry, _):
                        current, current_pp_loglik, state_key, beta = carry
                        shrink = jnp.sqrt(1.0 - beta * beta)
                        state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                        white = jax.random.normal(normal_key, current.shape, dtype=jnp.float64)
                        proposal = shrink * current + beta * white
                        proposal_pp_loglik = per_protocol_loglik_local(
                            proposal, obs_indices_pp, y_padded_pp, active_pp
                        )
                        uniforms = jax.random.uniform(
                            uniform_key, (n_protocols, current.shape[0]), dtype=jnp.float64
                        )
                        accepted = jnp.log(uniforms) < (proposal_pp_loglik - current_pp_loglik)
                        current_blocks = []
                        next_pp_loglik = []
                        for p in range(n_protocols):
                            accepted_p = accepted[p]
                            current_block = jnp.where(
                                accepted_p[:, None],
                                proposal[:, starts[p] : ends[p]],
                                current[:, starts[p] : ends[p]],
                            )
                            current_blocks.append(current_block)
                            next_pp_loglik.append(
                                jnp.where(accepted_p, proposal_pp_loglik[p], current_pp_loglik[p])
                            )
                        current = jnp.concatenate(current_blocks, axis=1)
                        current_pp_loglik = jnp.stack(next_pp_loglik, axis=0)
                        return (current, current_pp_loglik, state_key, beta), jnp.sum(accepted)

                    def segment_body(carry, _):
                        current, current_pp_loglik, state_key, beta = carry
                        (
                            (current, current_pp_loglik, state_key, _),
                            accepted_per_step,
                        ) = jax.lax.scan(
                            pcn_step,
                            (current, current_pp_loglik, state_key, beta),
                            None,
                            length=segment_length,
                        )
                        local_accepted = jnp.sum(accepted_per_step)
                        global_accepted = jax.lax.psum(local_accepted, axis_name="device")
                        seg_acceptance = global_accepted / (
                            segment_length * n_protocols * self.n_particles
                        )
                        new_beta = _segment_beta_update(beta, seg_acceptance) if self.adapt_beta else beta
                        return (current, current_pp_loglik, state_key, new_beta), (
                            beta,
                            seg_acceptance,
                        )

                    (
                        (state, final_pp_loglik, _, beta_final),
                        (beta_trace, acceptance_trace),
                    ) = jax.lax.scan(
                        segment_body,
                        (state, current_per_protocol_loglik0, key, beta0),
                        None,
                        length=n_segments,
                    )
                    loglik = jnp.sum(final_pp_loglik, axis=0)
                    return state, loglik, beta_final, beta_trace, acceptance_trace
            elif pcn_mode == "segment-blocked":
                # M7c "segment-blocked" kernel: same per-protocol-block
                # structure as "adaptive-blocked" below, wrapped in the same
                # outer/inner lax.scan-over-segments structure as
                # SharedProtocolGpnBackend's "segment-blocked" kernel.
                n_segments = self.n_pcn_segments
                segment_length = self.pcn_segment_length

                def pcn_local(state, loglik, key, obs_indices_pp, y_padded_pp, active_pp, beta_blocks0):
                    current_per_protocol_loglik0 = per_protocol_loglik_local(
                        state, obs_indices_pp, y_padded_pp, active_pp
                    )

                    def pcn_sweep(carry, _):
                        current, current_pp_loglik, state_key, beta_blocks = carry
                        accepted_this_sweep = jnp.zeros(n_protocols, dtype=jnp.float64)
                        for p in range(n_protocols):
                            state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                            block_current = current[:, starts[p] : ends[p]]
                            white = jax.random.normal(
                                normal_key, block_current.shape, dtype=jnp.float64
                            )
                            shrink_p = jnp.sqrt(1.0 - beta_blocks[p] * beta_blocks[p])
                            block_proposal = shrink_p * block_current + beta_blocks[p] * white
                            h_last_p = pushforward_fns[p](block_proposal)
                            proposal_loglik_p = loglik_from_h(
                                h_last_p, obs_indices_pp[p], y_padded_pp[p], active_pp[p]
                            )
                            accepted = jnp.log(
                                jax.random.uniform(
                                    uniform_key, (current.shape[0],), dtype=jnp.float64
                                )
                            ) < (proposal_loglik_p - current_pp_loglik[p])
                            block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                            current = current.at[:, starts[p] : ends[p]].set(block_new)
                            current_pp_loglik = current_pp_loglik.at[p].set(
                                jnp.where(accepted, proposal_loglik_p, current_pp_loglik[p])
                            )
                            accepted_this_sweep = accepted_this_sweep.at[p].add(jnp.sum(accepted))
                        return (current, current_pp_loglik, state_key, beta_blocks), accepted_this_sweep

                    def segment_body(carry, _):
                        current, current_pp_loglik, state_key, beta_blocks = carry
                        (
                            (current, current_pp_loglik, state_key, _),
                            accepted_per_sweep,
                        ) = jax.lax.scan(
                            pcn_sweep,
                            (current, current_pp_loglik, state_key, beta_blocks),
                            None,
                            length=segment_length,
                        )
                        local_accepted_blocks = jnp.sum(accepted_per_sweep, axis=0)
                        global_accepted_blocks = jax.lax.psum(
                            local_accepted_blocks, axis_name="device"
                        )
                        seg_acceptance_blocks = global_accepted_blocks / (
                            segment_length * self.n_particles
                        )
                        new_beta_blocks = _segment_beta_update(beta_blocks, seg_acceptance_blocks) if self.adapt_beta else beta_blocks
                        return (
                            (current, current_pp_loglik, state_key, new_beta_blocks),
                            (beta_blocks, seg_acceptance_blocks),
                        )

                    (
                        (state, final_pp_loglik, _, beta_blocks_final),
                        (beta_blocks_trace, acceptance_blocks_trace),
                    ) = jax.lax.scan(
                        segment_body,
                        (state, current_per_protocol_loglik0, key, beta_blocks0),
                        None,
                        length=n_segments,
                    )
                    loglik = jnp.sum(final_pp_loglik, axis=0)
                    return state, loglik, beta_blocks_final, beta_blocks_trace, acceptance_blocks_trace
            elif self.blocked:
                # M7b "adaptive-blocked" kernel: one pCN step = a SEQUENTIAL scan
                # over the n_protocols protocol blocks (P1..Pn), each proposing
                # ONLY its own 128-dim slice with its own beta_p and accepting
                # on the FULL likelihood ratio. Because the protocols never
                # interact statistically in this ablation (no shared types --
                # see the class docstring), the full joint likelihood ratio
                # restricted to one block reduces algebraically to that
                # protocol's OWN log-likelihood delta: the other protocols'
                # terms are identical on both sides of the ratio and cancel
                # exactly, so recomputing them here would be wasted compute, not
                # extra correctness -- this mirrors
                # ``per_protocol_loglik_local``'s existing design (computing
                # each protocol's pushforward once, never combined into the
                # others). ``lax.fori_loop`` over ``pcn_steps`` SWEEPS, each
                # sweep a fixed (Python-unrolled, static ``p``) sequence of
                # n_protocols block updates, per the task's implementation
                # notes.
                def pcn_local(state, loglik, key, obs_indices_pp, y_padded_pp, active_pp, beta_blocks):
                    shrink_blocks = jnp.sqrt(1.0 - beta_blocks * beta_blocks)
                    current_per_protocol_loglik0 = per_protocol_loglik_local(
                        state, obs_indices_pp, y_padded_pp, active_pp
                    )

                    def sweep_body(_, carry):
                        current, current_pp_loglik, state_key, accepted_totals = carry
                        for p in range(n_protocols):
                            state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                            block_current = current[:, starts[p] : ends[p]]
                            white = jax.random.normal(
                                normal_key, block_current.shape, dtype=jnp.float64
                            )
                            block_proposal = (
                                shrink_blocks[p] * block_current + beta_blocks[p] * white
                            )
                            h_last_p = pushforward_fns[p](block_proposal)
                            proposal_loglik_p = loglik_from_h(
                                h_last_p, obs_indices_pp[p], y_padded_pp[p], active_pp[p]
                            )
                            accepted = jnp.log(
                                jax.random.uniform(
                                    uniform_key, (current.shape[0],), dtype=jnp.float64
                                )
                            ) < (proposal_loglik_p - current_pp_loglik[p])
                            block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                            current = current.at[:, starts[p] : ends[p]].set(block_new)
                            current_pp_loglik = current_pp_loglik.at[p].set(
                                jnp.where(accepted, proposal_loglik_p, current_pp_loglik[p])
                            )
                            accepted_totals = accepted_totals.at[p].add(jnp.sum(accepted))
                        return current, current_pp_loglik, state_key, accepted_totals

                    accepted_totals0 = jnp.zeros(n_protocols, dtype=jnp.float64)
                    state, final_pp_loglik, _, accepted_totals = jax.lax.fori_loop(
                        0,
                        self.pcn_steps,
                        sweep_body,
                        (state, current_per_protocol_loglik0, key, accepted_totals0),
                    )
                    loglik = jnp.sum(final_pp_loglik, axis=0)
                    return state, loglik, accepted_totals / (self.pcn_steps * state.shape[0])
            else:

                def pcn_local(state, loglik, key, obs_indices_pp, y_padded_pp, active_pp, beta):
                    """Isotropic whitened pCN over the full state (one proposal
                    draw covering all ``n_protocols`` 128-dim protocol blocks at
                    once), but accept/reject is decided PER PROTOCOL BLOCK --
                    ``n_protocols`` independent Bernoulli draws per particle per
                    step, each thresholded against only that block's own
                    log-likelihood delta -- not one joint decision from the
                    summed log-likelihood (see the class docstring for why).
                    The incoming scalar ``loglik`` (this backend's public
                    per-particle total, matching ``self._loglik``'s output) is
                    recomputed here as its per-protocol breakdown at the start
                    of the loop, since the per-protocol terms are not
                    recoverable from the sum alone.
                    """
                    shrink = jnp.sqrt(1.0 - beta * beta)
                    current_per_protocol_loglik0 = per_protocol_loglik_local(
                        state, obs_indices_pp, y_padded_pp, active_pp
                    )

                    def body(_, carry):
                        current, current_per_protocol_loglik, state_key, accepted_total = carry
                        state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                        white = jax.random.normal(normal_key, current.shape, dtype=jnp.float64)
                        proposal = shrink * current + beta * white
                        proposal_per_protocol_loglik = per_protocol_loglik_local(
                            proposal, obs_indices_pp, y_padded_pp, active_pp
                        )
                        uniforms = jax.random.uniform(
                            uniform_key, (n_protocols, current.shape[0]), dtype=jnp.float64
                        )
                        accepted = jnp.log(uniforms) < (
                            proposal_per_protocol_loglik - current_per_protocol_loglik
                        )  # (n_protocols, P): one independent decision per protocol block

                        current_blocks = []
                        next_per_protocol_loglik = []
                        for p in range(n_protocols):
                            accepted_p = accepted[p]
                            current_block = jnp.where(
                                accepted_p[:, None],
                                proposal[:, starts[p] : ends[p]],
                                current[:, starts[p] : ends[p]],
                            )
                            current_blocks.append(current_block)
                            next_per_protocol_loglik.append(
                                jnp.where(
                                    accepted_p,
                                    proposal_per_protocol_loglik[p],
                                    current_per_protocol_loglik[p],
                                )
                            )
                        current = jnp.concatenate(current_blocks, axis=1)
                        current_per_protocol_loglik = jnp.stack(next_per_protocol_loglik, axis=0)
                        return (
                            current,
                            current_per_protocol_loglik,
                            state_key,
                            accepted_total + jnp.sum(accepted),
                        )

                    state, final_per_protocol_loglik, _, accepted_total = jax.lax.fori_loop(
                        0,
                        self.pcn_steps,
                        body,
                        (state, current_per_protocol_loglik0, key, jnp.asarray(0.0, dtype=jnp.float64)),
                    )
                    loglik = jnp.sum(final_per_protocol_loglik, axis=0)
                    acceptance = accepted_total / (self.pcn_steps * n_protocols * state.shape[0])
                    return state, loglik, acceptance

            def moments_local(state, local_weights, obs_indices_pp, y_padded_pp, active_pp):
                means, seconds, withins = [], [], []
                for p in range(n_protocols):
                    state_p = state[:, starts[p] : ends[p]]
                    h_last_p = pushforward_fns[p](state_p)
                    mean_p, second_p, within_p = moments_from_h(
                        h_last_p, local_weights, obs_indices_pp[p], y_padded_pp[p], active_pp[p]
                    )
                    means.append(mean_p)
                    seconds.append(second_p)
                    withins.append(within_p)
                return (
                    jnp.concatenate(means, axis=0),
                    jnp.concatenate(seconds, axis=0),
                    jnp.concatenate(withins, axis=0),
                )

            def summary_extended_local(state, local_weights, obs_indices_pp, y_padded_pp, active_pp, truth_pp):
                means, seconds, withins = [], [], []
                quantiles_list, cdfs_list, pits = [], [], []
                for p in range(n_protocols):
                    state_p = state[:, starts[p] : ends[p]]
                    h_last_p = pushforward_fns[p](state_p)
                    mean_p, second_p, within_p, quantiles_p, cdf_p, pit_p = summary_extended_from_h(
                        h_last_p,
                        local_weights,
                        obs_indices_pp[p],
                        y_padded_pp[p],
                        active_pp[p],
                        truth_pp[p],
                    )
                    means.append(mean_p)
                    seconds.append(second_p)
                    withins.append(within_p)
                    quantiles_list.append(quantiles_p)
                    cdfs_list.append(cdf_p)
                    pits.append(pit_p)
                return (
                    jnp.concatenate(means, axis=0),
                    jnp.concatenate(seconds, axis=0),
                    jnp.concatenate(withins, axis=0),
                    jnp.concatenate(quantiles_list, axis=1),
                    jnp.concatenate(cdfs_list, axis=1),
                    jnp.concatenate(pits, axis=0),
                )

            bundle_loglik = jax.pmap(
                loglik_local, in_axes=(0, None, None, None), devices=self.devices
            )
            if self.segment_mode:
                # M7c: see SharedProtocolGpnBackend's analogous comment -- the
                # segment kernels' between-segment beta update needs a
                # cross-device psum, so this pmap needs axis_name="device"
                # (unlike the pre-M7c _pcn pmap below).
                bundle_pcn = jax.pmap(
                    pcn_local,
                    in_axes=(0, 0, 0, None, None, None, None),
                    axis_name="device",
                    devices=self.devices,
                )
            else:
                bundle_pcn = jax.pmap(
                    pcn_local, in_axes=(0, 0, 0, None, None, None, None), devices=self.devices
                )
            bundle_moments = jax.pmap(
                moments_local, in_axes=(0, 0, None, None, None), devices=self.devices
            )
            bundle_summary_extended = jax.pmap(
                summary_extended_local,
                in_axes=(0, 0, None, None, None, None),
                axis_name="device",
                devices=self.devices,
            )
            return {
                "rb_chunk_size_local": rb_chunk_size_local,
                "loglik_from_h": loglik_from_h,
                "moments_from_h": moments_from_h,
                "summary_extended_from_h": summary_extended_from_h,
                "_loglik": bundle_loglik,
                "_pcn": bundle_pcn,
                "_moments": bundle_moments,
                "_summary_extended": bundle_summary_extended,
            }

        self._build_bundle = _build_bundle
        self._init = jax.pmap(initialise_local, devices=self.devices)
        # Eagerly build the largest bucket now (matches every pre-M10f
        # caller's single-bucket eager-build behaviour); other buckets
        # compile lazily on first use.
        self._bundles[self.n_max] = self._build_bundle(self.n_max)

    def n_max_for(self, n_obs: int) -> int:
        """See ``SharedProtocolGpnBackend.n_max_for``'s docstring."""
        for bucket in self.n_max_buckets:
            if bucket >= n_obs:
                return bucket
        raise ValueError(
            f"n_obs={n_obs} exceeds every bucket in n_max_buckets={self.n_max_buckets}"
        )

    def _bundle(self, n_max: int) -> dict:
        """See ``SharedProtocolGpnBackend._bundle``'s docstring."""
        if n_max not in self._bundles:
            if n_max not in self.n_max_buckets:
                raise ValueError(
                    f"n_max={n_max} is not one of this backend's n_max_buckets={self.n_max_buckets}"
                )
            self._bundles[n_max] = self._build_bundle(n_max)
        return self._bundles[n_max]

    def initialise(self, seed: int) -> jax.Array:
        keys = jax.random.split(jax.random.PRNGKey(seed), self.n_devices)
        result = self._init(keys)
        result.block_until_ready()
        return result

    def loglik(
        self, latent: jax.Array, obs_indices: np.ndarray, y_padded: np.ndarray, active: np.ndarray
    ) -> jax.Array:
        # M10f additive: bucket off the ACTUAL padded width (see
        # SharedProtocolGpnBackend.loglik's analogous comment) -- the
        # per-protocol split below must use the SAME width so its output
        # matches the shape the bucket's kernels were compiled for.
        bucket = int(y_padded.shape[0])
        bundle = self._bundle(bucket)
        pp_indices, pp_y, pp_active = split_flat_observations_by_protocol(
            obs_indices, y_padded, active,
            n_max_per_protocol=bucket, n_protocols=self.n_protocols,
        )
        result = bundle["_loglik"](latent, pp_indices, pp_y, pp_active)
        result.block_until_ready()
        return result

    def rejuvenate(
        self,
        latent: jax.Array,
        loglik: jax.Array,
        seed: int,
        obs_indices: np.ndarray,
        y_padded: np.ndarray,
        active: np.ndarray,
        beta: float | np.ndarray,
    ) -> tuple[jax.Array, jax.Array, float | np.ndarray | SegmentRejuvenationInfo]:
        """See ``SharedProtocolGpnBackend.rejuvenate``'s docstring: ``beta``
        is a scalar for ``pcn_mode in {"fixed","adaptive","segment"}`` and a
        length-``N_PCN_BLOCKS`` (per-protocol) array for
        ``pcn_mode in {"adaptive-blocked","segment-blocked"}``; the returned
        acceptance mirrors that shape, except in
        ``pcn_mode in {"segment","segment-blocked"}`` where it is instead a
        ``SegmentRejuvenationInfo`` (see that dataclass and
        ``SharedProtocolGpnBackend.rejuvenate``'s docstring).
        """
        bucket = int(y_padded.shape[0])
        bundle = self._bundle(bucket)
        pp_indices, pp_y, pp_active = split_flat_observations_by_protocol(
            obs_indices, y_padded, active,
            n_max_per_protocol=bucket, n_protocols=self.n_protocols,
        )
        keys = jax.random.split(jax.random.PRNGKey(seed), self.n_devices)
        beta_arg = np.asarray(beta, dtype=np.float64)
        if self.segment_mode:
            latent, loglik, beta_final, beta_trace, acceptance_trace = bundle["_pcn"](
                latent, loglik, keys, pp_indices, pp_y, pp_active, beta_arg
            )
            loglik.block_until_ready()
            beta_trace_np = np.asarray(beta_trace)[0]
            acceptance_trace_np = np.asarray(acceptance_trace)[0]
            beta_final_np = np.asarray(beta_final)[0]
            info = SegmentRejuvenationInfo(
                last_segment_acceptance=acceptance_trace_np[-1],
                first_segment_acceptance=acceptance_trace_np[0],
                beta_used_final_segment=beta_trace_np[-1],
                beta_next=beta_final_np,
                beta_trace=beta_trace_np,
                acceptance_trace=acceptance_trace_np,
            )
            return latent, loglik, info
        latent, loglik, local_acceptance = bundle["_pcn"](
            latent, loglik, keys, pp_indices, pp_y, pp_active, beta_arg
        )
        loglik.block_until_ready()
        if self.blocked:
            return latent, loglik, np.mean(np.asarray(local_acceptance), axis=0)
        return latent, loglik, float(np.mean(np.asarray(local_acceptance)))

    def posterior_moments(
        self,
        latent: jax.Array,
        weights: np.ndarray,
        obs_indices: np.ndarray,
        y_padded: np.ndarray,
        active: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bucket = int(y_padded.shape[0])
        bundle = self._bundle(bucket)
        pp_indices, pp_y, pp_active = split_flat_observations_by_protocol(
            obs_indices, y_padded, active,
            n_max_per_protocol=bucket, n_protocols=self.n_protocols,
        )
        local_weights = weights.reshape(self.n_devices, self.local_particles)
        result = bundle["_moments"](latent, local_weights, pp_indices, pp_y, pp_active)
        mean_sum, second_sum, within_sum = [np.sum(np.asarray(v), axis=0) for v in result]
        variance = np.maximum(second_sum - mean_sum * mean_sum, 0.0)
        return mean_sum, variance, within_sum

    def posterior_summary_extended(
        self,
        latent: jax.Array,
        weights: np.ndarray,
        obs_indices: np.ndarray,
        y_padded: np.ndarray,
        active: np.ndarray,
        truth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bucket = int(y_padded.shape[0])
        bundle = self._bundle(bucket)
        pp_indices, pp_y, pp_active = split_flat_observations_by_protocol(
            obs_indices, y_padded, active,
            n_max_per_protocol=bucket, n_protocols=self.n_protocols,
        )
        truth_pp = np.asarray(truth, dtype=np.float64).reshape(self.n_protocols, GRID_SIZE_PER_PROTOCOL)
        local_weights = weights.reshape(self.n_devices, self.local_particles)
        result = bundle["_summary_extended"](
            latent, local_weights, pp_indices, pp_y, pp_active, truth_pp
        )
        mean_sum, second_sum, within_sum = [np.sum(np.asarray(result[i]), axis=0) for i in range(3)]
        quantiles = np.asarray(result[3])[0]
        cdf_at_quantiles = np.asarray(result[4])[0]
        pit = np.asarray(result[5])[0]
        variance = np.maximum(second_sum - mean_sum * mean_sum, 0.0)
        return mean_sum, variance, within_sum, quantiles, cdf_at_quantiles, pit

    def host_particles(self, latent: jax.Array) -> np.ndarray:
        return np.asarray(latent).reshape(self.n_particles, self.grid_size)

    def shard_particles(self, particles: np.ndarray) -> jax.Array:
        shaped = np.asarray(particles, dtype=np.float64).reshape(
            self.n_devices, self.local_particles, self.grid_size
        )
        return jax.device_put(shaped, self.particle_sharding)

    def sample_prior_h(self, key: jax.Array, n_samples: int | None = None) -> np.ndarray:
        """Test utility: push ``n_samples`` fresh whitened prior draws to the
        joint ``n_candidates``-dim PENULTIMATE (pre-terminal-R) values, one
        independent depth-3 pushforward per protocol (protocol-major order,
        matching the flat-index convention).  Eager, single device.
        """
        n = self.n_particles if n_samples is None else int(n_samples)
        G = GRID_SIZE_PER_PROTOCOL
        keys = jax.random.split(key, self.n_protocols)
        chunk_size = compute_chunk_size(n, G, 256 * 1024 * 1024)
        h_parts = []
        for p in range(self.n_protocols):
            state_p = jax.random.normal(
                keys[p], (n, self.per_protocol_state_widths[p]), dtype=jnp.float64
            )
            h_p = pushforward_h(
                state_p,
                self._protocol_layer1_cholesky[p],
                self._inner_amplitudes,
                self._inner_lengthscales,
                self._jitter_const,
                G,
                self.protocol_depths[p],
                chunk_size,
            )
            h_parts.append(h_p)
        h_last = jnp.concatenate(h_parts, axis=1)
        h_last.block_until_ready()
        return np.asarray(h_last)

    def penultimate_outputs(self, latent: jax.Array) -> jax.Array:
        """K2a additive (K2_DESIGN.md's "New method" section; the only new
        code in this file): sharded analogue of ``sample_prior_h`` -- one
        pmap over the device axis that, per device, concatenates the
        per-protocol ``self._pushforward_fns`` pushforwards (each applied to
        its own state-column slice via ``self._protocol_state_bounds``,
        exactly ``per_protocol_loglik_local``'s slicing pattern above) into
        the joint ``(local_particles, n_candidates)`` penultimate output.
        Returns shape ``(n_devices, local_particles, n_candidates)``. Built
        lazily on first call (cached as ``self._penultimate``). ``latent``
        must already carry ``self.particle_sharding``'s ``(n_devices,
        local_particles, grid_size)`` layout.
        """
        if getattr(self, "_penultimate", None) is None:
            starts = [bound[0] for bound in self._protocol_state_bounds]
            ends = [bound[1] for bound in self._protocol_state_bounds]
            pushforward_fns = self._pushforward_fns
            n_protocols = self.n_protocols

            def penultimate_local(state: jax.Array) -> jax.Array:
                parts = []
                for p in range(n_protocols):
                    state_p = state[:, starts[p]:ends[p]]
                    parts.append(pushforward_fns[p](state_p))
                return jnp.concatenate(parts, axis=1)

            self._penultimate = jax.pmap(penultimate_local, devices=self.devices)
        result = self._penultimate(latent)
        result.block_until_ready()
        return result
