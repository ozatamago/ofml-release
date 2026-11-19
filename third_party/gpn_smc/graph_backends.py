"""gpn_smc.graph_backends -- SMC backends for generic library graphs (BENCH9, 2026-11-01b).

Additive module: nothing in ``backends.py`` / ``factorised_independent.py`` is touched.

* ``GraphSharedGpnBackend`` -- the shared model of a ``graph_library.GraphLibrary``.  Latent =
  whitened blocks concatenated in TYPE order (block width ``n_nodes_t * G``), one pCN block per
  type.  Pushforward: for each type in order, Cholesky of K(Z_t) (Z_t built from the parent
  outputs of THIS particle; types with identical Z share one Cholesky) times its whitened block;
  the penultimate vector is the protocol-ordered concatenation of the protocols' penultimate
  node outputs.  Terminal R: exactly the existing Rao-Blackwellised exact terminal
  (``build_terminal_rb_functions``; repeated observations of one cell are ordinary rows).
  pCN kernel = the generic "segment-blocked" kernel (same key-splitting order as
  ``SharedProtocolGpnBackend``'s generic "segment-blocked" kernel), with an exact incremental
  forward pass: updating block t recomputes only type t and its descendants (the other node
  outputs are carried).  Optional ``terminal_lowrank``: fixed-grid Nystrom terminal for the
  pCN accept/reject steps with the exact end-of-call correction (the same exact-SMC scheme as
  the S6-LR path of ``SharedProtocolGpnBackend``), evaluated on the DISTINCT observed cells
  (counts / y-sums), so its cost does not grow with the number of repeat visits.
  2026-11-03 additive ``l2_null_refresh=K`` (0 = off, the old kernel bit for bit): every K-th sweep the
  whitened blocks of the penultimate (leaf) types are resampled exactly from their conditional prior given
  the likelihood-relevant part ``A xi`` (``_refresh_structure`` / ``_make_refresh``; the graph version of
  ``SharedProtocolGpnBackend``'s S3 ``l2_null_refresh``).
* ``GraphFactorisedIndependentGpnBackend`` -- the factorised independent model: protocol p gets a
  private copy of its sub-graph (one private type per node), state width ``n_sub(p) * G``,
  one particle system per protocol (``FactorisedIndependentGpnBackend`` interface).

Both classes are duck-type compatible with the drivers (``u15_compare.run_gpn_arm`` /
``u15_indep_factorised.run_indep_factorised_arm``).
"""

from __future__ import annotations

import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .constants import GRID_SIZE_PER_PROTOCOL, PCN_SEGMENT_LENGTH
from .factorised_independent import FactorisedIndependentGpnBackend
from .graph_library import GraphLibrary, rbf_kernel_nd, stable_cholesky
from .library import ProtocolLibrarySpec
from .numerics import compute_chunk_size
from .pcn import SegmentRejuvenationInfo, _segment_beta_update
from .terminal_rb import (
    LOWRANK_H_MARGIN,
    LOWRANK_H_MAX,
    LOWRANK_H_MIN,
    LowRankTerminal,
    build_lowrank_moments,
    build_terminal_rb_functions,
)

GRAPH_PCN_MODES = ("segment-blocked",)


def _sq_dist_terms(zs: list[jax.Array], ell: float) -> jax.Array:
    """sum_d ((z_d[:, :, None] - z_d[:, None, :]) / ell)^2 (for one dim the arithmetic is the
    same as the existing ``(difference / ell) ** 2``)."""
    total = None
    for z in zs:
        diff = z[:, :, None] - z[:, None, :]
        term = (diff / ell) ** 2
        total = term if total is None else total + term
    return total


class GraphSharedGpnBackend:
    """Shared-model SMC backend of a library graph (see module docstring)."""

    LOWRANK_CAP_LADDER_STEP = 32

    def __init__(
        self,
        warped_x: np.ndarray,
        library_spec: ProtocolLibrarySpec,
        graph: GraphLibrary,
        noise_variance: float,
        kernel_jitter: float,
        n_particles: int,
        pcn_steps: int,
        *,
        pcn_mode: str = "segment-blocked",
        n_max_buckets: tuple[int, ...] | None = None,
        chunk_target_bytes: int = 256 * 1024 * 1024,
        pcn_segment_length: int | None = None,
        adapt_beta: bool = True,
        allow_repeat_observations: bool = False,
        terminal_lowrank: int = 0,
        terminal_lowrank_min_n: int = 0,
        lowrank_moments: bool | None = None,
        lr_chunk_target_bytes: int = 1024 * 1024 * 1024,
        incremental: bool = True,
        fast_chol: bool = False,
        l2_null_refresh: int = 0,
        mixing_diagnostics: bool = False,
        inner_collapsed_sweeps: int = 0,
        inner_collapsed_beta0: float = 0.3,
        inner_collapsed_mode: int = 1,
        inner_move: str = "pcn",
        inner_hmc_steps: int = 10,
        inner_ess_shrinks: int = 16,
        h_ess_shrinks: int = 0,
    ) -> None:
        devices = jax.devices("gpu")
        if not devices:
            raise RuntimeError("No JAX GPU devices are visible")
        if n_particles % len(devices):
            raise ValueError(f"particles={n_particles} must be divisible by devices={len(devices)}")
        if pcn_mode not in GRAPH_PCN_MODES:
            raise ValueError(f"GraphSharedGpnBackend supports pcn_mode in {GRAPH_PCN_MODES}, got {pcn_mode!r}")
        graph.validate()
        G = int(warped_x.shape[0])
        if G != GRID_SIZE_PER_PROTOCOL:
            raise ValueError(f"expected a {GRID_SIZE_PER_PROTOCOL}-point grid, got {G}")
        self.graph = graph
        self.G = G
        self.n_max_buckets = tuple(sorted(int(b) for b in (n_max_buckets if n_max_buckets else (100,))))
        if len(set(self.n_max_buckets)) != len(self.n_max_buckets) or any(b <= 0 for b in self.n_max_buckets):
            raise ValueError(f"bad n_max_buckets={self.n_max_buckets}")
        self.n_max = self.n_max_buckets[-1]
        self.devices = devices
        self.n_devices = len(devices)
        self.n_particles = int(n_particles)
        self.local_particles = self.n_particles // self.n_devices
        self.library_spec = library_spec
        self.pcn_steps = int(pcn_steps)
        self.pcn_mode = pcn_mode
        self.segment_mode = True
        self.blocked = True
        self.adapt_beta = bool(adapt_beta)
        self.allow_repeat_observations = bool(allow_repeat_observations)
        self.incremental = bool(incremental)
        # 2026-11-02 additive (S1): fast_chol=True reuses the Cholesky factor of the previous block's own
        # group for the next block of the same group (valid because the group's inputs did not change) and
        # skips the redundant symmetrisation inside cholesky.  Pure re-scheduling: same proposals, same
        # accept decisions, results identical to fast_chol=False (checked bit for bit in the S1 gate).
        self.fast_chol = bool(fast_chol)
        seg = PCN_SEGMENT_LENGTH if pcn_segment_length is None else int(pcn_segment_length)
        if seg <= 0 or self.pcn_steps % seg != 0:
            raise ValueError(f"pcn_steps={pcn_steps} must be a positive multiple of {seg}")
        self.pcn_segment_length = seg
        self.n_pcn_segments = self.pcn_steps // seg
        self.n_protocols = graph.n_protocols
        self.n_candidates = graph.n_candidates(G)
        self.grid_size = graph.latent_dim(G)  # whitened latent width
        self.n_pcn_blocks = graph.n_types
        self.da_subset = 0
        # 2026-11-03 additive (phase 3): exact null-space refresh of the PENULTIMATE (leaf) types every
        # ``l2_null_refresh``-th sweep (0 = off = the old kernel, bit for bit; see ``_refresh_structure``).
        self.l2_null_refresh = int(l2_null_refresh)
        if self.l2_null_refresh < 0:
            raise ValueError("l2_null_refresh must be >= 0")
        # 2026-11-04 additive: per-rejuvenation mixing diagnostics (host-side only; the kernel is untouched).
        self.mixing_diagnostics = bool(mixing_diagnostics)
        self.last_mixing_diag = None
        self.last_assim_diag = None
        # 2026-11-04 additive: collapsed (partially centred) pCN moves of the INNER types (0 = off = old kernel).
        # See ``_make_collapsed``.  Requires the null refresh with K = 1 (the move's re-expression step replaces
        # that sweep's plain refresh).
        self.inner_collapsed_sweeps = int(inner_collapsed_sweeps)
        self.inner_collapsed_beta0 = float(inner_collapsed_beta0)
        # mode 1: every inner move holds the penultimate outputs at the observed cells fixed.  mode 2: in
        # addition the layer-1 moves hold the middle-type outputs at the needed cells fixed (``_make_mid_collapse``).
        self.inner_collapsed_mode = int(inner_collapsed_mode)
        if self.inner_collapsed_mode not in (1, 2):
            raise ValueError("inner_collapsed_mode must be 1 or 2")
        # 2026-11-05 additive (option C): the collapsed inner move as ONE joint move of all inner blocks:
        # "pcn" = the 2026-11-04 per-block pCN moves (default), "hmc" = Hilbert-space HMC (exact Gaussian rotation +
        # gradient kicks, ``inner_hmc_steps`` leapfrog steps, step size adapted per segment to acceptance 0.65),
        # "ess" = elliptical slice sampling with at most ``inner_ess_shrinks`` bracket shrinks (stay put if none
        # accepted).  Target = the mode-1 collapsed density N(xi_in) N(h_obs; 0, S(xi_in)).
        self.inner_move = str(inner_move)
        self.inner_hmc_steps = int(inner_hmc_steps)
        self.inner_ess_shrinks = int(inner_ess_shrinks)
        if self.inner_move not in ("pcn", "hmc", "ess"):
            raise ValueError("inner_move must be pcn, hmc or ess")
        if self.inner_move != "pcn" and self.inner_collapsed_mode != 1:
            raise ValueError("inner_move hmc/ess requires inner_collapsed_mode 1")
        # 2026-11-05 additive: elliptical slice move of h_obs (penultimate outputs at the observed cells) under its
        # exact Gaussian conditional prior N(0, S(xi_in)) and the RB terminal likelihood, with at most
        # ``h_ess_shrinks`` evaluations (0 = off), followed by the C redraw with the new h_obs.  Needs the collapsed
        # moves on (inner_collapsed_sweeps > 0, mode 1).
        self.h_ess_shrinks = int(h_ess_shrinks)
        if self.h_ess_shrinks < 0 or (self.h_ess_shrinks > 0 and (int(inner_collapsed_sweeps) <= 0 or int(inner_collapsed_mode) != 1)):
            raise ValueError("h_ess_shrinks > 0 requires inner_collapsed_sweeps > 0 and inner_collapsed_mode 1")
        if self.inner_hmc_steps < 1 or self.inner_ess_shrinks < 1:
            raise ValueError("inner_hmc_steps and inner_ess_shrinks must be >= 1")
        if self.inner_collapsed_sweeps < 0:
            raise ValueError("inner_collapsed_sweeps must be >= 0")
        if self.inner_collapsed_sweeps > 0 and self.l2_null_refresh != 1:
            raise ValueError("inner_collapsed_sweeps > 0 requires l2_null_refresh == 1")
        if self.inner_collapsed_sweeps > 0 and int(terminal_lowrank) > 0:
            raise NotImplementedError("inner_collapsed_sweeps is implemented for the exact terminal only")
        self._cm_beta = None
        self.extra_mixing_diag_shapes = None
        self.terminal_lowrank = int(terminal_lowrank)
        if self.terminal_lowrank < 0 or 0 < self.terminal_lowrank < 8:
            raise ValueError("terminal_lowrank must be 0 or >= 8")
        self.terminal_lowrank_min_n = int(terminal_lowrank_min_n)
        self.lowrank_moments = (
            self.terminal_lowrank > 0 if lowrank_moments is None else bool(lowrank_moments)
        )
        self.lr_chunk_target_bytes = int(lr_chunk_target_bytes)
        self.last_lowrank_logw_corr = None
        self.last_lowrank_stats = None
        self.particle_sharding = NamedSharding(Mesh(np.asarray(devices), ("device",)), P("device"))
        self._chunk_target_bytes = int(chunk_target_bytes)
        self._jitter_const = float(kernel_jitter)
        self._noise_variance = float(noise_variance)

        # ---- static structure -------------------------------------------------------
        self.block_bounds = graph.block_bounds(G)
        n_types = graph.n_types
        self._group_of_type: dict[int, int] = {}
        for gi, grp in enumerate(graph.chol_groups):
            for t in grp:
                self._group_of_type[t] = gi
        # layer-1 factors (fixed numpy Cholesky of K(Z_t), jitter as the existing backend)
        self._layer1_chol_T: dict[int, jax.Array] = {}
        jitter_used = 0.0
        for t in range(n_types):
            if graph.is_layer1(t):
                ty = graph.types[t]
                n_t = len(graph.type_nodes[t])
                z = np.tile(np.asarray(warped_x, dtype=np.float64), n_t)[:, None]
                chol, jit = stable_cholesky(rbf_kernel_nd(z, ty.amplitude, ty.lengthscale), kernel_jitter)
                jitter_used = max(jitter_used, jit)
                self._layer1_chol_T[t] = jnp.asarray(chol.T, dtype=jnp.float64)
        self.layer1_jitter_used = jitter_used
        # dirty sets: types to recompute when block k is updated (k and its descendants)
        parent_types: list[set[int]] = [set() for _ in range(n_types)]
        for i, nd in enumerate(graph.nodes):
            for q in nd.parents:
                parent_types[graph.node_type[i]].add(graph.node_type[q])
        self._group_parent_types: dict[int, frozenset[int]] = {}
        for gi, grp in enumerate(graph.chol_groups):
            self._group_parent_types[gi] = frozenset(parent_types[grp[0]])
        self._dirty: list[frozenset[int]] = []
        for k in range(n_types):
            dirty = {k}
            for t in range(k + 1, n_types):
                if parent_types[t] & dirty:
                    dirty.add(t)
            self._dirty.append(frozenset(dirty))
        self._proto_nodes = np.asarray(graph.protocols, dtype=np.int32)
        max_n = max(len(nodes) * G for t, nodes in enumerate(graph.type_nodes) if not graph.is_layer1(t))
        self.chunk_size_local = compute_chunk_size(self.local_particles, max_n, chunk_target_bytes)
        self._bundles: dict[int, dict] = {}
        self._lr_cache: dict[tuple[int, int], Any] = {}
        self._refresh = self._refresh_structure() if self.l2_null_refresh > 0 else None
        if self.inner_collapsed_sweeps > 0:
            rset = set(self._refresh["types"])
            self._cm_inner = tuple(t for t in range(n_types) if t not in rset)
            self._cm_beta = np.full(len(self._cm_inner), self.inner_collapsed_beta0, dtype=np.float64)
            self.extra_mixing_diag_shapes = {
                "cm_acc": (self.n_pcn_segments, len(self._cm_inner)),
                "cm_beta": (self.n_pcn_segments, len(self._cm_inner)),
            }

        def initialise_local(key: jax.Array) -> jax.Array:
            return jax.random.normal(key, (self.local_particles, self.grid_size), dtype=jnp.float64)

        self._init = jax.pmap(initialise_local, devices=self.devices)
        self._build_forward()
        self._bundles[self.n_max] = self._build_bundle(self.n_max)

    # ------------------------------------------------------------------ forward pass
    def _build_forward(self) -> None:
        graph = self.graph
        G = self.G
        jitter = self._jitter_const
        bounds = self.block_bounds
        n_nodes = graph.n_nodes
        group_of_type = self._group_of_type
        layer1_chol_T = self._layer1_chol_T
        proto_nodes = self._proto_nodes
        n_protocols = self.n_protocols
        eye_cache: dict[int, np.ndarray] = {}

        def eye(n: int) -> np.ndarray:  # numpy constant (never a tracer)
            if n not in eye_cache:
                eye_cache[n] = np.eye(n, dtype=np.float64)
            return eye_cache[n]

        fast = self.fast_chol
        if fast:
            def chol_fn(cov):  # symmetric by construction: skipping (a + a^T)/2 changes no bit
                return jax.lax.linalg.cholesky(cov, symmetrize_input=False)
        else:
            chol_fn = jnp.linalg.cholesky
        self._chol_fn = chol_fn  # 2026-11-03 additive: the null-space refresh factorises with the same function

        def forward_chunk(state_chunk: jax.Array, cache: jax.Array | None, dirty: frozenset[int] | None,
                          reuse: Any = None, want: int | None = None):
            nodes: list[Any] = [None] * n_nodes
            chols: dict[int, jax.Array] = {}
            if reuse is not None:
                chols[reuse[0]] = reuse[1]
            for t in range(graph.n_types):
                ids = graph.type_nodes[t]
                if dirty is not None and t not in dirty:
                    for i in ids:
                        nodes[i] = cache[:, i, :]
                    continue
                start, end = bounds[t]
                xi = state_chunk[:, start:end]
                if graph.is_layer1(t):
                    f = xi @ layer1_chol_T[t]
                else:
                    gi = group_of_type[t]
                    if gi not in chols:
                        ty = graph.types[t]
                        zs = [
                            jnp.concatenate([nodes[graph.nodes[i].parents[d]] for i in ids], axis=1)
                            for d in range(ty.arity)
                        ]
                        n_t = len(ids) * G
                        cov = (
                            ty.amplitude**2 * jnp.exp(-0.5 * _sq_dist_terms(zs, ty.lengthscale))
                            + jitter * eye(n_t)[None, :, :]
                        )
                        chols[gi] = chol_fn(cov)
                    f = jnp.einsum("pij,pj->pi", chols[gi], xi)
                for s, i in enumerate(ids):
                    nodes[i] = f[:, s * G:(s + 1) * G]
            out = jnp.stack(nodes, axis=1)  # (c, n_nodes, G)
            if want is None:
                return out
            return out, chols.get(want)

        chunk = self.chunk_size_local

        def forward(state: jax.Array, cache: jax.Array | None = None, dirty: frozenset[int] | None = None,
                    reuse: Any = None, want: int | None = None):
            """``reuse=(group, chol (P,N,N))``: factor valid for this call (caller guarantees the group's inputs
            are unchanged); ``want=group``: also return that group's factor (P,N,N) (None if not computed)."""
            total = state.shape[0]
            if chunk >= total:
                return forward_chunk(state, cache, dirty, reuse, want)
            n_chunks = total // chunk
            st = state.reshape(n_chunks, chunk, state.shape[-1])
            ca = None if cache is None else cache.reshape(n_chunks, chunk, n_nodes, G)
            ru = None if reuse is None else reuse[1].reshape(n_chunks, chunk, *reuse[1].shape[1:])
            gi_r = None if reuse is None else reuse[0]

            def body(t):
                s_c, c_c, r_c = t
                return forward_chunk(s_c, c_c, dirty, None if r_c is None else (gi_r, r_c), want)

            out = jax.lax.map(body, (st, ca, ru))
            if want is None:
                return out.reshape(total, n_nodes, G)
            nodes_o, chol_o = out
            nodes_o = nodes_o.reshape(total, n_nodes, G)
            return nodes_o, (None if chol_o is None else chol_o.reshape(total, *chol_o.shape[2:]))

        def penult(nodes: jax.Array) -> jax.Array:
            return nodes[:, proto_nodes, :].reshape(nodes.shape[0], n_protocols * G)

        def pushforward(state: jax.Array) -> jax.Array:
            return penult(forward(state))

        self._forward = forward
        self._penult = penult
        self._pushforward = pushforward

    # ------------------------------------------------------------------ null-space refresh (2026-11-03)
    def _refresh_structure(self) -> dict:
        """Static structure of the exact null-space refresh (``l2_null_refresh > 0``).

        Refreshed types = the non-layer-1 types with no descendant type (``_dirty[t] == {t}``): the
        PENULTIMATE types.  For such a type t the terminal likelihood depends on its whitened block xi_t
        only through ``A xi_t`` with ``A`` = the rows of ``L_t = chol(K(Z_t) + jitter I)`` at the observed
        (node, grid) cells of t, and ``L_t`` does not depend on xi_t (only on earlier types).  The
        conditional prior of xi_t given ``A xi_t`` is therefore sampled exactly by
        ``xi' = eta + A^T (A A^T)^{-1} A (xi - eta)``, ``eta ~ N(0, I)``.  Inner types are NOT refreshed
        (their factor feeds the descendants' inputs, so their null space is not likelihood-free).

        Requirements (checked; true for every BENCH9 library): every protocol node belongs to a refreshed
        type, and all refreshed types have the same block width, arity, lengthscale and amplitude.
        Refreshed types are processed per Cholesky group (types with identical inputs share ``L``)."""
        graph, G = self.graph, self.G
        rtypes = tuple(
            t for t in range(graph.n_types)
            if self._dirty[t] == frozenset({t}) and not graph.is_layer1(t)
        )
        proto_types = sorted({graph.node_type[i] for i in graph.protocols})
        missing = [graph.types[t].name for t in proto_types if t not in rtypes]
        if missing:
            raise ValueError(
                f"{graph.name}: l2_null_refresh needs every protocol node in a leaf non-layer-1 type; "
                f"types {missing} hold protocol nodes but are not refreshable"
            )
        widths = {len(graph.type_nodes[t]) * G for t in rtypes}
        sigs = {(graph.types[t].arity, graph.types[t].lengthscale, graph.types[t].amplitude) for t in rtypes}
        if len(widths) != 1 or len(sigs) != 1:
            raise NotImplementedError(
                f"{graph.name}: l2_null_refresh needs equal block width / kernel over the refreshed types "
                f"(got widths {sorted(widths)}, kernels {sorted(sigs)})"
            )
        (arity, lengthscale, amplitude), = sigs
        rindex = {t: r for r, t in enumerate(rtypes)}
        groups = []
        for grp in graph.chol_groups:
            members = tuple(t for t in grp if t in rindex)
            if members:
                groups.append(members)
        group_pos = {t: gp for gp, members in enumerate(groups) for t in members}
        n_cand = self.n_candidates
        rt_tab = np.zeros(n_cand, dtype=np.int32)
        rw_tab = np.zeros(n_cand, dtype=np.int32)
        gp_tab = np.zeros(n_cand, dtype=np.int32)
        lt_tab = np.zeros(n_cand, dtype=np.int32)
        g_tab = np.zeros(n_cand, dtype=np.int32)
        par_tab = np.zeros((n_cand, arity), dtype=np.int32)
        for p, node in enumerate(graph.protocols):
            t = graph.node_type[node]
            sl = slice(p * G, (p + 1) * G)
            rt_tab[sl] = rindex[t]
            rw_tab[sl] = graph.node_slot[node] * G + np.arange(G)  # row of L_t (node slot-major, grid minor)
            gp_tab[sl] = group_pos[t]
            lt_tab[sl] = groups[group_pos[t]].index(t)
            g_tab[sl] = np.arange(G)
            par_tab[sl, :] = np.asarray(graph.nodes[node].parents, dtype=np.int32)[None, :]
        return {
            "types": rtypes,
            "rindex": rindex,
            "groups": tuple(groups),
            "width": widths.pop(),
            "arity": int(arity),
            "lengthscale": float(lengthscale),
            "amplitude": float(amplitude),
            "tables": {"rt": rt_tab, "rw": rw_tab, "gp": gp_tab, "lt": lt_tab, "g": g_tab, "par": par_tab},
        }

    def _make_refresh(self, n_max: int):
        """``refresh(current, current_nodes, key, obs_indices, active) -> (current, current_nodes)``: one
        exact null-space refresh of every refreshed type (``_refresh_structure``).  Same construction as
        ``SharedProtocolGpnBackend``'s ``l2_null_refresh_fn`` (gpn_smc/backends.py): repeated observations
        of one (type, row) cell constrain once; Gram ``A A^T`` from the kernel formula
        ``K(z_o, z_o') + jitter [same cell]`` (``= L L^T`` at the observed rows), masked to equal types,
        ``1e-10`` on the active diagonal, identity on inactive/duplicate rows (one Gram over all observed rows,
        block diagonal by type, as the reference's non-typed refresh); ``REFINE_STEPS``
        iterative-refinement steps against the exact rows ``A`` bring ``A (xi' - xi)`` to roundoff.
        ``eta`` has shape ``(n, n_refreshed_types, width)`` (for the PB-W16 graph: the reference's
        ``(n, n_l2, 2G)`` draw).  The node outputs of the refreshed types are recomputed as ``L_t xi'_t``
        (same einsum as the forward pass); the carried log-likelihood is left unchanged (it is invariant
        up to roundoff, as in the reference).  Particles are processed in chunks sized from
        ``2 * chunk_target_bytes`` (``--graph-chunk-mb``) for the per-particle footprint of all group factors
        + row tables + Gram (``side`` below)."""
        from .backends import REFINE_STEPS

        rs = self._refresh
        graph, G = self.graph, self.G
        N, arity = rs["width"], rs["arity"]
        amp, ls = rs["amplitude"], rs["lengthscale"]
        jitter = self._jitter_const
        chol_fn = self._chol_fn
        bounds = self.block_bounds
        rtypes, groups, rindex = rs["types"], rs["groups"], rs["rindex"]
        n_rt = len(rtypes)
        tabs = {k: jnp.asarray(v) for k, v in rs["tables"].items()}
        eye_n = np.eye(N, dtype=np.float64)
        eye_m = np.eye(n_max, dtype=np.float64)
        tril_strict = np.tril(np.ones((n_max, n_max), dtype=bool), k=-1)
        n_g = len(groups)
        # per particle: the n_g group factors (kept until the node update) + one transient covariance +
        # the (n_max, N) row tables + the (n_max, n_max) Gram / its factor / distance terms
        side = int(np.ceil(np.sqrt((n_g + 1) * N * N + 3 * n_max * N + 3 * n_max * n_max)))
        target = 2 * self._chunk_target_bytes
        group_nodes = [graph.type_nodes[members[0]] for members in groups]

        def prepare(obs_indices, active):
            rt = tabs["rt"][obs_indices]
            rw = tabs["rw"][obs_indices]
            gp = tabs["gp"][obs_indices]
            same_cell = (rt[:, None] == rt[None, :]) & (rw[:, None] == rw[None, :])
            act_b = active > 0.5
            dup_earlier = jnp.any(same_cell & tril_strict & act_b[None, :], axis=1)
            uact = (act_b & ~dup_earlier).astype(jnp.float64)
            return {
                "rt": rt,
                "rw": rw,
                "in_g": [gp == gpos for gpos in range(n_g)],
                "g": tabs["g"][obs_indices],
                "par": tabs["par"][obs_indices],
                "uact": uact,
                "same_cell": same_cell.astype(jnp.float64),
                "same_type": (rt[:, None] == rt[None, :]).astype(jnp.float64),
                "onehot": (rt[:, None] == jnp.arange(n_rt)[None, :]).astype(jnp.float64),
            }

        def solve(lc, rhs):
            return jax.scipy.linalg.cho_solve((lc, True), rhs[..., None])[..., 0]

        def chunk_fn(st_c, nodes_c, eta_c, prep, h_c=None):
            """One chunk: the group factors L_g (same expression as forward_chunk), the observed rows
            A (row o = L_{g(o)}[rw_o]), ONE Gram over all observed rows (block diagonal by type through
            ``same_type``; the reference's ``l2_null_refresh_fn`` structure), refinement, node update."""
            rw, uact = prep["rw"], prep["uact"]
            xi_all = jnp.stack([st_c[:, bounds[t][0]:bounds[t][1]] for t in rtypes], axis=1)  # (c, n_rt, N)
            d_all = xi_all - eta_c
            chols = []
            c_rows = None
            for gpos, members in enumerate(groups):
                ids = group_nodes[gpos]
                zs = [
                    jnp.concatenate([nodes_c[:, graph.nodes[i].parents[d]] for i in ids], axis=1)
                    for d in range(arity)
                ]
                cov = amp**2 * jnp.exp(-0.5 * _sq_dist_terms(zs, ls)) + jitter * eye_n[None, :, :]
                chol = chol_fn(cov)
                chols.append(chol)
                rows_g = chol[:, rw, :]  # (c, n_max, N)
                c_rows = rows_g if c_rows is None else jnp.where(prep["in_g"][gpos][None, :, None], rows_g, c_rows)
            if h_c is None:
                d_obs = d_all[:, prep["rt"], :]  # (c, n_max, N): xi - eta of each observation's type
                u = jnp.sum(c_rows * d_obs, axis=-1) * uact[None, :]
            else:
                # 2026-11-04 additive (inner collapsed moves): target the given observed-cell values h_c
                # (c, n_max) under the CURRENT rows A: u = h - A eta, so A xi' = h (re-expression of xi_t).
                u = (h_c - jnp.sum(c_rows * eta_c[:, prep["rt"], :], axis=-1)) * uact[None, :]
            # Gram = A A^T = K(z_o, z_o') + jitter [same cell] (= L L^T at the observed rows), equal types only
            zo = [nodes_c[:, prep["par"][:, d], prep["g"]] for d in range(arity)]  # (c, n_max) inputs at obs
            cov_o = amp**2 * jnp.exp(-0.5 * _sq_dist_terms(zo, ls)) + jitter * prep["same_cell"][None]
            pair = uact[:, None] * uact[None, :] * prep["same_type"]
            gram = cov_o * pair[None] + eye_m[None] * ((1.0 - uact) + 1e-10 * uact)[None, :, None]
            lc = jnp.linalg.cholesky(gram)
            w = solve(lc, u)
            # iterative refinement against the exact rows A: A (xi' - xi) -> roundoff
            for _it in range(REFINE_STEPS + 1):
                s_t = jnp.einsum("ot,pok->ptk", prep["onehot"], w[:, :, None] * c_rows)  # (c, n_rt, N)
                if _it == REFINE_STEPS:
                    break
                r = (u - jnp.sum(c_rows * s_t[:, prep["rt"], :], axis=-1)) * uact[None, :]
                w = w + solve(lc, r)
            xi_new = eta_c + s_t
            new_f: list[Any] = [None] * n_rt
            for gpos, members in enumerate(groups):
                for t in members:
                    new_f[rindex[t]] = jnp.einsum("pij,pj->pi", chols[gpos], xi_new[:, rindex[t]])
            return xi_new, jnp.stack(new_f, axis=1)

        def refresh(current, current_nodes, key, obs_indices, active, h_target=None):
            prep = prepare(obs_indices, active)
            n_p = current.shape[0]
            eta = jax.random.normal(key, (n_p, n_rt, N), dtype=jnp.float64)
            ch = compute_chunk_size(n_p, side, target)
            if ch >= n_p:
                nx, nf = chunk_fn(current, current_nodes, eta, prep, h_target)
            elif h_target is None:
                nc = n_p // ch
                resh = lambda a_: a_.reshape(nc, ch, *a_.shape[1:])  # noqa: E731
                nx, nf = jax.lax.map(
                    lambda t_: chunk_fn(t_[0], t_[1], t_[2], prep), (resh(current), resh(current_nodes), resh(eta))
                )
                nx = nx.reshape(n_p, n_rt, N)
                nf = nf.reshape(n_p, n_rt, N)
            else:  # 2026-11-04 additive (h_target given)
                nc = n_p // ch
                resh = lambda a_: a_.reshape(nc, ch, *a_.shape[1:])  # noqa: E731
                nx, nf = jax.lax.map(
                    lambda t_: chunk_fn(t_[0], t_[1], t_[2], prep, t_[3]),
                    (resh(current), resh(current_nodes), resh(eta), resh(h_target)),
                )
                nx = nx.reshape(n_p, n_rt, N)
                nf = nf.reshape(n_p, n_rt, N)
            for r, t in enumerate(rtypes):
                s0, e0 = bounds[t]
                current = current.at[:, s0:e0].set(nx[:, r])
                ids = np.asarray(graph.type_nodes[t], dtype=np.int32)
                current_nodes = current_nodes.at[:, ids, :].set(nf[:, r].reshape(n_p, len(ids), G))
            return current, current_nodes

        return refresh

    # ------------------------------------------------------------------ inner collapsed moves (2026-11-04)
    def _make_collapsed(self, n_max: int, forward):
        """Collapsed pCN moves of the inner (non-refreshed) types.

        Target over (xi_inner, xi_pen): N(xi; 0, I) l(h_obs) with h_obs = A(xi_inner) xi_pen the penultimate
        outputs at the observed cells (the RB terminal likelihood depends on xi_pen only through h_obs).
        Change variables xi_pen -> (h_obs, rest): under the prior, h_obs | xi_inner ~ N(0, S(xi_inner)) with
        S = K_pen(z_o, z_o') + jitter [same cell] (equal types only; the rows A A^T), and xi_pen | (xi_inner,
        h_obs) is the Gaussian conditional sampled exactly by the null refresh with target h_obs.  So the
        marginal target of (xi_inner, h_obs) is N(xi_inner) N(h_obs; 0, S(xi_inner)) l(h_obs).  A pCN move of
        an inner block with h_obs HELD FIXED is accepted with log ratio log N(h_obs; 0, S') - log N(h_obs; 0, S)
        (the likelihood cancels; pCN is prior-reversible).  After the inner moves, xi_pen is redrawn from its
        exact conditional given (xi_inner, h_obs) (``refresh(..., h_target=h_obs)``).  Each step leaves the
        posterior invariant; the composite is the partially-collapsed Gibbs sampler of (xi_inner, h_obs).

        Returns ``(prep_fn, h_obs_fn, log_marg_fn, inner_dirty)``."""
        rs = self._refresh
        graph, G = self.graph, self.G
        arity = rs["arity"]
        amp, ls = rs["amplitude"], rs["lengthscale"]
        jitter = self._jitter_const
        tabs = {k: jnp.asarray(v) for k, v in rs["tables"].items()}
        rset = set(rs["types"])
        inner_dirty = {k: frozenset(t for t in self._dirty[k] if t not in rset) for k in self._cm_inner}
        eye_m = np.eye(n_max, dtype=np.float64)
        tril_strict = np.tril(np.ones((n_max, n_max), dtype=bool), k=-1)
        proto_nodes = jnp.asarray(self._proto_nodes)
        side = int(np.ceil(np.sqrt(5 * n_max * n_max + graph.n_nodes * G)))
        target = self._chunk_target_bytes
        variant = getattr(self, "_cm_variant", "exact")  # test hook; only "exact" is a valid kernel

        def prep_fn(obs_indices, active):
            rt = tabs["rt"][obs_indices]
            rw = tabs["rw"][obs_indices]
            same_cell = (rt[:, None] == rt[None, :]) & (rw[:, None] == rw[None, :])
            act_b = active > 0.5
            dup_earlier = jnp.any(same_cell & tril_strict & act_b[None, :], axis=1)
            uact = (act_b & ~dup_earlier).astype(jnp.float64)
            return {
                "g": tabs["g"][obs_indices],
                "par": tabs["par"][obs_indices],
                "uact": uact,
                "same_cell": same_cell.astype(jnp.float64),
                "same_type": (rt[:, None] == rt[None, :]).astype(jnp.float64),
                "node": proto_nodes[obs_indices // G],
                "first": jnp.argmax(same_cell & act_b[None, :], axis=1),  # 2026-11-05: first active row of the cell
                "act": act_b.astype(jnp.float64),
            }

        def h_obs_fn(nodes, prep):
            return nodes[:, prep["node"], prep["g"]]  # (P, n_max)

        def lm_chunk(nodes_c, h_c, prep):
            uact = prep["uact"]
            zo = [nodes_c[:, prep["par"][:, d], prep["g"]] for d in range(arity)]
            cov_o = amp**2 * jnp.exp(-0.5 * _sq_dist_terms(zo, ls)) + jitter * prep["same_cell"][None]
            pair = uact[:, None] * uact[None, :] * prep["same_type"]
            gram = cov_o * pair[None] + eye_m[None] * (1.0 - uact)[None, :, None]
            lc = jnp.linalg.cholesky(gram)
            alpha = jax.scipy.linalg.solve_triangular(lc, (h_c * uact[None, :])[..., None], lower=True)[..., 0]
            quad = -0.5 * jnp.sum(alpha * alpha, axis=-1)
            logdet = jnp.sum(jnp.log(jnp.diagonal(lc, axis1=-2, axis2=-1)), axis=-1)
            if variant == "no_logdet":  # deliberately WRONG (gate power check only)
                return quad
            if variant == "accept_all":  # deliberately WRONG (gate power check only)
                return 0.0 * quad
            return quad - logdet

        def log_marg_fn(nodes, h, prep):
            n_p = nodes.shape[0]
            ch = compute_chunk_size(n_p, side, target)
            if ch >= n_p:
                return lm_chunk(nodes, h, prep)
            nc = n_p // ch
            out = jax.lax.map(lambda t_: lm_chunk(t_[0], t_[1], prep),
                              (nodes.reshape(nc, ch, *nodes.shape[1:]), h.reshape(nc, ch, h.shape[-1])))
            return out.reshape(n_p)

        def nu_chunk(nodes_c, z_c, prep):
            """2026-11-05: a draw nu ~ N(0, S) per particle (rows = observations; duplicates copy the first active
            row of their cell, inactive rows 0) from standard normals z_c (c, n_max)."""
            uact = prep["uact"]
            zo = [nodes_c[:, prep["par"][:, d], prep["g"]] for d in range(arity)]
            cov_o = amp**2 * jnp.exp(-0.5 * _sq_dist_terms(zo, ls)) + jitter * prep["same_cell"][None]
            pair = uact[:, None] * uact[None, :] * prep["same_type"]
            gram = cov_o * pair[None] + eye_m[None] * (1.0 - uact)[None, :, None]
            lc = jnp.linalg.cholesky(gram)
            nu = jnp.einsum("pij,pj->pi", lc, z_c * uact[None, :])
            if variant == "h_nu_white":  # deliberately WRONG (gate power check only): nu ~ N(0, I)
                nu = z_c * uact[None, :]
            return nu[:, prep["first"]] * prep["act"][None, :]

        def nu_fn(nodes, z, prep):
            n_p = nodes.shape[0]
            ch = compute_chunk_size(n_p, side, target)
            if ch >= n_p:
                return nu_chunk(nodes, z, prep)
            nc = n_p // ch
            out = jax.lax.map(lambda t_: nu_chunk(t_[0], t_[1], prep),
                              (nodes.reshape(nc, ch, *nodes.shape[1:]), z.reshape(nc, ch, z.shape[-1])))
            return out.reshape(n_p, z.shape[-1])

        if not hasattr(self, "_cm_nu_fns"):
            self._cm_nu_fns = {}
        self._cm_nu_fns[n_max] = nu_fn
        if not hasattr(self, "_cm_lm_chunks"):
            self._cm_lm_chunks = {}
        self._cm_lm_chunks[n_max] = lm_chunk  # 2026-11-05 additive (joint inner moves)
        return prep_fn, h_obs_fn, log_marg_fn, inner_dirty

    def _make_mid_collapse(self, n_max: int):
        """Mode 2 pieces (2026-11-04 additive).  Graphs of depth 3 inner layers only: layer-1 types L1, middle
        types M (all parents in L1), penultimate types (all parents in M).  For an observed cell (p, g) the
        penultimate input z_o is the vector of its parent (middle) node outputs at the SAME grid index g, so the
        "needed" middle cells are {(parent node, g)}.  Holding b = middle outputs at the needed cells (and h_obs)
        fixed, a layer-1 pCN move has log ratio sum_t [log N(b_t; 0, K_t(z'_t) + jitter I) - (same at z_t)]
        (z_t = layer-1 outputs at the needed cells); the penultimate prior density depends on the middle layer
        only through b, so it cancels.  Afterwards the middle blocks are redrawn exactly from their conditional
        given (layer-1, b) (null refresh with target b, full-row masked form).
        Returns (prep_mid, b_fn, lmb_fn, mid_redraw)."""
        graph, G = self.graph, self.G
        rset = set(self._refresh["types"])
        l1 = [t for t in range(graph.n_types) if graph.is_layer1(t)]
        mid = [t for t in range(graph.n_types) if t not in rset and not graph.is_layer1(t)]
        node_t = graph.node_type
        for t in rset:
            for i in graph.type_nodes[t]:
                if any(node_t[q] not in mid for q in graph.nodes[i].parents):
                    raise NotImplementedError("inner_collapsed_mode 2: penultimate parents must all be middle types")
        for t in mid:
            for i in graph.type_nodes[t]:
                if any(node_t[q] not in l1 for q in graph.nodes[i].parents):
                    raise NotImplementedError("inner_collapsed_mode 2: middle parents must all be layer-1 types")
        jitter = self._jitter_const
        bounds = self.block_bounds
        tabs = {}
        for t in mid:
            ids = graph.type_nodes[t]
            R = len(ids) * G
            tabs[t] = {
                "node": jnp.asarray(np.repeat(np.asarray(ids, dtype=np.int32), G)),
                "g": jnp.asarray(np.tile(np.arange(G, dtype=np.int32), len(ids))),
                "par": jnp.asarray(np.repeat(np.asarray([graph.nodes[i].parents for i in ids], dtype=np.int32), G, axis=0)),
                "R": R,
                "amp": graph.types[t].amplitude,
                "ls": graph.types[t].lengthscale,
                "arity": graph.types[t].arity,
            }
        groups = []
        for grp in graph.chol_groups:
            members = tuple(t for t in grp if t in mid)
            if members:
                groups.append(members)
        for t in mid:
            if not any(t in g_ for g_ in groups):
                groups.append((t,))
        rtabs = self._refresh["tables"]
        par_c = jnp.asarray(rtabs["par"])
        g_c = jnp.asarray(rtabs["g"])
        from .backends import REFINE_STEPS
        Rmax = max(tabs[t]["R"] for t in mid)
        side = int(np.ceil(np.sqrt(6 * Rmax * Rmax + graph.n_nodes * G)))
        target = self._chunk_target_bytes
        chol_fn = self._chol_fn
        mid_variant = getattr(self, "_cm_variant", "exact")  # test hook; only "exact" is a valid kernel

        def prep_mid(obs_indices, active):
            act = active > 0.5
            par_o = par_c[obs_indices]  # (n_max, arity_c)
            g_o = g_c[obs_indices]
            out = {}
            for t in mid:
                tb = tabs[t]
                hit = (par_o[:, :, None] == tb["node"][None, None, :]) & (g_o[:, None, None] == tb["g"][None, None, :])
                out[t] = jnp.any(hit & act[:, None, None], axis=(0, 1)).astype(jnp.float64)  # (R,)
            return out

        def b_fn(nodes, pm):
            return {t: nodes[:, tabs[t]["node"], tabs[t]["g"]] * pm[t][None, :] for t in mid}

        def cov_t(nodes_c, t):
            tb = tabs[t]
            z = [nodes_c[:, tb["par"][:, d], tb["g"]] for d in range(tb["arity"])]
            return tb["amp"] ** 2 * jnp.exp(-0.5 * _sq_dist_terms(z, tb["ls"])) + jitter * np.eye(tb["R"])[None]

        def lmb_chunk(nodes_c, b_c, pm):
            tot = 0.0
            for t in mid:
                msk = pm[t]
                gram = cov_t(nodes_c, t) * (msk[:, None] * msk[None, :])[None] + np.eye(tabs[t]["R"])[None] * (1.0 - msk)[None, :, None]
                lc = jnp.linalg.cholesky(gram)
                a = jax.scipy.linalg.solve_triangular(lc, b_c[t][..., None], lower=True)[..., 0]
                tot = tot - 0.5 * jnp.sum(a * a, -1)
                if mid_variant != "mid_no_logdet":  # "mid_no_logdet" is deliberately WRONG (gate power check only)
                    tot = tot - jnp.sum(jnp.log(jnp.diagonal(lc, axis1=-2, axis2=-1)), -1)
            return tot

        def _chunked(fn, n_p, *arrs):
            ch = compute_chunk_size(n_p, side, target)
            if ch >= n_p:
                return fn(*arrs)
            nc = n_p // ch
            resh = lambda a_: jax.tree_util.tree_map(lambda x: x.reshape(nc, ch, *x.shape[1:]), a_)  # noqa: E731
            out = jax.lax.map(lambda t_: fn(*t_), tuple(resh(a) for a in arrs))
            return jax.tree_util.tree_map(lambda x: x.reshape(n_p, *x.shape[2:]), out)

        def lmb_fn(nodes, b, pm):
            return _chunked(lambda n_, b_: lmb_chunk(n_, b_, pm), nodes.shape[0], nodes, b)

        def redraw_chunk(st_c, nodes_c, b_c, eta_c, pm):
            new_x, new_f = {}, {}
            for members in groups:
                t0 = members[0]
                cov = cov_t(nodes_c, t0)
                L = chol_fn(cov)
                for t in members:
                    msk = pm[t]
                    s0, e0 = bounds[t]
                    eta = eta_c[t]
                    gram = (cov_t(nodes_c, t) if t != t0 else cov) * (msk[:, None] * msk[None, :])[None] \
                        + np.eye(tabs[t]["R"])[None] * ((1.0 - msk) + 1e-10 * msk)[None, :, None]
                    lc = jnp.linalg.cholesky(gram)
                    u = (b_c[t] - jnp.einsum("pij,pj->pi", L, eta)) * msk[None]
                    w = jax.scipy.linalg.cho_solve((lc, True), u[..., None])[..., 0]
                    for _it in range(REFINE_STEPS + 1):
                        s_t = jnp.einsum("pij,pi->pj", L, w * msk[None])
                        if _it == REFINE_STEPS:
                            break
                        r = (u - jnp.einsum("pij,pj->pi", L, s_t)) * msk[None]
                        w = w + jax.scipy.linalg.cho_solve((lc, True), r[..., None])[..., 0]
                    xi = eta + s_t
                    new_x[t] = xi
                    new_f[t] = jnp.einsum("pij,pj->pi", L, xi)
            return new_x, new_f

        def mid_redraw(current, current_nodes, key, pm, b):
            n_p = current.shape[0]
            keys = jax.random.split(key, len(mid))
            eta = {t: jax.random.normal(keys[j], (n_p, tabs[t]["R"]), dtype=jnp.float64) for j, t in enumerate(mid)}
            nx, nf = _chunked(lambda s_, n_, b_, e_: redraw_chunk(s_, n_, b_, e_, pm), n_p, current, current_nodes, b, eta)
            for t in mid:
                s0, e0 = bounds[t]
                current = current.at[:, s0:e0].set(nx[t])
                ids = np.asarray(graph.type_nodes[t], dtype=np.int32)
                current_nodes = current_nodes.at[:, ids, :].set(nf[t].reshape(n_p, len(ids), G))
            return current, current_nodes

        return prep_mid, b_fn, lmb_fn, mid_redraw, tuple(l1), tuple(mid)

    def _make_joint_inner(self, n_max: int):
        """2026-11-05 additive: value / gradient of the collapsed log density lm(xi_in) = log N(h_obs; 0, S(xi_in))
        as a function of the concatenated inner whitened latent (all non-penultimate blocks), chunked over particles.
        Returns (inner_idx, inner_all_dirty, lm_state(state, nodes, h, prep), vg_state(state, nodes, h, prep))."""
        lm_chunk = self._cm_lm_chunks[n_max]
        forward = self._forward
        inner_idx = np.concatenate([np.arange(*self.block_bounds[k]) for k in self._cm_inner]).astype(np.int32)
        rset = set(self._refresh["types"])
        inner_all = frozenset(t for t in range(self.graph.n_types) if t not in rset)
        P = self.local_particles
        side_g = int(np.ceil(np.sqrt(16 * (128 * 128 + n_max * n_max) + 4 * self.graph.n_nodes * self.G)))
        cg = min(self.chunk_size_local, compute_chunk_size(P, side_g, self._chunk_target_bytes))

        def lm_c(st_c, nodes_c, h_c, prep):
            return lm_chunk(forward(st_c, nodes_c, inner_all), h_c, prep)

        def vg_c(st_c, nodes_c, h_c, prep):
            def f(x):
                return lm_c(st_c.at[:, inner_idx].set(x), nodes_c, h_c, prep)
            val, vjp = jax.vjp(f, st_c[:, inner_idx])
            return val, vjp(jnp.ones_like(val))[0]

        def _mapped(fn, state, nodes, h, prep):
            n_p = state.shape[0]
            if cg >= n_p:
                return fn(state, nodes, h, prep)
            nc = n_p // cg
            r = lambda a_: a_.reshape(nc, cg, *a_.shape[1:])  # noqa: E731
            out = jax.lax.map(lambda t_: fn(t_[0], t_[1], t_[2], prep), (r(state), r(nodes), r(h)))
            return jax.tree_util.tree_map(lambda a_: a_.reshape(n_p, *a_.shape[2:]), out)

        def lm_state(state, nodes, h, prep):
            return _mapped(lm_c, state, nodes, h, prep)

        def vg_state(state, nodes, h, prep):
            return _mapped(vg_c, state, nodes, h, prep)

        return jnp.asarray(inner_idx), inner_all, lm_state, vg_state

    def collapsed_logmarg_once(self, state, obs_indices, active):
        """Test hook (single device): log N(h_obs; 0, S(xi_inner)) of every particle of ``state``."""
        bundle = self._bundle(int(np.asarray(active).shape[0]))
        return bundle["_cm_logmarg_once"](state, obs_indices, active)

    def refresh_target_once(self, state, key, obs_indices, active, h_target):
        """Test hook (single device): refresh with an explicit observed-cell target (n, n_max)."""
        bundle = self._bundle(int(np.asarray(active).shape[0]))
        return bundle["_refresh_target_once"](state, key, obs_indices, active, h_target)

    def refresh_once(self, state, key, obs_indices, active):
        """Test hook (single device): one null-space refresh of ``state`` (n, latent_dim) with
        ``eta = normal(key, (n, n_refreshed_types, width))``.  Returns the refreshed state."""
        bundle = self._bundle(int(np.asarray(active).shape[0]))
        if bundle.get("_refresh_once") is None:
            raise RuntimeError("refresh_once needs l2_null_refresh > 0")
        return bundle["_refresh_once"](state, key, obs_indices, active)

    # ------------------------------------------------------------------ buckets
    def n_max_for(self, n_obs: int) -> int:
        for bucket in self.n_max_buckets:
            if bucket >= n_obs:
                return bucket
        raise ValueError(f"n_obs={n_obs} exceeds every bucket in n_max_buckets={self.n_max_buckets}")

    def _bundle(self, n_max: int) -> dict:
        if n_max not in self._bundles:
            if n_max not in self.n_max_buckets:
                raise ValueError(f"n_max={n_max} is not one of n_max_buckets={self.n_max_buckets}")
            self._bundles[n_max] = self._build_bundle(n_max)
        return self._bundles[n_max]

    def _use_lowrank(self, n_max: int) -> bool:
        return self.terminal_lowrank > 0 and n_max >= self.terminal_lowrank_min_n

    def _build_bundle(self, n_max: int) -> dict:
        spec = self.library_spec
        jitter = self._jitter_const
        G = self.G
        forward, penult, pushforward = self._forward, self._penult, self._pushforward
        bounds = self.block_bounds
        n_blocks = len(bounds)
        n_nodes = self.graph.n_nodes
        n_segments, segment_length = self.n_pcn_segments, self.pcn_segment_length
        n_particles = self.n_particles
        incremental = self.incremental
        dirty_sets = self._dirty
        rb_chunk = compute_chunk_size(self.local_particles, max(self.n_candidates, n_max), self._chunk_target_bytes)
        ll_chunk = compute_chunk_size(self.local_particles, n_max, self._chunk_target_bytes)
        loglik_from_h, moments_from_h, summary_extended_from_h = build_terminal_rb_functions(
            n_candidates=self.n_candidates,
            n_max=n_max,
            amplitude=spec.amplitude,
            lengthscale=spec.terminal_lengthscale,
            noise_variance=self._noise_variance,
            jitter=jitter,
            chunk_size=rb_chunk,
            loglik_chunk_size=ll_chunk,
        )
        adapt = self.adapt_beta

        def loglik_local(state, obs_indices, y_padded, active):
            return loglik_from_h(pushforward(state), obs_indices, y_padded, active)

        fast_chol = self.fast_chol
        group_of_type = self._group_of_type
        group_parent_types = self._group_parent_types

        def block_update(current, current_nodes, k, beta_blocks, state_key, reuse=None, want=None):
            """Draw the pCN proposal of block k; return (state_key, block_current, block_proposal,
            proposal_state, proposal_nodes, log_u)."""
            start, end = bounds[k]
            state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
            block_current = current[:, start:end]
            white = jax.random.normal(normal_key, block_current.shape, dtype=jnp.float64)
            shrink_k = jnp.sqrt(1.0 - beta_blocks[k] * beta_blocks[k])
            block_proposal = shrink_k * block_current + beta_blocks[k] * white
            proposal = current.at[:, start:end].set(block_proposal)
            chol_own = None
            if incremental and want is not None:
                nodes_prop, chol_own = forward(proposal, current_nodes, dirty_sets[k], reuse, want)
            elif incremental:
                nodes_prop = forward(proposal, current_nodes, dirty_sets[k])
            else:
                nodes_prop = forward(proposal)
            log_u = jnp.log(jax.random.uniform(uniform_key, (current.shape[0],), dtype=jnp.float64))
            if want is not None:
                return state_key, block_current, block_proposal, nodes_prop, log_u, chol_own
            return state_key, block_current, block_proposal, nodes_prop, log_u

        refresh_k = self.l2_null_refresh
        refresh_fn = self._make_refresh(n_max) if refresh_k > 0 else None
        # 2026-11-04 additive: inner collapsed moves (None = off = the old kernel; every branch below on
        # ``cm_parts`` is Python-level, so the traced program with the flag off is the old one).
        cm_sweeps = self.inner_collapsed_sweeps
        cm_parts = self._make_collapsed(n_max, forward) if cm_sweeps > 0 else None
        mid_parts = self._make_mid_collapse(n_max) if (cm_sweeps > 0 and self.inner_collapsed_mode == 2) else None
        joint_kind = self.inner_move if (cm_sweeps > 0 and self.inner_move != "pcn") else None
        joint_parts = self._make_joint_inner(n_max) if joint_kind is not None else None
        hmc_L, ess_K = self.inner_hmc_steps, self.inner_ess_shrinks
        h_K = self.h_ess_shrinks if cm_sweeps > 0 else 0
        nu_fn_h = self._cm_nu_fns[n_max] if h_K > 0 else None

        def h_ess(current_nodes, current_ll, h_fix, key, prep_c, ll_rows):
            """2026-11-05: elliptical slice update of h_obs given the inner nodes (prior N(0,S), RB likelihood).
            Returns (h_new, ll_new, n_accepted, n_extra_evaluations)."""
            k1, k2, k3, key = jax.random.split(key, 4)
            z = jax.random.normal(k1, h_fix.shape, dtype=jnp.float64)
            nu = nu_fn_h(current_nodes, z, prep_c)
            act = prep_c["act"][None, :]
            logy = current_ll + jnp.log(jax.random.uniform(k2, current_ll.shape, dtype=jnp.float64))
            th = jax.random.uniform(k3, current_ll.shape, dtype=jnp.float64) * (2 * np.pi)
            tmin, tmax = th - 2 * np.pi, th
            done = jnp.zeros(current_ll.shape, dtype=bool)
            h_new, ll_new = h_fix, current_ll
            n_ext = 0.0
            for _s in range(h_K):
                hp = jnp.where(act > 0.5, h_fix * jnp.cos(th)[:, None] + nu * jnp.sin(th)[:, None], h_fix)
                llp = ll_rows(hp)
                ok = (llp > logy) & ~done
                h_new = jnp.where(ok[:, None], hp, h_new)
                ll_new = jnp.where(ok, llp, ll_new)
                n_ext = n_ext + jnp.sum(~done) * (1.0 if _s > 0 else 0.0)
                done = done | ok
                tmin = jnp.where(~done & (th < 0), th, tmin)
                tmax = jnp.where(~done & (th >= 0), th, tmax)
                key, ks = jax.random.split(key)
                th = jnp.where(done, th, tmin + (tmax - tmin) * jax.random.uniform(ks, th.shape, dtype=jnp.float64))
            return h_new, ll_new, jnp.sum(done), n_ext
        cm_inner = self._cm_inner if cm_sweeps > 0 else ()
        n_cm = len(cm_inner)

        def make_pcn(ll_fn, refresh=None, cm=None):
            """Generic segment-blocked kernel; ``ll_fn(h) -> (c,)`` is the log-likelihood used
            for accept/reject.  Returns (pcn_local_core, carried) pieces.  2026-11-03 additive:
            ``refresh(current, current_nodes, key) -> (current, current_nodes)`` (None = old kernel)
            runs after the sweep's block moves on every sweep with ``(sweep_idx + 1) % K == 0``
            (sweep_idx = index within the segment, K = l2_null_refresh; the reference schedule), key
            ``fold_in(state_key, 0x53)`` so the main key stream is untouched."""

            def run(state, ll0, key, beta_blocks0, cm_beta0=None):
                nodes0 = forward(state)

                def pcn_sweep(carry, sweep_idx):
                    if cm is None:
                        current, current_nodes, current_ll, state_key, beta_blocks = carry
                    else:
                        current, current_nodes, current_ll, state_key, beta_blocks, cm_beta = carry
                    accepted_this_sweep = jnp.zeros(n_blocks, dtype=jnp.float64)
                    prev = None  # (group, chol) of the previous block's own group, valid for the current state
                    for k in range(n_blocks):
                        start, end = bounds[k]
                        if fast_chol and incremental:
                            own_gi = group_of_type.get(k)
                            reuse = None
                            if prev is not None and own_gi is not None and prev[0] == own_gi:
                                reuse = prev  # same group => same inputs (parents are earlier types, unchanged)
                            (state_key, block_current, block_proposal, nodes_prop, log_u, chol_own) = block_update(
                                current, current_nodes, k, beta_blocks, state_key, reuse, own_gi if own_gi is not None else -1
                            ) if own_gi is not None else (block_update(
                                current, current_nodes, k, beta_blocks, state_key) + (None,))
                            prev = (own_gi, chol_own) if (own_gi is not None and chol_own is not None) else None
                        else:
                            state_key, block_current, block_proposal, nodes_prop, log_u = block_update(
                                current, current_nodes, k, beta_blocks, state_key
                            )
                        proposal_ll = ll_fn(penult(nodes_prop))
                        accepted = log_u < (proposal_ll - current_ll)
                        block_new = jnp.where(accepted[:, None], block_proposal, block_current)
                        current = current.at[:, start:end].set(block_new)
                        current_nodes = jnp.where(accepted[:, None, None], nodes_prop, current_nodes)
                        current_ll = jnp.where(accepted, proposal_ll, current_ll)
                        accepted_this_sweep = accepted_this_sweep.at[k].add(jnp.sum(accepted))
                    if cm is not None:
                        # 2026-11-04 additive: collapsed inner sweeps with h_obs held fixed, then the exact
                        # re-expression / null refresh of the penultimate blocks with target h_obs (it replaces
                        # this sweep's plain refresh; same key).
                        prep_c, h_obs_fn, log_marg_fn, inner_dirty, obs_c = cm[:5]
                        ckey = jax.random.fold_in(state_key, 0xC0)
                        h_fix = h_obs_fn(current_nodes, prep_c)
                        lm_cur = log_marg_fn(current_nodes, h_fix, prep_c)
                        cm_acc = jnp.zeros(n_cm, dtype=jnp.float64)
                        if joint_kind is not None:
                            # 2026-11-05 additive: joint HMC / elliptical-slice move of all inner blocks on the
                            # collapsed target (h_obs held fixed), then the exact C redraw given h_obs.
                            # cm_acc: entry 0 = accepted count; for ess entry 1 = number of extra (shrink)
                            # evaluations; every other entry = accepted count.
                            inner_idx, inner_all, lm_state, vg_state = joint_parts
                            x0 = current[:, inner_idx]
                            n_acc = 0.0
                            n_shr = 0.0
                            for _m in range(cm_sweeps):
                                ckey, k1, k2, k3 = jax.random.split(ckey, 4)
                                if joint_kind == "hmc":
                                    hs = cm_beta[0]
                                    ch_, sh_ = jnp.cos(hs), jnp.sin(hs)
                                    v = jax.random.normal(k1, x0.shape, dtype=jnp.float64)
                                    lm0, g = vg_state(current, current_nodes, h_fix, prep_c)
                                    H0 = -lm0 + 0.5 * jnp.sum(x0 * x0, -1) + 0.5 * jnp.sum(v * v, -1)
                                    x = x0
                                    lm1 = lm0
                                    for _l in range(hmc_L):
                                        v = v + 0.5 * hs * g
                                        x, v = x * ch_ + v * sh_, -x * sh_ + v * ch_
                                        lm1, g = vg_state(current.at[:, inner_idx].set(x), current_nodes, h_fix, prep_c)
                                        v = v + 0.5 * hs * g
                                    H1 = -lm1 + 0.5 * jnp.sum(x * x, -1) + 0.5 * jnp.sum(v * v, -1)
                                    log_u = jnp.log(jax.random.uniform(k2, (x0.shape[0],), dtype=jnp.float64))
                                    acc_j = (log_u < (H0 - H1)) & jnp.isfinite(H1)
                                    x0 = jnp.where(acc_j[:, None], x, x0)
                                    n_acc = n_acc + jnp.sum(acc_j)
                                else:  # ess
                                    nu = jax.random.normal(k1, x0.shape, dtype=jnp.float64)
                                    lm0 = lm_state(current, current_nodes, h_fix, prep_c)
                                    logy = lm0 + jnp.log(jax.random.uniform(k2, lm0.shape, dtype=jnp.float64))
                                    th = jax.random.uniform(k3, lm0.shape, dtype=jnp.float64) * (2 * np.pi)
                                    tmin, tmax = th - 2 * np.pi, th
                                    done = jnp.zeros(lm0.shape, dtype=bool)
                                    xnew = x0
                                    for _s in range(ess_K):
                                        xp = x0 * jnp.cos(th)[:, None] + nu * jnp.sin(th)[:, None]
                                        lmp = lm_state(current.at[:, inner_idx].set(xp), current_nodes, h_fix, prep_c)
                                        ok = (lmp > logy) & ~done
                                        xnew = jnp.where(ok[:, None], xp, xnew)
                                        n_shr = n_shr + jnp.sum(~done) * (1.0 if _s > 0 else 0.0)
                                        done = done | ok
                                        tmin = jnp.where(~done & (th < 0), th, tmin)
                                        tmax = jnp.where(~done & (th >= 0), th, tmax)
                                        ckey, ks = jax.random.split(ckey)
                                        th_new = tmin + (tmax - tmin) * jax.random.uniform(ks, th.shape, dtype=jnp.float64)
                                        th = jnp.where(done, th, th_new)
                                    x0 = xnew
                                    n_acc = n_acc + jnp.sum(done)
                                current = current.at[:, inner_idx].set(x0)
                            current_nodes = forward(current, current_nodes, inner_all)
                            cm_acc = jnp.full(n_cm, 1.0, dtype=jnp.float64) * n_acc
                            if joint_kind == "ess":
                                cm_acc = cm_acc.at[1].set(n_shr)
                            if h_K > 0:  # cm_acc entries 2 / 3 = h-ESS accepted count / extra evaluations
                                h_fix, current_ll, h_acc, h_ext = h_ess(current_nodes, current_ll, h_fix,
                                                                        jax.random.fold_in(state_key, 0xE5), prep_c, cm[6])
                                cm_acc = cm_acc.at[2].set(h_acc).at[3].set(h_ext)
                            rkey = jax.random.fold_in(state_key, 0x53)
                            current, current_nodes = refresh(current, current_nodes, rkey, h_fix)
                            return (current, current_nodes, current_ll, state_key, beta_blocks, cm_beta), (
                                accepted_this_sweep, cm_acc)
                        if mid_parts is not None:
                            prep_mid, b_fn, lmb_fn, mid_redraw, l1_types, _mid_types = mid_parts
                            pm = prep_mid(obs_c, cm[5])
                        for _m in range(cm_sweeps):
                            if mid_parts is not None:
                                # mode 2: layer-1 moves with the middle outputs at the needed cells held fixed,
                                # then the exact middle redraw given them (h_obs untouched: S depends on them only)
                                b_fix = b_fn(current_nodes, pm)
                                lmb_cur = lmb_fn(current_nodes, b_fix, pm)
                                for j, k in enumerate(cm_inner):
                                    if k not in l1_types:
                                        continue
                                    s0, e0 = bounds[k]
                                    ckey, nk, uk = jax.random.split(ckey, 3)
                                    blk = current[:, s0:e0]
                                    bj = cm_beta[j]
                                    prop_blk = jnp.sqrt(1.0 - bj * bj) * blk + bj * jax.random.normal(
                                        nk, blk.shape, dtype=jnp.float64)
                                    proposal = current.at[:, s0:e0].set(prop_blk)
                                    nodes_p = forward(proposal, current_nodes, frozenset({k}))
                                    lm_p = lmb_fn(nodes_p, b_fix, pm)
                                    log_u = jnp.log(jax.random.uniform(uk, (current.shape[0],), dtype=jnp.float64))
                                    acc_c = log_u < (lm_p - lmb_cur)
                                    current = current.at[:, s0:e0].set(jnp.where(acc_c[:, None], prop_blk, blk))
                                    current_nodes = jnp.where(acc_c[:, None, None], nodes_p, current_nodes)
                                    lmb_cur = jnp.where(acc_c, lm_p, lmb_cur)
                                    cm_acc = cm_acc.at[j].add(jnp.sum(acc_c))
                                ckey, mkey = jax.random.split(ckey)
                                current, current_nodes = mid_redraw(current, current_nodes, mkey, pm, b_fix)
                                lm_cur = log_marg_fn(current_nodes, h_fix, prep_c)
                            for j, k in enumerate(cm_inner):
                                if mid_parts is not None and k in l1_types:
                                    continue
                                s0, e0 = bounds[k]
                                ckey, nk, uk = jax.random.split(ckey, 3)
                                blk = current[:, s0:e0]
                                bj = cm_beta[j]
                                prop_blk = jnp.sqrt(1.0 - bj * bj) * blk + bj * jax.random.normal(
                                    nk, blk.shape, dtype=jnp.float64)
                                proposal = current.at[:, s0:e0].set(prop_blk)
                                nodes_p = forward(proposal, current_nodes, inner_dirty[k])
                                lm_p = log_marg_fn(nodes_p, h_fix, prep_c)
                                log_u = jnp.log(jax.random.uniform(uk, (current.shape[0],), dtype=jnp.float64))
                                acc_c = log_u < (lm_p - lm_cur)
                                current = current.at[:, s0:e0].set(jnp.where(acc_c[:, None], prop_blk, blk))
                                current_nodes = jnp.where(acc_c[:, None, None], nodes_p, current_nodes)
                                lm_cur = jnp.where(acc_c, lm_p, lm_cur)
                                cm_acc = cm_acc.at[j].add(jnp.sum(acc_c))
                        if h_K > 0:  # 2026-11-05 (not recorded in cm_acc on the per-block pCN path)
                            h_fix, current_ll, _ha, _he = h_ess(current_nodes, current_ll, h_fix,
                                                                jax.random.fold_in(state_key, 0xE5), prep_c, cm[6])
                        rkey = jax.random.fold_in(state_key, 0x53)
                        current, current_nodes = refresh(current, current_nodes, rkey, h_fix)
                        return (current, current_nodes, current_ll, state_key, beta_blocks, cm_beta), (
                            accepted_this_sweep, cm_acc)
                    if refresh is not None:
                        rkey = jax.random.fold_in(state_key, 0x53)
                        if refresh_k == 1:
                            current, current_nodes = refresh(current, current_nodes, rkey)
                        else:
                            current, current_nodes = jax.lax.cond(
                                (sweep_idx + 1) % refresh_k == 0,
                                lambda c_: refresh(c_[0], c_[1], rkey),
                                lambda c_: c_,
                                (current, current_nodes),
                            )
                    return (current, current_nodes, current_ll, state_key, beta_blocks), accepted_this_sweep

                if cm is not None:  # 2026-11-04 additive: segment body with the collapsed-move beta adaptation
                    def segment_body_cm(carry, _):
                        current, current_nodes, current_ll, state_key, beta_blocks, cm_beta = carry
                        (current, current_nodes, current_ll, state_key, _, _), (acc_ps, cm_ps) = jax.lax.scan(
                            pcn_sweep, (current, current_nodes, current_ll, state_key, beta_blocks, cm_beta),
                            jnp.arange(segment_length), length=segment_length,
                        )
                        global_accepted = jax.lax.psum(jnp.sum(acc_ps, axis=0), axis_name="device")
                        seg_acc = global_accepted / (segment_length * n_particles)
                        new_beta = _segment_beta_update(beta_blocks, seg_acc) if adapt else beta_blocks
                        cm_glob = jax.lax.psum(jnp.sum(cm_ps, axis=0), axis_name="device")
                        cm_seg = cm_glob / (segment_length * cm_sweeps * n_particles)
                        if joint_kind == "hmc":  # 2026-11-05 additive: HMC step size, target acceptance 0.65
                            new_cm = jnp.clip(cm_beta * jnp.exp(0.5 * (cm_seg[0] - 0.65)), 0.002, 1.5)
                        elif joint_kind == "ess":
                            new_cm = cm_beta
                        else:
                            new_cm = jnp.clip(cm_beta * jnp.exp(0.5 * (cm_seg - 0.23)), 0.002, 1.0)
                        return (current, current_nodes, current_ll, state_key, new_beta, new_cm), (
                            beta_blocks, seg_acc, cm_beta, cm_seg)

                    (state_f, nodes_f, ll_f, _, beta_final, cm_final), (beta_trace, acc_trace, cmb_tr, cma_tr) = (
                        jax.lax.scan(segment_body_cm, (state, nodes0, ll0, key, beta_blocks0, cm_beta0), None,
                                     length=n_segments))
                    return state_f, nodes_f, ll_f, beta_final, beta_trace, acc_trace, cm_final, cmb_tr, cma_tr

                def segment_body(carry, _):
                    current, current_nodes, current_ll, state_key, beta_blocks = carry
                    (current, current_nodes, current_ll, state_key, _), accepted_per_sweep = jax.lax.scan(
                        pcn_sweep, (current, current_nodes, current_ll, state_key, beta_blocks),
                        jnp.arange(segment_length) if refresh is not None else None, length=segment_length,
                    )
                    local_accepted = jnp.sum(accepted_per_sweep, axis=0)
                    global_accepted = jax.lax.psum(local_accepted, axis_name="device")
                    seg_acc = global_accepted / (segment_length * n_particles)
                    new_beta = _segment_beta_update(beta_blocks, seg_acc) if adapt else beta_blocks
                    return (current, current_nodes, current_ll, state_key, new_beta), (beta_blocks, seg_acc)

                (state_f, nodes_f, ll_f, _, beta_final), (beta_trace, acc_trace) = jax.lax.scan(
                    segment_body, (state, nodes0, ll0, key, beta_blocks0), None, length=n_segments
                )
                return state_f, nodes_f, ll_f, beta_final, beta_trace, acc_trace

            return run

        def bind_refresh(obs_indices, active):
            if refresh_fn is None:
                return None
            return lambda c_, n_, k_: refresh_fn(c_, n_, k_, obs_indices, active)

        def pcn_local(state, loglik, key, obs_indices, y_padded, active, beta_blocks0):
            run = make_pcn(lambda h: loglik_from_h(h, obs_indices, y_padded, active),
                           bind_refresh(obs_indices, active))
            state_f, _nodes_f, ll_f, beta_final, beta_trace, acc_trace = run(state, loglik, key, beta_blocks0)
            return state_f, ll_f, beta_final, beta_trace, acc_trace

        bundle_pcn_cm = None
        cm_once = None
        rt_once = None
        if cm_parts is not None:  # 2026-11-04 additive
            prep_fn_c, h_obs_fn_c, log_marg_c, inner_dirty_c = cm_parts

            def pcn_local_cm(state, loglik, key, obs_indices, y_padded, active, beta_blocks0, cm_beta0):
                prep_c = prep_fn_c(obs_indices, active)
                refresh_t = lambda c_, n_, k_, h_: refresh_fn(c_, n_, k_, obs_indices, active, h_)  # noqa: E731
                run = make_pcn(lambda h: loglik_from_h(h, obs_indices, y_padded, active), refresh_t,
                               cm=(prep_c, h_obs_fn_c, log_marg_c, inner_dirty_c, obs_indices, active,
                                   lambda hr: loglik_from_h(hr, jnp.arange(hr.shape[-1], dtype=obs_indices.dtype),
                                                            y_padded, active)))
                (state_f, _nodes_f, ll_f, beta_final, beta_trace, acc_trace, cm_final, cmb_tr, cma_tr) = run(
                    state, loglik, key, beta_blocks0, cm_beta0)
                return state_f, ll_f, beta_final, beta_trace, acc_trace, cm_final, cmb_tr, cma_tr

            bundle_pcn_cm = jax.pmap(
                pcn_local_cm, in_axes=(0, 0, 0, None, None, None, None, None), axis_name="device",
                devices=self.devices,
            )

            def _cm_once(st_, oi_, ac_):
                nodes_ = forward(st_)
                pc_ = prep_fn_c(oi_, ac_)
                return log_marg_c(nodes_, h_obs_fn_c(nodes_, pc_), pc_)

            cm_once = jax.jit(_cm_once)
            rt_once = jax.jit(lambda st_, k_, oi_, ac_, h_: refresh_fn(st_, forward(st_), k_, oi_, ac_, h_)[0])

        def moments_local(state, local_weights, obs_indices, y_padded, active):
            return moments_from_h(pushforward(state), local_weights, obs_indices, y_padded, active)

        def summary_extended_local(state, local_weights, obs_indices, y_padded, active, truth):
            return summary_extended_from_h(pushforward(state), local_weights, obs_indices, y_padded, active, truth)

        bundle_summary_extended = jax.pmap(
            summary_extended_local, in_axes=(0, 0, None, None, None, None), axis_name="device", devices=self.devices
        )
        bundle_loglik = jax.pmap(loglik_local, in_axes=(0, None, None, None), devices=self.devices)
        bundle_pcn = jax.pmap(
            pcn_local, in_axes=(0, 0, 0, None, None, None, None), axis_name="device", devices=self.devices
        )
        bundle_moments = jax.pmap(moments_local, in_axes=(0, 0, None, None, None), devices=self.devices)
        bundle_moments_lr = None
        if self.lowrank_moments:
            lr_mom = LowRankTerminal(
                self.terminal_lowrank if self.terminal_lowrank > 0 else 64,
                spec.amplitude, spec.terminal_lengthscale, self._noise_variance, jitter,
            )
            mom_side = int(np.ceil(np.sqrt(self.n_candidates * (lr_mom.m + 3 * lr_mom.r) * 3.0)))
            mom_chunk = compute_chunk_size(self.local_particles, mom_side, self._chunk_target_bytes)
            moments_lr_fn = build_lowrank_moments(lr_mom, self.n_candidates, n_max, mom_chunk)

            def moments_lr_local(state, local_weights, obs_indices, y_padded, active):
                return moments_lr_fn(pushforward(state), local_weights, obs_indices, y_padded, active)

            bundle_moments_lr = jax.pmap(
                moments_lr_local, in_axes=(0, 0, None, None, None), axis_name="device", devices=self.devices
            )
        lr_builder = None
        if self._use_lowrank(n_max):
            lr_term = LowRankTerminal(
                self.terminal_lowrank, spec.amplitude, spec.terminal_lengthscale, self._noise_variance, jitter
            )

            def build_lr(cap: int):
                side = int(np.ceil(np.sqrt(8 * cap * (lr_term.m + 3 * lr_term.r) / 8.0)))
                chunk_lr = compute_chunk_size(self.local_particles, side, self.lr_chunk_target_bytes)

                def lr_pcn_local(state, loglik, key, obs_indices, y_padded, active, beta_blocks0,
                                 cell_idx, cnt, ysum):
                    valid = (cnt > 0.5).astype(jnp.float64)
                    sqrt_cnt = jnp.sqrt(cnt)
                    yty = jnp.sum(y_padded * y_padded * active)
                    n_act = jnp.sum(active)
                    nodes0 = forward(state)
                    h0 = penult(nodes0)
                    hmax = jax.lax.pmax(jnp.max(jnp.abs(h0[:, cell_idx]) * valid[None, :]), axis_name="device")
                    half = jnp.clip(LOWRANK_H_MARGIN * hmax, LOWRANK_H_MIN, LOWRANK_H_MAX)
                    u_g, c_g = lr_term.grid(half)

                    def lr_ll_chunk(h_c):
                        hc = h_c[:, cell_idx]
                        bm = lr_term.features(hc, u_g, c_g, valid[None, :])
                        bw = bm * sqrt_cnt[None, :, None]
                        g = jnp.einsum("pci,pcj->pij", bw, bw)
                        b = jnp.einsum("pci,c->pi", bm, ysum)
                        return lr_term.loglik_from_gram(g, b, yty, n_act)

                    def lr_ll(h):
                        n_p = h.shape[0]
                        if chunk_lr >= n_p:
                            return lr_ll_chunk(h)
                        nc = n_p // chunk_lr
                        out = jax.lax.map(lr_ll_chunk, h.reshape(nc, chunk_lr, h.shape[-1]))
                        return out.reshape(n_p)

                    ll_lr0 = lr_ll(h0)
                    run = make_pcn(lr_ll, bind_refresh(obs_indices, active))
                    state_f, nodes_f, ll_lr_f, beta_final, beta_trace, acc_trace = run(
                        state, ll_lr0, key, beta_blocks0
                    )
                    h_f = penult(nodes_f)
                    ll_exact = loglik_from_h(h_f, obs_indices, y_padded, active)
                    logw_corr = (ll_lr0 - loglik) + (ll_exact - ll_lr_f)
                    outside = (jnp.abs(h_f[:, cell_idx]) > half) & (valid[None, :] > 0.5)
                    stats = jnp.stack([
                        half,
                        jnp.max(jnp.abs(ll_exact - ll_lr_f)),
                        jnp.max(jnp.abs(ll_lr0 - loglik)),
                        jnp.sum(outside * cnt[None, :]).astype(jnp.float64),
                        n_act * h_f.shape[0],
                        jnp.sum(jnp.any(outside, axis=1)).astype(jnp.float64),
                    ])
                    return state_f, ll_exact, beta_final, beta_trace, acc_trace, logw_corr, stats

                return jax.pmap(
                    lr_pcn_local,
                    in_axes=(0, 0, 0, None, None, None, None, None, None, None),
                    axis_name="device",
                    devices=self.devices,
                )

            lr_builder = build_lr
        return {
            "loglik_from_h": loglik_from_h,
            "moments_from_h": moments_from_h,
            "_loglik": bundle_loglik,
            "_pcn": bundle_pcn,
            "_moments": bundle_moments,
            "_moments_lr": bundle_moments_lr,
            "_summary_extended": bundle_summary_extended,
            "_lr_builder": lr_builder,
            "_lr_r": (lr_term.r if lr_builder is not None else 0),
            "_lr_cache": {},
            "_pcn_cm": bundle_pcn_cm,  # 2026-11-04 additive (None unless inner_collapsed_sweeps > 0)
            "_cm_logmarg_once": cm_once,
            "_refresh_target_once": rt_once,
            "_refresh_once": (
                None if refresh_fn is None
                else jax.jit(lambda st_, k_, oi_, ac_: refresh_fn(st_, forward(st_), k_, oi_, ac_)[0])
            ),
        }

    # ------------------------------------------------------------------ interface
    def initialise(self, seed: int) -> jax.Array:
        if self.inner_collapsed_sweeps > 0:  # 2026-11-04 additive: fresh collapsed-move betas per trajectory
            self._cm_beta = np.full(len(self._cm_inner), self.inner_collapsed_beta0, dtype=np.float64)
        keys = jax.random.split(jax.random.PRNGKey(seed), self.n_devices)
        result = self._init(keys)
        result.block_until_ready()
        return result

    def loglik(self, latent, obs_indices, y_padded, active):
        bundle = self._bundle(int(y_padded.shape[0]))
        result = bundle["_loglik"](latent, obs_indices, y_padded, active)
        result.block_until_ready()
        return result

    def cell_tables(self, obs_indices: np.ndarray, y_padded: np.ndarray, active: np.ndarray,
                    n_max: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Distinct active cells (flat index), visit counts and y-sums, padded to a capacity
        ``cap`` (multiple of 32, <= n_max) -- the low-rank terminal Gram / right-hand side depend
        on the observations only through these."""
        act = np.asarray(active) > 0.5
        idx = np.asarray(obs_indices)[act]
        y = np.asarray(y_padded, dtype=np.float64)[act]
        uniq, inv = np.unique(idx, return_inverse=True)
        cnt = np.bincount(inv, minlength=uniq.size).astype(np.float64)
        ysum = np.bincount(inv, weights=y, minlength=uniq.size).astype(np.float64)
        need = int(uniq.size)
        step = self.LOWRANK_CAP_LADDER_STEP
        cap = min(int(n_max), max(step, -(-need // step) * step))
        if need > cap:
            raise ValueError(f"{need} distinct observed cells exceed capacity {cap}")
        cidx = np.zeros(cap, dtype=np.int32)
        ccnt = np.zeros(cap, dtype=np.float64)
        cy = np.zeros(cap, dtype=np.float64)
        cidx[:need], ccnt[:need], cy[:need] = uniq.astype(np.int32), cnt, ysum
        return cap, cidx, ccnt, cy

    def _mixing_stats(self, x0: np.ndarray, x1: np.ndarray) -> dict:
        """2026-11-04 additive diagnostics of one rejuvenation, per pCN block (= type): mean squared jump
        of the whitened latent from the post-resampling state ``x0`` to the final state ``x1`` divided by
        the block width (``msjd``; two independent N(0,1) draws give 2), fraction of particles whose block
        changed (``moved``), number of distinct block values before / after (``uniq_before/after``,
        random-projection hash) and the mean per-coordinate across-particle variance after
        (``var_after``; 1 under the prior)."""
        rng = np.random.default_rng(12345)
        out = {k: [] for k in ("msjd", "moved", "uniq_before", "uniq_after", "var_after")}
        for (s0, e0) in self.block_bounds:
            a, b = x0[:, s0:e0], x1[:, s0:e0]
            d = b - a
            out["msjd"].append(float(np.mean(np.sum(d * d, axis=1)) / (e0 - s0)))
            out["moved"].append(float(np.mean(np.any(d != 0.0, axis=1))))
            r = rng.normal(size=e0 - s0)
            out["uniq_before"].append(int(np.unique(a @ r).size))
            out["uniq_after"].append(int(np.unique(b @ r).size))
            out["var_after"].append(float(np.mean(np.var(b, axis=0))))
        return out

    def rejuvenate(self, latent, loglik, seed, obs_indices, y_padded, active, beta):
        keys = jax.random.split(jax.random.PRNGKey(seed), self.n_devices)
        _x0 = self.host_particles(latent) if self.mixing_diagnostics else None
        _t0 = time.time()
        beta_arg = np.asarray(beta, dtype=np.float64)
        n_max = int(y_padded.shape[0])
        bundle = self._bundle(n_max)
        if bundle["_lr_builder"] is not None:
            cap, cidx, ccnt, cy = self.cell_tables(obs_indices, y_padded, active, n_max)
            if cap not in bundle["_lr_cache"]:
                bundle["_lr_cache"][cap] = bundle["_lr_builder"](cap)
            (latent, loglik, beta_final, beta_trace, acceptance_trace, logw_corr, lr_stats) = bundle[
                "_lr_cache"
            ][cap](latent, loglik, keys, obs_indices, y_padded, active, beta_arg, cidx, ccnt, cy)
            logw_corr = np.asarray(logw_corr).reshape(self.n_particles)
            self.last_lowrank_logw_corr = logw_corr
            st = np.asarray(lr_stats)
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
        elif bundle.get("_pcn_cm") is not None:  # 2026-11-04 additive (inner collapsed moves)
            (latent, loglik, beta_final, beta_trace, acceptance_trace, cm_final, cmb_tr, cma_tr) = bundle["_pcn_cm"](
                latent, loglik, keys, obs_indices, y_padded, active, beta_arg, np.asarray(self._cm_beta)
            )
            self._cm_beta = np.asarray(cm_final)[0].astype(np.float64)
            self._last_cm = (np.asarray(cma_tr)[0], np.asarray(cmb_tr)[0])
        else:
            latent, loglik, beta_final, beta_trace, acceptance_trace = bundle["_pcn"](
                latent, loglik, keys, obs_indices, y_padded, active, beta_arg
            )
        loglik.block_until_ready()
        beta_trace_np = np.asarray(beta_trace)[0]
        acceptance_trace_np = np.asarray(acceptance_trace)[0]
        beta_final_np = np.asarray(beta_final)[0]
        if self.mixing_diagnostics:
            _d = self._mixing_stats(_x0, self.host_particles(latent))
            _d["rejuv_seconds"] = time.time() - _t0
            _d["acc"] = np.asarray(acceptance_trace_np, dtype=np.float64)
            _d["beta"] = np.asarray(beta_trace_np, dtype=np.float64)
            if self.inner_collapsed_sweeps > 0:
                _d["cm_acc"], _d["cm_beta"] = self._last_cm
            self.last_mixing_diag = _d
        info = SegmentRejuvenationInfo(
            last_segment_acceptance=acceptance_trace_np[-1],
            first_segment_acceptance=acceptance_trace_np[0],
            beta_used_final_segment=beta_trace_np[-1],
            beta_next=beta_final_np,
            beta_trace=beta_trace_np,
            acceptance_trace=acceptance_trace_np,
        )
        return latent, loglik, info

    def posterior_moments(self, latent, weights, obs_indices, y_padded, active, exact: bool = False):
        bundle = self._bundle(int(y_padded.shape[0]))
        local_weights = weights.reshape(self.n_devices, self.local_particles)
        if self.lowrank_moments and not exact:
            result = bundle["_moments_lr"](latent, local_weights, obs_indices, y_padded, active)
        else:
            result = bundle["_moments"](latent, local_weights, obs_indices, y_padded, active)
        mean_sum, second_sum, within_sum = [np.sum(np.asarray(v), axis=0) for v in result]
        variance = np.maximum(second_sum - mean_sum * mean_sum, 0.0)
        return mean_sum, variance, within_sum

    def posterior_summary_extended(self, latent, weights, obs_indices, y_padded, active, truth):
        """Same contract as ``SharedProtocolGpnBackend.posterior_summary_extended`` (mean, variance, within,
        quantiles (10, C), cdf_at_quantiles (10, C), pit (C,))."""
        bundle = self._bundle(int(y_padded.shape[0]))
        local_weights = weights.reshape(self.n_devices, self.local_particles)
        result = bundle["_summary_extended"](latent, local_weights, obs_indices, y_padded, active, truth)
        mean_sum, second_sum, within_sum = [np.sum(np.asarray(result[i]), axis=0) for i in range(3)]
        variance = np.maximum(second_sum - mean_sum * mean_sum, 0.0)
        return (mean_sum, variance, within_sum, np.asarray(result[3])[0], np.asarray(result[4])[0],
                np.asarray(result[5])[0])

    def host_particles(self, latent: jax.Array) -> np.ndarray:
        return np.asarray(latent).reshape(self.n_particles, self.grid_size)

    def shard_particles(self, particles: np.ndarray) -> jax.Array:
        shaped = np.asarray(particles, dtype=np.float64).reshape(
            self.n_devices, self.local_particles, self.grid_size
        )
        return jax.device_put(shaped, self.particle_sharding)

    def sample_prior_h(self, key: jax.Array, n_samples: int | None = None) -> np.ndarray:
        n = self.n_particles if n_samples is None else int(n_samples)
        state = jax.random.normal(key, (n, self.grid_size), dtype=jnp.float64)
        h = self._pushforward(state)
        h.block_until_ready()
        return np.asarray(h)

    def penultimate_outputs(self, latent: jax.Array) -> jax.Array:
        if getattr(self, "_penultimate", None) is None:
            self._penultimate = jax.pmap(self._pushforward, devices=self.devices)
        result = self._penultimate(latent)
        result.block_until_ready()
        return result


# ===========================================================================
# factorised independent model of a library graph
# ===========================================================================
class GraphFactorisedIndependentGpnBackend(FactorisedIndependentGpnBackend):
    """Q independent particle systems, protocol p = private copy of its sub-graph.

    Every node of the sub-graph gets its own private type (one node per type) with the kernel of
    its library type; the terminal GP is private to the protocol.  Protocols whose sub-graphs
    have the same topology share compiled kernels.  Interface = ``FactorisedIndependentGpnBackend``
    (``loglik`` / ``rejuvenate`` / ``protocol_moments`` / ``posterior_moments`` take no layer-1
    Cholesky argument here).
    """

    def __init__(
        self,
        warped_x: np.ndarray,
        library_spec: ProtocolLibrarySpec,
        graph: GraphLibrary,
        noise_variance: float,
        kernel_jitter: float,
        n_particles: int,
        pcn_steps: int,
        *,
        pcn_mode: str = "segment-blocked",
        pcn_segment_length: int | None = None,
        n_max_buckets_pp: tuple[int, ...] = (8, 16, 32, 64),
        chunk_target_bytes: int = 256 * 1024 * 1024,
        adapt_beta: bool = True,
    ) -> None:
        graph.validate()
        super().__init__(
            warped_x, library_spec, noise_variance, kernel_jitter, n_particles, pcn_steps,
            pcn_mode=pcn_mode, pcn_segment_length=pcn_segment_length, n_protocols=graph.n_protocols,
            protocol_depths=None, protocol_layer1_cholesky=None, n_max_buckets_pp=n_max_buckets_pp,
            chunk_target_bytes=chunk_target_bytes, adapt_beta=adapt_beta,
        )
        self.graph = graph
        G = self.G
        self._subgraphs = [graph.protocol_subgraph(p) for p in range(graph.n_protocols)]
        self.protocol_depths = tuple(graph.depth(p) for p in range(graph.n_protocols))
        self.state_widths = tuple(len(nodes) * G for nodes, _ in self._subgraphs)
        # per-node kernel descriptors of each protocol's sub-graph and its topology key
        self._topology_of_protocol: list[tuple] = []
        for nodes, local_parents in self._subgraphs:
            desc = []
            for g, parents in zip(nodes, local_parents):
                ty = graph.types[graph.node_type[g]]
                desc.append((ty.arity if parents else 0, ty.lengthscale, ty.amplitude, tuple(parents)))
            self._topology_of_protocol.append(tuple(desc))
        self.state_width = max(self.state_widths)  # interface parity (per-protocol widths differ in general)
        # fixed layer-1 factors keyed by (lengthscale, amplitude)
        self._l1_chol_T: dict[tuple[float, float], jax.Array] = {}
        for desc in {d for topo in self._topology_of_protocol for d in topo if d[0] == 0}:
            _, ls, amp, _ = desc
            chol, _jit = stable_cholesky(rbf_kernel_nd(np.asarray(warped_x)[:, None], amp, ls), kernel_jitter)
            self._l1_chol_T[(ls, amp)] = jnp.asarray(chol.T, dtype=jnp.float64)
        self._init_by_width: dict[int, Any] = {}
        self._gbundles: dict[tuple[tuple, int], dict] = {}

    # ---------------------------------------------------------------- state
    def _init_fn(self, width: int):
        if width not in self._init_by_width:
            def initialise_local(key: jax.Array) -> jax.Array:
                return jax.random.normal(key, (self.local_particles, width), dtype=jnp.float64)

            self._init_by_width[width] = jax.pmap(initialise_local, devices=self.devices)
        return self._init_by_width[width]

    def initialise(self, seed: int) -> list[jax.Array]:
        keys_p = jax.random.split(jax.random.PRNGKey(seed), self.n_protocols)
        out = []
        for p in range(self.n_protocols):
            res = self._init_fn(self.state_widths[p])(jax.random.split(keys_p[p], self.n_devices))
            res.block_until_ready()
            out.append(res)
        return out

    def host_particles(self, latent_p: jax.Array) -> np.ndarray:
        return np.asarray(latent_p).reshape(self.n_particles, -1)

    def shard_particles(self, particles: np.ndarray) -> jax.Array:
        arr = np.asarray(particles, dtype=np.float64)
        shaped = arr.reshape(self.n_devices, self.local_particles, arr.shape[-1])
        return jax.device_put(shaped, self.particle_sharding)

    # ---------------------------------------------------------------- kernels
    def _push_fn(self, topo: tuple):
        G = self.G
        jitter = self._jitter_const
        l1 = self._l1_chol_T
        chunk_size_local = self.chunk_size_local
        eye_g = np.eye(G, dtype=np.float64)
        n_nodes = len(topo)

        def chunk_fn(state_chunk: jax.Array) -> jax.Array:
            nodes: list[Any] = []
            for j, (arity, ls, amp, parents) in enumerate(topo):
                xi = state_chunk[:, j * G:(j + 1) * G]
                if arity == 0:
                    nodes.append(xi @ l1[(ls, amp)])
                else:
                    cov = amp**2 * jnp.exp(-0.5 * _sq_dist_terms([nodes[q] for q in parents], ls)) \
                        + jitter * eye_g[None, :, :]
                    nodes.append(jnp.einsum("pij,pj->pi", jnp.linalg.cholesky(cov), xi))
            return nodes[-1]

        def push(state_p: jax.Array) -> jax.Array:
            total = state_p.shape[0]
            if chunk_size_local >= total:
                return chunk_fn(state_p)
            n_chunks = total // chunk_size_local
            chunks = state_p.reshape(n_chunks, chunk_size_local, state_p.shape[-1])
            return jax.lax.map(chunk_fn, chunks).reshape(total, G)

        return push

    def _gbundle(self, topo: tuple, n_max: int) -> dict:
        key = (topo, n_max)
        if key in self._gbundles:
            return self._gbundles[key]
        if n_max not in self.n_max_buckets_pp:
            raise ValueError(f"n_max={n_max} is not in n_max_buckets_pp={self.n_max_buckets_pp}")
        G = self.G
        spec = self.library_spec
        jitter_const = self._jitter_const
        rb_chunk = compute_chunk_size(self.local_particles, max(G, n_max), self._chunk_target_bytes)
        ll_chunk = compute_chunk_size(self.local_particles, n_max, self._chunk_target_bytes)
        loglik_from_h, moments_from_h, _ = build_terminal_rb_functions(
            n_candidates=G, n_max=n_max, amplitude=spec.amplitude, lengthscale=spec.terminal_lengthscale,
            noise_variance=self._noise_variance, jitter=jitter_const, chunk_size=rb_chunk,
            loglik_chunk_size=ll_chunk,
        )
        push = self._push_fn(topo)

        def loglik_local(state, obs_idx, y_padded, active):
            return loglik_from_h(push(state), obs_idx, y_padded, active)

        n_segments, segment_length = self.n_pcn_segments, self.pcn_segment_length
        n_particles = self.n_particles
        adapt = self.adapt_beta

        def pcn_local(state, loglik, key, obs_idx, y_padded, active, beta0):
            def pcn_step(carry, _):
                current, current_ll, state_key, beta = carry
                state_key, normal_key, uniform_key = jax.random.split(state_key, 3)
                white = jax.random.normal(normal_key, current.shape, dtype=jnp.float64)
                shrink = jnp.sqrt(1.0 - beta * beta)
                proposal = shrink * current + beta * white
                proposal_ll = loglik_from_h(push(proposal), obs_idx, y_padded, active)
                accepted = jnp.log(
                    jax.random.uniform(uniform_key, (current.shape[0],), dtype=jnp.float64)
                ) < (proposal_ll - current_ll)
                current = jnp.where(accepted[:, None], proposal, current)
                current_ll = jnp.where(accepted, proposal_ll, current_ll)
                return (current, current_ll, state_key, beta), jnp.sum(accepted)

            def segment_body(carry, _):
                current, current_ll, state_key, beta = carry
                (current, current_ll, state_key, _), accepted_per_step = jax.lax.scan(
                    pcn_step, (current, current_ll, state_key, beta), None, length=segment_length
                )
                global_accepted = jax.lax.psum(jnp.sum(accepted_per_step), axis_name="device")
                seg_acceptance = global_accepted / (segment_length * n_particles)
                new_beta = _segment_beta_update(beta, seg_acceptance) if adapt else beta
                return (current, current_ll, state_key, new_beta), (beta, seg_acceptance)

            (state, final_ll, _, beta_final), (beta_trace, acceptance_trace) = jax.lax.scan(
                segment_body, (state, loglik, key, beta0), None, length=n_segments
            )
            return state, final_ll, beta_final, beta_trace, acceptance_trace

        def moments_local(state, local_weights, obs_idx, y_padded, active):
            return moments_from_h(push(state), local_weights, obs_idx, y_padded, active)

        bundle = {
            "_loglik": jax.pmap(loglik_local, in_axes=(0, None, None, None), devices=self.devices),
            "_pcn": jax.pmap(
                pcn_local, in_axes=(0, 0, 0, None, None, None, None), axis_name="device", devices=self.devices
            ),
            "_moments": jax.pmap(moments_local, in_axes=(0, 0, None, None, None), devices=self.devices),
            "push": push,
            "moments_from_h": moments_from_h,
        }
        self._gbundles[key] = bundle
        return bundle

    def loglik(self, p: int, latent_p, obs_idx, y, active):
        bundle = self._gbundle(self._topology_of_protocol[p], int(y.shape[0]))
        res = bundle["_loglik"](latent_p, obs_idx, y, active)
        res.block_until_ready()
        return res

    def rejuvenate(self, p: int, latent_p, loglik_p, seed, obs_idx, y, active, beta):
        bundle = self._gbundle(self._topology_of_protocol[p], int(y.shape[0]))
        keys = jax.random.split(jax.random.PRNGKey(seed), self.n_devices)
        latent_p, loglik_p, beta_final, beta_trace, acc_trace = bundle["_pcn"](
            latent_p, loglik_p, keys, obs_idx, y, active, np.asarray(float(beta), dtype=np.float64)
        )
        loglik_p.block_until_ready()
        acc = np.asarray(acc_trace)[0]
        bt = np.asarray(beta_trace)[0]
        info = SegmentRejuvenationInfo(
            last_segment_acceptance=float(acc[-1]),
            first_segment_acceptance=float(acc[0]),
            beta_used_final_segment=float(bt[-1]),
            beta_next=float(np.asarray(beta_final)[0]),
            beta_trace=bt,
            acceptance_trace=acc,
        )
        return latent_p, loglik_p, info

    def protocol_moments(self, p: int, latent_p, weights_p, obs_idx, y, active):
        bundle = self._gbundle(self._topology_of_protocol[p], int(y.shape[0]))
        local_weights = np.asarray(weights_p).reshape(self.n_devices, self.local_particles)
        res = bundle["_moments"](latent_p, local_weights, obs_idx, y, active)
        return tuple(np.sum(np.asarray(v), axis=0) for v in res)  # type: ignore[return-value]

    def sample_prior_h(self, key: jax.Array, n_samples: int | None = None) -> np.ndarray:
        n = self.n_particles if n_samples is None else int(n_samples)
        keys = jax.random.split(key, self.n_protocols)
        parts = []
        for p in range(self.n_protocols):
            st = jax.random.normal(keys[p], (n, self.state_widths[p]), dtype=jnp.float64)
            parts.append(self._push_fn(self._topology_of_protocol[p])(st))
        return np.asarray(jnp.concatenate(parts, axis=1))
