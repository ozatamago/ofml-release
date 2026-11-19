"""gpn_smc.pushforward -- pure pushforward functions moved verbatim, per
K2_DESIGN.md K2a: pushforward_h from gpn_deep_backend.py; the
pushforward_shared_h* family and shared_pcn_block_bounds* from
u15_protocol_backend.py. Bodies are character-identical to their sources
except these import lines and this module docstring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pushforward_h(
    state: jax.Array,
    first_cholesky: jax.Array,
    inner_amplitudes: jax.Array,
    inner_lengthscales: jax.Array,
    jitter: float,
    n_candidates: int,
    depth: int,
    chunk_size: int,
) -> jax.Array:
    """Whitened state ``(P, (depth-1)*n_candidates)`` -> ``h_{D-1}`` ``(P, n_candidates)``.

    Pure function of its arguments (no closure state) -- see the module
    docstring's "Memory" paragraph. ``inner_amplitudes``/``inner_lengthscales``
    have length ``depth-2`` (one entry per inner layer k=2..D-1, in that
    order); for ``depth==2`` they are length-0 and the inner loop below never
    executes, so ``h_last`` is exactly ``h_1 = state[:, :G] @ first_cholesky.T``.
    ``depth`` and ``chunk_size`` are ordinary Python ints (static under
    ``jit``/``pmap`` tracing), so the ``range(depth - 2)`` loop unrolls into
    a fixed sequence of per-layer Cholesky steps.
    """
    G = n_candidates
    total_particles = state.shape[0]
    first_chol_T = first_cholesky.T
    eye_g = jnp.eye(G, dtype=state.dtype)

    def chunk_fn(state_chunk: jax.Array) -> jax.Array:
        h = state_chunk[:, 0:G] @ first_chol_T
        for k in range(depth - 2):
            xi_k = state_chunk[:, (k + 1) * G : (k + 2) * G]
            amplitude = inner_amplitudes[k]
            lengthscale = inner_lengthscales[k]
            difference = h[:, :, None] - h[:, None, :]
            covariance = (amplitude**2) * jnp.exp(
                -0.5 * (difference / lengthscale) ** 2
            ) + jitter * eye_g[None, :, :]
            cholesky = jnp.linalg.cholesky(covariance)
            h = jnp.einsum("pij,pj->pi", cholesky, xi_k)
        return h

    if chunk_size >= total_particles:
        return chunk_fn(state)
    n_chunks = total_particles // chunk_size
    chunks = state.reshape(n_chunks, chunk_size, state.shape[-1])
    h_chunks = jax.lax.map(chunk_fn, chunks)
    return h_chunks.reshape(total_particles, G)

def pushforward_shared_h(
    state: jax.Array,
    layer1_cholesky: jax.Array,
    layer2_amplitude: float,
    layer2_lengthscale: float,
    jitter: float,
    grid_size: int,
    chunk_size: int,
) -> jax.Array:
    """Whitened per-TYPE state ``(P, 6*G)`` -> joint penultimate ``(P, 4*G)``.

    ``state`` columns, in order: ``xi_A(0:G)``, ``xi_D(G:2G)``,
    ``xi_B(2G:4G)``, ``xi_C(4G:6G)`` with ``G = grid_size`` (64).  Returns
    the 4*G-dim joint penultimate vector in protocol order P1, P2, P3, P4::

        P1 penultimate = B(A(x))  = h_B[:, 0:G]
        P2 penultimate = C(A(x))  = h_C[:, 0:G]
        P3 penultimate = B(D(x))  = h_B[:, G:2G]
        P4 penultimate = C(D(x))  = h_C[:, G:2G]

    ``layer1_cholesky`` is the SAME fixed ``(G, G)`` factor for both A and D
    (they share amplitude/lengthscale by precommitment) -- reused, not
    recomputed.  B and C get their own separate ``(2G, 2G)`` per-particle
    covariance/Cholesky (kept separate, not shared, even though their
    hypers are numerically equal in the precommitted spec: a future
    sharing-degree sweep, U15_PROTOCOL_BANDITS_PLAN.md's optional M9, could
    give them different hypers).  Chunked via ``jax.lax.map`` exactly as
    ``gpn_deep_backend.pushforward_h`` chunks its inner layers -- the same
    mandatory pattern for any ``(P, d, d)`` tensor, here ``d = 2*grid_size``
    (128) for the B/C stage.
    """
    G = grid_size
    total_particles = state.shape[0]
    layer1_chol_T = layer1_cholesky.T
    eye2g = jnp.eye(2 * G, dtype=state.dtype)
    sf2_b = layer2_amplitude**2
    sf2_c = layer2_amplitude**2
    ell_b = layer2_lengthscale
    ell_c = layer2_lengthscale

    def chunk_fn(state_chunk: jax.Array) -> jax.Array:
        xi_A = state_chunk[:, 0:G]
        xi_D = state_chunk[:, G : 2 * G]
        xi_B = state_chunk[:, 2 * G : 4 * G]
        xi_C = state_chunk[:, 4 * G : 6 * G]
        h_A = xi_A @ layer1_chol_T
        h_D = xi_D @ layer1_chol_T
        induced = jnp.concatenate([h_A, h_D], axis=1)  # (chunk, 2G)
        difference = induced[:, :, None] - induced[:, None, :]

        cov_b = sf2_b * jnp.exp(-0.5 * (difference / ell_b) ** 2) + jitter * eye2g[None, :, :]
        chol_b = jnp.linalg.cholesky(cov_b)
        h_B = jnp.einsum("pij,pj->pi", chol_b, xi_B)

        cov_c = sf2_c * jnp.exp(-0.5 * (difference / ell_c) ** 2) + jitter * eye2g[None, :, :]
        chol_c = jnp.linalg.cholesky(cov_c)
        h_C = jnp.einsum("pij,pj->pi", chol_c, xi_C)

        return jnp.concatenate(
            [h_B[:, 0:G], h_C[:, 0:G], h_B[:, G : 2 * G], h_C[:, G : 2 * G]], axis=1
        )

    if chunk_size >= total_particles:
        return chunk_fn(state)
    n_chunks = total_particles // chunk_size
    chunks = state.reshape(n_chunks, chunk_size, state.shape[-1])
    joint_chunks = jax.lax.map(chunk_fn, chunks)
    return joint_chunks.reshape(total_particles, 4 * G)

def pushforward_shared_h_general(
    state: jax.Array,
    layer1_cholesky: jax.Array,
    layer2_amplitude: float,
    layer2_lengthscale: float,
    jitter: float,
    grid_size: int,
    chunk_size: int,
    n_l2_types: int,
    protocol_order: str = "l2_major",
) -> jax.Array:
    """Generalisation of ``pushforward_shared_h`` to an arbitrary number of
    layer-2 types (``n_l2_types``), used by the M9 8-protocol library
    (``n_l2_types=4``: B, C, G, H) and any future sharing-degree sweep
    (U15_PROTOCOL_BANDITS_PLAN.md's optional M9 section). NOT used by the
    ``n_l2_types=2`` (M8) default construction path --
    ``SharedProtocolGpnBackend`` calls the ORIGINAL, byte-unchanged
    ``pushforward_shared_h`` in that case, so M8's self-test 1 regression
    check is unaffected by this function's existence.

    ``state`` columns, in order: ``xi_A(0:G)``, ``xi_D(G:2G)``, then
    ``n_l2_types`` blocks of width ``2*G`` each, one per layer-2 type in
    type order (e.g. for M9: ``xi_B(2G:4G)``, ``xi_C(4G:6G)``,
    ``xi_G(6G:8G)``, ``xi_H(8G:10G)``). Each layer-2 type gets its OWN
    separately-computed ``(2G,2G)`` covariance/Cholesky (never shared across
    types, even though every type has numerically identical hypers under
    the precommitted spec -- matching ``pushforward_shared_h``'s own
    precedent, a hook for a future divergent-hyper sweep), evaluated at the
    SAME concatenated ``[h_A, h_D]`` induced points as every other type.

    Output order depends on ``protocol_order``:

    * ``"l2_major"`` (default, M9's convention): for each layer-2 type j in
      order, ``[that type at A, that type at D]`` -- i.e. for M9's B,C,G,H:
      P1=B.A, P2=B.D, P3=C.A, P4=C.D, P5=G.A, P6=G.D, P7=H.A, P8=H.D,
      matching U15_PROTOCOL_BANDITS_PLAN.md's M9 section EXACTLY. This is a
      DIFFERENT convention from ``pushforward_shared_h``'s L1-major M8
      order (P1=B.A, P2=C.A, P3=B.D, P4=C.D) -- the two are NOT
      interchangeable at ``n_l2_types=2``, which is why the M8 construction
      path keeps calling the original function rather than this one.
    * ``"l1_major"`` (M10's convention, U15_PROTOCOL_BANDITS_PLAN.md's M10
      section: "index p = i*8 + (j-1) for i=0 -> A, i=1 -> B"): ALL layer-2
      types at A first (in type order), then all layer-2 types at D (=B in
      M10's naming) -- i.e. for M10's C1..C8: P0=C1.A, P1=C2.A, ...,
      P7=C8.A, P8=C1.D, P9=C2.D, ..., P15=C8.D. This is the SAME convention
      ``pushforward_shared_h`` uses at ``n_l2_types=2`` (P1=B.A, P2=C.A,
      P3=B.D, P4=C.D reads as A-block [B,C] then D-block [B,C]), generalised
      to any ``n_l2_types``.

    The default ``"l2_major"`` reproduces the pre-M10 M9 behaviour exactly
    (byte-identical call signature and output for any caller that never
    passes ``protocol_order``), so this parameter is purely additive.
    """
    G = grid_size
    total_particles = state.shape[0]
    layer1_chol_T = layer1_cholesky.T
    eye2g = jnp.eye(2 * G, dtype=state.dtype)
    sf2 = layer2_amplitude**2
    ell = layer2_lengthscale
    if protocol_order not in ("l2_major", "l1_major"):
        raise ValueError(f"protocol_order={protocol_order!r} must be 'l2_major' or 'l1_major'")

    def chunk_fn(state_chunk: jax.Array) -> jax.Array:
        xi_A = state_chunk[:, 0:G]
        xi_D = state_chunk[:, G : 2 * G]
        h_A = xi_A @ layer1_chol_T
        h_D = xi_D @ layer1_chol_T
        induced = jnp.concatenate([h_A, h_D], axis=1)  # (chunk, 2G)
        difference = induced[:, :, None] - induced[:, None, :]
        covariance = sf2 * jnp.exp(-0.5 * (difference / ell) ** 2) + jitter * eye2g[None, :, :]
        # MANDATORY shared-Cholesky reuse (M10 cost-control precommitment,
        # U15_PROTOCOL_BANDITS_PLAN.md's "Cost control" section): every one
        # of the n_l2_types layer-2 types is evaluated at the SAME induced
        # points with the SAME hypers, so `covariance` above is one and the
        # same matrix for all j -- factorise it ONCE per particle (chunk)
        # and reuse the single Cholesky factor for every type's independent
        # xi_j (the xi_j are independent whitened draws, so h_j = L @ xi_j
        # reproduces the exact joint law of n_l2_types i.i.d. GP draws,
        # exactly as one 8x-repeated factorisation would, just without
        # repeating the O(G^3) factorisation cost n_l2_types times).
        chol = jnp.linalg.cholesky(covariance)

        if protocol_order == "l1_major":
            a_parts = []
            d_parts = []
            for j in range(n_l2_types):
                xi_j = state_chunk[:, (2 + 2 * j) * G : (2 + 2 * j + 2) * G]
                h_j = jnp.einsum("pij,pj->pi", chol, xi_j)
                a_parts.append(h_j[:, 0:G])
                d_parts.append(h_j[:, G : 2 * G])
            return jnp.concatenate(a_parts + d_parts, axis=1)

        parts = []
        for j in range(n_l2_types):
            xi_j = state_chunk[:, (2 + 2 * j) * G : (2 + 2 * j + 2) * G]
            h_j = jnp.einsum("pij,pj->pi", chol, xi_j)
            parts.append(h_j[:, 0:G])
            parts.append(h_j[:, G : 2 * G])
        return jnp.concatenate(parts, axis=1)

    n_out = 2 * n_l2_types * G
    if chunk_size >= total_particles:
        return chunk_fn(state)
    n_chunks = total_particles // chunk_size
    chunks = state.reshape(n_chunks, chunk_size, state.shape[-1])
    joint_chunks = jax.lax.map(chunk_fn, chunks)
    return joint_chunks.reshape(total_particles, n_out)

def pushforward_shared_h_general_nl1(
    state: jax.Array,
    layer1_cholesky: jax.Array,
    layer2_amplitude: float,
    layer2_lengthscale: float,
    jitter: float,
    grid_size: int,
    chunk_size: int,
    n_l1_types: int,
    n_l2_types: int,
) -> jax.Array:
    """Further generalisation of ``pushforward_shared_h_general`` to an
    arbitrary number of layer-1 types (``n_l1_types``), used by the M11
    PB-L2W/PB-L2N families (U15_PROTOCOL_BANDITS_PLAN.md's "M11: widening
    LAYER-2 sharing" section). NOT used at ``n_l1_types=2`` --
    ``SharedProtocolGpnBackend`` keeps calling the byte-unchanged
    ``pushforward_shared_h``/``pushforward_shared_h_general`` in that case
    (dispatched in the constructor), so M8/M9/M10's regression self-test is
    completely unaffected by this function's existence.

    ``state`` columns, in order: ``n_l1_types`` layer-1 blocks of width
    ``G = grid_size`` each (``xi_0(0:G), xi_1(G:2G), ..., xi_{L1-1}``), then
    ``n_l2_types`` layer-2 blocks of width ``n_l1_types*G`` each, one per
    layer-2 type in type order. ``layer1_cholesky`` is the SAME fixed
    ``(G,G)`` factor reused for every one of the ``n_l1_types`` layer-1
    types (they all share amplitude/lengthscale by precommitment, exactly
    as A/D did before). Every layer-2 type is evaluated JOINTLY at the
    concatenation of ALL ``n_l1_types`` layer-1 outputs (layer-1-index-major,
    width ``n_l1_types*G``) -- so its covariance is the SAME
    ``(n_l1_types*G, n_l1_types*G)`` matrix for every ``j``, factorised
    ONCE per particle (chunk) and reused for all ``n_l2_types`` types'
    independent ``xi_j`` (the mandatory M10/M11 Cholesky-reuse
    precommitment, generalised).

    Output is in the precommitted l1-major protocol order
    ``p = i*n_l2_types + j`` (i = layer-1 index, j = layer-2 index): for
    each layer-1 index ``i`` in order, all ``n_l2_types`` layer-2 types'
    values AT that layer-1 type's outputs, in layer-2-type order. At
    ``n_l1_types=2`` this is EXACTLY ``pushforward_shared_h_general``'s
    ``protocol_order="l1_major"`` branch (verified by self-test 5's
    Cholesky-reuse-exactness check), but this function is never called at
    ``n_l1_types=2`` in production (see above).
    """
    G = grid_size
    total_particles = state.shape[0]
    layer1_chol_T = layer1_cholesky.T
    L1 = int(n_l1_types)
    l2_width = L1 * G
    eye_l2 = jnp.eye(l2_width, dtype=state.dtype)
    sf2 = layer2_amplitude**2
    ell = layer2_lengthscale
    l1_offset = L1 * G

    def chunk_fn(state_chunk: jax.Array) -> jax.Array:
        h_list = [
            state_chunk[:, i * G : (i + 1) * G] @ layer1_chol_T for i in range(L1)
        ]
        induced = jnp.concatenate(h_list, axis=1)  # (chunk, L1*G)
        difference = induced[:, :, None] - induced[:, None, :]
        covariance = sf2 * jnp.exp(-0.5 * (difference / ell) ** 2) + jitter * eye_l2[None, :, :]
        # MANDATORY shared-Cholesky reuse (M10/M11 cost-control
        # precommitment): every one of the n_l2_types layer-2 types shares
        # the SAME covariance -- factorise ONCE per particle (chunk) and
        # reuse for every type's independent xi_j.
        chol = jnp.linalg.cholesky(covariance)

        blocks_by_i: list[list[jax.Array]] = [[] for _ in range(L1)]
        for j in range(n_l2_types):
            xi_j = state_chunk[:, l1_offset + j * l2_width : l1_offset + (j + 1) * l2_width]
            h_j = jnp.einsum("pij,pj->pi", chol, xi_j)
            for i in range(L1):
                blocks_by_i[i].append(h_j[:, i * G : (i + 1) * G])
        out = []
        for i in range(L1):
            out.extend(blocks_by_i[i])
        return jnp.concatenate(out, axis=1)

    n_out = L1 * n_l2_types * G
    if chunk_size >= total_particles:
        return chunk_fn(state)
    n_chunks = total_particles // chunk_size
    chunks = state.reshape(n_chunks, chunk_size, state.shape[-1])
    joint_chunks = jax.lax.map(chunk_fn, chunks)
    return joint_chunks.reshape(total_particles, n_out)

def pushforward_shared_h_m12(
    state: jax.Array,
    layer1_cholesky: jax.Array,
    layer2_amplitude: float,
    layer2_lengthscale: float,
    jitter: float,
    grid_size: int,
    chunk_size: int,
    n_l1_types: int,
) -> jax.Array:
    """M12 mixed-depth pushforward (cross-depth warm-up library; see this
    module's "M12" constants block and U15_PROTOCOL_BANDITS_PLAN.md's "M12"
    section). ``state`` columns, in order: ``n_l1_types`` layer-1 blocks of
    width ``G`` each (``xi_A1(0:G), ..., xi_A{L1}``), then ONE layer-2 block
    of width ``L1*G`` (``xi_B``), then ONE layer-3 block of width ``L1*G``
    (``xi_C``) -- total width ``3*L1*G``.

    Algebra: ``h_A = [h_A1..h_A{L1}]`` (A-index-major, ``L1*G``-wide) is the
    fixed-grid layer-1 pushforward, reusing the SAME precomputed ``(G,G)``
    Cholesky for every layer-1 type. B is drawn JOINTLY at the ``L1*G``
    concatenated ``h_A`` points (ONE covariance/Cholesky, since amplitude/
    lengthscale are shared and identical for every "type" -- there is only
    one B here, unlike the multi-layer-2-type functions above): ``h_B =
    chol_B @ xi_B``, and ``h_B[i*G:(i+1)*G] = B(A_i(x))`` for each layer-1
    index ``i`` in the SAME order as ``h_A``. C is then drawn JOINTLY at the
    ``L1*G`` concatenated ``h_B`` points (a SECOND covariance/Cholesky, same
    hypers): ``h_C = chol_C @ xi_C``, ``h_C[i*G:(i+1)*G] = C(B(A_i(x)))``.

    Output (the penultimate values fed to the pooled terminal R) is
    ``concat(h_A, h_C)``, width ``2*L1*G`` -- protocol order is the
    SIMPLE set first (protocols 0..L1-1, penultimate ``h_A_i``, depth 2),
    then the TARGET set (protocols L1..2*L1-1, penultimate ``h_C_i``, depth
    4), matching ``U15_PROTOCOL_BANDITS_PLAN.md``'s M12 protocol-order
    precommitment exactly.
    """
    G = grid_size
    total_particles = state.shape[0]
    layer1_chol_T = layer1_cholesky.T
    L1 = int(n_l1_types)
    width = L1 * G
    eye = jnp.eye(width, dtype=state.dtype)
    sf2 = layer2_amplitude**2
    ell = layer2_lengthscale

    def chunk_fn(state_chunk: jax.Array) -> jax.Array:
        h_list = [state_chunk[:, i * G : (i + 1) * G] @ layer1_chol_T for i in range(L1)]
        h_A = jnp.concatenate(h_list, axis=1)  # (chunk, L1*G), A-index-major
        xi_B = state_chunk[:, width : 2 * width]
        xi_C = state_chunk[:, 2 * width : 3 * width]

        diff_A = h_A[:, :, None] - h_A[:, None, :]
        cov_b = sf2 * jnp.exp(-0.5 * (diff_A / ell) ** 2) + jitter * eye[None, :, :]
        chol_b = jnp.linalg.cholesky(cov_b)
        h_B = jnp.einsum("pij,pj->pi", chol_b, xi_B)  # (chunk, L1*G)

        diff_B = h_B[:, :, None] - h_B[:, None, :]
        cov_c = sf2 * jnp.exp(-0.5 * (diff_B / ell) ** 2) + jitter * eye[None, :, :]
        chol_c = jnp.linalg.cholesky(cov_c)
        h_C = jnp.einsum("pij,pj->pi", chol_c, xi_C)  # (chunk, L1*G)

        return jnp.concatenate([h_A, h_C], axis=1)  # (chunk, 2*L1*G)

    n_out = 2 * width
    if chunk_size >= total_particles:
        return chunk_fn(state)
    n_chunks = total_particles // chunk_size
    chunks = state.reshape(n_chunks, chunk_size, state.shape[-1])
    joint_chunks = jax.lax.map(chunk_fn, chunks)
    return joint_chunks.reshape(total_particles, n_out)

def pushforward_shared_h_lib_s(
    state: jax.Array,
    layer1_cholesky: jax.Array,
    layer2_amplitude: float,
    layer2_lengthscale: float,
    jitter: float,
    grid_size: int,
    chunk_size: int,
    n_l1_types: int,
) -> jax.Array:
    """LIB-S ("siblings", PB-SIB) pushforward: ``n_l1_types`` (6) UNIFORM
    depth-4 protocols sharing one B and one C layer, exactly
    ``pushforward_shared_h_m12``'s algebra generalised to an arbitrary
    ``n_l1_types`` with NO simple/depth-2 set. Reuses
    ``pushforward_shared_h_m12`` byte-for-byte (same state layout: n_l1_types
    layer-1 blocks, then one B block, then one C block, each width
    ``n_l1_types*grid_size``) and simply keeps only its ``h_C`` half
    (``concat(h_A, h_C)[:, width:2*width]``) -- every protocol here is depth
    4, so there is no separate depth-2 "simple" set to concatenate in front
    of it."""
    width = int(n_l1_types) * grid_size
    combined = pushforward_shared_h_m12(
        state,
        layer1_cholesky,
        layer2_amplitude,
        layer2_lengthscale,
        jitter,
        grid_size,
        chunk_size,
        n_l1_types,
    )
    return combined[:, width : 2 * width]

def pushforward_shared_h_battery(
    state: jax.Array,
    layer1_cholesky: jax.Array,
    layer2_amplitude: float,
    layer2_lengthscale: float,
    jitter: float,
    grid_size: int,
    chunk_size: int,
    n_l1_types: int,
    ramp_w: jax.Array,
) -> jax.Array:
    """LIB-B ("coverage battery", PB-BAT) pushforward: M12's mixed-depth
    algebra (``n_l1_types`` simple depth-2 protocols, ``n_l1_types`` target
    depth-4 protocols sharing one B and one C layer) PLUS two ramp
    evaluations ``S_B = R.B.w``, ``S_C = R.C.w`` at the KNOWN, fixed input
    ``ramp_w`` (width ``grid_size``) -- see this module's "M13a posterior
    screen" constants-block docstring for why these are the deterministic
    GP-conditional-mean ("BLUP") readouts, not extra-residual draws, and
    hence need no extra whitened state (``state`` is the SAME
    ``3*n_l1_types*grid_size``-wide layout as ``pushforward_shared_h_m12``).

    Output order (penultimate values fed to the pooled terminal R):
    ``concat(h_C, h_A, b_w, c_w)`` -- TARGET set first, then simple, then
    the two ramp blocks, width ``2*n_l1_types*grid_size + 2*grid_size``, per
    the workflow's explicit "[targets 256, simple 256, B(w) 64, C(w) 64]"
    precommitment (the OPPOSITE order from ``pushforward_shared_h_m12``'s
    own simple-first convention -- a deliberate, documented difference).
    """
    G = grid_size
    total_particles = state.shape[0]
    layer1_chol_T = layer1_cholesky.T
    L1 = int(n_l1_types)
    width = L1 * G
    eye = jnp.eye(width, dtype=state.dtype)
    sf2 = layer2_amplitude**2
    ell = layer2_lengthscale
    w = ramp_w

    def chunk_fn(state_chunk: jax.Array) -> jax.Array:
        h_list = [state_chunk[:, i * G : (i + 1) * G] @ layer1_chol_T for i in range(L1)]
        h_A = jnp.concatenate(h_list, axis=1)  # (chunk, width)
        xi_B = state_chunk[:, width : 2 * width]
        xi_C = state_chunk[:, 2 * width : 3 * width]

        diff_A = h_A[:, :, None] - h_A[:, None, :]
        cov_b = sf2 * jnp.exp(-0.5 * (diff_A / ell) ** 2) + jitter * eye[None, :, :]
        chol_b = jnp.linalg.cholesky(cov_b)
        h_B = jnp.einsum("pij,pj->pi", chol_b, xi_B)  # (chunk, width)

        diff_B = h_B[:, :, None] - h_B[:, None, :]
        cov_c = sf2 * jnp.exp(-0.5 * (diff_B / ell) ** 2) + jitter * eye[None, :, :]
        chol_c = jnp.linalg.cholesky(cov_c)
        h_C = jnp.einsum("pij,pj->pi", chol_c, xi_C)  # (chunk, width)

        # Deterministic BLUP readouts at the known ramp w: alpha = K_AA^{-1}
        # h = chol^{-T} chol^{-1} h = chol^{-T} xi (since h = chol @ xi) --
        # ONE triangular solve, no second one needed.
        alpha_B = jax.scipy.linalg.solve_triangular(
            jnp.swapaxes(chol_b, 1, 2), xi_B[..., None], lower=False
        )[..., 0]
        diff_Aw = h_A[:, :, None] - w[None, None, :]  # (chunk, width, G)
        cross_Aw = sf2 * jnp.exp(-0.5 * (diff_Aw / ell) ** 2)
        b_w = jnp.einsum("pij,pi->pj", cross_Aw, alpha_B)  # (chunk, G)

        alpha_C = jax.scipy.linalg.solve_triangular(
            jnp.swapaxes(chol_c, 1, 2), xi_C[..., None], lower=False
        )[..., 0]
        diff_Bw = h_B[:, :, None] - w[None, None, :]
        cross_Bw = sf2 * jnp.exp(-0.5 * (diff_Bw / ell) ** 2)
        c_w = jnp.einsum("pij,pi->pj", cross_Bw, alpha_C)  # (chunk, G)

        return jnp.concatenate([h_C, h_A, b_w, c_w], axis=1)  # (chunk, 2*width+2G)

    n_out = 2 * width + 2 * G
    if chunk_size >= total_particles:
        return chunk_fn(state)
    n_chunks = total_particles // chunk_size
    chunks = state.reshape(n_chunks, chunk_size, state.shape[-1])
    joint_chunks = jax.lax.map(chunk_fn, chunks)
    return joint_chunks.reshape(total_particles, n_out)

def shared_pcn_block_bounds_m12(grid_size: int, n_l1_types: int) -> tuple[tuple[int, int], ...]:
    """Blocked-pCN block boundaries for ``SharedProtocolGpnBackend``'s M12
    mixed-depth whitened state: ``n_l1_types`` fixed layer-1 blocks of width
    ``grid_size`` each, then ONE layer-2 (B) block and ONE layer-3 (C) block,
    each width ``n_l1_types*grid_size`` -- ``n_l1_types + 2`` blocks total
    (6 at the precommitted ``n_l1_types=4``)."""
    G = grid_size
    L1 = n_l1_types
    width = L1 * G
    bounds = [(i * G, (i + 1) * G) for i in range(L1)]
    bounds.append((L1 * G, L1 * G + width))
    bounds.append((L1 * G + width, L1 * G + 2 * width))
    return tuple(bounds)

def shared_pcn_block_bounds(
    grid_size: int, n_l2_types: int, n_l1_types: int = 2
) -> tuple[tuple[int, int], ...]:
    """Blocked-pCN block boundaries for ``SharedProtocolGpnBackend``'s
    whitened state, generalised from the hardcoded 4-block (A, D, B, C)
    tuple to an arbitrary number of layer-1 types (``n_l1_types``, M11
    additive, default 2 reproducing the pre-M11 signature/behaviour
    exactly) and layer-2 types: ``n_l1_types`` fixed layer-1 blocks of
    width ``grid_size`` each, then ``n_l2_types`` layer-2 blocks in type
    order, each width ``n_l1_types*grid_size``. At ``n_l1_types=2,
    n_l2_types=2`` this returns EXACTLY the same 4 tuples the pre-M9 code
    hardcoded literally (``((0,G),(G,2G),(2G,4G),(4G,6G))``) -- pure
    integer arithmetic, so M8's byte-reproducibility self-test is
    unaffected by using this helper in place of the old literal.
    """
    G = grid_size
    bounds = [(i * G, (i + 1) * G) for i in range(n_l1_types)]
    l2_width = n_l1_types * G
    start = n_l1_types * G
    for _ in range(n_l2_types):
        bounds.append((start, start + l2_width))
        start += l2_width
    return tuple(bounds)
