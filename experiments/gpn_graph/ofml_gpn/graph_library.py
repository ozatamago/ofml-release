"""ofml_gpn.graph_library -- generic library-graph language (copied from cx_bayes gpn_smc.graph_library,
BENCH9 2026-11-01b; generalised in ofml_gpn 2026-11-04, see "Self-feeding types" below).

Numpy-only (no jax): importable before the CUDA device selection.

A LIBRARY is a list of inner TYPES in draw order and a list of NODES.  A node is one
application of a type to an ordered tuple of parent nodes (``parents == ()`` means a
layer-1 node whose input is the warped grid).  Each node output is a length-G vector.
A type's GP is ONE function drawn jointly at all of its nodes' inputs: the input points
``Z_t`` stack, over the type's nodes (node order), the G parent-output tuples, shape
``(n_nodes_t * G, arity_t)``.  Kernel: ``amplitude^2 exp(-||z - z'||^2 / (2 l^2))``
(Euclidean over the arity dimensions).  Each protocol is one penultimate node; the shared
terminal R is drawn jointly at the P penultimate outputs (P*G points, 1-D input), in
protocol order.

Conventions enforced by ``GraphLibrary.validate``:

* node indices are topologically ordered (every parent index < the node's index);
* layer-1 types have ``arity == 1`` and nodes with ``parents == ()``; every other node
  has ``len(parents) == arity`` of its type.
* (origin gpn_smc only, REMOVED here) "every parent node belongs to an EARLIER type".  A library that
  satisfies it is OLD-VALID (``GraphLibrary.old_valid``) and is evaluated exactly as before.

Teacher RNG order (one ``default_rng(teacher_seed)``) of an OLD-VALID library: for each type in order
``f_t = chol(K(Z_t)) @ rng.normal(size=n_nodes_t * G)``; then
``truth = chol(K_R) @ rng.normal(size=P * G)``.  Noise = ``default_rng(noise_seed)
.normal(scale=noise_std, size=(P, G)).reshape(-1)`` (the existing rule).

Self-feeding types (ofml_gpn, 2026-11-04).  A type may be applied to its own output (M(M(A,B),A)) and two
types may feed each other at node level (C2(C1(A)) and C1(C2(A))).  Each type's GP is still ONE function
drawn jointly at all its nodes' inputs, f_t = chol(K(Z_t) + jitter I) xi_t, but Z_t is only known
progressively.  The type's nodes are therefore put in a CHOLESKY ORDER made of EVALUATION BLOCKS
(``GraphLibrary.schedule``), each block's inputs depending only on earlier blocks.  Because row block b of
a lower Cholesky factor depends only on the matrix entries of blocks <= b,

    f_b = L[b, <=b] xi_{<=b} = K_{b,<b} K_{<b}^{-1} f_{<b} + chol(K_{b|<b}) xi_b,

i.e. block b is drawn from the GP conditional given the type's earlier evaluations, with xi ~ N(0, I)
and xi -> node outputs deterministic.  Implementation: block b's rows are taken from the Cholesky factor
of the PREFIX blocks <= b (``EvalBlock.prefix``), which the progressive inputs make computable.

Schedule (``stage_policy``): "old" = one block per type in type order (requires old_valid; this is the
origin behaviour, bit for bit); "leaf" = (1) every NON-layer-1 node without children is a LEAF node and
goes into its type's final LEAF block (all leaf blocks after every other block); (2) the remaining nodes
are scheduled by greedy passes over the types in type order, each pass emitting, per type, the block of
its not-yet-evaluated non-leaf nodes whose parents are all evaluated; layer-1 types are one block.  Inside
a block the nodes are sorted by (parent tuple, node index) so that types applied to the same parents in
the same way share one Cholesky factor.  "auto" (default) = "old" if old_valid else "leaf".  The leaf
blocks are what the null refresh and the collapsed moves of ``graph_backends`` act on.

Latent layout: whitened blocks in TYPE order, block t of width n_nodes_t * G in t's Cholesky order
(node slot-major, grid minor); ``node_slot`` is the position in that order.  Teacher RNG order of a
"leaf"-policy library: for each evaluation block in schedule order, ``xi_b = rng.normal(size=n_b * G)``
and ``f_b = L_prefix[b rows] @ xi_{<=b}`` (L_prefix = stable_cholesky of the prefix kernel matrix, the
initial jitter being the largest jitter already used by that type); then the terminal as above.  For an
old-valid library under "leaf" this is NOT the old stream (use "old"/"auto" for that).

``get_graph_library(name)`` returns one of the BENCH9 libraries; ``"PB-W16"`` is the PB-W16
graph (validation only: ``draw_graph_teacher`` reproduces ``draw_pb_w16_teacher`` bit for
bit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable

import numpy as np

AMPLITUDE = 0.70
LAYER1_LENGTHSCALE = 1.0
INNER_LENGTHSCALE = 0.50
TERMINAL_LENGTHSCALE = 0.50
TERMINAL_AMPLITUDE = 0.70


@dataclass(frozen=True)
class GraphType:
    """One library type.  ``arity`` = number of kernel input dimensions (1 for layer-1
    types, whose input is the warped grid; the number of parents otherwise)."""

    name: str
    arity: int
    lengthscale: float
    amplitude: float = AMPLITUDE


@dataclass(frozen=True)
class GraphNode:
    """One application of ``type_name`` to the ordered parent nodes ``parents`` (indices
    into the library's node list); ``parents == ()``: layer-1 node (input warped_x)."""

    type_name: str
    parents: tuple[int, ...]


@dataclass(frozen=True)
class EvalBlock:
    """One evaluation block of a type (ofml_gpn 2026-11-04; see the module docstring).

    ``nodes``: the block's nodes in Cholesky order; ``offset``: number of the type's nodes before this block
    in the type's Cholesky order; ``prefix``: the type's nodes of this and all earlier blocks (Cholesky
    order), whose inputs define the prefix Cholesky factor; ``leaf``: leaf block (non-layer-1, every node
    childless, last block of its type); ``single``: the block is the whole type (offset 0, prefix = all)."""

    type_idx: int
    nodes: tuple[int, ...]
    stage: int
    offset: int
    prefix: tuple[int, ...]
    leaf: bool
    single: bool


@dataclass(frozen=True)
class GraphLibrary:
    name: str
    types: tuple[GraphType, ...]
    nodes: tuple[GraphNode, ...]
    protocols: tuple[int, ...]  # node index of each protocol's penultimate node
    terminal_lengthscale: float = TERMINAL_LENGTHSCALE
    terminal_amplitude: float = TERMINAL_AMPLITUDE
    stage_policy: str = "auto"  # ofml_gpn: "auto" | "old" | "leaf" (module docstring)

    # ------------------------------------------------------------------ derived
    @cached_property
    def type_index(self) -> dict[str, int]:
        return {t.name: i for i, t in enumerate(self.types)}

    @cached_property
    def node_type(self) -> tuple[int, ...]:
        return tuple(self.type_index[n.type_name] for n in self.nodes)

    @cached_property
    def type_nodes(self) -> tuple[tuple[int, ...], ...]:
        """Node indices of each type, in node order."""
        out: list[list[int]] = [[] for _ in self.types]
        for i, t in enumerate(self.node_type):
            out[t].append(i)
        return tuple(tuple(v) for v in out)

    @cached_property
    def node_slot(self) -> tuple[int, ...]:
        """Position of each node inside its type's block (0-based, the type's CHOLESKY order; for the
        "old" policy this is node order, as in the origin)."""
        slot = [0] * self.n_nodes
        for order in self.type_chol_order:
            for s_, i in enumerate(order):
                slot[i] = s_
        return tuple(slot)

    # ------------------------------------------------------------------ ofml_gpn 2026-11-04: schedule
    @cached_property
    def children(self) -> tuple[tuple[int, ...], ...]:
        ch: list[list[int]] = [[] for _ in self.nodes]
        for i, nd in enumerate(self.nodes):
            for q in nd.parents:
                ch[q].append(i)
        return tuple(tuple(v) for v in ch)

    @cached_property
    def old_valid(self) -> bool:
        """Every parent node belongs to an EARLIER type (the origin language)."""
        return all(self.node_type[q] < self.node_type[i] for i, nd in enumerate(self.nodes) for q in nd.parents)

    @cached_property
    def resolved_stage_policy(self) -> str:
        pol = self.stage_policy
        if pol not in ("auto", "old", "leaf"):
            raise ValueError(f"{self.name}: unknown stage_policy {pol!r}")
        if pol == "auto":
            return "old" if self.old_valid else "leaf"
        if pol == "old" and not self.old_valid:
            raise ValueError(f"{self.name}: stage_policy 'old' needs an old-valid library")
        return pol

    @cached_property
    def schedule(self) -> tuple[EvalBlock, ...]:
        """Evaluation blocks in evaluation order (module docstring)."""
        raw: list[tuple[int, tuple[int, ...]]] = []
        layer1 = [self.is_layer1(t) for t in range(self.n_types)]
        leafset = {i for i in range(self.n_nodes) if not self.children[i] and not layer1[self.node_type[i]]}
        if self.resolved_stage_policy == "old":
            raw = [(t, self.type_nodes[t]) for t in range(self.n_types)]
        else:
            done: set[int] = set()
            todo = [[i for i in self.type_nodes[t] if i not in leafset] for t in range(self.n_types)]

            def sort_key(i: int):
                return (self.nodes[i].parents, i)

            while any(todo):
                progressed = False
                for t in range(self.n_types):
                    ready = [i for i in todo[t] if all(q in done for q in self.nodes[i].parents)]
                    if ready:
                        raw.append((t, tuple(sorted(ready, key=sort_key))))
                        done.update(ready)
                        todo[t] = [i for i in todo[t] if i not in done]
                        progressed = True
                if not progressed:
                    raise ValueError(f"{self.name}: node graph is not acyclic")
            for t in range(self.n_types):
                lv = [i for i in self.type_nodes[t] if i in leafset]
                if lv:
                    raw.append((t, tuple(sorted(lv, key=sort_key))))
        blocks: list[EvalBlock] = []
        seen: list[list[int]] = [[] for _ in range(self.n_types)]
        stage = [0] * self.n_types
        n_blocks_of = [sum(1 for tt, _ in raw if tt == t) for t in range(self.n_types)]
        for t, ids in raw:
            off = len(seen[t])
            seen[t].extend(ids)
            is_last = stage[t] == n_blocks_of[t] - 1
            leaf = bool(ids) and all(i in leafset for i in ids) and is_last
            blocks.append(EvalBlock(t, tuple(ids), stage[t], off, tuple(seen[t]), leaf,
                                    off == 0 and n_blocks_of[t] == 1))
            stage[t] += 1
        return tuple(blocks)

    @cached_property
    def type_chol_order(self) -> tuple[tuple[int, ...], ...]:
        """Each type's nodes in Cholesky order (= node order for the "old" policy)."""
        out: list[list[int]] = [[] for _ in self.types]
        for e in self.schedule:
            out[e.type_idx].extend(e.nodes)
        return tuple(tuple(v) for v in out)

    def eval_key(self, e: "EvalBlock") -> tuple:
        """Cholesky-sharing signature of a non-layer-1 block: kernel and the ordered parent tuples of its
        prefix (blocks with equal keys have identical prefix kernel matrices in every particle)."""
        ty = self.types[e.type_idx]
        return (ty.arity, ty.lengthscale, ty.amplitude, tuple(self.nodes[i].parents for i in e.prefix))

    @property
    def n_types(self) -> int:
        return len(self.types)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_protocols(self) -> int:
        return len(self.protocols)

    def n_candidates(self, G: int) -> int:
        return self.n_protocols * G

    def latent_dim(self, G: int) -> int:
        return self.n_nodes * G

    def block_bounds(self, G: int) -> tuple[tuple[int, int], ...]:
        """Whitened-latent block (one per type, type order) ``(start, end)`` columns."""
        bounds = []
        start = 0
        for nodes in self.type_nodes:
            end = start + len(nodes) * G
            bounds.append((start, end))
            start = end
        return tuple(bounds)

    def is_layer1(self, type_idx: int) -> bool:
        nodes = self.type_nodes[type_idx]
        return bool(nodes) and all(len(self.nodes[i].parents) == 0 for i in nodes)

    @cached_property
    def chol_groups(self) -> tuple[tuple[int, ...], ...]:
        """Groups of NON-layer-1 type indices whose covariance matrices are identical
        (same arity/lengthscale/amplitude and identical ordered parent tuples per node):
        one Cholesky serves the whole group.  Groups and members are in type order."""
        sig_to_group: dict[Any, list[int]] = {}
        order: list[Any] = []
        for t, ty in enumerate(self.types):
            if self.is_layer1(t):
                continue
            sig = (
                ty.arity, ty.lengthscale, ty.amplitude,
                tuple(self.nodes[i].parents for i in self.type_nodes[t]),
            )
            if sig not in sig_to_group:
                sig_to_group[sig] = []
                order.append(sig)
            sig_to_group[sig].append(t)
        return tuple(tuple(sig_to_group[s]) for s in order)

    def depth(self, protocol: int) -> int:
        """Number of layers of protocol p including the terminal (= longest chain + 1)."""
        memo: dict[int, int] = {}

        def d(i: int) -> int:
            if i not in memo:
                memo[i] = 1 + max((d(q) for q in self.nodes[i].parents), default=0)
            return memo[i]

        return d(self.protocols[protocol]) + 1

    def protocol_subgraph(self, p: int) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """Nodes reachable backwards from protocol p's penultimate node (global indices in
        increasing order) and, for each, its parents in LOCAL numbering (position in that
        list).  Used by the independent model (each node gets a private type)."""
        seen: set[int] = set()
        stack = [self.protocols[p]]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            stack.extend(self.nodes[i].parents)
        nodes = tuple(sorted(seen))
        local = {g: k for k, g in enumerate(nodes)}
        parents = tuple(tuple(local[q] for q in self.nodes[g].parents) for g in nodes)
        return nodes, parents

    def validate(self) -> None:
        names = [t.name for t in self.types]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate type names")
        for i, nd in enumerate(self.nodes):
            if nd.type_name not in self.type_index:
                raise ValueError(f"{self.name}: node {i} has unknown type {nd.type_name!r}")
            ty = self.types[self.type_index[nd.type_name]]
            if len(nd.parents) == 0:
                if ty.arity != 1:
                    raise ValueError(f"{self.name}: layer-1 node {i} (type {ty.name}) needs arity 1")
            elif len(nd.parents) != ty.arity:
                raise ValueError(
                    f"{self.name}: node {i} has {len(nd.parents)} parents but type {ty.name} has arity {ty.arity}"
                )
            for q in nd.parents:
                if not (0 <= q < i):
                    raise ValueError(f"{self.name}: node {i} parent {q} must be an earlier node")
                # ofml_gpn: the origin's "parent belongs to an EARLIER type" rule is removed (self-feeding
                # types); ``old_valid`` reports whether it holds.
        for t in range(self.n_types):
            kinds = {len(self.nodes[i].parents) == 0 for i in self.type_nodes[t]}
            if len(kinds) > 1:
                raise ValueError(f"{self.name}: type {self.types[t].name} mixes layer-1 and inner nodes")
            if not self.type_nodes[t]:
                raise ValueError(f"{self.name}: type {self.types[t].name} has no node")
        if not self.protocols or any(not (0 <= p < self.n_nodes) for p in self.protocols):
            raise ValueError(f"{self.name}: bad protocols")
        if len(set(self.protocols)) != len(self.protocols):
            raise ValueError(f"{self.name}: duplicate protocol nodes")
        _ = self.schedule  # ofml_gpn: policy / acyclicity checks


# ---------------------------------------------------------------------------
# kernels / teacher (numpy)
# ---------------------------------------------------------------------------
def rbf_kernel_nd(z: np.ndarray, amplitude: float, lengthscale: float) -> np.ndarray:
    """Euclidean RBF over the last axis of ``z`` (shape ``(N, d)``): ``amplitude^2 *
    exp(-0.5 * sum_d ((z_i - z_j)_d / l)^2)``.  For ``d == 1`` the arithmetic is the same as
    ``gpn_smc.numerics.rbf_covariance`` (bit-identical)."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    diff = z[:, None, :] - z[None, :, :]
    return amplitude**2 * np.exp(-0.5 * np.sum((diff / lengthscale) ** 2, axis=-1))


def stable_cholesky(matrix: np.ndarray, initial_jitter: float) -> tuple[np.ndarray, float]:
    """Same procedure as ``gpn_smc.numerics.stable_cholesky`` (jax-free copy)."""
    jitter = initial_jitter
    identity = np.eye(len(matrix), dtype=np.float64)
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + jitter * identity), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError(f"Cholesky failed through jitter={jitter:g}")


def type_inputs(lib: GraphLibrary, type_idx: int, node_out: dict[int, np.ndarray],
                warped_x: np.ndarray) -> np.ndarray:
    """``Z_t`` of shape ``(n_nodes_t * G, arity_t)`` from the parent outputs ``node_out`` (node order)."""
    return nodes_inputs(lib, lib.type_nodes[type_idx], node_out, warped_x)


def nodes_inputs(lib: GraphLibrary, node_ids, node_out: dict[int, np.ndarray],
                 warped_x: np.ndarray) -> np.ndarray:
    """Kernel inputs of the given nodes of ONE type, stacked in the given order (ofml_gpn)."""
    rows = []
    for i in node_ids:
        parents = lib.nodes[i].parents
        if not parents:
            rows.append(np.asarray(warped_x, dtype=np.float64)[:, None])
        else:
            rows.append(np.stack([node_out[q] for q in parents], axis=1))
    return np.concatenate(rows, axis=0)


def draw_graph_teacher(
    lib: GraphLibrary,
    warped_x: np.ndarray,
    kernel_jitter: float,
    teacher_seed: int,
    noise_seed: int,
    noise_std: float,
    *,
    return_details: bool = False,
):
    """Teacher draw following the module docstring's RNG order.  Returns ``(truth, noise)``
    (each ``(P*G,)``, protocol-major) or, with ``return_details``, additionally a dict with
    ``node_out`` (node -> (G,)) and ``joint_penultimate`` ((P*G,))."""
    rng = np.random.default_rng(teacher_seed)
    G = int(np.asarray(warped_x).shape[0])
    node_out: dict[int, np.ndarray] = {}
    chol_cache: dict[Any, np.ndarray] = {}
    xi_of_type: dict[int, list[np.ndarray]] = {t: [] for t in range(lib.n_types)}
    jit_of_type: dict[int, float] = {}
    jit_blocks: list[float] = []
    for e in lib.schedule:  # ofml_gpn: evaluation blocks ("old" policy: one per type, type order = origin)
        t = e.type_idx
        ty = lib.types[t]
        z = nodes_inputs(lib, e.prefix, node_out, warped_x)
        key = (z.tobytes(), z.shape, ty.lengthscale, ty.amplitude)
        chol = chol_cache.get(key)
        if chol is None:
            chol, jit = stable_cholesky(rbf_kernel_nd(z, ty.amplitude, ty.lengthscale),
                                        max(kernel_jitter, jit_of_type.get(t, kernel_jitter)))
            jit_of_type[t] = jit
            chol_cache[key] = chol
        jit_blocks.append(jit_of_type.get(t, kernel_jitter))
        nodes = e.nodes
        xi_b = rng.normal(size=len(nodes) * G)
        xi_of_type[t].append(xi_b)
        if e.single:
            f_t = chol @ xi_b
        else:
            f_t = chol[e.offset * G:, :] @ np.concatenate(xi_of_type[t])
        for s, i in enumerate(nodes):
            node_out[i] = f_t[s * G:(s + 1) * G]
    joint = np.concatenate([node_out[i] for i in lib.protocols])
    k_r = rbf_kernel_nd(joint, lib.terminal_amplitude, lib.terminal_lengthscale)
    l_r, _ = stable_cholesky(k_r, kernel_jitter)
    truth = l_r @ rng.normal(size=lib.n_protocols * G)
    noise = np.random.default_rng(noise_seed).normal(
        scale=noise_std, size=(lib.n_protocols, G)
    ).reshape(-1)
    if return_details:
        latent = np.concatenate([np.concatenate(xi_of_type[t]) for t in range(lib.n_types)])
        return truth, noise, {"node_out": node_out, "joint_penultimate": joint,
                              "latent": latent, "jitter_blocks": jit_blocks}
    return truth, noise


# ---------------------------------------------------------------------------
# library definitions
# ---------------------------------------------------------------------------
class _Builder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.types: list[GraphType] = []
        self.nodes: list[GraphNode] = []

    def type(self, name: str, arity: int, lengthscale: float) -> str:
        self.types.append(GraphType(name, arity, lengthscale))
        return name

    def node(self, type_name: str, parents: tuple[int, ...] = ()) -> int:
        self.nodes.append(GraphNode(type_name, tuple(parents)))
        return len(self.nodes) - 1

    def build(self, protocols: list[int]) -> GraphLibrary:
        lib = GraphLibrary(self.name, tuple(self.types), tuple(self.nodes), tuple(protocols))
        lib.validate()
        return lib


def _w16() -> GraphLibrary:
    """PB-W16 pattern: L1 {A, B}, L2 C1..C8 each applied to A and B; protocols C1@A..C8@A,
    C1@B..C8@B (l1-major)."""
    b = _Builder("PB-W16")
    b.type("A", 1, LAYER1_LENGTHSCALE)
    b.type("B", 1, LAYER1_LENGTHSCALE)
    na, nb = b.node("A"), b.node("B")
    at_a, at_b = [], []
    for k in range(1, 9):
        b.type(f"C{k}", 1, INNER_LENGTHSCALE)
        at_a.append(b.node(f"C{k}", (na,)))
        at_b.append(b.node(f"C{k}", (nb,)))
    return b.build(at_a + at_b)


def _chain_layer_pairs(b: _Builder, names: list[str], pairs: list[tuple[int, int]],
                       arity: int = 1) -> list[int]:
    """Types ``names`` (two per pair, in order) applied to ``pairs`` of parent nodes (chain,
    arity 1: each type applied separately to the two parents of its pair).  Returns the new
    nodes in node order."""
    out = []
    for k, name in enumerate(names):
        b.type(name, arity, INNER_LENGTHSCALE)
        for parent in pairs[k]:
            out.append(b.node(name, (parent,)))
    return out


def _d4() -> GraphLibrary:
    b = _Builder("PB-D4")
    b.type("A", 1, LAYER1_LENGTHSCALE)
    b.type("B", 1, LAYER1_LENGTHSCALE)
    na, nb = b.node("A"), b.node("B")
    q = _chain_layer_pairs(b, ["C1", "C2"], [(na, nb), (na, nb)])  # q0=C1@A q1=C1@B q2=C2@A q3=C2@B
    pair_cycle = [(q[0], q[1]), (q[0], q[1]), (q[2], q[3]), (q[2], q[3]),
                  (q[0], q[2]), (q[0], q[2]), (q[1], q[3]), (q[1], q[3])]
    d = _chain_layer_pairs(b, [f"D{k}" for k in range(1, 9)], pair_cycle)
    return b.build(d)


def _d5_from(b: _Builder, q: list[int]) -> list[int]:
    """PB-D5's L3 and L4 on the four nodes q0..q3; returns the 16 L4 nodes."""
    r = _chain_layer_pairs(
        b, ["D1", "D2", "D3", "D4"],
        [(q[0], q[1]), (q[2], q[3]), (q[0], q[2]), (q[1], q[3])],
    )  # r0=D1@q0 r1=D1@q1 r2=D2@q2 r3=D2@q3 r4=D3@q0 r5=D3@q2 r6=D4@q1 r7=D4@q3
    e_pairs = [(r[0], r[1]), (r[2], r[3]), (r[4], r[5]), (r[6], r[7]),
               (r[0], r[2]), (r[1], r[3]), (r[4], r[6]), (r[5], r[7])]
    return _chain_layer_pairs(b, [f"E{k}" for k in range(1, 9)], e_pairs)


def _d5() -> GraphLibrary:
    b = _Builder("PB-D5")
    b.type("A", 1, LAYER1_LENGTHSCALE)
    b.type("B", 1, LAYER1_LENGTHSCALE)
    na, nb = b.node("A"), b.node("B")
    q = _chain_layer_pairs(b, ["C1", "C2"], [(na, nb), (na, nb)])
    return b.build(_d5_from(b, q))


def _d6() -> GraphLibrary:
    b = _Builder("PB-D6")
    b.type("A", 1, LAYER1_LENGTHSCALE)
    na = b.node("A")
    s = _chain_layer_pairs(b, ["B1", "B2"], [(na,), (na,)])  # s0=B1@A, s1=B2@A
    q = _chain_layer_pairs(b, ["C1", "C2"], [(s[0], s[1]), (s[0], s[1])])
    return b.build(_d5_from(b, q))


def _merge_layer(b: _Builder, names: list[str], arity: int, groups: list[tuple[int, ...]],
                 assignment: list[tuple[int, int]]) -> list[int]:
    """Merge types ``names`` (arity ``arity``); type k is applied to the two parent-tuples
    ``groups[assignment[k][0]]`` and ``groups[assignment[k][1]]``.  Returns new nodes."""
    out = []
    for k, name in enumerate(names):
        b.type(name, arity, INNER_LENGTHSCALE)
        for gi in assignment[k]:
            out.append(b.node(name, groups[gi]))
    return out


# C1,C2 on (g0,g1); C3,C4 on (g2,g3); C5,C6 on (g0,g2); C7,C8 on (g1,g3)
_MERGE_ASSIGNMENT = [(0, 1), (0, 1), (2, 3), (2, 3), (0, 2), (0, 2), (1, 3), (1, 3)]


def _mrg2() -> GraphLibrary:
    b = _Builder("PB-MRG2")
    a = []
    for k in range(1, 5):
        b.type(f"A{k}", 1, LAYER1_LENGTHSCALE)
        a.append(b.node(f"A{k}"))
    pairs = [(a[0], a[1]), (a[2], a[3]), (a[0], a[2]), (a[1], a[3])]
    c = _merge_layer(b, [f"C{k}" for k in range(1, 9)], 2, pairs, _MERGE_ASSIGNMENT)
    return b.build(c)


def _mrg3() -> GraphLibrary:
    b = _Builder("PB-MRG3")
    a = []
    for k in range(1, 5):
        b.type(f"A{k}", 1, LAYER1_LENGTHSCALE)
        a.append(b.node(f"A{k}"))
    triples = [(a[0], a[1], a[2]), (a[0], a[1], a[3]), (a[0], a[2], a[3]), (a[1], a[2], a[3])]
    c = _merge_layer(b, [f"C{k}" for k in range(1, 9)], 3, triples, _MERGE_ASSIGNMENT)
    return b.build(c)


def _mrg22() -> GraphLibrary:
    b = _Builder("PB-MRG22")
    a = []
    for k in range(1, 5):
        b.type(f"A{k}", 1, LAYER1_LENGTHSCALE)
        a.append(b.node(f"A{k}"))
    # B1, B2 (arity 2) each on pairs (A1,A2), (A3,A4): u0=B1@(A1,A2) u1=B1@(A3,A4) u2=B2@(A1,A2) u3=B2@(A3,A4)
    u = _merge_layer(b, ["B1", "B2"], 2, [(a[0], a[1]), (a[2], a[3])], [(0, 1), (0, 1)])
    pairs = [(u[0], u[1]), (u[2], u[3]), (u[0], u[3]), (u[1], u[2])]
    c = _merge_layer(b, [f"C{k}" for k in range(1, 9)], 2, pairs, _MERGE_ASSIGNMENT)
    return b.build(c)


GRAPH_LIBRARY_BUILDERS: dict[str, Callable[[], GraphLibrary]] = {
    "PB-W16": _w16,
    "PB-D4": _d4,
    "PB-D5": _d5,
    "PB-D6": _d6,
    "PB-MRG2": _mrg2,
    "PB-MRG3": _mrg3,
    "PB-MRG22": _mrg22,
}
_CACHE: dict[str, GraphLibrary] = {}


def get_graph_library(name: str) -> GraphLibrary:
    if name not in GRAPH_LIBRARY_BUILDERS:
        raise KeyError(f"unknown graph library {name!r}; choices {sorted(GRAPH_LIBRARY_BUILDERS)}")
    if name not in _CACHE:
        _CACHE[name] = GRAPH_LIBRARY_BUILDERS[name]()
    return _CACHE[name]
