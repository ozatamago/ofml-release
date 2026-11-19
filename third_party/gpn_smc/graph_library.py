"""gpn_smc.graph_library -- generic library-graph language (BENCH9, 2026-11-01b).

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

* the type list is topologically ordered (every parent node belongs to an EARLIER type)
  and node indices are topologically ordered (every parent index < the node's index);
* layer-1 types have ``arity == 1`` and nodes with ``parents == ()``; every other node
  has ``len(parents) == arity`` of its type.

Teacher RNG order (one ``default_rng(teacher_seed)``): for each type in order
``f_t = chol(K(Z_t)) @ rng.normal(size=n_nodes_t * G)``; then
``truth = chol(K_R) @ rng.normal(size=P * G)``.  Noise = ``default_rng(noise_seed)
.normal(scale=noise_std, size=(P, G)).reshape(-1)`` (the existing rule).

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
class GraphLibrary:
    name: str
    types: tuple[GraphType, ...]
    nodes: tuple[GraphNode, ...]
    protocols: tuple[int, ...]  # node index of each protocol's penultimate node
    terminal_lengthscale: float = TERMINAL_LENGTHSCALE
    terminal_amplitude: float = TERMINAL_AMPLITUDE

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
        """Position of each node inside its type's block (0-based, node order)."""
        slot = []
        counters = [0] * len(self.types)
        for t in self.node_type:
            slot.append(counters[t])
            counters[t] += 1
        return tuple(slot)

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
                if self.node_type[q] >= self.node_type[i]:
                    raise ValueError(f"{self.name}: node {i} parent {q} belongs to a later/same type")
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
    """``Z_t`` of shape ``(n_nodes_t * G, arity_t)`` from the parent outputs ``node_out``."""
    rows = []
    for i in lib.type_nodes[type_idx]:
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
    for t, ty in enumerate(lib.types):
        z = type_inputs(lib, t, node_out, warped_x)
        key = (z.tobytes(), z.shape, ty.lengthscale, ty.amplitude)
        chol = chol_cache.get(key)
        if chol is None:
            chol, _ = stable_cholesky(rbf_kernel_nd(z, ty.amplitude, ty.lengthscale), kernel_jitter)
            chol_cache[key] = chol
        nodes = lib.type_nodes[t]
        f_t = chol @ rng.normal(size=len(nodes) * G)
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
        return truth, noise, {"node_out": node_out, "joint_penultimate": joint}
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
