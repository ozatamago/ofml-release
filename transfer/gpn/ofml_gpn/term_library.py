"""ofml_gpn.term_library -- build a ``GraphLibrary`` from protocol TERMS (ofml_gpn, 2026-11-04).

A protocol is written as the term of its penultimate node, e.g. ``"C2(C1(A))"`` or ``"M(M(A,B),A)"``
(the terminal R is implicit: protocol = R o term).  Layer-1 types are nullary (``"A"``), chain types unary,
merge types binary (any arity is allowed).  Equal sub-terms are ONE node (so ``C1(A)`` is shared by
``R o C1 o A`` and ``R o C2 o C1 o A``).  Numpy-only.

``purpose1_library(...)`` is the purpose-1 transfer benchmark (layer-1 A, B; chains C1, C2; merge M;
terminal R).  ``PURPOSE1_DRAFT`` is the DRAFT split (sources / targets) following the author's plan; the final
split is still being decided, so every list is a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph_library import (
    AMPLITUDE,
    INNER_LENGTHSCALE,
    LAYER1_LENGTHSCALE,
    TERMINAL_AMPLITUDE,
    TERMINAL_LENGTHSCALE,
    GraphLibrary,
    GraphNode,
    GraphType,
)


def parse_term(text: str):
    """``"M(M(A,B),A)"`` -> ``("M", (("M", (("A", ()), ("B", ()))), ("A", ())))``."""
    s = text.replace(" ", "")
    pos = 0

    def term():
        nonlocal pos
        start = pos
        while pos < len(s) and (s[pos].isalnum() or s[pos] == "_"):
            pos += 1
        name = s[start:pos]
        if not name:
            raise ValueError(f"bad term {text!r} at {start}")
        args = []
        if pos < len(s) and s[pos] == "(":
            pos += 1
            args.append(term())
            while s[pos] == ",":
                pos += 1
                args.append(term())
            if s[pos] != ")":
                raise ValueError(f"bad term {text!r}: expected ')' at {pos}")
            pos += 1
        return (name, tuple(args))

    out = term()
    if pos != len(s):
        raise ValueError(f"bad term {text!r}: trailing text at {pos}")
    return out


def term_str(t) -> str:
    name, args = t
    return name if not args else f"{name}(" + ",".join(term_str(a) for a in args) + ")"


def term_types(t) -> set[str]:
    name, args = t
    out = {name}
    for a in args:
        out |= term_types(a)
    return out


@dataclass(frozen=True)
class TermLibrary:
    library: GraphLibrary
    protocol_terms: tuple[str, ...]
    sources: tuple[int, ...]  # protocol indices
    targets: tuple[int, ...]
    node_terms: tuple[str, ...]


def build_term_library(
    name: str,
    layer1: list[str],
    inner: dict[str, int],
    sources: list[str],
    targets: list[str],
    *,
    layer1_lengthscale: float = LAYER1_LENGTHSCALE,
    inner_lengthscale: float = INNER_LENGTHSCALE,
    amplitude: float = AMPLITUDE,
    terminal_lengthscale: float = TERMINAL_LENGTHSCALE,
    terminal_amplitude: float = TERMINAL_AMPLITUDE,
    require_target_types_in_sources: bool = True,
    stage_policy: str = "auto",
) -> TermLibrary:
    """Types in the order ``layer1`` then ``inner`` (dict name -> arity, insertion order).  Protocols = sources
    then targets (that order); nodes are created by a post-order walk of the protocol terms in protocol order,
    so node indices are topological.  Checks: arities, no duplicate protocol, and (default) every type used by a
    target also appears in some source."""
    types = [GraphType(n, 1, layer1_lengthscale, amplitude) for n in layer1]
    types += [GraphType(n, int(a), inner_lengthscale, amplitude) for n, a in inner.items()]
    arity = {n: 0 for n in layer1}
    arity.update({n: int(a) for n, a in inner.items()})
    nodes: list[GraphNode] = []
    node_terms: list[str] = []
    memo: dict = {}

    def make(t) -> int:
        if t in memo:
            return memo[t]
        nm, args = t
        if nm not in arity:
            raise ValueError(f"{name}: unknown type {nm!r} in {term_str(t)}")
        if len(args) != arity[nm]:
            raise ValueError(f"{name}: {term_str(t)} has {len(args)} arguments, type {nm} has arity {arity[nm]}")
        parents = tuple(make(a) for a in args)
        nodes.append(GraphNode(nm, parents))
        node_terms.append(term_str(t))
        memo[t] = len(nodes) - 1
        return memo[t]

    parsed_s = [parse_term(x) for x in sources]
    parsed_t = [parse_term(x) for x in targets]
    all_terms = parsed_s + parsed_t
    if len({term_str(t) for t in all_terms}) != len(all_terms):
        raise ValueError(f"{name}: duplicate protocol term")
    protocols = [make(t) for t in all_terms]
    if require_target_types_in_sources:
        src_types = set().union(*(term_types(t) for t in parsed_s)) if parsed_s else set()
        for t in parsed_t:
            missing = term_types(t) - src_types
            if missing:
                raise ValueError(f"{name}: target {term_str(t)} uses types {sorted(missing)} absent from every source")
    used = {n.type_name for n in nodes}
    types = [ty for ty in types if ty.name in used]
    lib = GraphLibrary(name, tuple(types), tuple(nodes), tuple(protocols),
                       terminal_lengthscale=terminal_lengthscale, terminal_amplitude=terminal_amplitude,
                       stage_policy=stage_policy)
    lib.validate()
    n_s = len(parsed_s)
    return TermLibrary(lib, tuple(term_str(t) for t in all_terms), tuple(range(n_s)),
                       tuple(range(n_s, len(all_terms))), tuple(node_terms))


def _chains2():
    return [f"C{k}({x})" for x in ("A", "B") for k in (1, 2)]


def _chains3():
    return [f"C{j}(C{k}({x}))" for x in ("A", "B") for k in (1, 2) for j in (1, 2)]


# DRAFT split (2026-11-04; to be finalised with the author).  Sources: all depth-2 chains R o C_k o X, the
# depth-2 merge in both orders, and half of the depth-3 chains with one order-swapped pair per layer-1 input
# (C2(C1(A)) / C1(C2(A)) seen together; on B the two same-type chains).  Targets: the other four depth-3
# chains (each the order swap or the same-type counterpart of a source) and three double merges.
PURPOSE1_DRAFT = {
    "sources": _chains2() + ["M(A,B)", "M(B,A)", "C2(C1(A))", "C1(C2(A))", "C1(C1(B))", "C2(C2(B))"],
    "targets": ["C1(C1(A))", "C2(C2(A))", "C2(C1(B))", "C1(C2(B))",
                "M(M(A,B),A)", "M(M(B,A),B)", "M(A,M(A,B))"],
}


def purpose1_library(sources: list[str] | None = None, targets: list[str] | None = None,
                     name: str = "PB-P1-DRAFT", **kw) -> TermLibrary:
    """Purpose-1 benchmark library: layer-1 A, B; chains C1, C2 (arity 1); merge M (arity 2); terminal R.
    BENCH9 constants (amplitude 0.70, lengthscale 1.0 layer 1 / 0.5 inner and R; grid/noise are the caller's:
    linspace(0, 4, 64), noise sd 0.1).  Defaults = ``PURPOSE1_DRAFT``."""
    return build_term_library(
        name, ["A", "B"], {"C1": 1, "C2": 1, "M": 2},
        list(PURPOSE1_DRAFT["sources"] if sources is None else sources),
        list(PURPOSE1_DRAFT["targets"] if targets is None else targets), **kw)
