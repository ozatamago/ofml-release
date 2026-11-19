"""Protocol strings with branches, merges and input blocks -> DAG (Experiment 3,
REPORTS/EXP3_MERGE_DAG_SPEC.md, Sec. 3.4).

Grammar (tokens are separated by single spaces, so ``steps.split()`` still counts operations):

    protocol := seq "O"
    seq      := ["@k"] head op*             (unary operations applied in order)
    head     := op | "M" "(" seq "|" seq ")"
    op       := one of A B C D (or any other non-structural token)

A head op reads the input block x^(k) given by its marker ``@k`` (default ``@1``); block k is the
design-variable columns [2(k-1), 2k). A merge ``M ( s1 | s2 )`` takes the outputs of branches s1 and
s2 as its first and second input, in that order. ``M`` followed by "(" is a merge; ``M`` elsewhere
is a unary op (the depth-5 chain benchmark).

The result is a ``PDAG``: input nodes have type "in" and no parents (one per input block used, in
order of first use), parents are ordered by role, and ``out`` is the observer node. A marker-free
chain such as ``"A B O"`` gives the same types/parents/out as ``dagnn.chain_from_steps``.
"""

from __future__ import annotations

import dataclasses
import functools

INPUT_TYPE = "in"
STRUCT = ("(", "|", ")")


@dataclasses.dataclass(frozen=True)
class PDAG:
    types: tuple[str, ...]
    parents: tuple[tuple[int, ...], ...]
    out: int
    blocks: tuple[int, ...]          # input block (1-based) of each input node, 0 for other nodes

    @property
    def key(self):
        return (self.types, self.parents, self.out, self.blocks)

    @property
    def n_blocks(self) -> int:
        return max(self.blocks)

    @property
    def x_dim(self) -> int:
        return 2 * self.n_blocks

    @property
    def input_nodes(self) -> tuple[int, ...]:
        return tuple(v for v, t in enumerate(self.types) if t == INPUT_TYPE)

    def is_chain(self) -> bool:
        return (self.blocks[0] == 1 and sum(b > 0 for b in self.blocks) == 1
                and all(p == ((i - 1,) if i > 0 else ()) for i, p in enumerate(self.parents))
                and self.out == len(self.types) - 1)


@functools.lru_cache(maxsize=4096)
def parse_protocol(steps: str) -> PDAG:
    toks = str(steps).split()
    if not toks or toks[-1] != "O":
        raise ValueError(f"protocol must end with the observer O: {steps!r}")
    types: list[str] = []
    parents: list[tuple[int, ...]] = []
    blocks: list[int] = []
    input_of: dict[int, int] = {}
    pos = 0

    def add(t: str, pa: tuple[int, ...], blk: int = 0) -> int:
        types.append(t); parents.append(pa); blocks.append(blk)
        return len(types) - 1

    def input_node(k: int) -> int:
        if k not in input_of:
            input_of[k] = add(INPUT_TYPE, (), k)
        return input_of[k]

    def seq() -> int:
        nonlocal pos
        blk = 1
        if pos < len(toks) and toks[pos].startswith("@"):
            blk = int(toks[pos][1:])
            if blk < 1:
                raise ValueError(f"input blocks are 1-based: {steps!r}")
            pos += 1
        if pos >= len(toks) or toks[pos] in STRUCT or toks[pos] == "O" and pos == len(toks) - 1:
            raise ValueError(f"empty branch in {steps!r}")
        if toks[pos] == "M" and pos + 1 < len(toks) and toks[pos + 1] == "(":
            pos += 2
            a = seq()
            if toks[pos] != "|":
                raise ValueError(f"expected '|' in {steps!r}")
            pos += 1
            b = seq()
            if toks[pos] != ")":
                raise ValueError(f"expected ')' in {steps!r}")
            pos += 1
            cur = add("M", (a, b))
        else:
            src = input_node(blk)
            cur = add(toks[pos], (src,))
            pos += 1
        while pos < len(toks) and toks[pos] not in STRUCT and not (toks[pos] == "O" and pos == len(toks) - 1):
            if toks[pos] == "M" and pos + 1 < len(toks) and toks[pos + 1] == "(":
                raise ValueError(f"a merge must start a branch: {steps!r}")
            if toks[pos].startswith("@"):
                raise ValueError(f"an input marker must start a branch: {steps!r}")
            cur = add(toks[pos], (cur,))
            pos += 1
        return cur

    body = seq() if len(toks) > 1 else input_node(1)
    if pos != len(toks) - 1:
        raise ValueError(f"trailing tokens in {steps!r}")
    out = add("O", (body,))
    dag = PDAG(tuple(types), tuple(parents), out, tuple(blocks))
    if sorted(set(b for b in blocks if b)) != list(range(1, dag.n_blocks + 1)):
        raise ValueError(f"input blocks must be 1..K without gaps: {steps!r}")
    return dag


def op_types(steps: str) -> list[str]:
    """Intermediate operation types of a protocol (structure tokens, markers and the observer removed)."""
    return [t for t in str(steps).split() if t not in STRUCT and not t.startswith("@")][:-1]


def x_of(row, x_dim: int) -> list[float]:
    """Design variable of a row as a list (keys x1..x{x_dim})."""
    return [float(row[f"x{i}"]) for i in range(1, x_dim + 1)]
