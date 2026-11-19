"""Per-protocol descriptor context for the OFML composition (OFML-GPN).

A protocol's *context descriptor* is a fixed-length vector computed from its step
sequence and appended to every operation's input during both fitting and
composition. A shared operation GP can then condition on protocol context, which
is what lets a single per-operation model resolve hidden, protocol-level
interactions that are invisible to the operation's local state alone:

* ``presence`` (L0): one bit per operation type (1 if the type appears). Resolves
  a *set-gate* interaction such as s7 three-body (readout term active iff
  {A,B,C} all present).
* ``counts`` (L1): the multiset -- how many times each operation type appears.
  Strictly subsumes presence; resolves *count-graded* effects such as s8
  dose-count saturation (readout saturates in the number of B repetitions),
  which presence collapses.

An ARD kernel simply learns to ignore the context dimensions for operations that
do not depend on them (large lengthscales), so appending context to every
operation is harmless; only the observer that actually depends on the interaction
uses them. This mirrors the standalone gp_network descriptor ablation, now native
to the ofml OFMLModel so the ablation runs on the primary benchmark (s7/s8).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .base import OFMLModel

# Operation vocabulary for the descriptor (excludes the observer ``O``, which is
# the readout, not a protocol ingredient). Matches gp_network's _CONTEXT_VOCAB.
CONTEXT_VOCAB: tuple[str, ...] = ("A", "B", "C", "D", "M")


def steps_context(steps: str, kind: str, vocab: tuple[str, ...] = CONTEXT_VOCAB) -> np.ndarray:
    """Descriptor vector for a protocol's ``steps`` (space-separated op string).

    ``kind`` in {"none", "presence", "counts"}. "none" -> empty vector."""

    if kind in (None, "none"):
        return np.zeros(0, dtype=float)
    counts = {t: 0 for t in vocab}
    for token in str(steps).split():
        if token in counts:
            counts[token] += 1
    if kind == "presence":
        return np.array([float(counts[t] > 0) for t in vocab], dtype=float)
    if kind == "counts":
        return np.array([float(counts[t]) for t in vocab], dtype=float)
    raise ValueError(f"Unknown context kind '{kind}'. Expected none/presence/counts.")


@dataclasses.dataclass
class ContextOFMLModel(OFMLModel):
    """OFMLModel that appends a per-protocol descriptor to every operation input.

    The descriptor is derived from each row's ``steps`` and re-appended fresh at
    every step (operations output the raw state; only their *input* is augmented),
    exactly matching the augmentation the fit-side examples receive."""

    context_kind: str = "none"

    def _augment_input(self, state: np.ndarray, steps: str) -> np.ndarray:
        ctx = steps_context(steps, self.context_kind)
        if ctx.size == 0:
            return state
        state = np.atleast_2d(np.asarray(state, dtype=float))
        return np.concatenate([state, np.tile(ctx, (state.shape[0], 1))], axis=1)
