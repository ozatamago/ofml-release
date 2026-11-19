"""True operation semantics for synthetic protocol-transfer benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


OperationMap = Mapping[str, Callable[[np.ndarray], np.ndarray]]


def apply_step(state: np.ndarray, step: str, semantic_variant: str = "default") -> np.ndarray:
    """Apply one true operation for a semantic variant."""

    operations = _operations_for_variant(semantic_variant)
    try:
        return operations[step](state)
    except KeyError as error:
        raise ValueError(f"Unknown synthetic operation '{step}'.") from error


def readout(
    state: np.ndarray,
    semantic_variant: str = "default",
    steps: Sequence[str] = (),
) -> float:
    """Evaluate the true scalar readout for a semantic variant."""

    if semantic_variant == "nonlinear":
        value = 2.0 * state[0] - 1.6 * state[1] + 1.2 * state[0] * state[1] - 0.1
        return float(1.4 * _scaled_sigmoid(value) + 0.25 * np.sin(2.0 * np.pi * (state[0] - state[1])))
    if semantic_variant == "nonlinear_shifted":
        value = 1.2 * state[0] - 2.0 * state[1] - 0.9 * state[0] * state[1] + 0.35
        return float(2.2 * _scaled_sigmoid(value) + 0.75 * np.sin(2.0 * np.pi * (state[0] + 0.3 * state[1])))
    if semantic_variant == "nonlinear_lipschitz":
        return float(1.15 * state[0] - 0.85 * state[1] + 0.35 * np.sin(np.pi * (state[0] + state[1])))
    if semantic_variant == "nonlinear_misspecified":
        return float(
            1.55
            * _scaled_sigmoid(
                1.3 * state[0]
                - 1.1 * state[1]
                + 0.45 * np.sin(18.0 * np.pi * state[0])
                + 0.35 * np.sin(16.0 * np.pi * state[1])
            )
            + 0.35 * np.sin(4.0 * np.pi * (state[0] - state[1]))
        )
    if semantic_variant == "nonlinear_three_body":
        base = 1.5 * state[0] - 1.25 * state[1] + 0.25 * state[0] * state[1] - 0.15
        interaction = (
            2.2 * state[0] * state[1]
            + 0.7 * np.sin(np.pi * state[0] * state[1])
            + 1.1 * np.sin(2.0 * np.pi * (state[0] - state[1]))
            if _contains_three_body(steps)
            else 0.0
        )
        return float(1.5 * _scaled_sigmoid(base) + interaction)

    family = _bo_family(semantic_variant)
    if family is not None:
        return _bo_readout(state, family=family, semantic_variant=semantic_variant, steps=steps)

    if semantic_variant == "linear_noncommutative":
        return float(0.9 * state[0] - 0.6 * state[1] + 0.2)
    if semantic_variant == "order_rotation":
        # low-noise, NON-saturating linear readout: the whole protocol is linear
        # in x (trivially fittable per protocol), so the ONLY difficulty is the
        # ORDER of the strongly non-commuting ops (see _order_rotation_operations).
        return float(1.2 * state[0] - 0.8 * state[1])
    if semantic_variant == "nonlinear_order":
        # linear readout on the tanh-composed state (nonlinearity lives in the ops)
        return float(1.2 * state[0] - 0.8 * state[1])
    if semantic_variant == "shifted":
        return float(0.8 * state[0] - 1.1 * state[1] + 0.35 * np.sin(1.4 * np.pi * state.sum()))
    if semantic_variant == "unstable":
        return float(1.1 * state[0] - 0.6 * state[1] + 0.8 * np.sin(12.0 * np.pi * state.sum()))
    if semantic_variant == "lipschitz":
        return float(0.7 * state[0] + 0.3 * state[1])
    if semantic_variant == "misspecified":
        return float(0.9 * state[0] - 0.4 * state[1] + 0.25 * np.sin(24.0 * np.pi * state.sum()))
    if semantic_variant == "three_body":
        linear = 1.1 * state[0] - 0.8 * state[1] - 0.2
        interaction = 3.0 * state[0] * state[1] if _contains_three_body(steps) else 0.0
        return float(_sigmoid(linear + interaction))
    if semantic_variant == "dose_count":
        # L1 count-graded dose saturation (#12): the readout gains a term that
        # SATURATES in the number of times the dose op B appears. Because B is an
        # identity in state (see _dose_count_operations), the effect is invisible
        # to a per-operation model that conditions only on the state -- it needs a
        # COUNT (multiset) descriptor of the protocol to resolve it. Counterpart to
        # three_body's presence gate, one descriptor level up.
        linear = 1.0 * state[0] - 0.7 * state[1] - 0.1
        count_b = sum(1 for step in steps if step == "B")
        dose = 2.5 * (1.0 - np.exp(-0.9 * count_b))
        return float(_sigmoid(linear + dose))
    return float(1.2 * state[0] - 0.7 * state[1] + 0.25 * np.sin(np.pi * state.sum()))


def _operations_for_variant(semantic_variant: str) -> OperationMap:
    family = _bo_family(semantic_variant)
    if family is not None:
        return _bo_operations(semantic_variant, family=family)
    if semantic_variant == "nonlinear":
        return _nonlinear_operations()
    if semantic_variant == "nonlinear_shifted":
        return _nonlinear_shifted_operations()
    if semantic_variant == "nonlinear_lipschitz":
        return _nonlinear_lipschitz_operations()
    if semantic_variant == "nonlinear_misspecified":
        return _nonlinear_misspecified_operations()
    if semantic_variant == "nonlinear_three_body":
        return _nonlinear_three_body_operations()
    if semantic_variant == "linear_noncommutative":
        return _linear_noncommutative_operations()
    if semantic_variant == "order_rotation":
        return _order_rotation_operations()
    if semantic_variant == "nonlinear_order":
        return _nonlinear_order_operations()
    if semantic_variant == "shifted":
        return _shifted_operations()
    if semantic_variant == "unstable":
        return _unstable_operations()
    if semantic_variant == "lipschitz":
        return _lipschitz_operations()
    if semantic_variant == "misspecified":
        return _misspecified_operations()
    if semantic_variant == "three_body":
        return _three_body_operations()
    if semantic_variant == "dose_count":
        return _dose_count_operations()
    return _default_operations()


def _bo_family(semantic_variant: str) -> str | None:
    # bo_all_multimodal first: it is a distinct family (multimodal ops AND
    # multimodal readout) and must not be swallowed by a prefix match.
    for family in ("bo_all_multimodal", "bo_smooth", "bo_multimodal_readout", "bo_operation_multimodal"):
        if semantic_variant == family or semantic_variant.startswith(f"{family}_"):
            return family
    return None


def _bo_operations(semantic_variant: str, family: str) -> OperationMap:
    if semantic_variant.endswith("_shifted"):
        return _bo_shifted_operations(family)
    if semantic_variant.endswith("_lipschitz"):
        return _bo_lipschitz_operations(family)
    if semantic_variant.endswith("_misspecified"):
        return _bo_misspecified_operations(family)
    if semantic_variant.endswith("_three_body"):
        return _bo_three_body_operations(family)
    if family in ("bo_operation_multimodal", "bo_all_multimodal"):
        return _bo_operation_multimodal_operations()
    return _bo_smooth_operations()


def _bo_readout(
    state: np.ndarray,
    family: str,
    semantic_variant: str,
    steps: Sequence[str],
) -> float:
    if semantic_variant.endswith("_shifted"):
        return _bo_shifted_readout(state, family=family)
    if semantic_variant.endswith("_lipschitz"):
        return float(0.85 * state[0] - 0.55 * state[1] + 0.35 * _branin_feature(state))
    if semantic_variant.endswith("_misspecified"):
        return float(
            0.7 * _ackley_feature(state)
            + 0.35 * _rastrigin_feature(state)
            + 0.25 * np.sin(18.0 * np.pi * state[0])
            - 0.2 * np.cos(16.0 * np.pi * state[1])
        )
    if semantic_variant.endswith("_three_body"):
        if family == "bo_all_multimodal":
            base = 0.75 * _ackley_feature(state) + 0.45 * _rastrigin_feature(state) + 0.25 * _levy_feature(state)
        else:
            base = 0.65 * _branin_feature(state) + 0.45 * _goldstein_price_feature(state)
        interaction = (
            1.6 * state[0] * state[1]
            + 0.7 * np.sin(2.0 * np.pi * state[0] * state[1])
            if _contains_three_body(steps)
            else 0.0
        )
        return float(base + interaction)
    if family in ("bo_multimodal_readout", "bo_all_multimodal"):
        return float(0.75 * _ackley_feature(state) + 0.45 * _rastrigin_feature(state) + 0.25 * _levy_feature(state))
    if family == "bo_operation_multimodal":
        return float(0.95 * _branin_feature(state) + 0.45 * _hartmann3_feature(state) + 0.25 * state[0] - 0.2 * state[1])
    return float(0.9 * _branin_feature(state) + 0.55 * _hartmann3_feature(state) + 0.25 * _goldstein_price_feature(state))


def _default_operations() -> OperationMap:
    return {
        "A": lambda s: np.array([s[0] + 0.35 * s[1], s[1]], dtype=float),
        "B": lambda s: np.array([s[0], 0.65 * s[1] + 0.15], dtype=float),
        "C": lambda s: np.array([np.sin(np.pi * s[0]) * 0.5 + 0.5, s[1]], dtype=float),
        "D": lambda s: np.array([s[0] * s[1] + 0.1, 0.8 * s[1] + 0.2 * s[0]], dtype=float),
        "M": lambda s: np.array([(s[0] + s[1]) / 2.0, s[0] - s[1]], dtype=float),
        "O": lambda s: np.array([s[0], np.cos(np.pi * s[1])], dtype=float),
    }


def _nonlinear_operations() -> OperationMap:
    return {
        "A": lambda s: np.array([2.0 * _sigmoid(2.4 * s[0] - 0.8 * s[1] + 0.1) - 0.8, s[1]], dtype=float),
        "B": lambda s: np.array([s[0], 2.0 * _sigmoid(2.0 * s[1] + 0.7 * s[0] - 0.2) - 0.8], dtype=float),
        "C": lambda s: np.array(
            [
                1.2 * _scaled_sigmoid(2.2 * s[0] + 0.8 * s[1] - 1.0),
                1.2 * _scaled_sigmoid(2.0 * s[1] - 0.6 * s[0] - 0.6),
            ],
            dtype=float,
        ),
        "D": lambda s: np.array(
            [
                1.1 * _scaled_sigmoid(1.8 * s[0] * s[1] + 0.6 * s[0] - 0.2),
                1.1 * _scaled_sigmoid(1.3 * s[1] + 0.5 * s[0] * s[1] - 0.1),
            ],
            dtype=float,
        ),
        "M": lambda s: np.array(
            [
                s[0] + 0.4 * np.tanh(s[0] - s[1]),
                s[1] + 0.4 * np.tanh(s[0] + s[1]),
            ],
            dtype=float,
        ),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _nonlinear_shifted_operations() -> OperationMap:
    return {
        "A": lambda s: np.array([1.8 * _sigmoid(1.7 * s[0] - 1.1 * s[1] + 0.35) - 0.65, s[1] + 0.08], dtype=float),
        "B": lambda s: np.array([s[0] + 0.08, 1.8 * _sigmoid(1.35 * s[1] + 0.9 * s[0] - 0.05) - 0.65], dtype=float),
        "C": lambda s: np.array(
            [
                1.05 * _scaled_sigmoid(2.0 * s[0] + 1.0 * s[1] - 0.8) + 0.1,
                1.05 * _scaled_sigmoid(1.8 * s[1] - 0.9 * s[0] - 0.35) - 0.1,
            ],
            dtype=float,
        ),
        "D": lambda s: np.array(
            [
                _scaled_sigmoid(1.4 * s[0] * s[1] + 0.65 * s[0] + 0.05),
                _scaled_sigmoid(0.9 * s[1] + 0.55 * s[0] * s[1] + 0.1),
            ],
            dtype=float,
        ),
        "M": lambda s: np.array(
            [
                s[0] + 0.35 * np.tanh(0.7 * s[0] - 1.2 * s[1] + 0.25),
                s[1] + 0.35 * np.tanh(1.1 * s[1] + 0.5 * s[0] - 0.2),
            ],
            dtype=float,
        ),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _nonlinear_lipschitz_operations() -> OperationMap:
    alpha = 1.45
    return {
        "A": lambda s: _nonlinear_operations()["A"](s),
        "B": lambda s: _nonlinear_operations()["B"](s),
        "C": lambda s: _nonlinear_operations()["C"](s),
        "D": lambda s: _nonlinear_operations()["D"](s),
        "M": lambda s: alpha * np.asarray(s, dtype=float) + 0.08 * np.tanh(2.0 * np.asarray(s, dtype=float) - 1.0),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _nonlinear_misspecified_operations() -> OperationMap:
    return {
        "A": lambda s: np.array(
            [
                1.8 * _sigmoid(2.0 * s[0] - 0.7 * s[1] + 0.35 * np.sin(22.0 * np.pi * s[0])) - 0.7,
                s[1],
            ],
            dtype=float,
        ),
        "B": lambda s: np.array(
            [
                s[0],
                1.8 * _sigmoid(1.5 * s[1] + 0.4 * s[0] + 0.35 * np.sin(20.0 * np.pi * s[1])) - 0.7,
            ],
            dtype=float,
        ),
        "C": lambda s: np.array(
            [
                1.1 * _scaled_sigmoid(2.4 * s[0] - 1.0 + 0.35 * np.sin(18.0 * np.pi * s[0])),
                1.1 * _scaled_sigmoid(2.1 * s[1] - 0.8 + 0.35 * np.sin(18.0 * np.pi * s[1])),
            ],
            dtype=float,
        ),
        "D": lambda s: _nonlinear_operations()["D"](s),
        "M": lambda s: np.array(
            [
                s[0] + 0.3 * np.tanh(1.2 * s[0] - 0.8 * s[1]) + 0.2 * np.sin(16.0 * np.pi * s.sum()),
                s[1] + 0.3 * np.tanh(1.0 * s[1] + 0.7 * s[0]) + 0.2 * np.sin(14.0 * np.pi * (s[0] - s[1])),
            ],
            dtype=float,
        ),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _nonlinear_three_body_operations() -> OperationMap:
    return {
        "A": lambda s: np.array([1.8 * _sigmoid(2.0 * s[0] + 0.35 * s[1] + 0.1) - 0.7, s[1]], dtype=float),
        "B": lambda s: np.array([s[0], 1.8 * _sigmoid(2.0 * s[1] + 0.35 * s[0] + 0.1) - 0.7], dtype=float),
        "C": lambda s: np.array(
            [
                1.1 * _scaled_sigmoid(2.8 * s[0] + 0.5 * s[1] - 0.4),
                1.1 * _scaled_sigmoid(2.6 * s[1] + 0.4 * s[0] - 0.35),
            ],
            dtype=float,
        ),
        "D": lambda s: _nonlinear_operations()["D"](s),
        "M": lambda s: _nonlinear_operations()["M"](s),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _linear_noncommutative_operations() -> OperationMap:
    """Purely affine, pairwise non-commuting operations.

    Each operation is an affine map s -> W s + b whose matrices do not commute
    (e.g. A and B are shears in different coordinates). Because affine maps are
    globally identified from any input region, a LinearOperationModel can compose
    them in an unseen order (e.g. B->C->A->O) by exact extrapolation, while a
    local MLP cannot. The readout (semantic_variant == "linear_noncommutative")
    is affine and order-independent, so the only difficulty is operation order.
    """

    return {
        "A": lambda s: np.array([s[0] + 0.6 * s[1], s[1]], dtype=float),
        "B": lambda s: np.array([s[0], 0.4 * s[0] + 0.5 * s[1]], dtype=float),
        "C": lambda s: np.array([0.7 * s[0] - 0.3 * s[1] + 0.1, 0.3 * s[0] + 0.8 * s[1]], dtype=float),
        "D": lambda s: np.array([0.5 * s[0] + 0.5 * s[1], 0.2 * s[0] + 0.6 * s[1] + 0.1], dtype=float),
        "M": lambda s: np.array([0.6 * s[0] + 0.4 * s[1], 0.5 * s[0] - 0.5 * s[1]], dtype=float),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _order_rotation_operations() -> OperationMap:
    """Strongly non-commuting, AREA-PRESERVING linear operations (rotation / swap /
    shear). Because they are near-permutations/rotations, ``||AB - BA||`` is large,
    so reordering the same modules changes the composed state -- and hence the linear
    readout -- by more than one output std. Yet each op is linear and non-saturating,
    so any single protocol is a globally-linear (trivially learnable) map of x: the
    difficulty is ENTIRELY the operation order. This isolates order-identifiability
    as sharply as ``dose``/repetition isolates count-identifiability. Only A/B/C/M are
    used by the order scenarios; D is provided for vocabulary completeness."""

    return {
        "A": lambda s: np.array([-s[1], s[0]], dtype=float),                 # rotation +90 deg
        "B": lambda s: np.array([s[1], s[0]], dtype=float),                  # coordinate swap / reflection
        "C": lambda s: np.array([s[0], 1.2 * s[0] + s[1]], dtype=float),     # strong shear
        "D": lambda s: np.array([s[0] - 0.7 * s[1], s[1]], dtype=float),     # shear (vocab completeness)
        "M": lambda s: np.array([0.866 * s[0] - 0.5 * s[1], 0.5 * s[0] + 0.866 * s[1]], dtype=float),  # mild rotation mixer
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _nonlinear_order_operations() -> OperationMap:
    """Strongly non-commuting NONLINEAR operations: tanh-squashed versions of the
    ``order_rotation`` rotation/swap/shear maps. Non-commutation (hence order
    sensitivity, order-gap ~ 1 y-std) is inherited from the linear maps, while the
    ``tanh(2.5·)`` makes each composed protocol a genuinely NONLINEAR map of x (a
    linear model explains only ~40% of per-protocol variance). So a linear-composition
    surrogate is mis-specified and -- like every terminal-only model -- cannot identify
    the operation ORDER from the scalar readout alone without intermediate observation.
    Linear readout. Only A/B/C/M are used by the order scenarios; D for completeness."""
    g = 2.5
    return {
        "A": lambda s: np.tanh(g * np.array([-s[1], s[0]], dtype=float)),
        "B": lambda s: np.tanh(g * np.array([s[1], s[0]], dtype=float)),
        "C": lambda s: np.tanh(g * np.array([s[0], 1.2 * s[0] + s[1]], dtype=float)),
        "D": lambda s: np.tanh(g * np.array([s[0] - 0.7 * s[1], s[1]], dtype=float)),
        "M": lambda s: np.tanh(g * np.array([0.866 * s[0] - 0.5 * s[1], 0.5 * s[0] + 0.866 * s[1]], dtype=float)),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _shifted_operations() -> OperationMap:
    return {
        "A": lambda s: np.array([s[0] + 0.55 * s[1] + 0.05, s[1]], dtype=float),
        "B": lambda s: np.array([s[0], 0.45 * s[1] + 0.25], dtype=float),
        "C": lambda s: np.array([np.sin(1.5 * np.pi * s[0]) * 0.45 + 0.55, s[1] + 0.05], dtype=float),
        "D": lambda s: np.array([0.7 * s[0] * s[1] + 0.2, 0.6 * s[1] + 0.35 * s[0]], dtype=float),
        "M": lambda s: np.array([0.6 * s[0] + 0.4 * s[1], 1.2 * (s[0] - s[1])], dtype=float),
        "O": lambda s: np.array([s[0] + 0.05, np.cos(0.8 * np.pi * s[1])], dtype=float),
    }


def _unstable_operations() -> OperationMap:
    return {
        "A": lambda s: np.array([s[0] + 0.4 * s[1] + 0.08 * np.sin(20.0 * np.pi * s[0]), s[1]], dtype=float),
        "B": lambda s: np.array([s[0], 0.7 * s[1] + 0.12 + 0.05 * np.sin(18.0 * np.pi * s[1])], dtype=float),
        "C": lambda s: np.array([0.5 + 0.45 * np.sin(6.0 * np.pi * s[0]), s[1]], dtype=float),
        "D": lambda s: np.array([s[0] * s[1] + 0.1, 0.8 * s[1] + 0.2 * s[0]], dtype=float),
        "M": lambda s: np.array([(s[0] + s[1]) / 2.0, 3.0 * (s[0] - s[1])], dtype=float),
        "O": lambda s: np.array([s[0], np.cos(6.0 * np.pi * s[1])], dtype=float),
    }


def _lipschitz_operations() -> OperationMap:
    alpha = 1.35
    return {
        "A": lambda s: np.array([s[0] + 0.35 * s[1], s[1]], dtype=float),
        "B": lambda s: np.array([s[0], 0.65 * s[1] + 0.15], dtype=float),
        "C": lambda s: np.array([np.sin(np.pi * s[0]) * 0.5 + 0.5, s[1]], dtype=float),
        "D": lambda s: np.array([s[0] * s[1] + 0.1, 0.8 * s[1] + 0.2 * s[0]], dtype=float),
        "M": lambda s: alpha * np.asarray(s, dtype=float),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _misspecified_operations() -> OperationMap:
    return {
        "A": lambda s: np.array([s[0] + 0.25 * s[1] + 0.28 * np.sin(28.0 * np.pi * s[0]), s[1]], dtype=float),
        "B": lambda s: np.array([s[0], 0.7 * s[1] + 0.12 + 0.25 * np.sin(26.0 * np.pi * s[1])], dtype=float),
        "C": lambda s: np.array(
            [
                0.5 + 0.45 * np.sin(18.0 * np.pi * s[0]) + 0.12 * (s[0] > 0.55),
                s[1],
            ],
            dtype=float,
        ),
        "D": lambda s: np.array([s[0] * s[1] + 0.1, 0.8 * s[1] + 0.2 * s[0]], dtype=float),
        "M": lambda s: np.array([0.55 * s[0] + 0.45 * s[1], s[0] - 0.8 * s[1]], dtype=float),
        "O": lambda s: np.array([s[0], s[1]], dtype=float),
    }


def _three_body_operations() -> OperationMap:
    return {
        "A": lambda s: np.array([s[0] + 0.4, s[1]], dtype=float),
        "B": lambda s: np.array([s[0], s[1] + 0.4], dtype=float),
        "C": lambda s: np.array([0.65 * s[0] + 0.35, 0.65 * s[1] + 0.35], dtype=float),
        "D": lambda s: np.array([s[0] * s[1] + 0.1, 0.8 * s[1] + 0.2 * s[0]], dtype=float),
        "M": lambda s: np.array([(s[0] + s[1]) / 2.0, s[0] - s[1]], dtype=float),
        "O": lambda s: np.array([s[0], s[1]], dtype=float),
    }


def _dose_count_operations() -> OperationMap:
    """L1 dose-count semantics: the dose op ``B`` is the IDENTITY in state, so its
    only effect is the count-graded saturating readout term (see ``readout``). ``A``
    is a genuine nonlinear transform. Because ``B`` never moves the state, a
    per-operation model that conditions only on the state cannot represent the dose
    effect; the protocol needs a COUNT descriptor. ``C``/``D``/``M`` are provided
    for vocabulary completeness -- the dose_count scenario protocols use only A/B."""

    return {
        "A": lambda s: np.array(
            [_sigmoid(1.3 * s[0] - 0.4 * s[1] + 0.1),
             _sigmoid(0.8 * s[1] + 0.6 * np.sin(2.5 * s[0]))],
            dtype=float,
        ),
        "B": lambda s: np.asarray(s, dtype=float),  # dose / mix cycle: identity in state
        "C": lambda s: np.array([0.65 * s[0] + 0.35, 0.65 * s[1] + 0.35], dtype=float),
        "D": lambda s: np.array([s[0] * s[1] + 0.1, 0.8 * s[1] + 0.2 * s[0]], dtype=float),
        "M": lambda s: np.array([(s[0] + s[1]) / 2.0, s[0] - s[1]], dtype=float),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _bo_smooth_operations() -> OperationMap:
    return {
        "A": lambda s: np.array(
            [
                0.62 * s[0] + 0.32 * _branin_feature(s) + 0.08 * s[1],
                0.72 * s[1] + 0.24 * _hartmann3_feature(s),
            ],
            dtype=float,
        ),
        "B": lambda s: np.array(
            [
                0.74 * s[0] + 0.22 * _goldstein_price_feature(s),
                0.58 * s[1] + 0.34 * _branin_feature(np.array([s[1], s[0]], dtype=float)) + 0.04,
            ],
            dtype=float,
        ),
        "C": lambda s: np.array(
            [
                0.55 * _branin_feature(s) + 0.35 * _hartmann3_feature(s) + 0.1,
                0.65 * s[1] + 0.25 * _goldstein_price_feature(s),
            ],
            dtype=float,
        ),
        "D": lambda s: np.array(
            [
                0.45 * _branin_feature(s) + 0.35 * _goldstein_price_feature(s) + 0.15 * s[0] * s[1],
                0.55 * _hartmann3_feature(s) + 0.30 * s[1] + 0.08,
            ],
            dtype=float,
        ),
        "M": lambda s: np.array(
            [
                0.58 * s[0] + 0.35 * s[1] + 0.15 * _branin_feature(s),
                0.36 * s[0] - 0.44 * s[1] + 0.28 * _goldstein_price_feature(s),
            ],
            dtype=float,
        ),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _bo_shifted_operations(family: str) -> OperationMap:
    base = _bo_operation_multimodal_operations() if family == "bo_operation_multimodal" else _bo_smooth_operations()
    return {
        "A": lambda s: base["A"](np.array([0.85 * s[0] + 0.10, 0.9 * s[1] + 0.05], dtype=float)) + np.array([0.08, -0.04]),
        "B": lambda s: base["B"](np.array([0.9 * s[0] + 0.05, 0.8 * s[1] + 0.12], dtype=float)) + np.array([-0.05, 0.10]),
        "C": lambda s: base["C"](np.array([0.8 * s[0] + 0.12, 0.85 * s[1] + 0.08], dtype=float)) + np.array([0.04, 0.05]),
        "D": lambda s: base["D"](np.array([0.75 * s[0] + 0.15, 0.82 * s[1] + 0.08], dtype=float)),
        "M": lambda s: np.array([0.72 * s[0] + 0.42 * s[1] + 0.08, 0.45 * s[0] - 0.35 * s[1] - 0.04], dtype=float),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _bo_lipschitz_operations(family: str) -> OperationMap:
    base = _bo_operation_multimodal_operations() if family == "bo_operation_multimodal" else _bo_smooth_operations()
    alpha = 1.42
    return {
        "A": base["A"],
        "B": base["B"],
        "C": base["C"],
        "D": base["D"],
        "M": lambda s: alpha * np.asarray(s, dtype=float) + 0.10 * np.array([_branin_feature(s), _hartmann3_feature(s)], dtype=float),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _bo_misspecified_operations(family: str) -> OperationMap:
    base = _bo_operation_multimodal_operations() if family == "bo_operation_multimodal" else _bo_smooth_operations()
    return {
        "A": lambda s: base["A"](s) + np.array([0.18 * np.sin(22.0 * np.pi * s[0]), 0.0], dtype=float),
        "B": lambda s: base["B"](s) + np.array([0.0, 0.18 * np.sin(20.0 * np.pi * s[1])], dtype=float),
        "C": lambda s: base["C"](s) + np.array([0.14 * np.sin(18.0 * np.pi * (s[0] + 0.15 * s[1])), 0.10 * (s[1] > 0.55)], dtype=float),
        "D": base["D"],
        "M": lambda s: base["M"](s) + 0.12 * np.array(
            [
                np.sin(16.0 * np.pi * (s[0] + s[1])),
                np.sin(14.0 * np.pi * (s[0] - s[1])),
            ],
            dtype=float,
        ),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _bo_three_body_operations(family: str) -> OperationMap:
    if family in ("bo_operation_multimodal", "bo_all_multimodal"):
        base = _bo_operation_multimodal_operations()
    else:
        base = _bo_smooth_operations()
    return {
        "A": base["A"],
        "B": base["B"],
        "C": base["C"],
        "D": base["D"],
        "M": base["M"],
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _bo_operation_multimodal_operations() -> OperationMap:
    return {
        "A": lambda s: np.array(
            [
                0.48 * s[0] + 0.38 * _rastrigin_feature(s) + 0.12 * _ackley_feature(s),
                0.70 * s[1] + 0.22 * np.sin(6.0 * np.pi * s[0]),
            ],
            dtype=float,
        ),
        "B": lambda s: np.array(
            [
                0.68 * s[0] + 0.22 * np.cos(5.0 * np.pi * s[1]),
                0.42 * s[1] + 0.42 * _ackley_feature(np.array([s[1], s[0]], dtype=float)),
            ],
            dtype=float,
        ),
        "C": lambda s: np.array(
            [
                0.45 * _rastrigin_feature(s) + 0.32 * _levy_feature(s),
                0.35 * _ackley_feature(s) + 0.35 * np.sin(4.0 * np.pi * (s[0] + s[1])),
            ],
            dtype=float,
        ),
        "D": lambda s: np.array(
            [
                0.4 * _ackley_feature(s) + 0.35 * _rastrigin_feature(s) + 0.1 * s[0] * s[1],
                0.4 * _levy_feature(s) + 0.25 * np.cos(5.0 * np.pi * s[1]),
            ],
            dtype=float,
        ),
        "M": lambda s: np.array(
            [
                0.55 * s[0] + 0.28 * s[1] + 0.20 * _rastrigin_feature(s),
                0.34 * s[0] - 0.42 * s[1] + 0.22 * _ackley_feature(s),
            ],
            dtype=float,
        ),
        "O": lambda s: np.asarray(s, dtype=float),
    }


def _bo_shifted_readout(state: np.ndarray, family: str) -> float:
    shifted = np.array([0.85 * state[0] + 0.12, 0.80 * state[1] - 0.08], dtype=float)
    if family == "bo_multimodal_readout":
        return float(0.9 * _ackley_feature(shifted) - 0.55 * _rastrigin_feature(shifted) + 0.35 * _levy_feature(shifted))
    if family == "bo_operation_multimodal":
        return float(0.75 * _goldstein_price_feature(shifted) + 0.55 * _branin_feature(shifted) - 0.25 * shifted[1])
    return float(0.8 * _branin_feature(shifted) - 0.65 * _hartmann3_feature(shifted) + 0.35 * _goldstein_price_feature(shifted))


def _branin_feature(state: np.ndarray) -> float:
    x = 15.0 * state[0] - 5.0
    y = 15.0 * state[1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    value = a * (y - b * x**2 + c * x - r) ** 2 + s * (1.0 - t) * np.cos(x) + s
    return float(1.0 - 2.0 * _sigmoid((value - 45.0) / 25.0))


def _hartmann3_feature(state: np.ndarray) -> float:
    x = np.array([_clip01(state[0]), _clip01(state[1]), _clip01(0.35 + 0.3 * state[0] - 0.2 * state[1])], dtype=float)
    alpha = np.array([1.0, 1.2, 3.0, 3.2], dtype=float)
    a = np.array(
        [
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
        ],
        dtype=float,
    )
    p = 1e-4 * np.array(
        [
            [3689.0, 1170.0, 2673.0],
            [4699.0, 4387.0, 7470.0],
            [1091.0, 8732.0, 5547.0],
            [381.0, 5743.0, 8828.0],
        ],
        dtype=float,
    )
    value = -float(np.sum(alpha * np.exp(-np.sum(a * (x - p) ** 2, axis=1))))
    return float(_scaled_sigmoid(-0.8 * value - 1.0))


def _goldstein_price_feature(state: np.ndarray) -> float:
    x = 4.0 * _clip01(state[0]) - 2.0
    y = 4.0 * _clip01(state[1]) - 2.0
    term1 = 1.0 + (x + y + 1.0) ** 2 * (
        19.0 - 14.0 * x + 3.0 * x**2 - 14.0 * y + 6.0 * x * y + 3.0 * y**2
    )
    term2 = 30.0 + (2.0 * x - 3.0 * y) ** 2 * (
        18.0 - 32.0 * x + 12.0 * x**2 + 48.0 * y - 36.0 * x * y + 27.0 * y**2
    )
    value = np.log(term1 * term2)
    return float(1.0 - 2.0 * _sigmoid((value - 8.0) / 2.5))


def _ackley_feature(state: np.ndarray) -> float:
    x = 8.0 * _clip01(state[0]) - 4.0
    y = 8.0 * _clip01(state[1]) - 4.0
    value = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(
        0.5 * (np.cos(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y))
    ) + np.e + 20.0
    return float(1.0 - 2.0 * _sigmoid((value - 8.0) / 3.5))


def _rastrigin_feature(state: np.ndarray) -> float:
    x = 8.0 * _clip01(state[0]) - 4.0
    y = 8.0 * _clip01(state[1]) - 4.0
    value = 20.0 + x**2 - 10.0 * np.cos(2.0 * np.pi * x) + y**2 - 10.0 * np.cos(2.0 * np.pi * y)
    return float(1.0 - 2.0 * _sigmoid((value - 30.0) / 12.0))


def _levy_feature(state: np.ndarray) -> float:
    x = 10.0 * _clip01(state[0]) - 5.0
    y = 10.0 * _clip01(state[1]) - 5.0
    w1 = 1.0 + (x - 1.0) / 4.0
    w2 = 1.0 + (y - 1.0) / 4.0
    value = (
        np.sin(np.pi * w1) ** 2
        + (w1 - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w1 + 1.0) ** 2)
        + (w2 - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w2) ** 2)
    )
    return float(1.0 - 2.0 * _sigmoid((value - 4.0) / 2.5))


def _contains_three_body(steps: Sequence[str]) -> bool:
    return {"A", "B", "C"}.issubset(set(steps))


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _scaled_sigmoid(value: float) -> float:
    return float(2.0 * _sigmoid(value) - 1.0)
