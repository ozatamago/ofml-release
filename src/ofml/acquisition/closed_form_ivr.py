"""Closed-form (fantasy-free) IVR / IVR-Jacobian acquisition for OFML-GPN.

For a fixed-hyperparameter GP, adding one observation is an EXACT rank-1 Cohn
variance reduction (label-independent), so the whole fantasy deepcopy+refit is
replaced by matrix updates. This ports the gp_network operation-level closed-form
IVR to the ofml OFMLModel (linear ``steps`` composition):

* operation_inputs(model, rows): the per-operation input points each row induces.
* operation_sensitivities(model, rows): finite-difference Jacobian of the final
  scalar output w.r.t. each operation's output (for Jacobian variance-attribution).
* closed_form_ivr_sampling(...): greedy batch selection scoring each candidate by
  sum over ops/coords/refs of weight * cov_n(e,c)^2/(var_n(c)+noise), with a
  rank-1 posterior downdate after each pick (diverse batch).

Only operates on operation models exposing a per-coordinate closed-form GP
(ClosedFormGPOperationModel.gps_ -> _ClosedFormGP with posterior_cov/noise_orig),
i.e. OFML-GPN. Neural operation models are skipped (fall back to fantasy).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from ofml.benchmarks.synthetic_protocol_transfer.splits import Row

from .base import AcquisitionResult


def _initial_states(rows: Sequence[Row], state_dim: int) -> np.ndarray:
    base = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
    if state_dim == 2:
        return base
    if state_dim < 2:
        return base[:, :state_dim]
    return np.concatenate([base, np.zeros((len(rows), state_dim - 2))], axis=1)


def _closed_form_gp(operation_model):
    """Return the per-coordinate closed-form GP list, or None if not GP-backed."""
    return getattr(operation_model, "gps_", None)


def _augment(model, state: np.ndarray, steps: str) -> np.ndarray:
    """Apply the OFMLModel's per-protocol context-augmentation hook (identity for
    context-free models). The recorded operation-input points must live in the
    SAME augmented space the shared GP was fit in (so posterior_cov / ALC over
    refs and candidates are consistent), hence we record the augmented state."""

    fn = getattr(model, "_augment_input", None)
    return fn(state, steps) if fn is not None else state


def operation_inputs(model, rows: Sequence[Row]) -> dict[str, np.ndarray]:
    """Per operation type, the input points ``rows`` induce (mean composition)."""

    recorded: dict[str, list[np.ndarray]] = defaultdict(list)
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[str(row["steps"])].append(row)
    for steps, grp in groups.items():
        state = _initial_states(grp, model.initial_state_dim)
        for operation in steps.split():
            aug = _augment(model, state, steps)
            recorded[operation].append(np.atleast_2d(aug).copy())
            mean, _ = model._operation_model(operation).predict_distribution(aug)
            state = np.atleast_2d(np.asarray(mean, dtype=float))
    return {t: np.concatenate(v, axis=0) for t, v in recorded.items()}


def operation_sensitivities(model, rows: Sequence[Row], eps: float = 1e-3) -> dict[str, list[dict]]:
    """Per operation instance: input points + d(final)/d(op output coord) (finite diff)."""

    result: dict[str, list[dict]] = defaultdict(list)
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[str(row["steps"])].append(row)

    for steps, grp in groups.items():
        ops = steps.split()

        def compose(perturb_index=None, perturb_coord=None, delta=0.0):
            state = _initial_states(grp, model.initial_state_dim)
            recorded_inputs = []
            for idx, operation in enumerate(ops):
                aug = _augment(model, state, steps)
                recorded_inputs.append(np.atleast_2d(aug).copy())
                mean, _ = model._operation_model(operation).predict_distribution(aug)
                mean = np.atleast_2d(np.asarray(mean, dtype=float)).copy()
                if idx == perturb_index:
                    mean[:, perturb_coord] = mean[:, perturb_coord] + delta
                state = mean
            return state[:, 0], recorded_inputs  # final scalar [M], inputs per op

        _, inputs_per_op = compose()
        # recompute output dims robustly from a base pass (raw op outputs)
        outdims = []
        s = _initial_states(grp, model.initial_state_dim)
        for operation in ops:
            mean, _ = model._operation_model(operation).predict_distribution(_augment(model, s, steps))
            mean = np.atleast_2d(np.asarray(mean, dtype=float))
            outdims.append(mean.shape[1])
            s = mean

        # CENTERED finite difference (cx_bayes stabilization): (f(+h)-f(-h))/(2h)
        # is O(h^2)-accurate vs the one-sided O(h) bias, which on a multimodal
        # (highly curved) composition was the dominant source of noisy sensitivity
        # weights and per-round ranking instability.
        for idx, operation in enumerate(ops):
            d = outdims[idx]
            partials = np.zeros((len(grp), d))
            for j in range(d):
                fp, _ = compose(idx, j, +eps)
                fm, _ = compose(idx, j, -eps)
                partials[:, j] = (fp - fm) / (2.0 * eps)
            result[operation].append({"inputs": inputs_per_op[idx], "partials": partials})
    return result


def operation_sensitivities_smc(model, rows: Sequence[Row], n_samples: int = 32,
                                eps: float = 1e-3, seed: int = 0) -> dict[str, list[dict]]:
    """SMC-averaged Jacobian attribution: the composed-output sensitivity to each
    operation output is estimated by CENTERED finite difference along S SAMPLED
    pushforward paths (with common-random-numbers + antithetic draws) and averaged
    as RMS over samples, instead of a single mean-composition linearization.

    On a multimodal composition, different sampled paths land in different basins
    with different local sensitivities; the mean-path Jacobian misses this, so its
    variance-reduction attribution is misdirected. The MC average over paths gives
    the *expected* squared sensitivity across the actual (multimodal) landscape.
    Reference points are the mean-composition op inputs (unchanged); only the
    weight is refined. ``partials`` returned as RMS-over-samples so that
    ``partials**2`` (used by build_reference_jacobian) == E_s[sensitivity^2]."""

    result: dict[str, list[dict]] = defaultdict(list)
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[str(row["steps"])].append(row)

    for steps, grp in groups.items():
        ops = steps.split()
        n = len(grp)
        s_n = n_samples + (n_samples % 2)  # even for antithetic
        state0 = _initial_states(grp, model.initial_state_dim)

        # mean pass: reference op-input points + output dims per op
        inputs_per_op, outdims = [], []
        s = state0
        for operation in ops:
            aug = _augment(model, s, steps)
            inputs_per_op.append(np.atleast_2d(aug).copy())
            mean, _ = model._operation_model(operation).predict_distribution(aug)
            mean = np.atleast_2d(np.asarray(mean, dtype=float))
            outdims.append(mean.shape[1])
            s = mean

        # common random numbers: fixed antithetic standard-normal per op index
        rng = np.random.default_rng(seed)
        zs = [np.concatenate([h := rng.standard_normal((s_n // 2, n, outdims[i])), -h], axis=0)
              for i in range(len(ops))]

        def sampled_compose(pidx=None, pcoord=None, delta=0.0):
            states = np.repeat(state0[None, :, :], s_n, axis=0)  # [S, N, d0]
            for idx, operation in enumerate(ops):
                flat = _augment(model, states.reshape(s_n * n, -1), steps)
                mean, var = model._operation_model(operation).predict_distribution(flat)
                dout = np.asarray(mean).shape[1]
                mean = np.asarray(mean, dtype=float).reshape(s_n, n, dout)
                std = np.sqrt(np.maximum(np.asarray(var, dtype=float), 0.0)).reshape(s_n, n, dout)
                out = mean + std * zs[idx]
                if idx == pidx:
                    out[:, :, pcoord] = out[:, :, pcoord] + delta
                states = out
            return states[:, :, 0]  # [S, N] final scalar

        for idx, operation in enumerate(ops):
            d = outdims[idx]
            partials = np.zeros((n, d))
            for j in range(d):
                sens = (sampled_compose(idx, j, +eps) - sampled_compose(idx, j, -eps)) / (2.0 * eps)
                partials[:, j] = np.sqrt(np.mean(sens ** 2, axis=0))  # RMS over sampled paths
            result[operation].append({"inputs": inputs_per_op[idx], "partials": partials})
    return result


def build_reference_jacobian(model, evaluation_rows: Sequence[Row],
                             smc_samples: int = 0, seed: int = 0) -> dict[str, dict]:
    sens = (operation_sensitivities_smc(model, evaluation_rows, n_samples=smc_samples, seed=seed)
            if smc_samples > 0 else operation_sensitivities(model, evaluation_rows))
    pts: dict[str, list[np.ndarray]] = defaultdict(list)
    wts: dict[str, list[np.ndarray]] = defaultdict(list)
    for op_type, instances in sens.items():
        for inst in instances:
            pts[op_type].append(inst["inputs"])
            wts[op_type].append(inst["partials"] ** 2)
    return {
        t: {"points": np.concatenate(pts[t], axis=0), "weights": np.concatenate(wts[t], axis=0)}
        for t in pts
    }


def closed_form_ivr_sampling(
    row_model,
    candidates: Sequence[Row],
    evaluation_rows: Sequence[Row],
    n: int,
    jacobian: bool = True,
    smc_samples: int = 0,
    seed: int = 0,
) -> AcquisitionResult:
    """Greedy closed-form IVR / IVR-Jacobian selection over a candidate pool.

    ``smc_samples>0`` uses the SMC-averaged (per-sampled-path) Jacobian attribution
    instead of the single mean-composition linearization."""

    if not candidates:
        raise ValueError("candidates must not be empty.")
    model = getattr(row_model, "model", None)
    if model is None:
        raise ValueError("closed-form IVR needs a fitted OFML row model (row_model.model).")

    refs_jac = build_reference_jacobian(model, evaluation_rows, smc_samples=smc_samples, seed=seed)
    # keep only closed-form (GP-backed) operation types with references
    op_types = [t for t in refs_jac if _closed_form_gp(model.operation_models.get(t)) is not None]
    if not op_types:
        raise ValueError("closed-form IVR requires GP-backed operation models (OFML-GPN).")

    cand_inputs = [operation_inputs(model, [c]) for c in candidates]

    # Per (op type, output coord): posterior covariance over [refs ; candidate pts].
    state: dict[tuple[str, int], dict] = {}
    for op_type in op_types:
        gps = _closed_form_gp(model.operation_models[op_type])
        refs = np.atleast_2d(refs_jac[op_type]["points"])
        weights = np.atleast_2d(refs_jac[op_type]["weights"])
        blocks, slices, cursor = [], [], refs.shape[0]
        for ci in range(len(candidates)):
            pts = cand_inputs[ci].get(op_type)
            if pts is None or len(pts) == 0:
                slices.append([])
                continue
            pts = np.atleast_2d(pts)
            blocks.append(pts)
            slices.append(list(range(cursor, cursor + pts.shape[0])))
            cursor += pts.shape[0]
        points = np.vstack([refs, *blocks]) if blocks else refs
        ref_idx = np.arange(refs.shape[0])
        for j, gp in enumerate(gps):
            w = weights[:, j] if jacobian else np.ones(refs.shape[0])
            state[(op_type, j)] = {
                "M": gp.posterior_cov(points, points),
                "ref": ref_idx,
                "w": w,
                "slices": slices,
                "noise": gp.noise_orig,
            }

    def score(ci: int) -> float:
        total = 0.0
        for st in state.values():
            for u in st["slices"][ci]:
                var_u = max(float(st["M"][u, u]), 0.0)  # clip negative posterior var
                denom = var_u + st["noise"]
                if denom > 1e-12:  # floor (cx_bayes): avoid tiny-denominator score blow-up
                    total += float(np.sum(st["w"] * (st["M"][st["ref"], u] ** 2))) / denom
        return total

    chosen, chosen_scores, remaining = [], [], list(range(len(candidates)))
    for _ in range(min(n, len(candidates))):
        best_c, best = None, -np.inf
        for ci in remaining:
            sc = score(ci)
            if sc > best:
                best, best_c = sc, ci
        if best_c is None:
            break
        chosen.append(best_c)
        chosen_scores.append(best)
        remaining.remove(best_c)
        for st in state.values():  # rank-1 posterior downdate by the picked candidate
            for u in st["slices"][best_c]:
                var_u = max(float(st["M"][u, u]), 0.0)
                denom = var_u + st["noise"]
                if denom > 1e-12:
                    col = st["M"][:, u].copy()
                    st["M"] = st["M"] - np.outer(col, col) / denom
                    # PSD repair (cx_bayes): restore symmetry + clamp diagonal >=0 so
                    # float asymmetry / negative variances do not compound over picks.
                    st["M"] = 0.5 * (st["M"] + st["M"].T)
                    np.fill_diagonal(st["M"], np.clip(np.diagonal(st["M"]), 0.0, None))

    return AcquisitionResult(
        selected_indices=tuple(chosen),
        selected_rows=tuple(candidates[i] for i in chosen),
        scores=tuple(chosen_scores),
    )
