"""Parameter counts for the Experiment 3 models (spec Sec. 5): analytic formulas, gradient counts
(param_match._grad_masks, ReLU -> LeakyReLU while counting) and matching to OFML-DAG-Ens."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .param_match import _grad_masks, mlp_params

OFML_LATENT, OFML_HIDDEN = 10, 32
N_UNARY, N_TYPES_CELL, N_TYPES_DAGNN = 4, 5, 6      # A-D ; A-D + M ; A-D + M + O


def ofml_dag_params(d: int = OFML_LATENT, w: int = OFML_HIDDEN, n_unary: int = N_UNARY) -> int:
    return 3 * d + n_unary * mlp_params(d, d, w) + mlp_params(2 * d, d, w) + mlp_params(d, 1, w)


def sharedcell_dag_params(k: int, w: int, d: int = OFML_LATENT, n_types: int = N_TYPES_CELL) -> int:
    return 3 * d + n_types * k + mlp_params(2 * d + k, d, w) + mlp_params(d, 1, w)


def dagnn_dag_params(n_layers: int, d: int, r: int, aggregate: str, n_types: int = N_TYPES_DAGNN) -> int:
    n = 3 * d + n_types * d + n_layers * (6 * d * d + 6 * d)
    n += n_layers * (2 * d if aggregate == "attention" else 2 * d * d + d)
    return n + (n_layers + 1) * d * r + r + r + 1


def match_dagnn_dag(target: int, n_layers: int, aggregate: str, tol: float = 0.10,
                    d_range=(8, 64), r_range=(8, 64)) -> dict:
    """Closest (d_h, r) within +-tol; ties -> most balanced |r - d_h|, then smaller d_h (as Exp. 1)."""
    best = None
    for d in range(d_range[0], d_range[1] + 1):
        for r in range(r_range[0], r_range[1] + 1):
            n = dagnn_dag_params(n_layers, d, r, aggregate)
            key = (abs(n - target), abs(r - d), d)
            if best is None or key < best[0]:
                best = (key, d, r, n)
    _, d, r, n = best
    rel = (n - target) / target
    if abs(rel) > tol:
        raise ValueError(f"no DAGNN ({aggregate}) config within {tol:.0%} of {target}")
    return dict(n_layers=n_layers, hidden_dim=d, readout_dim=r, aggregate=aggregate, n_params=n, rel_err=rel,
                target=target)


def match_sharedcell_dag(target: int, k: int, tol: float = 0.10, w_range=(4, 256)) -> dict:
    best = min(range(w_range[0], w_range[1] + 1), key=lambda w: abs(sharedcell_dag_params(k, w) - target))
    n = sharedcell_dag_params(k, best)
    rel = (n - target) / target
    if abs(rel) > tol:
        raise ValueError(f"no SharedCell-DAG config within {tol:.0%} of {target}")
    return dict(embed_dim=k, hidden_dim=best, n_params=n, rel_err=rel, target=target)


# --------------------------------------------------------------------------- gradient counting
def _std_y(rows):
    y = np.array([float(r["y"]) for r in rows])
    return torch.as_tensor((y - y.mean()) / (y.std() or 1.0), dtype=torch.float32)


def _groups(rows):
    from .dag_protocol import parse_protocol
    from .exp3_models import group_rows, x_matrix
    out = []
    for steps, idx in group_rows(rows).items():
        dag = parse_protocol(steps)
        out.append((idx, dag, torch.as_tensor(x_matrix(rows, idx, dag), dtype=torch.float32)))
    return out


def count_active(model: str, rows: Sequence[Mapping], cfg=None, seeds=(0, 1, 2)) -> dict:
    """Active (gradient-receiving) parameters of one member of `model` on these rows."""
    from . import exp3_models as E
    y, groups = _std_y(rows), _groups(rows)
    if model == "ofml_dag":
        cfg = cfg or E.DAGOFMLConfig()
        build = lambda: E._DAGComposedNet(cfg)                       # noqa: E731
        fwd = lambda net, x, dag: net.forward_dag(x, dag)              # noqa: E731
    elif model == "sharedcell_dag":
        build = lambda: E.SharedCellDAGNet(cfg)                      # noqa: E731
        fwd = lambda net, x, dag: net(x, dag)                          # noqa: E731
    elif model in ("dagnn", "dagnn_cat"):
        build = lambda: E.DAGNNDAGNet(cfg)                           # noqa: E731
        fwd = lambda net, x, dag: net(x, dag)                          # noqa: E731
    else:
        raise ValueError(model)

    def loss(net):
        l = 0.0
        for idx, dag, x in groups:
            l = l + ((fwd(net, x, dag) - y[idx]) ** 2).sum()
        return l / len(rows)
    return _grad_masks(build, loss, seeds)
