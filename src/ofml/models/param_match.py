"""Parameter counting and parameter matching for Experiment 1.

"Actually trained" (active) parameters = parameters that receive a non-zero loss gradient on the
data of the given depth. Unused operation modules, unused embedding rows, the unused trace
``align`` head of OFML, the chain-inactive DAGNN attention scorers (softmax over one parent == 1)
and input columns of always-zero features are excluded.

Two independent ways of counting are provided and cross-checked in the tests:
  * analytic formulas  (``*_params``)
  * gradient counting  (``count_active_*``): one backward pass of the training loss on real rows
    from several random initialisations; an element is active if its gradient is non-zero in ANY of
    them. During counting every ``torch.nn.ReLU`` is swapped for ``LeakyReLU(0.01)``: with ReLU a
    hidden unit that happens to be dead on all rows at a given initialisation gives exact-zero
    gradients for its whole fan-in/fan-out (observed: ~15% of OFML's weights at one init), which is a
    property of that random init, not of the architecture/data. Structurally unused parameters
    (unused op modules / embedding rows, constant-zero input features, chain-inactive attention)
    still get exactly zero (or None) gradients under LeakyReLU.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

OFML_LATENT, OFML_HIDDEN = 10, 32       # TerminalOnlyConfig(latent_dim=10, hidden_dim=32) of OFML-Ens


# --------------------------------------------------------------------------- analytic
def mlp_params(i: int, o: int, h: int, n_hidden: int = 2) -> int:
    return i * h + h + (max(1, n_hidden) - 1) * (h * h + h) + h * o + o


def ofml_params(n_ops: int, d: int = OFML_LATENT, h: int = OFML_HIDDEN) -> int:
    """Active params of TerminalOnlyOFML (one member): embed + used op MLPs + observer."""
    return 3 * d + n_ops * mlp_params(d, d, h) + mlp_params(d, 1, h)


def dagnn_params(n_types: int, n_layers: int, d_h: int, r: int, chain: bool = True) -> int:
    """Active params of one DAGNN member. n_types = #distinct node types used (ops + O)."""
    n = 3 * d_h                         # input projection W_in x + b
    n += n_types * d_h                  # used embedding rows
    n += n_layers * (6 * d_h * d_h + 6 * d_h)   # GRU cells (W_ih, W_hh, b_ih, b_hh)
    if not chain:
        n += n_layers * 2 * d_h         # attention scorers (only active with >1 parent)
    n += (n_layers + 1) * d_h * r + r + r + 1    # readout MLP
    return n


def dagnn_total_params(n_layers: int, d_h: int, r: int, vocab: int = 6) -> int:
    return 3 * d_h + vocab * d_h + n_layers * (6 * d_h * d_h + 8 * d_h) + (n_layers + 1) * d_h * r + 2 * r + 1


def sharedcell_params(n_ops: int, k: int, h: int, d: int = OFML_LATENT) -> int:
    """Active params of one SharedCell member: embed + used op embedding rows + shared cell + observer."""
    return 3 * d + n_ops * k + mlp_params(d + k, d, h) + mlp_params(d, 1, h)


def sharedcell_total_params(k: int, h: int, d: int = OFML_LATENT, vocab: int = 6) -> int:
    return 3 * d + vocab * k + mlp_params(d + k, d, h) + mlp_params(d, 1, h)


# --------------------------------------------------------------------------- helpers
def used_ops(rows: Sequence[Mapping]) -> list[str]:
    """Distinct intermediate operation types appearing in the rows."""
    s = set()
    for r in rows:
        s.update(str(r["steps"]).split()[:-1])
    return sorted(s)


def _grad_masks(build: Callable[[], torch.nn.Module], loss_fn: Callable[[torch.nn.Module], torch.Tensor],
                seeds: Sequence[int]) -> dict:
    nz_union, notnone, total = None, 0, 0
    for s in seeds:
        torch.manual_seed(s)
        net = _leaky(build())
        net.zero_grad(set_to_none=True)
        loss_fn(net).backward()
        masks = {}
        nn_count = 0
        for name, p in net.named_parameters():
            if p.grad is None:
                masks[name] = torch.zeros_like(p, dtype=torch.bool)
            else:
                masks[name] = p.grad != 0
                nn_count += p.numel()
        total = sum(p.numel() for p in net.parameters())
        notnone = nn_count
        nz_union = masks if nz_union is None else {k: nz_union[k] | masks[k] for k in masks}
    return dict(active=int(sum(int(m.sum()) for m in nz_union.values())),
                grad_not_none=int(notnone), total=int(total),
                per_param={k: int(v.sum()) for k, v in nz_union.items()})


def _leaky(net: torch.nn.Module) -> torch.nn.Module:
    for mod in list(net.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.ReLU):
                setattr(mod, name, torch.nn.LeakyReLU(0.01))
    return net


def _xy(rows):
    x = torch.as_tensor(np.array([[float(r["x1"]), float(r["x2"])] for r in rows]), dtype=torch.float32)
    y = np.array([float(r["y"]) for r in rows])
    y = torch.as_tensor((y - y.mean()) / (y.std() or 1.0), dtype=torch.float32)
    return x, y


# --------------------------------------------------------------------------- gradient counting
def count_active_ofml(rows: Sequence[Mapping], latent_dim: int = OFML_LATENT, hidden_dim: int = OFML_HIDDEN,
                      seeds: Sequence[int] = (0, 1, 2)) -> dict:
    from ofml.models.ofml.terminal_only import TerminalOnlyConfig, _ComposedNet
    cfg = TerminalOnlyConfig(latent_dim=latent_dim, hidden_dim=hidden_dim)
    x, y = _xy(rows)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[str(r["steps"])].append(i)

    def loss(net):
        l = 0.0
        for steps, idx in groups.items():
            l = l + ((net.forward_protocol(x[idx], steps.split()) - y[idx]) ** 2).sum()
        return l / len(rows)
    return _grad_masks(lambda: _ComposedNet(cfg), loss, seeds)


def count_active_dagnn(rows: Sequence[Mapping], cfg, seeds: Sequence[int] = (0, 1, 2)) -> dict:
    from ofml.models.dagnn import DAGNNNet, make_batches
    x, y = _xy(rows)
    batches = make_batches(rows, x.numpy().astype(float), cfg.batching, "cpu")

    def loss(net):
        l = 0.0
        for idx, xb, types, out, parents in batches:
            l = l + ((net(xb, types, out, parents=parents) - y[idx]) ** 2).sum()
        return l / len(rows)
    return _grad_masks(lambda: DAGNNNet(cfg), loss, seeds)


def count_active_sharedcell(rows: Sequence[Mapping], cfg, seeds: Sequence[int] = (0, 1, 2)) -> dict:
    from ofml.models.ofml.shared_cell import SharedCellNet, make_batches
    x, y = _xy(rows)
    batches = make_batches(rows, x.numpy().astype(float), cfg.batching, "cpu")

    def loss(net):
        l = 0.0
        for idx, args, _ in batches:
            l = l + ((net(*args) - y[idx]) ** 2).sum()
        return l / len(rows)
    return _grad_masks(lambda: SharedCellNet(cfg), loss, seeds)


def count_active_shared_mlp(rows: Sequence[Mapping], hidden_dim: int = 32, seeds: Sequence[int] = (0, 1, 2)) -> dict:
    """Shared-Ens member (_MLP on RowFeatureEncoder features)."""
    from ofml.models.features import RowFeatureEncoder
    from ofml.models.shared_mlp import _MLP
    enc = RowFeatureEncoder()
    X = torch.as_tensor(enc.fit_transform(rows), dtype=torch.float32)
    _, y = _xy(rows)
    return _grad_masks(lambda: _MLP(X.shape[1], hidden_dim, 0.0), lambda net: ((net(X) - y) ** 2).mean(), seeds)


def count_active_independent(rows: Sequence[Mapping], targets: Sequence[str], hidden_dim: int = 32,
                             seeds: Sequence[int] = (0, 1, 2)) -> dict:
    """Independent-Ens member: one _MLP(2->h->h->1) per target that has rows (summed over targets)."""
    from ofml.models.shared_mlp import _MLP
    out = dict(active=0, grad_not_none=0, total=0)
    for t in targets:
        tr = [r for r in rows if str(r["steps"]) == t]
        if not tr:
            continue
        x, y = _xy(tr)
        c = _grad_masks(lambda: _MLP(2, hidden_dim, 0.0), lambda net: ((net(x) - y) ** 2).mean(), seeds)
        for k in out:
            out[k] += c[k]
    return out


# --------------------------------------------------------------------------- matching
def ofml_target(rows: Sequence[Mapping]) -> dict:
    """OFML-Ens active params (one member) on these rows: gradient count + analytic formula."""
    g = count_active_ofml(rows)
    return dict(grad=g["active"], analytic=ofml_params(len(used_ops(rows))), grad_not_none=g["grad_not_none"],
                total=g["total"])


def match_dagnn(target: int, n_layers: int, n_types: int, tol: float = 0.10,
                d_range: tuple[int, int] = (8, 64), r_range: tuple[int, int] = (8, 64)) -> dict:
    """Closest (d_h, r) to the target within +-tol; ties -> most balanced (|r - d_h| smallest), then smaller d_h."""
    best = None
    for d in range(d_range[0], d_range[1] + 1):
        for r in range(r_range[0], r_range[1] + 1):
            n = dagnn_params(n_types, n_layers, d, r)
            key = (abs(n - target), abs(r - d), d)
            if best is None or key < best[0]:
                best = (key, d, r, n)
    _, d, r, n = best
    rel = (n - target) / target
    if abs(rel) > tol:
        raise ValueError(f"no DAGNN config within {tol:.0%} of {target}")
    return dict(n_layers=n_layers, hidden_dim=d, readout_dim=r, n_params=n, rel_err=rel, target=target)


def match_sharedcell(target: int, k: int, n_ops: int, tol: float = 0.10, h_range: tuple[int, int] = (4, 256)) -> dict:
    best = None
    for h in range(h_range[0], h_range[1] + 1):
        n = sharedcell_params(n_ops, k, h)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (h, n)
    h, n = best
    rel = (n - target) / target
    if abs(rel) > tol:
        raise ValueError(f"no SharedCell config within {tol:.0%} of {target}")
    return dict(embed_dim=k, hidden_dim=h, n_params=n, rel_err=rel, target=target)
