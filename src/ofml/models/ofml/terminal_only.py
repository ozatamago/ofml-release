"""Terminal-observation-only OFML (the paper's target setting).

Only the shared terminal observer O is measured: each protocol run yields a single
scalar y. Intermediate operation states are NEVER observed. The compositional
surrogate is therefore trained END-TO-END on the terminal scalar alone, with the
per-operation models as *latent* shared modules composed along each protocol's
step sequence and supervised only at the output.

Contrast with the trace-based OFML (``OFMLModel`` + operation-level examples),
which fits each operation model separately on observed intermediate input->output
pairs. That trace assumption is the *future-work* relaxation; this module is the
terminal-only target.

Design:
* Latent intermediate representation of dimension ``latent_dim`` (the headroom):
  prior experiments found ~2-2.5x the true state dimension works well; default 5
  (=2.5x the 2-D design input). The extra dimensions give the composition room to
  carry hidden protocol-level interactions forward that a minimal state cannot.
* Non-residual operations ``s <- f_op(s)`` (general module map, per the generalized
  OFML setting): each operation is a full learned map; an operation that never appears
  in training maps via its prior/initialisation (it is *not* forced to the identity).
* Row-level interface: fit(rows)/predict_distribution(rows) on (x1,x2,steps,y) only
  -- no operation_examples / trace.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .operation_models import _OperationMLP, _set_torch_seed

INTERMEDIATE_OPS = ("A", "B", "C", "D", "M")
OBSERVER = "O"


@dataclasses.dataclass(frozen=True)
class TerminalOnlyConfig:
    latent_dim: int = 5            # headroom ~2.5x the true 2-D state
    hidden_dim: int = 32
    n_hidden: int = 2              # number of hidden layers per op-module
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    max_epochs: int = 1500          # upper cap on epochs
    dropout: float = 0.0
    seed: int = 0
    early_stopping: bool = False    # hold out an internal val split of the TRAINING rows
    es_val_fraction: float = 0.2    # fraction of training rows used for early-stopping val
    es_patience: int = 40           # stop after this many checks without val improvement
    es_check_every: int = 10        # evaluate val loss every N epochs
    es_min_epochs: int = 50         # always train at least this many epochs
    trace_weight: float = 1.0       # weight of the auxiliary intermediate-observation loss
    trace_warmup_epochs: int = 0    # ramp trace_weight linearly 0->target over these epochs (0 = off)
    grad_clip: float = 0.0          # max grad-norm (0 = off); stabilizes trace-augmented training
    bounded_ops: bool = False       # tanh the latent after each op -> bounded state (prevents C^k iteration blow-up)
    direct_trace: bool = False      # supervise the latent DIRECTLY (no align); requires latent_dim == 2 (fixes the gauge)
    per_op_align: bool = False      # a SEPARATE observer (align) per operation type: O_A,O_B,O_C,O_M (weights regularized like the rest)


class _ComposedNet(torch.nn.Module):
    """Shared latent operation modules composed along a protocol's step string."""

    def __init__(self, cfg: TerminalOnlyConfig) -> None:
        super().__init__()
        d = cfg.latent_dim
        self.embed = torch.nn.Linear(2, d)
        self.ops = torch.nn.ModuleDict(
            {op: _OperationMLP(d, d, cfg.hidden_dim, cfg.dropout, cfg.n_hidden) for op in INTERMEDIATE_OPS}
        )
        self.obs = _OperationMLP(d, 1, cfg.hidden_dim, cfg.dropout, cfg.n_hidden)
        # linear alignment latent(d) -> observed 2-D state, used ONLY for the optional
        # intermediate-observation (trace) supervision; unused for terminal prediction.
        self.bounded_ops = cfg.bounded_ops
        self.direct_trace = cfg.direct_trace
        self.per_op_align = cfg.per_op_align
        if cfg.per_op_align:   # one observer per operation type (O_A,O_B,O_C,O_M,+D)
            self.aligns = torch.nn.ModuleDict({op: torch.nn.Linear(d, 2) for op in INTERMEDIATE_OPS})
        else:
            self.align = torch.nn.Linear(d, 2)
        # NON-residual composition (general module map h(s), per the generalized OFML
        # setting): no zero-init identity default; an unseen op maps via its prior/init.

    def forward_protocol(self, x: torch.Tensor, steps: Sequence[str]) -> torch.Tensor:
        s = self.embed(x)
        for op in steps[:-1]:  # intermediate ops; last step is the observer
            if op in self.ops:
                s = self.ops[op](s)  # non-residual: s <- f_op(s)
                if self.bounded_ops:
                    s = torch.tanh(s)  # bound the latent -> iteration cannot blow up
        return self.obs(s)[:, 0]  # [N] terminal scalar

    def forward_protocol_trace(self, x: torch.Tensor, steps: Sequence[str], obs_pos):
        """Terminal scalar + {position t: aligned latent} at observed intermediate
        positions ``obs_pos`` (indices into steps[:-1])."""
        obs_set = set(obs_pos)
        s = self.embed(x)
        captured = {}
        for t, op in enumerate(steps[:-1]):
            if op in self.ops:
                s = self.ops[op](s)
                if self.bounded_ops:
                    s = torch.tanh(s)
            if t in obs_set:
                if self.direct_trace:
                    captured[t] = s                    # no align: compare latent directly (needs d==2)
                elif self.per_op_align:
                    captured[t] = self.aligns[op](s)    # per-operation observer O_op
                else:
                    captured[t] = self.align(s)         # shared observer
        return self.obs(s)[:, 0], captured


def _group(rows: Sequence[Mapping[str, str]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[str(r["steps"])].append(i)
    return groups


@dataclasses.dataclass
class TerminalOnlyOFML:
    """End-to-end terminal-only compositional surrogate.

    ``n_members>1`` -> deep ensemble (predictive variance = across-member spread).
    ``mcd_samples>0`` (with dropout>0) -> MC-dropout predictive variance.
    Otherwise a point model (variance 0).
    """

    config: TerminalOnlyConfig = dataclasses.field(default_factory=TerminalOnlyConfig)
    n_members: int = 1
    mcd_samples: int = 0
    nets_: list | None = None
    y_mean_: float = 0.0
    y_std_: float = 1.0
    trace_mean_: object = None
    trace_std_: object = None

    def fit(self, rows: Sequence[Mapping[str, str]], trace=None) -> "TerminalOnlyOFML":
        import copy
        cfg = self.config
        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        y = np.array([float(r["y"]) for r in rows], dtype=float)
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        ys = (y - self.y_mean_) / self.y_std_
        n = len(rows)

        # optional intermediate-observation (trace) targets: trace[i] = list of
        # (position_t, np.array([2])) for observed intermediate steps of row i.
        use_trace = trace is not None and any(len(trace[i]) > 0 for i in range(n))
        if use_trace:
            allt = np.array([v for i in range(n) for (_, v) in trace[i]], dtype=float)
            self.trace_mean_ = allt.mean(0)
            self.trace_std_ = np.where(allt.std(0) < 1e-6, 1.0, allt.std(0))
        else:
            self.trace_mean_, self.trace_std_ = np.zeros(2), np.ones(2)

        # internal train/val row split for early stopping (from the TRAINING rows only;
        # disjoint from any outer/target validation used to score the model)
        es = cfg.early_stopping and n >= 20
        if es:
            rng = np.random.default_rng(cfg.seed)
            perm = rng.permutation(n)
            n_val = max(4, int(round(cfg.es_val_fraction * n)))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
        else:
            tr_idx = np.arange(n); val_idx = np.array([], dtype=int)

        def make_batches(idx_set):
            g = defaultdict(list)
            for i in idx_set:
                g[str(rows[i]["steps"])].append(i)
            out = []
            for steps, idx in g.items():
                xb = torch.as_tensor(x[idx], dtype=torch.float32)
                yb = torch.as_tensor(ys[idx], dtype=torch.float32)
                obs_pos, tb = [], None
                if use_trace and len(trace[idx[0]]) > 0:
                    obs_pos = [t for (t, _) in trace[idx[0]]]
                    tb = {}
                    for pos_k, t in enumerate(obs_pos):
                        tgt = np.array([trace[i][pos_k][1] for i in idx], dtype=float)  # [n,2]
                        tgt = (tgt - self.trace_mean_) / self.trace_std_
                        tb[t] = torch.as_tensor(tgt, dtype=torch.float32)
                out.append((steps.split(), xb, yb, len(idx), obs_pos, tb))
            return out

        tr_batches = make_batches(tr_idx); tr_total = float(len(tr_idx))
        val_batches = make_batches(val_idx) if es else []
        val_total = float(len(val_idx)) if es else 1.0

        def eval_loss(net, bs, tot):
            net.eval()
            with torch.no_grad():
                l = 0.0
                for steps, xb, yb, _, _, _ in bs:
                    l += float(torch.sum((net.forward_protocol(xb, steps) - yb) ** 2))
            net.train()
            return l / tot

        self.nets_ = []
        for member in range(max(1, self.n_members)):
            _set_torch_seed(cfg.seed + member)
            net = _ComposedNet(cfg)
            opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            net.train()
            best_val, best_state, since = float("inf"), None, 0
            for ep in range(cfg.max_epochs):
                opt.zero_grad()
                loss = torch.zeros(())
                trace_loss = torch.zeros(())
                for steps, xb, yb, _, obs_pos, tb in tr_batches:
                    if tb:
                        pred, captured = net.forward_protocol_trace(xb, steps, obs_pos)
                        loss = loss + torch.nn.functional.mse_loss(pred, yb, reduction="sum")
                        for t in obs_pos:
                            trace_loss = trace_loss + torch.nn.functional.mse_loss(captured[t], tb[t], reduction="sum")
                    else:
                        loss = loss + torch.nn.functional.mse_loss(net.forward_protocol(xb, steps), yb, reduction="sum")
                tw = cfg.trace_weight
                if cfg.trace_warmup_epochs > 0:
                    tw = cfg.trace_weight * min(1.0, (ep + 1) / cfg.trace_warmup_epochs)
                ((loss + tw * trace_loss) / tr_total).backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                opt.step()
                if es and ep >= cfg.es_min_epochs and (ep % cfg.es_check_every == 0):
                    vl = eval_loss(net, val_batches, val_total)
                    if vl < best_val - 1e-5:
                        best_val, best_state, since = vl, copy.deepcopy(net.state_dict()), 0
                    else:
                        since += 1
                        if since >= cfg.es_patience:
                            break
            if es and best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.nets_.append(net)
        return self

    def predict_distribution(self, rows: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        if not self.nets_:
            raise ValueError("TerminalOnlyOFML must be fit before predict.")
        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        mean = np.zeros(len(rows)); var = np.zeros(len(rows))
        use_mcd = self.mcd_samples > 0 and self.config.dropout > 0
        for steps, idx in _group(rows).items():
            xb = torch.as_tensor(x[idx], dtype=torch.float32)
            preds = []
            if use_mcd:
                net = self.nets_[0]; net.train()  # dropout on
                with torch.no_grad():
                    for _ in range(self.mcd_samples):
                        preds.append(net.forward_protocol(xb, steps.split()).numpy())
                net.eval()
            else:
                with torch.no_grad():
                    for net in self.nets_:
                        preds.append(net.forward_protocol(xb, steps.split()).numpy())
            p = np.stack(preds, axis=0)  # [K, n]
            m = p.mean(0) * self.y_std_ + self.y_mean_
            v = p.var(0) * (self.y_std_ ** 2) if p.shape[0] > 1 else np.zeros(p.shape[1])
            for k, i in enumerate(idx):
                mean[i] = m[k]; var[i] = v[k]
        return mean, var

    def predict(self, rows: Sequence[Mapping[str, str]]) -> np.ndarray:
        return self.predict_distribution(rows)[0]
