"""Linear-composition terminal-only OFML (OFML-Linear).

Each operation is a LEARNED LINEAR map in a latent space; the observer is linear.
A protocol is therefore a composition of linear maps -> a single linear map of the
design input x, matching a linearly-composed truth (e.g. the ``order_rotation``
benchmark: strongly non-commuting rotation/swap/shear ops with a linear readout).

Why this model exists. With a single scalar terminal readout the ops are observed
only through 1-D projections ``r . (product)`` of the composed state. For a general
nonlinear surrogate this makes the ORDER of non-commuting ops effectively
under-determined (nonlinear OFML fails to extrapolate held-out orders). But for
LINEAR ops at MATCHED latent capacity the system is OVER-DETERMINED (#observations
> #effective params after the latent gauge), hence identifiable: the global
optimum of the terminal-only least-squares objective recovers the shared op
matrices and extrapolates unseen orders essentially exactly.

The only obstacle is that the terminal-only objective is non-convex (bilinear in
the op matrices), so a single fit can stick in a local optimum. Training loss is a
reliable SELECTOR (runs that fit the training protocols to the noise floor also
solve held-out orders), so we train several restarts and keep the lowest-training-
loss members as the predictive ensemble.

Row interface: fit(rows)/predict_distribution(rows) on (x1,x2,steps,y) only.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch

INTERMEDIATE_OPS = ("A", "B", "C", "D", "M")


@dataclasses.dataclass(frozen=True)
class LinearComposedConfig:
    latent_dim: int = 2          # match the true state dim -> identifiable (no headroom)
    n_restarts: int = 10         # random restarts of the non-convex fit
    keep_best: int = 3           # ensemble the lowest-training-loss restarts
    learning_rate: float = 5e-3
    max_epochs: int = 4000
    seed: int = 0
    trace_weight: float = 1.0    # weight of the optional intermediate-observation loss


class _LinComposedNet(torch.nn.Module):
    """Shared per-operation LINEAR maps composed along a protocol step string."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(2, d)
        self.ops = torch.nn.ModuleDict({op: torch.nn.Linear(d, d) for op in INTERMEDIATE_OPS})
        self.obs = torch.nn.Linear(d, 1)
        self.align = torch.nn.Linear(d, 2)   # latent -> observed 2-D, for optional trace supervision

    def forward(self, x: torch.Tensor, steps: Sequence[str]) -> torch.Tensor:
        s = self.embed(x)
        for op in steps[:-1]:            # last step is the terminal observer
            if op in self.ops:
                s = self.ops[op](s)
        return self.obs(s)[:, 0]

    def forward_trace(self, x: torch.Tensor, steps: Sequence[str], obs_pos):
        obs_set = set(obs_pos); s = self.embed(x); captured = {}
        for t, op in enumerate(steps[:-1]):
            if op in self.ops:
                s = self.ops[op](s)
            if t in obs_set:
                captured[t] = self.align(s)
        return self.obs(s)[:, 0], captured


def _group(rows):
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g[str(r["steps"])].append(i)
    return g


@dataclasses.dataclass
class LinearComposedOFML:
    """Terminal-only OFML with linear per-operation maps and multi-restart training.

    Predictive mean/variance are the mean/spread of the ``keep_best`` lowest-training-
    loss restarts (which, when the fit reaches the noise floor, agree -> confident and
    correct on held-out orders)."""

    config: LinearComposedConfig = dataclasses.field(default_factory=LinearComposedConfig)
    members_: list | None = None
    train_losses_: list | None = None
    y_mean_: float = 0.0
    y_std_: float = 1.0

    def fit(self, rows: Sequence[Mapping[str, str]], trace=None) -> "LinearComposedOFML":
        cfg = self.config
        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        y = np.array([float(r["y"]) for r in rows], dtype=float)
        self.y_mean_ = float(y.mean()); self.y_std_ = float(y.std()) or 1.0
        n = len(rows)
        use_trace = trace is not None and any(len(trace[i]) > 0 for i in range(n))
        if use_trace:
            allt = np.array([v for i in range(n) for (_, v) in trace[i]], dtype=float)
            self.trace_mean_ = allt.mean(0); self.trace_std_ = np.where(allt.std(0) < 1e-6, 1.0, allt.std(0))
        else:
            self.trace_mean_, self.trace_std_ = np.zeros(2), np.ones(2)
        batches = []
        for steps, idx in _group(rows).items():
            xb = torch.as_tensor(x[idx], dtype=torch.float32)
            yb = torch.as_tensor((y[idx] - self.y_mean_) / self.y_std_, dtype=torch.float32)
            obs_pos, tb = [], None
            if use_trace and len(trace[idx[0]]) > 0:
                obs_pos = [t for (t, _) in trace[idx[0]]]
                tb = {}
                for pos_k, t in enumerate(obs_pos):
                    tgt = np.array([trace[i][pos_k][1] for i in idx], dtype=float)
                    tgt = (tgt - self.trace_mean_) / self.trace_std_
                    tb[t] = torch.as_tensor(tgt, dtype=torch.float32)
            batches.append((steps.split(), xb, yb, obs_pos, tb))
        total = float(len(rows))

        trained = []
        for r in range(cfg.n_restarts):
            torch.manual_seed(cfg.seed * 1000 + r)
            net = _LinComposedNet(cfg.latent_dim)
            opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
            net.train()
            last = float("inf")
            for _ in range(cfg.max_epochs):
                opt.zero_grad(); term = torch.zeros(()); tloss = torch.zeros(())
                for steps, xb, yb, obs_pos, tb in batches:
                    if tb:
                        pred, captured = net.forward_trace(xb, steps, obs_pos)
                        term = term + torch.nn.functional.mse_loss(pred, yb, reduction="sum")
                        for t in obs_pos:
                            tloss = tloss + torch.nn.functional.mse_loss(captured[t], tb[t], reduction="sum")
                    else:
                        term = term + torch.nn.functional.mse_loss(net(xb, steps), yb, reduction="sum")
                ((term + cfg.trace_weight * tloss) / total).backward(); opt.step()
                last = float(term / total)   # SELECT restarts by terminal loss (the deliverable)
            net.eval()
            trained.append((last, net))
        trained.sort(key=lambda t: t[0])
        keep = max(1, min(cfg.keep_best, cfg.n_restarts))
        self.train_losses_ = [t[0] for t in trained]
        self.members_ = [t[1] for t in trained[:keep]]
        return self

    def predict_distribution(self, rows: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        if not self.members_:
            raise ValueError("LinearComposedOFML must be fit before predict.")
        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        mean = np.zeros(len(rows)); var = np.zeros(len(rows))
        with torch.no_grad():
            for steps, idx in _group(rows).items():
                xb = torch.as_tensor(x[idx], dtype=torch.float32); sl = steps.split()
                preds = np.stack([net(xb, sl).numpy() for net in self.members_], axis=0)
                m = preds.mean(0) * self.y_std_ + self.y_mean_
                v = preds.var(0) * (self.y_std_ ** 2) if preds.shape[0] > 1 else np.zeros(preds.shape[1])
                for k, i in enumerate(idx):
                    mean[i] = m[k]; var[i] = v[k]
        return mean, var

    def predict(self, rows):
        return self.predict_distribution(rows)[0]

    @property
    def diagnostics_(self):
        return {"train_losses": self.train_losses_, "n_members": len(self.members_ or [])}
