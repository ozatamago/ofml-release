"""Models for Experiment 3 (merge DAGs, REPORTS/EXP3_MERGE_DAG_SPEC.md Sec. 4).

All models read rows with ``steps`` (a protocol string, ``dag_protocol`` grammar) and the design
variable ``x1..x{x_dim}`` (x_dim = 2 per input block). Every branch head reads its own block.

* DAGOFML          OFML-DAG-Ens. CPU, trained exactly like TerminalOnlyOFML (per member, full-batch
                   Adam, sum-of-squares over protocol groups). On a chain it IS TerminalOnlyOFML.
* SharedCellDAG    SharedCell-DAG-Ens: one shared MLP h([u_v; e_tau]) for every type (second input
                   zero-padded for unary ops), the OFML observer. GPU (dagnn._VmapEnsemble).
* DAGNNDAG         DAGNN-Ens (aggregate="attention", Thost & Chen 2021, as dagnn.DAGNNNet) and
                   DAGNN-Cat-Ens (aggregate="concat": m_v = W_m [h_c1; h_c2] + b_m), with one input
                   node per input block. GPU (dagnn._VmapEnsemble).
* SharedEnsX       Shared-Ens with x1..x4 (zero-padded) in the features (design 2).
* IndependentX     Independent-Ens whose per-target MLP reads the full design variable.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .dag_protocol import INPUT_TYPE, PDAG, parse_protocol
from .dagnn import VOCAB, GRUCell, _VmapEnsemble
from .ofml.operation_models import _OperationMLP, _set_torch_seed
from .ofml.terminal_only import INTERMEDIATE_OPS


# --------------------------------------------------------------------------- helpers
def group_rows(rows: Sequence[Mapping]) -> dict[str, list[int]]:
    """Row indices per protocol, in order of first appearance (as TerminalOnlyOFML)."""
    g: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        g[str(r["steps"])].append(i)
    return g


def x_matrix(rows: Sequence[Mapping], idx: Sequence[int], dag: PDAG) -> np.ndarray:
    return np.array([[float(rows[i][f"x{j}"]) for j in range(1, dag.x_dim + 1)] for i in idx], dtype=float)


def _block(x: torch.Tensor, k: int) -> torch.Tensor:
    return x[:, 2 * (k - 1):2 * k]


# --------------------------------------------------------------------------- OFML-DAG (CPU)
@dataclasses.dataclass(frozen=True)
class DAGOFMLConfig:
    latent_dim: int = 10
    hidden_dim: int = 32
    n_hidden: int = 2
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    max_epochs: int = 400
    seed: int = 0
    merge_arity: int = 2     # inputs of the type "M"; 1 = the chain benchmarks' unary M


class _DAGComposedNet(torch.nn.Module):
    """terminal_only._ComposedNet with typed modules composed along a DAG (same parameter creation order)."""

    def __init__(self, cfg: DAGOFMLConfig) -> None:
        super().__init__()
        d = cfg.latent_dim
        self.embed = torch.nn.Linear(2, d)
        self.ops = torch.nn.ModuleDict(
            {op: _OperationMLP(d * (cfg.merge_arity if op == "M" else 1), d, cfg.hidden_dim, 0.0, cfg.n_hidden)
             for op in INTERMEDIATE_OPS})
        self.obs = _OperationMLP(d, 1, cfg.hidden_dim, 0.0, cfg.n_hidden)
        self.align = torch.nn.Linear(d, 2)   # unused; keeps the initialisation stream of _ComposedNet

    def forward_dag(self, x: torch.Tensor, dag: PDAG) -> torch.Tensor:
        vals: dict[int, torch.Tensor] = {}
        for v, (t, pa) in enumerate(zip(dag.types, dag.parents)):
            if t == INPUT_TYPE:
                vals[v] = self.embed(_block(x, dag.blocks[v]))
            elif v == dag.out:
                return self.obs(vals[pa[0]])[:, 0]
            else:
                u = vals[pa[0]] if len(pa) == 1 else torch.cat([vals[c] for c in pa], -1)
                vals[v] = self.ops[t](u)
        raise ValueError("DAG without an observer")


@dataclasses.dataclass
class DAGOFML:
    """Seed-only deep ensemble of the typed-module DAG surrogate (training as TerminalOnlyOFML)."""

    config: DAGOFMLConfig = dataclasses.field(default_factory=DAGOFMLConfig)
    n_members: int = 5
    nets_: list | None = None
    y_mean_: float = 0.0
    y_std_: float = 1.0

    def fit(self, rows: Sequence[Mapping]) -> "DAGOFML":
        cfg = self.config
        y = np.array([float(r["y"]) for r in rows], dtype=float)
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        ys = (y - self.y_mean_) / self.y_std_
        batches = []
        for steps, idx in group_rows(rows).items():
            dag = parse_protocol(steps)
            batches.append((dag, torch.as_tensor(x_matrix(rows, idx, dag), dtype=torch.float32),
                            torch.as_tensor(ys[idx], dtype=torch.float32)))
        total = float(len(rows))
        self.nets_ = []
        for member in range(max(1, self.n_members)):
            _set_torch_seed(cfg.seed + member)
            net = _DAGComposedNet(cfg)
            opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            net.train()
            for _ in range(cfg.max_epochs):
                opt.zero_grad()
                loss = torch.zeros(())
                for dag, xb, yb in batches:
                    loss = loss + torch.nn.functional.mse_loss(net.forward_dag(xb, dag), yb, reduction="sum")
                ((loss + 1.0 * torch.zeros(())) / total).backward()   # as TerminalOnlyOFML (empty trace term)
                opt.step()
            net.eval()
            self.nets_.append(net)
        return self

    def _member_preds_std(self, rows: Sequence[Mapping]) -> np.ndarray:
        if not self.nets_:
            raise ValueError("DAGOFML must be fit before predict")
        out = np.zeros((len(self.nets_), len(rows)), dtype=np.float32)   # float32 like TerminalOnlyOFML
        with torch.no_grad():
            for steps, idx in group_rows(rows).items():
                dag = parse_protocol(steps)
                xb = torch.as_tensor(x_matrix(rows, idx, dag), dtype=torch.float32)
                for k, net in enumerate(self.nets_):
                    out[k, idx] = net.forward_dag(xb, dag).numpy()
        return out

    def predict_samples(self, rows: Sequence[Mapping]) -> np.ndarray:
        return self._member_preds_std(rows) * self.y_std_ + self.y_mean_

    def predict_distribution(self, rows: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
        p = self._member_preds_std(rows)                       # same float32 arithmetic as TerminalOnlyOFML
        mean = (p.mean(0) * self.y_std_ + self.y_mean_).astype(float)
        var = (p.var(0) * (self.y_std_ ** 2)).astype(float) if p.shape[0] > 1 else np.zeros(p.shape[1])
        return mean, var

    def predict(self, rows: Sequence[Mapping]) -> np.ndarray:
        return self.predict_distribution(rows)[0]


# --------------------------------------------------------------------------- GPU models (vmap ensembles)
def _dag_batches(rows, device) -> list[tuple]:
    """[(row_idx, (x,), {"dag": dag})] one group per protocol (no padding)."""
    out = []
    for steps, idx in group_rows(rows).items():
        dag = parse_protocol(steps)
        out.append((np.array(idx), (torch.as_tensor(x_matrix(rows, idx, dag), dtype=torch.float32, device=device),),
                    {"dag": dag}))
    return out


@dataclasses.dataclass(frozen=True)
class SharedCellDAGConfig:
    latent_dim: int = 10       # d (as OFML-DAG)
    embed_dim: int = 4         # k
    hidden_dim: int = 61       # w (param-matched)
    n_hidden: int = 2
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    steps: int = 400
    seed: int = 0


class SharedCellDAGNet(torch.nn.Module):
    def __init__(self, cfg: SharedCellDAGConfig) -> None:
        super().__init__()
        d, k, w = cfg.latent_dim, cfg.embed_dim, cfg.hidden_dim
        self.embed = torch.nn.Linear(2, d)
        self.type_emb = torch.nn.Embedding(len(VOCAB), k)
        self.cell = _OperationMLP(2 * d + k, d, w, 0.0, cfg.n_hidden)
        self.obs = _OperationMLP(d, 1, w, 0.0, cfg.n_hidden)

    def forward(self, x: torch.Tensor, dag: PDAG) -> torch.Tensor:
        n = x.shape[0]
        vals: dict[int, torch.Tensor] = {}
        for v, (t, pa) in enumerate(zip(dag.types, dag.parents)):
            if t == INPUT_TYPE:
                vals[v] = self.embed(_block(x, dag.blocks[v]))
            elif v == dag.out:
                return self.obs(vals[pa[0]])[:, 0]
            else:
                ins = [vals[c] for c in pa]
                if len(ins) == 1:
                    ins.append(torch.zeros_like(ins[0]))
                e = self.type_emb.weight[VOCAB.index(t)].expand(n, -1)
                vals[v] = self.cell(torch.cat(ins + [e], -1))
        raise ValueError("DAG without an observer")


class SharedCellDAG(_VmapEnsemble):
    def __init__(self, config: SharedCellDAGConfig | None = None, n_members: int = 5, device: str = "cpu",
                 vectorize: bool = True, cuda_graph: bool = True) -> None:
        super().__init__(n_members=n_members, device=device, vectorize=vectorize, cuda_graph=cuda_graph)
        self.cfg = config or SharedCellDAGConfig()

    def _build_net(self, seed: int) -> torch.nn.Module:
        return SharedCellDAGNet(self.cfg)

    def _batches(self, rows, x):          # x (x1, x2 only) is ignored: blocks are read from the rows
        return _dag_batches(rows, self.device)


@dataclasses.dataclass(frozen=True)
class DAGNNDAGConfig:
    n_layers: int = 1          # L
    hidden_dim: int = 24       # d_h
    readout_dim: int = 24      # r
    aggregate: str = "attention"   # "attention" (DAGNN-Ens) | "concat" (DAGNN-Cat-Ens)
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    steps: int = 400
    seed: int = 0


class DAGNNDAGNet(torch.nn.Module):
    """dagnn.DAGNNNet with one input node per input block (same parameter names for 'attention')."""

    def __init__(self, cfg: DAGNNDAGConfig) -> None:
        super().__init__()
        import math
        d, L = cfg.hidden_dim, cfg.n_layers
        self.n_layers, self.aggregate = L, cfg.aggregate
        self.inp = torch.nn.Linear(2, d)
        self.emb = torch.nn.Embedding(len(VOCAB), d)
        self.grus = torch.nn.ModuleList([GRUCell(d, d) for _ in range(L)])
        if cfg.aggregate == "attention":
            self.att_w1 = torch.nn.Parameter(torch.zeros(L, d))
            self.att_w2 = torch.nn.Parameter(torch.zeros(L, d))
            torch.nn.init.normal_(self.att_w1, 0.0, 1.0 / math.sqrt(d))
            torch.nn.init.normal_(self.att_w2, 0.0, 1.0 / math.sqrt(d))
        elif cfg.aggregate == "concat":
            self.msg = torch.nn.ModuleList([torch.nn.Linear(2 * d, d) for _ in range(L)])
        else:
            raise ValueError(cfg.aggregate)
        self.read1 = torch.nn.Linear((L + 1) * d, cfg.readout_dim)
        self.read_act = torch.nn.ReLU()
        self.read2 = torch.nn.Linear(cfg.readout_dim, 1)

    def _message(self, l: int, h_v_prev: torch.Tensor, hs: list[torch.Tensor]) -> torch.Tensor:
        if not hs:
            return torch.zeros_like(h_v_prev)
        if self.aggregate == "concat":
            if len(hs) == 1:
                hs = hs + [torch.zeros_like(hs[0])]
            return self.msg[l](torch.cat(hs, -1))
        if len(hs) == 1:
            return hs[0]                                         # softmax over one parent == 1
        hu = torch.stack(hs, 1)                                  # [N,P,d]
        logits = (h_v_prev @ self.att_w1[l])[:, None] + hu @ self.att_w2[l]
        alpha = torch.softmax(logits, dim=1)
        return (alpha[..., None] * hu).sum(1)

    def forward(self, x: torch.Tensor, dag: PDAG) -> torch.Tensor:
        n = x.shape[0]
        T = len(dag.types)
        h_prev = [self.inp(_block(x, dag.blocks[v])) if t == INPUT_TYPE
                  else self.emb.weight[VOCAB.index(t)].expand(n, -1) for v, t in enumerate(dag.types)]
        per_layer = [h_prev[dag.out]]
        for l in range(self.n_layers):
            h_cur: list[torch.Tensor] = []
            for v in range(T):
                m = self._message(l, h_prev[v], [h_cur[u] for u in dag.parents[v]])
                h_cur.append(self.grus[l](h_prev[v], m))
            per_layer.append(h_cur[dag.out])
            h_prev = h_cur
        z = torch.cat(per_layer, dim=-1)
        return self.read2(self.read_act(self.read1(z)))[:, 0]


class DAGNNDAG(_VmapEnsemble):
    def __init__(self, config: DAGNNDAGConfig | None = None, n_members: int = 5, device: str = "cpu",
                 vectorize: bool = True, cuda_graph: bool = True) -> None:
        super().__init__(n_members=n_members, device=device, vectorize=vectorize, cuda_graph=cuda_graph)
        self.cfg = config or DAGNNDAGConfig()

    def _build_net(self, seed: int) -> torch.nn.Module:
        return DAGNNDAGNet(self.cfg)

    def _batches(self, rows, x):
        return _dag_batches(rows, self.device)


# --------------------------------------------------------------------------- Shared-Ens / Independent-Ens with x of any width
N_X_FEATURES = 4


def _x_features(row: Mapping, n: int = N_X_FEATURES) -> list[float]:
    return [float(row.get(f"x{j}", 0.0)) for j in range(1, n + 1)]


def make_x_encoder():
    from .features import RowFeatureEncoder

    @dataclasses.dataclass
    class XRowFeatureEncoder(RowFeatureEncoder):
        """RowFeatureEncoder with x1..x4 (missing -> 0) in place of x1, x2."""

        def transform(self, rows):
            base = super().transform(rows)
            if base.shape[0] == 0:
                return np.zeros((0, self.n_features), dtype=float)
            X = np.array([_x_features(r) for r in rows], dtype=float)
            return np.concatenate([X, base[:, 2:]], axis=1)

        @property
        def n_features(self) -> int:
            return super().n_features + N_X_FEATURES - 2

    return XRowFeatureEncoder()


class SharedEnsX:
    """SharedMLPEnsemble(bootstrap=False) whose members encode x1..x4."""

    def __init__(self, n_members: int = 5, hidden_dim: int = 32, max_epochs: int = 400, seed: int = 0):
        from .shared_mlp import SharedMLPConfig
        self.n_members = n_members
        self.config = SharedMLPConfig(hidden_dim=hidden_dim, max_epochs=max_epochs, seed=seed)
        self.models_ = []

    def fit(self, rows):
        from .shared_mlp import SharedMLP
        self.models_ = []
        for j in range(self.n_members):
            cfg = dataclasses.replace(self.config, seed=self.config.seed + j)
            self.models_.append(SharedMLP(config=cfg, encoder=make_x_encoder()).fit(list(rows)))
        return self

    def predict_samples(self, rows):
        return np.asarray([m.predict(rows) for m in self.models_], dtype=float)

    def predict_distribution(self, rows):
        s = self.predict_samples(rows)
        return s.mean(0), s.var(0)


class IndependentX:
    """scripts/independent_ens.IndependentEnsemble with the per-target MLP reading x1..x{x_dim}."""

    def __init__(self, target_protocols: Sequence[str], n_members: int = 5, hidden_dim: int = 32,
                 max_epochs: int = 400, lr: float = 1e-2, weight_decay: float = 1e-4, seed: int = 0):
        self.targets_ = {t.replace(" ", "_") for t in target_protocols}
        self.M, self.hidden, self.epochs = n_members, hidden_dim, max_epochs
        self.lr, self.wd, self.seed = lr, weight_decay, seed
        self.nets_: dict[str, list] = {}
        self.norm_: dict[str, tuple[float, float]] = {}
        self.source_global_mean_ = 0.0
        self.source_y_var_ = 1.0

    @staticmethod
    def _key(row):
        return row.get("protocol", str(row.get("steps", "")).replace(" ", "_"))

    @staticmethod
    def _X(rows):
        dag = parse_protocol(str(rows[0]["steps"]))
        return np.array([[float(r[f"x{j}"]) for j in range(1, dag.x_dim + 1)] for r in rows], float)

    def fit(self, rows):
        from .shared_mlp import _MLP
        src_y = [float(r["y"]) for r in rows if self._key(r) not in self.targets_ and "y" in r]
        if src_y:
            self.source_global_mean_ = float(np.mean(src_y))
            self.source_y_var_ = float(np.var(src_y)) or 1.0
        self.nets_, self.norm_ = {}, {}
        for tp in self.targets_:
            tr = [r for r in rows if self._key(r) == tp and "y" in r]
            if not tr:
                continue
            X = self._X(tr)
            y = np.array([float(r["y"]) for r in tr], float)
            ym, ys = float(y.mean()), (float(y.std()) or 1.0)
            self.norm_[tp] = (ym, ys)
            xb = torch.as_tensor(X, dtype=torch.float32)
            yb = torch.as_tensor((y - ym) / ys, dtype=torch.float32)
            members = []
            for mi in range(self.M):
                _set_torch_seed(self.seed * 100 + mi)
                net = _MLP(X.shape[1], self.hidden, dropout=0.0)
                opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
                net.train()
                for _ in range(self.epochs):
                    opt.zero_grad()
                    loss = torch.nn.functional.mse_loss(net(xb), yb)
                    loss.backward(); opt.step()
                net.eval(); members.append(net)
            self.nets_[tp] = members
        return self

    def _member_preds(self, rows):
        out = np.empty((self.M, len(rows)), float)
        by: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            by.setdefault(self._key(r), []).append(i)
        for tp, idx in by.items():
            if tp in self.nets_:
                ym, ys = self.norm_[tp]
                xb = torch.as_tensor(self._X([rows[i] for i in idx]), dtype=torch.float32)
                with torch.no_grad():
                    for mi, net in enumerate(self.nets_[tp]):
                        out[mi, idx] = net(xb).cpu().numpy() * ys + ym
            else:
                out[:, idx] = self.source_global_mean_
        return out

    def predict_samples(self, rows):
        return self._member_preds(rows)

    def predict_distribution(self, rows):
        P = self._member_preds(rows)
        var = P.var(0)
        return P.mean(0), np.where(var <= 1e-12, self.source_y_var_, var)
