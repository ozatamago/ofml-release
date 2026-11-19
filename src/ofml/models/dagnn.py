"""DAGNN-Ens: an order-aware, fully shared DAG encoder baseline (Thost & Chen, ICLR 2021).

Experiment 1 of the AAAI-27 review response (REPORTS/EXP1_ORDER_AWARE_BASELINE_SPEC.md, Sec. 3.2).

Graph
-----
A protocol string ``"A B O"`` becomes the chain  in -> A -> B -> O  (node 0 is the input node;
the last node is the single terminal observer O). A general DAG can be given in
"node types + parent index lists" form (``dag_from_parents``); nodes must be topologically
ordered (every parent index < child index). Only chains are used in the experiment.

Model (unidirectional)
----------------------
* h^0_in = W_in x + b ;  h^0_v = E[tau(v)] (type embedding) for every other node.
* layer l = 1..L, nodes in topological order:
      m_v^l = sum_{u in P(v)} alpha_vu^l h_u^l,
      alpha_vu^l = softmax_u( w1^l . h_v^{l-1} + w2^l . h_u^l )   (m_v^l = 0 if P(v) is empty)
      h_v^l = GRU^l(input = h_v^{l-1}, hidden = m_v^l)
  (with a single parent alpha == 1 exactly, so the softmax is skipped: bit-identical.)
* readout:  y_hat = Linear(ReLU(Linear( concat_{l=0..L} h_O^l )))   (single sink -> max-pool = id).

Training follows TerminalOnlyOFML: y standardized on the training rows, full-batch Adam,
loss = sum of squared errors over all rows / n_rows, fixed number of updates, no early stopping.
The ensemble members (seed = config.seed + j) are stacked with ``torch.func.stack_module_state``
and trained simultaneously with ``vmap(functional_call)``; because Adam (incl. L2 weight decay)
is elementwise and the member losses are summed, this equals independent per-member training.

Batching: by default all rows are processed in one padded batch. For a chain, padding nodes are
appended AFTER the observer, so they never influence any node at or before the observer
(unidirectional message passing); the terminal readout gathers each row's own O node. This is
mathematically identical to a per-protocol-string forward (``batching="protocol"``), which is kept
for verification.

GPU: with ``device="cuda"`` the whole training step (forward + backward + Adam(capturable=True)) is
captured once in a CUDA graph after ``N_WARMUP`` eager steps and replayed for the remaining steps
(same updates; ~10x faster because the tiny kernels are launch-bound). Reruns on the same GPU are
bit-identical; CPU and GPU agree to ~1e-5 after a few steps, but training is chaotic, so per-point
predictions after 400 steps differ between devices.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch

VOCAB = ("A", "B", "C", "D", "M", "O")   # INTERMEDIATE_OPS + ("O",), fixed (same as OFML)
INPUT_TYPE = "in"


# --------------------------------------------------------------------------- graphs
@dataclasses.dataclass(frozen=True)
class DAG:
    """Topologically ordered DAG. types[0] must be the input node; out = the terminal node index."""

    types: tuple[str, ...]
    parents: tuple[tuple[int, ...], ...]
    out: int

    @property
    def key(self):
        return (self.types, self.parents, self.out)

    def is_chain(self) -> bool:
        return all(p == ((i - 1,) if i > 0 else ()) for i, p in enumerate(self.parents)) and self.out == len(self.types) - 1


def dag_from_parents(types: Sequence[str], parents: Sequence[Sequence[int]], out: int | None = None) -> DAG:
    """DAG from node types + parent lists (node 0 = input node, type 'in')."""
    types = tuple(str(t) for t in types)
    parents = tuple(tuple(int(u) for u in p) for p in parents)
    if len(types) != len(parents):
        raise ValueError("types and parents must have the same length")
    if types[0] != INPUT_TYPE or parents[0]:
        raise ValueError("node 0 must be the input node 'in' without parents")
    for v, p in enumerate(parents):
        for u in p:
            if not 0 <= u < v:
                raise ValueError(f"node {v}: parent {u} violates topological order")
    for t in types[1:]:
        if t not in VOCAB:
            raise ValueError(f"unknown node type {t!r}")
    return DAG(types, parents, len(types) - 1 if out is None else int(out))


def chain_from_steps(steps: str | Sequence[str]) -> DAG:
    """'A B O' -> in -> A -> B -> O."""
    ops = steps.split() if isinstance(steps, str) else list(steps)
    types = [INPUT_TYPE] + ops
    parents = [[]] + [[i] for i in range(len(ops))]
    return dag_from_parents(types, parents)


# --------------------------------------------------------------------------- modules
class GRUCell(torch.nn.Module):
    """torch.nn.GRUCell re-implemented with plain matmuls (vmap-friendly; same params/init/math)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.empty(3 * hidden_size))
        std = 1.0 / math.sqrt(hidden_size)
        for w in self.parameters():
            torch.nn.init.uniform_(w, -std, std)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gi = x @ self.weight_ih.T + self.bias_ih
        gh = h @ self.weight_hh.T + self.bias_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return n + z * (h - n)          # == (1 - z) * n + z * h


@dataclasses.dataclass(frozen=True)
class DAGNNConfig:
    n_layers: int = 1          # L
    hidden_dim: int = 24       # d_h
    readout_dim: int = 24      # r (width of the readout MLP hidden layer)
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    steps: int = 400           # full-batch Adam updates
    seed: int = 0
    batching: str = "padded"   # "padded" | "protocol"


class DAGNNNet(torch.nn.Module):
    def __init__(self, cfg: DAGNNConfig) -> None:
        super().__init__()
        d = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.inp = torch.nn.Linear(2, d)
        self.emb = torch.nn.Embedding(len(VOCAB), d)
        self.grus = torch.nn.ModuleList([GRUCell(d, d) for _ in range(cfg.n_layers)])
        # attention scorers w1^l (on h_v^{l-1}) and w2^l (on h_u^l); inactive on chains (alpha == 1)
        self.att_w1 = torch.nn.Parameter(torch.zeros(cfg.n_layers, d))
        self.att_w2 = torch.nn.Parameter(torch.zeros(cfg.n_layers, d))
        torch.nn.init.normal_(self.att_w1, 0.0, 1.0 / math.sqrt(d))
        torch.nn.init.normal_(self.att_w2, 0.0, 1.0 / math.sqrt(d))
        self.read1 = torch.nn.Linear((cfg.n_layers + 1) * d, cfg.readout_dim)
        self.read_act = torch.nn.ReLU()
        self.read2 = torch.nn.Linear(cfg.readout_dim, 1)

    def forward(self, x: torch.Tensor, type_onehot: torch.Tensor, out_sel: torch.Tensor,
                parents: tuple[tuple[int, ...], ...]) -> torch.Tensor:
        """x [N,2]; type_onehot [N,T,V] one-hot node types (column 0 = input node, ignored);
        out_sel [N,T] one-hot of each row's terminal node; parents: shared topological template
        of length T. Returns [N].

        The embedding lookup is one-hot @ E and the terminal gather is a one-hot sum: bit-identical
        to index lookups, but deterministic on CUDA and free of host syncs (CUDA-graph capturable)."""
        N, T = out_sel.shape
        e_all = type_onehot @ self.emb.weight                                           # [N,T,d]
        h_prev = [self.inp(x)] + [e_all[:, v] for v in range(1, T)]
        per_layer = [torch.stack(h_prev, 1)]                       # [N,T,d] for l = 0
        for l in range(self.n_layers):
            gru = self.grus[l]
            h_cur: list[torch.Tensor] = []
            for v in range(T):
                pv = parents[v]
                if len(pv) == 0:
                    m = torch.zeros_like(h_prev[v])
                elif len(pv) == 1:
                    m = h_cur[pv[0]]                                # softmax over one parent == 1
                else:
                    hu = torch.stack([h_cur[u] for u in pv], 1)    # [N,P,d]
                    logits = (h_prev[v] @ self.att_w1[l])[:, None] + hu @ self.att_w2[l]   # [N,P]
                    alpha = torch.softmax(logits, dim=1)
                    m = (alpha[..., None] * hu).sum(1)
                h_cur.append(gru(h_prev[v], m))
            per_layer.append(torch.stack(h_cur, 1))
            h_prev = h_cur
        sel = out_sel[:, :, None]                                                       # [N,T,1]
        z = torch.cat([(H * sel).sum(1) for H in per_layer], dim=-1)  # h_O^l per row -> [N,(L+1)d]
        return self.read2(self.read_act(self.read1(z)))[:, 0]


# --------------------------------------------------------------------------- batching
def _row_graph(row: Mapping) -> DAG:
    g = row.get("dag")
    if g is not None:
        return g if isinstance(g, DAG) else dag_from_parents(*g)
    return chain_from_steps(str(row["steps"]))


def _type_ids(types: Sequence[str]) -> list[int]:
    return [0] + [VOCAB.index(t) for t in types[1:]]


def make_batches(rows: Sequence[Mapping], x: np.ndarray, mode: str, device) -> list[tuple]:
    """Return [(row_index_array, x_tensor, type_onehot, out_sel, parents_template)]."""
    graphs = [_row_graph(r) for r in rows]
    batches = []
    chain_idx = [i for i, g in enumerate(graphs) if g.is_chain()] if mode == "padded" else []
    if chain_idx:
        T = max(len(graphs[i].types) for i in chain_idx)
        pad = VOCAB.index("O")
        types = np.full((len(chain_idx), T), pad, dtype=np.int64)
        out = np.zeros(len(chain_idx), dtype=np.int64)
        for k, i in enumerate(chain_idx):
            ids = _type_ids(graphs[i].types)
            types[k, :len(ids)] = ids
            out[k] = graphs[i].out
        parents = tuple(((v - 1,) if v > 0 else ()) for v in range(T))
        batches.append((np.array(chain_idx), types, out, parents))
    done = set(chain_idx)
    groups: dict = defaultdict(list)
    for i, g in enumerate(graphs):
        if i not in done:
            groups[g.key].append(i)
    for (types_s, parents, out_v), idx in groups.items():
        ids = np.tile(np.array(_type_ids(types_s), dtype=np.int64), (len(idx), 1))
        batches.append((np.array(idx), ids, np.full(len(idx), out_v, dtype=np.int64), parents))
    res = []
    for idx, types, out, parents in batches:
        T = types.shape[1]
        onehot = np.eye(len(VOCAB), dtype=np.float32)[types]          # [n,T,V]
        sel = np.eye(T, dtype=np.float32)[out]                        # [n,T]
        res.append((idx,
                    torch.as_tensor(x[idx], dtype=torch.float32, device=device),
                    torch.as_tensor(onehot, device=device),
                    torch.as_tensor(sel, device=device),
                    parents))
    return res


# --------------------------------------------------------------------------- vectorized ensemble trainer
def _vmapped(base: torch.nn.Module, args: tuple, static: dict):
    """vmap(functional_call) over stacked member params; tensors in args are shared (in_dim None),
    non-tensor structure (e.g. the parents template) is bound as keyword arguments."""
    from torch.func import functional_call, vmap

    def f(p, b, *a):
        return functional_call(base, (p, b), a, static)
    return vmap(f, in_dims=(0, 0) + (None,) * len(args))


class _VmapEnsemble:
    """Shared fit/predict machinery for a stack of identically shaped member networks.

    Subclasses define ``_build_net(seed)`` and ``_batches(rows, x)`` returning
    [(row_idx, tensor_args, static_kwargs)] where ``net(*tensor_args, **static_kwargs) -> [n]``.
    """

    N_WARMUP = 3   # eager steps before CUDA-graph capture (they are real updates, counted in `steps`)

    def __init__(self, n_members: int = 5, device: str = "cpu", vectorize: bool = True,
                 cuda_graph: bool = True) -> None:
        self.n_members = n_members
        self.device = torch.device(device)
        self.vectorize = vectorize
        self.cuda_graph = cuda_graph
        self.params_ = None
        self.y_mean_ = 0.0
        self.y_std_ = 1.0

    # to be provided by subclasses
    cfg: object

    def _build_net(self, seed: int) -> torch.nn.Module:  # pragma: no cover
        raise NotImplementedError

    def _batches(self, rows, x):  # pragma: no cover
        raise NotImplementedError

    def _init_members(self):
        nets = []
        for j in range(self.n_members):
            torch.manual_seed(self.cfg.seed + j)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.cfg.seed + j)
            nets.append(self._build_net(self.cfg.seed + j).to(self.device).eval())   # dropout-free
        return nets

    def fit(self, rows: Sequence[Mapping]) -> "_VmapEnsemble":
        from torch.func import stack_module_state
        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        y = np.array([float(r["y"]) for r in rows], dtype=float)
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        ys = (y - self.y_mean_) / self.y_std_
        n_total = float(len(rows))
        batches = self._batches(rows, x)
        yb = [torch.as_tensor(ys[idx], dtype=torch.float32, device=self.device) for idx, _, _ in batches]
        nets = self._init_members()
        self.base_ = nets[0]
        cfg = self.cfg
        if self.vectorize:
            params, buffers = stack_module_state(nets)
            params = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
            base = nets[0]
            fvs = [_vmapped(base, args, static) for _, args, static in batches]
            use_graph = self.cuda_graph and self.device.type == "cuda" and cfg.steps > self.N_WARMUP
            opt = torch.optim.Adam(list(params.values()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                                   capturable=use_graph)

            def train_step():
                loss = 0.0
                for (idx, args, static), fv, yt in zip(batches, fvs, yb):
                    pred = fv(params, buffers, *args)                     # [M, n]
                    loss = loss + ((pred - yt) ** 2).sum()
                (loss / n_total).backward()   # sum over members -> per-member gradients unchanged
                opt.step()

            if not use_graph:
                for _ in range(cfg.steps):
                    opt.zero_grad()
                    train_step()
            else:
                # whole-step CUDA graph (forward + backward + Adam): same updates, no launch overhead
                side = torch.cuda.Stream(device=self.device)
                side.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(side):
                    for _ in range(self.N_WARMUP):
                        opt.zero_grad(set_to_none=True)
                        train_step()
                torch.cuda.current_stream(self.device).wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                opt.zero_grad(set_to_none=True)
                with torch.cuda.graph(graph):
                    train_step()                                         # captured, NOT executed
                for _ in range(cfg.steps - self.N_WARMUP):
                    graph.replay()
                torch.cuda.synchronize(self.device)
                del graph
            self.params_ = {k: v.detach() for k, v in params.items()}
            self.buffers_ = buffers
        else:
            for net in nets:
                opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
                for _ in range(cfg.steps):
                    opt.zero_grad()
                    loss = 0.0
                    for (idx, args, static), yt in zip(batches, yb):
                        loss = loss + ((net(*args, **static) - yt) ** 2).sum()
                    (loss / n_total).backward()
                    opt.step()
            params, buffers = stack_module_state(nets)
            self.params_ = {k: v.detach() for k, v in params.items()}
            self.buffers_ = buffers
        return self

    def _member_preds_std(self, rows: Sequence[Mapping]) -> np.ndarray:
        """[M, N] predictions on the standardized y scale."""
        if self.params_ is None:
            raise ValueError("model must be fit before predict")
        x = np.array([[float(r["x1"]), float(r["x2"])] for r in rows], dtype=float)
        out = np.zeros((self.n_members, len(rows)))
        with torch.no_grad():
            for idx, args, static in self._batches(rows, x):
                fv = _vmapped(self.base_, args, static)
                out[:, idx] = fv(self.params_, self.buffers_, *args).cpu().numpy()
        return out

    def predict_samples(self, rows: Sequence[Mapping]) -> np.ndarray:
        return self._member_preds_std(rows) * self.y_std_ + self.y_mean_

    def predict_distribution(self, rows: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
        # identical to TerminalOnlyOFML: mean/var (ddof=0) across members on the std scale, then rescale
        p = self._member_preds_std(rows)
        mean = p.mean(0) * self.y_std_ + self.y_mean_
        var = p.var(0) * (self.y_std_ ** 2) if p.shape[0] > 1 else np.zeros(p.shape[1])
        return mean, var

    def predict(self, rows: Sequence[Mapping]) -> np.ndarray:
        return self.predict_distribution(rows)[0]


class DAGNNEnsemble(_VmapEnsemble):
    """Deep ensemble (seed-only) of DAGNN regressors over protocol DAGs."""

    def __init__(self, config: DAGNNConfig | None = None, n_members: int = 5, device: str = "cpu",
                 vectorize: bool = True, cuda_graph: bool = True) -> None:
        super().__init__(n_members=n_members, device=device, vectorize=vectorize, cuda_graph=cuda_graph)
        self.cfg = config or DAGNNConfig()

    def _build_net(self, seed: int) -> torch.nn.Module:
        return DAGNNNet(self.cfg)

    def _batches(self, rows, x):
        return [(idx, (xb, types, out), {"parents": parents})
                for idx, xb, types, out, parents in make_batches(rows, x, self.cfg.batching, self.device)]
