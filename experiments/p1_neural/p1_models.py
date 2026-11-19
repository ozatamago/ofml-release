"""Neural baselines for purpose 1 (c), the GP-matched benchmark (REPORTS/EXPERIMENT_DESIGN_2026-11-04.md 1.8).

Five seed-only deep ensembles (no resampling of the data), all trained the same way (full-batch Adam with L2
weight decay, fixed number of steps, mean squared error on the response standardised with the SOURCE rows' mean
and sd):

* ``ofml``       OFML-Ens: typed modules composed along the term.  s0 = Linear(1 -> d)(x) (shared input embedding,
                 as w16u_ens / exp3 DAGOFML), A, B: MLP d -> d, C1, C2: MLP d -> d, M: MLP [u1; u2] (2d) -> d (role
                 order kept), R: MLP d -> 1.  Every typed MLP is ``_OperationMLP(n_hidden, width)``.
* ``sharedcell`` SharedCell-DAG-Ens (exp3_models.SharedCellDAGNet): ONE MLP h([u1; u2; e_tau]) for A, B, C1, C2, M
                 (u2 = 0 for unary types; for A, B u1 = s0), a learned type embedding e_tau (dim ``embed_dim``),
                 the same s0 and a separate R MLP d -> 1.
* ``dagnn``      DAGNN-Ens (Thost & Chen 2021, as exp3_models.DAGNNDAGNet aggregate="attention"): one input node
                 h0 = W_in x, other nodes h0 = E[tau]; L layers of (attention-weighted sum of the parents' states,
                 GRU update); readout Linear(ReLU(Linear(concat_l h_R^l))).
* ``dagnn_cat``  DAGNN-Cat-Ens: as dagnn, the message is W_m [h_c1; h_c2] + b_m (role order, h_c2 = 0 for one parent).
* ``shared``     Shared-Ens: one MLP on [x, #A, #B, #C1, #C2, #M] (use counts of the types in the protocol; R is in
                 every protocol, so its count is a constant and is left out).

Evaluation trick (exact): every node value is a deterministic function of the grid value x and of the node's
subterm, so each model is evaluated once per DISTINCT subterm on the 64 grid points (level by level, every type
batched over its nodes of the level), giving a [17, 64] table of predictions; the training loss gathers the rows'
cells from that table.  The loss, its gradient and the predictions are those of the per-row computation (only the
float rounding differs).  Repeated observations of a cell are separate rows.

The members are stacked (torch.func.stack_module_state) and trained together with vmap(functional_call); the loss
is the sum of the member losses and Adam is elementwise, so this equals independent per-member training.
Member m is initialised under torch.manual_seed(seed_base + m) (seed_base = problem seed), the same initial weights
at every refit (cold restart from the same initialisation, as w16u_ens).
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OFML_SRC = HERE.parents[1] / "src"          # ofplang/ofml/src
GPN_GRAPH = HERE.parent / "gpn_graph"
for _p in (str(OFML_SRC), str(GPN_GRAPH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import p1_common as P  # noqa: E402  (numpy only)
from ofml.models.dagnn import GRUCell  # noqa: E402
from ofml.models.ofml.operation_models import _OperationMLP  # noqa: E402

METHODS = ("shared", "dagnn", "dagnn_cat", "sharedcell", "ofml")
METHOD_LABEL = {"shared": "Shared-Ens", "dagnn": "DAGNN-Ens", "dagnn_cat": "DAGNN-Cat-Ens",
                "sharedcell": "SharedCell-DAG-Ens", "ofml": "OFML-Ens"}
LEAF_TYPES = ("A", "B")
INNER_TYPES = ("C1", "C2", "M")
NODE_TYPES = LEAF_TYPES + INNER_TYPES           # types with a learned module (R separately)
ARITY = {"A": 0, "B": 0, "C1": 1, "C2": 1, "M": 2}


# --------------------------------------------------------------------------- terms
def parse_term(s: str):
    """'M(M(A,B),A)' -> ('M', ('M', ('A',), ('B',)), ('A',)); leaves are 1-tuples."""
    s = s.replace(" ", "")
    pos = 0

    def term():
        nonlocal pos
        st = pos
        while pos < len(s) and s[pos].isalnum():
            pos += 1
        name = s[st:pos]
        if pos < len(s) and s[pos] == "(":
            pos += 1
            kids = [term()]
            while s[pos] == ",":
                pos += 1
                kids.append(term())
            assert s[pos] == ")"
            pos += 1
            return (name,) + tuple(kids)
        return (name,)
    t = term()
    assert pos == len(s), s
    return t


def depth(t) -> int:
    return 1 + max((depth(c) for c in t[1:]), default=0)


def type_counts(t) -> dict:
    c = {k: 0 for k in NODE_TYPES}
    stack = [t]
    while stack:
        u = stack.pop()
        c[u[0]] += 1
        stack.extend(u[1:])
    return c


@dataclasses.dataclass(frozen=True)
class Structure:
    """Distinct subterms of the 17 protocol terms, grouped by depth (level 1 = A, B)."""
    roots: tuple            # protocol p -> its term (inside R)
    levels: tuple           # tuple of tuples of subterms, level 1..max
    counts: np.ndarray      # (17, 5) use counts of A, B, C1, C2, M

    @staticmethod
    def build() -> "Structure":
        roots = tuple(parse_term(s) for s in P.PROTOCOL_TERMS)
        seen: dict = {}

        def visit(u):
            for c in u[1:]:
                visit(c)
            seen.setdefault(u, depth(u))
        for r in roots:
            visit(r)
        D = max(seen.values())
        levels = tuple(tuple(u for u in seen if seen[u] == k) for k in range(1, D + 1))
        counts = np.array([[type_counts(r)[k] for k in NODE_TYPES] for r in roots], dtype=np.float32)
        for r in roots:
            for u in seen:
                for c in u[1:]:
                    assert ARITY[u[0]] == len(u) - 1
        return Structure(roots, levels, counts)


STRUCT = Structure.build()


# --------------------------------------------------------------------------- configs
@dataclasses.dataclass(frozen=True)
class NetConfig:
    n_hidden: int = 2          # hidden layers of every MLP (DAGNN: number of DAGNN layers L, see DAGNN_MAP)
    width: int = 32            # hidden width (DAGNN: readout width r)
    latent_dim: int = 4        # node state d (DAGNN: d_h); ignored by Shared-Ens
    embed_dim: int = 4         # SharedCell type embedding k (fixed, exp3 default)
    lr: float = 1e-2
    wd: float = 1e-4
    steps: int = 400
    n_members: int = 5
    dagnn_map: str = "state_latent"   # "state_latent": d_h = latent, L = n_hidden, r = width (default)
                                      # "state_width":  d_h = width,  L = n_hidden, r = width, latent unused

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def dagnn_dims(cfg: NetConfig) -> tuple[int, int, int]:
    """(L, d_h, r) of the DAGNN nets from the common search space."""
    if cfg.dagnn_map == "state_latent":
        return cfg.n_hidden, cfg.latent_dim, cfg.width
    if cfg.dagnn_map == "state_width":
        return cfg.n_hidden, cfg.width, cfg.width
    raise ValueError(cfg.dagnn_map)


# --------------------------------------------------------------------------- nets (one member; vmapped)
class _Base(torch.nn.Module):
    """forward(x [G, 1]) -> [17, G] predictions on the standardised scale."""
    S = STRUCT


class OFMLNet(_Base):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        d, w, nh = cfg.latent_dim, cfg.width, cfg.n_hidden
        self.embed = torch.nn.Linear(1, d)
        self.ops = torch.nn.ModuleDict({t: _OperationMLP(d * max(1, ARITY[t]), d, w, 0.0, nh) for t in NODE_TYPES})
        self.obs = _OperationMLP(d, 1, w, 0.0, nh)

    def forward(self, x):
        G = x.shape[0]
        s0 = self.embed(x)
        vals = {}
        for lev in self.S.levels:
            for t in NODE_TYPES:
                us = [u for u in lev if u[0] == t]
                if not us:
                    continue
                if ARITY[t] == 0:
                    inp = torch.cat([s0] * len(us), 0)
                else:
                    inp = torch.cat([torch.cat([vals[c] for c in u[1:]], -1) for u in us], 0)
                out = self.ops[t](inp)
                for k, u in enumerate(us):
                    vals[u] = out[k * G:(k + 1) * G]
        z = torch.cat([vals[r] for r in self.S.roots], 0)
        return self.obs(z)[:, 0].reshape(len(self.S.roots), G)


class SharedCellNet(_Base):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        d, k, w, nh = cfg.latent_dim, cfg.embed_dim, cfg.width, cfg.n_hidden
        self.embed = torch.nn.Linear(1, d)
        self.type_emb = torch.nn.Embedding(len(NODE_TYPES), k)
        self.cell = _OperationMLP(2 * d + k, d, w, 0.0, nh)
        self.obs = _OperationMLP(d, 1, w, 0.0, nh)

    def forward(self, x):
        G = x.shape[0]
        s0 = self.embed(x)
        zero = torch.zeros_like(s0)
        vals = {}
        for lev in self.S.levels:
            rows = []
            for u in lev:
                t = u[0]
                ins = [s0] if ARITY[t] == 0 else [vals[c] for c in u[1:]]
                if len(ins) == 1:
                    ins.append(zero)
                e = self.type_emb.weight[NODE_TYPES.index(t)].expand(G, -1)
                rows.append(torch.cat(ins + [e], -1))
            out = self.cell(torch.cat(rows, 0))
            for k, u in enumerate(lev):
                vals[u] = out[k * G:(k + 1) * G]
        z = torch.cat([vals[r] for r in self.S.roots], 0)
        return self.obs(z)[:, 0].reshape(len(self.S.roots), G)


DAGNN_VOCAB = NODE_TYPES + ("R",)


class DAGNNNet(_Base):
    """Nodes: the input node 'in' (parent of every A, B node), one node per distinct subterm, one R node per
    protocol (parent = its term).  Unidirectional, so a node's states depend only on its subterm (exact sharing)."""

    def __init__(self, cfg: NetConfig, aggregate: str):
        super().__init__()
        L, d, r = dagnn_dims(cfg)
        self.L, self.aggregate = L, aggregate
        self.inp = torch.nn.Linear(1, d)
        self.emb = torch.nn.Embedding(len(DAGNN_VOCAB), d)
        self.grus = torch.nn.ModuleList([GRUCell(d, d) for _ in range(L)])
        if aggregate == "attention":
            self.att_w1 = torch.nn.Parameter(torch.zeros(L, d))
            self.att_w2 = torch.nn.Parameter(torch.zeros(L, d))
            torch.nn.init.normal_(self.att_w1, 0.0, 1.0 / math.sqrt(d))
            torch.nn.init.normal_(self.att_w2, 0.0, 1.0 / math.sqrt(d))
        elif aggregate == "concat":
            self.msg = torch.nn.ModuleList([torch.nn.Linear(2 * d, d) for _ in range(L)])
        else:
            raise ValueError(aggregate)
        self.read1 = torch.nn.Linear((L + 1) * d, r)
        self.read2 = torch.nn.Linear(r, 1)
        # node order: 'in', subterms level by level, then the 17 R nodes
        IN = ("in",)
        self.nodes = [IN] + [u for lev in self.S.levels for u in lev] + [("R", r_) for r_ in self.S.roots]
        self.node_levels = [[IN]] + [list(lev) for lev in self.S.levels] + [[("R", r_) for r_ in self.S.roots]]

    @staticmethod
    def _parents(u):
        if u[0] == "in":
            return []
        if u[0] in LEAF_TYPES:
            return [("in",)]
        return list(u[1:])

    def _messages(self, l, us, h_prev, h_cur, G):
        d = h_prev[us[0]].shape[-1]
        if self.aggregate == "concat":
            rows = []
            for u in us:
                ps = self._parents(u)
                if not ps:
                    rows.append(None)
                    continue
                hs = [h_cur[p] for p in ps]
                if len(hs) == 1:
                    hs.append(torch.zeros_like(hs[0]))
                rows.append(torch.cat(hs, -1))
            if all(r_ is None for r_ in rows):
                return [h_prev[u].new_zeros(G, d) for u in us]
            out = self.msg[l](torch.cat([r_ for r_ in rows if r_ is not None], 0))
            res, k = [], 0
            for r_ in rows:
                if r_ is None:
                    res.append(out.new_zeros(G, d))
                else:
                    res.append(out[k * G:(k + 1) * G])
                    k += 1
            return res
        res = []
        for u in us:
            ps = self._parents(u)
            if not ps:
                res.append(h_prev[u].new_zeros(G, d))
            elif len(ps) == 1:
                res.append(h_cur[ps[0]])                       # softmax over one parent == 1
            else:
                hu = torch.stack([h_cur[p] for p in ps], 1)    # [G,P,d]
                logits = (h_prev[u] @ self.att_w1[l])[:, None] + hu @ self.att_w2[l]
                alpha = torch.softmax(logits, dim=1)
                res.append((alpha[..., None] * hu).sum(1))
        return res

    def forward(self, x):
        G = x.shape[0]
        h_prev = {}
        for u in self.nodes:
            if u[0] == "in":
                h_prev[u] = self.inp(x)
            else:
                h_prev[u] = self.emb.weight[DAGNN_VOCAB.index(u[0])].expand(G, -1)
        R_nodes = self.node_levels[-1]
        per_layer = [torch.cat([h_prev[u] for u in R_nodes], 0)]
        for l in range(self.L):
            h_cur = {}
            for lev in self.node_levels:
                m = self._messages(l, lev, h_prev, h_cur, G)
                hp = torch.cat([h_prev[u] for u in lev], 0)
                out = self.grus[l](hp, torch.cat(m, 0))
                for k, u in enumerate(lev):
                    h_cur[u] = out[k * G:(k + 1) * G]
            per_layer.append(torch.cat([h_cur[u] for u in R_nodes], 0))
            h_prev = h_cur
        z = torch.cat(per_layer, -1)
        return self.read2(torch.relu(self.read1(z)))[:, 0].reshape(len(self.S.roots), G)


class SharedNet(_Base):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.mlp = _OperationMLP(1 + len(NODE_TYPES), 1, cfg.width, 0.0, cfg.n_hidden)
        self.register_buffer("counts", torch.as_tensor(self.S.counts), persistent=False)

    def forward(self, x):
        G, Pn = x.shape[0], self.counts.shape[0]
        feat = torch.cat([x[None].expand(Pn, G, 1), self.counts[:, None, :].expand(Pn, G, -1)], -1)
        return self.mlp(feat.reshape(Pn * G, -1))[:, 0].reshape(Pn, G)


def build_net(method: str, cfg: NetConfig) -> torch.nn.Module:
    if method == "ofml":
        return OFMLNet(cfg)
    if method == "sharedcell":
        return SharedCellNet(cfg)
    if method == "dagnn":
        return DAGNNNet(cfg, "attention")
    if method == "dagnn_cat":
        return DAGNNNet(cfg, "concat")
    if method == "shared":
        return SharedNet(cfg)
    raise ValueError(method)


def n_params(method: str, cfg: NetConfig) -> int:
    """Trainable parameters of ONE member (every parameter receives gradient in this benchmark)."""
    return int(sum(p.numel() for p in build_net(method, cfg).parameters()))


# --------------------------------------------------------------------------- ensemble trainer
class Ensemble:
    """Seed-only ensemble of ``n_members`` nets of one method; ``fit`` = cold refit from the same initialisation."""

    N_WARMUP = 3

    def __init__(self, method: str, cfg: NetConfig, seed_base: int, device="cpu", cuda_graph: bool = True,
                 y_mean: float = 0.0, y_std: float = 1.0):
        from torch.func import stack_module_state
        self.method, self.cfg, self.seed_base = method, cfg, int(seed_base)
        self.device = torch.device(device)
        self.cuda_graph = cuda_graph
        self.y_mean, self.y_std = float(y_mean), float(y_std)
        nets = []
        for m in range(cfg.n_members):
            torch.manual_seed(self.seed_base + m)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed_base + m)
            nets.append(build_net(method, cfg).to(self.device).eval())
        p0, b0 = stack_module_state(nets)
        self.init_params = {k: v.detach().clone() for k, v in p0.items()}
        self.buffers = {k: v.detach() for k, v in b0.items()}
        self.base = nets[0]
        self.x = torch.as_tensor(P.WX, dtype=torch.float32, device=self.device)[:, None]
        self.params = None

    def _fwd(self):
        from torch.func import functional_call, vmap
        base, x = self.base, self.x

        def f(p, b):
            return functional_call(base, (p, b), (x,))
        return vmap(f, in_dims=(0, 0))

    def fit(self, cells: np.ndarray, y: np.ndarray) -> "Ensemble":
        cfg = self.cfg
        dev = self.device
        params = {k: v.clone().requires_grad_(True) for k, v in self.init_params.items()}
        buffers = self.buffers
        fv = self._fwd()
        idx = torch.as_tensor(np.asarray(cells, dtype=np.int64), device=dev)
        ys = torch.as_tensor((np.asarray(y, float) - self.y_mean) / self.y_std, dtype=torch.float32, device=dev)
        use_graph = self.cuda_graph and dev.type == "cuda" and cfg.steps > self.N_WARMUP
        opt = torch.optim.Adam(list(params.values()), lr=cfg.lr, weight_decay=cfg.wd, capturable=use_graph)

        def step():
            tab = fv(params, buffers).reshape(cfg.n_members, -1)       # [M, 17*G]
            pred = tab[:, idx]
            loss = ((pred - ys) ** 2).mean(1).sum()                     # sum of member MSEs
            loss.backward()
            opt.step()

        if not use_graph:
            for _ in range(cfg.steps):
                opt.zero_grad(set_to_none=True)
                step()
        else:
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(self.N_WARMUP):
                    opt.zero_grad(set_to_none=True)
                    step()
            torch.cuda.current_stream(dev).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                step()
            for _ in range(cfg.steps - self.N_WARMUP):
                graph.replay()
            torch.cuda.synchronize(dev)
            del graph
        self.params = {k: v.detach() for k, v in params.items()}
        return self

    def member_table(self) -> np.ndarray:
        """[M, 17, G] member predictions on the ORIGINAL scale (float64)."""
        with torch.no_grad():
            tab = self._fwd()(self.params, self.buffers)
        return tab.double().cpu().numpy() * self.y_std + self.y_mean

    def member_targets(self) -> np.ndarray:
        """[M, 448] member predictions at the target candidates (candidate order of p1_common)."""
        tab = self.member_table().reshape(self.cfg.n_members, -1)
        return tab[:, P.tgt_flat_cells()]


# --------------------------------------------------------------------------- member-batched functional forwards (fast path)
# The same nets, written with an explicit member axis (x [M, N, in], weights [M, out, in] from
# stack_module_state, so the parameters and their initialisation are those of the nn.Modules above).
# No vmap; the loss can be torch.compile'd.  Checked against the vmap path in tests/check_models.py.
def _lin(p, name, x):
    return torch.baddbmm(p[name + ".bias"].unsqueeze(1), x, p[name + ".weight"].transpose(1, 2))


def _mlp(p, prefix, x, n_hidden):
    nh = max(1, n_hidden)
    for j in range(nh + 1):
        x = _lin(p, f"{prefix}.net.{3 * j}", x)
        if j < nh:
            x = torch.relu(x)
    return x


def _split(out, us, G, vals):
    for k, u in enumerate(us):
        vals[u] = out[:, k * G:(k + 1) * G]


def f_ofml(p, x, cfg: NetConfig, S: Structure = STRUCT):
    M, G = x.shape[0], x.shape[1]
    s0 = _lin(p, "embed", x)
    vals = {}
    for lev in S.levels:
        for t in NODE_TYPES:
            us = [u for u in lev if u[0] == t]
            if not us:
                continue
            if ARITY[t] == 0:
                inp = torch.cat([s0] * len(us), 1)
            else:
                inp = torch.cat([torch.cat([vals[c] for c in u[1:]], -1) for u in us], 1)
            _split(_mlp(p, f"ops.{t}", inp, cfg.n_hidden), us, G, vals)
    z = torch.cat([vals[r] for r in S.roots], 1)
    return _mlp(p, "obs", z, cfg.n_hidden)[..., 0].reshape(M, len(S.roots), G)


def f_sharedcell(p, x, cfg: NetConfig, S: Structure = STRUCT):
    M, G = x.shape[0], x.shape[1]
    s0 = _lin(p, "embed", x)
    zero = torch.zeros_like(s0)
    E = p["type_emb.weight"]                                   # [M, 5, k]
    vals = {}
    for lev in S.levels:
        rows = []
        for u in lev:
            t = u[0]
            ins = [s0] if ARITY[t] == 0 else [vals[c] for c in u[1:]]
            if len(ins) == 1:
                ins.append(zero)
            e = E[:, NODE_TYPES.index(t)][:, None, :].expand(M, G, E.shape[-1])
            rows.append(torch.cat(ins + [e], -1))
        _split(_mlp(p, "cell", torch.cat(rows, 1), cfg.n_hidden), lev, G, vals)
    z = torch.cat([vals[r] for r in S.roots], 1)
    return _mlp(p, "obs", z, cfg.n_hidden)[..., 0].reshape(M, len(S.roots), G)


def _gru(p, l, x, h):
    gi = _lin2(p, f"grus.{l}", "ih", x)
    gh = _lin2(p, f"grus.{l}", "hh", h)
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return n + z * (h - n)


def _lin2(p, pre, kind, x):
    return torch.baddbmm(p[f"{pre}.bias_{kind}"].unsqueeze(1), x, p[f"{pre}.weight_{kind}"].transpose(1, 2))


def f_dagnn(p, x, cfg: NetConfig, aggregate: str, S: Structure = STRUCT):
    M, G = x.shape[0], x.shape[1]
    L = dagnn_dims(cfg)[0]
    IN = ("in",)
    node_levels = [[IN]] + [list(lev) for lev in S.levels] + [[("R", r_) for r_ in S.roots]]
    E = p["emb.weight"]                                        # [M, 6, d]
    d = E.shape[-1]
    h_prev = {IN: _lin(p, "inp", x)}
    for lev in node_levels[1:]:
        for u in lev:
            h_prev[u] = E[:, DAGNN_VOCAB.index(u[0])][:, None, :].expand(M, G, d)
    R_nodes = node_levels[-1]
    per_layer = [torch.cat([h_prev[u] for u in R_nodes], 1)]
    for l in range(L):
        h_cur = {}
        for lev in node_levels:
            pars = [DAGNNNet._parents(u) for u in lev]
            if aggregate == "concat":
                if not pars[0]:
                    m = x.new_zeros(M, G * len(lev), d)
                else:
                    rows = []
                    for ps in pars:
                        hs = [h_cur[q] for q in ps]
                        if len(hs) == 1:
                            hs.append(torch.zeros_like(hs[0]))
                        rows.append(torch.cat(hs, -1))
                    m = _lin(p, f"msg.{l}", torch.cat(rows, 1))
            else:
                ms = []
                w1, w2 = p["att_w1"][:, l], p["att_w2"][:, l]         # [M, d]
                for u, ps in zip(lev, pars):
                    if not ps:
                        ms.append(x.new_zeros(M, G, d))
                    elif len(ps) == 1:
                        ms.append(h_cur[ps[0]])
                    else:
                        hu = torch.stack([h_cur[q] for q in ps], 2)                      # [M,G,P,d]
                        logits = torch.einsum("mgd,md->mg", h_prev[u], w1)[..., None] \
                            + torch.einsum("mgpd,md->mgp", hu, w2)
                        a = torch.softmax(logits, dim=2)
                        ms.append((a[..., None] * hu).sum(2))
                m = torch.cat(ms, 1)
            hp = torch.cat([h_prev[u] for u in lev], 1)
            _split(_gru(p, l, hp, m), lev, G, h_cur)
        per_layer.append(torch.cat([h_cur[u] for u in R_nodes], 1))
        h_prev = h_cur
    z = torch.cat(per_layer, -1)
    return _lin(p, "read2", torch.relu(_lin(p, "read1", z)))[..., 0].reshape(M, len(S.roots), G)


_COUNTS: dict = {}


def f_shared(p, x, cfg: NetConfig, S: Structure = STRUCT):
    M, G = x.shape[0], x.shape[1]
    key = (str(x.device), x.dtype)
    if key not in _COUNTS:                 # created once per device (not inside a CUDA-graph capture)
        _COUNTS[key] = torch.as_tensor(S.counts, dtype=x.dtype, device=x.device)
    c = _COUNTS[key]
    Pn = c.shape[0]
    feat = torch.cat([x[:, None].expand(M, Pn, G, 1), c[None, :, None, :].expand(M, Pn, G, c.shape[1])], -1)
    return _mlp(p, "mlp", feat.reshape(M, Pn * G, -1), cfg.n_hidden)[..., 0].reshape(M, Pn, G)


def functional_forward(method: str, cfg: NetConfig):
    if method == "ofml":
        return lambda p, x: f_ofml(p, x, cfg)
    if method == "sharedcell":
        return lambda p, x: f_sharedcell(p, x, cfg)
    if method == "dagnn":
        return lambda p, x: f_dagnn(p, x, cfg, "attention")
    if method == "dagnn_cat":
        return lambda p, x: f_dagnn(p, x, cfg, "concat")
    if method == "shared":
        return lambda p, x: f_shared(p, x, cfg)
    raise ValueError(method)


class FastEnsemble(Ensemble):
    """Ensemble with the member-batched functional forward and a DENSE form of the row loss.

    With n_c rows at cell c (repeats allowed) and S_c / Q_c the sum of their (standardised) responses / squares,
        (1/n) sum_i (f(c_i) - y_i)^2 = (1/n) sum_c (n_c f_c^2 - 2 S_c f_c + Q_c),
    the same loss value and gradient as the per-row MSE (exact algebra; only float rounding differs).  It needs no
    gather (whose CUDA backward uses atomics and is not bit-reproducible) and its shapes do not change between
    rounds, so ``compile=True`` (torch.compile of the loss) compiles once.  ``n_max`` is kept for the interface."""

    def __init__(self, *a, n_max: int = 280, compile: bool = False, **k):
        k.setdefault("cuda_graph", True)
        super().__init__(*a, **k)
        self.n_max = int(n_max)
        self.fwd = functional_forward(self.method, self.cfg)
        self.xb = self.x[None].expand(self.cfg.n_members, -1, -1).contiguous()
        if self.method == "shared":
            with torch.no_grad():
                self.fwd(self.init_params, self.xb)      # caches the count features on the device
        M = self.cfg.n_members

        def loss_fn(params, w, s, q):
            f = self.fwd(params, self.xb).reshape(M, -1)               # [M, 17*G]
            return (w * f * f - 2.0 * s * f).sum() + q * M              # sum of the member MSEs
        self.loss_fn = torch.compile(loss_fn, dynamic=False) if compile else loss_fn

    def fit(self, cells, y):
        cfg, dev = self.cfg, self.device
        n = len(cells)
        NC = P.N_PROTOCOLS * P.G
        ys_ = (np.asarray(y, float) - self.y_mean) / self.y_std
        cells = np.asarray(cells, dtype=np.int64)
        w = np.bincount(cells, minlength=NC) / n
        s_ = np.bincount(cells, weights=ys_, minlength=NC) / n
        wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
        st = torch.as_tensor(s_, dtype=torch.float32, device=dev)
        qt = torch.tensor(float(np.mean(ys_ ** 2)), dtype=torch.float32, device=dev)
        params = {k: v.clone().requires_grad_(True) for k, v in self.init_params.items()}
        use_graph = self.cuda_graph and dev.type == "cuda" and cfg.steps > self.N_WARMUP
        opt = torch.optim.Adam(list(params.values()), lr=cfg.lr, weight_decay=cfg.wd, capturable=use_graph)

        def step():
            self.loss_fn(params, wt, st, qt).backward()
            opt.step()
        if not use_graph:
            for _ in range(cfg.steps):
                opt.zero_grad(set_to_none=True)
                step()
        else:
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(self.N_WARMUP):
                    opt.zero_grad(set_to_none=True)
                    step()
            torch.cuda.current_stream(dev).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                step()
            for _ in range(cfg.steps - self.N_WARMUP):
                graph.replay()
            torch.cuda.synchronize(dev)
            del graph
        self.params = {k: v.detach() for k, v in params.items()}
        return self

    def member_table(self):
        with torch.no_grad():
            tab = self.fwd(self.params, self.xb)
        return tab.double().cpu().numpy() * self.y_std + self.y_mean
