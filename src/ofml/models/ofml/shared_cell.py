"""OFML-SharedCell-Ens: OFML with the per-operation-type parameters removed (Exp. 1 ablation).

Spec: REPORTS/EXP1_ORDER_AWARE_BASELINE_SPEC.md, Sec. 3.3.

    s_0 = Linear(2 -> d)                       (same as TerminalOnlyOFML.embed, d = 10)
    for each intermediate operation tau:   s <- f_theta([s, e_tau])     (ONE MLP shared by all types)
    y_hat = obs(s)                             (same observer MLP as OFML)

f_theta = _OperationMLP(d + k, d, h, dropout=0, n_hidden=2) and obs = _OperationMLP(d, 1, h, 0, 2) are
the exact module class OFML uses (same activations / init); e_tau = Embedding(len(VOCAB), k).
Composition is unbounded (no tanh), as in OFML-Ens. Training / prediction API and the
across-member variance definition are those of TerminalOnlyOFML (ensemble of n_members seeds
config.seed + j, trained simultaneously via vmap(functional_call); see ofml.models.dagnn).

Batching: by default all rows are processed in one padded batch of operation sequences; rows
whose protocol is shorter keep their state unchanged (torch.where) at padded positions, which is
mathematically identical to a per-protocol forward (``batching="protocol"``).
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from ..dagnn import VOCAB, _VmapEnsemble
from .operation_models import _OperationMLP


@dataclasses.dataclass(frozen=True)
class SharedCellConfig:
    latent_dim: int = 10       # d (same as OFML-Ens)
    embed_dim: int = 4         # k (type-embedding size)
    hidden_dim: int = 41       # h (param-matched)
    n_hidden: int = 2
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    steps: int = 400
    seed: int = 0
    batching: str = "padded"   # "padded" | "protocol"


class SharedCellNet(torch.nn.Module):
    def __init__(self, cfg: SharedCellConfig) -> None:
        super().__init__()
        d, k, h = cfg.latent_dim, cfg.embed_dim, cfg.hidden_dim
        self.embed = torch.nn.Linear(2, d)
        self.type_emb = torch.nn.Embedding(len(VOCAB), k)
        self.cell = _OperationMLP(d + k, d, h, 0.0, cfg.n_hidden)
        self.obs = _OperationMLP(d, 1, h, 0.0, cfg.n_hidden)

    def forward(self, x: torch.Tensor, op_onehot: torch.Tensor, op_mask: torch.Tensor) -> torch.Tensor:
        """x [N,2]; op_onehot [N,K,V] one-hot intermediate-op types; op_mask [N,K] bool (True = real op).
        Embedding lookup = one-hot @ E (bit-identical; deterministic and CUDA-graph capturable)."""
        s = self.embed(x)
        e_all = op_onehot @ self.type_emb.weight                                       # [N,K,k]
        for t in range(op_onehot.shape[1]):
            s_new = self.cell(torch.cat([s, e_all[:, t]], dim=-1))
            s = torch.where(op_mask[:, t:t + 1], s_new, s)
        return self.obs(s)[:, 0]


def _ops(row: Mapping) -> list[str]:
    return str(row["steps"]).split()[:-1]   # last step is the observer O


def make_batches(rows: Sequence[Mapping], x: np.ndarray, mode: str, device) -> list[tuple]:
    if mode == "padded":
        groups = {"all": list(range(len(rows)))}
    else:
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[str(r["steps"])].append(i)
    out = []
    pad = VOCAB.index("O")
    for idx in groups.values():
        K = max(len(_ops(rows[i])) for i in idx)
        types = np.full((len(idx), K), pad, dtype=np.int64)
        mask = np.zeros((len(idx), K), dtype=bool)
        for a, i in enumerate(idx):
            ops = _ops(rows[i])
            types[a, :len(ops)] = [VOCAB.index(o) for o in ops]
            mask[a, :len(ops)] = True
        idx = np.array(idx)
        out.append((idx, (torch.as_tensor(x[idx], dtype=torch.float32, device=device),
                          torch.as_tensor(np.eye(len(VOCAB), dtype=np.float32)[types], device=device),
                          torch.as_tensor(mask, device=device)), {}))
    return out


class SharedCellOFML(_VmapEnsemble):
    """Seed-only deep ensemble of the shared-cell composed surrogate (API of TerminalOnlyOFML)."""

    def __init__(self, config: SharedCellConfig | None = None, n_members: int = 5, device: str = "cpu",
                 vectorize: bool = True, cuda_graph: bool = True) -> None:
        super().__init__(n_members=n_members, device=device, vectorize=vectorize, cuda_graph=cuda_graph)
        self.cfg = config or SharedCellConfig()

    @property
    def config(self) -> SharedCellConfig:
        return self.cfg

    def _build_net(self, seed: int) -> torch.nn.Module:
        return SharedCellNet(self.cfg)

    def _batches(self, rows, x):
        return make_batches(rows, x, self.cfg.batching, self.device)
