#!/usr/bin/env python3
"""PB-W16U active learning with OFML-Ens (typed-MLP deep ensemble, terminal-only), same teachers, initial
design, candidates, policies and record schema as ``w16u_gpn.py``.

Model (exp2_ofml_ens.py generalised to 2 layer-1 types x 8 layer-2 types):
    s0 = Linear(1 -> d)(x)            shared input embedding (x = uniform grid value in [0, 4])
    s1 = L1_i(s0)  (d -> d MLP)       i in {A, B}
    s2 = C_j(s1)   (d -> d MLP)       j in {C1..C8}
    y  = R(s2)     (d -> 1 MLP)       shared terminal observer
Members are initialised from the paper's ``_OperationMLP`` modules under ``_set_torch_seed(base + m)`` (base =
the teacher's inference seed), y standardised on the training rows, full-batch Adam with L2 weight decay, a
cold refit from the same initialisation every round. Predictive mean / variance = across-member mean /
population variance (as ``TerminalOnlyOFML.predict_distribution``); IVR uses the sample covariance of the
member predictions (``np.cov`` + 1e-8 I, as exp2 / ``terminal_only_lib.acquire('ivr')``).

All trajectories (reps x policies) advance in lock-step: at every round all of them have the same number of
observations, so all ensembles of a round are trained as ONE batched model (K = B x M independent nets; the loss
is a sum of per-net MSEs and Adam / weight decay are elementwise, so this equals independent fits in exact
arithmetic). CAUTION (found 2026-11-04): in float32 the batched kernels round differently for different batch
sizes, and the adaptive policies amplify these ulp-level differences (rep 0: chosen points differ from round 3,
NRMSE by up to 0.19 between B=4 and B=64 on the GPU). Production therefore runs every trajectory in its own batch
(``--policies <one> --reps <one>``, CPU, 1 thread; slurm/ens_cpu_rep.sbatch), which is bit-reproducible.
Outputs per (rep, policy): ``<out>/ens/rep_XXXX_<policy>.json`` and ``.npz`` (mean / var / PIT per round).
Checkpoint: ``<out>/ckpt/ens_<tag>.json`` after every round (observed lists, records, RNG states).

Optional uncertainty techniques (2026-11-08, config keys; the defaults leave the code path and the results
bit-identical to the earlier runs, and the keys are absent from the stored config unless set):
  * ``prior_scale`` (float, default 0.0) -- randomized prior functions (Osband, Aslanides & Cassirer, NeurIPS
    2018). Member m's output (standardised y units) is f_m(x) + prior_scale * p_m(x), where p_m is a fixed,
    untrained net of the same typed architecture initialised by the same module code under the seed
    ``seed_base + PRIOR_SEED_OFFSET + m`` (PRIOR_SEED_OFFSET = 10**9; trainable members use ``seed_base + m``
    with seed_base = teacher inference seed ~5.4e7, so the two seed sets never meet for any rep < 10**8).
    p_m has no gradients and is not in the optimiser; it is used both in the loss and in prediction.
  * ``anchor`` (bool, default False) -- anchored ensembles / randomized MAP sampling (Pearce, Leibfried &
    Brintrup, AISTATS 2020): Adam(weight_decay=0) plus the explicit gradient term wd * (theta - theta_0), theta_0
    = the member's own initial parameters, instead of the coupled L2 term wd * theta of Adam(weight_decay=wd).
    Applied to every trainable tensor (as Adam's weight decay is); tensors that receive no gradient in a fit
    (types without training points under ``packed``) are skipped, as Adam skips them.

Optional initialisation (2026-11-13, config key ``init``; absent or "default" leaves the code path and the results
bit-identical to the earlier runs, and the key is absent from the stored config unless set):
  * ``init`` = "he" -- the modules are built under the same seed as in the default, and then every weight is redrawn
    from the same generator: embedding weight ~ N(0, 1); every hidden Linear of every typed MLP and of the observer
    ~ N(0, 2 / fan_in); the last Linear of each MLP ~ N(0, 1 / fan_in); all biases 0. Identical to
    ``ens_train_diag.HeInit`` (the class used by the pre-stage probes of 2026-11-13).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w16u_common as C  # noqa: E402

OFML_SRC = HERE.parent / "src"
BASELINE_CONFIG = dict(latent_dim=5, hidden=32, n_hidden=2, lr=1e-2, wd=1e-4, steps=400, n_members=5)
CONFIG_KEYS = tuple(BASELINE_CONFIG) + ("packed", "prior_scale", "anchor", "init")
INIT_KINDS = ("default", "he")
PRIOR_SEED_OFFSET = 10 ** 9      # prior net m of a seed base: seed_base + PRIOR_SEED_OFFSET + m
L1_TYPES = ("A", "B")
L2_TYPES = tuple(f"C{j + 1}" for j in range(C.N_L2))


class EnsembleInit:
    """Cache of member initial parameters per seed base (built once with the original modules)."""

    def __init__(self, cfg: dict):
        import torch
        sys.path.insert(0, str(OFML_SRC))
        from ofml.models.ofml.operation_models import _OperationMLP, _set_torch_seed
        self.torch, self.cfg = torch, cfg
        self._mlp, self._seed = _OperationMLP, _set_torch_seed
        self.cache: dict[int, list] = {}
        self.prior_cache: dict[int, list] = {}

    def member(self, seed: int) -> dict:
        torch = self.torch
        d, h, nh = self.cfg["latent_dim"], self.cfg["hidden"], self.cfg["n_hidden"]
        self._seed(seed)
        embed = torch.nn.Linear(1, d)
        ops = {op: self._mlp(d, d, h, 0.0, nh) for op in L1_TYPES + L2_TYPES}
        obs = self._mlp(d, 1, h, 0.0, nh)
        kind = self.cfg.get("init", "default")
        if kind not in INIT_KINDS:
            raise ValueError(f"init must be one of {INIT_KINDS}, got {kind!r}")
        if kind == "he":      # module docstring; the default branch draws nothing more from the generator
            with torch.no_grad():
                torch.nn.init.normal_(embed.weight, 0.0, 1.0)
                embed.bias.zero_()
                for mod in list(ops.values()) + [obs]:
                    lins = [l for l in mod.net if isinstance(l, torch.nn.Linear)]
                    for i, l in enumerate(lins):
                        fan_in = l.weight.shape[1]
                        std = (2.0 / fan_in) ** 0.5 if i < len(lins) - 1 else (1.0 / fan_in) ** 0.5
                        torch.nn.init.normal_(l.weight, 0.0, std)
                        l.bias.zero_()

        def layers(mod):
            return [(l.weight.detach().T.clone(), l.bias.detach().clone())
                    for l in mod.net if isinstance(l, torch.nn.Linear)]
        return dict(We=embed.weight.detach().T.clone(), be=embed.bias.detach().clone(),
                    ops={k: layers(v) for k, v in ops.items()}, obs=layers(obs))

    def members(self, seed_base: int) -> list:
        if seed_base not in self.cache:
            self.cache[seed_base] = [self.member(seed_base + m) for m in range(self.cfg["n_members"])]
        return self.cache[seed_base]

    def prior_members(self, seed_base: int) -> list:
        """Fixed random prior nets (randomized prior functions): same architecture, disjoint seeds."""
        if seed_base not in self.prior_cache:
            self.prior_cache[seed_base] = [self.member(seed_base + PRIOR_SEED_OFFSET + m)
                                           for m in range(self.cfg["n_members"])]
        return self.prior_cache[seed_base]


class BatchedEnsemble:
    """K = B*M nets (dataset-major, member-minor), trained jointly."""

    def __init__(self, init: EnsembleInit, seed_bases: list[int], device, packed: bool | None = None,
                 _prior: bool = False):
        torch = init.torch
        self.torch, self.device = torch, device
        # packed (v2, 2026-11-04): every typed MLP is applied only to the points of its own type (gather /
        # scatter) instead of to all points and then selected. Same function and gradients; float32 rounding
        # differs, so v1 (packed=False) results are reproduced only with the old path.
        self.packed = bool(init.cfg.get("packed", False)) if packed is None else bool(packed)
        self.M, self.B = init.cfg["n_members"], len(seed_bases)
        mem = [mm for sb in seed_bases for mm in (init.prior_members(sb) if _prior else init.members(sb))]
        self.params = []

        def leaf(t):
            t = t.to(device=device, dtype=torch.float32).contiguous().requires_grad_(True)
            self.params.append(t)
            return t

        self.We = leaf(torch.stack([m["We"] for m in mem]))           # [K,1,d]
        self.be = leaf(torch.stack([m["be"] for m in mem]))           # [K,d]
        n_layers = len(mem[0]["obs"])

        def group(types):
            if self.packed:   # one leaf per (type, layer): no strided slices of a stacked tensor in backward
                return [[(leaf(torch.stack([m["ops"][op][li][0] for m in mem])),
                          leaf(torch.stack([m["ops"][op][li][1] for m in mem]))) for li in range(n_layers)]
                        for op in types]
            return [(leaf(torch.stack([torch.stack([m["ops"][op][li][0] for op in types]) for m in mem])),
                     leaf(torch.stack([torch.stack([m["ops"][op][li][1] for op in types]) for m in mem])))
                    for li in range(n_layers)]
        self.A = group(L1_TYPES)
        self.Cc = group(L2_TYPES)
        self.R = [(leaf(torch.stack([m["obs"][li][0] for m in mem])), leaf(torch.stack([m["obs"][li][1] for m in mem])))
                  for li in range(n_layers)]
        # optional techniques (module docstring); off by default
        self.prior_scale = 0.0 if _prior else float(init.cfg.get("prior_scale", 0.0) or 0.0)
        self.anchor = False if _prior else bool(init.cfg.get("anchor", False))
        self.prior_net = None
        if _prior:
            for t in self.params:
                t.requires_grad_(False)
        elif self.prior_scale > 0.0:
            self.prior_net = BatchedEnsemble(init, seed_bases, device, self.packed, _prior=True)

    @staticmethod
    def _mlp_group(torch, s, layers):
        h = s.unsqueeze(1)                       # [K,1,N,d]
        for li, (W, b) in enumerate(layers):     # W [K,T,in,out]
            h = torch.matmul(h, W) + b.unsqueeze(2)
            if li < len(layers) - 1:
                h = torch.relu(h)
        return h                                 # [K,T,N,out]

    def _tables(self, idx, n_types: int):
        """Per type t: positions (K, L_t) of the points of type t (padding -> dummy slot N)."""
        torch = self.torch
        K, N = idx.shape
        out = []
        for t in range(n_types):
            m = idx == t
            cnt = m.sum(1)
            L = int(cnt.max().item())
            if L == 0:
                continue
            order = torch.argsort((~m).to(torch.int8), dim=1, stable=True)[:, :L]
            valid = torch.arange(L, device=idx.device)[None, :] < cnt[:, None]
            out.append((t, torch.where(valid, order, torch.full_like(order, N))))
        return out

    def _typed_packed(self, s, layers, tables):
        torch = self.torch
        K, N, d = s.shape
        s_pad = torch.cat([s, s.new_zeros(K, 1, d)], dim=1)
        dout = layers[0][-1][0].shape[-1]
        out = s.new_zeros(K, N + 1, dout)
        for t, sel in tables:
            h = s_pad.gather(1, sel[..., None].expand(-1, -1, d))
            for li, (W, b) in enumerate(layers[t]):
                h = torch.bmm(h, W) + b.unsqueeze(1)
                if li < len(layers[t]) - 1:
                    h = torch.relu(h)
            out = out.scatter(1, sel[..., None].expand(-1, -1, dout), h)
        return out[:, :N]

    def forward(self, x, a_idx, c_idx, tables=None):
        torch = self.torch
        if self.packed:
            ta, tc = tables if tables is not None else (self._tables(a_idx, len(L1_TYPES)),
                                                         self._tables(c_idx, len(L2_TYPES)))
            s = x.unsqueeze(-1) * self.We + self.be.unsqueeze(1)
            s = self._typed_packed(s, self.A, ta)
            s = self._typed_packed(s, self.Cc, tc)
            h = s
            for li, (W, b) in enumerate(self.R):
                h = torch.bmm(h, W) + b.unsqueeze(1)
                if li < len(self.R) - 1:
                    h = torch.relu(h)
            return h[..., 0]
        s = x.unsqueeze(-1) * self.We + self.be.unsqueeze(1)                          # [K,N,d]
        ha = self._mlp_group(torch, s, self.A)
        d = ha.shape[-1]
        s = torch.take_along_dim(ha, a_idx[:, None, :, None].expand(-1, 1, -1, d), dim=1)[:, 0]
        hc = self._mlp_group(torch, s, self.Cc)
        s = torch.take_along_dim(hc, c_idx[:, None, :, None].expand(-1, 1, -1, d), dim=1)[:, 0]
        h = s
        for li, (W, b) in enumerate(self.R):
            h = torch.bmm(h, W) + b.unsqueeze(1)
            if li < len(self.R) - 1:
                h = torch.relu(h)
        return h[..., 0]

    def _tensors(self, flat_kn: np.ndarray):
        torch = self.torch
        p = flat_kn // C.G
        x = torch.as_tensor(C.WARPED_X[flat_kn % C.G], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(p // C.N_L2, dtype=torch.long, device=self.device)
        c = torch.as_tensor(p % C.N_L2, dtype=torch.long, device=self.device)
        return x, a, c

    def fit(self, flat_bn: np.ndarray, y_bn: np.ndarray, steps: int, lr: float, wd: float,
            mask_bn: np.ndarray | None = None, anchor_params: list | None = None) -> None:
        """``mask_bn`` (HPO only): 1 for real rows, 0 for padding, so datasets of different size share one
        batch; each net's loss is the mean over its real rows (same gradients as an unpadded fit)."""
        torch = self.torch
        if mask_bn is None:
            y_mean = y_bn.mean(1)
            y_std = y_bn.std(1)
        else:
            cnt = mask_bn.sum(1)
            y_mean = (y_bn * mask_bn).sum(1) / cnt
            y_std = np.sqrt((((y_bn - y_mean[:, None]) ** 2) * mask_bn).sum(1) / cnt)
        self.y_mean, self.y_std = y_mean, np.where(y_std > 0, y_std, 1.0)
        ys = (y_bn - self.y_mean[:, None]) / self.y_std[:, None]
        rep = lambda v: np.repeat(v, self.M, axis=0)  # noqa: E731
        x, a, c = self._tensors(rep(flat_bn))
        y = torch.as_tensor(rep(ys), dtype=torch.float32, device=self.device)
        mk = None if mask_bn is None else torch.as_tensor(rep(mask_bn), dtype=torch.float32, device=self.device)
        tabs = (self._tables(a, len(L1_TYPES)), self._tables(c, len(L2_TYPES))) if self.packed else None
        prior_tr = None
        if self.prior_net is not None:   # fixed prior outputs on the training rows (constant during the fit)
            with torch.no_grad():
                prior_tr = self.prior_scale * self.prior_net.forward(x, a, c, tabs)
        anchor0 = None
        if self.anchor:                  # theta_0 = this member's initialisation (``anchor_params``: test only)
            anchor0 = [p.detach().clone() for p in self.params] if anchor_params is None else anchor_params
        # v2 (packed) also uses the fused Adam kernel (same update rule, different rounding); v1 keeps the default.
        opt = torch.optim.Adam(self.params, lr=lr, weight_decay=(0.0 if self.anchor else wd),
                               **({"fused": True} if self.packed else {}))
        for _ in range(steps):
            opt.zero_grad()
            out = self.forward(x, a, c, tabs)
            if prior_tr is not None:
                out = out + prior_tr
            if mk is None:
                loss = ((out - y) ** 2).mean(1).sum()
            else:
                loss = (((out - y) ** 2 * mk).sum(1) / mk.sum(1)).sum()
            loss.backward()
            if anchor0 is not None:
                with torch.no_grad():
                    for p, p0 in zip(self.params, anchor0):
                        if p.grad is not None:
                            p.grad.add_(p - p0, alpha=wd)
            opt.step()

    def predict_members(self, flat_p: np.ndarray) -> np.ndarray:
        torch = self.torch
        K = self.B * self.M
        x, a, c = self._tensors(np.broadcast_to(flat_p, (K, len(flat_p))).copy())
        with torch.no_grad():
            out = self.forward(x, a, c)
            if self.prior_net is not None:
                out = out + self.prior_scale * self.prior_net.forward(x, a, c)
            out = out.double().cpu().numpy()
        out = out.reshape(self.B, self.M, -1)
        return out * self.y_std[:, None, None] + self.y_mean[:, None, None]


def fit_predict(init: EnsembleInit, cfg: dict, seed_bases: list[int], flat_bn: np.ndarray, y_bn: np.ndarray,
                device) -> np.ndarray:
    """B ensembles on B training sets of equal size -> member predictions at all 1024 candidates [B,M,1024]."""
    ens = BatchedEnsemble(init, seed_bases, device)
    ens.fit(np.atleast_2d(flat_bn), np.atleast_2d(y_bn), cfg["steps"], cfg["lr"], cfg["wd"])
    return ens.predict_members(np.arange(C.N_CAND))


def get_device(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def summarise(S: np.ndarray, want_cov: bool):
    mean = S.mean(0)
    var = S.var(0)                      # population variance (paper)
    cov = (np.cov(S.T) + 1e-8 * np.eye(S.shape[1])) if want_cov else None
    return mean, var, cov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0-15")
    ap.add_argument("--policies", default=",".join(C.POLICIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="", help="JSON file with the frozen Ens config (default: exp2 baseline)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="", help="checkpoint name (default from reps/policies)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--stop-after-round", type=int, default=-1, help="test only")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--model-dir", default="ens", help="result sub-directory (ens = v1 config, ens_v2 = v2 config)")
    a = ap.parse_args()
    import torch
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    cfg = dict(BASELINE_CONFIG)
    if a.config:
        cfg.update({k: v for k, v in json.loads(Path(a.config).read_text()).items() if k in CONFIG_KEYS})
    cfg["latent_dim"], cfg["hidden"], cfg["n_hidden"], cfg["steps"], cfg["n_members"] = (
        int(cfg["latent_dim"]), int(cfg["hidden"]), int(cfg["n_hidden"]), int(cfg["steps"]), int(cfg["n_members"]))
    device = get_device(a.device)
    out_dir = Path(a.out)
    reps = C.parse_range(a.reps)
    pols = a.policies.split(",")
    trajs = [(r, p) for r in reps for p in pols
             if not (a.skip_existing and C.result_done(C.result_path(out_dir, a.model_dir, r, p)))]
    print(f"OFML-Ens W16U: {len(trajs)} trajectories, cfg={cfg}, device={device}", flush=True)
    if not trajs:
        print("W16U_ENS_DONE", flush=True)
        return
    tag = a.tag or f"r{a.reps}_p{'-'.join(pols)}".replace(",", "_")
    ck = out_dir / "ckpt" / f"{a.model_dir}_{tag}.json"
    teachers = {r: C.load_teacher(r) for r in reps}
    init = EnsembleInit(cfg)
    B = len(trajs)
    seed_bases = [C.teacher_seeds(r)[2] for r, _ in trajs]
    state = dict(observed=[list(C.INITIAL_FLAT) for _ in trajs], rec=[{k: [] for k in REC_KEYS} for _ in trajs],
                 rng=[C.policy_rng(r, p).bit_generator.state for r, p in trajs], trajs=[list(t) for t in trajs])
    arrs = [dict(mean=[], var=[], pit=[]) for _ in trajs]
    if ck.exists():
        z = json.loads(ck.read_text())
        if z["trajs"] != state["trajs"] or z["cfg"] != cfg:
            raise RuntimeError(f"{ck} belongs to another run")
        state = z["state"]
        az = np.load(ck.with_suffix(".npz"))
        for b in range(B):
            for k in arrs[b]:
                arrs[b][k] = list(az[f"{k}_{b}"])
        print(f"[ckpt] resume at n={len(state['observed'][0])}", flush=True)
    rngs = []
    for st in state["rng"]:
        g = np.random.default_rng()
        g.bit_generator.state = st
        rngs.append(g)
    t_start = time.time()
    while True:
        n = len(state["observed"][0])
        t = n - len(C.INITIAL_FLAT)
        rt = time.time()
        flat = np.array(state["observed"], dtype=np.int64)                         # [B,n]
        y = np.stack([teachers[r][0][flat[b]] + teachers[r][1][flat[b]] for b, (r, _) in enumerate(trajs)])
        S_all = fit_predict(init, cfg, seed_bases, flat, y, device)                 # [B,M,1024]
        t_fit = time.time() - rt
        for b, (r, p) in enumerate(trajs):
            truth = teachers[r][0]
            mean, var, cov = summarise(S_all[b], want_cov=(p == "ivr" and n < C.T_TOTAL))
            pit = C.gaussian_pit(mean, var, truth)
            m = C.round_metrics(mean, var, pit, truth)
            rec = state["rec"][b]
            rec["n_obs"].append(n)
            for k in METRIC_KEYS:
                rec[k].append(m[k])
            rec["fit_seconds"].append(t_fit / B)
            arrs[b]["mean"].append(mean.astype(np.float32))
            arrs[b]["var"].append(var.astype(np.float32))
            arrs[b]["pit"].append(pit.astype(np.float32))
            if n < C.T_TOTAL:
                pick = C.choose(p, state["observed"][b], var, cov, rngs[b])
                assert pick not in state["observed"][b]
                rec["chosen"].append(pick)
                state["observed"][b].append(pick)
        print(f"n={n} fit+predict {t_fit:.1f}s (B={B}) total {time.time() - t_start:.0f}s "
              f"mean NRMSE {np.mean([s['nrmse'][-1] for s in state['rec']]):.4f}", flush=True)
        if n >= C.T_TOTAL:
            break
        state["rng"] = [g.bit_generator.state for g in rngs]
        ck.parent.mkdir(parents=True, exist_ok=True)
        tmpz = ck.with_name("." + ck.stem + ".tmp.npz")
        np.savez(tmpz, **{f"{k}_{b}": np.asarray(arrs[b][k], dtype=np.float32) for b in range(B) for k in arrs[b]})
        os.replace(tmpz, ck.with_suffix(".npz"))
        C.atomic_write_json(ck, dict(trajs=state["trajs"], cfg=cfg, state=state))
        if a.stop_after_round >= 0 and t + 1 >= a.stop_after_round:
            print(f"[test] stop after round {t + 1}", flush=True)
            return
    for b, (r, p) in enumerate(trajs):
        truth, noise = teachers[r]
        res = dict(model="ens", rep=r, policy=p, status="ok", family=C.FAMILY, teacher_seed=C.teacher_seeds(r)[0],
                   teacher_sha256=C.sha256_arrays(truth, noise), initial=list(C.INITIAL_FLAT),
                   observed=state["observed"][b], config=dict(cfg, device=str(device), seed_base=seed_bases[b]),
                   policy_seconds=time.time() - t_start, rounds=state["rec"][b])
        rp = C.result_path(out_dir, a.model_dir, r, p)
        rp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(rp.with_suffix(".npz"), **{k: np.stack(v).astype(np.float32) for k, v in arrs[b].items()},
                 n_obs=np.asarray(state["rec"][b]["n_obs"]))
        C.atomic_write_json(rp, res)
    for f in (ck, ck.with_suffix(".npz")):
        if f.exists():
            f.unlink()
    print("W16U_ENS_DONE", flush=True)


METRIC_KEYS = ("nrmse", "rmse", "ise", "iv", "max_var", "max_abs_err", "max_var_zeroshot", "max_abs_err_zeroshot",
               "coverage", "coverage_zeroshot", "nrmse_by_protocol")
REC_KEYS = ("n_obs",) + METRIC_KEYS + ("chosen", "fit_seconds")


if __name__ == "__main__":
    main()
