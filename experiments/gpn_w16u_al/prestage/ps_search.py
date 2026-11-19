#!/usr/bin/env python3
"""Search programs of the pre-stage (REPORTS/ENS_PRESTAGE_PLAN.md 4.5, 4.6, rule 6, E0 step 3, E1, E4, E6).

The GP, the expected improvement and the candidate generator are the functions of
``runs/prestage_probe/bo_rehearsal.py`` (the program tested in P7), called unchanged.

sub-commands
    rehearse  --seeds 0-19            E0 step 3: this selection code on the synthetic problems of P7 (row 11)
    e1        --shard i --nshards n   E1: gate of the first --count Sobol points that are in Psi
    e1sum                             E1: tables, and the 32 initial points of E4 (runs/prestage/e4/init_points.json)
    worker    --camp e4|e6            one asynchronous worker of E4 / E6 (all state is in files, under a file lock)
    status    --camp e4|e6
    finalize  --camp e4|e6            end-of-search records, the set S (E4) or the 5 settings of E6

A campaign directory runs/prestage/<camp>/ holds trials/tNNNN/{spec.json, ckpt.json, result.json},
select/sNNNN.json (record of every selection), loo_49.json / loo_73.json, ext.json (rule 6 or the E6 extension).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import ps_common as P
from ps_common import C

sys.path.insert(0, str(P.EXP / "runs" / "prestage_probe"))
import bo_rehearsal as R  # noqa: E402

LEVELS7 = (7, 8, 7, 0, 0, 0, 2)
MAX_GATE = 32
NAMES = ("w", "L", "d", "eta", "K", "wd", "init")


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------- selection (shared)
def fit_with_pending(X, y, Pn, noise_sd):
    """Plan 4.5 step 1 (= the 'new' branch of bo_rehearsal.run_async). Returns the GP fitted to D alone (gp0),
    the GP refitted with the fantasised running trials (same kernel), and the reference value z-hat-min."""
    gp0, mu0, sd0 = R.fit_gp(X, y, noise_sd=noise_sd)
    if len(Pn):
        yf = np.append(y, gp0.predict(Pn) * sd0 + mu0)
        Xf = np.vstack([X, Pn])
        gp, mu, sd = R.fit_gp(Xf, yf, noise_sd=noise_sd, optimise=False, kernel=gp0.kernel_)
    else:
        Xf, yf, gp, mu, sd = X, y.copy(), gp0, mu0, sd0
    best = float(gp.predict(Xf).min())
    return dict(gp0=gp0, mu0=mu0, sd0=sd0, gp=gp, mu=mu, sd=sd, Xf=Xf, yf=yf, best=best)


def pick(gp, best, U, gate_fn, max_gate=MAX_GATE):
    """Plan 4.5 steps 3-4: candidates in decreasing EI, the first of at most ``max_gate`` that passes the gate."""
    m, s = gp.predict(U, return_std=True)
    e = R.ei(m, s, best)
    order = np.argsort(-e, kind="stable")
    rejected = []
    for i in order[:max_gate]:
        if gate_fn(int(i)):
            return int(i), e, rejected
        rejected.append(int(i))
    return None, e, rejected


def loo(X, y, noise_sd, gp=None):
    """Leave-one-out check with the kernel fixed at the fit on all of the data (as in bo_rehearsal.recommend).
    Returns (rho_LOO, sd of the standardised residuals, kernel description)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    if gp is None:
        gp, _, _ = R.fit_gp(X, y, noise_sd=noise_sd)
    mu, sd = y.mean(), max(y.std(), 1e-12)
    ys = (y - mu) / sd
    pred, res = [], []
    for i in range(len(X)):
        m = np.arange(len(X)) != i
        g2 = GaussianProcessRegressor(gp.kernel_, alpha=gp.alpha, optimizer=None).fit(X[m], ys[m])
        mi, si = g2.predict(X[i][None], return_std=True)
        pred.append(float(mi[0]))
        res.append(float((ys[i] - mi[0]) / np.sqrt(si[0] ** 2 + gp.alpha)))
    rho = float(np.corrcoef(pred, ys)[0, 1]) if len(X) > 2 else float("nan")
    return rho, float(np.std(res)), kernel_desc(gp)


def kernel_desc(gp) -> dict:
    k = gp.kernel_
    return dict(alpha2=float(k.k1.constant_value), ell=[float(v) for v in np.atleast_1d(k.k2.length_scale)])


def candidates_box(rng, X, y, lo, hi, n=4000):
    """bo_rehearsal.candidates for a box [lo, hi] (rule 6: the expanded part has coordinates outside [0, 1]).
    The draws are made in the same order; for lo = 0, hi = 1 the result equals bo_rehearsal.candidates."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    xb = X[np.argsort(y)[:5]]
    local = xb[rng.integers(0, len(xb), n // 2)] + rng.normal(0, 0.08, (n // 2, X.shape[1]))
    cand = np.clip(np.vstack([lo + (hi - lo) * rng.random((n, X.shape[1])), local]), lo, hi)
    for i, k in enumerate(R.LEVELS):
        if k:
            cand[:, i] = np.round(cand[:, i] * (k - 1)) / (k - 1)
    seen = {tuple(np.round(x, 6)) for x in X}
    return cand[[tuple(np.round(c, 6)) not in seen for c in cand]]


# ------------------------------------------------------------------------------------- E0 step 3
REH_MAX_GATE = MAX_GATE


def rehearse_one(seed: int, scen: str = "P16S1", W: int = 16, nm: float = 0.5) -> dict:
    """bo_rehearsal.run_async ('new', gate, W asynchronous workers) with the selection functions of this file."""
    prob = R.Problem(seed, scen)
    ranks = []
    rng = np.random.default_rng([seed, 21])
    X = R.sobol_points(seed, R.N_INIT, prob, True)
    y = prob.trial(X)
    pend, n_fallback = [], 0
    for _ in range(R.N_TOTAL - R.N_INIT):
        Pn = np.array(pend).reshape(len(pend), X.shape[1])
        g = fit_with_pending(X, y, Pn, prob.noise * nm)
        cand = R.candidates(rng, g["Xf"], g["yf"])
        i, _, rej = pick(g["gp"], g["best"], cand, lambda j: bool(prob.feasible(cand[j])[0]),
                         max_gate=REH_MAX_GATE or len(cand))
        ranks.append(len(rej) + 1 if i is not None else None)
        if i is None:                                     # plan 4.5 step 4: next Sobol point that passes the gate
            n_fallback += 1
            seen = {tuple(np.round(x, 6)) for x in g["Xf"]}
            x = next(u for u in R.sobol_points(seed, R.N_TOTAL * 4, prob, True) if tuple(np.round(u, 6)) not in seen)
        else:
            x = cand[i]
        pend.append(x)
        if len(pend) == W:
            x = pend.pop(int(rng.integers(len(pend))))
            X, y = np.vstack([X, x]), np.append(y, prob.trial(x[None]))
    if pend:
        X, y = np.vstack([X, np.array(pend)]), np.append(y, prob.trial(np.array(pend)))
    out = R.recommend(prob, X, y, True, nm)
    out["n_fallback"] = n_fallback
    out["ranks"] = ranks                                  # EI rank of the chosen candidate (None: Sobol fallback)
    return out


def cmd_rehearse(a) -> None:
    from multiprocessing import Pool
    R.LEVELS = tuple(int(v) for v in a.levels.split(","))
    R.N_INIT, R.N_TOTAL = a.n_init, a.n_total
    ref = json.loads((P.EXP / "runs" / "prestage_probe" / "reh_I_d7_n96_noise.json").read_text())["P16S1"]
    seeds = C.parse_range(a.seeds)
    # (a) the stored result is reproduced by the unchanged rehearsal program with these arguments
    kind, gate, q = R.METHODS["new_gate_nx05_a16"]
    prob = R.Problem(seeds[0], "P16S1")
    Xr, yr = R.run(prob, seeds[0], kind, gate, q)
    rep = R.recommend(prob, Xr, yr, gate, 0.5)
    st = ref[str(seeds[0])]["new_gate_nx05_a16"]
    same = all(abs(rep[k] - st[k]) < 1e-12 for k in ("obs", "pm", "conf5"))
    log(f"stored result of seed {seeds[0]} reproduced by bo_rehearsal with levels {R.LEVELS}: {same} "
        f"(conf5 {rep['conf5']:.6f} vs stored {st['conf5']:.6f})")
    with Pool(a.procs, initializer=_reh_init, initargs=(R.LEVELS, R.N_INIT, R.N_TOTAL, a.max_gate)) as pool:
        rows = pool.map(rehearse_one, seeds, chunksize=1)
    out = dict(levels=R.LEVELS, n_init=R.N_INIT, n_total=R.N_TOTAL, reproduced=bool(same), seeds=seeds, keys={})
    for k in ("conf5", "pm", "obs"):
        mine = np.array([r[k] for r in rows])
        theirs = np.array([ref[str(s)]["new_gate_nx05_a16"][k] for s in seeds])
        d = mine - theirs
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        out["keys"][k] = dict(mine=float(mine.mean()), stored=float(theirs.mean()), mean_diff=float(d.mean()), se=se,
                              n_identical=int((np.abs(d) < 1e-12).sum()),
                              passed=bool(abs(d.mean()) <= 2 * se))
        log(f"{k}: mean regret x1000 mine {1e3 * mine.mean():.2f} stored {1e3 * theirs.mean():.2f} "
            f"paired diff {1e3 * d.mean():+.3f} (SE {1e3 * se:.3f}) identical {out['keys'][k]['n_identical']}/{len(d)} "
            f"pass {out['keys'][k]['passed']}")
    out["n_fallback"] = [r["n_fallback"] for r in rows]
    out["max_gate"] = a.max_gate
    rk = [v for r in rows for v in r["ranks"]]
    got = np.array([v for v in rk if v is not None])
    out["ranks"] = dict(n_selections=len(rk), n_fallback=int(sum(v is None for v in rk)),
                        n_rank1=int((got == 1).sum()), n_rank_le32=int((got <= 32).sum()),
                        quantiles={str(q): float(np.quantile(got, q)) for q in (0.5, 0.75, 0.9, 0.99, 1.0)})
    log(f"max_gate {a.max_gate}: Sobol fallbacks {out['ranks']['n_fallback']}/{len(rk)}; EI rank of the chosen candidate "
        f"{out['ranks']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    log(f"PS_REHEARSE_DONE pass(conf5)={out['keys']['conf5']['passed']} reproduced={same}")


def _reh_init(levels, n_init, n_total, max_gate):
    global REH_MAX_GATE
    R.LEVELS, R.N_INIT, R.N_TOTAL, REH_MAX_GATE = levels, n_init, n_total, max_gate


# ------------------------------------------------------------------------------------------ Sobol points
_SOB = {}


def sobol_settings(sp: dict, tcpu: dict) -> list[dict]:
    """Settings of the points of the Sobol sequence (7 dims, scrambled, seed 0) that are in Psi, in sequence order;
    a setting that equals an earlier one after rounding is dropped."""
    key = json.dumps(sp, sort_keys=True)
    if key not in _SOB:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            U = qmc.Sobol(7, scramble=True, seed=0).random(8192)
        out, seen = [], set()
        for i, u in enumerate(U):
            psi = P.from_u(u, sp)
            if P.in_psi(psi, sp, tcpu):
                k = P.key_of(psi)
                if k not in seen:
                    seen.add(k)
                    out.append(dict(psi=P.norm_psi(psi), sobol_index=i))
        _SOB[key] = out
    return _SOB[key]


# ---------------------------------------------------------------------------------------------------- E1
def cmd_e1(a) -> None:
    import torch
    dev = torch.device(a.device)
    sp, tcpu = P.load_space(), P.load_tcpu()
    pts = sobol_settings(sp, tcpu)[a.offset:a.offset + a.count]
    order = sorted(range(len(pts)), key=lambda i: -P.t_est(pts[i]["psi"], tcpu))      # heavy ones spread over shards
    mine = [pts[i] for j, i in enumerate(order) if j % a.nshards == a.shard]
    log(f"E1 shard {a.shard}/{a.nshards}: {len(mine)} settings on {dev}")
    for q in mine:
        P.gate_cached(q["psi"], dev, log=log)
    log("PS_E1_DONE")


def cmd_e1sum(a) -> None:
    sp, tcpu = P.load_space(), P.load_tcpu()
    pts = sobol_settings(sp, tcpu)
    rows = []
    for q in pts:
        f = P.RUN / "gates" / f"{P.key_of(q['psi'])}.json"
        if not f.exists():
            break
        rows.append(dict(q, gate=P.read_json(f), u=P.to_u(q["psi"], sp).tolist()))
    n = len(rows)
    out = dict(n=n, space=sp, tau=P.TAU, overall={}, by_dim={}, fail_conditions={}, changes={})
    for t in P.TAUS:
        out["overall"][str(t)] = int(sum(r["gate"]["passed"][str(t)] for r in rows))
    bins = dict(w=[(v, v) for v in P.WIDTHS], L=[(v, v) for v in range(1, 9)], d=[(v, v) for v in range(2, 9)],
                eta=[(1e-4, 10 ** -3.5), (10 ** -3.5, 1e-3), (1e-3, 10 ** -2.5), (10 ** -2.5, 1.0001e-2)],
                K=[(400, 800), (800, 1600), (1600, 3200), (3200, 6401)],
                wd=[(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1.0001e-2)],
                init=[("default", "default"), ("he", "he")])
    for name, bs in bins.items():
        tab = []
        for lo, hi in bs:
            sel = [r for r in rows if (r["psi"][name] == lo if lo == hi else lo <= r["psi"][name] < hi)]
            if sel:
                tab.append(dict(lo=lo, hi=hi, n=len(sel),
                                passed={str(t): int(sum(r["gate"]["passed"][str(t)] for r in sel)) for t in P.TAUS}))
        out["by_dim"][name] = tab
    for t in P.TAUS:
        fc = {"finite": 0, "kappa": 0, "nrmse": 0, "kappa_and_nrmse": 0}
        for r in rows:
            g = r["gate"]
            if g["passed"][str(t)]:
                continue
            if not g["finite"]:
                fc["finite"] += 1
            else:
                kb, nb = g["kappa_mean"] > t, not g["cond_nrmse"]
                fc["kappa_and_nrmse" if kb and nb else ("kappa" if kb else "nrmse")] += 1
        out["fail_conditions"][str(t)] = fc
    out["changes"] = {"1.0_vs_1.25": int(sum(r["gate"]["passed"]["1.0"] != r["gate"]["passed"]["1.25"] for r in rows)),
                      "1.25_vs_1.5": int(sum(r["gate"]["passed"]["1.25"] != r["gate"]["passed"]["1.5"] for r in rows))}
    out["gate_seconds"] = dict(mean=float(np.mean([r["gate"]["seconds"] for r in rows])),
                               max=float(np.max([r["gate"]["seconds"] for r in rows])))
    passing = [r for r in rows if r["gate"]["passed"][str(P.TAU)]]
    out["n_pass_tau"] = len(passing)
    (P.RUN / "e1").mkdir(parents=True, exist_ok=True)
    (P.RUN / "e1" / "summary.json").write_text(json.dumps(out, indent=1))
    (P.RUN / "e1" / "gates_table.json").write_text(json.dumps(
        [dict(psi=r["psi"], sobol_index=r["sobol_index"], kappa=r["gate"]["kappa_mean"], nrmse=r["gate"]["nrmse_mean"],
              cov95=float(np.mean(r["gate"]["cov95"])), finite=r["gate"]["finite"], passed=r["gate"]["passed"],
              seconds=r["gate"]["seconds"]) for r in rows], indent=0))
    log(f"E1: {n} settings gated; passed {out['overall']} ; fail conditions {out['fail_conditions']}; "
        f"changes {out['changes']}")
    for name, tab in out["by_dim"].items():
        log(f"  {name}: " + "  ".join(f"{b['lo'] if b['lo'] == b['hi'] else f'{b['lo']:.3g}-{b['hi']:.3g}'}:"
                                      f"{b['passed'][str(P.TAU)]}/{b['n']}" for b in tab))
    if len(passing) >= 32 and a.write_init:
        init = passing[:32]
        order = sorted(range(32), key=lambda i: (-P.t_est(init[i]["psi"], tcpu), i))
        pts32 = [dict(psi=init[i]["psi"], sobol_index=init[i]["sobol_index"], sobol_rank=i,
                      t_est=P.t_est(init[i]["psi"], tcpu)) for i in order]
        (P.RUN / "e4").mkdir(parents=True, exist_ok=True)
        f = P.RUN / "e4" / "init_points.json"
        if f.exists():
            log(f"{f} exists; not rewritten")
        else:
            f.write_text(json.dumps(pts32, indent=1))
            log(f"wrote {f}: t_est {pts32[0]['t_est'] / 3600:.2f} h .. {pts32[-1]['t_est'] / 3600:.2f} h; the 32nd "
                f"initial point is the {init[-1]['sobol_index'] + 1}th Sobol point")
    log(f"PS_E1SUM_DONE n={n} pass@{P.TAU}={len(passing)}")


# --------------------------------------------------------------------------------------------- campaign
class Camp:
    def __init__(self, name: str):
        assert name in ("e4", "e6")
        self.name, self.dir = name, P.RUN / name
        self.obj, self.fit = ("J", "fits") if name == "e4" else ("J_pass", "fits_pass")
        self.n_init, self.base, self.ext = (32, 96, 16) if name == "e4" else (0, 16, 16)
        self.loo_blocks = {49: (49, 72), 73: (73, 96)} if name == "e4" else {}
        e3 = P.read_json(P.RUN / "e3" / "summary.json")
        self.noise = float(e3["delta_gp"][self.obj])
        self.sp, self.tcpu = P.load_space(), P.load_tcpu()

    def first_no(self) -> int:
        if self.name == "e4":
            return 1
        return max(int(d.name[1:]) for d in (P.RUN / "e4" / "trials").iterdir()) + 1

    def tdir(self, no: int) -> Path:
        return self.dir / "trials" / f"t{no:04d}"

    def snapshot(self) -> dict:
        """Everything is read from the files (call under the lock)."""
        self.sp = P.load_space()
        specs = []
        td = self.dir / "trials"
        if td.exists():
            for d in sorted(td.iterdir()):
                if (d / "spec.json").exists():
                    s = P.read_json(d / "spec.json")
                    s["result"] = P.read_json(d / "result.json") if (d / "result.json").exists() else None
                    specs.append(s)
        specs.sort(key=lambda s: s["no"])
        rows = []                                              # data D: fitted trials with a value
        for nm in ("shallow", "deep"):
            r = P.read_json(P.RUN / "e3" / f"{nm}_c0" / "result.json")
            if r.get(self.fit) and r.get(self.obj) is not None:
                rows.append(dict(psi=P.REFS[nm], val=r[self.obj], src=f"ref_{nm}", no=0, res=r))
        prev = []
        if self.name == "e6":
            for d in sorted((P.RUN / "e4" / "trials").iterdir()):
                s = P.read_json(d / "spec.json")
                s["result"] = P.read_json(d / "result.json")
                prev.append(s)
        for s in prev + specs:
            r = s["result"]
            if r is not None and r.get("status") == "ok" and r.get(self.fit) and r.get(self.obj) is not None:
                rows.append(dict(psi=s["psi"], val=r[self.obj], src=s["source"], no=s["no"], res=r))
        running = [s for s in specs if s["result"] is None]
        tried = {P.key_of(P.REFS["shallow"]), P.key_of(P.REFS["deep"])} | {P.key_of(s["psi"]) for s in prev + specs}
        return dict(specs=specs, rows=rows, running=running, tried=tried, prev=prev)

    def arrays(self, st: dict):
        X = np.array([P.to_u(r["psi"], self.sp) for r in st["rows"]])
        y = np.minimum(np.array([r["val"] for r in st["rows"]], dtype=float), 1.5)
        Pn = np.array([P.to_u(s["psi"], self.sp) for s in st["running"]]).reshape(len(st["running"]), 7)
        return X, y, Pn


def owner() -> dict:
    return dict(job=P.my_job(), host=socket.gethostname(), pid=os.getpid())


def is_orphan(s: dict, live) -> bool:
    o = s["owner"]
    if o["host"] == socket.gethostname() and not Path(f"/proc/{o['pid']}").exists():
        return True
    if o["job"].startswith("pid") or live is None:
        return False
    return o["job"] not in live


def write_spec(camp: Camp, s: dict) -> None:
    C.atomic_write_json(camp.tdir(s["no"]) / "spec.json", {k: v for k, v in s.items() if k != "result"})


def abnormal(camp: Camp, s: dict, why: str) -> None:
    C.atomic_write_json(camp.tdir(s["no"]) / "result.json",
                        dict(psi=s["psi"], status="abnormal", fits=False, fits_pass=False, J=None, J_pass=None,
                             error=why))
    log(f"trial {s['no']}: recorded as an abnormal end ({why})")


def edge_conditions(psi: dict, sp: dict) -> list[str]:
    """Rule 6: edge conditions of a setting for the current range (after an expansion: within 0.05 of the new end)."""
    u = P.to_u(psi, sp)
    out = []
    if u[3] <= sp["lo"][3] + 0.05:
        out.append("eta_lo")
    if u[3] >= sp["hi"][3] - 0.05:
        out.append("eta_hi")
    if u[5] >= sp["hi"][5] - 0.05:
        out.append("wd_hi")
    if psi["d"] == P._rnd(2 + 6 * sp["hi"][2]):
        out.append("d_hi")
    return out


def decide_ext(camp: Camp, st: dict) -> int | None:
    """Total number of trials of the campaign after the decision on the extension (None: not all base trials are
    finished yet). E4: rule 6. E6: step 2 of E6."""
    f = camp.dir / "ext.json"
    if f.exists():
        return P.read_json(f)["limit"]
    base = st["specs"][:camp.base]
    if len(base) < camp.base or any(s["result"] is None for s in base):
        return None
    X, y, _ = camp.arrays(st)
    rec = dict(time=time.strftime("%F %T"), n_data=len(X))
    if camp.name == "e4":
        gp, mu, sd = R.fit_gp(X, y, noise_sd=camp.noise)
        z = gp.predict(X)
        q = int(np.argmin(z))
        psi = st["rows"][q]["psi"]
        conds = edge_conditions(psi, camp.sp)
        rec.update(argmin_psi=psi, argmin_u=X[q].tolist(), argmin_src=st["rows"][q]["src"], argmin_no=st["rows"][q]["no"],
                   zhat_min=float(z[q]), conditions=conds, kernel=kernel_desc(gp), space_before=camp.sp)
        if conds:
            sp = json.loads(json.dumps(camp.sp))
            if "eta_lo" in conds:
                sp["lo"][3] = 0.5 * np.log10(1 / 3)
            if "eta_hi" in conds:
                sp["hi"][3] = 1 + 0.5 * np.log10(3)
            if "wd_hi" in conds:
                sp["hi"][5] = 1 + np.log10(3) / sp["u6_div"]
            if "d_hi" in conds:
                sp["hi"][2] = 10 / 6
            sp["expanded"] = conds
            C.atomic_write_json(P.RUN / "space.json", sp)
            rec["space_after"] = sp
            camp.sp = sp
        rec["limit"] = camp.base + (camp.ext if conds else 0)
    else:
        own = st["specs"][:camp.base]
        vals = {s["no"]: (s["result"][camp.obj] if s["result"].get("status") == "ok" and s["result"].get(camp.fit)
                          else None) for s in own}
        own_no = {s["no"] for s in own}
        init_best = min(r["val"] for r in st["rows"] if r["no"] not in own_no)
        new_best = False
        for i, s in enumerate(own):
            v = vals[s["no"]]
            earlier = [vals[t["no"]] for t in own[:i] if vals[t["no"]] is not None]
            if i >= 8 and v is not None and v < min([init_best] + earlier):
                new_best = True
        rec.update(init_best=init_best, own_values=[vals[s["no"]] for s in own], new_best_in_9_16=new_best)
        rec["limit"] = camp.base + (camp.ext if new_best else 0)
    C.atomic_write_json(f, rec)
    log(f"extension decision: {json.dumps(rec)[:600]}")
    return rec["limit"]


def next_spec(camp: Camp, dev):
    """Under the lock: an orphan trial to resume, the next trial to start, None (wait) or 'done'."""
    st = camp.snapshot()
    live = P.live_jobs()
    for s in st["running"]:
        if is_orphan(s, live):
            if s["attempts"] >= 2:
                abnormal(camp, s, "second abnormal end")
                continue
            s["attempts"] += 1
            s["owner"] = owner()
            write_spec(camp, s)
            log(f"trial {s['no']}: taken over for the single resume (attempt {s['attempts']})")
            return s
    st = camp.snapshot()
    n = len(st["specs"])
    limit = camp.base if n < camp.base else decide_ext(camp, st)
    if limit is None:
        return None
    if n >= limit:
        return "done"
    no = camp.first_no() + n if camp.name == "e6" else n + 1
    idx = n + 1                                               # 1-based index in the campaign
    if idx <= camp.n_init:
        q = P.read_json(camp.dir / "init_points.json")[idx - 1]
        spec = dict(no=no, psi=q["psi"], source="init", sobol_index=q["sobol_index"], sel=None)
    else:
        spec = select(camp, st, no, idx, dev)
    spec.update(u=P.to_u(spec["psi"], camp.sp).tolist(), key=P.key_of(spec["psi"]), owner=owner(), attempts=1,
                started=time.strftime("%F %T"), t_est=P.t_est(spec["psi"], camp.tcpu))
    write_spec(camp, spec)
    log(f"trial {no} [{spec['source']}] started: {spec['psi']} (t_est {spec['t_est'] / 3600:.2f} h)")
    return spec


def sobol_next(camp: Camp, st: dict, dev, rejected: list) -> dict:
    busy = st["tried"]
    for q in sobol_settings(camp.sp, camp.tcpu):
        if P.key_of(q["psi"]) in busy:
            continue
        g = P.gate_cached(q["psi"], dev, log=log)
        if P.gate_pass(g):
            return q
        rejected.append(q["psi"])
    raise RuntimeError("the Sobol sequence is exhausted")


def select(camp: Camp, st: dict, no: int, idx: int, dev) -> dict:
    """Plan 4.5 steps 1-4 for the trial with number ``no`` (index ``idx`` in the campaign)."""
    X, y, Pn = camp.arrays(st)
    rec = dict(no=no, time=time.strftime("%F %T"), n_data=len(X), n_running=len(Pn), min_J=float(y.min()),
               noise_sd=camp.noise)
    g = fit_with_pending(X, y, Pn, camp.noise)
    rho, rsd, kd = loo(X, y, camp.noise, gp=g["gp0"])
    rec.update(kernel=kd, rho_loo=rho, loo_resid_sd=rsd, zhat_min=g["best"],
               zhat_min_J=float(g["best"] * g["sd"] + g["mu"]))
    mode = "ei"
    for start, (lo, hi) in camp.loo_blocks.items():
        if lo <= idx <= hi:
            f = camp.dir / f"loo_{start}.json"
            if not f.exists():
                C.atomic_write_json(f, dict(no=no, rho_loo=rho, loo_resid_sd=rsd, kernel=kd, n_data=len(X),
                                            sobol=bool(rho < 0.5), time=rec["time"]))
                log(f"rho_LOO before trial {start}: {rho:.3f} -> {'Sobol' if rho < 0.5 else 'EI'} for {lo}-{hi}")
            if P.read_json(f)["sobol"]:
                mode = "sobol_loo"
    rejected, chosen = [], None
    if mode == "ei":
        rng = np.random.default_rng(no)
        expanded = bool(camp.sp["expanded"])
        cand = (candidates_box(rng, g["Xf"], g["yf"], camp.sp["lo"], camp.sp["hi"]) if expanded
                else R.candidates(rng, g["Xf"], g["yf"]))
        psis, U, seen = [], [], set(st["tried"])
        n_out = n_dup = n_failed = 0
        for c in cand:
            psi = P.norm_psi(P.from_u(c, camp.sp))
            k = P.key_of(psi)
            if not P.in_psi(psi, camp.sp, camp.tcpu):
                n_out += 1
                continue
            if k in seen:
                n_dup += 1
                continue
            seen.add(k)
            gf = P.RUN / "gates" / f"{k}.json"
            if gf.exists() and not P.gate_pass(P.read_json(gf)):
                n_failed += 1
                continue
            psis.append(psi)
            U.append(P.to_u(psi, camp.sp))
        rec.update(n_cand=len(cand), n_outside=n_out, n_tried_or_dup=n_dup, n_known_gate_fail=n_failed, n_valid=len(psis))
        if psis:
            U = np.array(U)

            def gate_fn(i):
                return P.gate_pass(P.gate_cached(psis[i], dev, log=log))

            i, e, rej = pick(g["gp"], g["best"], U, gate_fn)
            rejected = [psis[j] for j in rej]
            rec["ei_max"] = float(e.max())
            if i is not None:
                m, s = g["gp"].predict(U[i][None], return_std=True)
                chosen = dict(psi=psis[i], source="ei")
                rec.update(ei_chosen=float(e[i]), zhat_chosen=float(m[0]), omega_chosen=float(s[0]))
    if chosen is None:
        q = sobol_next(camp, st, dev, rejected)
        chosen = dict(psi=q["psi"], source="sobol_loo" if mode == "sobol_loo" else "sobol", sobol_index=q["sobol_index"])
    rec.update(mode=mode, chosen=chosen, gate_rejected=rejected)
    C.atomic_write_json(camp.dir / "select" / f"s{no:04d}.json", rec)
    return dict(no=no, sel=dict(rho_loo=rho, ei_max=rec.get("ei_max"), mode=mode), **chosen)


def cmd_worker(a) -> None:
    import torch
    R.LEVELS = LEVELS7
    dev = torch.device(a.device)
    camp = Camp(a.camp)
    log(f"worker {owner()} campaign {a.camp} device {dev} noise {camp.noise:.5f} tau {P.TAU}")
    while True:
        with P.file_lock(camp.dir / "lock"):
            spec = next_spec(camp, dev)
        if spec == "done":
            break
        if spec is None:
            time.sleep(60)
            continue
        no = spec["no"]
        while True:
            try:
                res = P.run_trial(spec["psi"], P.R_TUNE, C.POLICIES, 0, dev, camp.tdir(no),
                                  log=lambda m, no=no: log(f"t{no:04d} {m}"))
                log(f"trial {no} finished: status {res['status']} J {res['J']} J_pass {res['J_pass']} kappa_bar "
                    f"{res['kappa_bar']} fits {res['fits']} seconds {res['seconds']:.0f}")
                break
            except Exception as e:  # noqa: BLE001
                log(f"trial {no} failed: {type(e).__name__}: {e}")
                with P.file_lock(camp.dir / "lock"):
                    s = P.read_json(camp.tdir(no) / "spec.json")
                    if s["attempts"] >= 2:
                        abnormal(camp, s, f"{type(e).__name__}: {e}")
                        break
                    s["attempts"] += 1
                    write_spec(camp, s)
    log("PS_WORKER_DONE")


def cmd_status(a) -> None:
    camp = Camp(a.camp)
    st = camp.snapshot()
    done = [s for s in st["specs"] if s["result"] is not None]
    print(f"{a.camp}: started {len(st['specs'])} finished {len(done)} running {len(st['running'])} data {len(st['rows'])}")
    for s in st["specs"]:
        r = s["result"]
        txt = "running" if r is None else (f"{r['status']} J {r['J']:.4f} Jp {r['J_pass']:.4f} kb {r['kappa_bar']:.2f} "
                                           f"fits {r['fits']} {r['seconds'] / 60:.0f} min" if r.get("J") is not None
                                           else r["status"])
        p = s["psi"]
        print(f"  {s['no']:3d} {s['source']:9s} w{p['w']:<4d} L{p['L']} d{p['d']} eta {p['eta']:.2e} K {p['K']:4d} "
              f"wd {p['wd']:.1e} {p['init'][:2]} est {s['t_est'] / 3600:4.1f}h att {s['attempts']} | {txt}")
    for f in sorted(camp.dir.glob("loo_*.json")) + sorted(camp.dir.glob("ext.json")):
        print(f.name, json.dumps(P.read_json(f))[:400])


# --------------------------------------------------------------------------------------------- finalize
def cmd_finalize(a) -> None:
    R.LEVELS = LEVELS7
    camp = Camp(a.camp)
    st = camp.snapshot()
    assert not st["running"], "trials are still running"
    X, y, _ = camp.arrays(st)
    rows = st["rows"]
    gp, mu, sd = R.fit_gp(X, y, noise_sd=camp.noise)
    z = gp.predict(X)
    rho, rsd, kd = loo(X, y, camp.noise, gp=gp)
    oz = [int(i) for i in np.argsort(z, kind="stable")]
    oj = [int(i) for i in np.argsort([r["val"] for r in rows], kind="stable")]
    for i, r in enumerate(rows):
        r["zhat"], r["key"] = float(z[i]), P.key_of(r["psi"])
    out = dict(camp=a.camp, n_trials=len(st["specs"]), n_data=len(rows), kernel=kd, rho_loo=rho, loo_resid_sd=rsd,
               space=camp.sp, noise_sd=camp.noise, mu=float(mu), sd=float(sd),
               argmin_zhat=dict(psi=rows[oz[0]]["psi"], u=X[oz[0]].tolist(), src=rows[oz[0]]["src"],
                                no=rows[oz[0]]["no"], zhat=float(z[oz[0]]), J=rows[oz[0]]["val"],
                                edge=edge_conditions(rows[oz[0]]["psi"], camp.sp)),
               n_gate_pass_not_fit=int(sum(1 for s in st["specs"] if not (s["result"].get("status") == "ok"
                                                                           and s["result"].get(camp.fit)))),
               not_fit=[dict(no=s["no"], psi=s["psi"], status=s["result"].get("status"),
                             kappa_bar=s["result"].get("kappa_bar")) for s in st["specs"]
                        if not (s["result"].get("status") == "ok" and s["result"].get(camp.fit))],
               sources={k: int(sum(s["source"] == k for s in st["specs"])) for k in ("init", "ei", "sobol", "sobol_loo")})
    jmin = rows[oj[0]]["val"]
    near = [r for r in rows if r["val"] <= jmin + 0.03 and r["res"].get("aulc")]
    if a.camp == "e4":
        out["near_best"] = dict(n=len(near), share_smallest={
            p: float(np.mean([min(r["res"]["aulc"], key=r["res"]["aulc"].get) == p for r in near])) for p in C.POLICIES})
    out["table"] = [dict(no=r["no"], src=r["src"], psi=r["psi"], val=r["val"], zhat=r["zhat"],
                         J=r["res"].get("J"), J_pass=r["res"].get("J_pass"), aulc=r["res"].get("aulc"),
                         kappa_bar=r["res"].get("kappa_bar"), t_est=P.t_est(r["psi"], camp.tcpu),
                         seconds=r["res"].get("seconds")) for r in rows]
    chosen = []

    def add(i, why):
        k = rows[i]["key"]
        for c in chosen:
            if c["key"] == k:
                c["why"].append(why)
                return
        chosen.append(dict(key=k, psi=rows[i]["psi"], why=[why], src=rows[i]["src"], no=rows[i]["no"],
                           zhat=rows[i]["zhat"], val=rows[i]["val"]))

    if a.camp == "e4":
        taken = []
        for i in oz:                                       # item 1: 5 by z-hat with the distance rule
            if len(taken) == 5:
                break
            p = rows[i]["psi"]
            if any((p["w"], p["L"], p["init"]) == (rows[j]["psi"]["w"], rows[j]["psi"]["L"], rows[j]["psi"]["init"])
                   and np.linalg.norm(X[i][2:6] - X[j][2:6]) <= 0.1 for j in taken):
                continue
            taken.append(i)
            add(i, f"zhat{len(taken)}")
        for n_, i in enumerate(oj[:3]):                    # item 2: 3 by J
            add(i, f"J{n_ + 1}")
        init = [i for i in oj if rows[i]["src"] == "init"]
        if init:
            add(init[0], "best_init")                      # item 3
        for i, r in enumerate(rows):                       # item 4
            if r["src"].startswith("ref_"):
                add(i, r["src"])
        present = {c["src"] for c in chosen}
        for nm in ("shallow", "deep"):                     # a reference that does not fit has no value but stays in S
            if f"ref_{nm}" not in present:
                chosen.append(dict(key=P.key_of(P.REFS[nm]), psi=P.REFS[nm], why=[f"ref_{nm}"], src=f"ref_{nm}", no=0,
                                   zhat=None, val=None))
        ei_vals = [r["val"] for r in rows if r["src"] == "ei"]
        init_vals = [r["val"] for r in rows if r["src"] == "init"]
        out["c2_second"] = bool(ei_vals and init_vals and min(ei_vals) < min(init_vals))
    else:
        for n_, i in enumerate(oz[:3]):
            add(i, f"zhat{n_ + 1}")
        for n_, i in enumerate(oj):
            if len(chosen) >= 5:
                break
            add(i, f"Jpass{n_ + 1}")
    out["S"] = chosen
    C.atomic_write_json(camp.dir / "final.json", out)
    log(f"{a.camp} final: {len(rows)} data, rho_LOO {rho:.3f}, kernel {kd}")
    log(f"argmin zhat: {out['argmin_zhat']}")
    for c in chosen:
        log(f"  S: {c['why']} no {c['no']} {c['src']} {c['psi']} val {c['val']} zhat {c['zhat']}")
    log("PS_FINALIZE_DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("rehearse")
    s.add_argument("--seeds", default="0-19")
    s.add_argument("--levels", default="5,4,7,0,0,0,2")
    s.add_argument("--n-init", type=int, default=32)
    s.add_argument("--n-total", type=int, default=96)
    s.add_argument("--procs", type=int, default=20)
    s.add_argument("--max-gate", type=int, default=MAX_GATE, help="0: no limit (as in the rehearsal of P7)")
    s.add_argument("--out", type=Path, default=P.RUN / "e0" / "step3.json")
    s = sub.add_parser("e1")
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--nshards", type=int, default=1)
    s.add_argument("--count", type=int, default=256)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--device", default="cuda")
    s = sub.add_parser("e1sum")
    s.add_argument("--write-init", action="store_true")
    for name in ("worker", "status", "finalize"):
        s = sub.add_parser(name)
        s.add_argument("--camp", required=True)
        s.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dict(rehearse=cmd_rehearse, e1=cmd_e1, e1sum=cmd_e1sum, worker=cmd_worker, status=cmd_status,
         finalize=cmd_finalize)[a.cmd](a)


if __name__ == "__main__":
    main()
