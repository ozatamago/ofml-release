#!/usr/bin/env python3
"""Hyper-parameter search of one neural method for purpose 1 (c) (design 1.8.6): GP + EI Bayesian optimisation.

Objective (minimised): mean NRMSE over the 4 tuning functions (tune_00..03, never production) x t in
{0, 20, 50, 100, 200}, each trajectory run EXACTLY as production (IVR on the member covariance, cold refit at every
t = 0..200, 400 Adam steps, 5 members).  The 4 trajectories of a trial run as 4 parallel single-thread processes.

Space (all methods), ``--space v1``: hidden layers 1..3 (int), width 8..64 (log2, rounded to an integer width), latent
dim 2..8 (int; not for Shared-Ens), learning rate [1e-3, 3e-2] (log), weight decay [1e-6, 1e-3] (log).
``--space wide`` (2026-11-05, after Shared-Ens and SharedCell-DAG-Ens selected upper-bound values in v1): hidden layers
1..5, width 4..128, latent dim 2..8, learning rate [1e-3, 1e-1], weight decay [1e-6, 1e-2].  It contains v1, so the
v1 trials of a method are continued as data (copy the jsonl, raise --trials).
Trials 0..n_init-1: scrambled Sobol (seeded); then argmax EI of a GP (Matern 5/2 ARD + white noise, sklearn) on
the standardised objective (same proposal code as ens_hpo.py of the Section 6.3 experiments, not included).  Non-finite trials enter the GP capped
at Y_CAP and are never selected.  ``--parallel q > 1``: asynchronous q-batch (constant liar = current best).

    python p1_hpo.py --method ofml --out runs/hpo_<date>/ofml.jsonl --trials 50 [--parallel 1]
Resumable: trials already in the jsonl are kept (the proposal for trial k depends only on trials < k and the seed
when --parallel 1).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N_TUNE = 4
Y_CAP = 1.5
SPACE_ALL = dict(n_hidden=(1, 3), log2_width=(3, 6), latent_dim=(2, 8), log10_lr=(math.log10(1e-3), math.log10(3e-2)),
                 log10_wd=(-6, -3))
SPACES = dict(v1=SPACE_ALL,
              wide=dict(n_hidden=(1, 5), log2_width=(2, 7), latent_dim=(2, 8), log10_lr=(-3, -1), log10_wd=(-6, -2)))


class Space:
    def __init__(self, method: str, dagnn_map: str = "state_latent", space: str = "v1"):
        # latent_dim is not a parameter of Shared-Ens, nor of the DAGNN variants under dagnn_map = state_width
        no_latent = method == "shared" or (method in ("dagnn", "dagnn_cat") and dagnn_map == "state_width")
        sp = SPACES[space]
        self.names = [k for k in sp if not (no_latent and k == "latent_dim")]
        self.lo = np.array([sp[k][0] for k in self.names], float)
        self.hi = np.array([sp[k][1] for k in self.names], float)

    @property
    def dim(self):
        return len(self.names)

    def decode(self, u) -> dict:
        v = dict(zip(self.names, self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)))
        out = dict(n_hidden=int(round(v["n_hidden"])), width=int(round(2 ** v["log2_width"])),
                   lr=float(10 ** v["log10_lr"]), wd=float(10 ** v["log10_wd"]))
        if "latent_dim" in v:
            out["latent_dim"] = int(round(v["latent_dim"]))
        return out

    def encode(self, c: dict) -> np.ndarray:
        v = dict(n_hidden=c["n_hidden"], log2_width=math.log2(c["width"]), latent_dim=c.get("latent_dim", 0),
                 log10_lr=math.log10(c["lr"]), log10_wd=math.log10(c["wd"]))
        return (np.array([v[k] for k in self.names]) - self.lo) / (self.hi - self.lo)

    def snap(self, u):
        return self.encode(self.decode(u))

    def boundary(self, c: dict, tol=0.02) -> dict:
        u = self.encode(c)
        return {k: ("lower" if u[i] <= tol else "upper") for i, k in enumerate(self.names) if u[i] <= tol or u[i] >= 1 - tol}


def expected_improvement(mu, sd, best):
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-12)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def propose(space: Space, X, yv, rng, n_cand: int = 20000):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    ys = (yv - yv.mean()) / max(yv.std(), 1e-12)
    kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.full(X.shape[1], 0.3),
                                                     length_scale_bounds=(1e-2, 1e1), nu=2.5) \
        + WhiteKernel(1e-3, (1e-6, 1e-1))
    gp = GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=5, random_state=0).fit(X, ys)
    best = ys.min()
    cand = rng.random((n_cand, X.shape[1]))
    xb = X[np.argsort(ys)[:5]]
    local = np.clip(xb[rng.integers(0, len(xb), n_cand // 2)] + rng.normal(0, 0.08, (n_cand // 2, X.shape[1])), 0, 1)
    cand = np.array([space.snap(c) for c in np.vstack([cand, local])])
    seen = {tuple(np.round(x, 6)) for x in X}
    cand = cand[np.array([tuple(np.round(c, 6)) not in seen for c in cand])]
    mu, sd = gp.predict(cand, return_std=True)
    return cand[int(np.argmax(expected_improvement(mu, sd, best)))], str(gp.kernel_)


_W: dict = {}


def _init(threads: int, device: str, compile_: bool) -> None:
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _W["device"], _W["compile"] = device, compile_


def _eval_one(args):
    """One tuning trajectory -> (u, nrmse list, status, seconds)."""
    method, cfgd, u, T = args
    import p1_models as PM
    import p1_run as R
    prob = PM.P.load_problem(PM.P.problem_path(R.PROBLEM_ROOT, u, True))
    cfg = PM.NetConfig(**cfgd)
    if _W["compile"]:
        import torch
        torch._dynamo.reset()
    t0 = time.time()
    out = R.run_trajectory(method, cfg, prob, T, device=_W["device"], compile_=_W["compile"])
    return u, out["rec"]["nrmse"], out["rec"]["nrmse_by_target"], out["status"], time.time() - t0


def score(per: dict, T: int) -> float:
    import p1_run as R
    if any(len(v) < T + 1 or not np.all(np.isfinite(v)) for v in per.values()):
        return float("nan")
    return float(np.mean([R.tuning_score(v) for v in per.values()]))


def read_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=1, help="trials in flight (1 = plain sequential BO)")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dagnn-map", default="state_latent")
    ap.add_argument("--space", default="v1", choices=sorted(SPACES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--compile", type=int, default=0)
    a = ap.parse_args()
    import multiprocessing as mp
    from scipy.stats import qmc
    import p1_models as PM
    space = Space(a.method, a.dagnn_map, a.space)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trials = read_jsonl(out)
    sobol = qmc.Sobol(space.dim, scramble=True, seed=a.seed).random(16)
    fixed = dict(steps=a.steps, n_members=a.members, dagnn_map=a.dagnn_map)
    pool = mp.get_context("spawn").Pool(N_TUNE * a.parallel, initializer=_init, initargs=(a.threads, a.device, bool(a.compile)))
    print(f"{a.method}: space {a.space} {space.names}; {len(trials)} trials on file; parallel {a.parallel}; fixed {fixed}",
          flush=True)
    pending: dict = {}
    next_k = len(trials)

    def make(k):
        if k < a.n_init:
            return "sobol", space.decode(sobol[k]), ""
        X = np.array([space.encode(t["config"]) for t in trials] + [space.encode(v[1]) for v in pending.values()])
        yv = np.array([t["objective"] for t in trials], float)
        yv = np.where(np.isfinite(yv), np.minimum(yv, Y_CAP), Y_CAP)
        yv = np.concatenate([yv, np.full(len(pending), yv.min())])
        u, info = propose(space, X, yv, np.random.default_rng([a.seed, k]))
        return "bo", space.decode(u), info

    while len(trials) < a.trials:
        while len(pending) < a.parallel and next_k < a.trials:
            kind, c, info = make(next_k)
            cfgd = dict(PM.NetConfig().to_dict(), **c, **fixed)
            if "latent_dim" not in space.names:
                cfgd["latent_dim"] = PM.NetConfig().latent_dim      # unused (Shared-Ens; DAGNN with state_width)
            jobs = [pool.apply_async(_eval_one, ((a.method, cfgd, u, a.T),)) for u in range(N_TUNE)]
            pending[next_k] = (kind, c, cfgd, info, jobs, time.time())
            next_k += 1
        done = [k for k, v in pending.items() if all(j.ready() for j in v[4])]
        if not done:
            time.sleep(1.0)
            continue
        for k in sorted(done):
            kind, c, cfgd, info, jobs, t0 = pending.pop(k)
            res = [j.get() for j in jobs]
            per = {u: nr for u, nr, _, _, _ in res}
            obj = score(per, a.T)
            rec = dict(trial=k, kind=kind, method=a.method, config=c, full_config=cfgd,
                       n_params_per_member=PM.n_params(a.method, PM.NetConfig(**cfgd)), objective=obj,
                       by_time={str(t): float(np.mean([per[u][t] for u in per])) for t in (0, 20, 50, 100, 200)
                                if all(t < len(per[u]) for u in per)},
                       per_tune={str(u): nr for u, nr in per.items()},
                       nrmse_by_target_t0={str(u): bt[0] for u, _, bt, _, _ in res},
                       status={str(u): s for u, _, _, s, _ in res}, traj_seconds={str(u): s for u, _, _, _, s in res},
                       wall_seconds=time.time() - t0, gp_kernel=info, n_inflight=a.parallel, device=a.device, compile=bool(a.compile),
                       seed=a.seed, space=a.space)
            trials.append(rec)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            fin = [t for t in trials if np.isfinite(t["objective"])]
            best = min(fin, key=lambda t: t["objective"]) if fin else None
            print(f"trial {k:2d} [{kind}] obj {obj:.4f} cfg {c} params {rec['n_params_per_member']} "
                  f"({rec['wall_seconds']:.0f}s) | best {best['objective'] if best else float('nan'):.4f} "
                  f"@ {best['trial'] if best else -1} | done {len(trials)}", flush=True)
    pool.close()
    pool.join()
    fin = [t for t in trials if np.isfinite(t["objective"])]
    best = min(fin, key=lambda t: t["objective"])
    sel = dict(best["full_config"], method=a.method, selected_trial=best["trial"], objective=best["objective"],
               by_time=best["by_time"], n_params_per_member=best["n_params_per_member"],
               boundary=space.boundary(best["config"]), n_trials=len(trials), space=a.space)
    (out.parent / f"selected_{a.method}.json").write_text(json.dumps(sel, indent=1))
    print("SELECTED", json.dumps(sel), flush=True)


if __name__ == "__main__":
    main()
