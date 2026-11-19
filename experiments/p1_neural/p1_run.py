#!/usr/bin/env python3
"""One trajectory (method x problem) of purpose 1 (c) for a neural ensemble (design 1.8.5).

t = 0: fit on the 80 source rows -> NRMSE(0).  t = 1..T: IVR on the sample covariance of the member predictions at
the 448 target candidates (p1_common.choose_ivr; observed cells stay candidates), observe with p1_common.observe
(shared noise table, repeats allowed), cold refit on ALL rows so far (same member initialisation), NRMSE(t).
Response standardised with the 80 source rows' mean and sd (fixed for the whole trajectory).
Covariance = np.cov(member predictions) (ddof = 1) + COV_JITTER * I (recorded).

    python p1_run.py --method ofml --problem rep:0 --out runs/<tag> [--config cfg.json] [--T 200] [--members 5]

Output: <out>/<method>/<kind>_<NN>.json (records) and .npz (mean / var of the 448 candidates per t).
Resumable: a checkpoint (chosen candidates so far) is written every --ckpt-every rounds; the data at round t are a
deterministic function of the chosen list, so a resumed run refits exactly the same data.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COV_JITTER = 1e-8
PROBLEM_ROOT = HERE.parent / "gpn_graph" / "runs" / "p1_main" / "problems"
EVAL_TIMES = (0, 20, 50, 100, 200)


def load(kind_rep: str):
    import p1_models as PM
    kind, r = kind_rep.split(":")
    assert kind in ("rep", "tune")
    return PM.P.load_problem(PM.P.problem_path(PROBLEM_ROOT, int(r), kind == "tune")), kind, int(r)


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def run_trajectory(method: str, cfg, prob, T: int, device="cpu", log=None, ckpt: Path | None = None,
                   ckpt_every: int = 20, stop_on_nan: bool = True, impl: str = "fast", compile_: bool = False,
                   stop_after: int | None = None) -> dict:
    import torch
    import p1_models as PM
    P = PM.P
    y_mean, y_std = float(np.mean(prob.src_y)), float(np.std(prob.src_y))
    if impl == "fast":
        ens = PM.FastEnsemble(method, cfg, seed_base=prob.seed, device=device, y_mean=y_mean, y_std=y_std,
                              n_max=len(prob.src_cells) + T, compile=compile_)
    else:
        ens = PM.Ensemble(method, cfg, seed_base=prob.seed, device=device, y_mean=y_mean, y_std=y_std)
    rec = dict(t=[], nrmse=[], nrmse_by_target=[], chosen=[], y_obs=[], fit_seconds=[], max_var=[],
               mean_var=[], min_cov_eig=[])
    means, vars_ = [], []
    chosen: list[int] = []
    if ckpt is not None and ckpt.exists():
        z = json.loads(ckpt.read_text())
        assert z["method"] == method and z["config"] == cfg.to_dict() and z["seed"] == prob.seed and z["T"] == T
        rec, chosen = z["rec"], z["chosen"]
        a = np.load(ckpt.with_suffix(".npz"))
        means, vars_ = list(a["mean"]), list(a["var"])
        if log:
            log(f"[ckpt] resume after t={rec['t'][-1]}")
    visits = np.zeros(P.N_TGT_CELLS, dtype=np.int64)
    cells = list(map(int, prob.src_cells))
    ys = list(map(float, prob.src_y))
    for c in chosen:                                # replay the data (deterministic)
        ys.append(P.observe(prob, c, visits))
        cells.append(P.candidate_to_cell(c))
    t = len(chosen)
    status = "ok"
    tgt_truth = prob.truth_tgt
    base_rmse = np.sqrt(np.mean((tgt_truth - tgt_truth.mean()) ** 2))
    t0 = time.time()
    S = None

    def fit_members():
        ts = time.time()
        ens.fit(np.array(cells), np.array(ys))
        S_ = ens.member_targets()                                        # [M, 448], original scale
        if device != "cpu":
            torch.cuda.synchronize()
        return S_, time.time() - ts

    def covariance(S_):
        return np.cov(S_.T) + COV_JITTER * np.eye(S_.shape[1])

    while True:
        if not rec["t"] or rec["t"][-1] < t:
            S, fs = fit_members()
            if not np.all(np.isfinite(S)):
                status = f"nonfinite_at_t{t}"
                if log:
                    log(status)
                if stop_on_nan:
                    break
            mean = S.mean(0)
            var = S.var(0, ddof=1) if S.shape[0] > 1 else np.zeros(S.shape[1])
            rec["t"].append(t)
            rec["nrmse"].append(P.nrmse(mean, prob))
            err = (mean - tgt_truth).reshape(len(P.TARGETS), P.G)
            rec["nrmse_by_target"].append([float(np.sqrt(np.mean(e ** 2)) / base_rmse) for e in err])
            rec["fit_seconds"].append(fs)
            rec["max_var"].append(float(var.max()))
            rec["mean_var"].append(float(var.mean()))
            rec["min_cov_eig"].append(float(np.linalg.eigvalsh(covariance(S))[0]) if np.all(np.isfinite(S)) else float("nan"))
            means.append(mean.astype(np.float32))
            vars_.append(var.astype(np.float32))
            if log and (t % 10 == 0 or t == T):
                log(f"t={t:3d} nrmse {rec['nrmse'][-1]:.4f} fit {fs:.2f}s elapsed {time.time() - t0:.0f}s")
            if ckpt is not None and t % ckpt_every == 0 and 0 < t < T:
                tmp = ckpt.with_name("." + ckpt.stem + ".tmp.npz")
                np.savez(tmp, mean=np.stack(means), var=np.stack(vars_))
                os.replace(tmp, ckpt.with_suffix(".npz"))
                atomic_json(ckpt, dict(method=method, config=cfg.to_dict(), seed=prob.seed, T=T, rec=rec,
                                       chosen=chosen))
                if stop_after is not None and t >= stop_after:          # test only: simulate a kill
                    status = "stopped"
                    break
        elif S is None:                                                  # resumed at a recorded round
            S, _ = fit_members()
        if t >= T:
            break
        c = P.choose_ivr(covariance(S))
        y = P.observe(prob, c, visits)
        chosen.append(int(c))
        rec["chosen"].append(int(c))
        rec["y_obs"].append(y)
        cells.append(P.candidate_to_cell(c))
        ys.append(y)
        t += 1
        S = None
    return dict(status=status, rec=rec, mean=np.stack(means) if means else np.zeros((0, 448)),
                var=np.stack(vars_) if vars_ else np.zeros((0, 448)), seconds=time.time() - t0,
                y_mean=y_mean, y_std=y_std)


def tuning_score(nrmse: list) -> float:
    """Mean NRMSE at t in EVAL_TIMES (those <= the trajectory's T)."""
    return float(np.mean([nrmse[t] for t in EVAL_TIMES if t < len(nrmse)]))


def config_from(args_cfg: str, members: int | None, steps: int | None, dagnn_map: str | None):
    import p1_models as PM
    d = json.loads(Path(args_cfg).read_text()) if args_cfg else {}
    keys = {f.name for f in PM.dataclasses.fields(PM.NetConfig)}
    d = {k: v for k, v in d.items() if k in keys}
    if members:
        d["n_members"] = members
    if steps:
        d["steps"] = steps
    if dagnn_map:
        d["dagnn_map"] = dagnn_map
    return PM.NetConfig(**d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--problem", required=True, help="rep:N (production, N=0..15) or tune:N (N=0..3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--members", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--dagnn-map", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--impl", default="fast", choices=("fast", "vmap"))
    ap.add_argument("--compile", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    import p1_models as PM
    assert a.method in PM.METHODS, a.method
    prob, kind, r = load(a.problem)
    cfg = config_from(a.config, a.members, a.steps, a.dagnn_map)
    od = Path(a.out) / a.method
    od.mkdir(parents=True, exist_ok=True)
    res_path = od / f"{kind}_{r:02d}.json"
    if a.skip_existing and res_path.exists() and json.loads(res_path.read_text()).get("status") == "ok":
        print("exists", res_path)
        return
    ckpt = Path(a.out) / "ckpt" / f"{a.method}_{kind}_{r:02d}.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(f"[{a.method} {kind}:{r}] {s}", flush=True)  # noqa: E731
    log(f"cfg {cfg.to_dict()} params/member {PM.n_params(a.method, cfg)} device {a.device}")
    out = run_trajectory(a.method, cfg, prob, a.T, device=a.device, log=log, ckpt=ckpt, impl=a.impl,
                         compile_=bool(a.compile))
    np.savez(res_path.with_suffix(".npz"), mean=out["mean"], var=out["var"])
    import platform
    meta = dict(method=a.method, label=PM.METHOD_LABEL[a.method], problem=f"{kind}:{r}", teacher_seed=prob.seed,
                T=a.T, config=cfg.to_dict(), n_params_per_member=PM.n_params(a.method, cfg),
                cov_jitter=COV_JITTER, y_mean=out["y_mean"], y_std=out["y_std"], status=out["status"],
                seconds=out["seconds"], host=socket.gethostname(), device=a.device,
                gpu=(torch.cuda.get_device_name(0) if a.device != "cpu" else ""), torch=torch.__version__,
                python=platform.python_version(), slurm_job=os.environ.get("SLURM_JOB_ID", ""),
                threads=a.threads, impl=a.impl, compile=bool(a.compile), tuning_score=tuning_score(out["rec"]["nrmse"]),
                aulc=PM.P.aulc(np.asarray(out["rec"]["nrmse"])) if len(out["rec"]["nrmse"]) > 1 else None,
                rounds=out["rec"])
    atomic_json(res_path, meta)
    for f in (ckpt, ckpt.with_suffix(".npz")):
        if f.exists():
            f.unlink()
    log(f"done status={out['status']} {out['seconds']:.0f}s nrmse(0)={out['rec']['nrmse'][0]:.4f} "
        f"nrmse(T)={out['rec']['nrmse'][-1]:.4f}")


if __name__ == "__main__":
    main()
