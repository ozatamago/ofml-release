#!/usr/bin/env python3
"""Time ONE fit (5 members, 400 Adam steps) of every method on the source rows + n_extra target rows.

    python bench_fit.py --device cuda --impl fast --compile 0 --out runs/bench/<tag>.json
Configs: min (1 layer, width 8, latent 2), mid (2, 32, 5), max (3, 64, 8).  The first fit (warm-up / compile) is
timed separately; then --reps fits are timed.  Rows: tune_00 source rows + n_extra target cells (first n_extra
candidates of a fixed random order, as a stand-in for IVR picks).
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

CONFIGS = dict(min=dict(n_hidden=1, width=8, latent_dim=2), mid=dict(n_hidden=2, width=32, latent_dim=5),
               max=dict(n_hidden=3, width=64, latent_dim=8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--impl", default="fast", choices=("fast", "vmap"))
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--methods", default="shared,dagnn,dagnn_cat,sharedcell,ofml")
    ap.add_argument("--configs", default="min,mid,max")
    ap.add_argument("--n-extra", default="0,200")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    import p1_models as PM
    import p1_run as R
    prob = PM.P.load_problem(PM.P.problem_path(R.PROBLEM_ROOT, 0, True))
    order = np.random.default_rng(5).permutation(PM.P.N_TGT_CELLS)
    res = dict(host=socket.gethostname(), device=a.device, impl=a.impl, compile=a.compile, threads=a.threads,
               gpu=torch.cuda.get_device_name(0) if a.device != "cpu" else "", torch=torch.__version__,
               slurm_job=os.environ.get("SLURM_JOB_ID", ""), rows=[])
    for meth in a.methods.split(","):
        for cn in a.configs.split(","):
            cfg = PM.NetConfig(**CONFIGS[cn])
            for ne in map(int, a.n_extra.split(",")):
                cells = np.concatenate([prob.src_cells, np.array([PM.P.candidate_to_cell(c) for c in order[:ne]], dtype=np.int64)])
                y = np.concatenate([prob.src_y, prob.truth[cells[len(prob.src_cells):]]])
                if a.compile:
                    torch._dynamo.reset()          # one compiled graph per config (avoid the recompile limit)
                if a.impl == "fast":
                    ens = PM.FastEnsemble(meth, cfg, prob.seed, device=a.device, compile=bool(a.compile),
                                          n_max=len(cells))
                else:
                    ens = PM.Ensemble(meth, cfg, prob.seed, device=a.device)
                t0 = time.time()
                ens.fit(cells, y)
                S = ens.member_targets()
                first = time.time() - t0
                ts = []
                for _ in range(a.reps):
                    t0 = time.time()
                    ens.fit(cells, y)
                    S = ens.member_targets()
                    ts.append(time.time() - t0)
                row = dict(method=meth, config=cn, n_rows=len(cells), first_fit=first, fit_seconds=float(np.median(ts)),
                           n_params=PM.n_params(meth, cfg), finite=bool(np.all(np.isfinite(S))))
                res["rows"].append(row)
                print(json.dumps(row), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
