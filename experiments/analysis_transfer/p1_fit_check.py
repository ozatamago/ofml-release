"""Diagnostic (2026-11-18, not a production script): do the section 6.2 ensembles fit their training points?

For each method and rep, replay the chosen target candidates of the saved production trajectory
(p1_neural/runs/main_2026-11-08/<method>/rep_NN.json), refit the ensemble exactly as p1_run.py does
(FastEnsemble, same config, same member seeds) on D_t for t in TIMES, and report
kappa_t = RMSE(ensemble mean at the training cells - observed y) / NOISE_SD (0.10).
The same is repeated with STEPS_LONG Adam steps (only the number of steps changed) to see whether
the training error is still decreasing at the production 400 steps.
"""
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "p1_neural"
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
torch.set_num_threads(1)
import p1_models as PM  # noqa: E402
import p1_run as R  # noqa: E402

P = PM.P
METHODS = sys.argv[1].split(",")
REPS = [int(r) for r in sys.argv[2].split(",")]
TIMES = (0, 50, 100, 200)
STEPS_LONG = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
COMPILE = True

for method in METHODS:
    for rep in REPS:
        js = json.loads((HERE / "runs/main_2026-11-08" / method / f"rep_{rep:02d}.json").read_text())
        prob, _, _ = R.load(f"rep:{rep}")
        names = {f.name for f in fields(PM.NetConfig)}
        cfg0 = {k: v for k, v in js["config"].items() if k in names}
        chosen = js["rounds"]["chosen"]
        y_mean, y_std = float(np.mean(prob.src_y)), float(np.std(prob.src_y))
        visits = np.zeros(P.N_TGT_CELLS, dtype=np.int64)
        cells = list(map(int, prob.src_cells)); ys = list(map(float, prob.src_y))
        data = {0: (list(cells), list(ys))}
        for t, c in enumerate(chosen, start=1):
            ys.append(P.observe(prob, c, visits)); cells.append(P.candidate_to_cell(c))
            if t in TIMES:
                data[t] = (list(cells), list(ys))
        assert np.allclose(ys[len(prob.src_y):], js["rounds"]["y_obs"])
        out = []
        for steps in (cfg0["steps"], STEPS_LONG):
            cfg = PM.NetConfig(**{**cfg0, "steps": steps})
            row = []
            for t in TIMES:
                c_, y_ = data[t]
                ens = PM.FastEnsemble(method, cfg, seed_base=prob.seed, device="cpu", y_mean=y_mean, y_std=y_std,
                                      n_max=len(prob.src_cells) + 200, compile=COMPILE)
                ts = time.time()
                ens.fit(np.array(c_), np.array(y_))
                mean = ens.member_table().reshape(cfg.n_members, -1).mean(0)
                kap = float(np.sqrt(np.mean((mean[np.array(c_)] - np.array(y_)) ** 2)) / P.NOISE_SD)
                tgt = ens.member_targets().mean(0)
                nr = float(P.nrmse(tgt, prob))
                if steps == cfg0["steps"] and t in js["rounds"]["t"]:
                    assert abs(nr - js["rounds"]["nrmse"][t]) < 2e-3, (nr, js["rounds"]["nrmse"][t])
                row.append(f"t={t:3d} kappa {kap:5.2f} nrmse {nr:.3f} ({time.time() - ts:.0f}s)")
            out.append(f"  steps {steps:5d}: " + " | ".join(row))
        print(f"{method:10s} rep {rep:2d}  (n_train at t=0,50,100,200: 80,130,180,280)", flush=True)
        print("\n".join(out), flush=True)
