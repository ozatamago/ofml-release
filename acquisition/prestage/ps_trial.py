#!/usr/bin/env python3
"""One trial of the pre-stage (plan 4.2): all trajectories (functions x acquisition functions) of a setting as one
bundle, through ``ps_common.run_trial``.
usage: ps_trial.py --psi shallow --reps 1000-1015 --c 0 --device cuda --out runs/prestage/e3/shallow_c0
       --psi is a name of ps_common.REFS or a JSON file holding a setting dict(w, L, d, eta, K, wd, init)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ps_common as P
from ps_common import C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi", required=True)
    ap.add_argument("--reps", default="1000-1015")
    ap.add_argument("--pols", default=",".join(C.POLICIES))
    ap.add_argument("--c", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--stop-after-n", type=int, default=None, help="test only")
    ap.add_argument("--no-split", action="store_true")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    psi = P.REFS[a.psi] if a.psi in P.REFS else json.loads(Path(a.psi).read_text())
    dev = torch.device(a.device)
    t0 = time.time()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"trial psi={P.norm_psi(psi)} reps={a.reps} pols={a.pols} c={a.c} device={dev} out={a.out}")
    res = None
    for attempt in (1, 2):                      # plan 4.2: one resume after an abnormal end
        try:
            res = P.run_trial(psi, C.parse_range(a.reps), a.pols.split(","), a.c, dev, a.out, log=log,
                              stop_after_n=a.stop_after_n, no_split=a.no_split)
            break
        except Exception as e:  # noqa: BLE001
            log(f"attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt == 2:
                C.atomic_write_json(a.out / "result.json", dict(psi=P.norm_psi(psi), status="abnormal", fits=False,
                                                                 fits_pass=False, J=None, J_pass=None, error=str(e)))
                sys.exit(1)
    if res is not None:
        log(f"PS_TRIAL_DONE status={res['status']} J={res['J']} J_pass={res['J_pass']} kappa_bar={res['kappa_bar']} "
            f"fits={res['fits']} seconds={time.time() - t0:.0f}")


if __name__ == "__main__":
    main()
