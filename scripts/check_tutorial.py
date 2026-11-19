"""Compare the outputs of the tutorial (README, steps 2 and 3) with the published run records.

Step 2 (deep ensembles, CPU, one thread) uses the settings of the paper's runs, so the error curves must equal the
published ones:
  Section 6.2, OFML-Ens, true function 2, t = 0..10   vs transfer/ensembles/runs/main_2026-11-08/ofml/rep_02.json
  Section 6.3, OFML-Ens (shallow reference), rep 0, IVR
                                                      vs acquisition/runs/prestage/prod_eval_fz/shallow/ens
Step 3 (OFML-GPN, GPU) uses fewer particles and pCN updates than the paper; only the completion of the runs is checked.

usage: python scripts/check_tutorial.py [--gpu]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TUT = ROOT / "tutorial_runs"
TOL = 1e-9


def compare(name: str, got, ref) -> bool:
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    ok = got.shape == ref.shape and np.allclose(got, ref, rtol=0, atol=TOL)
    diff = float(np.max(np.abs(got - ref))) if got.shape == ref.shape else float("nan")
    print(f"[{'ok' if ok else 'DIFFERS'}] {name}: {len(got)} values, max |difference| = {diff:.3g}")
    return ok


def main() -> None:
    ok = True
    t = json.loads((TUT / "transfer_ensembles" / "ofml" / "rep_02.json").read_text())["rounds"]["nrmse"]
    r = json.loads((ROOT / "transfer" / "ensembles" / "runs" / "main_2026-11-08" / "ofml" / "rep_02.json").read_text())["rounds"]["nrmse"]
    ok &= compare("Section 6.2, OFML-Ens, NRMSE(t), t = 0..10", t, r[:len(t)])

    t = json.loads((TUT / "acquisition" / "ens" / "rep_0000_ivr.json").read_text())["rounds"]["nrmse"]
    r = json.loads((ROOT / "acquisition" / "runs" / "prestage" / "prod_eval_fz" / "shallow" / "ens"
                    / "rep_0000_ivr.json").read_text())["rounds"]["nrmse"]
    ok &= compare("Section 6.3, OFML-Ens (shallow reference), IVR, NRMSE per round", t, r)

    if "--gpu" in sys.argv:
        for p in (TUT / "transfer_gpn" / "rep_02.json", TUT / "acquisition" / "gpn" / "rep_0000_ivr.json"):
            st = json.loads(p.read_text())["status"] if p.exists() else "missing"
            good = p.exists() and st in ("ok", "excluded")
            ok &= good
            print(f"[{'ok' if good else 'FAILED'}] {p.relative_to(ROOT)}: status {st}")
    print("all checks passed" if ok else "some checks failed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
