"""PB-W16U: apply the Section 6.2 (GP-matched benchmark) exclusion rule to the OFML-GPN runs (2026-11-08).

Rule: a true function is excluded if, in any of the four policy runs, the ESS right after the weight update and
before the resampling decision, divided by the number of particles, falls below 0.001 at any assimilation
(the 8 initial points included).  This is the quantity ``ess_pre``; the pre-registered PB-W16U rule looked at the
ESS at the start of each round instead, which cannot fall below the 0.65 resampling threshold.

Prints the excluded reps with the policies and minima, and the paired AULC difference IVR - space-filling on the
remaining reps and on all 16 (same trapezoid rule and bootstrap as analyze_w16u.py).

usage: sensitivity_p1_rule.py runs/2026-11-04_main
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_w16u import aulc, boot_ci  # noqa: E402

POLICIES = ("ivr", "maxvar", "spacefill", "random")
THRESH = 1e-3


def min_ess_pre(js: dict) -> float:
    vals = [v for v in js["rounds"]["ess_pre"] if v == v]
    vals += [d["ess_pre"] for d in js.get("warmup", {}).get("diag", []) if d.get("ess_pre") == d.get("ess_pre")]
    return float(min(vals))


def main() -> None:
    run = Path(sys.argv[1])
    reps = list(range(16))
    hits = {}
    curves = {}
    for r in reps:
        for p in POLICIES:
            js = json.loads((run / "gpn" / f"rep_{r:04d}_{p}.json").read_text())
            m = min_ess_pre(js)
            if m < THRESH:
                hits.setdefault(r, []).append((p, m))
            curves[(r, p)] = np.asarray(js["rounds"]["nrmse"], float)
    keep = [r for r in reps if r not in hits]
    print("excluded:", {r: [(p, f"{m:.5f}") for p, m in v] for r, v in sorted(hits.items())})
    print("kept:", keep)
    for name, rs in (("kept", keep), ("all16", reps)):
        d = np.array([aulc(curves[(r, "ivr")]) - aulc(curves[(r, "spacefill")]) for r in rs])
        lo, hi = boot_ci(d)
        print(f"{name}: IVR - spacefill AULC = {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], IVR smaller {int((d < 0).sum())}/{len(d)}")


if __name__ == "__main__":
    main()
