"""Sensitivity analysis of Section 6.3 under the exclusion rule of Section 6.2 (Appendix, sec:acq-sensitivity).

1. Applies the Section 6.2 rule (ess_pre / P < 0.001 at any assimilation, the 8 initial points included) to the four
   OFML-GPN runs of each of the 16 evaluation true functions, and counts the runs that hit it per acquisition function
   and whether any hit happened during the assimilation of the initial points (``warmup``).
2. On the remaining true functions, recomputes the quantities of fig:exp-acquisition (b) for OFML-GPN with IVR
   (mean over the true functions of the maximum posterior variance, the maximum absolute error of the posterior mean,
   and the 95% coverage, each over the 512 grid points of the 8 protocols with an initial point and of the 8 without),
   with the same functions as gpn_w16u_al/plot_w16u_uniform_split.py.
The paired AULC difference IVR - space-filling on the remaining true functions is printed by
gpn_w16u_al/sensitivity_p1_rule.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "experiments" / "gpn_w16u_al"
sys.path.insert(0, str(W))
sys.path.insert(0, str(ROOT / "scripts"))
import w16u_common as C  # noqa: E402
from plot_w16u_uniform_split import per_rep  # noqa: E402

RUN = W / "runs" / "2026-11-04_main"
POLICIES = ("ivr", "maxvar", "spacefill", "random")
THRESH = 1e-3
PAPER = {"hits": {"ivr": 2, "maxvar": 2, "spacefill": 4, "random": 4}, "n_excluded": 8,
         "max_var_with": (0.39, 0.03), "max_var_without": (0.38, 0.03),
         "max_err_with": (1.26, 0.48), "max_err_without": (1.06, 0.38), "cov95": (0.907, 0.969)}


def main() -> None:
    hits = {p: [] for p in POLICIES}
    warm_hit = False
    for r in range(16):
        for p in POLICIES:
            js = json.loads((RUN / "gpn" / f"rep_{r:04d}_{p}.json").read_text())
            rounds = [v for v in js["rounds"]["ess_pre"] if v == v]
            warm = [d["ess_pre"] for d in js.get("warmup", {}).get("diag", []) if d.get("ess_pre") == d.get("ess_pre")]
            if min(rounds + warm) < THRESH:
                hits[p].append(r)
                warm_hit |= bool(warm) and min(warm) < THRESH
    excluded = sorted({r for v in hits.values() for r in v})
    keep = [r for r in range(16) if r not in excluded]
    print("runs that hit the rule per acquisition function:", {p: len(v) for p, v in hits.items()},
          "(paper:", PAPER["hits"], ")")
    print(f"excluded true functions: {len(excluded)} {excluded} (paper: {PAPER['n_excluded']})")
    print("a hit during the assimilation of the initial points:", warm_hit, "(paper: none)")
    print("remaining:", keep)

    data = [per_rep(RUN, "gpn", r, C.load_teacher(r)[0]) for r in keep]
    n = np.load(RUN / "gpn" / "rep_0000_ivr.npz")["n_obs"]
    print(f"\nOFML-GPN, IVR, mean over the {len(keep)} remaining true functions; n_obs {int(n[0])} -> {int(n[-1])}")
    cov = []
    for key in ("max_var_with", "max_var_without", "max_err_with", "max_err_without"):
        c = np.stack([d[key] for d in data]).mean(0)
        got = (round(float(c[0]), 2), round(float(c[-1]), 2))
        print(f"  {key:16s} {c[0]:.4f} -> {c[-1]:.4f}   rounded {got[0]:.2f} -> {got[1]:.2f}   paper {PAPER[key]}"
              + ("" if got == PAPER[key] else "   <-- differs"))
    for name in ("with", "without"):
        cov.append(np.stack([d[f"cov95_{name}"] for d in data]).mean(0))
    lo, hi = float(min(c.min() for c in cov)), float(max(c.max() for c in cov))
    got = (round(lo, 3), round(hi, 3))
    print(f"  95% coverage, both sets, all n_obs: [{lo:.4f}, {hi:.4f}]   rounded {got}   paper {PAPER['cov95']}"
          + ("" if got == PAPER["cov95"] else "   <-- differs"))


if __name__ == "__main__":
    main()
