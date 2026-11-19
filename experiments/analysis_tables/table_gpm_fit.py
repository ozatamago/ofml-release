"""Fit of the deep ensembles of Section 6.2 to their training points (Appendix, tab:gpm-fit): means over the 12 true
functions of the comparison.

Input: the outputs of experiments/analysis_transfer/p1_fit_check.py, one block per (method, rep):
  analysis_transfer/fit12/first4_reps_2_11_17_28.txt   rep 2, 11, 17, 28 (all four methods, one file)
  analysis_transfer/fit12/<method>_r<rep>.txt          the other 8 reps (commands in fit12/cmds.txt)
Each block holds kappa_t and NRMSE(t) at t = 0, 50, 100, 200 for 400 and 4000 Adam steps.  kappa_t is the RMSE of
the posterior mean at the training points divided by the noise standard deviation 0.1.
The inputs are printed with 2 (kappa) or 3 (NRMSE) decimals; the means are computed in decimal arithmetic and rounded half
up (e.g. the mean 0.3335 of OFML-Ens is printed as 0.334).
"""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIT = ROOT / "experiments" / "analysis_transfer" / "fit12"
COMPARED = [2, 4, 5, 9, 11, 12, 15, 17, 19, 23, 27, 28]
METHODS = [("ofml", "OFML-Ens"), ("sharedcell", "Shared-DAG-Ens"), ("dagnn_cat", "DAGNN-Cat-Ens"), ("shared", "Shared-Ens")]
# values printed in arXiv_TMLR/appendix_ja.tex (tab:gpm-fit)
PAPER = {"ofml": "0.09 0.15 0.49 0.83 0.64 0.312 0.334", "sharedcell": "0.34 0.42 0.75 1.03 1.04 0.302 0.347",
         "dagnn_cat": "0.13 0.17 0.44 1.10 1.15 0.325 0.354", "shared": "1.70 1.51 1.60 1.68 1.50 0.723 0.724"}
HEAD = re.compile(r"^(\w+)\s+rep\s+(\d+)\s")
LINE = re.compile(r"steps\s+(\d+):(.*)")
CELL = re.compile(r"t=\s*(\d+) kappa\s+([\d.]+) nrmse\s+([\d.]+)")


def parse() -> dict:
    res: dict = {}
    for p in sorted(FIT.glob("*.txt")):
        if p.name == "cmds.txt":
            continue
        key = None
        for line in p.read_text().splitlines():
            m = HEAD.match(line)
            if m:
                key = (m.group(1), int(m.group(2)))
                continue
            m = LINE.search(line)
            if m and key:
                steps = int(m.group(1))
                for t, k, n in CELL.findall(m.group(2)):
                    res[(key[0], key[1], steps, int(t))] = (Decimal(k), Decimal(n))
    return res


def main() -> None:
    res = parse()
    print(f"{'method':16s}{'k0':>6s}{'k50':>6s}{'k100':>6s}{'k200':>6s}{'k200@4000':>11s}{'nrmse@400':>11s}{'nrmse@4000':>11s}")
    bad = 0
    for m, label in METHODS:
        reps = [r for r in COMPARED if (m, r, 400, 200) in res]
        assert reps == COMPARED, (m, sorted(set(COMPARED) - set(reps)))
        mean = lambda f: sum(f(r) for r in COMPARED) / Decimal(len(COMPARED))  # noqa: E731
        q = lambda x, d: x.quantize(Decimal(d), rounding=ROUND_HALF_UP)  # noqa: E731
        v = [mean(lambda r, t=t: res[(m, r, 400, t)][0]) for t in (0, 50, 100, 200)]
        v.append(mean(lambda r: res[(m, r, 4000, 200)][0]))
        v.append(mean(lambda r: res[(m, r, 400, 200)][1]))
        v.append(mean(lambda r: res[(m, r, 4000, 200)][1]))
        s = " ".join([str(q(x, "0.01")) for x in v[:5]] + [str(q(x, "0.001")) for x in v[5:]])
        ok = s == PAPER[m]
        bad += not ok
        w = s.split()
        print(f"{label:16s}" + "".join(f"{x:>6s}" for x in w[:4]) + f"{w[4]:>11s}{w[5]:>11s}{w[6]:>11s}"
              + ("" if ok else f"   <-- paper: {PAPER[m]}"))
    print("\nall values match the paper" if bad == 0 else f"\n{bad} row(s) differ from the paper")


if __name__ == "__main__":
    main()
