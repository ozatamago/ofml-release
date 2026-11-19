"""Values of fig:exp-acquisition (b) quoted in the text of Section 6.3 (16 true functions, IVR).

For OFML-GPN and OFML-Ens (setting s04), prints the mean over the 16 true functions of the maximum posterior variance,
the maximum absolute error of the posterior mean and the 95% coverage, on the 8 protocols with an initial point and on
the 8 without, at 8 and at 100 observations and their range over all numbers of observations (the same functions as
plot_w16u_uniform_split.py, which draws the panel). It then checks the two statements of the text on OFML-Ens:
the maximum error does not fall below 0.96 up to 100 observations, and the 95% coverage stays between 0.07 and 0.26.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import w16u_common as C  # noqa: E402
from plot_w16u_uniform_split import per_rep  # noqa: E402

RUN = HERE / "runs" / "2026-11-04_main"
MODELS = (("OFML-GPN", "gpn"), ("OFML-Ens", "../prestage/prod_eval_fz/s04/ens"))


def main() -> None:
    curves = {}
    for label, d in MODELS:
        data = [per_rep(RUN, d, r, C.load_teacher(r)[0]) for r in range(16)]
        print(label)
        for key, name in (("max_var", "max posterior variance"), ("max_err", "max absolute error"), ("cov95", "95% coverage")):
            for part in ("with", "without"):
                c = np.stack([x[f"{key}_{part}"] for x in data]).mean(0)
                curves[(label, key, part)] = c
                print(f"  {name:22s} {part:7s} initial point: n=8 {c[0]:.3f}  n=100 {c[-1]:.3f}  range [{c.min():.3f}, {c.max():.3f}]")
    err_min = min(curves[("OFML-Ens", "max_err", p)].min() for p in ("with", "without"))
    cov = np.concatenate([curves[("OFML-Ens", "cov95", p)] for p in ("with", "without")])
    ok1 = round(float(err_min), 2) >= 0.96
    ok2 = (round(float(cov.min()), 2), round(float(cov.max()), 2)) == (0.07, 0.26)
    print(f"\nOFML-Ens: smallest max absolute error {err_min:.3f} (paper: does not fall below 0.96)" + ("" if ok1 else "   <-- differs"))
    print(f"OFML-Ens: 95% coverage between {cov.min():.3f} and {cov.max():.3f} (paper: 0.07 to 0.26)" + ("" if ok2 else "   <-- differs"))
    print("all values match the paper" if ok1 and ok2 else "some values differ from the paper")


if __name__ == "__main__":
    main()
