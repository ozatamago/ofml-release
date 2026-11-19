"""AULC(T) of OFML-Ens per setting on the 16 evaluation true functions (Appendix, tab:acq-ens-settings and the
differences IVR - space-filling with 95% intervals in the text of sec:acq-ens-settings).

The per-setting statistics are those of acquisition/analyze_w16u.py (mean over the 16 true functions; the interval is
the 95% paired percentile bootstrap over true functions with 10,000 resamples).  If a summary does not exist yet, this
script runs analyze_w16u.py for it.

  s04       runs/prestage/prod_eval_fz/s04/ens       (the setting of the main text)
  p02       runs/prestage/prod_eval_fz/p02/ens
  shallow   runs/prestage/prod_eval_fz/shallow/ens   (浅い参照, the shallow reference)
  old       runs/2026-11-04_main/ens_v2              (旧設定, the old setting)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "acquisition"
RUN = W / "runs" / "2026-11-04_main"
SETTINGS = [("s04", "../prestage/prod_eval_fz/s04/ens", "analysis_s04"),
            ("p02", "../prestage/prod_eval_fz/p02/ens", "analysis_p02"),
            ("shallow", "../prestage/prod_eval_fz/shallow/ens", "analysis_shallow"),
            ("old", "ens_v2", "analysis_16reps")]
POL = ("ivr", "maxvar", "spacefill", "random")
# values printed in arXiv_TMLR/appendix_ja.tex
PAPER_AULC = {"s04": (0.737, 0.739, 0.809, 0.867, 0.788), "p02": (0.931, 0.968, 0.793, 0.901, 0.898),
              "shallow": (0.699, 0.699, 0.877, 0.967, 0.810), "old": (0.898, 0.919, 0.779, 0.847, 0.861),
              "OFML-GPN": (0.479, 0.546, 0.645, 0.664, 0.584)}
PAPER_DIFF = {"s04": (-0.072, -0.125, -0.022), "shallow": (-0.178, -0.213, -0.144), "p02": (+0.137, +0.055, +0.217),
              "old": (+0.119, +0.076, +0.168)}


def summary(ens_dir: str, sub: str) -> dict:
    p = RUN / sub / "summary.json"
    if not p.exists():
        subprocess.run([sys.executable, str(W / "analyze_w16u.py"), str(RUN), "--ens-dir", ens_dir,
                        "--outdir", str(RUN / sub)], check=True, cwd=W, stdout=subprocess.DEVNULL)
    s = json.loads(p.read_text())
    assert s["reps"] == list(range(16)), s["reps"]
    return s


def main() -> None:
    bad = 0
    print(f"{'setting':10s}" + "".join(f"{p:>11s}" for p in POL) + f"{'mean':>11s}")
    rows = {}
    for name, ens_dir, sub in SETTINGS:
        s = summary(ens_dir, sub)
        rows[name] = s
        v = [s["per_model"]["ens"][p]["aulc_mean"] for p in POL]
        v.append(sum(v) / 4)
        ok = all(f"{a:.3f}" == f"{b:.3f}" for a, b in zip(v, PAPER_AULC[name]))
        bad += not ok
        print(f"{name:10s}" + "".join(f"{x:11.3f}" for x in v) + ("" if ok else "   <-- differs from the paper"))
    g = [rows["s04"]["per_model"]["gpn"][p]["aulc_mean"] for p in POL]
    g.append(sum(g) / 4)
    ok = all(f"{a:.3f}" == f"{b:.3f}" for a, b in zip(g, PAPER_AULC["OFML-GPN"]))
    bad += not ok
    print(f"{'OFML-GPN':10s}" + "".join(f"{x:11.3f}" for x in g) + ("" if ok else "   <-- differs from the paper"))
    print("\nIVR - space-filling (OFML-Ens): mean [95% interval]")
    for name in ("s04", "shallow", "p02", "old"):
        d = rows[name]["per_model"]["ens"]["diffs"]["ivr-spacefill"]
        v = (d["aulc_mean"], *d["aulc_ci"])
        ok = all(f"{a:+.3f}" == f"{b:+.3f}" for a, b in zip(v, PAPER_DIFF[name]))
        bad += not ok
        print(f"{name:10s}{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" + ("" if ok else "   <-- differs from the paper"))
    print("\nall values match the paper" if bad == 0 else f"\n{bad} line(s) differ from the paper")


if __name__ == "__main__":
    main()
