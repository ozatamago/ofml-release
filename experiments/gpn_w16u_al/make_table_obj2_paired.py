"""Appendix table (2026-11-10): paired AULC(T) differences between acquisition policies on PB-W16U, which used to be
the right panel of Figure 4.  Reads ``analysis_16reps/summary.json`` (written by analyze_w16u.py: 16 reps,
OFML-Ens = ens_v2, 95% paired percentile bootstrap over reps with 10,000 resamples).  The difference is
(first policy) - (second policy) with the same surrogate; the count is the number of reps where the first policy has
the smaller AULC.  The right panel also showed SF - Rand, which is Rand - SF with the sign flipped; it is not repeated.

usage: make_table_obj2_paired.py runs/2026-11-04_main [analysis_subdir (default analysis_16reps; analysis_s04 since 2026-11-18)]
output: outputs/tables/table_obj2_paired_{ja,en}.tex (scripts/ofml_paths.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import ofml_paths  # noqa: E402
OUT = ofml_paths.OUT_TABLES
ROWS = [("ivr", "spacefill"), ("maxvar", "spacefill"), ("random", "spacefill"), ("ivr", "random"), ("maxvar", "random")]
NAME = {"ja": {"ivr": "IVR", "maxvar": "MaxVar", "spacefill": "空間充填", "random": "ランダム"},
        "en": {"ivr": "IVR", "maxvar": "MaxVar", "spacefill": "Space-filling", "random": "Random"}}
HEAD = {"ja": ("比較", "差の平均", "95\\% 区間", "本数"), "en": ("Comparison", "Mean", "95\\% interval", "Count")}


def cell(d):
    lo, hi = d["aulc_ci"]
    return f"${d['aulc_mean']:+.3f}$ & $[{lo:+.3f},{hi:+.3f}]$ & {d['aulc_wins']}/{d['n']}"


def main() -> None:
    run = Path(sys.argv[1])
    sub = sys.argv[2] if len(sys.argv) > 2 else "analysis_16reps"   # 2026-11-18: analysis_s04 = OFML-Ens s04
    s = json.loads((run / sub / "summary.json").read_text())
    assert s["reps"] == list(range(16)), s["reps"]
    for lang in ("ja", "en"):
        h, nm = HEAD[lang], NAME[lang]
        body = []
        for p, q in ROWS:
            body.append(f"{nm[p]} $-$ {nm[q]} & " + cell(s["per_model"]["gpn"]["diffs"][f"{p}-{q}"]) + " & "
                        + cell(s["per_model"]["ens"]["diffs"][f"{p}-{q}"]) + " \\\\")
        tex = ("\\begin{tabular}{@{}l rrr rrr@{}}\n\\toprule\n"
               " & \\multicolumn{3}{c}{OFML-GPN} & \\multicolumn{3}{c}{OFML-Ens}\\\\\n"
               "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
               f"{h[0]} & {h[1]} & {h[2]} & {h[3]} & {h[1]} & {h[2]} & {h[3]}\\\\\n\\midrule\n"
               + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")
        (OUT / f"table_obj2_paired_{lang}.tex").write_text(tex)
        print(tex)


if __name__ == "__main__":
    main()
