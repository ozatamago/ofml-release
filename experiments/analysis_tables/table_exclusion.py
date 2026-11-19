"""Per-true-function exclusion decisions of Section 6.2 (Appendix, tab:gpm-exclusion).

Rule (fixed before the results were seen): for OFML-GPN (16384 particles, including the assimilation of D_0) and
Independent-GPN (65536 particles per particle system), the effective sample size divided by the number of particles
is recorded after each observation is assimilated, before the resampling decision (``ess_pre_over_P``).  A true
function for which this value falls below 0.001 at least once for either model is excluded from the comparison of
all models.

Inputs (all in this repository):
  runs/p1_main/<method>/rep_NN.json                finished runs (also runs stopped by the rule, status "excluded")
  runs/p1_main/<method>/stopped/rep_NN_rounds.json runs of the first 16 true functions that were stopped early

Outputs:
  runs/p1_main/exclusion/judgments.csv   one row per drawn true function (rep 0-28) and the rep 29-33 run after
                                         the five replacements had been collected
  stdout                                 the counts of tab:gpm-exclusion
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments" / "gpn_graph" / "runs" / "p1_main"
THRESHOLD = 1e-3
INITIAL = list(range(16))
REPLACEMENT = list(range(16, 29))       # drawn in order until five comparable true functions were collected
AFTER = list(range(29, 34))             # run after the five had been collected; not counted in the table
COMPARED_EXPECTED = [2, 4, 5, 9, 11, 12, 15, 17, 19, 23, 27, 28]


def load_rounds(method: str, rep: int):
    p = RUN / method / f"rep_{rep:02d}.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d["status"], d["rounds"], "rep_json"
    p = RUN / method / "stopped" / f"rep_{rep:02d}_rounds.json"
    if p.exists():
        return "stopped", json.loads(p.read_text())["rounds"], "stopped_rounds"
    return None, None, None


def judge(rounds):
    """Return (minimum ratio, phase of the first observation below the threshold or '')."""
    vals = [(r["ess_pre_over_P"], r.get("phase", "")) for r in rounds if r.get("ess_pre_over_P") is not None]
    if not vals:
        return float("nan"), ""
    first = next((ph for v, ph in vals if v < THRESHOLD), "")
    return min(v for v, _ in vals), first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUN / "exclusion" / "judgments.csv"))
    a = ap.parse_args()
    rows = []
    for rep in INITIAL + REPLACEMENT + AFTER:
        row = {"rep": rep, "group": "initial" if rep in INITIAL else ("replacement" if rep in REPLACEMENT else "after")}
        for method, key in (("ofml_gpn", "ofml"), ("indep_gpn", "indep")):
            status, rounds, src = load_rounds(method, rep)
            if rounds is None:
                row.update({f"{key}_record": "not_run", f"{key}_min_ratio": "", f"{key}_hit": "", f"{key}_first_hit_phase": ""})
                continue
            mn, first = judge(rounds)
            row.update({f"{key}_record": f"{src}:{status}", f"{key}_min_ratio": f"{mn:.6g}",
                        f"{key}_hit": int(bool(first)), f"{key}_first_hit_phase": first})
        row["compared"] = int(row["ofml_hit"] == 0 and row["indep_hit"] == 0)
        rows.append(row)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    def counts(group):
        rs = [r for r in rows if r["group"] == group]
        return (len(rs),
                sum(r["ofml_first_hit_phase"] == "source" for r in rs),
                sum(r["ofml_first_hit_phase"] == "target" for r in rs),
                sum(r["indep_hit"] == 1 for r in rs),
                sum(r["compared"] for r in rs))
    head = ("drawn", "OFML-GPN hit in D_0", "OFML-GPN hit in target", "Independent-GPN hit", "compared")
    print("group        " + " | ".join(head))
    tot = [0] * 5
    for g in ("initial", "replacement"):
        c = counts(g)
        tot = [x + y for x, y in zip(tot, c)]
        print(f"{g:12s} " + " | ".join(str(x) for x in c))
    print(f"{'total':12s} " + " | ".join(str(x) for x in tot))
    compared = [r["rep"] for r in rows if r["compared"] and r["group"] != "after"]
    print("compared reps:", compared, "(matches the paper)" if compared == COMPARED_EXPECTED else "(DIFFERS from the paper)")
    both = [r["rep"] for r in rows if r["indep_hit"] == 1 and r["ofml_hit"] == 1]
    print("hit by Independent-GPN and also by OFML-GPN:", both)
    print("after the five replacements:", [(r["rep"], "compared" if r["compared"] else "excluded") for r in rows if r["group"] == "after"])


if __name__ == "__main__":
    main()
