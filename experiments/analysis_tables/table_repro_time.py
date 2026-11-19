"""Median computation time of one run per surrogate model (Appendix, tab:repro-time).

Definitions (time = wall-clock seconds recorded in the run records):
  Section 6.2, OFML-GPN / Independent-GPN : sum of rounds[].seconds of runs/p1_main/<method>/rep_NN.json
                                             (one round per assimilated observation; OFML-GPN includes D_0)
  Section 6.2, deep ensembles             : 'seconds' of p1_neural/runs/main_2026-11-08/<method>/rep_NN.json
  Section 6.3, OFML-GPN                   : 'policy_seconds' of gpn_w16u_al/runs/2026-11-04_main/gpn/*.json
                                             (excludes the assimilation of the initial points shared by the four
                                             acquisition functions, which is 'warmup.seconds')
  Section 6.3, OFML-Ens                   : 'policy_seconds' of gpn_w16u_al/runs/prestage/prod_eval_fz/s04/ens/*.json
Section 6.2 uses the 12 true functions of the comparison.  The machine of a run is config.host / host; the OFML-GPN
records of Section 6.3 for rep 0-4 lack config.host, and runs/2026-11-04_main/gpn/hosts.json (from the Slurm logs)
gives the machine of every rep.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
P1 = EXP / "gpn_graph" / "runs" / "p1_main"
NN = EXP / "p1_neural" / "runs" / "main_2026-11-08"
W = EXP / "gpn_w16u_al" / "runs"
COMPARED = [2, 4, 5, 9, 11, 12, 15, 17, 19, 23, 27, 28]
DEVICE = {"xeon6130-v100": "V100 / Xeon Gold 6130", "xeon6530-h100nvl": "H100 NVL / Xeon Gold 6530"}
# values printed in arXiv_TMLR/appendix_ja.tex (tab:repro-time), for comparison
PAPER = {("6.2", "OFML-GPN", "xeon6130-v100"): (8, "16.8 h"), ("6.2", "OFML-GPN", "xeon6530-h100nvl"): (4, "8.2 h"),
         ("6.2", "Independent-GPN", "xeon6130-v100"): (8, "4.5 h"),
         ("6.2", "Independent-GPN", "xeon6530-h100nvl"): (4, "1.7 h"),
         ("6.2", "OFML-Ens", "cpu"): (12, "13.7 min"), ("6.2", "Shared-DAG-Ens", "cpu"): (12, "40.7 min"),
         ("6.2", "DAGNN-Cat-Ens", "cpu"): (12, "33.8 min"), ("6.2", "Shared-Ens", "cpu"): (12, "9.5 min"),
         ("6.3", "OFML-GPN", "xeon6130-v100"): (20, "79.7 min"), ("6.3", "OFML-GPN", "xeon6530-h100nvl"): (44, "24.8 min"),
         ("6.3", "OFML-Ens", "cpu"): (64, "13.5 h")}


def fmt(sec: float, unit: str) -> str:
    return f"{sec / 3600:.1f} h" if unit == "h" else f"{sec / 60:.1f} min"


def main() -> None:
    times: dict = defaultdict(list)
    for method, label in (("ofml_gpn", "OFML-GPN"), ("indep_gpn", "Independent-GPN")):
        for r in COMPARED:
            d = json.loads((P1 / method / f"rep_{r:02d}.json").read_text())
            times[("6.2", label, d["config"]["host"])].append(sum(x["seconds"] for x in d["rounds"]))
    for method, label in (("ofml", "OFML-Ens"), ("sharedcell", "Shared-DAG-Ens"), ("dagnn_cat", "DAGNN-Cat-Ens"),
                          ("shared", "Shared-Ens")):
        for r in COMPARED:
            d = json.loads((NN / method / f"rep_{r:02d}.json").read_text())
            assert d["device"] == "cpu" and d["threads"] == 1, (method, r)
            times[("6.2", label, "cpu")].append(d["seconds"])
    hosts = json.loads((W / "2026-11-04_main" / "gpn" / "hosts.json").read_text())["host_by_rep"]
    for p in sorted((W / "2026-11-04_main" / "gpn").glob("rep_*.json")):
        d = json.loads(p.read_text())
        h = hosts[str(d["rep"])]
        assert d["config"].get("host") in (None, h), p
        times[("6.3", "OFML-GPN", h)].append(d["policy_seconds"])
    for p in sorted((W / "prestage" / "prod_eval_fz" / "s04" / "ens").glob("rep_*.json")):
        times[("6.3", "OFML-Ens", "cpu")].append(json.loads(p.read_text())["policy_seconds"])

    print(f"{'section':8s}{'surrogate':18s}{'machine':32s}{'runs':>5s}{'median':>11s}   paper")
    for key, (n_paper, v_paper) in PAPER.items():
        xs = times[key]
        unit = "h" if v_paper.endswith(" h") else "min"
        med = fmt(st.median(xs), unit)
        mark = "" if (len(xs) == n_paper and med == v_paper) else "   <-- differs"
        dev = DEVICE.get(key[2], "CPU, 1 thread (Xeon Gold 6530)")
        print(f"{key[0]:8s}{key[1]:18s}{dev:32s}{len(xs):5d}{med:>11s}   {n_paper} runs, {v_paper}{mark}")


if __name__ == "__main__":
    main()
