"""Extract the per-observation records of the stopped runs of Section 6.2 from their checkpoints.

The runs of the first 16 true functions that were stopped before T=200 (OFML-GPN rep 0, 3, 6, 7, 10, 13, 14 and
Independent-GPN rep 3, 6, 7) have no ``rep_NN.json``; their per-observation records are stored only as the string
``rounds_json`` inside ``<method>/ckpt/rep_NN.npz``.  The checkpoints (several GB in total) are not published.  This
script copies, for each such run, the fields needed for the exclusion rule (``n``, ``cell``, ``phase``,
``ess_pre_over_P``, ``seconds``) into ``runs/p1_main/<method>/stopped/rep_NN_rounds.json``.

It was run once by the authors on the original run directory:

    python extract_stopped_rounds.py --src <original runs/p1_main> --dst <this repository>/transfer/gpn/runs/p1_main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEEP = ("n", "cell", "phase", "ess_pre_over_P", "seconds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original runs/p1_main directory that contains <method>/ckpt/")
    ap.add_argument("--dst", required=True, help="runs/p1_main directory of this repository")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    for method in ("ofml_gpn", "indep_gpn"):
        done = {p.stem for p in (src / method).glob("rep_*.json")}
        for ck in sorted((src / method / "ckpt").glob("rep_*.npz")):
            if ck.stem in done:
                continue          # a finished run: its rep_NN.json is published
            d = np.load(ck, allow_pickle=False)
            rounds = json.loads(str(d["rounds_json"]))
            out = {"method": method, "rep": int(ck.stem.split("_")[1]), "status": "stopped",
                   "source": "checkpoint rounds_json",
                   "rounds": [{k: r[k] for k in KEEP if k in r} for r in rounds]}
            p = dst / method / "stopped" / f"{ck.stem}_rounds.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out, indent=1))
            print(p, len(rounds))


if __name__ == "__main__":
    main()
