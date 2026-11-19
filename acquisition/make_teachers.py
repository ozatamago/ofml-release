#!/usr/bin/env python3
"""Write the PB-W16U teacher cache (teachers/rep_XXXX.npz) with u15_compare's OWN functions, and check

(a) truth/noise from u15_compare.draw_family_teacher == the re-implementation in w16u_common (bit-identical),
    and == the `truth` stored in the ProtocolUCB BENCH9 production npz where one exists;
(b) the initial design (u15_compare's run_replicate_all_arms recipe) == w16u_common.INITIAL_FLAT, and == the
    production config_json's `initial_flat`.

cx_bayes is only imported (read-only). Run with the cx_bayes venv; CPU is enough:
    JAX_PLATFORMS=cpu ~/program/cx_bayes/ucb/.venv/bin/python make_teachers.py --reps 0-15,1000-1007
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w16u_common as C  # noqa: E402

PROD_DIRS = [C.CX_BAYES / "ucb/noisy_ucb_experiments/results/u15_comparison/bench9_t100_2026-11-02/shared_gpn",
             C.CX_BAYES / "ucb/noisy_ucb_experiments/results/u15_comparison/bench9_t100_2026-11-02/indep_gpn_f"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0-15,1000-1007")
    ap.add_argument("--report", default="", help="optional JSON file for the test results")
    args = ap.parse_args()
    sys.path.insert(0, str(C.U15_EXPERIMENTS))
    sys.path.insert(0, str(C.CX_BAYES))
    import u15_compare as U  # noqa: E402
    import u11_exact_gpn_ucb_initial_design as base  # noqa: E402  (u15_compare.run_worker's base module)

    entry = U.FAMILY_REGISTRY[C.FAMILY]
    assert entry["family_index"] == C.FAMILY_INDEX and U.family_n_protocols(entry) == C.N_PROTOCOLS
    assert U.family_protocol_order(entry) == "l1_major" and U.family_n_l2_types(entry) == C.N_L2
    src = json.loads(U.DEFAULT_GRID_SOURCE.read_text())
    x_u10 = np.asarray(src["teacher"]["x"], dtype=np.float64)
    warped = U.resolve_warped_x(U.family_grid(entry), np.asarray(src["teacher"]["warped_x"], dtype=np.float64))
    assert np.array_equal(warped, C.WARPED_X), "uniform grid differs"
    # (b) initial design, exactly as run_replicate_all_arms builds it.
    grid_indices = base.initial_indices(x_u10, U.family_initial_points_per_protocol(entry))
    initial_flat = [p * U.GRID_SIZE_PER_PROTOCOL + idx for p in U.family_initial_protocols(entry) for idx in grid_indices]
    out = dict(initial_flat_u15=initial_flat, initial_flat_ours=list(C.INITIAL_FLAT),
               initial_match=initial_flat == list(C.INITIAL_FLAT), grid_index=grid_indices, reps={})
    print(f"(b) initial design u15={initial_flat} ours={list(C.INITIAL_FLAT)} match={out['initial_match']}")
    assert out["initial_match"]

    C.TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    for rep in C.parse_range(args.reps):
        ts = U.SEED_BASE + U.FAMILY_INDEX_SEED_SPACING * entry["family_index"] + 10 * rep
        assert (ts, ts + 1, ts + 2) == C.teacher_seeds(rep)
        truth, noise = U.draw_family_teacher(base, C.FAMILY, entry, warped, 1e-8, ts, ts + 1, 0.10)
        t2, n2 = C.draw_teacher_reimpl(rep)
        rec = dict(teacher_seed=ts, reimpl_bit_identical=bool(np.array_equal(truth, t2) and np.array_equal(noise, n2)))
        for d in PROD_DIRS:
            f = d / f"{C.FAMILY}_r{rep:04d}" / C.FAMILY / d.name / f"rep_{rep:04d}.npz"
            if f.exists():
                with np.load(f, allow_pickle=False) as z:
                    cfg = json.loads(str(z["config_json"]))
                    rec[f"prod_{d.name}_truth_bit_identical"] = bool(np.array_equal(z["truth"], truth))
                    rec[f"prod_{d.name}_initial_flat_match"] = cfg["initial_flat"] == list(C.INITIAL_FLAT)
                    rec[f"prod_{d.name}_seed_match"] = cfg["seeds"]["teacher_seed"] == ts
        digest = C.sha256_arrays(truth, noise)
        path = C.teacher_path(rep)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                assert str(z["sha256"]) == digest, f"{path} exists with different content"
        else:
            tmp = path.with_name("." + path.stem + ".tmp.npz")
            np.savez(tmp, truth=truth, noise=noise, sha256=np.asarray(digest), teacher_seed=np.asarray(ts),
                     noise_seed=np.asarray(ts + 1), family=np.asarray(C.FAMILY))
            os.replace(tmp, path)
        rec["sha256"] = digest
        out["reps"][rep] = rec
        print(f"rep {rep}: {rec}")
        assert rec["reimpl_bit_identical"]
        assert all(v for k, v in rec.items() if k.startswith("prod_"))
    if args.report:
        Path(args.report).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
