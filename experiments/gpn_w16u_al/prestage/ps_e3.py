#!/usr/bin/env python3
"""E3 of REPORTS/ENS_PRESTAGE_PLAN.md: noise of a trial, test of the two paths, test of the lower limit of the L2
coefficient. Writes runs/prestage/e3/summary.json (and runs/prestage/space.json when --write-space is given)."""
import argparse
import numpy as np
import ps_common as P
from ps_common import C

I0 = P.N_KAPPA0 - len(C.INITIAL_FLAT)          # index of n = 50 in the per-round arrays


def prod_summary(run_dir, reps, pols=C.POLICIES):
    """Production-path summary of a setting: per-function AULC per acquisition function, J, kappa-bar, fit."""
    A = {p: {} for p in pols}
    kb, finite, secs, missing = [], True, [], []
    for r in reps:
        for p in pols:
            t = P.prod_trajectory(run_dir, r, p)
            if t is None:
                missing.append((r, p))
                continue
            A[p][r] = t["aulc"]
            kb.append(float(np.mean(t["kappa"][I0:])))
            finite &= t["finite"]
            secs.append(t["seconds"])
    if missing:
        return dict(complete=False, missing=missing)
    aulc = {p: float(np.mean(list(A[p].values()))) for p in pols}
    kbar = float(np.mean(kb))
    kr = float(np.mean([np.mean(P.prod_trajectory(run_dir, r, "random")["kappa"][I0:]) for r in reps])) \
        if "random" in pols else None
    return dict(complete=True, per_function=A, aulc=aulc, J=float(np.mean([aulc[p] for p in pols])),
                J_pass=aulc.get("random"), kappa_bar=kbar, kappa_bar_random=kr, finite=bool(finite),
                fits=bool(finite and kbar <= P.TAU), fits_by_tau={str(t): bool(finite and kbar <= t) for t in P.TAUS},
                seconds_mean=float(np.mean(secs)), seconds_max=float(np.max(secs)))


def tstat(d):
    d = np.asarray(d, float)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-space", action="store_true")
    ap.add_argument("--l2-extra", action="store_true", help="include the test with wd = 1e-8 (after the limit was lowered)")
    a = ap.parse_args()
    e3 = P.RUN / "e3"
    out = dict(tau=P.TAU, trials={}, delta_hat={}, prod={}, path_test={}, l2_test={})
    res = {}
    for nm in ("shallow", "deep", "old"):
        for c in range(4):
            res[nm, c] = P.read_json(e3 / f"{nm}_c{c}" / "result.json")
    res["deep_wd0", 0] = P.read_json(e3 / "deep_wd0_c0" / "result.json")
    for (nm, c), r in res.items():
        out["trials"][f"{nm}_c{c}"] = {k: r.get(k) for k in ("status", "J", "J_pass", "aulc", "kappa_bar", "kappa_bar_random",
                                                           "fits", "fits_pass", "fits_by_tau", "seconds", "resumes", "split")}
        print(f"trial {nm:9s} c={c} status {r['status']} J {r['J']:.4f} J_pass {r['J_pass']:.4f} kappa_bar {r['kappa_bar']:.3f} "
              f"fits {r['fits']} aulc " + " ".join(f"{p} {r['aulc'][p]:.4f}" for p in C.POLICIES) + f" seconds {r['seconds']:.0f}")
    for nm in ("shallow", "deep", "old"):
        out["delta_hat"][nm] = {k: float(np.std([res[nm, c][k] for c in range(4)], ddof=1)) for k in ("J", "J_pass")}
        out["delta_hat"][nm]["mean_J"] = float(np.mean([res[nm, c]["J"] for c in range(4)]))
        out["delta_hat"][nm]["mean_J_pass"] = float(np.mean([res[nm, c]["J_pass"] for c in range(4)]))
        print(f"delta-hat {nm}: J {out['delta_hat'][nm]['J']:.5f} J_pass {out['delta_hat'][nm]['J_pass']:.5f}")
    out["delta_gp"] = {k: 0.5 * min(out["delta_hat"]["shallow"][k], out["delta_hat"]["deep"][k]) for k in ("J", "J_pass")}
    print("delta_GP", out["delta_gp"])

    # test of the two paths
    ok_model, ok_fixed, complete = True, True, True
    for nm in ("shallow", "deep", "old"):
        ps = prod_summary(P.RUN / "prod_tune" / nm, P.R_TUNE)
        if not ps["complete"]:
            print(f"production path of {nm}: {len(ps['missing'])} trajectories missing")
            complete = False
            continue
        out["prod"][nm] = {k: v for k, v in ps.items() if k != "per_function"}
        print(f"production {nm}: J {ps['J']:.4f} J_pass {ps['J_pass']:.4f} kappa_bar {ps['kappa_bar']:.3f} fits {ps['fits']} "
              f"seconds/trajectory mean {ps['seconds_mean']:.0f} max {ps['seconds_max']:.0f}")
        bun = {(t["rep"], t["policy"]): t["aulc"] for t in res[nm, 0]["trajs"]}
        for p in C.POLICIES:
            D = np.array([ps["per_function"][p][r] - bun[r, p] for r in P.R_TUNE])
            row = dict(mean=float(D.mean()), sd=float(D.std(ddof=1)), max_abs=float(np.abs(D).max()))
            if p in ("ivr", "maxvar"):
                row["t"] = tstat(D)
                row["passed"] = bool(abs(row["t"]) <= 3.04)
                ok_model &= row["passed"]
            else:
                row["passed"] = bool(row["max_abs"] <= 0.002)
                ok_fixed &= row["passed"]
            out["path_test"][f"{nm}_{p}"] = row
            print(f"  path test {nm:8s} {p:9s} mean D {row['mean']:+.5f} sd {row['sd']:.5f} max|D| {row['max_abs']:.5f} "
                  + (f"t {row['t']:+.2f} " if "t" in row else "") + f"pass {row['passed']}")
    out["path_test_passed"] = bool(complete and ok_model and ok_fixed)
    out["path_complete"] = complete

    # lower limit of the L2 coefficient
    j0, j1 = P.per_function_J(res["deep", 0]), P.per_function_J(res["deep_wd0", 0])
    d = np.array([j1[r] - j0[r] for r in P.R_TUNE])
    t = tstat(d)
    keep = abs(t) <= 2.13
    out["l2_test"] = {"mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)), "se": float(d.std(ddof=1) / 4), "t": t,
                      "keep_1e-6": bool(keep)}
    print(f"L2 test (wd 0 minus wd 1e-6, deep reference): mean {d.mean():+.5f} SE {d.std(ddof=1) / 4:.5f} t {t:+.2f} -> "
          f"{'keep 1e-6' if keep else 'lower limit 1e-8'}")
    f8 = e3 / "deep_wd1e-8_c0" / "result.json"
    if f8.exists():
        j8 = P.per_function_J(P.read_json(f8))
        d8 = np.array([j1[r] - j8[r] for r in P.R_TUNE])
        out["l2_test"]["extra_1e-8"] = dict(mean_diff=float(d8.mean()), se=float(d8.std(ddof=1) / 4), t=tstat(d8))
        print(f"L2 test (wd 0 minus wd 1e-8): mean {d8.mean():+.5f} t {tstat(d8):+.2f}")
    sp = dict(P.DEFAULT_SPACE)
    if not keep:
        sp.update(wd_base=1e-8, u6_div=6.0)
    out["space"] = sp
    C.atomic_write_json(e3 / "summary.json", out)
    if a.write_space:
        f = P.RUN / "space.json"
        if f.exists():
            print(f"{f} exists; not rewritten")
        else:
            C.atomic_write_json(f, sp)
            print(f"wrote {f}: {sp}")
    print(f"PS_E3_DONE path_test_passed={out['path_test_passed']} keep_1e-6={keep}")


if __name__ == "__main__":
    main()
