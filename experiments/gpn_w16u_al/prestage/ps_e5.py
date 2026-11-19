#!/usr/bin/env python3
"""E5 and the confirmation of E6 of REPORTS/ENS_PRESTAGE_PLAN.md (sections 4.6, E5, E6).

  prep --camp e4 : the set S of runs/prestage/e4/final.json and the old setting -> production configs, the command
                   file of the production path on R_conf (64 trajectories per setting, CPU) and the command file of
                   the bundle path on R_conf (GPU; reported only, rule 2).
  prep --camp e6 : the 5 settings of runs/prestage/e6/final.json -> command file of the random trajectories on R_conf
                   by the production path (settings that E5 already computes are not computed twice).
  sum  --camp e4 : psi-star (plan 4.6), C1, C2, difference of the two paths.  Writes runs/prestage/e5/summary.json.
  sum  --camp e6 : psi-pass (plan E6 step 3).  Writes runs/prestage/e6/confirm.json.
Run from the directory of this file.
"""
import argparse
import json
import numpy as np
import ps_common as P
from ps_common import C
from ps_e3 import prod_summary

E5 = P.RUN / "e5"
# 2026-11-15: the production path flushes subnormal numbers to zero (prestage/fz.py; author's decision, REPORTS/修正記録.md).
# Its results are in prod_conf_fz. The directory prod_conf holds the partial results of the path without the flush.
CONF = P.RUN / "prod_conf_fz"
REPS = f"{P.R_CONF[0]}-{P.R_CONF[-1]}"


def prod_cmd(name: str, r: int, p: str) -> str:
    return (f"$PY prestage/fz.py w16u_ens.py --reps {r} --policies {p} --out runs/prestage/{CONF.name}/{name} "
            f"--config runs/prestage/configs/e5_{name}.json --device cpu --threads 1 --tag r{r}_{p} --model-dir ens "
            f"--skip-existing >> runs/prestage/{CONF.name}/logs/{name}_r{r}_{p}.log 2>&1")


def cmd_prep(a) -> None:
    tcpu = P.load_tcpu()
    (CONF / "logs").mkdir(parents=True, exist_ok=True)
    (E5 / "psi").mkdir(parents=True, exist_ok=True)
    (E5 / "logs").mkdir(parents=True, exist_ok=True)
    mf = E5 / "manifest.json"
    if a.camp == "e4":
        assert not mf.exists(), f"{mf} exists"
        fin = P.read_json(P.RUN / "e4" / "final.json")
        man = []
        for i, c in enumerate(fin["S"]):
            name = {"ref_shallow": "shallow", "ref_deep": "deep"}.get(c["src"], f"s{i + 1:02d}")
            man.append(dict(name=name, key=c["key"], psi=P.norm_psi(c["psi"]), why=c["why"], src=c["src"], no=c["no"],
                            zhat=c["zhat"], val=c["val"], in_S=True))
        assert P.key_of(P.REFS["old"]) not in {m["key"] for m in man}
        man.append(dict(name="old", key=P.key_of(P.REFS["old"]), psi=P.norm_psi(P.REFS["old"]), why=["old"], src="old",
                        no=0, zhat=None, val=None, in_S=False))
        assert len({m["key"] for m in man}) == len(man) and len(man) <= 12
        for m in man:
            m["t_est"] = 0.0 if m["name"] == "old" else P.t_est(m["psi"], tcpu)   # the old setting is outside Psi
            C.atomic_write_json(P.RUN / "configs" / f"e5_{m['name']}.json", P.prod_config(m["psi"]))
            C.atomic_write_json(E5 / "psi" / f"{m['name']}.json", m["psi"])
        order = sorted(man, key=lambda m: -m["t_est"])
        prod = [prod_cmd(m["name"], r, p) for m in order for r in P.R_CONF for p in C.POLICIES]
        gpu = [f"cd prestage && $PY -u ps_trial.py --psi ../runs/prestage/e5/psi/{m['name']}.json --reps {REPS} --c 0 "
               f"--device cuda --out ../runs/prestage/e5/bundle/{m['name']} > ../runs/prestage/e5/logs/bundle_{m['name']}.log 2>&1"
               for m in order]
        (E5 / "prod.cmds").write_text("\n".join(prod) + "\n")
        (E5 / "gpu.cmds").write_text("\n".join(gpu) + "\n")
        C.atomic_write_json(mf, man)
        for m in man:
            print(f"{m['name']:8s} {m['why']} src {m['src']} no {m['no']} est {m['t_est'] / 3600:.2f} h {m['psi']}")
        print(f"PS_E5_PREP_DONE settings={len(man)} prod_lines={len(prod)} gpu_lines={len(gpu)}")
    else:
        man5 = {m["key"]: m for m in P.read_json(mf)}
        fin = P.read_json(P.RUN / "e6" / "final.json")
        assert len(fin["S"]) <= 5
        man, prod = [], []
        for i, c in enumerate(fin["S"]):
            shared = man5.get(c["key"])
            name = shared["name"] if shared else f"p{i + 1:02d}"
            m = dict(name=name, key=c["key"], psi=P.norm_psi(c["psi"]), why=c["why"], src=c["src"], no=c["no"],
                     zhat=c["zhat"], val=c["val"], shared_with_e5=bool(shared), t_est=P.t_est(c["psi"], tcpu))
            man.append(m)
            if not shared:
                C.atomic_write_json(P.RUN / "configs" / f"e5_{name}.json", P.prod_config(m["psi"]))
                prod += [prod_cmd(name, r, "random") for r in P.R_CONF]
            print(f"{name:8s} {m['why']} src {m['src']} no {m['no']} shared {bool(shared)} {m['psi']}")
        (P.RUN / "e6" / "prod.cmds").write_text("\n".join(prod) + ("\n" if prod else ""))
        C.atomic_write_json(P.RUN / "e6" / "manifest.json", man)
        print(f"PS_E6_PREP_DONE settings={len(man)} prod_lines={len(prod)}")


def cmd_cmds(a) -> None:
    """Command file of the production path of E5 from the existing manifest (no other file is changed)."""
    assert a.camp == "e4"
    (CONF / "logs").mkdir(parents=True, exist_ok=True)
    order = sorted(P.read_json(E5 / "manifest.json"), key=lambda m: -m["t_est"])
    prod = [prod_cmd(m["name"], r, p) for m in order for r in P.R_CONF for p in C.POLICIES]
    f = E5 / f"{CONF.name}.cmds"
    f.write_text("\n".join(prod) + "\n")
    print(f"PS_E5_CMDS_DONE {f} lines={len(prod)}")


def cmd_sum(a) -> None:
    import ps_search as S
    if a.camp == "e4":
        man = P.read_json(E5 / "manifest.json")
        fin = P.read_json(P.RUN / "e4" / "final.json")
        sp = P.load_space()
        rows, incomplete = [], []
        for m in man:
            ps = prod_summary(CONF / m["name"], P.R_CONF)
            if not ps["complete"]:
                incomplete.append((m["name"], len(ps["missing"])))
                continue
            row = dict(m, J_cpu=ps["J"], J_pass_cpu=ps["J_pass"], aulc_cpu=ps["aulc"], kappa_bar_cpu=ps["kappa_bar"],
                       finite=ps["finite"], fits_conf=ps["fits"], fits_by_tau=ps["fits_by_tau"],
                       seconds_mean=ps["seconds_mean"], seconds_max=ps["seconds_max"],
                       edge=S.edge_conditions(m["psi"], sp))
            fb = E5 / "bundle" / m["name"] / "result.json"
            if fb.exists():
                b = P.read_json(fb)
                row.update(J_bundle_conf=b.get("J"), bundle_status=b.get("status"), bundle_kappa_bar=b.get("kappa_bar"),
                           path_diff=None if b.get("J") is None else ps["J"] - b["J"])
            rows.append(row)
        if incomplete:
            print("incomplete:", incomplete)
        cand = [r for r in rows if r["in_S"] and r["fits_conf"]]
        cand.sort(key=lambda r: (r["J_cpu"], np.inf if r["zhat"] is None else r["zhat"]))
        out = dict(tau=P.TAU, complete=not incomplete, incomplete=incomplete, rows=rows,
                   production_path=f"prestage/fz.py w16u_ens.py (subnormal numbers flushed to zero), results in {CONF.name}")
        for r in sorted(rows, key=lambda r: r["J_cpu"]):
            print(f"{r['name']:8s} {'+'.join(r['why']):24s} src {r['src']:11s} J_cpu(conf) {r['J_cpu']:.4f} "
                  f"kappa_bar {r['kappa_bar_cpu']:.3f} fits {r['fits_conf']} J(tune) "
                  f"{'-' if r['val'] is None else format(r['val'], '.4f')} zhat "
                  f"{'-' if r['zhat'] is None else format(r['zhat'], '.4f')} bundle(conf) "
                  f"{'-' if r.get('J_bundle_conf') is None else format(r['J_bundle_conf'], '.4f')} "
                  f"s/traj {r['seconds_mean']:.0f} edge {r['edge']} "
                  + " ".join(f"{p} {r['aulc_cpu'][p]:.4f}" for p in C.POLICIES))
        if not incomplete:
            if cand:
                st = cand[0]
                out["psi_star"] = st
                out["rule9_failed"] = False
                out["C1"] = bool(st["val"] is not None and st["fits_conf"])
                c2 = dict(rho_loo=fin["rho_loo"], first=bool(fin["rho_loo"] >= 0.5), second=bool(fin["c2_second"]),
                          third=bool(st["src"] == "ei"), fourth=bool(not st["edge"]))
                c2["all"] = bool(c2["first"] and c2["second"] and c2["third"] and c2["fourth"])
                out["C2"] = c2
                tune_best = min((r for r in rows if r["in_S"] and r["val"] is not None), key=lambda r: r["val"])
                out["h1_tune_best_differs"] = bool(tune_best["key"] != st["key"])
                out["second_best_gap"] = None if len(cand) < 2 else cand[1]["J_cpu"] - st["J_cpu"]
                print(f"psi-star: {st['name']} {st['psi']} src {st['src']} J_cpu(conf) {st['J_cpu']:.4f}")
                print(f"C1 {out['C1']}  C2 {c2}  gap to the second {out['second_best_gap']}")
            else:
                out["psi_star"], out["rule9_failed"] = None, True
                print("no setting of S fits on R_conf by the production path: the pre-stage fails (rule 9)")
        C.atomic_write_json(E5 / "summary.json", out)
        print(f"PS_E5_SUM_DONE complete={not incomplete}")
    else:
        man = P.read_json(P.RUN / "e6" / "manifest.json")
        rows, incomplete = [], []
        for m in man:
            ps = prod_summary(CONF / m["name"], P.R_CONF, pols=("random",))
            if not ps["complete"]:
                incomplete.append((m["name"], len(ps["missing"])))
                continue
            rows.append(dict(m, J_pass_cpu=ps["J_pass"], kappa_bar_random_cpu=ps["kappa_bar"], finite=ps["finite"],
                             fits_conf=ps["fits"], seconds_mean=ps["seconds_mean"]))
        cand = sorted((r for r in rows if r["fits_conf"]), key=lambda r: r["J_pass_cpu"])
        for r in sorted(rows, key=lambda r: r["J_pass_cpu"]):
            print(f"{r['name']:8s} {'+'.join(r['why']):20s} src {r['src']:11s} J_pass_cpu(conf) {r['J_pass_cpu']:.4f} "
                  f"kappa_bar(random) {r['kappa_bar_random_cpu']:.3f} fits {r['fits_conf']} J_pass(tune) {r['val']:.4f}")
        out = dict(tau=P.TAU, complete=not incomplete, incomplete=incomplete, rows=rows,
                   psi_pass=(cand[0] if cand and not incomplete else None))
        if not incomplete:
            print("psi-pass:", json.dumps(out["psi_pass"]) if cand else "not defined (no setting fits)")
        C.atomic_write_json(P.RUN / "e6" / "confirm.json", out)
        print(f"PS_E6_SUM_DONE complete={not incomplete}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("prep", "cmds", "sum"):
        s = sub.add_parser(nm)
        s.add_argument("--camp", required=True, choices=("e4", "e6"))
    a = ap.parse_args()
    dict(prep=cmd_prep, cmds=cmd_cmds, sum=cmd_sum)[a.cmd](a)


if __name__ == "__main__":
    main()
