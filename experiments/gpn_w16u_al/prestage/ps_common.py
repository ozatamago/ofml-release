#!/usr/bin/env python3
"""Pre-stage of OFML-Ens (REPORTS/ENS_PRESTAGE_PLAN.md, 2026-11-13): settings, search space, gate, trial.

A setting psi is a dict(w, L, d, eta, K, wd, init) with init in {"default", "he"}; 5 members, prior / anchor off.
The trial and the gate call the production functions ``w16u_ens.fit_predict`` / ``summarise``,
``w16u_common.choose`` / ``round_metrics`` and ``analyze_w16u.aulc``; the only difference from the production path
is that several trajectories are trained as one bundle (plan 4.2).

Seed replicate c (plan 2.2): member seed base = inference seed of the teacher + 1000003 c; the random policy uses
``C.policy_rng`` for c = 0 and ``default_rng([teacher seed, 7, policy index, c])`` for c >= 1.
kappa_n = RMSE of the ensemble mean on the observed points against the observed y, divided by the noise sd 0.10.
"""
from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
import w16u_common as C  # noqa: E402
import w16u_ens as E  # noqa: E402
from analyze_w16u import aulc  # noqa: E402

RUN = EXP / "runs" / "prestage"
R_TUNE = tuple(range(1000, 1016))
R_CONF = tuple(range(1016, 1032))
TAUS = (1.0, 1.25, 1.5)
TAU = 1.25                       # fixed 2026-11-13 (decision D4, recorded in REPORTS/修正記録.md)
NRMSE_GATE = 0.95
N_KAPPA0 = 50                    # kappa-bar averages n = 50..100
W_CAP = 5.2e6                    # weights of one member
T_CAP = 6 * 3600.0               # seconds of one production trajectory
WIDTHS = (16, 32, 64, 128, 256, 512, 1024)
ETA0, K0 = 1e-4, 400

_SH = dict(w=64, L=1, d=2, eta=0.001071249597445359, K=1600, wd=0.0005457683354166773, init="default")
REFS = {
    "shallow": _SH,
    "deep": dict(_SH, L=4, wd=1e-6),
    "old": dict(w=4, L=2, d=2, eta=0.003000783133850832, K=400, wd=0.001, init="default"),
    "shallow_wd0": dict(_SH, wd=0.0),
    "deep_wd0": dict(_SH, L=4, wd=0.0),
    "deep_wd1e-8": dict(_SH, L=4, wd=1e-8),
    "e0_w32l3d5he": dict(_SH, w=32, L=3, d=5, init="he"),
}


# ----------------------------------------------------------------------------------------------- settings
def norm_psi(psi: dict) -> dict:
    return dict(w=int(psi["w"]), L=int(psi["L"]), d=int(psi["d"]), eta=float(psi["eta"]), K=int(psi["K"]),
                wd=float(psi["wd"]), init=str(psi["init"]))


def key_of(psi: dict) -> str:
    p = norm_psi(psi)
    s = json.dumps([p["w"], p["L"], p["d"], p["eta"].hex(), p["K"], p["wd"].hex(), p["init"]])
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cfg_of(psi: dict) -> dict:
    """Model config of the production script for a setting (``init`` is absent for the default initialisation)."""
    p = norm_psi(psi)
    cfg = dict(E.BASELINE_CONFIG, latent_dim=p["d"], hidden=p["w"], n_hidden=p["L"], lr=p["eta"], wd=p["wd"],
               steps=p["K"], n_members=5, packed=True)
    if p["init"] == "he":
        cfg["init"] = "he"
    return cfg


def prod_config(psi: dict) -> dict:
    """Content of the JSON config file read by ``w16u_ens.py --config`` for a setting."""
    p = norm_psi(psi)
    out = dict(latent_dim=p["d"], hidden=p["w"], n_hidden=p["L"], lr=p["eta"], wd=p["wd"], steps=p["K"],
               n_members=5, packed=True)
    if p["init"] == "he":
        out["init"] = "he"
    return out


def n_weights(w: int, L: int, d: int) -> float:
    return 11.0 * ((L - 1) * w * w + 2 * d * w)


STRUCTS = tuple((w, L) for w in WIDTHS for L in range(1, 9) if n_weights(w, L, 8) <= W_CAP)   # 43 structures


# ------------------------------------------------------------------------------------------- search space
DEFAULT_SPACE = dict(wd_base=1e-6, u6_div=4.0, lo=[0.0] * 7, hi=[1.0] * 7, expanded=[])


def load_space(path: Path | None = None) -> dict:
    path = path or (RUN / "space.json")
    return json.loads(path.read_text()) if path.exists() else json.loads(json.dumps(DEFAULT_SPACE))


def load_tcpu() -> dict:
    """{"d8": {"<w>_<L>": seconds per step}, "d12": {...}} measured by ps_tcpu.py (E0 step 4)."""
    return json.loads((RUN / "tcpu.json").read_text())


def to_u(psi: dict, sp: dict) -> np.ndarray:
    p = norm_psi(psi)
    return np.array([(math.log2(p["w"]) - 4) / 6, (p["L"] - 1) / 7, (p["d"] - 2) / 6,
                     0.5 * math.log10(p["eta"] / ETA0), 0.25 * math.log2(p["K"] / K0),
                     math.log10(p["wd"] / sp["wd_base"]) / sp["u6_div"], 1.0 if p["init"] == "he" else 0.0])


def _rnd(x: float) -> int:
    return int(math.floor(x + 0.5))


def from_u(u, sp: dict) -> dict:
    u = [float(v) for v in u]
    return dict(w=2 ** _rnd(4 + 6 * u[0]), L=_rnd(1 + 7 * u[1]), d=_rnd(2 + 6 * u[2]), eta=ETA0 * 10 ** (2 * u[3]),
                K=_rnd(K0 * 2 ** (4 * u[4])), wd=sp["wd_base"] * 10 ** (sp["u6_div"] * u[5]),
                init="he" if u[6] >= 0.5 else "default")


def t_est(psi: dict, tcpu: dict) -> float:
    """93 K t_cpu(w, L): estimate of the time of one production trajectory, seconds."""
    p = norm_psi(psi)
    return 93.0 * p["K"] * tcpu["d12" if p["d"] > 8 else "d8"][f"{p['w']}_{p['L']}"]


def in_psi(psi: dict, sp: dict, tcpu: dict) -> bool:
    p = norm_psi(psi)
    if p["w"] not in WIDTHS or not 1 <= p["L"] <= 8 or not K0 <= p["K"] <= 6400:
        return False
    u, eps = to_u(p, sp), 1e-9
    if not (sp["lo"][2] - eps <= u[2] <= sp["hi"][2] + eps and sp["lo"][3] - eps <= u[3] <= sp["hi"][3] + eps
            and sp["lo"][5] - eps <= u[5] <= sp["hi"][5] + eps):
        return False
    if n_weights(p["w"], p["L"], p["d"]) > W_CAP:
        return False
    return t_est(p, tcpu) <= T_CAP


# ----------------------------------------------------------------------------------------------- helpers
@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def live_jobs() -> set[str] | None:
    """Slurm job ids of this user that are queued or running on this host (None if squeue is unavailable)."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%A"], capture_output=True,
                             text=True, timeout=60)
        if out.returncode != 0:
            return None
        return set(out.stdout.split())
    except Exception:
        return None


def my_job() -> str:
    return os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def spacefill_design() -> np.ndarray:
    """O_100 of the space-filling trajectory (the same for every function)."""
    obs = list(C.INITIAL_FLAT)
    while len(obs) < C.T_TOTAL:
        obs.append(C.spacefill_pick(obs))
    return np.asarray(obs, dtype=np.int64)


def initial_split(psi: dict, B: int, device) -> int:
    """Number of parts the bundle is trained in (plan H21): parts of at most about 8 GB on the GPU."""
    if device.type != "cuda":
        return 1
    p = norm_psi(psi)
    est = 16.0 * 5 * B * n_weights(p["w"], p["L"], p["d"])      # float32 x (param, grad, 2 Adam moments)
    split = 1
    while est / split > 8e9 and split < B:
        split *= 2
    return split


def fit_predict_robust(init, cfg, sbase, flat, y, device, st: dict) -> np.ndarray:
    """``E.fit_predict`` on the bundle in st["split"] parts; a GPU out-of-memory error doubles the number of parts
    (then waits and retries when the parts cannot be made smaller)."""
    import torch
    cuda = device.type == "cuda"
    waits = 0
    while True:
        oom = False
        try:
            parts = np.array_split(np.arange(len(sbase)), min(st["split"], len(sbase)))
            out = [E.fit_predict(init, cfg, [sbase[i] for i in ix], flat[ix], y[ix], device) for ix in parts]
            if cuda:
                torch.cuda.empty_cache()
            return np.concatenate(out)
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            st["oom"] = st.get("oom", 0) + 1
            if st["split"] < len(sbase):
                st["split"] *= 2
            else:
                waits += 1
                if waits > 30:
                    raise RuntimeError("GPU out of memory with a bundle of one trajectory per part")
                time.sleep(60)


# -------------------------------------------------------------------------------------------------- gate
def gate_compute(psi: dict, device, reps=R_TUNE) -> dict:
    """Plan 4.4: one training on O_100 of the space-filling trajectory per function, one bundle."""
    t0 = time.time()
    psi = norm_psi(psi)
    cfg = cfg_of(psi)
    init = E.EnsembleInit(cfg)
    teachers = {r: C.load_teacher(r) for r in reps}
    flat = np.tile(spacefill_design(), (len(reps), 1))
    y = np.stack([teachers[r][0][flat[b]] + teachers[r][1][flat[b]] for b, r in enumerate(reps)])
    sbase = [C.teacher_seeds(r)[2] for r in reps]
    st = dict(split=initial_split(psi, len(reps), device), oom=0)
    S = fit_predict_robust(init, cfg, sbase, flat, y, device, st)
    finite = bool(np.isfinite(S).all())
    kap, nr, cov = [], [], []
    for b, r in enumerate(reps):
        truth = teachers[r][0]
        mean, var, _ = E.summarise(S[b], want_cov=False)
        kap.append(float(np.sqrt(np.mean((mean[flat[b]] - y[b]) ** 2)) / C.NOISE_STD))
        if finite:
            m = C.round_metrics(mean, var, C.gaussian_pit(mean, var, truth), truth)
            nr.append(float(m["nrmse"]))
            cov.append(float(m["coverage"][2]))
        else:
            nr.append(float("nan"))
            cov.append(float("nan"))
    km, nm = float(np.mean(kap)), float(np.mean(nr))
    c3 = bool(finite and nm < NRMSE_GATE)
    return dict(psi=psi, key=key_of(psi), reps=list(reps), finite=finite, kappa100=kap, nrmse100=nr, cov95=cov,
                kappa_mean=km, nrmse_mean=nm, cond_nrmse=c3,
                passed={str(t): bool(finite and km <= t and c3) for t in TAUS},
                failed_conditions=[n for n, ok in (("finite", finite), ("kappa", finite and km <= TAU),
                                                   ("nrmse", c3)) if not ok],
                split=st["split"], oom=st["oom"], seconds=time.time() - t0, device=str(device))


def gate_cached(psi: dict, device, log=None) -> dict:
    f = RUN / "gates" / f"{key_of(psi)}.json"
    if f.exists():
        return read_json(f)
    g = gate_compute(psi, device)
    f.parent.mkdir(parents=True, exist_ok=True)
    C.atomic_write_json(f, g)
    if log:
        log(f"gate {g['key']} {psi}: kappa {g['kappa_mean']:.3f} nrmse {g['nrmse_mean']:.3f} "
            f"pass@{TAU} {g['passed'][str(TAU)]} ({g['seconds']:.0f}s)")
    return g


def gate_pass(g: dict, tau: float = TAU) -> bool:
    return bool(g["passed"][str(tau)])


# ------------------------------------------------------------------------------------------------- trial
def run_trial(psi: dict, reps, pols, c: int, device, out_dir: Path, log=print, stop_after_n: int | None = None,
              no_split: bool = False):
    """All trajectories (reps x pols) of a setting as one bundle. Saves a checkpoint after every n and resumes from
    it. Returns the result (also in out_dir/result.json), or None when stopped by ``stop_after_n`` (test)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rp, ck = out_dir / "result.json", out_dir / "ckpt.json"
    if rp.exists():
        return read_json(rp)
    psi = norm_psi(psi)
    reps, pols = [int(r) for r in reps], list(pols)
    cfg = cfg_of(psi)
    init = E.EnsembleInit(cfg)
    trajs = [(r, p) for r in reps for p in pols]
    teachers = {r: C.load_teacher(r) for r in reps}
    sbase = [C.teacher_seeds(r)[2] + 1000003 * c for r, _ in trajs]
    ident = dict(psi=psi, reps=reps, pols=pols, c=int(c))
    if ck.exists():
        z = read_json(ck)
        if z["ident"] != ident:
            raise RuntimeError(f"{ck} belongs to another trial")
        obs, rec, st, elapsed, resumes = z["obs"], z["rec"], z["st"], z["elapsed"], z["resumes"] + 1
        rngs = []
        for s in z["rng"]:
            g = np.random.default_rng()
            g.bit_generator.state = s
            rngs.append(g)
        log(f"resume at n={len(obs[0])} (resume {resumes})")
    else:
        obs = [list(C.INITIAL_FLAT) for _ in trajs]
        rec = [dict(nrmse=[], cov95=[], kappa=[], iv=[]) for _ in trajs]
        rngs = [C.policy_rng(r, p) if c == 0 else
                np.random.default_rng([C.teacher_seeds(r)[0], 7, C.POLICIES.index(p), c]) for r, p in trajs]
        st = dict(split=1 if no_split else initial_split(psi, len(trajs), device), oom=0)
        elapsed, resumes = 0.0, 0
    t0 = time.time()
    status = "ok"
    while True:
        n = len(obs[0])
        flat = np.array(obs, dtype=np.int64)
        y = np.stack([teachers[r][0][flat[b]] + teachers[r][1][flat[b]] for b, (r, _) in enumerate(trajs)])
        S_all = fit_predict_robust(init, cfg, sbase, flat, y, device, st)
        if not np.isfinite(S_all).all():
            status = "nonfinite"
            log(f"non-finite prediction at n={n}")
            break
        for b, (r, p) in enumerate(trajs):
            truth = teachers[r][0]
            mean, var, cov = E.summarise(S_all[b], want_cov=(p == "ivr" and n < C.T_TOTAL))
            m = C.round_metrics(mean, var, C.gaussian_pit(mean, var, truth), truth)
            rec[b]["nrmse"].append(float(m["nrmse"]))
            rec[b]["cov95"].append(float(m["coverage"][2]))
            rec[b]["kappa"].append(float(np.sqrt(np.mean((mean[flat[b]] - y[b]) ** 2)) / C.NOISE_STD))
            rec[b]["iv"].append(float(m["iv"]))
            if n < C.T_TOTAL:
                pick = C.choose(p, obs[b], var, cov, rngs[b])
                assert pick not in obs[b]
                obs[b].append(pick)
        if n % 10 == 0 or n >= C.T_TOTAL:
            log(f"n={n} elapsed {elapsed + time.time() - t0:.0f}s mean NRMSE "
                f"{np.mean([q['nrmse'][-1] for q in rec]):.4f} split {st['split']}")
        if n >= C.T_TOTAL:
            break
        C.atomic_write_json(ck, dict(ident=ident, obs=obs, rec=rec, st=st, resumes=resumes,
                                     rng=[g.bit_generator.state for g in rngs],
                                     elapsed=elapsed + time.time() - t0))
        if stop_after_n is not None and n >= stop_after_n:
            log(f"[test] stopped after n={n}")
            return None
    res = summarise_trial(ident, trajs, obs, rec, status)
    res.update(seconds=elapsed + time.time() - t0, resumes=resumes, split=st["split"], oom=st["oom"],
               device=str(device), cfg=cfg, key=key_of(psi), job=my_job())
    C.atomic_write_json(rp, res)
    if ck.exists():
        ck.unlink()
    return res


def summarise_trial(ident: dict, trajs, obs, rec, status: str) -> dict:
    pols = ident["pols"]
    out = dict(ident, status=status, trajs=[])
    if status != "ok":
        out.update(aulc={}, J=None, J_pass=None, kappa_bar=None, kappa_bar_random=None, fits=False, fits_pass=False,
                   n_reached=len(rec[0]["nrmse"]) + len(C.INITIAL_FLAT))
        return out
    i0 = N_KAPPA0 - len(C.INITIAL_FLAT)
    for b, (r, p) in enumerate(trajs):
        cnt = np.bincount(np.asarray(obs[b]) // C.G, minlength=C.N_PROTOCOLS)
        out["trajs"].append(dict(rep=r, policy=p, observed=obs[b], aulc=float(aulc(np.asarray(rec[b]["nrmse"]))),
                                 kappa_bar=float(np.mean(rec[b]["kappa"][i0:])),
                                 n_protocols=int((cnt > 0).sum()), max_share=float(cnt.max() / len(obs[b])),
                                 **rec[b]))
    A = {p: float(np.mean([t["aulc"] for t in out["trajs"] if t["policy"] == p])) for p in pols}
    kb = float(np.mean([t["kappa_bar"] for t in out["trajs"]]))
    kr = (float(np.mean([t["kappa_bar"] for t in out["trajs"] if t["policy"] == "random"]))
          if "random" in pols else None)
    full = set(pols) == set(C.POLICIES)
    out.update(aulc=A, J=float(np.mean([A[p] for p in C.POLICIES])) if full else None, J_pass=A.get("random"),
               kappa_bar=kb, kappa_bar_random=kr, fits=bool(kb <= TAU),
               fits_pass=bool(kr is not None and kr <= TAU), fits_by_tau={str(t): bool(kb <= t) for t in TAUS})
    return out


def per_function_J(res: dict) -> dict:
    """{rep: mean AULC over the 4 acquisition functions} of a trial result."""
    out = {}
    for t in res["trajs"]:
        out.setdefault(t["rep"], []).append(t["aulc"])
    return {r: float(np.mean(v)) for r, v in out.items()}


def prod_trajectory(run_dir: Path, rep: int, policy: str, model_dir: str = "ens"):
    """Production output of one trajectory -> dict(aulc, nrmse[93], kappa[93], finite) or None if missing.
    kappa_n is recomputed from the stored predictive mean (float32) and the observed points."""
    f = C.result_path(Path(run_dir), model_dir, rep, policy)
    if not C.result_done(f):
        return None
    r = read_json(f)
    z = np.load(f.with_suffix(".npz"))
    truth, noise = C.load_teacher(rep)
    obs = np.asarray(r["observed"], dtype=np.int64)
    nr = np.asarray(r["rounds"]["nrmse"], dtype=float)
    mean = z["mean"].astype(float)                       # [93, 1024]
    n0 = len(C.INITIAL_FLAT)
    kap = np.array([np.sqrt(np.mean((mean[i][obs[:n0 + i]] - (truth + noise)[obs[:n0 + i]]) ** 2)) / C.NOISE_STD
                    for i in range(len(nr))])
    finite = bool(np.isfinite(nr).all() and np.isfinite(mean).all() and np.isfinite(z["var"]).all())
    return dict(aulc=float(aulc(nr)), nrmse=nr, kappa=kap, finite=finite, observed=r["observed"],
                seconds=float(r["policy_seconds"]), cov95=np.asarray([c[2] for c in r["rounds"]["coverage"]]))
