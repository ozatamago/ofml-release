"""GPU smoke of OFML-GPN (ofml_gpn general graph backend) on the purpose-1 DRAFT library, one teacher.

Design (driver pieces re-implemented from cx_bayes u15_compare.run_gpn_arm, not imported):
  * sources: N_SRC grid points each (indices round(linspace(4, 59, N_SRC))), assimilated one at a time in
    point-major order (point k of every source, then k+1);
  * then N_TGT adaptive points by UCB (mean + sqrt(log(n + 2)) sd, the u15 c_schedule) restricted to the TARGET
    protocols, repeats allowed, fresh noise per visit (sd 0.1);
  * SMC: assimilate_u15 (ESS threshold 0.65, systematic resampling, derive_stream_seed streams), segment-blocked
    pCN with per-block beta adaptation, l2_null_refresh 1, inner collapsed sweeps (default 1, mode 1, multi-block
    leaves non-centred), fast_chol.
  * logged per round: seconds, pre-resampling ESS, resampled flag, rejuvenation seconds, plain/cm acceptance and
    beta; at n = N_SRC * n_sources and at the end: posterior quantiles (exact RB terminal) -> coverage of the
    truth by the 50/80/90/95/99% central bands on target / source protocols.
usage: run_p1_smoke.py OUT_DIR [--particles 4096] [--pcn-steps 96] [--cm 1] [--n-src 8] [--n-tgt 20] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import numpy as np  # noqa: E402

import ofml_gpn as OG  # noqa: E402
from ofml_gpn.coverage_quantiles import BAND_LEVELS, COVERAGE_LEVELS, shared_quantiles  # noqa: E402
from ofml_gpn.graph_backends import GraphSharedGpnBackend  # noqa: E402
from ofml_gpn.numerics import c_schedule, normalize_logweights  # noqa: E402
from ofml_gpn.term_library import purpose1_library  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--particles", type=int, default=4096)
ap.add_argument("--pcn-steps", type=int, default=96)
ap.add_argument("--cm", type=int, default=1)
ap.add_argument("--cm-multi", default="noncentred")
ap.add_argument("--n-src", type=int, default=8)
ap.add_argument("--n-tgt", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk-mb", type=int, default=512)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

G = 64
WX = np.linspace(0.0, 4.0, G)
JIT, NOISE = 1e-8, 0.10
TL = purpose1_library()
LIB = TL.library
P = LIB.n_protocols
NC = P * G
FAMILY_SEED = 47_000_000 + 1000 * args.seed
truth, _noise_fixed = OG.draw_graph_teacher(LIB, WX, JIT, FAMILY_SEED, FAMILY_SEED + 1, NOISE)
noise_rng = np.random.default_rng(FAMILY_SEED + 3)  # fresh noise per visit

src_pts = np.round(np.linspace(4, 59, args.n_src)).astype(int)
initial = [p * G + int(g) for g in src_pts for p in TL.sources]
n0 = len(initial)
T_total = n0 + args.n_tgt
buckets = tuple(sorted({b for b in (40, 64, 100, 128, 160, 200) if b < T_total} | {8 * math.ceil(T_total / 8)}))
tgt_mask = np.isin(np.arange(NC) // G, TL.targets)
src_mask = np.isin(np.arange(NC) // G, TL.sources)

t_build = time.time()
b = GraphSharedGpnBackend(
    WX, OG.ProtocolLibrarySpec(), LIB, NOISE**2, JIT, args.particles, args.pcn_steps, n_max_buckets=buckets,
    allow_repeat_observations=True, fast_chol=True, chunk_target_bytes=args.chunk_mb << 20, l2_null_refresh=1,
    inner_collapsed_sweeps=args.cm, inner_collapsed_multi=args.cm_multi, mixing_diagnostics=True,
)
print(f"[setup] {LIB.name}: P={P} nodes={LIB.n_nodes} latent={b.grid_size} types="
      f"{[t.name for t in LIB.types]} legacy={b._legacy} R-classes="
      f"{[(c['kind'], [LIB.types[t].name for t in c['types']]) for c in b._refresh['classes']]} "
      f"U-protocol nodes={[TL.node_terms[i] for i in b._u_proto_nodes]} cm_inner="
      f"{[LIB.types[t].name for t in getattr(b, '_cm_inner', ())]} buckets={buckets} devices={b.devices} "
      f"build {time.time() - t_build:.1f}s", flush=True)

latent = b.initialise(FAMILY_SEED + 2)
logw = np.full(b.n_particles, -math.log(b.n_particles))
ll = np.zeros(b.n_particles)
observed: list[int] = []
ys: list[float] = []
beta = np.full(b.n_pcn_blocks, 0.30)  # PcnBetaState seed (segment-blocked)
n_rej = 0
rounds = []
cover = {}


def y_of(c):
    return float(truth[c] + noise_rng.normal(scale=NOISE))


def coverage(tag):
    idx, yp, act = OG.padded_observations_u15(observed, np.zeros(NC), n_max=b.n_max_for(len(observed)),
                                              observed_y=np.asarray(ys))
    w, _ = normalize_logweights(logw)
    t0 = time.time()
    q, qerr = shared_quantiles(b, latent, w, idx, yp, act)
    lv = [round(v, 4) for v in COVERAGE_LEVELS]
    res = {"n": len(observed), "seconds": time.time() - t0, "max_cdf_err": float(np.max(qerr))}
    for band, (lo, hi) in BAND_LEVELS.items():
        inside = (truth >= q[lv.index(lo)]) & (truth <= q[lv.index(hi)])
        res[f"cov{band}_targets"] = float(inside[tgt_mask].mean())
        res[f"cov{band}_sources"] = float(inside[src_mask].mean())
        res[f"cov{band}_per_protocol"] = [float(inside[p * G:(p + 1) * G].mean()) for p in range(P)]
    res["width95_targets"] = float(np.mean((q[lv.index(0.975)] - q[lv.index(0.025)])[tgt_mask]))
    mean, var, _ = b.posterior_moments(latent, w, idx, yp, act, exact=True)
    res["rmse_targets"] = float(np.sqrt(np.mean((mean - truth)[tgt_mask] ** 2)))
    res["rmse_sources"] = float(np.sqrt(np.mean((mean - truth)[src_mask] ** 2)))
    cover[tag] = res
    np.save(out / f"quantiles_{tag}.npy", q.astype(np.float32))
    print(f"[coverage {tag}] n={len(observed)} cov95 targets {res['cov95_targets']:.3f} sources "
          f"{res['cov95_sources']:.3f}; cov50/80/90/99 targets "
          f"{res['cov50_targets']:.2f}/{res['cov80_targets']:.2f}/{res['cov90_targets']:.2f}/{res['cov99_targets']:.2f};"
          f" rmse targets {res['rmse_targets']:.3f}; {res['seconds']:.1f}s", flush=True)


t_start = time.time()
while len(observed) < T_total:
    t0 = time.time()
    if len(observed) < n0:
        c = initial[len(observed)]
    else:
        idx, yp, act = OG.padded_observations_u15(observed, np.zeros(NC), n_max=b.n_max_for(len(observed)),
                                                  observed_y=np.asarray(ys))
        w, _ = normalize_logweights(logw)
        mean, var, _ = b.posterior_moments(latent, w, idx, yp, act)
        ucb = mean + c_schedule(len(observed)) * np.sqrt(np.maximum(var, 0.0))
        cand = np.flatnonzero(tgt_mask)
        c = int(cand[np.argmax(ucb[cand])])
    t_acq = time.time() - t0
    observed.append(c)
    ys.append(y_of(c))
    b.last_assim_diag = None
    b.last_mixing_diag = None
    latent, ll, logw, ess, acc, n_rej, rejuv = OG.assimilate_u15(
        b, latent, ll, logw, observed, np.zeros(NC), 0.65, beta, FAMILY_SEED + 2, n0, n_rej, observed_y=np.asarray(ys))
    rec = {"n": len(observed), "cell": c, "protocol": c // G, "seconds": time.time() - t0, "acq_seconds": t_acq,
           "ess_pre_over_P": float(b.last_assim_diag["ess_pre"]) / b.n_particles if b.last_assim_diag else None,
           "resampled": bool(rejuv)}
    if acc is not None:
        beta = np.asarray(acc.beta_next)
        md = b.last_mixing_diag or {}
        rec.update({"rejuv_seconds": float(md.get("rejuv_seconds", np.nan)),
                    "acc_last": np.asarray(acc.last_segment_acceptance).round(3).tolist(),
                    "beta_last": np.asarray(acc.beta_used_final_segment).round(4).tolist(),
                    "msjd": [round(v, 4) for v in md.get("msjd", [])],
                    "uniq_after": md.get("uniq_after"),
                    "n_unique_ancestors": (b.last_assim_diag or {}).get("n_unique_ancestors")})
        if args.cm:
            rec["cm_acc_last"] = np.asarray(md["cm_acc"])[-1].round(3).tolist() if "cm_acc" in md else None
            rec["cm_beta"] = np.asarray(b._cm_beta).round(4).tolist()
    rounds.append(rec)
    print("[round] " + json.dumps(rec), flush=True)
    if len(observed) == n0:
        coverage("after_sources")
    json.dump({"rounds": rounds, "coverage": cover}, open(out / "progress.json", "w"), indent=1)
coverage("final")
rj = [r["rejuv_seconds"] for r in rounds if r.get("rejuv_seconds") is not None]
summary = {
    "library": LIB.name, "sources": [TL.protocol_terms[p] for p in TL.sources],
    "targets": [TL.protocol_terms[p] for p in TL.targets], "args": vars(args), "devices": str(b.devices),
    "n_rejuvenations": n_rej, "total_seconds": time.time() - t_start,
    "mean_round_seconds": float(np.mean([r["seconds"] for r in rounds])),
    "mean_rejuv_seconds": float(np.mean(rj)) if rj else None,
    "mean_plain_round_seconds": float(np.mean([r["seconds"] for r in rounds if not r["resampled"]] or [np.nan])),
    "min_ess_pre_over_P": float(np.nanmin([r["ess_pre_over_P"] for r in rounds if r["ess_pre_over_P"] is not None])),
    "coverage": cover, "rounds": rounds,
}
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print("[done] " + json.dumps({k: v for k, v in summary.items() if k not in ("rounds", "coverage")}), flush=True)
