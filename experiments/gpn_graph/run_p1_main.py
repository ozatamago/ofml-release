"""Purpose 1 (c), GP-matched benchmark: OFML-GPN production runner, one problem (true function) per process.

Plan: REPORTS/EXPERIMENT_DESIGN_2026-11-04.md 1.8 (1.8.2 data, 1.8.5 protocol, 1.8.6 inference settings).
* Data / noise / IVR / NRMSE: p1_common (unchanged).  Problem = runs/p1_main/problems/{rep,tune}_XX.npz.
* Model: ofml_gpn GraphSharedGpnBackend on p1_common.library() (types A, B, C1, C2, M shared across all 17
  protocols; terminal R Rao-Blackwellised).  Kernels / noise fixed to the truth.
* SMC: the 80 source rows (src_cells, src_y, in that order) are assimilated one at a time, then t = 1..T:
  IVR (p1_common.choose_ivr) on the EXACT RB-terminal posterior covariance of f over the 448 target cells,
  observation p1_common.observe (repeats allowed, noise table), one SMC update.  assimilate_u15 with ESS
  threshold 0.65, systematic resampling, segment-blocked pCN (default 96 steps, per-block beta adaptation, beta0
  0.30), l2_null_refresh 1, no collapsed moves (inner_collapsed_sweeps 0), fast_chol, exact terminal likelihood.
* Records (out/<stem>.json + .npz): NRMSE(t), chosen candidates, max posterior variance, 50/80/95% coverage of
  the 448 target cells, posterior mean / variance / PIT per t; per assimilation round (all 280) pre-resampling
  ESS / P, resampling flag, distinct ancestors, pCN acceptance and beta per segment and block, seconds.
* Checkpoint every --checkpoint-every adaptive steps (out/ckpt/<stem>.npz); rerunning resumes.

usage: run_p1_main.py --problem runs/p1_main/problems/rep_00.npz --out runs/p1_main/ofml_gpn
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p1_gpn_tools as TL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--problem", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--particles", type=int, default=16384)
ap.add_argument("--pcn-steps", type=int, default=96)
ap.add_argument("--T", type=int, default=200)
ap.add_argument("--ess-fraction", type=float, default=0.65)
ap.add_argument("--beta0", type=float, default=0.30)
ap.add_argument("--chunk-mb", type=int, default=512)
ap.add_argument("--cov-chunk", type=int, default=64)
ap.add_argument("--buckets", default="40,64,100,128,160,200,240,280")
ap.add_argument("--checkpoint-every", type=int, default=10)
ap.add_argument("--stop-after-t", type=int, default=-1, help="timing only: stop (after a checkpoint) at this t")
ap.add_argument("--compile-cache", default="")
ap.add_argument("--skip-existing", action="store_true")
ap.add_argument("--abort-on-exclusion", action="store_true",
                help="stop as soon as a round's pre-resampling ESS/P < 0.001 (plan 1.8.5 rule); status 'excluded', exit code 3")
# 2026-11-06 additive diagnostics (defaults = the production runner unchanged)
ap.add_argument("--seed-offset", type=int, default=0, help="inference seed = problem seed + 4 + 7919 * offset")
ap.add_argument("--oracle-types", default="", help="comma-separated types (e.g. A,B) fixed to the truth, never moved")
ap.add_argument("--diag-every", type=int, default=0,
                help=">0: every this many t, posterior of every node output vs the teacher (cov95, mean, sd)")
args = ap.parse_args()
TL.jax_setup(args.compile_cache)

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import p1_common as C  # noqa: E402
from ofml_gpn import ProtocolLibrarySpec  # noqa: E402
from ofml_gpn.graph_backends import GraphSharedGpnBackend  # noqa: E402

prob_path = Path(args.problem)
stem = prob_path.stem
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
f_json, f_npz, f_ck = out / f"{stem}.json", out / f"{stem}.npz", out / "ckpt" / f"{stem}.npz"
if args.skip_existing and f_json.exists() and json.load(open(f_json)).get("status") == "ok":
    print(f"[skip] {f_json} exists", flush=True)
    sys.exit(0)

prob = C.load_problem(prob_path)
TLIB = C.library()
LIB = TLIB.library
T = int(args.T)
n0 = len(prob.src_cells)
tgt_cells = C.tgt_flat_cells()
truth_tgt = prob.truth_tgt
buckets = tuple(int(v) for v in args.buckets.split(","))
assert buckets[-1] >= n0 + T, f"largest bucket {buckets[-1]} < {n0 + T}"
INFERENCE_SEED = prob.seed + 4 + 7919 * args.seed_offset  # (seed, seed+1 teacher; seed+2/3 noises); offset 0 = production
ORACLE = [v for v in args.oracle_types.split(",") if v]
TYPE_NAMES = [t.name for t in LIB.types]
FROZEN = tuple(TYPE_NAMES.index(v) for v in ORACLE)
DIAG = args.diag_every > 0 or bool(ORACLE)
if DIAG:  # the teacher's node outputs and whitened latent (same draw as make_problem)
    import ofml_gpn.graph_library as GL  # noqa: E402
    _tr, _, TEACH = GL.draw_graph_teacher(LIB, C.WX, C.JIT, prob.seed, prob.seed + 1, C.NOISE_SD, return_details=True)
    assert np.allclose(_tr, prob.truth), "teacher redraw differs from the problem file"
    NODE_TRUTH = np.stack([TEACH["node_out"][i] for i in range(len(LIB.nodes))])  # (n_nodes, G)
    NODE_TYPE = np.array([TYPE_NAMES.index(n.type_name) for n in LIB.nodes])

t_build = time.time()
b = GraphSharedGpnBackend(
    C.WX, ProtocolLibrarySpec(), LIB, C.NOISE_SD ** 2, C.JIT, args.particles, args.pcn_steps,
    n_max_buckets=buckets, allow_repeat_observations=True, fast_chol=True, chunk_target_bytes=args.chunk_mb << 20,
    l2_null_refresh=1, inner_collapsed_sweeps=0, mixing_diagnostics=True, frozen_blocks=FROZEN)
summ = TL.MixtureSummary(b, tgt_cells, chunk=args.cov_chunk)
block_names = [t.name for t in LIB.types]
config = dict(method="OFML-GPN", problem=str(prob_path), seed=prob.seed, inference_seed=INFERENCE_SEED,
              particles=b.n_particles, pcn_steps=b.pcn_steps, pcn_segments=b.n_pcn_segments, ess_fraction=args.ess_fraction,
              beta0=args.beta0, l2_null_refresh=1, inner_collapsed_sweeps=0, terminal="exact", fast_chol=True,
              buckets=list(buckets), T=T, n_source_rows=n0, pcn_blocks=block_names, legacy=bool(b._legacy),
              devices=str(b.devices), seed_offset=args.seed_offset, oracle_types=ORACLE, diag_every=args.diag_every,
              **TL.host_info())
print(f"[setup] {stem}: {LIB.name} P={LIB.n_protocols} latent={b.grid_size} blocks={block_names} "
      f"particles={b.n_particles} pcn={b.pcn_steps} buckets={buckets} devices={b.devices} "
      f"build {time.time() - t_build:.1f}s", flush=True)

sys_ = TL.SmcSystem(b, INFERENCE_SEED, n0, args.ess_fraction, args.beta0)
if FROZEN:  # oracle: every particle carries the teacher's whitened blocks of the frozen types
    _x = np.asarray(b.host_particles(sys_.latent)).copy()
    for k in FROZEN:
        s0, e0 = b.block_bounds[k]
        _x[:, s0:e0] = TEACH["latent"][s0:e0]
    sys_.latent = b.shard_particles(_x)
    _n = np.asarray(b._forward(jnp.asarray(_x[:4])))
    for k in FROZEN:
        _err = max(float(np.max(np.abs(_n[:, i] - NODE_TRUTH[i]))) for i in np.flatnonzero(NODE_TYPE == k))
        print(f"[oracle] {TYPE_NAMES[k]} fixed to the truth; max |node - teacher| {_err:.2e}", flush=True)
        assert _err < 1e-6
diag = dict(t=[], cov95=[], mean=[], sd=[])  # node posteriors at t = 0, diag_every, ...


def node_diag(t):
    """Weighted posterior of every node output (G cells each) vs the teacher: central 95% coverage, mean, sd."""
    w, _ = sys_.weights()
    nodes = np.asarray(b._forward(jnp.asarray(b.host_particles(sys_.latent)))).reshape(b.n_particles, -1)  # (P, n_nodes*G)
    order = np.argsort(nodes, axis=0)
    cw = np.cumsum(w[order], axis=0)
    lo = np.take_along_axis(nodes, order, 0)[np.argmax(cw >= 0.025, axis=0), np.arange(nodes.shape[1])]
    hi = np.take_along_axis(nodes, order, 0)[np.argmax(cw >= 0.975, axis=0), np.arange(nodes.shape[1])]
    tr = NODE_TRUTH.reshape(-1)
    cov = ((tr >= lo - 1e-9) & (tr <= hi + 1e-9)).reshape(NODE_TRUTH.shape).mean(axis=1)  # tol: oracle types
    mean = w @ nodes
    sd = np.sqrt(np.maximum(w @ (nodes * nodes) - mean * mean, 0.0))
    diag["t"].append(t)
    diag["cov95"].append(cov)
    diag["mean"].append(mean.reshape(NODE_TRUTH.shape).astype(np.float32))
    diag["sd"].append(sd.reshape(NODE_TRUTH.shape).astype(np.float32))
    by_type = " ".join(f"{TYPE_NAMES[k]}:{cov[NODE_TYPE == k].mean():.3f}" for k in range(len(TYPE_NAMES)))
    rmse = " ".join(f"{TYPE_NAMES[k]}:{np.sqrt(np.mean((mean.reshape(NODE_TRUTH.shape) - NODE_TRUTH)[NODE_TYPE == k] ** 2)):.3f}"
                    for k in range(len(TYPE_NAMES)))
    print(f"[nodes t={t}] cov95 {by_type} | rmse {rmse} | lineages0 {np.unique(sys_.lineage0).size} "
          f"lineages_src {np.unique(sys_.lineage_src).size if sys_.lineage_src is not None else -1}", flush=True)
visits = np.zeros(C.N_TGT_CELLS, dtype=np.int64)
rounds: list[dict] = []          # one per assimilation (n = 1 .. n0 + T)
steps: list[dict] = []           # one per t = 0 .. T (posterior summary, then the IVR choice for t < T)
arr = dict(mean=[], var=[], pit=[])
t_resume = 0
if f_ck.exists():
    z = TL.load_npz(f_ck)
    sys_.load(z, "s_")
    visits = z["visits"].copy()
    rounds = json.loads(str(z["rounds_json"]))
    steps = json.loads(str(z["steps_json"]))
    for k in arr:
        arr[k] = list(z["arr_" + k])
    t_resume = len(steps)
    print(f"[ckpt] resume: n={len(sys_.observed)} t={t_resume} (steps recorded {len(steps)})", flush=True)


def checkpoint():
    TL.save_npz_atomic(f_ck, **sys_.state("s_"), visits=visits, rounds_json=json.dumps(rounds),
                       steps_json=json.dumps(steps),
                       **{"arr_" + k: np.asarray(v, dtype=np.float32).reshape(-1, C.N_TGT_CELLS) for k, v in arr.items()})


def log_round(rec):
    short = {k: v for k, v in rec.items() if k not in ("acc_trace", "beta_trace", "uniq_after")}
    if "acc_trace" in rec:
        short["acc_last"] = np.round(rec["acc_trace"][-1], 3).tolist()
        short["beta_last"] = np.round(rec["beta_trace"][-1], 4).tolist()
    print("[round] " + json.dumps(short), flush=True)


ABORTED = False


def hits_rule(rec) -> bool:
    v = rec.get("ess_pre_over_P")
    return bool(args.abort_on_exclusion and v is not None and np.isfinite(v) and v < 1e-3)


t_start = time.time()
# ---- sources (t = 0 prefix): 80 rows in the stored order
while len(sys_.observed) < n0:
    i = len(sys_.observed)
    t0 = time.time()
    rec = sys_.assimilate(int(prob.src_cells[i]), float(prob.src_y[i]))
    rec.update(phase="source", seconds=time.time() - t0)
    rounds.append(rec)
    log_round(rec)
    if hits_rule(rec):
        ABORTED = True
        print(f"[abort] exclusion rule hit at source round n={len(sys_.observed)}: ESS/P={rec['ess_pre_over_P']:.2e}", flush=True)
        checkpoint()
        break
    if len(sys_.observed) == n0 or (args.checkpoint_every > 0 and len(sys_.observed) % 20 == 0):
        checkpoint()

# ---- adaptation
t = len(steps)
while t <= T and not ABORTED:
    t0 = time.time()
    obs_idx, y_pad, active = sys_.padded()
    w, ess = sys_.weights()
    mean, cov, pit = summ(sys_.latent, w, obs_idx, y_pad, active, truth_tgt)
    var = np.maximum(np.diag(cov), 0.0)
    t_sum = time.time() - t0
    st = dict(t=t, n=len(sys_.observed), nrmse=C.nrmse(mean, prob), max_var=float(var.max()),
              mean_var=float(var.mean()), coverage=TL.coverage_from_pit(pit), ess_over_P=float(ess) / b.n_particles,
              summary_seconds=t_sum)
    arr["mean"].append(mean.astype(np.float32))
    arr["var"].append(var.astype(np.float32))
    arr["pit"].append(pit.astype(np.float32))
    if DIAG and args.diag_every > 0 and t % args.diag_every == 0:
        node_diag(t)
    if t < T:
        t1 = time.time()
        c = C.choose_ivr(cov)
        st["acq_seconds"] = time.time() - t1
        y = C.observe(prob, c, visits)
        if DIAG:  # one-step-ahead standardised residual of the observation (Gaussian moment match, no truth used)
            st["z_next"] = float((y - mean[c]) / math.sqrt(var[c] + C.NOISE_SD ** 2))
        rec = sys_.assimilate(C.candidate_to_cell(c), y)
        rec.update(phase="target", t=t + 1, candidate=c, visit=int(visits[c]))
        st["chosen"] = c
        rec["seconds"] = time.time() - t0
        rounds.append(rec)
        st["seconds"] = rec["seconds"]
        log_round(rec)
        if hits_rule(rec):
            ABORTED = True
            print(f"[abort] exclusion rule hit at t={t + 1}: ESS/P={rec['ess_pre_over_P']:.2e}", flush=True)
    else:
        st["seconds"] = time.time() - t0
    steps.append(st)
    print(f"[t={t}] nrmse={st['nrmse']:.4f} maxvar={st['max_var']:.4f} cov50/80/95="
          f"{'/'.join(f'{v:.3f}' for v in st['coverage'])} summ={t_sum:.2f}s total={st['seconds']:.1f}s", flush=True)
    t += 1
    tt = t  # number of steps recorded
    if ABORTED:
        checkpoint()
        break
    if (args.checkpoint_every > 0 and tt % args.checkpoint_every == 0 and tt <= T) or tt - 1 == args.stop_after_t:
        checkpoint()
    if args.stop_after_t >= 0 and tt - 1 >= args.stop_after_t and tt <= T:
        break

status = "excluded" if ABORTED else ("ok" if len(steps) == T + 1 else "partial")
n_seg, n_blk = b.n_pcn_segments, b.n_pcn_blocks
ess_pre = [r["ess_pre_over_P"] for r in rounds]
result = dict(
    status=status, config=config, total_seconds_this_process=time.time() - t_start, resumed_from_t=t_resume,
    n_rejuvenations=sys_.n_rej, exclusion=TL.exclusion_flag(ess_pre),
    nrmse=[s["nrmse"] for s in steps], aulc=(C.aulc(np.array([s["nrmse"] for s in steps])) if status == "ok" else None),
    chosen=[s["chosen"] for s in steps if "chosen" in s], steps=steps, rounds=rounds)
TL.save_json_atomic(f_json, result)
TL.save_npz_atomic(
    f_npz, nrmse=np.array([s["nrmse"] for s in steps]), chosen=np.array(result["chosen"], dtype=np.int64),
    max_var=np.array([s["max_var"] for s in steps]), coverage=np.array([s["coverage"] for s in steps]),
    step_seconds=np.array([s["seconds"] for s in steps]),
    mean=np.asarray(arr["mean"], np.float32), var=np.asarray(arr["var"], np.float32), pit=np.asarray(arr["pit"], np.float32),
    round_n=np.array([r["n"] for r in rounds]), round_cell=np.array([r["cell"] for r in rounds]),
    ess_pre_over_P=np.array(ess_pre, dtype=np.float64), resampled=np.array([r["resampled"] for r in rounds]),
    n_unique_ancestors=np.array([r["n_unique_ancestors"] for r in rounds]),
    acc=TL.pad_traces(rounds, "acc_trace", n_seg, n_blk), beta=TL.pad_traces(rounds, "beta_trace", n_seg, n_blk),
    round_seconds=np.array([r["seconds"] for r in rounds]),
    rejuv_seconds=np.array([r.get("rejuv_seconds", np.nan) for r in rounds]),
    pcn_blocks=np.array(block_names), truth_tgt=truth_tgt,
    **({} if not DIAG else dict(
        z_next=np.array([s.get("z_next", np.nan) for s in steps]),
        log_evidence_increment=np.array([r.get("log_evidence_increment", np.nan) for r in rounds]),
        lineages0=np.array([r.get("lineages0", -1) for r in rounds]),
        lineages_src=np.array([-1 if r.get("lineages_src") is None else r["lineages_src"] for r in rounds]),
        msjd=np.array([r.get("msjd", [np.nan] * len(block_names)) for r in rounds]),
        node_t=np.array(diag["t"]), node_cov95=np.array(diag["cov95"]), node_mean=np.array(diag["mean"]),
        node_sd=np.array(diag["sd"]), node_truth=NODE_TRUTH, node_type=NODE_TYPE)))
if status == "ok" and f_ck.exists():
    f_ck.unlink()
print(f"[done] {stem} status={status} t={len(steps) - 1} rejuv={sys_.n_rej} "
      f"min ESS/P={result['exclusion']['min_ess_pre_over_P']:.4f} excluded={result['exclusion']['excluded']} "
      f"nrmse(0)={steps[0]['nrmse'] if steps else float('nan'):.4f} "
      f"nrmse(last)={steps[-1]['nrmse'] if steps else float('nan'):.4f} "
      f"{time.time() - t_start:.0f}s", flush=True)
sys.exit(3 if ABORTED else 0)
