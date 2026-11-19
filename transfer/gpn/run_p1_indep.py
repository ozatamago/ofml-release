"""Purpose 1 (c), GP-matched benchmark: Independent-GPN production runner, one problem (true function) per process.

Plan: REPORTS/EXPERIMENT_DESIGN_2026-11-04.md 1.8 (Independent-GPN = for every protocol, its own composition of
GPs with the same structure; no sharing across protocols).
* Model: for each of the 7 targets a private one-protocol library (ofml_gpn.term_library.build_term_library):
  inside the protocol, nodes of the same type share one function (e.g. both C1 nodes of R(C1(C1(A))), both M
  nodes of R(M(M(A,B),A))), and its own terminal R; nothing is shared between protocols.  The types are renamed
  canonically (layer-1 L1, L2 / unary U1, U2 / binary M1 in order of first appearance) -- the kernels only
  depend on the kind (layer 1: lengthscale 1.0; inner and R: 0.5; amplitude 0.70), so the prior is that of the
  plan, and isomorphic targets (C1(C1(A)) ~ C2(C2(A)) ...) share one compiled backend.
* Without sharing the source data say nothing about the targets, so it is not used: t = 0 is the prior.
* SMC: one independent system per target (default 65536 particles, pCN 384; ESS 0.65, systematic resampling,
  segment-blocked pCN with per-block beta adaptation, l2_null_refresh 1, no collapsed moves, exact terminal).
  A round updates only the system of the chosen target.
* IVR: p1_common.choose_ivr on the block-diagonal (7 x 64) covariance of f over the 448 target cells.
* Records as run_p1_main.py; per round the ESS / pCN diagnostics are those of the updated system.

usage: run_p1_indep.py --problem runs/p1_main/problems/rep_00.npz --out runs/p1_main/indep_gpn
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p1_gpn_tools as TL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--problem", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--particles", type=int, default=65536)
ap.add_argument("--pcn-steps", type=int, default=384)
ap.add_argument("--T", type=int, default=200)
ap.add_argument("--ess-fraction", type=float, default=0.65)
ap.add_argument("--beta0", type=float, default=0.30)
ap.add_argument("--chunk-mb", type=int, default=512)
ap.add_argument("--cov-chunk", type=int, default=256)
ap.add_argument("--buckets", default="16,32,48,64,96,128,160,200")
ap.add_argument("--checkpoint-every", type=int, default=20)
ap.add_argument("--stop-after-t", type=int, default=-1, help="timing only: stop (after a checkpoint) at this t")
ap.add_argument("--compile-cache", default="")
ap.add_argument("--skip-existing", action="store_true")
ap.add_argument("--abort-on-exclusion", action="store_true",
                help="stop as soon as a round's pre-resampling ESS/P < 0.001 (plan 1.8.5 rule); status 'excluded', exit code 3")
args = ap.parse_args()
TL.jax_setup(args.compile_cache)

import numpy as np  # noqa: E402

import p1_common as C  # noqa: E402
from ofml_gpn import ProtocolLibrarySpec  # noqa: E402
from ofml_gpn.graph_backends import GraphSharedGpnBackend  # noqa: E402
from ofml_gpn.term_library import build_term_library, parse_term, term_str  # noqa: E402

prob_path = Path(args.problem)
stem = prob_path.stem
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
f_json, f_npz, f_ck = out / f"{stem}.json", out / f"{stem}.npz", out / "ckpt" / f"{stem}.npz"
if args.skip_existing and f_json.exists() and json.load(open(f_json)).get("status") == "ok":
    print(f"[skip] {f_json} exists", flush=True)
    sys.exit(0)

prob = C.load_problem(prob_path)
T = int(args.T)
G = C.G
NT = len(C.TARGETS)
buckets = tuple(int(v) for v in args.buckets.split(","))
assert buckets[-1] >= T
LAYER1 = {"A", "B"}
ARITY = {"C1": 1, "C2": 1, "M": 2}


def canonical(term: str):
    """Rename types by kind in order of first appearance (post-order) -> (canonical term, mapping)."""
    mapping: dict[str, str] = {}
    count = {"L": 0, "U": 0, "M": 0}

    def walk(t):
        nm, args_ = t
        new_args = tuple(walk(a) for a in args_)
        if nm not in mapping:
            kind = "L" if nm in LAYER1 else ("U" if ARITY[nm] == 1 else "M")
            count[kind] += 1
            mapping[nm] = f"{kind}{count[kind]}"
        return (mapping[nm], new_args)

    return term_str(walk(parse_term(term))), mapping


backends: dict[str, tuple] = {}
targets = []
t_build = time.time()
for j, p in enumerate(C.TARGETS):
    term = C.PROTOCOL_TERMS[p]
    can, mapping = canonical(term)
    if can not in backends:
        kinds = sorted(set(mapping.values()))
        layer1 = [k for k in kinds if k[0] == "L"]
        inner = {k: (1 if k[0] == "U" else 2) for k in kinds if k[0] != "L"}
        tl = build_term_library(f"IND-{can}", layer1, inner, [can], [], require_target_types_in_sources=False)
        b = GraphSharedGpnBackend(
            C.WX, ProtocolLibrarySpec(), tl.library, C.NOISE_SD ** 2, C.JIT, args.particles, args.pcn_steps,
            n_max_buckets=buckets, allow_repeat_observations=True, fast_chol=True,
            chunk_target_bytes=args.chunk_mb << 20, l2_null_refresh=1, inner_collapsed_sweeps=0, mixing_diagnostics=True)
        assert b.n_candidates == G
        summ = TL.MixtureSummary(b, np.arange(G), chunk=args.cov_chunk)
        backends[can] = (b, summ, [t.name for t in tl.library.types])
        print(f"[setup] backend {can}: latent={b.grid_size} blocks={backends[can][2]} legacy={b._legacy} "
              f"particles={b.n_particles} pcn={b.pcn_steps}", flush=True)
    targets.append(dict(j=j, protocol=p, term=term, canonical=can, mapping=mapping))
print(f"[setup] {stem}: {len(backends)} distinct backends for {NT} targets, buckets={buckets}, "
      f"build {time.time() - t_build:.1f}s", flush=True)

systems = []
for tg in targets:
    b, summ, names = backends[tg["canonical"]]
    systems.append(TL.SmcSystem(b, prob.seed + 10 + tg["j"], 0, args.ess_fraction, args.beta0))
config = dict(method="Independent-GPN", problem=str(prob_path), seed=prob.seed,
              inference_seeds=[s.seed for s in systems], particles=args.particles, pcn_steps=args.pcn_steps,
              ess_fraction=args.ess_fraction, beta0=args.beta0, l2_null_refresh=1, inner_collapsed_sweeps=0,
              terminal="exact", fast_chol=True, buckets=list(buckets), T=T, source_data_used=False,
              targets=[dict(term=tg["term"], canonical=tg["canonical"], mapping=tg["mapping"],
                            pcn_blocks=backends[tg["canonical"]][2]) for tg in targets],
              **TL.host_info())

visits = np.zeros(C.N_TGT_CELLS, dtype=np.int64)
rounds: list[dict] = []
steps: list[dict] = []
arr = dict(mean=[], var=[], pit=[])
t_resume = 0
if f_ck.exists():
    z = TL.load_npz(f_ck)
    for j, s in enumerate(systems):
        s.load(z, f"s{j}_")
    visits = z["visits"].copy()
    rounds = json.loads(str(z["rounds_json"]))
    steps = json.loads(str(z["steps_json"]))
    for k in arr:
        arr[k] = list(z["arr_" + k])
    t_resume = len(steps)
    print(f"[ckpt] resume at t={t_resume}", flush=True)


def checkpoint():
    st = {}
    for j, s in enumerate(systems):
        st.update(s.state(f"s{j}_"))
    TL.save_npz_atomic(f_ck, **st, visits=visits, rounds_json=json.dumps(rounds), steps_json=json.dumps(steps),
                       **{"arr_" + k: np.asarray(v, dtype=np.float32).reshape(-1, C.N_TGT_CELLS) for k, v in arr.items()})


def block_summary(j):
    s = systems[j]
    b, summ, _ = backends[targets[j]["canonical"]]
    obs_idx, y_pad, active = s.padded()
    w, ess = s.weights()
    return summ(s.latent, w, obs_idx, y_pad, active, prob.truth_tgt[j * G:(j + 1) * G]) + (float(ess) / b.n_particles,)


ABORTED = False
t_start = time.time()
cache = [None] * NT      # (mean, cov, pit, ess) per target; recomputed only for the updated system
t = len(steps)
while t <= T:
    t0 = time.time()
    for j in range(NT):
        if cache[j] is None:
            cache[j] = block_summary(j)
    mean = np.concatenate([c[0] for c in cache])
    pit = np.concatenate([c[2] for c in cache])
    cov = np.zeros((C.N_TGT_CELLS, C.N_TGT_CELLS))
    for j in range(NT):
        cov[j * G:(j + 1) * G, j * G:(j + 1) * G] = cache[j][1]
    var = np.maximum(np.diag(cov), 0.0)
    t_sum = time.time() - t0
    st = dict(t=t, nrmse=C.nrmse(mean, prob), max_var=float(var.max()), mean_var=float(var.mean()),
              coverage=TL.coverage_from_pit(pit), ess_over_P=[c[3] for c in cache], summary_seconds=t_sum)
    arr["mean"].append(mean.astype(np.float32))
    arr["var"].append(var.astype(np.float32))
    arr["pit"].append(pit.astype(np.float32))
    if t < T:
        t1 = time.time()
        c = C.choose_ivr(cov)
        st["acq_seconds"] = time.time() - t1
        y = C.observe(prob, c, visits)
        j, g = divmod(c, G)
        rec = systems[j].assimilate(g, y)
        cache[j] = None
        rec.update(phase="target", t=t + 1, candidate=c, target=j, visit=int(visits[c]), seconds=time.time() - t0)
        st["chosen"] = c
        st["seconds"] = rec["seconds"]
        rounds.append(rec)
        short = {k: v for k, v in rec.items() if k not in ("acc_trace", "beta_trace", "uniq_after")}
        if "acc_trace" in rec:
            short["acc_last"] = np.round(rec["acc_trace"][-1], 3).tolist()
        print("[round] " + json.dumps(short), flush=True)
        v = rec.get("ess_pre_over_P")
        if args.abort_on_exclusion and v is not None and np.isfinite(v) and v < 1e-3:
            ABORTED = True
            print(f"[abort] exclusion rule hit at t={t + 1} (target {j}): ESS/P={v:.2e}", flush=True)
    else:
        st["seconds"] = time.time() - t0
    steps.append(st)
    print(f"[t={t}] nrmse={st['nrmse']:.4f} maxvar={st['max_var']:.4f} cov50/80/95="
          f"{'/'.join(f'{v:.3f}' for v in st['coverage'])} summ={t_sum:.2f}s total={st['seconds']:.1f}s", flush=True)
    t += 1
    if ABORTED:
        checkpoint()
        break
    if (args.checkpoint_every > 0 and t % args.checkpoint_every == 0 and t <= T) or t - 1 == args.stop_after_t:
        checkpoint()
    if args.stop_after_t >= 0 and t - 1 >= args.stop_after_t and t <= T:
        break

status = "excluded" if ABORTED else ("ok" if len(steps) == T + 1 else "partial")
n_seg = max(b.n_pcn_segments for b, _, _ in backends.values())
n_blk = max(b.n_pcn_blocks for b, _, _ in backends.values())
ess_pre = [r["ess_pre_over_P"] for r in rounds]
result = dict(
    status=status, config=config, total_seconds_this_process=time.time() - t_start, resumed_from_t=t_resume,
    n_rejuvenations=[s.n_rej for s in systems], n_obs_per_target=[len(s.observed) for s in systems],
    exclusion=TL.exclusion_flag(ess_pre),
    nrmse=[s["nrmse"] for s in steps], aulc=(C.aulc(np.array([s["nrmse"] for s in steps])) if status == "ok" else None),
    chosen=[s["chosen"] for s in steps if "chosen" in s], steps=steps, rounds=rounds)
TL.save_json_atomic(f_json, result)
TL.save_npz_atomic(
    f_npz, nrmse=np.array([s["nrmse"] for s in steps]), chosen=np.array(result["chosen"], dtype=np.int64),
    max_var=np.array([s["max_var"] for s in steps]), coverage=np.array([s["coverage"] for s in steps]),
    step_seconds=np.array([s["seconds"] for s in steps]),
    mean=np.asarray(arr["mean"], np.float32), var=np.asarray(arr["var"], np.float32), pit=np.asarray(arr["pit"], np.float32),
    round_target=np.array([r["target"] for r in rounds], dtype=np.int64),
    ess_pre_over_P=np.array(ess_pre, dtype=np.float64), resampled=np.array([r["resampled"] for r in rounds]),
    n_unique_ancestors=np.array([r["n_unique_ancestors"] for r in rounds]),
    acc=TL.pad_traces(rounds, "acc_trace", n_seg, n_blk), beta=TL.pad_traces(rounds, "beta_trace", n_seg, n_blk),
    round_seconds=np.array([r["seconds"] for r in rounds]),
    rejuv_seconds=np.array([r.get("rejuv_seconds", np.nan) for r in rounds]), truth_tgt=prob.truth_tgt)
if status == "ok" and f_ck.exists():
    f_ck.unlink()
print(f"[done] {stem} status={status} t={len(steps) - 1} rejuv={[s.n_rej for s in systems]} "
      f"min ESS/P={result['exclusion']['min_ess_pre_over_P']:.4f} excluded={result['exclusion']['excluded']} "
      f"nrmse(0)={steps[0]['nrmse']:.4f} nrmse(last)={steps[-1]['nrmse']:.4f} {time.time() - t_start:.0f}s", flush=True)
sys.exit(3 if ABORTED else 0)
