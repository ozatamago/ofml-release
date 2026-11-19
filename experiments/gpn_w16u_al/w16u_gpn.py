#!/usr/bin/env python3
"""PB-W16U active learning with OFML-GPN (RB-SMC on the cx_bayes ``gpn_smc`` shared backend).

Inference = the ProtocolUCB BENCH9 production fast path for PB-W16U (``--particles 16384 --pcn-steps 96
--pcn-mode segment-blocked-obs --l2-null-refresh 1 --terminal-lowrank 64 --terminal-lowrank-min-n 120``,
buckets 40,60,80,100), built by importing ``gpn_smc`` (read-only) with the same constructor arguments as
``u15_compare.run_worker`` and assimilated with the same ``gpn_smc.assimilate_u15`` call and PcnBetaState as
``u15_compare.run_gpn_arm``. Differences to production: acquisition (IVR / MaxVar / space-filling / random
instead of UCB), no repeated observations, and the posterior summary is EXACT (mixture of per-particle
Rao-Blackwellised terminal GPs; production's UCB uses the Nystrom moments).

One process = one teacher (rep): the warm-up (8 initial points) is assimilated once and forked for every
policy (paired design). Output per (rep, policy): ``<out>/gpn/rep_XXXX_<policy>.json`` (per-round records) and
``rep_XXXX_<policy>.npz`` (per-round posterior mean / variance / PIT at all 1024 points, float32).
Checkpoints: ``<out>/ckpt/gpn_rep_XXXX_<policy>.npz`` every ``--checkpoint-every`` rounds (deleted when the
policy finishes) and ``gpn_rep_XXXX_warm.npz`` (the warm-up state).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w16u_common as C  # noqa: E402

PROD_FLAGS = dict(particles=16384, pcn_steps=96, pcn_mode="segment-blocked-obs", l2_null_refresh=1,
                  terminal_lowrank=64, terminal_lowrank_min_n=120, n_max_buckets=(40, 60, 80, 100),
                  ess_fraction=0.65, pcn_beta=0.30)


# ---------------------------------------------------------------------------------------------------------
# Exact posterior summary of the RB-SMC mixture (mean, variance or full covariance, PIT at the truth).

class MixtureSummary:
    """Per particle the terminal GP R is conditioned analytically on the observations given that particle's
    penultimate values h_i; the mixture has mean mu = sum_i w_i mu_i and covariance
    sum_i w_i (Sigma_i + mu_i mu_i^T) - mu mu^T (exact; generalises exp2_gpn_rb.MixtureCovariance to all
    candidates, with the per-particle covariance never materialised: sum_i w_i K_i(h,h) is a fused weighted
    reduction and sum_i w_i V_i^T V_i one GEMM per chunk). PIT = sum_i w_i Phi((f - mu_i)/s_i), the same
    definition as SharedProtocolGpnBackend.posterior_summary_extended."""

    def __init__(self, backend, noise_var: float, jitter: float, chunk: int = 128):
        import jax
        import jax.numpy as jnp
        self.backend = backend
        sf2, ell = C.AMPLITUDE ** 2, C.TERM_LS
        local = backend.local_particles
        if local % chunk:
            raise ValueError(f"local particles {local} not divisible by chunk {chunk}")
        n_chunks = local // chunk
        m = backend.n_candidates

        def kern(d):
            return sf2 * jnp.exp(-0.5 * (d / ell) ** 2)

        def make(full: bool):
            def per_device(h, w, obs_idx, y, active, truth):
                n_max = obs_idx.shape[0]
                eye = jnp.eye(n_max, dtype=jnp.float64)

                def body(carry, inp):
                    hc, wc = inp
                    zo = hc[:, obs_idx]
                    koo = kern(zo[:, :, None] - zo[:, None, :]) * (active[None, :, None] * active[None, None, :])
                    koo = koo + eye[None] * ((noise_var + jitter) * active + (1.0 - active))[None, :, None]
                    chol = jnp.linalg.cholesky(koo)
                    kco = kern(hc[:, :, None] - zo[:, None, :]) * active[None, None, :]      # (c, m, n)
                    rhs = jnp.broadcast_to(y[None, :, None], (hc.shape[0], n_max, 1))
                    tmp = jax.scipy.linalg.solve_triangular(chol, rhs, lower=True)
                    alpha = jax.scipy.linalg.solve_triangular(jnp.swapaxes(chol, 1, 2), tmp, lower=False)[..., 0]
                    mu = jnp.einsum("pmn,pn->pm", kco, alpha)
                    v = jax.scipy.linalg.solve_triangular(chol, jnp.swapaxes(kco, 1, 2), lower=True)  # (c, n, m)
                    var_i = jnp.maximum(sf2 - jnp.sum(v * v, axis=1), 0.0)
                    sd_i = jnp.sqrt(jnp.maximum(var_i, 1e-15))
                    s1, s2d, sw, sp, s2 = carry
                    s1 = s1 + wc @ mu
                    s2d = s2d + wc @ (var_i + mu * mu)
                    sw = sw + wc @ var_i
                    sp = sp + wc @ jax.scipy.special.ndtr((truth[None, :] - mu) / sd_i)
                    if full:
                        kk = jnp.einsum("p,pmk->mk", wc, kern(hc[:, :, None] - hc[:, None, :]))
                        sq = jnp.sqrt(wc)
                        vw = (v * sq[:, None, None]).reshape(-1, m)
                        muw = mu * sq[:, None]
                        s2 = s2 + kk - vw.T @ vw + muw.T @ muw
                    return (s1, s2d, sw, sp, s2), None

                z = jnp.zeros(m, jnp.float64)
                init = (z, z, z, z, jnp.zeros((m, m) if full else (1, 1), jnp.float64))
                xs = (h.reshape(n_chunks, chunk, m), w.reshape(n_chunks, chunk))
                out, _ = jax.lax.scan(body, init, xs)
                return out

            return jax.pmap(per_device, in_axes=(0, 0, None, None, None, None), devices=backend.devices)

        self._fn = {False: make(False), True: make(True)}

    def __call__(self, latent, weights: np.ndarray, obs_idx, y, active, truth, full: bool):
        """-> mean (m,), var (m,), within (m,), pit (m,), cov (m,m) or None."""
        h = self.backend.penultimate_outputs(latent)
        w = weights.reshape(self.backend.n_devices, self.backend.local_particles)
        s1, s2d, sw, sp, s2 = [np.sum(np.asarray(a), axis=0) for a in self._fn[bool(full)](
            h, w, obs_idx, y, active, truth)]
        mean = s1
        if full:
            cov = s2 - np.outer(mean, mean)
            cov = 0.5 * (cov + cov.T)
            var = np.maximum(np.diag(cov).copy(), 0.0)
        else:
            cov = None
            var = np.maximum(s2d - mean * mean, 0.0)
        return mean, var, sw, np.clip(sp, 0.0, 1.0), cov


# ---------------------------------------------------------------------------------------------------------
# SMC state (one trajectory), production assimilation call.

class SmcState:
    def __init__(self, ctx: dict, rep: int):
        gs, backend, a = ctx["gs"], ctx["backend"], ctx["args"]
        self.ctx = ctx
        self.inference_seed = C.teacher_seeds(rep)[2]
        self.n0 = len(C.INITIAL_FLAT)
        self.latent = backend.initialise(self.inference_seed)
        self.logw = np.full(backend.n_particles, -math.log(backend.n_particles))
        self.old_ll = np.zeros(backend.n_particles)
        self.observed: list[int] = []
        self.resamples = 0
        self.pcn = gs.PcnBetaState(mode=a.pcn_mode, fixed_beta=a.pcn_beta, n_blocks=backend.n_pcn_blocks)
        self.last_acc = float("nan")
        self.last_beta = float("nan")
        self.last = {}          # diagnostics of the latest assimilation

    def fork(self) -> "SmcState":
        new = copy.copy(self)
        new.logw = self.logw.copy()
        new.old_ll = self.old_ll.copy()
        new.observed = list(self.observed)
        new.pcn = copy.deepcopy(self.pcn)
        new.last = dict(self.last)
        return new  # latent is an immutable jax array

    def assimilate(self, index: int, noisy: np.ndarray) -> None:
        gs, backend, a = self.ctx["gs"], self.ctx["backend"], self.ctx["args"]
        self.observed.append(int(index))
        beta = self.pcn.current(len(self.observed))
        backend.last_assim_diag = None
        backend.last_lowrank_stats = None
        t0 = time.time()
        (self.latent, self.old_ll, self.logw, _ess, acc, self.resamples, rejuv) = gs.assimilate_u15(
            backend, self.latent, self.old_ll, self.logw, self.observed, noisy, a.ess_fraction, beta,
            self.inference_seed, self.n0, self.resamples)
        acc_now, beta_now = float("nan"), float("nan")
        if acc is not None:   # segment mode -> SegmentRejuvenationInfo (u15_compare._consume_assimilate_result)
            acc_now = float(np.mean(np.asarray(acc.last_segment_acceptance)))
            beta_now = float(np.mean(np.asarray(acc.beta_used_final_segment)))
            self.last_acc, self.last_beta = acc_now, beta_now
            self.pcn.update(acc)
        ad = getattr(backend, "last_assim_diag", None) or {}
        ls = getattr(backend, "last_lowrank_stats", None) or {}
        self.last = dict(ess_pre=float(ad.get("ess_pre", np.nan)) / backend.n_particles,
                         resampled=bool(ad.get("resampled", False)),
                         n_unique_ancestors=int(ad.get("n_unique_ancestors", -1)),
                         acc_now=acc_now, beta_now=beta_now,
                         acc_min_block=(float(np.min(np.asarray(acc.last_segment_acceptance)))
                                        if acc is not None else float("nan")),
                         lr_max_abs_dll=float(ls.get("max_abs_dll_final", np.nan)),
                         assim_seconds=time.time() - t0)

    def padded(self, noisy: np.ndarray):
        gs, backend = self.ctx["gs"], self.ctx["backend"]
        return gs.padded_observations_u15(self.observed, noisy, n_max=backend.n_max_for(len(self.observed)))

    # checkpoint ---------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        b = self.ctx["backend"]
        return dict(particles=b.host_particles(self.latent), logweights=self.logw, old_loglik=self.old_ll,
                    observed=np.asarray(self.observed, dtype=np.int64), resamples=self.resamples,
                    pcn_fixed_beta=self.pcn.fixed_beta, pcn_beta=self.pcn.beta,
                    pcn_beta_blocks=np.asarray(self.pcn.beta_blocks, dtype=np.float64),
                    last_acc=self.last_acc, last_beta=self.last_beta, last_json=json.dumps(self.last),
                    inference_seed=self.inference_seed, n_particles=b.n_particles)

    def load_state(self, z: dict) -> None:
        b = self.ctx["backend"]
        if int(z["inference_seed"]) != self.inference_seed or int(z["n_particles"]) != b.n_particles:
            raise RuntimeError("checkpoint belongs to another configuration")
        self.latent = b.shard_particles(z["particles"])
        self.logw = z["logweights"].copy()
        self.old_ll = z["old_loglik"].copy()
        self.observed = [int(i) for i in z["observed"]]
        self.resamples = int(z["resamples"])
        self.pcn.fixed_beta = float(z["pcn_fixed_beta"])
        self.pcn.beta = float(z["pcn_beta"])
        self.pcn.beta_blocks = z["pcn_beta_blocks"].copy()
        self.last_acc, self.last_beta = float(z["last_acc"]), float(z["last_beta"])
        self.last = json.loads(str(z["last_json"]))


def save_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------------------------------------

REC_KEYS = ("n_obs", "nrmse", "rmse", "ise", "iv", "max_var", "max_abs_err", "max_var_zeroshot",
            "max_abs_err_zeroshot", "coverage", "coverage_zeroshot", "nrmse_by_protocol", "ess", "ess_pre",
            "resampled", "n_unique_ancestors", "pcn_acceptance", "pcn_acceptance_now", "pcn_acc_min_block",
            "pcn_beta", "lr_max_abs_dll", "chosen", "summary_seconds", "assim_seconds", "seconds")


def run_policy(ctx: dict, rep: int, policy: str, warm: SmcState, truth: np.ndarray, noisy: np.ndarray,
               out_dir: Path) -> dict:
    gs, backend, a, summ = ctx["gs"], ctx["backend"], ctx["args"], ctx["summary"]
    ck = out_dir / "ckpt" / f"gpn_rep_{rep:04d}_{policy}.npz"
    st = warm.fork()
    rng = C.policy_rng(rep, policy)
    rec = {k: [] for k in REC_KEYS}
    arr = dict(mean=[], var=[], pit=[])
    t_start = time.time()
    if ck.exists():
        z = load_npz(ck)
        st.load_state(z)
        rec = json.loads(str(z["rec_json"]))
        for k in arr:
            arr[k] = list(z["arr_" + k])
        rng.bit_generator.state = json.loads(str(z["rng_json"]))
        print(f"[ckpt] rep {rep} {policy}: resume at n={len(st.observed)}", flush=True)
    resumed_n = len(st.observed) if ck.exists() else -1
    n0 = len(C.INITIAL_FLAT)
    while True:
        n = len(st.observed)
        t = n - n0
        if a.checkpoint_every > 0 and t > 0 and t % a.checkpoint_every == 0 and n != resumed_n:
            save_npz_atomic(ck, **st.state_dict(), rec_json=json.dumps(rec),
                            rng_json=json.dumps(rng.bit_generator.state),
                            **{"arr_" + k: np.asarray(v, dtype=np.float32).reshape(-1, C.N_CAND)
                               for k, v in arr.items()})
            if a.stop_after_round >= 0 and t >= a.stop_after_round:
                print(f"[test] stop after round {t}", flush=True)
                return dict(status="stopped")
        rt = time.time()
        obs_idx, y_pad, active = st.padded(noisy)
        weights, ess = gs.normalize_logweights(st.logw)
        mean, var, _within, pit, cov = summ(st.latent, weights, obs_idx, y_pad, active, truth,
                                            full=(policy == "ivr"))
        t_summary = time.time() - rt
        m = C.round_metrics(mean, var, pit, truth)
        rec["n_obs"].append(n)
        for k in ("nrmse", "rmse", "ise", "iv", "max_var", "max_abs_err", "max_var_zeroshot",
                  "max_abs_err_zeroshot", "coverage", "coverage_zeroshot", "nrmse_by_protocol"):
            rec[k].append(m[k])
        rec["ess"].append(float(ess) / backend.n_particles)
        rec["ess_pre"].append(st.last.get("ess_pre", float("nan")))
        rec["resampled"].append(bool(st.last.get("resampled", False)))
        rec["n_unique_ancestors"].append(int(st.last.get("n_unique_ancestors", -1)))
        rec["pcn_acceptance"].append(st.last_acc)
        rec["pcn_acceptance_now"].append(st.last.get("acc_now", float("nan")))
        rec["pcn_acc_min_block"].append(st.last.get("acc_min_block", float("nan")))
        rec["pcn_beta"].append(st.last_beta)
        rec["lr_max_abs_dll"].append(st.last.get("lr_max_abs_dll", float("nan")))
        rec["assim_seconds"].append(st.last.get("assim_seconds", float("nan")))
        rec["summary_seconds"].append(t_summary)
        arr["mean"].append(mean.astype(np.float32))
        arr["var"].append(var.astype(np.float32))
        arr["pit"].append(pit.astype(np.float32))
        if n >= C.T_TOTAL:
            rec["seconds"].append(time.time() - rt)
            break
        pick = C.choose(policy, st.observed, var, cov, rng)
        assert pick not in st.observed
        rec["chosen"].append(pick)
        st.assimilate(pick, noisy)
        rec["seconds"].append(time.time() - rt)
        if a.verbose:
            print(f"  rep {rep} {policy} n={n} nrmse={m['nrmse']:.4f} cov95={m['coverage'][2]:.3f} "
                  f"ess={ess / backend.n_particles:.3f} pre={st.last['ess_pre']:.3f} acc={st.last_acc:.3f} "
                  f"summ={t_summary:.2f}s assim={st.last['assim_seconds']:.2f}s", flush=True)
    excl = C.exclusion_check(rec["ess"], rec["pcn_acceptance"], rec["lr_max_abs_dll"])
    result = dict(model="gpn", rep=rep, policy=policy, status="ok", family=C.FAMILY,
                  teacher_seed=C.teacher_seeds(rep)[0], teacher_sha256=C.sha256_arrays(truth, noisy - truth),
                  initial=list(C.INITIAL_FLAT), observed=list(st.observed), config=ctx["config"],
                  warmup=ctx["warmup_info"], resamples=st.resamples, policy_seconds=time.time() - t_start,
                  exclusion=excl, rounds=rec)
    return result, arr


def build_backend(a):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(C.CX_BAYES))
    import gpn_smc as gs  # sets JAX_ENABLE_X64 before jax is imported
    import jax
    if a.compile_cache:
        jax.config.update("jax_compilation_cache_dir", a.compile_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
    spec = gs.ProtocolLibrarySpec()
    assert (spec.amplitude, spec.layer1_lengthscale, spec.layer2_lengthscale, spec.terminal_lengthscale) == (
        C.AMPLITUDE, C.L1_LS, C.L2_LS, C.TERM_LS)
    kw = dict(pcn_mode=a.pcn_mode, n_l2_types=C.N_L2, protocol_order="l1_major", n_l1_types=C.N_L1,
              n_l3_types=0, n_max_buckets=tuple(a.n_max_buckets))
    if a.pcn_mode == "segment-blocked-obs":
        kw.update(l2_null_refresh=a.l2_null_refresh, terminal_lowrank=a.terminal_lowrank,
                  terminal_lowrank_min_n=a.terminal_lowrank_min_n,
                  allow_repeat_observations=bool(a.backend_allow_repeats))
    t0 = time.time()
    backend = gs.SharedProtocolGpnBackend(C.WARPED_X, spec, C.NOISE_VAR, C.KERNEL_JITTER, a.particles,
                                          a.pcn_steps, **kw)
    backend.mixing_diagnostics = True   # only makes gpn_smc.assimilate_u15 record the pre-resampling ESS
    assert backend.n_protocols == C.N_PROTOCOLS and backend.n_candidates == C.N_CAND
    print(f"backend ready ({time.time() - t0:.1f}s): {backend.n_particles} particles on {backend.n_devices} "
          f"GPU(s) {jax.devices()}, buckets {backend.n_max_buckets}, kw={kw}", flush=True)
    return gs, backend


def add_inference_args(ap) -> None:
    p = PROD_FLAGS
    ap.add_argument("--particles", type=int, default=p["particles"])
    ap.add_argument("--pcn-steps", type=int, default=p["pcn_steps"])
    ap.add_argument("--pcn-mode", default=p["pcn_mode"])
    ap.add_argument("--l2-null-refresh", type=int, default=p["l2_null_refresh"])
    ap.add_argument("--terminal-lowrank", type=int, default=p["terminal_lowrank"])
    ap.add_argument("--terminal-lowrank-min-n", type=int, default=p["terminal_lowrank_min_n"])
    ap.add_argument("--n-max-buckets", type=lambda s: [int(v) for v in s.split(",")],
                    default=list(p["n_max_buckets"]))
    ap.add_argument("--ess-fraction", type=float, default=p["ess_fraction"])
    ap.add_argument("--pcn-beta", type=float, default=p["pcn_beta"])
    ap.add_argument("--backend-allow-repeats", type=int, default=0,
                    help="backend flag only (production BENCH9 used 1); observations are never repeated here")
    ap.add_argument("--chunk", type=int, default=128, help="particles per chunk of the mixture summary")
    ap.add_argument("--compile-cache", default="", help="JAX persistent compilation cache dir (speed only)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="0")
    ap.add_argument("--policies", default=",".join(C.POLICIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--stop-after-round", type=int, default=-1, help="test only: stop after this round")
    ap.add_argument("--verbose", action="store_true")
    add_inference_args(ap)
    a = ap.parse_args()
    out_dir = Path(a.out)
    gs, backend = build_backend(a)
    summary = MixtureSummary(backend, C.NOISE_VAR, C.KERNEL_JITTER, a.chunk)
    config = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(a).items()}
    import socket
    import jax
    config.update(n_devices=backend.n_devices, jax_x64=True, host=socket.gethostname(),
                  gpu=[d.device_kind for d in jax.devices()], slurm_job=os.environ.get("SLURM_JOB_ID", ""),
                  slurm_array=f"{os.environ.get('SLURM_ARRAY_JOB_ID', '')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '')}")
    ctx = dict(gs=gs, backend=backend, summary=summary, args=a, config=config)
    for rep in C.parse_range(a.reps):
        todo = [p for p in a.policies.split(",")
                if not (a.skip_existing and C.result_done(C.result_path(out_dir, "gpn", rep, p)))]
        if not todo:
            print(f"rep {rep}: all policies done", flush=True)
            continue
        truth, noise = C.load_teacher(rep)
        noisy = truth + noise
        wck = out_dir / "ckpt" / f"gpn_rep_{rep:04d}_warm.npz"
        warm = SmcState(ctx, rep)
        tw = time.time()
        warm_diag = []
        if wck.exists():
            z = load_npz(wck)
            warm.load_state(z)
            ctx["warmup_info"] = json.loads(str(z["info_json"]))
            print(f"rep {rep}: warm-up loaded from {wck}", flush=True)
        else:
            for idx in C.INITIAL_FLAT:
                warm.assimilate(idx, noisy)
                warm_diag.append(dict(warm.last))
            ctx["warmup_info"] = dict(seconds=time.time() - tw, resamples=warm.resamples, diag=warm_diag)
            if a.checkpoint_every > 0:
                save_npz_atomic(wck, **warm.state_dict(), info_json=json.dumps(ctx["warmup_info"]))
            print(f"rep {rep}: warm-up {time.time() - tw:.1f}s resamples={warm.resamples}", flush=True)
        for policy in todo:
            try:
                res = run_policy(ctx, rep, policy, warm, truth, noisy, out_dir)
                if isinstance(res, dict) and res.get("status") == "stopped":
                    continue
                r, arr = res
            except Exception as e:  # keep the sweep alive, record the failure
                import traceback
                traceback.print_exc()
                C.atomic_write_json(C.result_path(out_dir, "gpn", rep, policy).with_suffix(".fail.json"),
                                    dict(model="gpn", rep=rep, policy=policy, status="fail",
                                         error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-3000:]))
                continue
            rp = C.result_path(out_dir, "gpn", rep, policy)
            save_npz_atomic(rp.with_suffix(".npz"), **{k: np.stack(v).astype(np.float32) for k, v in arr.items()},
                            n_obs=np.asarray(r["rounds"]["n_obs"]))
            C.atomic_write_json(rp, r)
            ck = out_dir / "ckpt" / f"gpn_rep_{rep:04d}_{policy}.npz"
            if ck.exists():
                ck.unlink()
            rd = r["rounds"]
            print(f"rep {rep:3d} {policy:9s} NRMSE {rd['nrmse'][0]:.4f}->{rd['nrmse'][-1]:.4f} "
                  f"cov95 {rd['coverage'][0][2]:.3f}->{rd['coverage'][-1][2]:.3f} "
                  f"ESS/P min {min(rd['ess']):.3f} pre-min {np.nanmin(rd['ess_pre']):.3f} "
                  f"acc min {np.nanmin(rd['pcn_acceptance']):.3f} excl={r['exclusion']} "
                  f"({r['policy_seconds']:.0f}s)", flush=True)
    print("W16U_GPN_DONE", flush=True)


if __name__ == "__main__":
    main()
