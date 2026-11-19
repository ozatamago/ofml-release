"""Purpose 1 (c), GP-matched benchmark: helpers shared by the two GP runners (run_p1_main.py = OFML-GPN,
run_p1_indep.py = Independent-GPN).  p1_common.py (data, noise table, IVR, NRMSE) is NOT changed; this file only
adds what the GP runners need on top of it.

* ``MixtureSummary``: exact posterior of f at a set of candidate cells under the RB-SMC mixture.  Per particle the
  terminal GP R is conditioned analytically on the observations given that particle's penultimate values h_i
  (the same kernel, jitter and masking as ``ofml_gpn.terminal_rb``), so the mixture has mean
  mu = sum_i w_i mu_i and covariance sum_i w_i (Sigma_i + mu_i mu_i^T) - mu mu^T (Sigma_i = the conditional
  covariance of f, no observation noise).  PIT at the truth = sum_i w_i Phi((f - mu_i) / s_i); the central
  alpha band of the mixture contains the truth iff |PIT - 1/2| <= alpha / 2 (the mixture CDF is continuous and
  increasing), so the 50/80/95% coverages are exact without quantile bisection.  Generalises
  ``acquisition/w16u_gpn.MixtureSummary`` to a subset of the backend's candidates.
* ``SmcSystem``: one particle system (backend + latent + weights + pCN betas) with the production assimilation
  call (``ofml_gpn.assimilate_u15``, systematic resampling at ESS < 0.65 P, derive_stream_seed streams) and
  per-round diagnostics (pre-resampling ESS / P, resampling flag, distinct ancestors, per-block pCN acceptance
  and beta of every segment, rejuvenation seconds), plus checkpoint state.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

COVERAGE_BANDS = (0.50, 0.80, 0.95)


def jax_setup(compile_cache: str = "") -> None:
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    if compile_cache:
        jax.config.update("jax_compilation_cache_dir", compile_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)


def host_info() -> dict:
    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                              os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return dict(host=platform.node(), gpu=gpu, slurm_job=os.environ.get("SLURM_JOB_ID", ""),
                slurm_array=f"{os.environ.get('SLURM_ARRAY_JOB_ID', '')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '')}",
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""))


# ---------------------------------------------------------------------------------------------------------
class MixtureSummary:
    def __init__(self, backend, cells: np.ndarray, chunk: int = 64):
        import jax
        import jax.numpy as jnp
        spec = backend.library_spec
        sf2, ell = float(spec.amplitude) ** 2, float(spec.terminal_lengthscale)
        noise_var, jitter = float(backend._noise_variance), float(backend._jitter_const)
        self.backend = backend
        self.cells = np.asarray(cells, dtype=np.int32)
        m = int(self.cells.size)
        local = backend.local_particles
        chunk = min(int(chunk), local)
        if local % chunk:
            raise ValueError(f"local particles {local} not divisible by chunk {chunk}")
        n_chunks = local // chunk
        cells_j = jnp.asarray(self.cells)

        def kern(d):
            return sf2 * jnp.exp(-0.5 * (d / ell) ** 2)

        def per_device(h, w, obs_idx, y, active, truth):
            n_max = obs_idx.shape[0]
            eye = jnp.eye(n_max, dtype=jnp.float64)

            def body(carry, inp):
                hfull, wc = inp
                hc = hfull[:, cells_j]
                zo = hfull[:, obs_idx]
                koo = kern(zo[:, :, None] - zo[:, None, :]) * (active[None, :, None] * active[None, None, :])
                koo = koo + eye[None] * ((noise_var + jitter) * active + (1.0 - active))[None, :, None]
                chol = jnp.linalg.cholesky(koo)
                kco = kern(hc[:, :, None] - zo[:, None, :]) * active[None, None, :]        # (c, m, n)
                rhs = jnp.broadcast_to(y[None, :, None], (hc.shape[0], n_max, 1))
                tmp = jax.scipy.linalg.solve_triangular(chol, rhs, lower=True)
                alpha = jax.scipy.linalg.solve_triangular(jnp.swapaxes(chol, 1, 2), tmp, lower=False)[..., 0]
                mu = jnp.einsum("pmn,pn->pm", kco, alpha)
                v = jax.scipy.linalg.solve_triangular(chol, jnp.swapaxes(kco, 1, 2), lower=True)  # (c, n, m)
                var_i = jnp.maximum(sf2 - jnp.sum(v * v, axis=1), 0.0)
                sd_i = jnp.sqrt(jnp.maximum(var_i, 1e-15))
                s1, sp, s2 = carry
                s1 = s1 + wc @ mu
                sp = sp + wc @ jax.scipy.special.ndtr((truth[None, :] - mu) / sd_i)
                kk = jnp.einsum("p,pmk->mk", wc, kern(hc[:, :, None] - hc[:, None, :]))
                sq = jnp.sqrt(wc)
                vw = (v * sq[:, None, None]).reshape(-1, m)
                muw = mu * sq[:, None]
                s2 = s2 + kk - vw.T @ vw + muw.T @ muw
                return (s1, sp, s2), None

            z = jnp.zeros(m, jnp.float64)
            init = (z, z, jnp.zeros((m, m), jnp.float64))
            xs = (h.reshape(n_chunks, chunk, h.shape[-1]), w.reshape(n_chunks, chunk))
            out, _ = jax.lax.scan(body, init, xs)
            return out

        self._fn = jax.pmap(per_device, in_axes=(0, 0, None, None, None, None), devices=backend.devices)

    def __call__(self, latent, weights, obs_idx, y, active, truth_cells):
        """-> mean (m,), cov (m, m), pit (m,)."""
        b = self.backend
        h = b.penultimate_outputs(latent)
        w = np.asarray(weights, dtype=np.float64).reshape(b.n_devices, b.local_particles)
        s1, sp, s2 = [np.sum(np.asarray(a), axis=0) for a in self._fn(
            h, w, obs_idx, y, active, np.asarray(truth_cells, dtype=np.float64))]
        cov = s2 - np.outer(s1, s1)
        cov = 0.5 * (cov + cov.T)
        return s1, cov, np.clip(sp, 0.0, 1.0)


def coverage_from_pit(pit: np.ndarray) -> list[float]:
    return [float(np.mean(np.abs(pit - 0.5) <= 0.5 * a)) for a in COVERAGE_BANDS]


# ---------------------------------------------------------------------------------------------------------
class SmcSystem:
    """One RB-SMC particle system on a GraphSharedGpnBackend."""

    def __init__(self, backend, inference_seed: int, n_initial: int, ess_fraction: float, beta0: float = 0.30):
        self.b = backend
        self.seed = int(inference_seed)
        self.n_initial = int(n_initial)
        self.ess_fraction = float(ess_fraction)
        self.latent = backend.initialise(self.seed)
        self.logw = np.full(backend.n_particles, -math.log(backend.n_particles))
        self.ll = np.zeros(backend.n_particles)
        self.observed: list[int] = []
        self.ys: list[float] = []
        self.beta = np.full(backend.n_pcn_blocks, float(beta0))
        self.n_rej = 0
        # 2026-11-06 additive (diagnostics): ancestor ids of the current particles in the initial population
        # (lineage0) and in the population at the end of the sources (lineage_src, set at n = n_initial).
        self.lineage0 = np.arange(backend.n_particles)
        self.lineage_src: np.ndarray | None = None

    def padded(self):
        from ofml_gpn import padded_observations_u15
        return padded_observations_u15(self.observed, np.zeros(self.b.n_candidates),
                                       n_max=self.b.n_max_for(len(self.observed)),
                                       observed_y=np.asarray(self.ys, dtype=np.float64))

    def weights(self):
        from ofml_gpn.numerics import normalize_logweights
        return normalize_logweights(self.logw)

    def assimilate(self, cell: int, y: float) -> dict:
        from ofml_gpn import assimilate_u15
        b = self.b
        self.observed.append(int(cell))
        self.ys.append(float(y))
        b.last_assim_diag = None
        b.last_mixing_diag = None
        t0 = time.time()
        self.latent, self.ll, self.logw, ess, acc, self.n_rej, rejuv = assimilate_u15(
            b, self.latent, self.ll, self.logw, self.observed, np.zeros(b.n_candidates), self.ess_fraction,
            self.beta, self.seed, self.n_initial, self.n_rej, observed_y=np.asarray(self.ys, dtype=np.float64))
        ad = b.last_assim_diag or {}
        rec = dict(n=len(self.observed), cell=int(cell), assim_seconds=time.time() - t0,
                   ess_pre_over_P=float(ad.get("ess_pre", np.nan)) / b.n_particles,
                   resampled=bool(ad.get("resampled", False)),
                   n_unique_ancestors=int(ad.get("n_unique_ancestors", -1)))
        if "log_evidence_increment" in ad:
            idx = ad.get("indices")
            if idx is not None:
                self.lineage0 = self.lineage0[idx]
                if self.lineage_src is not None:
                    self.lineage_src = self.lineage_src[idx]
            if len(self.observed) == self.n_initial:
                self.lineage_src = np.arange(b.n_particles)
            rec.update(log_evidence_increment=float(ad["log_evidence_increment"]),
                       lineages0=int(np.unique(self.lineage0).size),
                       lineages_src=(int(np.unique(self.lineage_src).size) if self.lineage_src is not None else None))
        if acc is not None:
            md = b.last_mixing_diag or {}
            self.beta = np.asarray(acc.beta_next, dtype=np.float64).copy()
            rec.update(rejuv_seconds=float(md.get("rejuv_seconds", np.nan)),
                       acc_trace=np.asarray(acc.acceptance_trace, dtype=np.float64).reshape(b.n_pcn_segments, -1).tolist(),
                       beta_trace=np.asarray(acc.beta_trace, dtype=np.float64).reshape(b.n_pcn_segments, -1).tolist(),
                       msjd=[float(v) for v in md.get("msjd", [])],
                       uniq_after=[int(v) for v in md.get("uniq_after", [])])
        return rec

    # checkpoint ------------------------------------------------------------------------------------------
    def state(self, prefix: str) -> dict:
        return {prefix + "particles": self.b.host_particles(self.latent), prefix + "logw": self.logw,
                prefix + "ll": self.ll, prefix + "observed": np.asarray(self.observed, dtype=np.int64),
                prefix + "ys": np.asarray(self.ys, dtype=np.float64), prefix + "beta": self.beta,
                prefix + "n_rej": np.int64(self.n_rej), prefix + "seed": np.int64(self.seed)}

    def load(self, z: dict, prefix: str) -> None:
        if int(z[prefix + "seed"]) != self.seed or z[prefix + "particles"].shape[0] != self.b.n_particles:
            raise RuntimeError("checkpoint belongs to another configuration")
        self.latent = self.b.shard_particles(z[prefix + "particles"])
        self.logw = z[prefix + "logw"].copy()
        self.ll = z[prefix + "ll"].copy()
        self.observed = [int(i) for i in z[prefix + "observed"]]
        self.ys = [float(v) for v in z[prefix + "ys"]]
        self.beta = z[prefix + "beta"].copy()
        self.n_rej = int(z[prefix + "n_rej"])


def save_npz_atomic(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def save_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def pad_traces(rounds: list[dict], key: str, n_seg: int, n_blk: int) -> np.ndarray:
    """(n_rounds, n_seg, n_blk) float array of a per-round trace (NaN where the round had no rejuvenation)."""
    out = np.full((len(rounds), n_seg, n_blk), np.nan)
    for i, r in enumerate(rounds):
        if key in r:
            a = np.asarray(r[key], dtype=np.float64)
            out[i, :a.shape[0], :a.shape[1]] = a
    return out


def exclusion_flag(ess_pre: list[float], threshold: float = 1e-3) -> dict:
    """Plan 1.8.5 (fixed 2026-11-05): a function is excluded if any round's pre-resampling ESS / P < 0.001."""
    e = np.asarray([v for v in ess_pre if v is not None and np.isfinite(v)], dtype=np.float64)
    mn = float(e.min()) if e.size else float("nan")
    return dict(threshold=threshold, min_ess_pre_over_P=mn, excluded=bool(e.size and mn < threshold),
                n_rounds_below=int(np.sum(e < threshold)))
