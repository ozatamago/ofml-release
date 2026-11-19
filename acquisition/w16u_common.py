"""Shared, numpy-only definitions for the PB-W16U active-learning experiment (design 2026-11-04, aim 2).

Importable from both the JAX (cx_bayes) environment and the torch (ofml) environment.

Benchmark = ProtocolUCB BENCH9 PB-W16U (family_index 129): depth-3 chain, layer 1 {A, B}, layer 2 C1..C8 each
applied to A and B, shared terminal R; protocol p = i*8 + j is R . C_{j+1} . L1_i (i=0: A, i=1: B), i.e. the
L1-major order (A,C1)..(A,C8),(B,C1)..(B,C8). Uniform grid linspace(0,4,64) per protocol; flat candidate
index = p*64 + k (1024 candidates). Teacher drawn from the learner's own prior (GP-matched).

Teachers are generated once by ``make_teachers.py`` with u15_compare's own functions and cached under
``teachers/``; both runners load the cache and check its sha256, so truth is bit-identical across the two
Python environments.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# gpn_smc is distributed in third_party/ (copy of cx_bayes/gpn_smc, see third_party/gpn_smc/PROVENANCE.md).
# make_teachers.py alone also needs the unpublished cx_bayes experiments (OFML_CX_BAYES); the teachers are cached.
CX_BAYES = HERE.parent / "third_party"
U15_EXPERIMENTS = Path(os.environ.get("OFML_CX_BAYES", "cx_bayes")) / "ucb" / "noisy_ucb_experiments" / "experiments"
TEACHER_DIR = HERE / "teachers"

FAMILY, FAMILY_INDEX = "PB-W16U", 129
N_L1, N_L2, G = 2, 8, 64
N_PROTOCOLS = N_L1 * N_L2
N_CAND = N_PROTOCOLS * G
WARPED_X = np.linspace(0.0, 4.0, G)          # u15_compare.UNIFORM_WARPED_X
AMPLITUDE, L1_LS, L2_LS, TERM_LS = 0.70, 1.0, 0.50, 0.50
NOISE_STD = 0.10
NOISE_VAR = NOISE_STD ** 2
KERNEL_JITTER = 1e-8
SEED_BASE, FAMILY_SPACING = 41_000_000, 100_000
INITIAL_PROTOCOLS = tuple(range(0, N_PROTOCOLS, 2))   # BENCH9 PB-W16U registry entry
INITIAL_GRID_INDEX = 32                               # initial_indices(u10 x, 1) = argmin|x| (verified in tests)
INITIAL_FLAT = tuple(p * G + INITIAL_GRID_INDEX for p in INITIAL_PROTOCOLS)
T_TOTAL = 100
POLICIES = ("ivr", "maxvar", "spacefill", "random")
COVERAGE_LEVELS = (0.5, 0.8, 0.95)
IVR_NOISE_VAR = 0.01


def protocol_types(p: int) -> tuple[int, int]:
    """(layer-1 type index, layer-2 type index) of protocol p (L1-major)."""
    return divmod(int(p), N_L2)


def teacher_seeds(rep: int) -> tuple[int, int, int]:
    """(teacher_seed, noise_seed, inference_seed), BENCH9 rule."""
    s = SEED_BASE + FAMILY_SPACING * FAMILY_INDEX + 10 * int(rep)
    return s, s + 1, s + 2


def rbf_covariance(values: np.ndarray, amplitude: float, lengthscale: float) -> np.ndarray:
    difference = values[:, None] - values[None, :]
    return amplitude**2 * np.exp(-0.5 * (difference / lengthscale) ** 2)


def stable_cholesky(matrix: np.ndarray, initial_jitter: float) -> tuple[np.ndarray, float]:
    jitter = initial_jitter
    identity = np.eye(len(matrix), dtype=np.float64)
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + jitter * identity), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError(f"Cholesky failed through jitter={jitter:g}")


def draw_teacher_reimpl(rep: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent re-implementation of u15_compare.draw_pb_w16_teacher on the uniform grid (test only;
    production code uses the cache written from u15_compare itself)."""
    teacher_seed, noise_seed, _ = teacher_seeds(rep)
    rng = np.random.default_rng(teacher_seed)
    l1, _ = stable_cholesky(rbf_covariance(WARPED_X, AMPLITUDE, L1_LS), KERNEL_JITTER)
    f_a = l1 @ rng.normal(size=G)
    f_b = l1 @ rng.normal(size=G)
    l2, _ = stable_cholesky(rbf_covariance(np.concatenate([f_a, f_b]), AMPLITUDE, L2_LS), KERNEL_JITTER)
    a_parts, b_parts = [], []
    for _ in range(N_L2):
        f2 = l2 @ rng.normal(size=2 * G)
        a_parts.append(f2[:G])
        b_parts.append(f2[G:])
    pen = np.concatenate(a_parts + b_parts)
    lr, _ = stable_cholesky(rbf_covariance(pen, AMPLITUDE, TERM_LS), KERNEL_JITTER)
    truth = lr @ rng.normal(size=N_CAND)
    noise = np.random.default_rng(noise_seed).normal(scale=NOISE_STD, size=(N_PROTOCOLS, G)).reshape(-1)
    return truth, noise


def sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def teacher_path(rep: int) -> Path:
    return TEACHER_DIR / f"rep_{int(rep):04d}.npz"


def load_teacher(rep: int) -> tuple[np.ndarray, np.ndarray]:
    """(truth, noise) from the cache, sha256-checked."""
    path = teacher_path(rep)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run make_teachers.py --reps {rep} first")
    with np.load(path, allow_pickle=False) as z:
        truth, noise = z["truth"].astype(np.float64), z["noise"].astype(np.float64)
        digest = str(z["sha256"])
        if int(z["teacher_seed"]) != teacher_seeds(rep)[0]:
            raise RuntimeError(f"{path}: teacher_seed mismatch")
    if sha256_arrays(truth, noise) != digest:
        raise RuntimeError(f"{path}: sha256 mismatch")
    return truth, noise


# ---------------------------------------------------------------------------------------------------------
# Policies (numpy, shared by both surrogates).

def vdc_order(n: int) -> list[int]:
    bits = int(math.log2(n))
    return [int(format(k, f"0{bits}b")[::-1], 2) for k in range(n)]


SPACEFILL_SEQUENCE = np.array(
    [(k % N_PROTOCOLS) * G + vdc_order(G)[k // N_PROTOCOLS] for k in range(N_CAND)], dtype=np.int64)


def spacefill_position(t: int, unobserved: set[int]) -> int:
    """exp2_common.spacefill_position generalised to 16 protocols over all 1024 candidates: cycle the
    protocols, van der Corput order within each, skip observed points. Returns a flat index."""
    vdc = vdc_order(G)
    k = t
    while True:
        flat = (k % N_PROTOCOLS) * G + vdc[(k // N_PROTOCOLS) % G]
        if flat in unobserved:
            return flat
        k += 1


def spacefill_pick(observed: list[int] | set[int]) -> int:
    """First unobserved point of the fixed space-filling sequence (equivalent to spacefill_position, see
    tests/test_policies.py)."""
    obs = set(int(i) for i in observed)
    for flat in SPACEFILL_SEQUENCE:
        if int(flat) not in obs:
            return int(flat)
    raise RuntimeError("no unobserved candidate left")


def policy_rng(rep: int, policy: str) -> np.random.Generator:
    """RNG of a policy (only 'random' consumes it); depends on rep and policy only, so it is the same for
    both surrogates."""
    return np.random.default_rng([teacher_seeds(rep)[0], 7, POLICIES.index(policy)])


def unobserved_mask(observed) -> np.ndarray:
    mask = np.ones(N_CAND, dtype=bool)
    mask[np.asarray(list(observed), dtype=np.int64)] = False
    return mask


def ivr_scores(cov: np.ndarray, noise_var: float = IVR_NOISE_VAR) -> np.ndarray:
    """score(x) = sum_{x'} k(x', x)^2 / (k(x, x) + noise_var), reference set = all 1024 candidates."""
    return np.sum(cov * cov, axis=0) / (np.diag(cov) + noise_var)


def choose(policy: str, observed: list[int], var: np.ndarray | None, cov: np.ndarray | None,
           rng: np.random.Generator) -> int:
    """Next flat index; never an observed point."""
    mask = unobserved_mask(observed)
    cand = np.flatnonzero(mask)
    if policy == "random":
        return int(rng.choice(cand))
    if policy == "spacefill":
        return spacefill_pick(observed)
    if policy == "maxvar":
        return int(cand[np.argmax(var[cand])])
    if policy == "ivr":
        s = ivr_scores(cov)
        return int(cand[np.argmax(s[cand])])
    raise ValueError(policy)


# ---------------------------------------------------------------------------------------------------------
# Metrics.

def truth_baseline_rmse(truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - truth.mean()) ** 2)))


ZERO_SHOT_PROTOCOLS = tuple(p for p in range(N_PROTOCOLS) if p not in INITIAL_PROTOCOLS)   # odd protocols
ZERO_SHOT_FLAT = np.array([p * G + k for p in ZERO_SHOT_PROTOCOLS for k in range(G)], dtype=np.int64)


def gaussian_pit(mean: np.ndarray, var: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Gaussian PIT Phi((f - mu)/sd) (used for OFML-Ens, as in exp2)."""
    from math import erf
    sd = np.sqrt(np.maximum(var, 1e-24))
    z = (truth - mean) / sd
    return 0.5 * (1.0 + np.vectorize(erf)(z / math.sqrt(2.0)))


def round_metrics(mean: np.ndarray, var: np.ndarray, pit: np.ndarray, truth: np.ndarray) -> dict:
    err2 = (mean - truth) ** 2
    abs_err = np.sqrt(err2)
    rmse = float(np.sqrt(np.mean(err2)))
    zs = ZERO_SHOT_FLAT
    return dict(
        nrmse=rmse / truth_baseline_rmse(truth),
        rmse=rmse,
        ise=float(np.mean(err2)),
        iv=float(np.mean(var)),
        max_var=float(np.max(var)),
        max_abs_err=float(np.max(abs_err)),
        max_var_zeroshot=float(np.max(var[zs])),
        max_abs_err_zeroshot=float(np.max(abs_err[zs])),
        coverage=[float(np.mean(np.abs(pit - 0.5) <= a / 2)) for a in COVERAGE_LEVELS],
        coverage_zeroshot=[float(np.mean(np.abs(pit[zs] - 0.5) <= a / 2)) for a in COVERAGE_LEVELS],
        nrmse_by_protocol=[float(np.sqrt(np.mean(err2[p * G:(p + 1) * G])) /
                                 truth_baseline_rmse(truth)) for p in range(N_PROTOCOLS)],
    )


# ---------------------------------------------------------------------------------------------------------
# Small IO helpers.

def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def result_path(out_dir: Path, model: str, rep: int, policy: str) -> Path:
    return Path(out_dir) / model / f"rep_{int(rep):04d}_{policy}.json"


def result_done(path: Path) -> bool:
    if not Path(path).exists():
        return False
    try:
        return json.loads(Path(path).read_text()).get("status") == "ok"
    except (json.JSONDecodeError, OSError):
        return False


def exclusion_check(ess_over_p: list[float], acceptance: list[float], lr_dll: list[float]) -> dict:
    """Preset exclusion rule (design 2.6 / BENCH9 spec 4), inference diagnostics only."""
    ess = np.asarray(ess_over_p, dtype=float)
    acc = np.asarray(acceptance, dtype=float)
    dll = np.asarray(lr_dll, dtype=float)
    low = np.nan_to_num(acc, nan=1.0) < 0.05
    run, longest = 0, 0
    for v in low:
        run = run + 1 if v else 0
        longest = max(longest, run)
    reasons = []
    if np.any(~np.isfinite(ess)):
        reasons.append("nonfinite ESS")
    if ess.size and np.nanmin(ess) < 0.10:
        reasons.append(f"ESS/P min {np.nanmin(ess):.3f} < 0.10")
    if longest >= 10:
        reasons.append(f"pCN acceptance < 0.05 for {longest} consecutive rounds")
    if dll.size and np.any(np.isfinite(dll)) and np.nanmax(dll) > 1e-3:
        reasons.append(f"low-rank max|dll| {np.nanmax(dll):.2e} > 1e-3")
    return dict(excluded=bool(reasons), reasons=reasons)
