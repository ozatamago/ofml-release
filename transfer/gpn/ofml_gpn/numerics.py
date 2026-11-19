"""gpn_smc.numerics -- pure numpy/math helpers moved verbatim, per
K2_DESIGN.md K2a: rbf_covariance, stable_cholesky, normalize_logweights,
systematic_resample, derive_stream_seed, c_schedule, EXTENDED_QUANTILE_LEVELS
from u11_exact_gpn_ucb_initial_design.py; compute_chunk_size from
gpn_deep_backend.py. Bodies are character-identical to their sources except
these import lines and this module docstring; u11's module also imports
matplotlib/PIL at module scope -- these copies drop that dependency
entirely (only the pure functions below are copied, not the module).
"""

from __future__ import annotations

import math

import numpy as np


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

def compute_chunk_size(
    n_particles: int, n_candidates: int, target_bytes: int = 256 * 1024 * 1024
) -> int:
    """Largest power-of-two-derived divisor of ``n_particles`` whose
    per-chunk ``(chunk, n_candidates, n_candidates)`` FP64 tensor is <=
    ``target_bytes``. Same doubling search as
    ``u13_rff_backend.py``'s ``_summary_n_chunks`` loop; if ``n_particles``
    is not evenly divisible past some point the search simply stops there
    (the returned chunk size always exactly divides ``n_particles``).
    """
    bytes_per_particle = 8 * n_candidates * n_candidates
    max_chunk = (
        max(1, target_bytes // bytes_per_particle)
        if bytes_per_particle > 0
        else n_particles
    )
    n_chunks = 1
    chunk_size = n_particles
    while (
        chunk_size > max_chunk
        and n_chunks < n_particles
        and n_particles % (n_chunks * 2) == 0
    ):
        n_chunks *= 2
        chunk_size = n_particles // n_chunks
    return chunk_size

def normalize_logweights(logweights: np.ndarray) -> tuple[np.ndarray, float]:
    shifted = logweights - float(np.max(logweights))
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    ess = float(1.0 / np.sum(weights * weights))
    return weights, ess

def systematic_resample(weights: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions).astype(np.int64)

def derive_stream_seed(
    inference_seed: int, n_initial: int, len_observed: int, purpose: int
) -> int:
    """Collision-resistant internal SMC RNG seed derivation.

    Pools ``(inference_seed, n_initial, len_observed, purpose)`` through
    ``numpy.random.SeedSequence`` instead of combining them with a small
    linear offset (the previous ``inference_seed + K*n_initial +
    len_observed`` scheme). A linear offset can alias: when a caller holds
    ``n_initial`` fixed across many runs and spaces ``inference_seed`` by a
    small constant step (as U13's replicate protocol does), two different
    (run, round) pairs can land on the exact same integer and therefore draw
    the identical resampling permutation / rejuvenation RNG stream, silently
    correlating supposedly-independent replicates. Seed pooling removes that
    failure mode without touching any top-level precommitted seed protocol.
    ``purpose`` distinguishes independent streams that would otherwise share
    the same (inference_seed, n_initial, len_observed) triple (e.g.
    resampling vs. pCN rejuvenation).
    """
    sequence = np.random.SeedSequence(
        [int(inference_seed), int(n_initial), int(len_observed), int(purpose)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])

def c_schedule(n_observations: int) -> float:
    return float(math.sqrt(math.log(n_observations + 2.0)))

EXTENDED_QUANTILE_LEVELS: tuple[float, ...] = (
    0.005,
    0.025,
    0.05,
    0.10,
    0.25,
    0.75,
    0.90,
    0.95,
    0.975,
    0.995,
)
