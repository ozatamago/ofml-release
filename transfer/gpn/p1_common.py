"""Purpose 1 (c), GP-matched benchmark: data and metrics shared by EVERY method (GP and neural).

Plan: REPORTS/EXPERIMENT_DESIGN_2026-11-04.md section 1.8 (agreed with the author on 2026-11-05).
numpy only at import time.  Only ``make_problem`` needs ofml_gpn (imported lazily; it pulls in jax).  Envs
without jax (the neural baselines) read the problems written by ``python p1_common.py dump OUT_DIR`` with
``load_problem``.

* Library: ofml_gpn.term_library.purpose1_library() -- 10 sources then 7 targets (protocol-major order).
* Grid: linspace(0, 4, 64).  Kernels: amplitude 0.70, lengthscale 1.0 (A, B) / 0.5 (C1, C2, M, R).
* Functions: rep r in 0..11 -> teacher seed MAIN_BASE + 1000 r; tuning functions u in 0..3 -> TUNE_BASE + 1000 u.
* Source data: 8 grid points per source (SOURCE_GRID), one noisy observation each (80 rows).
* Target noise: a table (target, grid point, visit) so that the k-th observation of the same cell has the same
  noise value for every method (observations may repeat).
* Adaptation: T = 200 IVR steps over the 448 target cells (observed cells stay candidates).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("OFML_GPN_ALLOW_ANY_GRID", "1")

G = 64
WX = np.linspace(0.0, 4.0, G)
NOISE_SD = 0.10
JIT = 1e-8
T_ADAPT = 200
SOURCE_GRID = tuple(int(i) for i in np.round(np.linspace(4, 59, 8)))  # 4, 12, 20, 28, 35, 43, 51, 59
MAIN_BASE = 47_100_000   # production functions (rep 0..11)
TUNE_BASE = 47_200_000   # neural hyper-parameter search functions (0..3), disjoint from production
N_REPS = 12
N_TUNE = 4

# protocol order of ofml_gpn.term_library.purpose1_library() (checked in make_problem)
PROTOCOL_TERMS = ("C1(A)", "C2(A)", "C1(B)", "C2(B)", "M(A,B)", "M(B,A)",
                  "C2(C1(A))", "C1(C2(A))", "C1(C1(B))", "C2(C2(B))",
                  "C1(C1(A))", "C2(C2(A))", "C2(C1(B))", "C1(C2(B))",
                  "M(M(A,B),A)", "M(M(B,A),B)", "M(A,M(A,B))")  # R(.) is applied to each
N_PROTOCOLS = len(PROTOCOL_TERMS)
SOURCES = tuple(range(10))           # protocol indices 0..9
TARGETS = tuple(range(10, 17))       # protocol indices 10..16


def library():
    """ofml_gpn TermLibrary of the benchmark (imports jax)."""
    from ofml_gpn.term_library import purpose1_library
    t = purpose1_library()
    assert tuple(t.protocol_terms) == PROTOCOL_TERMS and tuple(t.sources) == SOURCES and tuple(t.targets) == TARGETS
    return t
N_TGT_CELLS = len(TARGETS) * G       # 448


def teacher_seed(rep: int, tune: bool = False) -> int:
    return (TUNE_BASE if tune else MAIN_BASE) + 1000 * int(rep)


@dataclass(frozen=True)
class Problem:
    seed: int
    truth: np.ndarray          # (N_PROTOCOLS * G,) protocol-major, noise-free f_p(x)
    src_cells: np.ndarray      # (80,) flat cell indices p * G + g
    src_y: np.ndarray          # (80,)
    tgt_noise: np.ndarray      # (7, G, T_ADAPT + 1) noise of the k-th visit of (target j, grid g)

    @property
    def truth_tgt(self) -> np.ndarray:
        return self.truth[tgt_flat_cells()]


def make_problem(rep: int, tune: bool = False) -> Problem:
    from ofml_gpn import graph_library as GL
    seed = teacher_seed(rep, tune)
    truth, _ = GL.draw_graph_teacher(library().library, WX, JIT, seed, seed + 1, NOISE_SD)
    rng = np.random.default_rng(seed + 2)
    src_cells = np.array([p * G + g for g in SOURCE_GRID for p in SOURCES], dtype=np.int64)  # point-major
    src_y = truth[src_cells] + NOISE_SD * rng.standard_normal(len(src_cells))
    tgt_noise = NOISE_SD * np.random.default_rng(seed + 3).standard_normal((len(TARGETS), G, T_ADAPT + 1))
    return Problem(seed, np.asarray(truth, float), src_cells, src_y, tgt_noise)


def tgt_flat_cells() -> np.ndarray:
    """Flat cell indices (p * G + g) of the 448 target candidates, in candidate order (target-major)."""
    return np.array([p * G + g for p in TARGETS for g in range(G)], dtype=np.int64)


def candidate_to_cell(c: int) -> int:
    return int(tgt_flat_cells()[c])


def observe(prob: Problem, c: int, visits: np.ndarray) -> float:
    """Noisy observation of candidate c (0..447); ``visits`` (448,) counts earlier visits and is incremented."""
    j, g = divmod(int(c), G)
    k = int(visits[c])
    visits[c] += 1
    return float(prob.truth[TARGETS[j] * G + g] + prob.tgt_noise[j, g, k])


def ivr_scores(cov: np.ndarray, noise_var: float = NOISE_SD ** 2) -> np.ndarray:
    """IVR(c) = sum_{c'} Cov(c', c)^2 / (Var(c) + noise_var) over the 448 target candidates (cov: (448, 448))."""
    return np.sum(cov * cov, axis=0) / (np.diag(cov) + noise_var)


def choose_ivr(cov: np.ndarray) -> int:
    return int(np.argmax(ivr_scores(cov)))


def nrmse(pred_tgt: np.ndarray, prob: Problem) -> float:
    """NRMSE over the 448 target cells: RMSE(prediction) / RMSE(constant mean of the truth)."""
    f = prob.truth_tgt
    return float(np.sqrt(np.mean((pred_tgt - f) ** 2)) / np.sqrt(np.mean((f - f.mean()) ** 2)))


def aulc(curve: np.ndarray) -> float:
    """AULC(T) = (1/T) sum_{t=0}^{T-1} (NRMSE(t) + NRMSE(t+1)) / 2 for curve = NRMSE(0..T)."""
    c = np.asarray(curve, float)
    return float(np.mean(0.5 * (c[:-1] + c[1:])))


def save_problem(prob: Problem, path) -> None:
    np.savez(path, seed=prob.seed, truth=prob.truth, src_cells=prob.src_cells, src_y=prob.src_y,
             tgt_noise=prob.tgt_noise)


def load_problem(path) -> Problem:
    d = np.load(path)
    return Problem(int(d["seed"]), d["truth"], d["src_cells"], d["src_y"], d["tgt_noise"])


def problem_path(root, rep: int, tune: bool = False) -> Path:
    return Path(root) / f"{'tune' if tune else 'rep'}_{int(rep):02d}.npz"


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "dump":
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for tune, n in ((False, N_REPS), (True, N_TUNE)):
        for r in range(n):
            save_problem(make_problem(r, tune), problem_path(out, r, tune))
    print(f"wrote {N_REPS} production + {N_TUNE} tuning problems to {out}")
