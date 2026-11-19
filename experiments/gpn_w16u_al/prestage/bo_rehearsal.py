#!/usr/bin/env python3
"""Dress rehearsal of the stage-2 optimiser on synthetic objectives (2026-11-13). No network is trained here.

Question: at the measured noise level of J and a budget of 48 trials, which parts of the optimiser matter
(trainability gate, noise handling, recommendation rule, confirmation on fresh functions, batch size)?

Synthetic problem (5 normalised dims: log2 w [5 levels], L [4 levels], d [7 levels], log eta, log(eta K)):
    f(u)      population objective: 0.75 + anisotropic quadratic (2 strong dims, 3 weak, one interaction; sd SIG_F over
              the feasible region) + a rough smooth term (sd 0.01).
    cliff     infeasible ("does not train") when u5 < c0 + c1 u2; the objective there is 1.0 + N(0, 0.02^2).
    g(u)      pilot-set bias (the pilot mean differs from the population mean): smooth, sd BIAS.
    noise     seed-level chaos of one trial: N(0, NOISE^2).
    trial     y = f + g + noise.            confirmation: y' = f + g' + noise', g' sd BIAS_C, noise' sd NOISE_C.
Regret is always measured on f (population), relative to the best feasible point of a dense sample.

Optimisers (48 trials each):
    sobol            48 Sobol points
    old              16 Sobol + 32 EI as in ens_hpo.py (learned white noise, incumbent = best observed, the cliff
                     values enter the GP, q = 8 constant liar)
    new              16 Sobol + 32 EI with known noise, incumbent = best posterior mean at evaluated points,
                     q = 8 with posterior-mean fantasies
    *_gate           the same with a free trainability gate (infeasible proposals are rejected before the trial)
    new_gate_q1      new_gate, sequential
Recommendation rules: best observed ("obs"), best posterior mean ("pm"), and top-k by posterior mean followed by a
confirmation on fresh functions ("conf3", "conf5", "conf8").

usage: bo_rehearsal.py --problems 200 --procs 64 --out rehearsal.json
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import norm, qmc

LEVELS = (5, 4, 7, 0, 0)          # 0 = continuous
N_INIT, N_TOTAL, Q = 16, 48, 8
CLIFF = (0.1, 0.3)             # range of the cliff offset c0 (0.1-0.3: about 1/4 of the box is infeasible)
SCEN = {  # SIG_F, NOISE, BIAS, NOISE_C, BIAS_C
    "base_P8S2": (0.05, 0.0088, 0.0125, 0.0088, 0.0088),
    "P4S1": (0.05, 0.0177, 0.0175, 0.0088, 0.0088),
    "P4S4": (0.05, 0.0088, 0.0175, 0.0088, 0.0088),
    "weak_signal": (0.025, 0.0088, 0.0125, 0.0088, 0.0088),
    "P16S1": (0.05, 0.0088, 0.0088, 0.0088, 0.0088),
    "flat": (0.01, 0.0088, 0.0125, 0.0088, 0.0088),
}


def snap(u):
    u = np.clip(np.atleast_2d(u).astype(float), 0, 1)
    for i, k in enumerate(LEVELS):
        if k:
            u[:, i] = np.round(u[:, i] * (k - 1)) / (k - 1)
    return u


class RFF:
    def __init__(self, rng, sd, ell, D=300):
        self.w = rng.normal(0, 1 / ell, (D, len(LEVELS)))
        self.b = rng.uniform(0, 2 * np.pi, D)
        self.c = sd * np.sqrt(2.0 / D)

    def __call__(self, u):
        return self.c * np.cos(u @ self.w.T + self.b).sum(1)


class Problem:
    def __init__(self, seed, scen):
        self.sig_f, self.noise, self.bias, self.noise_c, self.bias_c = SCEN[scen]
        rng = np.random.default_rng([seed, 11])
        nd = len(LEVELS)
        strong = rng.choice(nd, 2, replace=False)
        self.a = rng.uniform(0, 0.15, nd)
        self.a[strong] = rng.uniform(0.5, 1.5, 2)
        self.o = rng.uniform(-0.1, 1.1, nd)
        j, k = rng.choice(nd, 2, replace=False)
        self.jk, self.rho = (j, k), rng.uniform(-0.6, 0.6)
        self.c0, self.c1 = rng.uniform(*CLIFF), rng.uniform(0.0, 0.3)
        self.rough = RFF(rng, 0.01, 0.25)
        self.g = RFF(rng, self.bias, 0.3)
        self.gc = RFF(rng, self.bias_c, 0.3)
        self.rng = np.random.default_rng([seed, 12])
        dense = snap(np.random.default_rng([seed, 13]).random((40000, nd)))
        dense = dense[self.feasible(dense)]
        self.scale = 1.0
        self.scale = self.sig_f / self._quad(dense).std()
        self.fmin = self.f(dense).min()

    def feasible(self, u):
        u = np.atleast_2d(u)
        return u[:, 4] >= self.c0 + self.c1 * u[:, 1]

    def _quad(self, u):
        z = u - self.o
        j, k = self.jk
        q = (self.a * z ** 2).sum(1) + 2 * self.rho * np.sqrt(self.a[j] * self.a[k]) * z[:, j] * z[:, k]
        return self.scale * q

    def f(self, u):
        u = np.atleast_2d(u)
        return 0.75 + self._quad(u) + self.rough(u)

    def trial(self, u):
        u = np.atleast_2d(u)
        y = self.f(u) + self.g(u) + self.rng.normal(0, self.noise, len(u))
        bad = ~self.feasible(u)
        y[bad] = 1.0 + self.rng.normal(0, 0.02, bad.sum())
        return y

    def confirm(self, u):
        u = np.atleast_2d(u)
        return self.f(u) + self.gc(u) + self.rng.normal(0, self.noise_c, len(u))

    def regret(self, u):
        u = np.atleast_2d(u)
        r = self.f(u) - self.fmin
        r[~self.feasible(u)] = 1.0 - self.fmin
        return r


def fit_gp(X, y, noise_sd=None, optimise=True, kernel=None):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    mu, sd = y.mean(), max(y.std(), 1e-12)
    ys = (y - mu) / sd
    if kernel is None:
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.full(X.shape[1], 0.3),
                                                           length_scale_bounds=(1e-2, 1e1), nu=2.5)
        if noise_sd is None:
            kernel = kernel + WhiteKernel(1e-3, (1e-6, 1e-1))
    alpha = 1e-10 if noise_sd is None else (noise_sd / sd) ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(kernel, alpha=alpha, normalize_y=False,
                                      n_restarts_optimizer=3 if optimise else 0,
                                      optimizer="fmin_l_bfgs_b" if optimise else None, random_state=0).fit(X, ys)
    return gp, mu, sd


def ei(mu, sd, best):
    sd = np.maximum(sd, 1e-12)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def candidates(rng, X, y, n=4000):
    xb = X[np.argsort(y)[:5]]
    local = xb[rng.integers(0, len(xb), n // 2)] + rng.normal(0, 0.08, (n // 2, X.shape[1]))
    cand = snap(np.vstack([rng.random((n, X.shape[1])), local]))
    seen = {tuple(np.round(x, 6)) for x in X}
    return cand[[tuple(np.round(c, 6)) not in seen for c in cand]]


def sobol_points(seed, n, prob, gate):
    eng = qmc.Sobol(len(LEVELS), scramble=True, seed=seed)
    out = []
    while len(out) < n:
        for u in snap(eng.random(16)):
            if (not gate or prob.feasible(u)[0]) and len(out) < n:
                out.append(u)
    return np.array(out)


def run(prob, seed, kind, gate, q):
    """kind in {'sobol', 'old', 'new'} -> X, y of the 48 trials and the final GP used for recommendations."""
    rng = np.random.default_rng([seed, 21])
    kind, nm = (kind.split("@")[0], float(kind.split("@")[1])) if "@" in kind else (kind, 1.0)
    if isinstance(q, tuple):
        return run_async(prob, seed, kind, gate, q[1], nm, rng)
    if kind == "sobol":
        X = sobol_points(seed, N_TOTAL, prob, gate)
        return X, prob.trial(X)
    X = sobol_points(seed, N_INIT, prob, gate)
    y = prob.trial(X)
    while len(X) < N_TOTAL:
        Xf, yf, kern = X.copy(), y.copy(), None
        new = []
        for _ in range(min(q, N_TOTAL - len(X))):
            if kind == "old":
                gp, mu, sd = fit_gp(Xf, np.minimum(yf, 1.5))
                best = (yf.min() - mu) / sd
            else:
                ok = prob.feasible(Xf) if gate else np.ones(len(Xf), bool)
                gp, mu, sd = fit_gp(Xf[ok], yf[ok], noise_sd=(prob.noise * nm if nm else None), optimise=kern is None, kernel=kern)
                kern = gp.kernel_
                best = gp.predict(Xf[ok]).min()
            cand = candidates(rng, Xf, yf)
            if gate:
                cand = cand[prob.feasible(cand)]
            m, s = gp.predict(cand, return_std=True)
            x = cand[int(np.argmax(ei(m, s, best)))]
            new.append(x)
            lie = yf.min() if kind == "old" else float(gp.predict(x[None])[0]) * sd + mu
            Xf, yf = np.vstack([Xf, x]), np.append(yf, lie)
        new = np.array(new)
        X, y = np.vstack([X, new]), np.append(y, prob.trial(new))
    return X, y


def run_async(prob, seed, kind, gate, W, nm, rng):
    """W asynchronous workers: every selection sees all finished trials and W - 1 running ones (fantasised);
    after each selection one running trial (chosen at random: durations differ) finishes."""
    X = sobol_points(seed, N_INIT, prob, gate)
    y = prob.trial(X)
    pend = []
    for _ in range(N_TOTAL - N_INIT):
        P = np.array(pend).reshape(len(pend), X.shape[1])
        Xf = np.vstack([X, P])
        if kind == "old":
            yf = np.append(y, np.full(len(pend), y.min()))
            gp, mu, sd = fit_gp(Xf, np.minimum(yf, 1.5))
            best = (yf.min() - mu) / sd
        else:
            gp, mu, sd = fit_gp(X, y, noise_sd=(prob.noise * nm if nm else None))
            yf = np.append(y, gp.predict(P) * sd + mu) if len(pend) else y.copy()
            if len(pend):
                gp, mu, sd = fit_gp(Xf, yf, noise_sd=(prob.noise * nm if nm else None), optimise=False, kernel=gp.kernel_)
            best = gp.predict(Xf).min()
        cand = candidates(rng, Xf, yf)
        if gate:
            cand = cand[prob.feasible(cand)]
        m, s = gp.predict(cand, return_std=True)
        pend.append(cand[int(np.argmax(ei(m, s, best)))])
        if len(pend) == W:
            x = pend.pop(int(rng.integers(len(pend))))
            X, y = np.vstack([X, x]), np.append(y, prob.trial(x[None]))
    if pend:
        X, y = np.vstack([X, np.array(pend)]), np.append(y, prob.trial(np.array(pend)))
    return X, y


def recommend(prob, X, y, gate, nm=1.0):
    ok = (prob.feasible(X) if gate else (y < 0.97))
    if ok.sum() < 6:
        ok = np.ones(len(X), bool)
    out = {"obs": float(prob.regret(X[np.argmin(y)])[0])}
    gp, mu, sd = fit_gp(X[ok], y[ok], noise_sd=(prob.noise * nm if nm else None))
    pm = gp.predict(X[ok])
    order = np.argsort(pm)
    Xo = X[ok][order]
    out["pm"] = float(prob.regret(Xo[0])[0])
    for k in (3, 5, 8):
        top = Xo[:k]
        out[f"conf{k}"] = float(prob.regret(top[np.argmin(prob.confirm(top))])[0])
    top = X[np.argsort(y)[:5]]
    out["obs_conf5"] = float(prob.regret(top[np.argmin(prob.confirm(top))])[0])
    # leave-one-out predictive check of the final GP (fixed hyper-parameters)
    from sklearn.gaussian_process import GaussianProcessRegressor
    ys = (y[ok] - mu) / sd
    loo = []
    for i in range(ok.sum()):
        m = np.arange(ok.sum()) != i
        g2 = GaussianProcessRegressor(gp.kernel_, alpha=gp.alpha, optimizer=None).fit(X[ok][m], ys[m])
        loo.append(float(g2.predict(X[ok][i][None])[0]))
    out["loo_r"] = float(np.corrcoef(loo, ys)[0, 1])
    out["best_after_init"] = bool(np.argmin(y) >= N_INIT)
    out["n_infeasible"] = int((~prob.feasible(X)).sum())
    return out


METHODS = {"sobol": ("sobol", False, 1), "sobol_gate": ("sobol", True, 1), "old": ("old", False, Q),
           "old_gate": ("old", True, Q), "new": ("new", False, Q), "new_gate": ("new", True, Q),
           "new_gate_q1": ("new", True, 1), "new_gate_q4": ("new", True, 4),
           "old_gate_q16": ("old", True, 16), "new_gate_q16": ("new", True, 16),
           "new_gate_a8": ("new", True, ("a", 8)), "new_gate_a16": ("new", True, ("a", 16)),
           "old_gate_a8": ("old", True, ("a", 8)), "old_gate_a16": ("old", True, ("a", 16)),
           "new_gate_nx05": ("new@0.5", True, Q), "new_gate_nx2": ("new@2", True, Q),
           "new_gate_nx4": ("new@4", True, Q), "new_gate_nx025": ("new@0.25", True, Q),
           "new_gate_nx0125": ("new@0.125", True, Q), "new_gate_est": ("new@0", True, Q),
           "new_gate_est_a16": ("new@0", True, ("a", 16)), "new_gate_nx05_a16": ("new@0.5", True, ("a", 16))}


def one(args):
    seed, scen = args
    res = {}
    for name, (kind, gate, q) in METHODS.items():
        prob = Problem(seed, scen)
        X, y = run(prob, seed, kind, gate, q)
        res[name] = recommend(prob, X, y, gate, float(kind.split('@')[1]) if '@' in kind else 1.0)
    return scen, seed, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--scen", default=",".join(SCEN))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--levels", default="5,4,7,0,0")
    ap.add_argument("--n-init", type=int, default=16)
    ap.add_argument("--n-total", type=int, default=48)
    ap.add_argument("--cliff", default="0.1,0.3")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    global LEVELS, N_INIT, N_TOTAL, CLIFF
    LEVELS = tuple(int(v) for v in a.levels.split(","))
    N_INIT, N_TOTAL = a.n_init, a.n_total
    CLIFF = tuple(float(v) for v in a.cliff.split(","))
    for m in [m for m in METHODS if m not in a.methods.split(",")]:
        del METHODS[m]
    print(f"levels {LEVELS} n_init {N_INIT} n_total {N_TOTAL} cliff {CLIFF} methods {list(METHODS)}", flush=True)
    jobs = [(s, sc) for sc in a.scen.split(",") for s in range(a.problems)]
    with Pool(a.procs) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    out = {}
    for scen, seed, res in rows:
        out.setdefault(scen, {})[seed] = res
    a.out.write_text(json.dumps(out))
    for scen, d in out.items():
        print(f"== {scen} (SIG_F, NOISE, BIAS, NOISE_C, BIAS_C) = {SCEN[scen]}; {len(d)} problems; mean regret x1000 "
              f"(90% in brackets)")
        keys = ("obs", "pm", "conf3", "conf5", "conf8", "obs_conf5")
        print(f"{'method':14s}" + "".join(f"{k:>16s}" for k in keys) + "   loo_r  best>init  n_infeas")
        for m in METHODS:
            v = {k: np.array([d[s][m][k] for s in d]) for k in keys}
            print(f"{m:14s}" + "".join(f"{1e3 * v[k].mean():8.1f} [{1e3 * np.quantile(v[k], 0.9):5.1f}]" for k in keys)
                  + f"   {np.mean([d[s][m]['loo_r'] for s in d]):5.2f}  {np.mean([d[s][m]['best_after_init'] for s in d]):8.2f}"
                  + f"  {np.mean([d[s][m]['n_infeasible'] for s in d]):7.1f}")


if __name__ == "__main__":
    main()
