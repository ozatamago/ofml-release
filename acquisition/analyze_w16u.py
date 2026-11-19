#!/usr/bin/env python3
"""Analysis + figures for the PB-W16U active-learning experiment (re-runnable as more reps finish).

Reads ``<run>/{gpn,ens}/rep_XXXX_<policy>.json``. A rep enters the paired analysis only when all four policies
are present for every requested model and the GPN runs pass the preset exclusion rule (an excluded rep is
dropped for both models and all policies).

AULC = (1/92) * trapezoid integral of pooled NRMSE over n = 8..100 (the mean height of the learning curve).
Paired differences per surrogate: policy - space-filling and policy - random (negative = policy better),
95% paired percentile bootstrap over reps (10,000 resamples), win count = #reps with a negative difference.

Outputs (in ``--outdir``, default ``<run>/analysis``): ``fig_w16u_nrmse.{pdf,png}`` (learning curves + paired
AULC differences), ``fig_w16u_uniform.{pdf,png}`` (max posterior variance, max |error|, coverage vs n),
``summary.md``, ``table_w16u_al.tex``, ``summary.json``.

    ~/program/ofml/.venv/bin/python analyze_w16u.py runs/<date>_<tag>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import w16u_common as C  # noqa: E402

POL_LABEL = {"ivr": "IVR", "maxvar": "MaxVar", "spacefill": "Space-filling", "random": "Random"}
POL_COLOR = {"ivr": "#009E73", "maxvar": "#0072B2", "spacefill": "#E69F00", "random": "#7f7f7f"}  # Okabe-Ito
MODEL_LABEL = {"gpn": "OFML-GPN", "ens": "OFML-Ens"}
MODEL_LS = {"gpn": "-", "ens": "--"}
N_GRID = np.arange(len(C.INITIAL_FLAT), C.T_TOTAL + 1)


MODEL_DIRS = {"gpn": "gpn", "ens": "ens"}   # model key -> result sub-directory (``--ens-dir`` changes "ens")


def load(run: Path, models: list[str]) -> tuple[dict, list[int], dict]:
    data: dict = {m: {} for m in models}
    for m in models:
        for f in sorted((run / MODEL_DIRS[m]).glob("rep_*_*.json")):
            if f.name.endswith(".fail.json"):
                continue
            r = json.loads(f.read_text())
            if r.get("status") != "ok":
                continue
            data[m].setdefault(r["rep"], {})[r["policy"]] = r
    reps_all = sorted(set.intersection(*[{k for k, v in data[m].items() if set(C.POLICIES) <= set(v)}
                                          for m in models]))
    excluded = {}
    if "gpn" in models:
        for rep in reps_all:
            why = [f"{p}: {', '.join(data['gpn'][rep][p]['exclusion']['reasons'])}" for p in C.POLICIES
                   if data["gpn"][rep][p]["exclusion"]["excluded"]]
            if why:
                excluded[rep] = why
    reps = [r for r in reps_all if r not in excluded]
    return data, reps, excluded


def curve(data, m, reps, p, key="nrmse") -> np.ndarray:
    out = []
    for rep in reps:
        rd = data[m][rep][p]["rounds"]
        assert rd["n_obs"] == list(N_GRID), (m, rep, p)
        out.append(np.asarray(rd[key], dtype=float))
    return np.asarray(out)


def aulc(c: np.ndarray) -> np.ndarray:
    return np.trapezoid(c, N_GRID, axis=-1) / (N_GRID[-1] - N_GRID[0])


def boot_ci(d: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    bm = d[idx].mean(1)
    return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def se(x: np.ndarray, axis=0) -> np.ndarray:
    return np.std(x, axis=axis, ddof=1) / np.sqrt(x.shape[axis]) if x.shape[axis] > 1 else np.zeros_like(np.mean(x, axis))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--models", default="gpn,ens")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--no-bands", action="store_true")
    ap.add_argument("--ens-dir", default="ens",
                    help="sub-directory used as OFML-Ens (ens = v1 config, ens_v2 = widened-search config)")
    ap.add_argument("--ens-label", default="", help="legend label of OFML-Ens (default from --ens-dir)")
    ap.add_argument("--compare-ens", default="", help="comma list of Ens sub-directories to tabulate side by side")
    a = ap.parse_args()
    MODEL_DIRS["ens"] = a.ens_dir
    if a.ens_label:
        MODEL_LABEL["ens"] = a.ens_label
    elif a.ens_dir != "ens":
        MODEL_LABEL["ens"] = f"OFML-Ens ({a.ens_dir.replace('ens_', '')})"
    models = [m for m in a.models.split(",") if (a.run / MODEL_DIRS[m]).exists()]
    outdir = a.outdir or (a.run / ("analysis" if a.ens_dir == "ens" else f"analysis_{a.ens_dir}"))
    if a.compare_ens:
        compare_ens(a.run, a.compare_ens.split(","), outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data, reps, excluded = load(a.run, models)
    print(f"models {models}; complete paired reps: {reps}; excluded: {excluded}")
    if not reps:
        print("nothing to analyse yet")
        return
    summ: dict = dict(reps=reps, excluded=excluded, models=models, per_model={})
    lines = [f"# PB-W16U active learning: analysis ({len(reps)} reps: {reps})", ""]
    if excluded:
        lines += [f"Excluded reps (preset rule): {excluded}", ""]
    tex_rows = []
    for m in models:
        pm: dict = {}
        cur = {p: curve(data, m, reps, p) for p in C.POLICIES}
        au = {p: aulc(cur[p]) for p in C.POLICIES}
        fin = {p: cur[p][:, -1] for p in C.POLICIES}
        lines += [f"## {MODEL_LABEL[m]}", "",
                  "| policy | AULC mean (SE) | final NRMSE mean (SE) | cov95 at n=100 | cov80 | cov50 |",
                  "|---|---|---|---|---|---|"]
        for p in C.POLICIES:
            cov = curve(data, m, reps, p, "coverage")  # (R, 93, 3)
            pm[p] = dict(aulc=au[p].tolist(), final=fin[p].tolist(), aulc_mean=float(au[p].mean()),
                         aulc_se=float(se(au[p])), final_mean=float(fin[p].mean()), final_se=float(se(fin[p])),
                         cov_final=[float(v) for v in cov[:, -1, :].mean(0)])
            lines.append(f"| {POL_LABEL[p]} | {au[p].mean():.4f} ({se(au[p]):.4f}) | {fin[p].mean():.4f} "
                         f"({se(fin[p]):.4f}) | {cov[:, -1, 2].mean():.3f} | {cov[:, -1, 1].mean():.3f} | "
                         f"{cov[:, -1, 0].mean():.3f} |")
        lines += ["", "| comparison | dAULC mean [95% CI] | wins | d final NRMSE mean [95% CI] | wins |",
                  "|---|---|---|---|---|"]
        diffs = {}
        for ref in ("spacefill", "random"):
            for p in C.POLICIES:
                if p == ref or (ref == "random" and p == "spacefill" and False):
                    continue
                da, df = au[p] - au[ref], fin[p] - fin[ref]
                ca, cf = boot_ci(da), boot_ci(df, seed=1)
                diffs[f"{p}-{ref}"] = dict(aulc_mean=float(da.mean()), aulc_ci=ca, aulc_wins=int((da < 0).sum()),
                                           final_mean=float(df.mean()), final_ci=cf, final_wins=int((df < 0).sum()),
                                           n=len(da), aulc_diff=da.tolist())
                lines.append(f"| {POL_LABEL[p]} - {POL_LABEL[ref]} | {da.mean():+.4f} [{ca[0]:+.4f}, {ca[1]:+.4f}] | "
                             f"{(da < 0).sum()}/{len(da)} | {df.mean():+.4f} [{cf[0]:+.4f}, {cf[1]:+.4f}] | "
                             f"{(df < 0).sum()}/{len(df)} |")
                tex_rows.append((m, p, ref, da.mean(), ca, int((da < 0).sum()), df.mean(), cf, int((df < 0).sum()), len(da)))
        pm["diffs"] = diffs
        if m == "gpn":
            ess = min(min(data[m][r][p]["rounds"]["ess"]) for r in reps for p in C.POLICIES)
            pre = min(np.nanmin(data[m][r][p]["rounds"]["ess_pre"]) for r in reps for p in C.POLICIES)
            acc = min(np.nanmin(data[m][r][p]["rounds"]["pcn_acceptance"]) for r in reps for p in C.POLICIES)
            hours = sum(data[m][r][p]["policy_seconds"] for r in reps for p in C.POLICIES) / 3600
            pm["diagnostics"] = dict(min_ess_over_p=ess, min_ess_pre_over_p=float(pre), min_pcn_acc=float(acc),
                                     policy_gpu_hours=hours)
            lines += ["", f"Diagnostics: min ESS/P (round start) {ess:.3f}, min pre-resampling ESS/P {pre:.3f}, "
                          f"min pCN acceptance {acc:.3f}; GPU hours (policies, excl. warm-up) {hours:.2f}."]
        lines.append("")
        summ["per_model"][m] = pm
    (outdir / "summary.md").write_text("\n".join(lines))
    (outdir / "summary.json").write_text(json.dumps(summ, indent=1))
    write_tex(outdir / "table_w16u_al.tex", summ, tex_rows, reps)
    fig_nrmse(data, models, reps, summ, outdir, bands=not a.no_bands)
    fig_uniform(data, models, reps, outdir)
    print("\n".join(lines))
    print(f"wrote {outdir}")


def compare_ens(run: Path, dirs: list[str], outdir: Path) -> None:
    """Side-by-side AULC / final NRMSE / coverage of several Ens result directories (same reps)."""
    outdir.mkdir(parents=True, exist_ok=True)
    per = {}
    for d in dirs:
        MODEL_DIRS["ens"] = d
        data, reps, _ = load(run, ["ens"])
        per[d] = (data, reps)
    reps = sorted(set.intersection(*[set(r) for _, r in per.values()]))
    lines = [f"# OFML-Ens configurations compared ({len(reps)} common reps)", "",
             "| policy | " + " | ".join(f"{d}: AULC / NRMSE(100) / cov95(100)" for d in dirs) + " |",
             "|---|" + "---|" * len(dirs)]
    for p in C.POLICIES:
        cells = []
        for d in dirs:
            data = per[d][0]
            c = curve(data, "ens", reps, p)
            cov = curve(data, "ens", reps, p, "coverage")
            cells.append(f"{aulc(c).mean():.4f} ({se(aulc(c)):.4f}) / {c[:, -1].mean():.4f} / {cov[:, -1, 2].mean():.3f}")
        lines.append(f"| {POL_LABEL[p]} | " + " | ".join(cells) + " |")
    if len(dirs) == 2:
        lines += ["", f"Paired AULC difference {dirs[1]} - {dirs[0]} per policy (95% bootstrap CI, wins = {dirs[1]} better):"]
        for p in C.POLICIES:
            dd = aulc(curve(per[dirs[1]][0], "ens", reps, p)) - aulc(curve(per[dirs[0]][0], "ens", reps, p))
            ci = boot_ci(dd)
            lines.append(f"- {POL_LABEL[p]}: {dd.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}], wins {(dd < 0).sum()}/{len(dd)}")
    (outdir / "compare_ens.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    MODEL_DIRS["ens"] = dirs[-1]


def write_tex(path: Path, summ, rows, reps) -> None:
    out = ["% generated by analyze_w16u.py", r"\begin{tabular}{llcccc}", r"\toprule",
           # paper notation: t counts observations after the 8 initial points, T = 92; no new abbreviations
           r"Surrogate & Policy & $\operatorname{AULC}(T)$ & $\operatorname{NRMSE}(T)$ & "
           r"$\Delta\operatorname{AULC}(T)$ from Space-filling & $\Delta\operatorname{AULC}(T)$ from Random \\",
           r"\midrule"]
    for m, pm in summ["per_model"].items():
        for p in C.POLICIES:
            cells = []
            for ref in ("spacefill", "random"):
                # table sign: policy - baseline (negative = the policy is better); wins = #reps the policy is better
                d = pm["diffs"].get(f"{p}-{ref}")
                cells.append("--" if d is None else
                             f"${d['aulc_mean']:+.3f}$ [${d['aulc_ci'][0]:+.3f}$, ${d['aulc_ci'][1]:+.3f}$] ({d['aulc_wins']}/{d['n']})")
            out.append(f"{MODEL_LABEL[m]} & {POL_LABEL[p]} & ${pm[p]['aulc_mean']:.3f}\\pm{pm[p]['aulc_se']:.3f}$ & "
                       f"${pm[p]['final_mean']:.3f}\\pm{pm[p]['final_se']:.3f}$ & {cells[0]} & {cells[1]} \\\\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out += [r"\end{tabular}", f"% reps: {reps}"]
    path.write_text("\n".join(out) + "\n")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE.parent / "scripts"))
    try:
        import fig_style
        fig_style.apply()
    except Exception:  # noqa: BLE001
        pass
    return plt


def fig_nrmse(data, models, reps, summ, outdir: Path, bands: bool) -> None:
    plt = _style()
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(7.0, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    for m in models:
        for p in C.POLICIES:
            c = np.log(curve(data, m, reps, p))
            mu = np.exp(c.mean(0))
            ax.plot(N_GRID, curve(data, m, reps, p).mean(0), color=POL_COLOR[p], ls=MODEL_LS[m], lw=1.6)
            if bands and len(reps) > 1:
                c2 = curve(data, m, reps, p)
                s = se(c2)
                ax.fill_between(N_GRID, c2.mean(0) - s, c2.mean(0) + s, color=POL_COLOR[p], alpha=0.10, lw=0)
            del mu
    ax.set_yscale("log")
    ax.set_xlabel("number of observations $n$")
    ax.set_ylabel("pooled NRMSE")
    ax.set_xlim(N_GRID[0], N_GRID[-1])
    ax.set_title(f"PB-W16U, mean over {len(reps)} rep{'s' if len(reps) > 1 else ''}", fontsize=10)
    # paired AULC differences
    bx = fig.add_subplot(gs[0, 1])
    ylab, y = [], 0
    marker = {"gpn": "o", "ens": "s"}
    for ref in ("spacefill", "random"):
        for p in C.POLICIES:
            if p == ref:
                continue
            for k, m in enumerate(models):
                d = summ["per_model"][m]["diffs"].get(f"{p}-{ref}")
                if d is None:
                    continue
                yy = y + (0.18 if k == 0 else -0.18)
                bx.errorbar(d["aulc_mean"], yy, xerr=[[d["aulc_mean"] - d["aulc_ci"][0]], [d["aulc_ci"][1] - d["aulc_mean"]]],
                            fmt=marker[m], color=POL_COLOR[p], mfc=POL_COLOR[p] if m == "gpn" else "white",
                            ms=4.5, capsize=2, lw=1.2)
            short = {"ivr": "IVR", "maxvar": "MaxVar", "spacefill": "SF", "random": "Rand"}
            ylab.append((y, f"{short[p]} − {short[ref]}"))
            y -= 1
        y -= 0.5
    bx.axvline(0, color="#555", lw=0.8)
    bx.set_yticks([v for v, _ in ylab])
    bx.set_yticklabels([t for _, t in ylab], fontsize=8)
    bx.set_xlabel("paired ΔAULC (95% CI)", fontsize=9)
    bx.grid(axis="y", visible=False)
    # separate legends: colour = acquisition function, line style / marker = surrogate
    h1 = [Line2D([], [], color=POL_COLOR[p], lw=2.2) for p in C.POLICIES]
    h2 = [Line2D([], [], color="k", ls=MODEL_LS[m], lw=1.5, marker=marker[m], ms=4.5,
                 mfc="k" if m == "gpn" else "white") for m in models]
    l1 = fig.legend(h1, [POL_LABEL[p] for p in C.POLICIES], loc="upper center", ncol=4, fontsize=8,
                    title="acquisition (colour)", title_fontsize=8, bbox_to_anchor=(0.36, -0.02), frameon=False)
    fig.add_artist(l1)
    fig.legend(h2, [MODEL_LABEL[m] for m in models], loc="upper center", ncol=2, fontsize=8,
               title="surrogate (line / marker)", title_fontsize=8, bbox_to_anchor=(0.82, -0.02), frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_w16u_nrmse.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_uniform(data, models, reps, outdir: Path) -> None:
    """Uniform-convergence check (aim 3, Theorem 5.2): IVR runs of both surrogates."""
    plt = _style()
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
    p = "ivr"
    for m in models:
        for key, zs, col in (("max_var", False, "#000000"), ("max_var_zeroshot", True, "#999999")):
            c = curve(data, m, reps, p, key)
            axs[0].plot(N_GRID, c.mean(0), color=col, ls=MODEL_LS[m])
        for key, zs, col in (("max_abs_err", False, "#000000"), ("max_abs_err_zeroshot", True, "#999999")):
            c = curve(data, m, reps, p, key)
            axs[1].plot(N_GRID, c.mean(0), color=col, ls=MODEL_LS[m])
        cov = curve(data, m, reps, p, "coverage")   # (R, 93, 3)
        for j, (lev, col) in enumerate(zip(C.COVERAGE_LEVELS, ("#56B4E9", "#0072B2", "#000000"))):
            axs[2].plot(N_GRID, cov[:, :, j].mean(0), color=col, ls=MODEL_LS[m])
    for lev, col in zip(C.COVERAGE_LEVELS, ("#56B4E9", "#0072B2", "#000000")):
        axs[2].axhline(lev, color=col, lw=0.7, ls=":")
    axs[0].set_yscale("log"); axs[1].set_yscale("log")
    axs[0].set_title(r"$\max_x s_n^2(x)$", fontsize=10)
    axs[1].set_title(r"$\max_x |\mu_n(x)-f(x)|$", fontsize=10)
    axs[2].set_title("coverage: 50% (light), 80% (blue), 95% (black)", fontsize=8)
    for ax in axs:
        ax.set_xlabel("$n$")
        ax.set_xlim(N_GRID[0], N_GRID[-1])
    axs[2].set_ylim(0, 1.02)
    h = [Line2D([], [], color="k", ls=MODEL_LS[m]) for m in models] + \
        [Line2D([], [], color="#000000"), Line2D([], [], color="#999999")]
    lab = [f"{MODEL_LABEL[m]}-IVR" for m in models] + ["all 16 protocols", "8 zero-shot protocols"]
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_w16u_uniform.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
