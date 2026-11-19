"""PB-W16U, IVR runs: uniform-convergence check split by whether a protocol has an initial point (2026-11-08).

Colour = surrogate (OFML-GPN green, OFML-Ens blue); line style = protocol set
(solid: the 8 protocols with an initial point, dashed: the 8 without).  Panels: max posterior variance,
max absolute error, 95% coverage, each over the 512 grid cells of the set; mean over the 16 reps.
The per-set maxima are recomputed from the stored per-round mean / var / pit and the cached teachers; the
"without" set and the union are checked against the stored round metrics.

usage: plot_w16u_uniform_split.py runs/2026-11-04_main [--ens-dir ens_v2] [--outdir DIR]
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
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
import fig_style as S  # noqa: E402

COL = {"gpn": "#2ca02c", "ens": "#3B75AE"}
LAB = {"gpn": "OFML-GPN", "ens": "OFML-Ens"}
G = C.G
INIT = np.array([p * G + k for p in C.INITIAL_PROTOCOLS for k in range(G)])
NOINIT = C.ZERO_SHOT_FLAT


def per_rep(run: Path, d: str, rep: int, truth: np.ndarray) -> dict:
    z = np.load(run / d / f"rep_{rep:04d}_ivr.npz")
    js = json.loads((run / d / f"rep_{rep:04d}_ivr.json").read_text())["rounds"]
    mean, var, pit = z["mean"], z["var"], z["pit"]
    err = np.abs(mean - truth[None, :])
    out = {}
    for name, idx in (("with", INIT), ("without", NOINIT)):
        out[f"max_var_{name}"] = var[:, idx].max(1)
        out[f"max_err_{name}"] = err[:, idx].max(1)
        out[f"cov95_{name}"] = (np.abs(pit[:, idx] - 0.5) <= 0.475).mean(1)
    # checks against the stored round metrics
    assert np.allclose(np.maximum(out["max_var_with"], out["max_var_without"]), js["max_var"], rtol=1e-5, atol=1e-9)
    assert np.allclose(out["max_var_without"], js["max_var_zeroshot"], rtol=1e-5, atol=1e-9)
    assert np.allclose(np.maximum(out["max_err_with"], out["max_err_without"]), js["max_abs_err"], rtol=1e-5, atol=1e-7)
    assert np.allclose(out["cov95_without"], np.array(js["coverage_zeroshot"])[:, 2], atol=1e-9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ens-dir", default="ens_v2")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--lang", default="en", choices=["en", "ja"], help="ja: Japanese titles, axis label and legend")
    a = ap.parse_args()
    outdir = a.outdir or (a.run / "analysis_uniform_split")
    outdir.mkdir(parents=True, exist_ok=True)
    reps = list(range(16))
    truths = {r: C.load_teacher(r)[0] for r in reps}
    data = {m: [per_rep(a.run, d, r, truths[r]) for r in reps] for m, d in (("gpn", "gpn"), ("ens", a.ens_dir))}
    n = np.load(a.run / "gpn" / "rep_0000_ivr.npz")["n_obs"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    S.apply()
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.6))
    rows = []
    for m in ("gpn", "ens"):
        for name, ls in (("with", "-"), ("without", "--")):
            for ax, key in zip(axs, ("max_var", "max_err", "cov95")):
                c = np.stack([d[f"{key}_{name}"] for d in data[m]]).mean(0)
                ax.plot(n, c, color=COL[m], ls=ls, lw=1.6)
                rows.append((m, name, key, c[0], c[-1]))
    axs[2].axhline(0.95, color="#444444", lw=0.8, ls=":")
    axs[0].set_yscale("log"); axs[1].set_yscale("log")
    J = a.lang == "ja"
    fkw = {}
    if J:
        import os
        from matplotlib import font_manager
        fkw = {"fontproperties": font_manager.FontProperties(fname=str(__import__("ofml_paths").JA_FONT), size=10)}
    titles = (('事後分散の最大値', '事後平均の誤差の最大値', '95% 被覆率') if J else
              ("max posterior variance", "max absolute error", "95% coverage"))
    for ax, tt in zip(axs, titles):
        ax.set_title(tt, fontsize=10, fontweight="normal", **fkw)
    axs[2].set_ylim(0, 1.02)
    for ax in axs:
        ax.set_xlim(n[0], n[-1])
    if J:
        axs[1].set_xlabel("観測の数（初期点を含む）", fontproperties=fkw["fontproperties"].copy())
        axs[1].xaxis.label.set_fontsize(plt.rcParams["axes.labelsize"])
    else:
        axs[1].set_xlabel("number of observations")
    h = [Line2D([], [], color=COL["gpn"], lw=1.6), Line2D([], [], color=COL["ens"], lw=1.6),
         Line2D([], [], color="#444444", ls="-", lw=1.6), Line2D([], [], color="#444444", ls="--", lw=1.6)]
    lab = ((["OFML-GPN (IVR)", "OFML-Ens (IVR)", "初期点をもつ 8 個のプロトコル", "初期点をもたない 8 個のプロトコル"]) if J else
           ["OFML-GPN (IVR)", "OFML-Ens (IVR)", "8 protocols with an initial point", "8 protocols without"])
    lkw = {"prop": font_manager.FontProperties(fname=fkw["fontproperties"].get_file(), size=plt.rcParams["legend.fontsize"])} if J else {}
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02), **lkw)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_w16u_uniform_split{'_ja' if J else ''}.{ext}", dpi=200)
    print("model set metric n=8 n=100")
    for r in rows:
        print(*r[:3], f"{r[3]:.3f}", f"{r[4]:.3f}")


if __name__ == "__main__":
    main()
