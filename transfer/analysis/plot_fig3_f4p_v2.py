"""Figure 3, version 2 (2026-11-18): colours of arXiv_TMLR/tools/figs/ofml_palette.py (shared with figure 4) and two layouts.
  --layout arxiv   : two rows (GP surrogates / deep-ensemble surrogates), as in version 1
  --layout compact : one row of three panels with all six surrogates (for a page-limited version such as the AAAI
                     camera-ready); GP surrogates solid, ensembles dashed, legend under the panels in three columns

Version 1 text:
Figure 3 of the arXiv (TMLR) version (section 6.2), redrawn in the style of ChenLiu-1996/figures4papers (trial, 2026-11-18).

Same data and same NRMSE(t) as plot_main_fig.py (which is not modified); only the layout and the encoding change.
  row (a): the two GP surrogates (OFML-GPN, Independent-GPN), 2 lines per panel
  row (b): the four deep-ensemble surrogates, 4 lines per panel
  columns: all 7 target protocols, the 4 serial ones, the 3 merge ones; the same log y axis in all panels
  colour = how the surrogate shares and composes modules (blue = OFML: typed modules composed along the DAG,
  red = shared modules composed along the DAG, grey = no composition); band = mean +/- standard error over the reps.
Drawn at twice the printed size (text 16 pt, lines 2.5); included at \\textwidth it prints with 8 pt text.

usage: plot_fig3_f4p.py --reps 2,4,5,9,11,12,15,17,19,23,27,28 --lang ja --out STEM [--band sd|se]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'scripts'))
import ofml_paths as _OP  # noqa: E402
EXP = str(_OP.TRANSFER)
sys.path.insert(0, os.path.join(EXP, "gpn"))
import p1_common as P  # noqa: E402

RUN = os.path.join(EXP, "gpn/runs/p1_main")
NN = os.path.join(EXP, "ensembles/runs/main_2026-11-08")
FONT = str(_OP.JA_FONT)
G = 64
sys.path.insert(0, str(_OP.ROOT / "scripts"))
import ofml_palette as PAL  # noqa: E402
SC = PAL.SURROGATE
# (label, directory, colour, line style, row)
METHODS = [
    ("OFML-GPN", f"{RUN}/ofml_gpn", SC["OFML-GPN"], "-", 0),
    ("Independent-GPN", f"{RUN}/indep_gpn", SC["Independent-GPN"], "-", 0),
    ("OFML-Ens", f"{NN}/ofml", SC["OFML-Ens"], "-", 1),
    ("Shared-DAG-Ens", f"{NN}/sharedcell", SC["Shared-DAG-Ens"], "-", 1),
    ("DAGNN-Cat-Ens", f"{NN}/dagnn_cat", SC["DAGNN-Cat-Ens"], "-", 1),
    ("Shared-Ens", f"{NN}/shared", SC["Shared-Ens"], "-", 1),
]
GROUPS = [("all", np.arange(7 * G)), ("serial", np.arange(4 * G)), ("merge", np.arange(4 * G, 7 * G))]
TXT = {
    "ja": dict(col={"all": "全体（7 個）", "serial": "直列の群（4 個）", "merge": "合流の群（3 個）"},
               row=("(a) GP の代理モデル", "(b) 深層アンサンブルの代理モデル"),
               x="ターゲットで追加した観測の数 $t$", y="NRMSE（対数目盛）",
               desc={"OFML-GPN": "型ごとのモジュール", "Independent-GPN": "共有しない",
                     "OFML-Ens": "型ごとのモジュール", "Shared-DAG-Ens": "全型で共有のモジュール",
                     "DAGNN-Cat-Ens": "全型で共有の GRU", "Shared-Ens": "合成しない"}),
    "en": dict(col={"all": "all (7)", "serial": "serial (4)", "merge": "merge (3)"},
               row=("(a) GP surrogates", "(b) deep-ensemble surrogates"),
               x="acquired target observations $t$", y="NRMSE (log scale)",
               desc={"OFML-GPN": "typed modules", "Independent-GPN": "no sharing",
                     "OFML-Ens": "typed modules", "Shared-DAG-Ens": "one shared module",
                     "DAGNN-Cat-Ens": "one shared GRU", "Shared-Ens": "no composition"}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True)
    ap.add_argument("--lang", default="ja", choices=["ja", "en"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", default="se", choices=["se", "sd"])
    ap.add_argument("--layout", default="arxiv", choices=["arxiv", "compact"])
    a = ap.parse_args()
    L = TXT[a.lang]
    reps = [int(x) for x in a.reps.split(",")]
    t = np.arange(201)
    F = {r: P.load_problem(P.problem_path(RUN + "/problems", r)).truth_tgt for r in reps}

    def nr(pred, f, idx):
        return np.sqrt(np.mean((pred[:, idx] - f[idx]) ** 2, axis=1)) / f[idx].std()

    if a.lang == "ja":
        font_manager.fontManager.addfont(FONT)
        family = [font_manager.FontProperties(fname=FONT).get_name()]
    else:
        family = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams.update({
        "font.family": family, "font.size": 16, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 2.0, "xtick.major.width": 2.0, "ytick.major.width": 2.0, "xtick.major.size": 5,
        "ytick.major.size": 5, "legend.frameon": False, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
        "axes.titlesize": 16, "axes.titleweight": "normal",
    })
    compact = a.layout == "compact"
    W, H = (13.0, 5.4) if compact else (13.0, 8.6)
    fig = plt.figure(figsize=(W, H))
    if compact:
        pw, ph, x0, dx = 0.285, 0.58, 0.075, 0.31
        rows_y = (0.33,)
        row_axes = [fig.add_axes([x0 + k * dx, rows_y[0], pw, ph]) for k in range(3)]
        axes = [row_axes, row_axes]
    else:
        pw, ph, x0, dx = 0.200, 0.28, 0.075, 0.232
        rows_y = (0.60, 0.13)
        axes = [[fig.add_axes([x0 + k * dx, y, pw, ph]) for k in range(3)] for y in rows_y]
    YL = (0.15, 2.0)
    for lab, d, col, ls, row in METHODS:
        means = {r: np.load(f"{d}/rep_{r:02d}.npz")["mean"] for r in reps}
        for k, (g, idx) in enumerate(GROUPS):
            c = np.array([nr(means[r], F[r], idx) for r in reps])
            m = c.mean(0)
            s = c.std(0, ddof=1) / (np.sqrt(len(reps)) if a.band == "se" else 1.0)
            ax = axes[row][k]
            ax.fill_between(t, np.clip(m - s, YL[0], None), m + s, color=col, alpha=0.15, lw=0)
            ax.plot(t, m, color=col, ls=(PAL.DASH if (compact and row == 1) else ls), lw=3.0 if lab == "OFML-GPN" else 2.5)
    ticks = [0.2, 0.3, 0.5, 1.0, 1.5]
    for row in (range(1) if compact else range(2)):
        for k, (g, _) in enumerate(GROUPS):
            ax = axes[row][k]
            ax.set_yscale("log"); ax.set_ylim(*YL); ax.set_xlim(0, 200); ax.set_xticks([0, 100, 200])
            ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in ticks]))
            ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_formatter(NullFormatter())
            if k > 0:
                ax.set_yticklabels([])
            ax.set_title(L["col"][g], pad=8)
        axes[row][0].set_ylabel(L["y"])
        if not compact:
            fig.text(x0 - 0.06, rows_y[row] + ph + 0.072, L["row"][row], ha="left", va="bottom", fontsize=17)
    fig.text(x0 + 1.5 * dx - (dx - pw) / 2, 0.235 if compact else 0.045, L["x"], ha="center", va="top")
    if compact:
        hh = [Line2D([], [], color=col, ls=(PAL.DASH if r == 1 else "-"), lw=3.0 if lab == "OFML-GPN" else 2.5)
              for lab, _, col, ls, r in METHODS]
        ll = [lab for lab, *_ in METHODS]
        order = [0, 2, 1, 3, 4, 5]      # matplotlib fills legends column-wise: columns (OFML-GPN, OFML-Ens), ...
        fig.legend([hh[i] for i in order], [ll[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=3, handlelength=2.4, columnspacing=1.6, fontsize=16)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(f"{a.out}.{ext}", dpi=300)
        print("wrote", a.out)
        return
    # legends: one per row, in the column right of the panels
    for row in range(2):
        hh = [Line2D([], [], color=col, ls=ls, lw=3.0 if lab == "OFML-GPN" else 2.5) for lab, _, col, ls, r in METHODS if r == row]
        ll = [f"{lab}\n{L['desc'][lab]}" for lab, _, _, _, r in METHODS if r == row]
        lax = fig.add_axes([x0 + 3 * dx - 0.005, rows_y[row], 1 - (x0 + 3 * dx), ph]); lax.set_axis_off()
        lax.legend(hh, ll, loc="center left", bbox_to_anchor=(0.0, 0.5), handlelength=2.0, labelspacing=0.9, fontsize=15)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
