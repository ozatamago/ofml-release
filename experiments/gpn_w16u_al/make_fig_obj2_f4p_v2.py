"""Figure 4, version 2 (2026-11-18): the colours of arXiv_TMLR/tools/figs/ofml_palette.py, shared with figure 3.
IVR and MaxVar take the colour of the surrogate (solid / dashed); space-filling and random, which choose points
without the surrogate, are black and light grey.  In (b) OFML-GPN is blue and OFML-Ens teal.
Version 1 (make_fig_obj2_f4p.py) is unchanged; it drew the figure now in the paper.

Figure 4 of the arXiv (TMLR) version (section 6.3), redrawn in the style of ChenLiu-1996/figures4papers (2026-11-17).

Same data and same quantities as make_fig_obj2_combined.py; only the layout and the encoding change.
  top row     (a) NRMSE learning curves, one panel per surrogate (OFML-GPN, OFML-Ens), 4 lines per panel
                  (blue = acquisition functions that use the predictive uncertainty, red / grey = the others),
                  band = mean +/- standard error over the 16 true functions; top right cell = legends.
  bottom row  (b) IVR runs: max posterior variance, max absolute error, 95% coverage;
                  blue = OFML-GPN, red = OFML-Ens; solid = 8 protocols with an initial point, dashed = 8 without.
The figure is drawn at twice the printed size with the figures4papers sizes (text 16 pt, lines 2.5, axes 2);
included at \\textwidth (6.5 in) it prints with 8 pt text.

usage: make_fig_obj2_f4p.py runs/2026-11-04_main --ens-dir <dir with rep_XXXX_<policy>.json/.npz> --ens-label <text>
                            [--lang ja|en] [--out STEM]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_w16u as A  # noqa: E402
import w16u_common as C  # noqa: E402
import plot_w16u_uniform_split as U  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import ofml_paths as _OP  # noqa: E402
FONT_DIR = str(_OP.JA_FONT_DIR)
import ofml_palette as PAL  # noqa: E402
GREY = PAL.GREY
SUR_COL = {"gpn": PAL.SURROGATE["OFML-GPN"], "ens": PAL.SURROGATE["OFML-Ens"]}


def pol_style(m, p):
    if p in PAL.NO_SURROGATE_ACQ:
        return PAL.NO_SURROGATE_ACQ[p]
    return SUR_COL[m], PAL.UNCERTAINTY_ACQ_LS[p]
TXT = {
    "ja": dict(
        x="観測の数（初期点の 8 個を含む）", y_a="NRMSE（対数目盛）",
        a_gpn="OFML-GPN\nハイパーパラメータは真の値", a_ens="OFML-Ens\n{lab}",
        b=("事後分散の最大値", "事後平均の誤差の最大値", "95% 被覆率"),
        row_a="(a) 獲得関数の比較", row_b="(b) IVR で観測を選んだときの、事後分散の最大値、事後平均の誤差の最大値、95% 被覆率",
        pol={"ivr": "IVR（予測の不確実性を使う）", "maxvar": "MaxVar（予測の不確実性を使う）",
             "spacefill": "空間充填", "random": "ランダム"},
        leg_a="(a) の線：観測を選んだ獲得関数\nIVR と MaxVar は予測の不確実性を使う", leg_b="(b) の線：代理モデルとプロトコル",
        sur={"gpn": "OFML-GPN", "ens": "OFML-Ens"},
        with_="初期点をもつ 8 個のプロトコル", without="初期点をもたない 8 個のプロトコル", nominal="名目の値 0.95（被覆率）"),
    "en": dict(
        x="number of observations (incl. the 8 initial points)", y_a="NRMSE (log scale)",
        a_gpn="OFML-GPN\ntrue hyperparameters", a_ens="OFML-Ens\n{lab}",
        b=("max posterior variance", "max error of the posterior mean", "95% coverage"),
        row_a="(a) comparison of the acquisition functions", row_b="(b) IVR runs: max posterior variance, max error of the posterior mean, and 95% coverage",
        pol={"ivr": "IVR (uses uncertainty)", "maxvar": "MaxVar (uses uncertainty)",
             "spacefill": "Space-filling", "random": "Random"},
        leg_a="lines in (a): acquisition function\nIVR and MaxVar use the uncertainty", leg_b="lines in (b): surrogate and protocols",
        sur={"gpn": "OFML-GPN", "ens": "OFML-Ens"},
        with_="8 protocols with an initial point", without="8 protocols without one", nominal="nominal 0.95 (coverage)"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ens-dir", type=Path, required=True)
    ap.add_argument("--ens-label", required=True)
    ap.add_argument("--lang", default="ja", choices=["en", "ja"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    L = TXT[a.lang]
    ens_dir = a.ens_dir.resolve()

    A.MODEL_DIRS["ens"] = str(ens_dir)          # run / <absolute path> = the absolute path
    data, reps, excl = A.load(a.run, ["gpn", "ens"])
    assert len(reps) == 16 and not excl, (reps, excl)
    truths = {r: C.load_teacher(r)[0] for r in range(16)}
    bdata = {m: [U.per_rep(a.run, str(d), r, truths[r]) for r in range(16)]
             for m, d in (("gpn", "gpn"), ("ens", ens_dir))}
    n = np.load(a.run / "gpn" / "rep_0000_ivr.npz")["n_obs"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

    fs = 16
    if a.lang == "ja":
        fpath = os.path.join(FONT_DIR, "HaranoAjiGothic-Medium-glyf.ttf")
        font_manager.fontManager.addfont(fpath)
        family = [font_manager.FontProperties(fname=fpath).get_name()]
    else:
        family = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams.update({
        "font.family": family, "font.size": fs, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 2.0, "xtick.major.width": 2.0, "ytick.major.width": 2.0, "xtick.major.size": 5,
        "ytick.major.size": 5, "legend.frameon": False, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
        "axes.titlesize": fs, "axes.titleweight": "normal", "axes.titlelocation": "left",
    })

    W, H = 13.0, 11.6
    fig = plt.figure(figsize=(W, H))
    pw, ph = 0.245, 0.30
    xs = (0.075, 0.395, 0.715)
    ph = 0.27
    axg = fig.add_axes([xs[0], 0.57, pw, ph]); axe = fig.add_axes([xs[1], 0.57, pw, ph])
    axl = fig.add_axes([0.655, 0.47, 0.345, 0.47])
    axb = [fig.add_axes([x, 0.10, pw, ph]) for x in xs]
    fig.text(0.012, 0.975, L["row_a"], ha="left", va="top", fontsize=17)
    fig.text(0.012, 0.455, L["row_b"], ha="left", va="top", fontsize=17)

    # (a): one panel per surrogate, same y range
    lo, hi = np.inf, 0.0
    for ax, m in ((axg, "gpn"), (axe, "ens")):
        for p in C.POLICIES:
            c = A.curve(data, m, reps, p)
            mu, s = c.mean(0), A.se(c)
            col, ls = pol_style(m, p)
            ax.fill_between(A.N_GRID, mu - s, mu + s, color=col, alpha=0.15, lw=0)
            ax.plot(A.N_GRID, mu, color=col, ls=ls, lw=2.5)
            lo, hi = min(lo, (mu - s).min()), max(hi, (mu + s).max())
    for ax, title in ((axg, L["a_gpn"]), (axe, L["a_ens"].format(lab=a.ens_label))):
        ax.set_yscale("log")
        ax.set_ylim(lo * 0.92, hi * 1.08)
        ax.set_title(title, pad=10, linespacing=1.3)
    axg.set_ylabel(L["y_a"])
    ticks = [v for v in (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5) if lo * 0.92 <= v <= hi * 1.08]
    for ax in (axg, axe):
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in ticks]))
        ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_formatter(NullFormatter())

    # (b): theorem quantities of the IVR runs
    for m in ("gpn", "ens"):
        for name, ls in (("with", "-"), ("without", (0, (5, 2)))):
            for ax, key in zip(axb, ("max_var", "max_err", "cov95")):
                c = np.stack([d[f"{key}_{name}"] for d in bdata[m]]).mean(0)
                ax.plot(n, c, color=SUR_COL[m], ls=ls, lw=2.5)
    axb[2].axhline(0.95, color=GREY, lw=1.5, ls=":")
    axb[0].set_yscale("log"); axb[1].set_yscale("log"); axb[2].set_ylim(0, 1.02)
    for ax, vals in ((axb[0], [0.01, 0.03, 0.1, 0.3, 1.0]), (axb[1], [0.3, 0.5, 1.0, 2.0, 3.0])):
        y0, y1 = ax.get_ylim()
        v = [t for t in vals if y0 <= t <= y1]
        ax.yaxis.set_major_locator(FixedLocator(v)); ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in v]))
        ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_formatter(NullFormatter())
    for ax, title in zip(axb, L["b"]):
        ax.set_title(title, pad=10)
    for ax in [axg, axe] + axb:
        ax.set_xlim(n[0], n[-1]); ax.set_xticks([8, 25, 50, 75, 100])

    # legends in the top right cell
    axl.set_axis_off()
    sep = " の " if a.lang == "ja" else " "
    pname = {"ivr": "IVR", "maxvar": "MaxVar"}
    h_a, l_a = [], []
    for m in ("gpn", "ens"):
        for p in ("ivr", "maxvar"):
            c_, s_ = pol_style(m, p)
            h_a.append(Line2D([], [], color=c_, ls=s_, lw=2.5))
            l_a.append(f"{L['sur'][m]}{sep}{pname[p]}" if a.lang == "ja" else f"{pname[p]} ({L['sur'][m]})")
    for p in ("spacefill", "random"):
        c_, s_ = pol_style("gpn", p)
        h_a.append(Line2D([], [], color=c_, ls=s_, lw=2.5))
        l_a.append(L["pol"][p])
    la = axl.legend(h_a, l_a, title=L["leg_a"], loc="upper left",
                    bbox_to_anchor=(-0.02, 1.0), alignment="left", handlelength=2.0, handletextpad=0.6, labelspacing=0.45)
    axl.add_artist(la)
    h_b = [Line2D([], [], color=SUR_COL[m], lw=2.5) for m in ("gpn", "ens")]
    l_b = [L["sur"][m] for m in ("gpn", "ens")]
    h_b += [Line2D([], [], color="black", ls="-", lw=2.5), Line2D([], [], color="black", ls=(0, (5, 2)), lw=2.5),
            Line2D([], [], color=GREY, ls=":", lw=1.5)]
    l_b += [L["with_"], L["without"], L["nominal"]]
    axl.legend(h_b, l_b, title=L["leg_b"], loc="lower left", bbox_to_anchor=(-0.02, -0.02), alignment="left",
               handlelength=2.0, handletextpad=0.6, labelspacing=0.45)
    # one x label per row, centred under the panels of the row
    fig.text((xs[0] + xs[1] + pw) / 2, 0.505, L["x"], ha="center", va="top")
    fig.text((xs[0] + xs[2] + pw) / 2, 0.04, L["x"], ha="center", va="top")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300)
    print("wrote", a.out)
    for m in ("gpn", "ens"):
        print(m, {p: round(float(A.curve(data, m, reps, p).mean(1).mean()), 4) for p in C.POLICIES})


if __name__ == "__main__":
    main()
