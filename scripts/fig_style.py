"""Shared plotting style for the OFML paper figures (unified palette, clean look, bottom legend).
Goal: neat, consistent figures like a well-made multi-panel regret plot — light grid, no top/right
spines, one shared bottom legend, the SAME colour for a given series across every figure."""
from __future__ import annotations
import matplotlib.pyplot as plt

# Exact dimensions from aaai2027.sty.  Drawing at these widths prevents LaTeX
# from shrinking text after the figure has been generated.
AAAI_TEXT_WIDTH_IN = 7.0
AAAI_COLUMN_WIDTH_IN = (7.0 - 0.375) / 2.0
AAAI_MIN_FONT_PT = 9.5

# one consistent colour per series, reused across Fig 2/3/4
COLORS = {
    "Independent-Ens": "#7f7f7f", "Shared-Ens": "#ff7f0e", "OFML-Ens": "#1f77b4",
    "OFML-linear": "#2ca02c", "OFML-nonlinear": "#2ca02c",
    "random": "#7f7f7f", "maxvar": "#1f77b4", "ivr": "#2ca02c",
    "floor": "#d62728",
}
CI_ALPHA = 0.16


def apply():
    plt.rcParams.update({
        # AAAI submission requirements: no Type 3 fonts (embed TrueType), and all
        # labels at least 9 pt AT FINAL PRINT SIZE -- figures are drawn at the
        # exact 7.0 in text width or 3.3125 in column width from aaai2027.sty.
        # We use 9.5 pt as a small audit/rounding safety margin.
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": AAAI_MIN_FONT_PT,
        "ytick.labelsize": AAAI_MIN_FONT_PT,
        "legend.fontsize": AAAI_MIN_FONT_PT,
        "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#c8c8c8", "axes.linewidth": 0.9,
        "axes.grid": True, "grid.color": "#e9e9e9", "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold",
        "xtick.color": "#444", "ytick.color": "#444",
        "legend.frameon": False, "lines.linewidth": 1.7,
    })


def bottom_legend(fig, handles, labels, ncol=None, y=0.005):
    fig.legend(handles, labels, loc="lower center", ncol=ncol or len(labels),
               bbox_to_anchor=(0.5, y), frameon=False, handlelength=1.9, columnspacing=1.8,
               borderaxespad=0.0)
