"""第 6.2 節のベンチマークの 17 個のプロトコル（ソース 10、ターゲット 7）を横一列に描く。
各プロトコルは木として描き、葉（1 層目の節点）を上、終端の観測器 R を下に置く。節点の高さは葉からの距離。
出力：arXiv_TMLR/figures/fig_obj1_protocols.pdf（と確認用の png）。`--lang ja` で日本語版 fig_obj1_protocols_ja.pdf。"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

LANG = 'ja' if '--lang' in sys.argv and sys.argv[sys.argv.index('--lang') + 1] == 'ja' else 'en'
# Japanese labels in Harano Aji Gothic, as in the Japanese Figure 1; node letters keep the default font
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'scripts'))
import ofml_paths as _OP  # noqa: E402
HARANO = str(_OP.JA_FONT_DIR)
JA = {'serial, depth 3': '直列・深さ 3', 'merge, depth 3': '合流・深さ 3', 'serial, depth 4': '直列・深さ 4',
      'merge, depth 4': '合流・深さ 4', 'Source protocols': 'ソースプロトコル', 'Target protocols': 'ターゲットプロトコル'}


def label_font(bold):
    if LANG != 'ja':
        return {'fontweight': 'bold'} if bold else {}
    # the paper's figure used the OpenType files; the glyf TrueType files of scripts/otf2ttf_haranoaji.py also work
    w = 'Bold' if bold else 'Medium'
    f = os.path.join(HARANO, f'HaranoAjiGothic-{w}.otf')
    if not os.path.exists(f):
        f = os.path.join(HARANO, f'HaranoAjiGothic-{w}-glyf.ttf')
    return {'fontproperties': font_manager.FontProperties(fname=f)}


def tr(s):
    return JA.get(s, s) if LANG == 'ja' else s
from matplotlib.patches import Circle, FancyArrowPatch

# protocol trees as nested tuples: (type, child, child, ...); leaves are 1-tuples
A, B = ('A',), ('B',)
def T(t, *c): return (t,) + c
SOURCES = [
    ('serial, depth 3', [T('O', T('C', A)), T('O', T('D', A)), T('O', T('C', B)), T('O', T('D', B))]),
    ('merge, depth 3', [T('O', T('M', A, B)), T('O', T('M', B, A))]),
    ('serial, depth 4', [T('O', T('D', T('C', A))), T('O', T('C', T('D', A))), T('O', T('C', T('C', B))),
                         T('O', T('D', T('D', B)))]),
]
TARGETS = [
    ('serial, depth 4', [T('O', T('C', T('C', A))), T('O', T('D', T('D', A))), T('O', T('D', T('C', B))),
                         T('O', T('C', T('D', B)))]),
    ('merge, depth 4', [T('O', T('M', T('M', A, B), A)), T('O', T('M', T('M', B, A), B)), T('O', T('M', A, T('M', A, B)))]),
]
# 2026-11-11: serial types C1, C2 renamed C, D; terminal type R renamed O (observation) (the author's choice).
# Okabe-Ito based colours per type (fill), except C (formerly C1) in yellow-green (#8BC34A, 2026-11-08); text stays black
TEXT_ON: dict = {}   # all letters black
# palette chosen 2026-11-08 for colour-vision deficiency (same as Figure 1): pairwise OKLab distance x100 >= 8 under
# protan/deutan/tritan simulation and >= 16 with normal vision
COL = {'A': '#F0E442', 'B': '#56B4E9', 'C': '#8BC34A', 'D': '#D55E00', 'M': '#CC79A7', 'O': '#CCCCCC'}


def height(n):
    return 0 if len(n) == 1 else 1 + max(height(c) for c in n[1:])


def leaves(n):
    return 1 if len(n) == 1 else sum(leaves(c) for c in n[1:])


def layout(n, x0, pos, edges, parent=None):
    """Place leaves left to right from x0 (unit spacing); an inner node sits above the mean of its children."""
    if len(n) == 1:
        x = x0
        nxt = x0 + 1
    else:
        xs, nxt = [], x0
        for c in n[1:]:
            cx, nxt = layout(c, nxt, pos, edges, None)
            xs.append(cx)
        x = sum(xs) / len(xs)
    key = len(pos)
    pos.append((x, height(n), n[0]))
    if len(n) > 1:
        # children were appended just before; find them by walking back
        pass
    return x, nxt


def draw_tree(ax, n, x0, r):
    """Draw tree n with its leftmost leaf at x0; return (right edge, centre x)."""
    nodes = []

    def rec(m, x_start):
        if len(m) == 1:
            nodes.append((x_start, 0, m[0], []))  # leaves at y = 0 (top)
            return x_start, x_start + 1, len(nodes) - 1
        child_ids, xs, nxt = [], [], x_start
        for c in m[1:]:
            cx, nxt, cid = rec(c, nxt)
            xs.append(cx)
            child_ids.append(cid)
        x = sum(xs) / len(xs)
        nodes.append((x, -height(m), m[0], child_ids))   # parents below their children
        return x, nxt, len(nodes) - 1

    rec(n, x0)
    for (x, y, t, ch) in nodes:
        for ci, cid in enumerate(ch):
            cx, cy = nodes[cid][0], nodes[cid][1]
            if len(ch) == 2:
                # 2026-11-18 (the author): the two inputs of a merge node are not joined into one arrow; the first
                # (left) parent enters at 1/4 and the second (right) parent at 3/4 of the node's width
                tx = x - r / 2 if ci == 0 else x + r / 2
                ty = y + r * 0.8660254
                ym = y + 0.5
                if abs(cx - tx) < 1e-9:
                    ax.add_patch(FancyArrowPatch((cx, cy - r), (tx, ty), arrowstyle='-|>', mutation_scale=4.5,
                                                 lw=0.6, color='#444444', shrinkA=0, shrinkB=0))
                else:
                    ax.plot([cx, cx, tx], [cy - r, ym, ym], color='#444444', lw=0.6, solid_capstyle='butt',
                            solid_joinstyle='miter')
                    ax.add_patch(FancyArrowPatch((tx, ym), (tx, ty), arrowstyle='-|>', mutation_scale=4.5,
                                                 lw=0.6, color='#444444', shrinkA=0, shrinkB=0))
                continue
            if abs(cx - x) < 1e-9:   # straight down
                ax.add_patch(FancyArrowPatch((cx, cy - r), (x, y + r), arrowstyle='-|>', mutation_scale=4.5,
                                             lw=0.6, color='#444444', shrinkA=0, shrinkB=0))
            else:                     # orthogonal elbow: down, across just above the parent, down into it
                ym = y + 0.5          # halfway between the parent and the level above it
                ax.plot([cx, cx, x], [cy - r, ym, ym], color='#444444', lw=0.6, solid_capstyle='butt',
                        solid_joinstyle='miter')
                ax.add_patch(FancyArrowPatch((x, ym), (x, y + r), arrowstyle='-|>', mutation_scale=4.5,
                                             lw=0.6, color='#444444', shrinkA=0, shrinkB=0))
    for (x, y, t, ch) in nodes:
        ax.add_patch(Circle((x, y), r, facecolor=COL[t], edgecolor='#333333', lw=0.5, zorder=3))
        ax.text(x, y, t, ha='center', va='center', fontsize=5.2, zorder=4,
                color=TEXT_ON.get(t, 'black'))
    return x0 + leaves(n), sum(nd[0] for nd in nodes if nd[1] == height(n)) / 1.0


def main():
    gap_p, gap_g, gap_st = 0.45, 0.9, 1.6       # between protocols, groups, sources/targets
    r = 0.34
    fig, ax = plt.subplots(figsize=(6.5, 1.42))
    x = 0.0
    spans = []                                    # (label, x_left, x_right, side)
    for side, groups in (('Source protocols', SOURCES), ('Target protocols', TARGETS)):
        side_left = x
        for gi, (lab, trees) in enumerate(groups):
            gl = x
            for ti, t in enumerate(trees):
                right, _ = draw_tree(ax, t, x, r)
                x = right - 1 + 1 + (gap_p if ti < len(trees) - 1 else 0)
            spans.append((lab, gl - 0.5, x - 0.5, 'group'))
            x += gap_g if gi < len(groups) - 1 else 0
        spans.append((side, side_left - 0.5, x - 0.5, 'side'))
        if side.startswith('Source'):
            sep = x + gap_st / 2 - 0.5
            x += gap_st
    ax.axvline(sep, ymin=0.02, ymax=0.98, color='#888888', lw=0.8, ls=(0, (3, 2)))
    for lab, xl, xr, kind in spans:
        y = -3.85 if kind == 'group' else 0.68
        if kind == 'group':
            ax.plot([xl, xl, xr, xr], [y + 0.18, y, y, y + 0.18], color='#666666', lw=0.6)
            ax.text((xl + xr) / 2, y - 0.25, tr(lab), ha='center', va='top', fontsize=6.5, **label_font(False))
        else:
            ax.text((xl + xr) / 2, y, tr(lab), ha='center', va='bottom', fontsize=7.5, **label_font(True))
    ax.set_xlim(-0.8, x - 0.2)
    ax.set_ylim(-4.6, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    base = 'fig_obj1_protocols' + ('_ja' if LANG == 'ja' else '')
    out = str(_OP.OUT_FIGURES / f'{base}.pdf')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)   # 2026-11-11: no extra white margin above the caption
    fig.savefig(str(_OP.OUT_FIGURES / f'{base}.png'), dpi=300, bbox_inches='tight', pad_inches=0.03)
    print('wrote', out, 'x extent', x)


if __name__ == '__main__':
    main()
