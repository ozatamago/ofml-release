"""第 6.3 節のベンチマークの 16 個のプロトコルを、図 2（plot_protocols.py）と同じ描き方で横一列に描く。
左に初期点をもつ 8 個（2 層目が G1、G3、G5、G7）、右に初期点をもたない 8 個（G2、G4、G6、G8）を置き、破線で分ける。
各側の中は、1 層目が E の 4 個、F の 4 個の順で、その間を少し空ける（群の括弧と名前は付けない）。

2026-11-18：型の名前を本文に合わせた（旧 D、E、F1〜F8 を E、F、G1〜G8 に読み替えた）。
日本語の文字は glyf 版の TTF で埋め込む（.otf だと PDF で日本語が消えることがあるため）。

色（2026-11-08 に色覚に配慮して選び、2026-11-18 に名前だけを替えて引き継いだ。check_palette_cvd.py の OKLab 距離 ×100）：
E #4AD0BA、F #8285EB、G1〜G8 #F4A437（同じ役割の型なので同じ色にし、番号の文字で区別する）、終端 Q #DDDDDD。
4 色のすべての組で、通常の色覚で 16.8 以上、1・2・3 型の色覚のシミュレーションで 7.9 以上。
1 層目と 2 層目の色は、図 2 の型とは別の型であることを示すため、図 2 の型の色と変えた。
終端の型は、図 2 の O と同じく灰色にした。文字はすべて黒。

usage: plot_protocols_w16u.py [--lang ja]
出力：arXiv_TMLR/figures/fig_obj2_protocols{,_ja}.pdf と、このディレクトリの png。
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch

LANG = 'ja' if '--lang' in sys.argv and sys.argv[sys.argv.index('--lang') + 1] == 'ja' else 'en'
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'scripts'))
import ofml_paths as _OP  # noqa: E402
FONTS = str(_OP.JA_FONT_DIR)
COL = {'E': '#4AD0BA', 'F': '#8285EB', 'G': '#F4A437', 'Q': '#DDDDDD'}
TXT = {
    'en': {'with': 'Protocols with an initial point (8)', 'without': 'Protocols without an initial point (8)',
           'E': 'first layer E', 'F': 'first layer F'},
    'ja': {'with': '初期点をもつプロトコル（8 個）', 'without': '初期点をもたないプロトコル（8 個）',
           'E': '1 層目が E', 'F': '1 層目が F'},
}[LANG]


def font(bold):
    if LANG != 'ja':
        return {'fontweight': 'bold'} if bold else {}
    f = os.path.join(FONTS, 'HaranoAjiGothic-Bold-glyf.ttf' if bold else 'HaranoAjiGothic-Medium-glyf.ttf')
    return {'fontproperties': font_manager.FontProperties(fname=f)}


def chain(ax, x, types, r):
    """A depth-3 serial protocol drawn top (first layer) to bottom (terminal) at column x."""
    ys = [0, -1, -2]
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((x, y0 - r), (x, y1 + r), arrowstyle='-|>', mutation_scale=4.5,
                                     lw=0.6, color='#444444', shrinkA=0, shrinkB=0))
    for y, t in zip(ys, types):
        ax.add_patch(Circle((x, y), r, facecolor=COL[t[0]], edgecolor='#333333', lw=0.5, zorder=3))
        ax.text(x, y, t, ha='center', va='center', fontsize=5.2, zorder=4, color='black')


def main():
    r = 0.34
    gap_p, gap_g, gap_s = 0.45, 0.9, 1.6       # between protocols, groups, sides (as in figure 2)
    fig, ax = plt.subplots(figsize=(6.5, 1.0))
    x = 0.0
    spans = []
    for side, gs in (('with', (1, 3, 5, 7)), ('without', (2, 4, 6, 8))):
        sl = x
        for gi, first in enumerate(('E', 'F')):
            gl = x
            for k, g in enumerate(gs):
                chain(ax, x, (first, f'G{g}', 'Q'), r)
                x += 1 + (gap_p if k < len(gs) - 1 else 0)
            spans.append((TXT[first], gl - 0.5, x - 0.5, 'group'))
            x += gap_g if gi == 0 else 0
        spans.append((TXT[side], sl - 0.5, x - 0.5, 'side'))
        if side == 'with':
            sep = x + gap_s / 2 - 0.5
            x += gap_s
    ax.axvline(sep, ymin=0.02, ymax=0.98, color='#888888', lw=0.8, ls=(0, (3, 2)))
    for lab, xl, xr, kind in spans:
        # 2026-11-18: no bracket or label for the first-layer groups (the author: the node letters show them)
        if kind == 'side':
            ax.text((xl + xr) / 2, 0.68, lab, ha='center', va='bottom', fontsize=7.5, **font(True))
    ax.set_xlim(-0.8, x - 0.2)
    ax.set_ylim(-2.55, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    plt.rcParams['pdf.fonttype'] = 42
    base = 'fig_obj2_protocols' + ('_ja' if LANG == 'ja' else '')
    out = str(_OP.OUT_FIGURES / f'{base}.pdf')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)   # same margins as figure 2
    fig.savefig(str(_OP.OUT_FIGURES / f'{base}.png'), dpi=300,
                bbox_inches='tight', pad_inches=0.03)
    print('wrote', out, 'x extent', x)


if __name__ == '__main__':
    main()
