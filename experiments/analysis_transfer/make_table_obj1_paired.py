"""付録の表（2026-11-11）：第 6.2 節の代理モデルの組ごとの対応のある差（12 本）。
# 2026-11-11: 表示名 SharedCell-DAG-Ens を Shared-DAG-Ens に替えた（結果のフォルダ名 sharedcell はそのまま）。
差は（先に書いた代理モデル）−（後に書いた代理モデル）。95% 区間は rep を重複を許して選び直すブートストラップ 10000 回
（paired_stats.py と同じ乱数の種 20261108、同じ選び直しの添字）。本数は先に書いた代理モデルの値の方が小さかった rep の数。
出力：arXiv_TMLR/generated_exp/table_obj1_paired_{ja,en}.tex"""
import os, sys
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'scripts'))
import ofml_paths as _OP  # noqa: E402
EXP = str(_OP.EXP)
sys.path.insert(0, os.path.join(EXP, 'gpn_graph'))
import p1_common as P
RUN = os.path.join(EXP, 'gpn_graph/runs/p1_main'); NN = os.path.join(EXP, 'p1_neural/runs/main_2026-11-08')
OUT = str(_OP.OUT_TABLES)
VALID = [2, 4, 5, 9, 11, 12, 15, 17, 19, 23, 27, 28]
GR = [('all', np.arange(448)), ('ser', np.arange(256)), ('mer', np.arange(256, 448))]
M = {'OFML-GPN': f'{RUN}/ofml_gpn', 'Independent-GPN': f'{RUN}/indep_gpn', 'OFML-Ens': f'{NN}/ofml',
     'Shared-DAG-Ens': f'{NN}/sharedcell', 'DAGNN-Cat-Ens': f'{NN}/dagnn_cat', 'Shared-Ens': f'{NN}/shared'}
PAIRS = [('OFML-GPN', 'Independent-GPN'), ('OFML-GPN', 'OFML-Ens'), ('Shared-DAG-Ens', 'Shared-Ens'),
         ('OFML-Ens', 'Shared-DAG-Ens'), ('OFML-Ens', 'DAGNN-Cat-Ens')]
KEYS = ['n0', 'nT', 'aulc']
V = {}
for r in VALID:
    f = P.load_problem(P.problem_path(RUN + '/problems', r)).truth_tgt
    for m, d in M.items():
        mean = np.load(f'{d}/rep_{r:02d}.npz')['mean']
        for g, idx in GR:
            c = np.sqrt(np.mean((mean[:, idx] - f[idx]) ** 2, axis=1)) / f[idx].std()
            for k, v in (('n0', c[0]), ('aulc', np.mean((c[:-1] + c[1:]) / 2)), ('nT', c[200])):
                V.setdefault((m, g, k), []).append(v)
rng = np.random.default_rng(20261108)
B = rng.integers(0, len(VALID), size=(10000, len(VALID)))
def cell(a, b, g, k):
    d = np.array(V[a, g, k]) - np.array(V[b, g, k]); bs = d[B].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    return f'${d.mean():+.2f}$ & $[{lo:+.2f},{hi:+.2f}]$ & {int((d < 0).sum())}/{len(d)}'
GN = {'ja': {'all': '全体', 'ser': '直列', 'mer': '合流'}, 'en': {'all': 'All', 'ser': 'Serial', 'mer': 'Merge'}}
HEAD = {'ja': ('組', '群', '差の平均', '95\\% 区間', '本数'), 'en': ('Pair', 'Group', 'Mean', '95\\% interval', 'Count')}
for lang in ('ja', 'en'):
    h = HEAD[lang]; rows = []
    for a, b in PAIRS:
        for i, (g, _) in enumerate(GR):
            first = f'{a} $-$ {b}' if i == 0 else ''
            rows.append(f'{first} & {GN[lang][g]} & ' + ' & '.join(cell(a, b, g, k) for k in KEYS) + ' \\\\')
        rows.append('\\addlinespace')
    rows = rows[:-1]
    tex = ('\\begin{tabular}{@{}ll rrr rrr rrr@{}}\n\\toprule\n'
           ' & & \\multicolumn{3}{c}{$\\NRMSE(0)$} & \\multicolumn{3}{c}{$\\NRMSE(T)$} & \\multicolumn{3}{c}{$\\AULC(T)$}\\\\\n'
           '\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(l){9-11}\n'
           f'{h[0]} & {h[1]} & {h[2]} & {h[3]} & {h[4]} & {h[2]} & {h[3]} & {h[4]} & {h[2]} & {h[3]} & {h[4]}\\\\\n\\midrule\n'
           + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')
    open(os.path.join(OUT, f'table_obj1_paired_{lang}.tex'), 'w').write(tex)
print(open(os.path.join(OUT, 'table_obj1_paired_ja.tex')).read())
