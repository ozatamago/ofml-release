"""目的 1 の表（論文用）：規則に当たらなかった 7 本での NRMSE(0)、AULC(200)、NRMSE(200)。全体・直列・合流。
各列で平均が最小の値を太字にする。日本語版と英語版の tex を arXiv_TMLR/generated_exp/ に書く。"""
import os, sys
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'scripts'))
import ofml_paths as _OP  # noqa: E402
EXP = str(_OP.TRANSFER)
sys.path.insert(0, os.path.join(EXP, 'gpn'))
import p1_common as P
RUN = os.path.join(EXP, 'gpn/runs/p1_main'); NN = os.path.join(EXP, 'ensembles/runs/main_2026-11-08')
OUT = str(_OP.OUT_TABLES)
VALID = [2, 4, 5, 9, 11, 12, 15]
# 2026-11-10: --reps 2,4,... で rep を替え、--suffix _v14 などで別の名前に書く（既定は従来どおり 7 本、table_obj1_{lang}.tex）
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('--reps', default=''); _ap.add_argument('--suffix', default='')
_A = _ap.parse_args()
if _A.reps:
    VALID = [int(x) for x in _A.reps.split(',')]
GR = [('all', np.arange(448)), ('ser', np.arange(256)), ('mer', np.arange(256, 448))]
M = [('OFML-GPN', f'{RUN}/ofml_gpn'), ('Independent-GPN', f'{RUN}/indep_gpn'), ('OFML-Ens', f'{NN}/ofml'),
     ('Shared-DAG-Ens', f'{NN}/sharedcell'), ('DAGNN-Cat-Ens', f'{NN}/dagnn_cat'), ('Shared-Ens', f'{NN}/shared')]
V = {}
for r in VALID:
    f = P.load_problem(P.problem_path(RUN + '/problems', r)).truth_tgt
    for name, d in M:
        mu = np.load(f'{d}/rep_{r:02d}.npz')['mean']
        for g, idx in GR:
            c = np.sqrt(np.mean((mu[:, idx] - f[idx]) ** 2, 1)) / f[idx].std()
            for k, v in (('n0', c[0]), ('aulc', np.mean((c[:-1] + c[1:]) / 2)), ('nT', c[200])):
                V.setdefault((name, g, k), []).append(v)
# 2026-11-11: 列の順を NRMSE(0)、NRMSE(T)、AULC(T) に（著者の指示）
cols = [(g, k) for g, _ in GR for k in ('n0', 'nT', 'aulc')]
mean = {(n, g, k): np.mean(V[n, g, k]) for n, _ in M for g, k in cols}
best = {c: min(mean[n, c[0], c[1]] for n, _ in M) for c in cols}
def row(n):
    cells = []
    for g, k in cols:
        v = mean[n, g, k]; s = f'{v:.3f}'
        cells.append(f'\\textbf{{{s}}}' if abs(v - best[g, k]) < 5e-4 else s)
    return n + ' & ' + ' & '.join(cells) + ' \\\\'
body = '\n'.join(row(n) for n, _ in M)
head = {'ja': ('代理モデル', '全体', '直列', '合流'), 'en': ('Surrogate model', 'All', 'Serial', 'Merge')}
for lang in ('ja', 'en'):
    h = head[lang]
    tex = (f"""\\begin{{tabular}}{{@{{}}l rrr rrr rrr@{{}}}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{{h[1]}}} & \\multicolumn{{3}}{{c}}{{{h[2]}}} & \\multicolumn{{3}}{{c}}{{{h[3]}}}\\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}\\cmidrule(l){{8-10}}
{h[0]} & $\\NRMSE(0)$ & $\\NRMSE(T)$ & $\\AULC(T)$ & $\\NRMSE(0)$ & $\\NRMSE(T)$ & $\\AULC(T)$ & $\\NRMSE(0)$ & $\\NRMSE(T)$ & $\\AULC(T)$\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
""")
    open(os.path.join(OUT, f'table_obj1{_A.suffix}_{lang}.tex'), 'w').write(tex)
print(open(os.path.join(OUT, f'table_obj1{_A.suffix}_ja.tex')).read())
