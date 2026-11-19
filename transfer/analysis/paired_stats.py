"""目的 1(c)：規則に当たらずに t=200 まで終わった rep での NRMSE(0)、AULC(200)、NRMSE(200) と、対応のある差（提案 − 比較、負なら提案が良い）。
95% 区間は rep を重複を許して選び直すブートストラップ 10000 回。勝ち数は提案の誤差が小さかった rep の数。群は全体・直列・合流。"""
import os, sys, json
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'scripts'))
import ofml_paths as _OP  # noqa: E402
EXP = str(_OP.TRANSFER)
sys.path.insert(0, os.path.join(EXP, 'gpn'))
import p1_common as P
RUN = os.path.join(EXP, 'gpn/runs/p1_main'); NN = os.path.join(EXP, 'ensembles/runs/main_2026-11-08')
G = 64; GR = {'全体': np.arange(448), '直列': np.arange(256), '合流': np.arange(256, 448)}
VALID = [2, 4, 5, 9, 11, 12, 15]
M = {'OFML-GPN': f'{RUN}/ofml_gpn', 'Independent-GPN': f'{RUN}/indep_gpn', 'OFML-Ens': f'{NN}/ofml',
     'SharedCell-DAG-Ens': f'{NN}/sharedcell', 'DAGNN-Cat-Ens': f'{NN}/dagnn_cat', 'Shared-Ens': f'{NN}/shared'}
def nr(pred, f, idx):
    return np.sqrt(np.mean((pred[:, idx] - f[idx]) ** 2, axis=1)) / f[idx].std()
def metrics(c):
    return {'NRMSE(0)': c[0], 'AULC(200)': np.mean((c[:-1] + c[1:]) / 2), 'NRMSE(200)': c[200]}
V = {}  # (method, group, metric) -> array over reps
for r in VALID:
    f = P.load_problem(P.problem_path(RUN + '/problems', r)).truth_tgt
    for m, d in M.items():
        mean = np.load(f'{d}/rep_{r:02d}.npz')['mean']
        for g, idx in GR.items():
            for k, v in metrics(nr(mean, f, idx)).items():
                V.setdefault((m, g, k), []).append(v)
rng = np.random.default_rng(20261108)
B = rng.integers(0, len(VALID), size=(10000, len(VALID)))
out = [f'rep {VALID}（{len(VALID)} 本）', '']
for g in GR:
    out += [f'## {g}', '', '| 手法 | NRMSE(0) | AULC(200) | NRMSE(200) |', '|---|---|---|---|']
    for m in M:
        out.append(f'| {m} | ' + ' | '.join(f'{np.mean(V[m, g, k]):.3f}' for k in ['NRMSE(0)', 'AULC(200)', 'NRMSE(200)']) + ' |')
    out += ['', '| 比較 | ΔNRMSE(0) | ΔAULC(200) | ΔNRMSE(200) |', '|---|---|---|---|']
    for a, b in [('OFML-GPN', 'Independent-GPN'), ('OFML-GPN', 'OFML-Ens'), ('OFML-Ens', 'SharedCell-DAG-Ens'),
                 ('OFML-Ens', 'DAGNN-Cat-Ens'), ('OFML-Ens', 'Shared-Ens')]:
        cells = []
        for k in ['NRMSE(0)', 'AULC(200)', 'NRMSE(200)']:
            d = np.array(V[a, g, k]) - np.array(V[b, g, k]); bs = d[B].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {int((d < 0).sum())}/{len(d)}')
        out.append(f'| {a} − {b} | ' + ' | '.join(cells) + ' |')
    out.append('')
txt = '\n'.join(out)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paired_stats_valid7.md'), 'w').write(txt)
print(txt)
