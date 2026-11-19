"""目的 1(c)：OFML-GPN の 95% 被覆率（PIT が (0.025, 0.975) に入る格子の点の割合）を、rep ごと、群（全体・直列・合流）ごとに
t = 0, 50, 100, 200 で表にする。zeroshot_vs_coverage.py と同じ定義。usage: coverage_by_t.py [REPS]（既定は規則に当たらなかった 14 本）"""
import os, sys
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'scripts'))
import ofml_paths as _OP  # noqa: E402
RUN = str(_OP.TRANSFER / 'gpn' / 'runs' / 'p1_main')
REPS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [2, 4, 5, 9, 11, 12, 15, 17, 19, 23, 27, 28, 29, 32]
GR = {'全体': np.arange(448), '直列': np.arange(256), '合流': np.arange(256, 448)}
TS = [0, 50, 100, 200]
C = {}
for r in REPS:
    pit = np.load(f'{RUN}/ofml_gpn/rep_{r:02d}.npz')['pit']
    for g, i in GR.items():
        for t in TS:
            C[r, g, t] = np.mean((pit[t, i] > 0.025) & (pit[t, i] < 0.975))
out = [f'OFML-GPN の 95% 被覆率、rep {REPS}（{len(REPS)} 本）', '']
for g in GR:
    out += [f'## {g}（格子の点 {len(GR[g])} 個）', '', '| rep | ' + ' | '.join(f't={t}' for t in TS) + ' |', '|---' * (len(TS) + 1) + '|']
    for r in REPS:
        out.append(f'| {r} | ' + ' | '.join(f'{C[r, g, t]:.3f}' for t in TS) + ' |')
    for lab, fn in (('平均', np.mean), ('最小', np.min), ('最大', np.max)):
        out.append(f'| {lab} | ' + ' | '.join(f'{fn([C[r, g, t] for r in REPS]):.3f}' for t in TS) + ' |')
    out.append('')
txt = '\n'.join(out)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'coverage_by_t_{len(REPS)}reps.md'), 'w').write(txt)
print(txt)
