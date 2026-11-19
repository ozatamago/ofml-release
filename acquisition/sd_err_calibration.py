"""2026-11-11: PB-W16U, IVR runs: does the posterior standard deviation track the error?
Per rep and observation count n in {8, 50, 100}: Spearman correlation between s_t(x) and |mu_t(x) - f(x)| over the
1024 grid cells, and the median of |mu_t - f| / s_t (0.674 for a calibrated Gaussian). Mean (min, max) over 16 reps.
OFML-Ens = ens_v2; its stored var is the population variance (divided by 5), as in Figure 4(b).
usage: sd_err_calibration.py runs/2026-11-04_main"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import w16u_common as C
run = Path(sys.argv[1])
ENS = sys.argv[2] if len(sys.argv) > 2 else "ens_v2"   # 2026-11-18: absolute path of the s04 results allowed
for m, d in (("OFML-GPN", "gpn"), ("OFML-Ens", ENS)):
    a = []
    for r in range(16):
        f = C.load_teacher(r)[0]
        z = np.load(run / d / f"rep_{r:04d}_ivr.npz"); n = z["n_obs"]
        row = []
        for t in (8, 50, 100):
            k = int(np.where(n == t)[0][0]); err = np.abs(z["mean"][k] - f); sd = np.sqrt(np.maximum(z["var"][k], 0))
            row.append((spearmanr(sd, err).correlation, np.median(err / np.maximum(sd, 1e-12))))
        a.append(row)
    a = np.array(a)
    for j, t in enumerate((8, 50, 100)):
        print(f"| {m} | {t} | {a[:,j,0].mean():.2f} ({a[:,j,0].min():.2f}, {a[:,j,0].max():.2f}) | {a[:,j,1].mean():.2f} ({a[:,j,1].min():.2f}, {a[:,j,1].max():.2f}) |")
