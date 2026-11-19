#!/usr/bin/env python3
"""Per running trajectory of E5: time of the rounds that ran entirely after the memory move (snapshot mig_snapshot.txt,
2026-11-15 12:04) divided by the time of the same number of rounds that ended before the pinning (snapshot
pin_snapshot.txt, 11:09). The two sets of rounds of one trajectory have adjacent n, so the ratio is not biased by the
start of new trajectories. Prints, per setting, the number of trajectories and quantiles of the two times and of the ratio."""
import re, collections, pathlib, statistics as st
H = pathlib.Path(__file__).resolve().parent
L = H.parent.parent / "prod_conf" / "logs"
pin = dict(l.split() for l in open(H / "pin_snapshot.txt"))
mig = dict(l.split() for l in open(H / "mig_snapshot.txt"))
pat = re.compile(r"^n=(\d+) fit\+predict ([\d.]+)s")
out = collections.defaultdict(list)
for f in sorted(L.glob("*.log")):
    if f.name not in pin or f.name not in mig:
        continue
    t = [float(m[2]) for m in (pat.match(l) for l in open(f)) if m]
    kp, km = int(pin[f.name]), int(mig[f.name])
    post = t[km + 1:]                      # row km was in progress at the snapshot after the memory move
    pre = t[max(0, kp - len(post)):kp][-5:] if post else []
    post = post[:5]
    if post and pre:
        out[f.name.split("_r")[0]].append((st.mean(pre), st.mean(post), st.mean(post) / st.mean(pre)))
def q(v):
    v = sorted(v)
    return f"min {v[0]:.2f} quartile {v[len(v) // 4]:.2f} median {st.median(v):.2f} quartile {v[(3 * len(v)) // 4]:.2f} max {v[-1]:.2f}"
for name, rows in out.items():
    print(f"{name}: trajectories {len(rows)}")
    print("  seconds per round before the pinning:  ", q([r[0] for r in rows]))
    print("  seconds per round after the memory move:", q([r[1] for r in rows]))
    print("  ratio after / before:                   ", q([r[2] for r in rows]))
