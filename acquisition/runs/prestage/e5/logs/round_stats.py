#!/usr/bin/env python3
"""Round times of the production trajectories of E5 for the rounds that ran entirely after the pinning (2026-11-15 11:08:47)."""
import re, sys, collections, pathlib, statistics as st
L = pathlib.Path(__file__).resolve().parent.parent.parent / "prod_conf" / "logs"
SNAP = pathlib.Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else "pin_snapshot.txt")
snap = dict(l.split() for l in open(SNAP))
T0 = SNAP.stat().st_mtime
pat = re.compile(r"^n=(\d+) fit\+predict ([\d.]+)s .*total ([\d.]+)s")
post, pre, state = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
for f in sorted(L.glob("*.log")):
    name = f.name.split("_r")[0]
    rows = [(int(m[1]), float(m[3])) for m in (pat.match(l) for l in open(f)) if m]
    done = "W16U_ENS_DONE" in f.read_text()[-200:]
    state[name].append(("done" if done else "run", rows[-1][0] if rows else None, rows[-1][1] if rows else 0.0))
    new_after = f.name not in snap and rows and f.stat().st_mtime > T0 and not (done and f.stat().st_mtime - rows[-1][1] < T0)
    k = int(snap[f.name]) if f.name in snap else (0 if new_after else 10 ** 9)
    for i in range(1, len(rows)):
        d = rows[i][1] - rows[i - 1][1]
        (post if i >= k + 1 else pre)[name].append(d)
for name in state:
    s = state[name]
    run = [x for x in s if x[0] == "run"]
    q = lambda v: (f"n {len(v)} median {st.median(v):.0f} s mean {st.mean(v):.0f} s max {max(v):.0f} s" if v else "none")
    print(f"{name:8s} logs {len(s)} done {len(s) - len(run)} running {len(run)} n of running "
          f"{sorted(x[1] for x in run if x[1])[:1]}..{sorted(x[1] for x in run if x[1])[-1:]} | rounds before the pinning: {q(pre[name])} | "
          f"rounds entirely after the pinning: {q(post[name])}")
    if len(s) - len(run):
        tot = [x[2] for x in s if x[0] == "done"]
        print(f"         finished trajectories: total seconds median {st.median(tot):.0f} max {max(tot):.0f}")
