"""Progress of the purpose-1 (c) GP production runs from their Slurm logs (runs/p1_main/slurm_logs/p1{main,indep}_rep_*):
per problem the latest log, rounds done, t reached, rejuvenations, mean rejuvenation seconds, min pre-resampling
ESS / P and the exclusion flag of plan 1.8.5 (any round < 0.001).  usage: p1_status.py [log_dir]"""
import glob, json, os, re, sys
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "runs", "p1_main", "slurm_logs")
latest = {}
for f in sorted(glob.glob(os.path.join(d, "p1*_rep_*.out")), key=os.path.getmtime):
    head = open(f).readline()
    m = re.search(r"method=(\w+) problem=\S*/(\w+)\.npz", head)
    if m:
        latest[(m.group(1), m.group(2))] = (f, head)
for (meth, prob), (f, head) in sorted(latest.items()):
    rs, t, done = [], -1, ""
    for line in open(f):
        if line.startswith("[round] "):
            rs.append(json.loads(line[8:]))
        elif line.startswith("[t="):
            t = int(line[3:line.index("]")])
        elif line.startswith("[done]") or line.startswith("rc="):
            done = line.strip()[:60]
    rj = [r for r in rs if r.get("resampled")]
    ess = [r["ess_pre_over_P"] for r in rs if r.get("ess_pre_over_P") is not None]
    mn = min(ess) if ess else float("nan")
    host = re.search(r"host=(\S+)", head).group(1)
    print(f"{meth:5s} {prob}  {host:6s} rounds {len(rs):3d} t {t:3d} rejuv {len(rj):3d} "
          f"mean rejuv {sum(r['rejuv_seconds'] for r in rj) / max(1, len(rj)):6.1f}s  min ESS/P {mn:.4f}"
          f"{'  EXCLUDED (<0.001)' if mn < 1e-3 else ''}  {done}")
