"""Purpose 1 (c): automatic replacement of functions excluded by the ESS rule (2026-11-08, author's decision).

Goal: TARGET more valid functions from rep 16 on (valid = OFML-GPN and Independent-GPN both reach t = 200 without a
round of pre-resampling ESS/P < 0.001).  Flow per function:
  OFML-GPN worker (p1_auto_ofml.sbatch, --abort-on-exclusion)
    rc 0 -> ok: submit Independent-GPN (p1_auto_indep.sbatch, same host) and the 4 tuned neural methods (xeon6530-h100nvl CPU)
    rc 3 -> excluded (stopped at the round that hit the rule)
  after every worker / Independent-GPN job: `launch <host>` submits new OFML-GPN workers while
    valid + YIELD * in_flight < TARGET  and  in_flight < MAX_IN_FLIGHT  and  failures < MAX_FAILED.
  YIELD = 7/16 (share of the first 16 functions that passed).  Each new worker takes the next unused rep number.
Which valid functions enter the paper: the first TARGET valid ones in rep order (fixed before seeing any result).

usage:
  p1_auto.py launch HOST [N]       submit up to N workers on HOST (default: as many as the rule allows)
  p1_auto.py set REP KIND STATUS   KIND ofml|indep|neural, STATUS queued|running|ok|excluded|failed|submitted
  p1_auto.py followups REP HOST    submit Independent-GPN on HOST and the neural methods on xeon6530-h100nvl
  p1_auto.py status
State: runs/p1_main/auto/state.json (lock: state.lock), events: runs/p1_main/auto/events.log.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

GG = Path(__file__).resolve().parents[3]          # transfer/gpn
NN = GG.parent / "ensembles"
AUTO = GG / "runs/p1_main/auto"
SLURM = GG / "runs/p1_main/slurm"
FIRST_REP, LAST_REP = 16, 39          # problems rep_16 .. rep_39 exist
TARGET = 5
YIELD = 7 / 16
MAX_IN_FLIGHT = 8
MAX_FAILED = 2
NEURAL_ENV = "OUT=runs/main_2026-11-08,CFGDIR=runs/cfg_2026-11-08"
NEURAL_ARRAY = "0,2,3,4"              # shared, dagnn_cat, sharedcell, ofml (DAGNN-Ens waits for its tuning)


@contextmanager
def locked():
    AUTO.mkdir(parents=True, exist_ok=True)
    with open(AUTO / "state.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        p = AUTO / "state.json"
        st = json.loads(p.read_text()) if p.exists() else {"reps": {}}
        yield st
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
        os.replace(tmp, p)
        fcntl.flock(lk, fcntl.LOCK_UN)


def log(msg):
    AUTO.mkdir(parents=True, exist_ok=True)
    with open(AUTO / "events.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {socket.gethostname()} {msg}\n")


def counts(st):
    reps = st["reps"]
    valid = sum(1 for r in reps.values() if r.get("ofml") == "ok" and r.get("indep") not in ("excluded", "failed"))
    in_flight = sum(1 for r in reps.values() if r.get("ofml") in ("queued", "running"))
    failed = sum(1 for r in reps.values() for k in ("ofml", "indep") if r.get(k) == "failed")
    return valid, in_flight, failed


def run_on(host, cmd):
    if socket.gethostname() == host:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True)


def sbatch(host, script, env, extra=""):
    cmd = f"cd {GG} && sbatch --parsable --export=ALL,{env} {extra} {script}"
    r = run_on(host, cmd)
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed on {host}: {r.stderr.strip()}")
    return r.stdout.strip()


def launch(host, n=None):
    with locked() as st:
        made = []
        while n is None or len(made) < n:
            valid, in_flight, failed = counts(st)
            if not (valid + YIELD * in_flight < TARGET and in_flight < MAX_IN_FLIGHT and failed < MAX_FAILED):
                break
            used = {int(k) for k in st["reps"]}
            free = [r for r in range(FIRST_REP, LAST_REP + 1) if r not in used]
            if not free:
                log("no problem files left; stop launching")
                break
            rep = free[0]
            job = sbatch(host, SLURM / "p1_auto_ofml.sbatch", f"REP={rep}", f"--job-name=p1auto_main_{rep:02d}")
            st["reps"][str(rep)] = dict(ofml="queued", ofml_host=host, ofml_job=job)
            made.append(rep)
            log(f"launch rep {rep} OFML-GPN on {host} job {job} (valid {valid}, in flight {in_flight + 1})")
        valid, in_flight, failed = counts(st)
    print(f"launched {made}; valid {valid}, in flight {in_flight}, failed {failed}")


def set_status(rep, kind, status):
    with locked() as st:
        st["reps"].setdefault(str(int(rep)), {})[kind] = status
    log(f"rep {int(rep)} {kind} -> {status}")


def followups(rep, host):
    rep = int(rep)
    job = sbatch(host, SLURM / "p1_auto_indep.sbatch", f"REP={rep}", f"--job-name=p1auto_indep_{rep:02d}")
    ncmd = (f"cd {NN} && sbatch --parsable --job-name=p1n_auto_{rep:02d} --array={NEURAL_ARRAY} "
            f"--export=ALL,{NEURAL_ENV},REPS={rep} slurm/main_traj.sbatch")
    r = run_on("xeon6530-h100nvl", ncmd)
    njob = r.stdout.strip() if r.returncode == 0 else f"FAILED: {r.stderr.strip()}"
    with locked() as st:
        e = st["reps"].setdefault(str(rep), {})
        e.update(indep="queued", indep_host=host, indep_job=job, neural="submitted", neural_job=njob)
    log(f"rep {rep} follow-ups: Independent-GPN on {host} job {job}; neural on xeon6530-h100nvl job {njob}")


def status():
    with locked() as st:
        valid, in_flight, failed = counts(st)
        print(f"valid {valid} / target {TARGET}, OFML-GPN in flight {in_flight}, failed {failed}")
        for k in sorted(st["reps"], key=int):
            print(k, st["reps"][k])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "launch":
        launch(a[1], int(a[2]) if len(a) > 2 else None)
    elif a[0] == "set":
        set_status(a[1], a[2], a[3])
    elif a[0] == "followups":
        followups(a[1], a[2])
    elif a[0] == "status":
        status()
    else:
        sys.exit(f"unknown command {a[0]}")
