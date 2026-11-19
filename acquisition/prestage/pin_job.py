#!/usr/bin/env python3
"""Give every w16u_ens.py process of one Slurm job its own logical CPU.

Reason (2026-11-15, banyan): NUMA nodes 0 and 1 had almost no free memory (page cache), so the memory of all processes
of the job was allocated on nodes 2 and 3, and the automatic NUMA balancing of the kernel kept all 112 single-thread
processes on the logical CPUs of nodes 2 and 3 (48 of the 112 allocated logical CPUs), three to five processes per
logical CPU, while the 64 allocated logical CPUs of nodes 0 and 1 were idle.

A process that is pinned to a logical CPU of node 0 or 1 keeps its memory on node 2 or 3 (first touch; the kernel does
not move it because nodes 0 and 1 have no free memory). The script therefore also moves the private memory of every
pinned process to the NUMA node of its logical CPU with migratepages(8) (the clean page cache of that node is reclaimed
by the kernel): once 60 s or more after the process was first seen, and once more 600 s or more after that.

The script changes only the CPU affinity and the physical location of the memory of processes of the given job that
belong to the calling user, and only within the affinity mask that Slurm gave to the job. It does not change any
computed value.

  pin_job.py JOBID            one pass, then exit
  pin_job.py JOBID --loop S   repeat every S seconds until the cgroup of the job disappears
"""
import os
import subprocess
import sys
import time

JOB = sys.argv[1]
TAG = f"/job_{JOB}/"
CG = f"/sys/fs/cgroup/system.slice/localhost_slurmstepd.scope/job_{JOB}"
UID = os.getuid()


def node_of_cpu() -> dict:
    out = {}
    base = "/sys/devices/system/node"
    for d in os.listdir(base):
        if d.startswith("node") and d[4:].isdigit():
            for part in open(f"{base}/{d}/cpulist").read().strip().split(","):
                a, _, b = part.partition("-")
                for c in range(int(a), int(b or a) + 1):
                    out[c] = int(d[4:])
    return out


NODE = node_of_cpu()


def siblings_of_cpu() -> dict:
    """logical CPU -> the other logical CPUs of the same physical core."""
    out = {}
    for c in NODE:
        try:
            txt = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
        except FileNotFoundError:
            continue
        s = set()
        for part in txt.split(","):
            a, _, b = part.partition("-")
            s.update(range(int(a), int(b or a) + 1))
        out[c] = s - {c}
    return out


SIBLINGS = siblings_of_cpu()


def procs() -> list:
    """(pid, affinity of the main thread, logical CPU on which the main thread ran last) of the target processes."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if os.stat(f"/proc/{d}").st_uid != UID:
                continue
            if not open(f"/proc/{d}/comm").read().startswith("python"):
                continue
            if b"w16u_ens.py" not in open(f"/proc/{d}/cmdline", "rb").read():
                continue
            if TAG not in open(f"/proc/{d}/cgroup").read():
                continue
            psr = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[36])
            out.append((int(d), os.sched_getaffinity(int(d)), psr))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return out


def pin(pid: int, cpu: int) -> bool:
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            os.sched_setaffinity(int(t), {cpu})
        return True
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


SEEN = {}          # pid -> [time first seen, number of migrations, time of the last migration]
NODES = sorted(set(NODE.values()))


def migrate(now: float, ps: list) -> None:
    alive = {pid for pid, _, _ in ps}
    for pid in list(SEEN):
        if pid not in alive:
            del SEEN[pid]
    moved = 0
    for pid, aff, _ in ps:
        rec = SEEN.setdefault(pid, [now, 0, 0.0])
        if len(aff) != 1:
            continue
        due = (rec[1] == 0 and now - rec[0] >= 60.0) or (rec[1] == 1 and now - rec[2] >= 600.0)
        if not due:
            continue
        nd = NODE[next(iter(aff))]
        others = ",".join(str(n) for n in NODES if n != nd)
        r = subprocess.run(["migratepages", str(pid), others, str(nd)], capture_output=True)
        rec[1] += 1
        rec[2] = now
        moved += int(r.returncode == 0)
    if moved:
        print(f"{time.strftime('%F %T')} moved the memory of {moved} processes to the node of their logical CPU", flush=True)


def one_pass(allowed: set, do_migrate: bool = False) -> set:
    ps = procs()
    if do_migrate:
        migrate(time.time(), ps)
    for _, aff, _ in ps:
        if len(aff) > 1:
            allowed = allowed | aff                    # the mask that Slurm gave to the job
    taken = {next(iter(aff)) for _, aff, _ in ps if len(aff) == 1}
    todo = [(pid, psr) for pid, aff, psr in ps if len(aff) > 1]
    free = sorted(allowed - taken)
    done = []
    busy = set(taken)
    # Processes are served in the order of their process numbers (the order of the command file). Preference:
    # (1) a free logical CPU on the NUMA node where the process runs whose sibling thread is not in use,
    # (2) such a logical CPU on any node, (3) any free logical CPU on the same node, (4) any free logical CPU.
    # With this order the first 64 processes of a job get 64 different physical cores (2026-11-17).
    for pid, psr in sorted(todo):
        if not free:
            break
        same = [c for c in free if NODE.get(c) == NODE.get(psr)]
        whole = [c for c in free if not (SIBLINGS.get(c, set()) & busy)]
        whole_same = [c for c in whole if NODE.get(c) == NODE.get(psr)]
        c = (whole_same or whole or same or free)[0]
        free.remove(c)
        busy.add(c)
        done.append((pid, c, pin(pid, c)))
    if done or len(todo) != len(done):
        by_node = {}
        for _, c, ok in done:
            if ok:
                by_node[NODE.get(c)] = by_node.get(NODE.get(c), 0) + 1
        print(f"{time.strftime('%F %T')} processes {len(ps)} pinned-before {len(taken)} newly-pinned "
              f"{sum(ok for _, _, ok in done)} by-node {dict(sorted(by_node.items()))} unpinned-left "
              f"{len(todo) - len(done)} free-left {len(free)}", flush=True)
    return allowed


def main() -> None:
    allowed = set()
    if "--loop" not in sys.argv:
        one_pass(allowed)
        return
    every = float(sys.argv[sys.argv.index("--loop") + 1])
    while os.path.isdir(CG):
        allowed = one_pass(allowed, do_migrate=True)
        time.sleep(every)
    print(f"{time.strftime('%F %T')} the cgroup of job {JOB} is gone; exit", flush=True)


if __name__ == "__main__":
    main()
