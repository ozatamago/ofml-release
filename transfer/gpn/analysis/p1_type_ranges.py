"""Purpose-1 draft library: input/output ranges of every operation type of the TRUE functions (teachers).

For each teacher (seed 47_000_000 + 1000 s, the smoke-run convention) and each type, the input values of its
nodes (layer 1: x on linspace(0,4,64); others: the parents' outputs) and output values, split by whether the
node lies in a source protocol, a target protocol only, or both.  Coverage of a target-only input point:
within one lengthscale (Euclidean, 0.5) of some input point of a source node of the same type.

    JAX_PLATFORMS=cpu OFML_GPN_ALLOW_ANY_GRID=1 <ucb venv python> analysis/p1_type_ranges.py [N_TEACHERS]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ofml_gpn import graph_library as GL  # noqa: E402
from ofml_gpn.term_library import purpose1_library  # noqa: E402

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
T = purpose1_library()
L = T.library
WX = np.linspace(0.0, 4.0, 64)


def subtree(i, acc):
    acc.add(i)
    for q in L.nodes[i].parents:
        subtree(q, acc)
    return acc


src_nodes, tgt_nodes = set(), set()
for k, p in enumerate(L.protocols):
    (src_nodes if k in T.sources else tgt_nodes).update(subtree(p, set()))
tgt_only = tgt_nodes - src_nodes
print("protocol terms:", T.protocol_terms)
print("target-only nodes:", [T.node_terms[i] for i in sorted(tgt_only)])

rows = {}
for s in range(NT):
    seed = 47_000_000 + 1000 * s
    truth, _, det = GL.draw_graph_teacher(L, WX, 1e-8, seed, seed + 1, 0.1, return_details=True)
    out = det["node_out"]
    per = {}
    for i, nd in enumerate(L.nodes):
        z = WX[:, None] if not nd.parents else np.stack([out[q] for q in nd.parents], axis=1)
        d = per.setdefault(nd.type_name, {"in_src": [], "in_tgt": [], "out": []})
        d["out"].append(out[i])
        (d["in_tgt"] if i in tgt_only else d["in_src"]).append(z)
    zr = np.stack([out[p] for p in L.protocols], axis=1)  # R reads the protocol nodes (one value per protocol)
    per["R"] = {"in_src": [], "in_tgt": [], "out": [truth]}
    for k, p in enumerate(L.protocols):
        (per["R"]["in_src"] if k in T.sources else per["R"]["in_tgt"]).append(out[p][:, None])
    for ty, d in per.items():
        r = rows.setdefault(ty, {"in_src": [], "in_tgt": [], "out": [], "cov": []})
        for key in ("in_src", "in_tgt", "out"):
            r[key] += d[key]
        if d["in_tgt"] and d["in_src"]:
            S = np.concatenate(d["in_src"]); Q = np.concatenate(d["in_tgt"])
            dist = np.min(np.linalg.norm(Q[:, None, :] - S[None, :, :], axis=-1), axis=1)
            r["cov"].append(float(np.mean(dist <= 0.5)))

print(f"\n{NT} teachers; ranges are [min, max] over all teachers and grid points")
for ty, r in rows.items():
    o = np.concatenate(r["out"])
    line = f"{ty:3s} out [{o.min():+.2f}, {o.max():+.2f}]"
    for key in ("in_src", "in_tgt"):
        if r[key]:
            z = np.concatenate(r[key])
            line += f" | {key} " + " x ".join(f"[{z[:, j].min():+.2f}, {z[:, j].max():+.2f}]" for j in range(z.shape[1]))
    if r["cov"]:
        line += f" | target-only inputs within 0.5 of a source input: mean {np.mean(r['cov']):.2f} (min {np.min(r['cov']):.2f})"
    print(line)

# per-teacher ranges: median over teachers of the per-teacher [min, max]
print("\nper teacher: median over teachers of [min, max] (and of the range width)")
for ty, r in rows.items():
    def per_teacher(lst, n_per):
        chunks = [np.concatenate(lst[k * n_per:(k + 1) * n_per]) for k in range(NT)]
        lo = np.median([c.min(0) for c in chunks], axis=0); hi = np.median([c.max(0) for c in chunks], axis=0)
        return " x ".join(f"[{a:+.2f}, {b:+.2f}]" for a, b in zip(np.atleast_1d(lo), np.atleast_1d(hi)))
    line = f"{ty:3s} out {per_teacher(r['out'], len(r['out']) // NT)}"
    for key in ("in_src", "in_tgt"):
        if r[key]:
            line += f" | {key} {per_teacher(r[key], len(r[key]) // NT)}"
    print(line)
