"""Triangle counting in the COUNTING semiring on a power-law (Chung-Lu style) undirected graph.

Edges are oriented from the lower to the higher (degree, id) rank, so every triangle
{a < b < c} is counted exactly once as the wedge a -> b -> c closed by the edge a -> c.

Two algebra forms run on the same instance (both results are returned):
  materialized:  wedges = combine(E(i,j), E(j,k))            # every 2-path (i, j, k)
                 closed = mask(wedges, E(i,k))               # semijoin on (i, k)
                 count  = reduce(closed, keep=())
  fused:         W      = combine(E(i,j), E(j,k), keep=(i,k))  # wedge counts per (i, k), reduce fused
                 T      = combine(W, E(i,k))                   # elementwise: wedges x 1 on closed pairs
                 count  = reduce(T, keep=())

Golden: for every oriented edge (i, j) intersect the out-neighbour sets N+(i) and N+(j) by
probing the smaller into the larger (hash intersection). Hotspot event: candidate third vertices
examined -- an adjacency entry probed by the golden, a wedge (i, j, k) produced by the join.

native_ops convention: the trace holds both forms, so native_ops = 2 x golden probes and the
harness ratio is the mean of the two per-form ratios listed in `extra`.
"""
from __future__ import annotations

import random

from algebra import COUNTING, Relation, combine, mask, reduce, rename, current_trace

NAME = "triangle_count"
CATEGORY = "graph"
SEMIRINGS = ["counting"]
STRUCTURE = "sparse join (one-shot, no fixpoint)"
HOTSPOT_EVENT = "candidate third vertices examined: adjacency entries probed (golden) / wedges (i,j,k) produced (join)"
SCALES = {"small": {"n": 3000, "m": 15000}, "large": {"n": 20000, "m": 120000}}
GAMMA = 2.5          # power-law exponent of the expected-degree sequence


def make_instance(scale: str, seed: int):
    """Chung-Lu style: expected degree of rank-i vertex proportional to (i+1)^(-1/(gamma-1));
    endpoints are sampled proportionally to the expected degrees until m distinct edges exist
    (self-loops and duplicates rejected)."""
    p = SCALES[scale]
    rng = random.Random(seed)
    n, m = p["n"], p["m"]
    w = [(i + 1) ** (-1.0 / (GAMMA - 1.0)) for i in range(n)]
    edges = set()
    while len(edges) < m:
        batch = rng.choices(range(n), weights=w, k=2 * (m - len(edges)) + 16)
        for a, b in zip(batch[0::2], batch[1::2]):
            if a != b:
                edges.add((a, b) if a < b else (b, a))
            if len(edges) >= m:
                break
    deg = [0] * n
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    rank = {v: r for r, v in enumerate(sorted(range(n), key=lambda v: (deg[v], v)))}
    oriented = sorted((a, b) if rank[a] < rank[b] else (b, a) for a, b in edges)
    return {"n": n, "m": m, "oriented": oriented, "deg": deg}


# ---------------------------------------------------------------------------------------------
def native(inst):
    """Edge-iterator with hash intersection of the out-neighbour sets; probes the smaller set."""
    n = inst["n"]
    out = [set() for _ in range(n)]
    for i, j in inst["oriented"]:
        out[i].add(j)
    probes = 0
    tri = 0
    for i, j in inst["oriented"]:
        a, b = (out[i], out[j]) if len(out[i]) <= len(out[j]) else (out[j], out[i])
        probes += len(a)
        for k in a:
            if k in b:
                tri += 1
    native.last_counts = {"probes": probes, "triangles": tri}
    # claude: decision start — native_ops counts the golden probes once per algebra form (materialized, fused) so the harness ratio is the mean of the two per-form ratios
    return {"triangles": tri}, 2 * probes
    # claude: decision end


# ---------------------------------------------------------------------------------------------
def _tag_totals(records, prefix):
    out = {"records": 0, "candidates": 0, "mul_ops": 0, "add_ops": 0, "n_out": 0}
    for r in records:
        if r.tag == prefix or r.tag.startswith(prefix + "/"):
            out["records"] += 1
            out["candidates"] += r.candidates
            out["mul_ops"] += r.mul_ops
            out["add_ops"] += r.add_ops
            out["n_out"] += r.n_out
    return out


def algebra(inst):
    n = inst["n"]
    dom = {"i": n, "j": n, "k": n}
    E = Relation(("i", "j"), COUNTING, {e: 1 for e in inst["oriented"]}, dom)
    Ejk = rename(E, {"i": "j", "j": "k"})
    Eik = rename(E, {"j": "k"})
    tr = current_trace()
    with tr.tag("materialized"):
        wedges = combine(E, Ejk)                          # (i,j) x (j,k) -> (i,j,k)
        closed = mask(wedges, Eik)                        # keep wedges whose (i,k) is an edge
        c1 = reduce(closed, keep=())
    with tr.tag("fused"):
        W = combine(E, Ejk, keep=("i", "k"))              # wedge counts per (i,k)
        T = combine(W, Eik)                               # x 1 on closed pairs
        c2 = reduce(T, keep=())
    recs = tr.records
    algebra.last_info = {"materialized": _tag_totals(recs, "materialized"),
                         "fused": _tag_totals(recs, "fused"),
                         "wedges": len(wedges), "closed_wedges": len(closed),
                         "wedge_pairs": len(W), "closed_pairs": len(T)}
    return {"triangles_materialized": int(c1.get(())), "triangles_fused": int(c2.get(()))}


def compare(a, b) -> bool:
    return a["triangles"] == b["triangles_materialized"] == b["triangles_fused"]


def extra_info(inst, res):
    li = algebra.last_info
    g = native.last_counts
    deg = inst["deg"]
    outdeg = {}
    indeg = {}
    for i, j in inst["oriented"]:
        outdeg[i] = outdeg.get(i, 0) + 1
        indeg[j] = indeg.get(j, 0) + 1
    return {
        "golden": g,
        "triangles": res,
        "max_degree": max(deg), "max_outdegree": max(outdeg.values()),
        "wedges_oriented": li["wedges"],
        "wedges_unoriented_derived": sum(d * (d - 1) // 2 for d in deg),   # derived: sum_j C(d(j), 2)
        "closed_wedges": li["closed_wedges"], "wedge_pairs": li["wedge_pairs"],
        "materialized": {**li["materialized"], "ratio_mul_vs_probes": li["materialized"]["mul_ops"] / g["probes"],
                         "ratio_cand_vs_probes": li["materialized"]["candidates"] / g["probes"]},
        "fused": {**li["fused"], "ratio_mul_vs_probes": li["fused"]["mul_ops"] / g["probes"],
                  "ratio_cand_vs_probes": li["fused"]["candidates"] / g["probes"]},
    }
