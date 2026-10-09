"""Max-min fuzzy relational inference: fuzzy transitive closure plus a batch of fuzzy queries.

    (R o S)(x, z) = max_y min(R(x, y), S(y, z))            composition in the FUZZY (max, min) semiring
    R*            = max_{k >= 1} R^k                        fuzzy transitive closure (fixpoint)
    out_q(z)      = max_x min(q(x), R*(x, z))               one fuzzy query = one composition with R*

Algebra program (one fixpoint round). It is the datalog_tc chain with the semiring swapped:
    C        = combine(rename(Delta, y->m), rename(R, x->m), keep=(x, y))   # Delta o R, max-merged
    X, Delta = update(X, C)                                                  # max-merge; Delta = pairs whose value rose
The antijoin of datalog_tc (mask(C, X, negate=True)) is gone: in FUZZY a pair already in X can still
improve, so membership in X is not a reason to drop a candidate; `update` does the change detection.
Queries: out = combine(q(qid, x), rename(X, y->z), keep=(qid, z)).

Golden: widest-path (max-min) Dijkstra from every element, plain Python and heapq, followed by plain
loops for the queries. Both sides count "(x, y, z) min-evaluations": the reference counts one per
out-edge (y, z) relaxed from a settled element y (and one per (q(x), R*(x, z)) pair in the query
phase); the algebra counts one per matching (Delta(x, y), R(y, z)) pair and one per matching
(q(qid, x), R*(x, z)) pair. The reference settles every (x, y) once; the round-synchronous fixpoint
re-derives a pair every time its value rises, which is where ratio_mul > 1 comes from.
"""
from __future__ import annotations

import heapq
import random

from algebra import FUZZY, Relation, combine, update, rename, fixpoint

NAME = "fuzzy_inference"
CATEGORY = "logic"
SEMIRINGS = ["fuzzy"]
STRUCTURE = "sparse-fixpoint"
HOTSPOT_EVENT = ("(x, y, z) min-evaluations: one per (R*(x,y), R(y,z)) pair examined in the closure "
                 "plus one per (q(x), R*(x,z)) pair in the query phase")
# claude: decision start — the task asked for n=2000 at large scale; measured pure-Python time at
# n=1000 was 52 s for the algebra program alone (2026-09-25, prototype, seed 0), so "large" is set to
# n=800 to stay under the 40 s bound and the requested size is kept as the extra scale "spec_large".
SCALES = {
    "small": {"n": 300, "d": 6, "Q": 32, "qsize": 4, "levels": 100},
    "large": {"n": 800, "d": 6, "Q": 128, "qsize": 4, "levels": 100},
    "spec_large": {"n": 2000, "d": 6, "Q": 128, "qsize": 4, "levels": 100},
}
# claude: decision end


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    n, d, levels = p["n"], p["d"], p["levels"]
    # claude: decision start — membership degrees are quantized to `levels` equal steps in (0, 1], as
    # in a fuzzy rule base; quantization keeps ties exact and removes sub-tolerance value differences.
    rel = {}
    for x in range(n):
        ys = set()
        while len(ys) < d:
            y = rng.randrange(n)
            if y != x:
                ys.add(y)
        for y in sorted(ys):
            rel[(x, y)] = rng.randint(1, levels) / levels
    # claude: decision end
    queries = {}
    for qid in range(p["Q"]):
        for x in rng.sample(range(n), p["qsize"]):
            queries[(qid, x)] = rng.randint(1, levels) / levels
    return {"n": n, "Q": p["Q"], "rel": rel, "queries": queries}


def native(inst):
    """Widest-path Dijkstra from every element x: pop the element y with the largest tentative
    value, relax every out-edge (y, z) once with min(value(y), R(y, z)). The source's own out-edges
    are base facts (X_0 = R on the algebra side) and are not counted; every later relaxation is one
    min-evaluation. Then one min per (q(x) > 0, R*(x, z) > 0) pair for the queries."""
    n = inst["n"]
    adj = [[] for _ in range(n)]
    for (x, y), w in inst["rel"].items():
        adj[x].append((y, w))
    closure = {}
    mins = 0
    for s in range(n):
        best = {}
        heap = []
        for y, w in adj[s]:
            if w > best.get(y, 0.0):
                best[y] = w
                heapq.heappush(heap, (-w, y))
        settled = set()
        while heap:
            negw, y = heapq.heappop(heap)
            w = -negw
            if y in settled or w < best[y]:
                continue                        # stale heap entry
            settled.add(y)
            for z, w2 in adj[y]:
                mins += 1
                c = w2 if w2 < w else w         # min(value(y), R(y, z))
                if c > best.get(z, 0.0):
                    best[z] = c
                    heapq.heappush(heap, (-c, z))
        for y, w in best.items():
            closure[(s, y)] = w
    rows = [[] for _ in range(n)]
    for (x, z), w in closure.items():
        rows[x].append((z, w))
    answers = {}
    for (qid, x), mu in inst["queries"].items():
        for z, w in rows[x]:
            mins += 1
            c = w if w < mu else mu
            key = (qid, z)
            if c > answers.get(key, 0.0):
                answers[key] = c
    return {"closure": closure, "answers": answers}, mins


def algebra(inst):
    n = inst["n"]
    R = Relation(("x", "y"), FUZZY, inst["rel"], {"x": n, "y": n})
    Rm = rename(R, {"x": "m"})                                      # (m, y)

    # claude: decision start — datalog_tc's antijoin `mask(C, X, negate=True)` is dropped: it is only
    # valid when a stored value cannot improve (Boolean OR); in FUZZY the max-merge inside `update`
    # both accumulates and detects the pairs whose value rose, which is the exact semi-naive delta.
    def step(X, D, k):
        C = combine(rename(D, {"y": "m"}), Rm, keep=("x", "y"))     # (x, m) x (m, y) -> (x, y)
        return update(X, C)
    # claude: decision end

    X, info = fixpoint(step, R)
    Qrel = Relation(("qid", "x"), FUZZY, inst["queries"], {"qid": inst["Q"], "x": n})
    out = combine(Qrel, rename(X, {"y": "z"}), keep=("qid", "z"))   # (qid, x) x (x, z) -> (qid, z)
    algebra.last_info = {"rounds": info["rounds"], "delta_sizes": info["delta_sizes"],
                         "closure_size": len(X), "base_size": len(R)}
    return {"closure": dict(X.items()), "answers": dict(out.items())}


def _same(a: dict, b: dict, tol: float = 1e-9) -> bool:
    if len(a) != len(b):
        return False
    for k, v in a.items():
        w = b.get(k)
        if w is None or abs(v - w) > tol:
            return False
    return True


def compare(a, b) -> bool:
    return _same(a["closure"], b["closure"]) and _same(a["answers"], b["answers"])


def extra_info(inst, res):
    info = algebra.last_info
    # every round joins Delta_k with R, so the closure work is sum_k |Delta_k| * d; the reference
    # settles each closure pair once; the ratio of the two sums is the re-derivation factor
    sum_delta = info["base_size"] + sum(info["delta_sizes"])
    return {"rounds": info["rounds"], "closure_size": info["closure_size"],
            "closure_density": info["closure_size"] / (inst["n"] * inst["n"]),
            "sum_delta_sizes": sum_delta,
            "rederivation_factor": sum_delta / info["closure_size"],
            "max_delta": max(info["delta_sizes"]) if info["delta_sizes"] else 0,
            "answers": len(res["answers"])}
