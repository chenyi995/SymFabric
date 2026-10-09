"""Single-source shortest paths (integer weights) and BFS hop distance on a random sparse digraph.

Three algebra programs run on the same instance and `algebra()` returns all three results:

  (a) frontier relaxation in MINPLUS (semi-naive Bellman-Ford):
        cand = combine(frontier(u), E(u, v), keep=(v,));  dist, frontier' = update(dist, cand)
      fixpoint until the frontier is empty.
  (b) Dijkstra with SELECT: one select_k(open, 1) per settled vertex (MINPLUS order), the popped
      vertex is deleted from open (antijoin), its out-edges are relaxed with combine, candidates
      into settled vertices are masked out, the rest is min-merged into open with update.
  (c) BFS levels in BOOLEAN: frontier fixpoint with a mask against the visited set; the level of
      every newly visited vertex is written with a scan (constant = round + 1) into a MINPLUS
      relation.

Golden: heapq Dijkstra (lazy deletion) and queue BFS. Hotspot event: edge relaxations (one per
out-edge scanned from a settled / dequeued vertex). Per-program counts and ratios are reported in
`extra_info`; the select and delete cost of (b) is reported separately from its relaxations.

native_ops convention: the harness divides the trace total of all three programs by native_ops,
so native_ops counts the golden's hotspot events once per algebra program that recomputes the
same result: 2 x Dijkstra relaxations (for (a) and (b)) + 1 x BFS relaxations (for (c)). The
harness ratio is then the work-weighted mean of the three per-program ratios listed in `extra`.
"""
from __future__ import annotations

import heapq
import math
import random
from collections import deque

from algebra import (MINPLUS, BOOLEAN, Relation, combine, mask, update, rename, select_k, scan,
                     fixpoint, current_trace)

NAME = "graph_bfs_sssp"
CATEGORY = "graph"
SEMIRINGS = ["minplus", "boolean"]
STRUCTURE = "frontier-fixpoint + select-loop"
HOTSPOT_EVENT = "edge relaxations: out-edges scanned from a settled (Dijkstra) or dequeued (BFS) vertex"
SCALES = {"small": {"n": 2000, "m": 10000}, "large": {"n": 20000, "m": 100000}}
W_MAX = 10


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    n, m = p["n"], p["m"]
    edges = {}
    while len(edges) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v and (u, v) not in edges:
            edges[(u, v)] = rng.randint(1, W_MAX)
    source = rng.randrange(n)
    return {"n": n, "m": m, "source": source,
            "edges": sorted((u, v, w) for (u, v), w in edges.items())}


# ---------------------------------------------------------------------------------------------
def native(inst):
    """heapq Dijkstra with lazy deletion + queue BFS. Counts one relaxation per out-edge scanned
    from a settled (Dijkstra) or dequeued (BFS) vertex."""
    n, s = inst["n"], inst["source"]
    adj = [[] for _ in range(n)]
    for u, v, w in inst["edges"]:
        adj[u].append((v, w))
    # Dijkstra
    dist = {s: 0}
    heap = [(0, s)]
    settled = set()
    relax_d = 0
    while heap:
        d, u = heapq.heappop(heap)
        if u in settled or d > dist[u]:
            continue
        settled.add(u)
        for v, w in adj[u]:
            relax_d += 1
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    # BFS
    hops = {s: 0}
    dq = deque([s])
    relax_b = 0
    while dq:
        u = dq.popleft()
        for v, _ in adj[u]:
            relax_b += 1
            if v not in hops:
                hops[v] = hops[u] + 1
                dq.append(v)
    native.last_counts = {"dijkstra_relaxations": relax_d, "bfs_relaxations": relax_b}
    # claude: decision start — native_ops counts Dijkstra once per SSSP algebra program (a, b) and BFS once (c), so the harness ratio is the work-weighted mean of the per-program ratios
    native_ops = 2 * relax_d + relax_b
    # claude: decision end
    return {"dist": dist, "hops": hops}, native_ops


# ---------------------------------------------------------------------------------------------
def _tag_totals(records, prefix):
    out = {"records": 0, "candidates": 0, "mul_ops": 0, "add_ops": 0}
    for r in records:
        if r.tag == prefix or r.tag.startswith(prefix + "/"):
            out["records"] += 1
            out["candidates"] += r.candidates
            out["mul_ops"] += r.mul_ops
            out["add_ops"] += r.add_ops
    return out


def _sssp_frontier(inst, E):
    """(a) semi-naive Bellman-Ford: relax the whole frontier every round."""
    n, s = inst["n"], inst["source"]
    X0 = Relation(("u",), MINPLUS, {(s,): 0.0}, {"u": n})
    relax_per_round = []

    def step(X, D, k):
        cand = combine(D, E, keep=("v",))            # (u) x (u,v) -> (v): min_u D(u) + w(u,v)
        relax_per_round.append(current_trace().records[-1].candidates)
        cand = rename(cand, {"v": "u"})
        return update(X, cand)                       # X'(v) = min(X(v), cand(v)); D' = improved

    X, info = fixpoint(step, X0)
    info["relax_per_round"] = relax_per_round
    return X, info


def _sssp_dijkstra_select(inst, E_rows):
    """(b) Dijkstra: one select per settled vertex. E_rows[u] is the out-edge row of u (CSR row
    as a relation); the popped vertex selects its row, an index access."""
    n, s = inst["n"], inst["source"]
    dom = {"u": n}
    dist = Relation(("u",), MINPLUS, None, dom)             # settled vertices = closed set
    open_ = Relation(("u",), MINPLUS, {(s,): 0.0}, dom)
    empty_row = Relation(("u", "v"), MINPLUS, None, {"u": n, "v": n})
    pops = 0
    sum_open = 0
    max_open = 0
    tr = current_trace()
    while len(open_) > 0:
        sum_open += len(open_)
        max_open = max(max_open, len(open_))
        with tr.tag("pop"):
            # claude: decision start — key=value gives the same order as the semiring's `better` (min value, ties by key) and is the cheaper interpreter path; the recorded candidates are identical
            best = select_k(open_, 1, key=lambda k, v: v)   # candidates = |open|
            # claude: decision end
            open_ = mask(open_, best, negate=True)          # point delete: candidates = |open|
        with tr.tag("relax"):
            dist, _ = update(dist, best)                    # settle the popped vertex
            u = next(iter(best.keys()))[0]
            cand = combine(best, E_rows.get(u, empty_row), keep=("v",))   # relax out-edges of u
            cand = rename(cand, {"v": "u"})
            cand = mask(cand, dist, negate=True)            # drop candidates into settled vertices
            open_, _ = update(open_, cand)                  # min-merge into open
        pops += 1
    return dist, {"pops": pops, "sum_open": sum_open, "max_open": max_open}


def _bfs_levels(inst, Eb):
    """(c) BFS: BOOLEAN frontier fixpoint with a mask against the visited set; the level counter
    is the fixpoint round, written to a MINPLUS relation with a constant scan."""
    n, s = inst["n"], inst["source"]
    X0 = Relation.from_set(("u",), [(s,)], {"u": n})
    levels = [Relation(("u",), MINPLUS, {(s,): 0.0}, {"u": n})]

    def step(X, D, k):
        cand = combine(D, Eb, keep=("v",))           # frontier x E -> reached vertices (OR)
        cand = rename(cand, {"v": "u"})
        new = mask(cand, X, negate=True)             # visited check
        X2, D2 = update(X, new)                      # D2 = newly visited = next frontier
        lv = scan(D2, lambda key, v, lvl=float(k + 1): [(key, lvl)], fn_name="const_level",
                  sr=MINPLUS)                        # level counter: round + 1
        levels[0], _ = update(levels[0], lv)
        return X2, D2

    X, info = fixpoint(step, X0)
    return levels[0], info


def algebra(inst):
    n = inst["n"]
    dom = {"u": n, "v": n}
    E = Relation(("u", "v"), MINPLUS, {(u, v): float(w) for u, v, w in inst["edges"]}, dom)
    Eb = Relation.from_set(("u", "v"), [(u, v) for u, v, _ in inst["edges"]], dom)
    # claude: decision start — program (b) reads E as one row relation per source vertex (CSR rows, an index access) because the interpreter's combine streams the larger operand and would read all of E once per pop
    rows: dict[int, dict] = {}
    for u, v, w in inst["edges"]:
        rows.setdefault(u, {})[(u, v)] = float(w)
    E_rows = {u: Relation(("u", "v"), MINPLUS, d, dom) for u, d in rows.items()}
    # claude: decision end
    tr = current_trace()
    with tr.tag("frontier"):
        Xa, info_a = _sssp_frontier(inst, E)
    with tr.tag("dijkstra"):
        Xb, info_b = _sssp_dijkstra_select(inst, E_rows)
    with tr.tag("bfs"):
        Xc, info_c = _bfs_levels(inst, Eb)
    recs = tr.records
    algebra.last_info = {
        "frontier": {**info_a, **_tag_totals(recs, "frontier")},
        "dijkstra": {**info_b, "total": _tag_totals(recs, "dijkstra"),
                     "pop": _tag_totals(recs, "dijkstra/pop"),
                     "relax": _tag_totals(recs, "dijkstra/relax")},
        "bfs": {**info_c, **_tag_totals(recs, "bfs")},
    }
    return {"dist_frontier": {k[0]: int(v) for k, v in Xa.items()},
            "dist_dijkstra": {k[0]: int(v) for k, v in Xb.items()},
            "hops": {k[0]: int(v) for k, v in Xc.items()}}


def compare(a, b) -> bool:
    return (a["dist"] == b["dist_frontier"] and a["dist"] == b["dist_dijkstra"]
            and a["hops"] == b["hops"])


def extra_info(inst, res):
    li = algebra.last_info
    g = native.last_counts
    rd, rb = g["dijkstra_relaxations"], g["bfs_relaxations"]
    fa = li["frontier"]
    db = li["dijkstra"]
    bc = li["bfs"]
    relax_a = sum(fa["relax_per_round"])
    relax_b = db["relax"]["mul_ops"]
    relax_c = bc["mul_ops"]
    return {
        "golden": g,
        "frontier": {"relaxations": relax_a, "ratio_relax_vs_dijkstra": relax_a / rd,
                     "rounds": fa["rounds"], "delta_sizes": fa["delta_sizes"],
                     "relax_per_round": fa["relax_per_round"],
                     "candidates": fa["candidates"], "ratio_cand_vs_dijkstra": fa["candidates"] / rd},
        "dijkstra": {"relaxations": relax_b, "ratio_relax_vs_dijkstra": relax_b / rd,
                     "pops": db["pops"], "select_candidates": db["sum_open"],
                     "delete_candidates": db["pop"]["candidates"] - db["sum_open"],
                     "pop_candidates_select_plus_delete": db["pop"]["candidates"],
                     "ratio_pop_cand_vs_dijkstra": db["pop"]["candidates"] / rd,
                     "avg_open": db["sum_open"] / max(1, db["pops"]), "max_open": db["max_open"],
                     "relax_candidates": db["relax"]["candidates"],
                     "ratio_relax_cand_vs_dijkstra": db["relax"]["candidates"] / rd},
        "bfs": {"relaxations": relax_c, "ratio_relax_vs_bfs": relax_c / rb, "rounds": bc["rounds"],
                "delta_sizes": bc["delta_sizes"], "candidates": bc["candidates"],
                "ratio_cand_vs_bfs": bc["candidates"] / rb},
    }
