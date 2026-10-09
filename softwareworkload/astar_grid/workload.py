"""A* on a 2-D grid with random obstacles, 4-connected, unit costs, Manhattan heuristic.

Algebra program (select loop, one select per expanded node):
    best_f  = select_k(f_open, 1)                        # pop the open node with the smallest f
    f_open  = mask(f_open, best_f, negate=True)          # point delete of the popped node
    best_g  = combine(best_f, scan(best_f, neg_h))       # g = f + (-h): MINPLUS mul with the looked-up heuristic
    closed  = update(closed, best_g)                     # expanded set
    cand_g  = combine(best_g, E_row[u], keep=(m,))       # relax the grid edges (u, m): g + 1
    cand_g  = mask(cand_g, closed, negate=True)          # never re-open an expanded node (consistent h)
    f_cand  = combine(cand_g, scan(cand_g, lookup_h))    # f = g + h on the changed candidates only
    f_open  = update(f_open, f_cand)                     # min-merge; stop when the goal is selected
f is maintained incrementally: the heuristic is added to the candidates produced by an expansion
(at most 4 per step), not recomputed over the whole open list. The popped node's g is recovered
from f with the inverse of the (min,+) mul, g = f + (-h).

Golden: heapq A* with (f, node) ordering and lazy deletion; it counts node expansions and edge
relaxations (every free neighbour scanned from an expanded node). HOTSPOT_EVENT and native_ops
are the edge relaxations; expansions are reported in `extra` on both sides.

Variant (a) of graph_bfs_sssp (global (min,+) frontier relaxation, Bellman-Ford) runs on the same
grid inside a nested trace so that its counts appear in `extra` (relaxations vs A*) without
entering the harness ratio of the A* program: it is the plan's section 4.5 counter-example
(a global relaxation loses the heuristic's pruning), not a second solution of the workload.
"""
from __future__ import annotations

import heapq
import random
from collections import deque

from algebra import (MINPLUS, Relation, combine, mask, update, rename, select_k, scan, fixpoint,
                     current_trace, use_trace)

NAME = "astar_grid"
CATEGORY = "search"
SEMIRINGS = ["minplus"]
STRUCTURE = "select-loop"
HOTSPOT_EVENT = "edge relaxations: free 4-neighbours scanned from an expanded node"
SCALES = {"small": {"H": 60, "W": 60}, "large": {"H": 250, "W": 250}}
OBSTACLE_DENSITY = 0.3
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    H, W = p["H"], p["W"]
    free = [rng.random() >= OBSTACLE_DENSITY for _ in range(H * W)]

    def nbrs(u):
        r, c = divmod(u, W)
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and free[rr * W + cc]:
                yield rr * W + cc

    # claude: decision start — start and goal lie in the largest connected component (start = its top-left-most cell, goal = its cell farthest from start by Manhattan distance) so a path exists and is long
    comp_of = [-1] * (H * W)
    comps = []
    for u in range(H * W):
        if free[u] and comp_of[u] < 0:
            cid = len(comps)
            comp_of[u] = cid
            dq = deque([u])
            members = [u]
            while dq:
                x = dq.popleft()
                for y in nbrs(x):
                    if comp_of[y] < 0:
                        comp_of[y] = cid
                        dq.append(y)
                        members.append(y)
            comps.append(members)
    big = max(comps, key=len)
    start = min(big, key=lambda u: (u // W + u % W, u // W))
    sr, sc = divmod(start, W)
    goal = max(big, key=lambda u: (abs(u // W - sr) + abs(u % W - sc), -u))
    # claude: decision end
    edges = [(u, v) for u in range(H * W) if free[u] for v in nbrs(u)]
    return {"H": H, "W": W, "free": free, "start": start, "goal": goal, "edges": edges,
            "component_size": len(big)}


# ---------------------------------------------------------------------------------------------
def native(inst):
    """heapq A*: (f, node) ordering, lazy deletion, closed set (consistent heuristic, no reopen).
    Neighbours are computed from the coordinates, independently of the edge list."""
    H, W, free, s, goal = inst["H"], inst["W"], inst["free"], inst["start"], inst["goal"]
    gr, gc = divmod(goal, W)

    def h(u):
        return abs(u // W - gr) + abs(u % W - gc)

    g = {s: 0}
    heap = [(h(s), s)]
    closed = set()
    expansions = 0
    relax = 0
    cost = None
    while heap:
        f, u = heapq.heappop(heap)
        if u in closed or f > g[u] + h(u):
            continue
        closed.add(u)
        expansions += 1
        if u == goal:
            cost = g[u]
            break
        r, c = divmod(u, W)
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                v = rr * W + cc
                if free[v]:
                    relax += 1
                    if v in closed:
                        continue
                    ng = g[u] + 1
                    if ng < g.get(v, 1 << 60):
                        g[v] = ng
                        heapq.heappush(heap, (ng + h(v), v))
    native.last_counts = {"expansions": expansions, "relaxations": relax}
    return {"cost": cost, "expansions": expansions}, relax


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


def _astar(inst, E_rows, h):
    n_cells = inst["H"] * inst["W"]
    s, goal = inst["start"], inst["goal"]
    dom = {"n": n_cells}
    closed = Relation(("n",), MINPLUS, None, dom)
    f_open = Relation(("n",), MINPLUS, {(s,): float(h(s))}, dom)      # f(s) = g(s) + h(s) = h(s)
    empty_row = Relation(("n", "m"), MINPLUS, None, {"n": n_cells, "m": n_cells})
    tr = current_trace()
    expansions = 0
    sum_open = 0
    max_open = 0
    cost = None
    while len(f_open) > 0:
        sum_open += len(f_open)
        max_open = max(max_open, len(f_open))
        with tr.tag("pop"):
            # claude: decision start — key=value gives the same order as the semiring's `better` (min f, ties by node id) and is the cheaper interpreter path; the recorded candidates are identical
            best_f = select_k(f_open, 1, key=lambda k, v: v)          # candidates = |open|
            # claude: decision end
            u = next(iter(best_f.keys()))[0]
            expansions += 1
            if u == goal:
                cost = int(best_f.get((u,)))                          # h(goal) = 0, so f = g
                break
            f_open = mask(f_open, best_f, negate=True)                # point delete
        with tr.tag("expand"):
            neg_h = scan(best_f, lambda k, v: [(k, -float(h(k[0])))], fn_name="neg_h")
            best_g = combine(best_f, neg_h)                           # g = f + (-h)
            closed, _ = update(closed, best_g)
            cand_g = combine(best_g, E_rows.get(u, empty_row), keep=("m",))   # relaxations
            cand_g = rename(cand_g, {"m": "n"})
            cand_g = mask(cand_g, closed, negate=True)                # no reopening
        with tr.tag("heuristic"):
            # claude: decision start — f = g + h is formed for the changed candidates only (h looked up by key with a scan) instead of re-joining the whole open list with H every step
            h_c = scan(cand_g, lambda k, v: [(k, float(h(k[0])))], fn_name="lookup_h")
            f_c = combine(cand_g, h_c)                                # f = g + h
            # claude: decision end
        with tr.tag("open_update"):
            f_open, _ = update(f_open, f_c)
    return cost, {"expansions": expansions, "sum_open": sum_open, "max_open": max_open}


def _bellman_ford_frontier(inst, E_rows):
    """Variant (a) of graph_bfs_sssp on the grid: global (min,+) frontier relaxation from the
    start until no distance changes. The frontier's edge rows are gathered by a scan (index
    access, at most 4 edges per cell) because the interpreter's combine streams the larger
    operand and this fixpoint runs for hundreds of rounds."""
    n_cells = inst["H"] * inst["W"]
    s = inst["start"]
    dom = {"n": n_cells, "m": n_cells}
    X0 = Relation(("n",), MINPLUS, {(s,): 0.0}, {"n": n_cells})
    empty = {}
    relax_per_round = []

    def gather_row(k, v):
        row = E_rows.get(k[0])
        return row.items() if row is not None else empty.items()

    def step(X, D, k):
        rows = scan(D, gather_row, schema=("n", "m"), fn_name="gather_row", domain=dom)
        cand = combine(D, rows, keep=("m",))
        relax_per_round.append(current_trace().records[-1].candidates)
        cand = rename(cand, {"m": "n"})
        return update(X, cand)

    X, info = fixpoint(step, X0)
    info["relax_per_round"] = relax_per_round
    return X, info


def algebra(inst):
    H, W, goal = inst["H"], inst["W"], inst["goal"]
    n_cells = H * W
    gr, gc = divmod(goal, W)

    def h(u):
        return abs(u // W - gr) + abs(u % W - gc)

    dom = {"n": n_cells, "m": n_cells}
    rows: dict[int, dict] = {}
    for u, v in inst["edges"]:
        rows.setdefault(u, {})[(u, v)] = 1.0
    E_rows = {u: Relation(("n", "m"), MINPLUS, d, dom) for u, d in rows.items()}
    tr = current_trace()
    with tr.tag("astar"):
        cost, info_a = _astar(inst, E_rows, h)
    recs = tr.records
    # the Bellman-Ford counter-example runs in its own trace (see the module docstring)
    # claude: decision start — the Bellman-Ford variant is a measurement of the plan's section 4.5 counter-example, not a second solution, so it runs in a nested trace and is reported in extra only
    with use_trace() as tr_bf:
        Xbf, info_bf = _bellman_ford_frontier(inst, E_rows)
    # claude: decision end
    bf_cost = int(Xbf.get((goal,)))
    algebra.last_info = {
        "astar": {**info_a, "total": _tag_totals(recs, "astar"),
                  "pop": _tag_totals(recs, "astar/pop"), "expand": _tag_totals(recs, "astar/expand"),
                  "heuristic": _tag_totals(recs, "astar/heuristic"),
                  "open_update": _tag_totals(recs, "astar/open_update")},
        "bellman_ford": {"rounds": info_bf["rounds"], "delta_sizes": info_bf["delta_sizes"],
                         "relaxations": sum(info_bf["relax_per_round"]),
                         "cells_reached": len(Xbf), "trace": tr_bf.summary(), "cost": bf_cost},
    }
    return {"cost": cost, "expansions": info_a["expansions"], "bf_cost": bf_cost}


def compare(a, b) -> bool:
    return a["cost"] is not None and a["cost"] == b["cost"] and b["bf_cost"] == a["cost"]


def extra_info(inst, res):
    li = algebra.last_info
    g = native.last_counts
    a = li["astar"]
    bf = li["bellman_ford"]
    relax_a = a["expand"]["mul_ops"] - a["expansions"] + 1     # expand muls minus the g = f + (-h) op per pop (goal pop has no relax)
    return {
        "golden": g,
        "component_size": inst["component_size"],
        "astar": {"expansions": a["expansions"], "expansions_ratio": a["expansions"] / g["expansions"],
                  "relaxations": relax_a, "ratio_relax_vs_golden": relax_a / g["relaxations"],
                  "select_candidates": a["sum_open"],
                  "delete_candidates": a["pop"]["candidates"] - a["sum_open"],
                  "pop_candidates_select_plus_delete": a["pop"]["candidates"],
                  "ratio_pop_cand_vs_relax": a["pop"]["candidates"] / g["relaxations"],
                  "avg_open": a["sum_open"] / max(1, a["expansions"]), "max_open": a["max_open"],
                  "heuristic_mul_ops": a["heuristic"]["mul_ops"],
                  "expand_candidates": a["expand"]["candidates"],
                  "open_update_candidates": a["open_update"]["candidates"]},
        "bellman_ford": {"relaxations": bf["relaxations"], "ratio_relax_vs_astar_golden": bf["relaxations"] / g["relaxations"],
                         "rounds": bf["rounds"], "cells_reached": bf["cells_reached"],
                         "trace_candidates": bf["trace"]["candidates"], "trace_mul_ops": bf["trace"]["mul_ops"],
                         "cost": bf["cost"]},
    }
