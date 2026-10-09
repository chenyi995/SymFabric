"""Non-linear transitive closure by semi-naive evaluation.

    path(x, z) :- edge(x, z).
    path(x, z) :- path(x, y), path(y, z).

The rule is quadratic in path, so the semi-naive delta needs two join terms (Ullman's rewriting):
    Delta_{i+1} = ( Delta_i |x| X_i  U  X_{i-1} |x| Delta_i ) - X_i,      X_i = X_{i-1} U Delta_i.
The first term covers Delta |x| Delta and Delta |x| X_old; the second covers X_old |x| Delta.

Algebra program (one fixpoint round; X_old is the state before this round's delta was merged):
    C1       = combine(rename(Delta, j->m), rename(X, i->m), keep=(i, j))       # Delta |x| X_new
    C2       = combine(rename(X_old, j->m), rename(Delta, i->m), keep=(i, j))   # X_old |x| Delta
    U, _     = update(C1, C2)                                                    # OR-union
    New      = mask(U, X, negate=True)
    X, Delta = update(X, New)

Golden: BFS from every source vertex (the same reference algorithm as datalog_tc); it counts one
(path(s,x), edge(x,y)) pair examined = one AND. The algebra counts one (path(x,y), path(y,z))
pair examined = one AND. Same kind of event, different program: the ratio measures how much more
work the non-linear program does in exchange for its fewer rounds.

Two diagnostic single-delta variants exist for the decomposition document (not the primary
program):
  * "relaxed": Delta |x| X_new only. Complete for transitive closure, because E is a subset of X
    and every path has a last-edge decomposition, but it is not a semi-naive rewriting of the
    rule: it degenerates into the right-linear program (Delta |x| E) plus redundant work.
  * "strict": Delta |x| X_old only, the linear rewriting applied blindly. Wrong: it misses
    Delta |x| Delta. On the edges a->b, b->c it derives nothing.
"""
from __future__ import annotations

import random
from collections import deque

from algebra import BOOLEAN, Relation, combine, mask, update, rename, fixpoint, use_trace
from softwareworkload.datalog_tc import workload as _linear

NAME = "datalog_nonlinear_tc"
CATEGORY = "logic"
SEMIRINGS = ["boolean"]
STRUCTURE = "sparse-fixpoint"
HOTSPOT_EVENT = ("join pairs: (path(x,y), path(y,z)) pairs examined by the algebra versus "
                 "(path(s,x), edge(x,y)) pairs examined by the BFS golden, one per AND")
# claude: decision start — the same generator as datalog_tc (random digraph plus a chain through
# n // 8 vertices) at smaller n, because the non-linear program's work grows like |TC| * n on a
# dense closure; datalog_tc's own sizes would take minutes in pure Python.
SCALES = {"small": {"n": 100, "m": 300}, "large": {"n": 250, "m": 750}}
# claude: decision end


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    return random_digraph(p["n"], p["m"], seed)


def random_digraph(n: int, m: int, seed: int):
    """Identical to softwareworkload/datalog_tc/workload.py::make_instance, parametrised by n, m."""
    rng = random.Random(seed)
    edges = set()
    chain = n // 8
    for i in range(chain - 1):
        edges.add((i, i + 1))
    while len(edges) < m:
        edges.add((rng.randrange(n), rng.randrange(n)))
    return {"n": n, "edges": sorted(edges)}


def native(inst):
    """BFS closure from every source; one relaxation per out-edge scanned from a vertex already
    known reachable from the source (the source's own out-edges are base facts)."""
    n = inst["n"]
    succ = [[] for _ in range(n)]
    for a, b in inst["edges"]:
        succ[a].append(b)
    closure = set()
    relax = 0
    for s in range(n):
        seen = set(succ[s])
        frontier = deque(seen)
        while frontier:
            x = frontier.popleft()
            for y in succ[x]:
                relax += 1
                if y not in seen:
                    seen.add(y)
                    frontier.append(y)
        closure.update((s, t) for t in seen)
    native.last_result = closure
    return closure, relax


def _algebra(inst, variant: str = "two_term"):
    n = inst["n"]
    dom = {"i": n, "j": n}
    E = Relation.from_set(("i", "j"), inst["edges"], dom)
    prev = [Relation(("i", "j"), BOOLEAN, None, dom)]          # X_old, empty before round 0

    def step(X, D, k):
        Xold = prev[0]
        Dm = rename(D, {"j": "m"})
        if variant == "two_term":
            # claude: decision start — X_old is carried in a closure variable; the fixpoint
            # driver only passes (X, Delta) and X_old = X minus Delta would cost a full antijoin
            # pass over X every round.
            C1 = combine(Dm, rename(X, {"i": "m"}), keep=("i", "j"))         # Delta |x| X_new
            C2 = combine(rename(Xold, {"j": "m"}), rename(D, {"i": "m"}), keep=("i", "j"))  # X_old |x| Delta
            U, _ = update(C1, C2)                                           # union
            # claude: decision end
        elif variant == "relaxed":
            U = combine(Dm, rename(X, {"i": "m"}), keep=("i", "j"))          # Delta |x| X_new only
        elif variant == "strict":
            U = combine(Dm, rename(Xold, {"i": "m"}), keep=("i", "j"))       # Delta |x| X_old only
        else:
            raise ValueError(variant)
        New = mask(U, X, negate=True)
        prev[0] = X
        return update(X, New)

    X, info = fixpoint(step, E)
    return X, info


def algebra(inst):
    X, info = _algebra(inst, "two_term")
    algebra.last_info = info
    return set(X.keys())


def compare(a, b) -> bool:
    return a == b


def _run_traced(fn, inst):
    with use_trace() as tr:
        out = fn(inst)
    return out, tr.summary()


def extra_info(inst, res):
    info = algebra.last_info
    golden = native.last_result
    out = {"rounds": info["rounds"], "delta_sizes": info["delta_sizes"], "tc_size": len(res),
           "n_edges": len(inst["edges"])}
    # the linear program of datalog_tc on the same instance
    lin, s = _run_traced(_linear.algebra, inst)
    out["linear_rounds"] = _linear.algebra.last_info["rounds"]
    out["linear_candidates"] = s["candidates"]
    out["linear_mul_ops"] = s["mul_ops"]
    out["linear_ok"] = lin == golden
    # single-delta diagnostics
    for variant in ("relaxed", "strict"):
        (X, vinfo), s = _run_traced(lambda i, v=variant: _algebra(i, v), inst)
        out[f"{variant}_rounds"] = vinfo["rounds"]
        out[f"{variant}_mul_ops"] = s["mul_ops"]
        out[f"{variant}_candidates"] = s["candidates"]
        out[f"{variant}_tc_size"] = len(X)
        out[f"{variant}_ok"] = set(X.keys()) == golden
    return out
