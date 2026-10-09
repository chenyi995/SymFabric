"""Datalog transitive closure (linear recursion) by semi-naive evaluation.

    path(i, j) :- edge(i, j).
    path(i, k) :- path(i, j), edge(j, k).

Algebra program (one fixpoint round):
    C     = combine(rename(Delta, j->m), rename(E, i->m), keep=(i, j))   # Delta join E, project
    New   = mask(C, X, negate=True)                                        # drop known facts
    X, Delta = update(X, New)                                              # OR-merge, new delta

Golden: BFS from every source vertex. Both sides count "edge relaxations": the reference counts
every out-edge scanned from a dequeued vertex; the algebra counts every matching (Delta, E) pair.
"""
from __future__ import annotations

import random
from collections import deque

from algebra import BOOLEAN, Relation, combine, mask, update, rename, fixpoint

NAME = "datalog_tc"
CATEGORY = "logic"
SEMIRINGS = ["boolean"]
STRUCTURE = "sparse-fixpoint"
HOTSPOT_EVENT = "edge relaxations: (path(i,j), edge(j,k)) pairs examined, one per AND"
SCALES = {"small": {"n": 200, "m": 600}, "large": {"n": 2500, "m": 10000}}


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    n, m = p["n"], p["m"]
    edges = set()
    # a long chain through the first n // 8 vertices forces many fixpoint rounds
    chain = n // 8
    for i in range(chain - 1):
        edges.add((i, i + 1))
    while len(edges) < m:
        edges.add((rng.randrange(n), rng.randrange(n)))
    return {"n": n, "edges": sorted(edges)}


def native(inst):
    """BFS from every source. Counts one relaxation per out-edge scanned from a vertex that is
    already known to be reachable from the source (the base edges out of the source itself are
    facts, not relaxations; the algebra's round 0 joins Delta_0 = E with E in the same way)."""
    n = inst["n"]
    adj = [[] for _ in range(n)]
    for a, b in inst["edges"]:
        adj[a].append(b)
    reach = set()
    relax = 0
    for s in range(n):
        visited = set()
        dq = deque()
        for t in adj[s]:
            if t not in visited:
                visited.add(t)
                dq.append(t)
        while dq:
            x = dq.popleft()
            for y in adj[x]:
                relax += 1
                if y not in visited:
                    visited.add(y)
                    dq.append(y)
        reach.update((s, t) for t in visited)
    return reach, relax


def algebra(inst):
    n = inst["n"]
    E = Relation.from_set(("i", "j"), inst["edges"], {"i": n, "j": n})
    Em = rename(E, {"i": "m"})

    def step(X, D, k):
        C = combine(rename(D, {"j": "m"}), Em, keep=("i", "j"))
        New = mask(C, X, negate=True)
        return update(X, New)

    X, info = fixpoint(step, E)
    algebra.last_info = info
    return set(X.keys())


def compare(a, b) -> bool:
    return a == b


def extra_info(inst, res):
    return {"rounds": algebra.last_info["rounds"], "tc_size": len(res)}
