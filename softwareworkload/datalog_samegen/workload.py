"""Same-generation Datalog program on a random rooted tree, by semi-naive evaluation.

    sg(x, y) :- flat(x, y).
    sg(x, y) :- up(x, x1), sg(x1, y1), down(y1, y).

The recursive rule is linear in sg, so the single-delta semi-naive form is exact:
    Delta' = (up |x| Delta |x| down) - X.
The three-way join is executed as two combine calls. Join order "up_first" (the default):
    T = combine(up(x, x1), Delta(x1, y1), keep=(x, y1))      # (up, sg) pairs
    C = combine(T(x, y1), down(y1, y), keep=(x, y))          # (partial, down) pairs
Join order "down_first":
    T = combine(Delta(x1, y1), down(y1, y), keep=(x1, y))    # (sg, down) pairs
    C = combine(up(x, x1), T(x1, y), keep=(x, y))            # (up, partial) pairs

Goldens (plain Python sets, no algebra imports):
  * naive fixpoint: recompute the whole rule body from the full sg every round until no change;
  * semi-naive fixpoint: recompute the body from the previous round's new facts only.
Both count the same hotspot event as the algebra: one event per (up, sg) pair combined plus one
per (partial, down) pair combined, in the same loop nesting as join order "up_first".
`native()` returns the semi-naive count as native_ops (the ratio that matters) and exposes the
naive count through `native.last` / `extra_info`.
"""
from __future__ import annotations

import random

from algebra import Relation, combine, mask, update, rename, fixpoint, use_trace

NAME = "datalog_samegen"
CATEGORY = "logic"
SEMIRINGS = ["boolean"]
STRUCTURE = "sparse-fixpoint"
HOTSPOT_EVENT = ("join pairs: (up(x,x1), sg(x1,y1)) pairs combined plus (partial(x,y1), down(y1,y)) "
                 "pairs combined, one per AND")
# n nodes; level k holds min(b^k, remaining) nodes; `flat` ordered sibling pairs in total.
SCALES = {"small": {"n": 400, "b": 3, "flat": 120}, "large": {"n": 5000, "b": 4, "flat": 800}}
JOIN_ORDER = "up_first"


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    n, b, f = p["n"], p["b"], p["flat"]
    # levels: 1, b, b^2, ... nodes, the last level truncated so that exactly n nodes exist
    levels = [[0]]
    nxt = 1
    while nxt < n:
        size = min(len(levels[-1]) * b, n - nxt)
        levels.append(list(range(nxt, nxt + size)))
        nxt += size
    parent = [-1] * n
    children = [[] for _ in range(n)]
    for k in range(1, len(levels)):
        prev = levels[k - 1]
        for v in levels[k]:
            par = prev[rng.randrange(len(prev))]
            parent[v] = par
            children[par].append(v)
    # ordered sibling pairs (x, y), x != y, same parent, grouped by depth
    pairs_by_depth = []
    for k in range(1, len(levels)):
        pairs = []
        for par in levels[k - 1]:
            ch = children[par]
            for x in ch:
                for y in ch:
                    if x != y:
                        pairs.append((x, y))
        pairs_by_depth.append(pairs)
    total_pairs = sum(len(p_) for p_ in pairs_by_depth)
    # claude: decision start — flat pairs are drawn per depth with a quota proportional to the
    # number of sibling pairs at that depth (stratified sampling), because one flat pair at depth
    # k induces about b^(2(d-k)) sg facts and a uniform draw would make |sg| depend on whether a
    # single near-root pair happened to be drawn.
    quotas = [round(f * len(p_) / total_pairs) for p_ in pairs_by_depth]
    flat = set()
    for pairs, q in zip(pairs_by_depth, quotas):
        flat.update(rng.sample(pairs, min(q, len(pairs))))
    # claude: decision end
    up = [(v, parent[v]) for v in range(1, n)]            # child -> parent
    down = [(parent[v], v) for v in range(1, n)]          # parent -> child
    return {"n": n, "depth": len(levels) - 1, "level_sizes": [len(l) for l in levels],
            "up": up, "down": down, "flat": sorted(flat), "flat_quota_by_depth": quotas}


def _children(inst):
    ch = [[] for _ in range(inst["n"])]
    for par, v in inst["down"]:
        ch[par].append(v)
    return ch


def _golden_naive(children, flat):
    """Naive evaluation: X_{r+1} = flat  U  body(X_r), starting from X_0 = flat, until X_{r+1} = X_r.
    Every round re-derives every fact derivable from the full X_r."""
    X = set(flat)
    events = 0
    rounds = 0
    while True:
        rounds += 1
        body = set()
        for x1, y1 in X:
            cy = children[y1]
            for x in children[x1]:
                events += 1                       # (up, sg) pair
                for y in cy:
                    events += 1                   # (partial, down) pair
                    body.add((x, y))
        if body <= X:
            return X, events, rounds
        X |= body


def _golden_seminaive(children, flat):
    """Semi-naive evaluation with sets: the body is evaluated on last round's new facts only."""
    X = set(flat)
    D = set(flat)
    events = 0
    rounds = 0
    while D:
        rounds += 1
        new = set()
        for x1, y1 in D:
            cy = children[y1]
            for x in children[x1]:
                events += 1                       # (up, sg) pair
                for y in cy:
                    events += 1                   # (partial, down) pair
                    if (x, y) not in X:
                        new.add((x, y))
        X |= new
        D = new
    return X, events, rounds


def native(inst):
    """Both goldens; the returned count is the semi-naive one (the reference that matters)."""
    children = _children(inst)
    sg_naive, ev_naive, r_naive = _golden_naive(children, inst["flat"])
    sg_semi, ev_semi, r_semi = _golden_seminaive(children, inst["flat"])
    assert sg_naive == sg_semi, "the two goldens disagree"
    native.last = {"naive_events": ev_naive, "naive_rounds": r_naive,
                   "seminaive_events": ev_semi, "seminaive_rounds": r_semi}
    return sg_semi, ev_semi


def _algebra(inst, order: str):
    n = inst["n"]
    UP = Relation.from_set(("x", "x1"), inst["up"], {"x": n, "x1": n})
    DOWN = Relation.from_set(("y1", "y"), inst["down"], {"y1": n, "y": n})
    FLAT = Relation.from_set(("x", "y"), inst["flat"], {"x": n, "y": n})

    def step(X, D, k):
        D1 = rename(D, {"x": "x1", "y": "y1"})                  # Delta(x1, y1)
        # claude: decision start — the three-way join runs as two binary combines; "up_first"
        # is the default because its two candidate counts are exactly the two pair counts of the
        # golden's loop nesting. Measured over 10 small seeds: the second join is identical in
        # both orders and the first differs by -4.7% to +7.1%, random in sign.
        if order == "up_first":
            T = combine(UP, D1, keep=("x", "y1"))                # (x,x1) |x| (x1,y1) -> (x,y1)
            C = combine(T, DOWN, keep=("x", "y"))                # (x,y1) |x| (y1,y)  -> (x,y)
        else:
            T = combine(D1, DOWN, keep=("x1", "y"))              # (x1,y1) |x| (y1,y) -> (x1,y)
            C = combine(UP, T, keep=("x", "y"))                  # (x,x1) |x| (x1,y)  -> (x,y)
        # claude: decision end
        New = mask(C, X, negate=True)                            # drop already-known facts
        return update(X, New)

    X, info = fixpoint(step, FLAT)
    return X, info


def algebra(inst, order: str = JOIN_ORDER):
    X, info = _algebra(inst, order)
    algebra.last_info = info
    return set(X.keys())


def compare(a, b) -> bool:
    return a == b


def extra_info(inst, res):
    info = dict(algebra.last_info)
    out = {"rounds": info["rounds"], "delta_sizes": info["delta_sizes"], "sg_size": len(res),
           "depth": inst["depth"], "n_flat": len(inst["flat"]),
           "flat_quota_by_depth": inst["flat_quota_by_depth"], "join_order": JOIN_ORDER}
    out.update(getattr(native, "last", {}))
    if inst["n"] <= SCALES["small"]["n"]:
        # candidates of both join orders, each under its own trace (small scale only)
        for order in ("up_first", "down_first"):
            with use_trace() as tr:
                _algebra(inst, order)
            s = tr.summary()
            out[f"candidates_{order}"] = s["by_op"]["combine"]["candidates"]
            out[f"mul_ops_{order}"] = s["mul_ops"]
    return out
