"""Delete-relaxation heuristics h^add and h^max for STRIPS planning (as in Fast Downward).

A task has propositions 0..P-1, actions a with preconditions pre(a), add effects add(a) and an
integer cost(a) >= 1, and an initial state I. The relaxed reachability costs are the least
fixpoint of (propositions never reached stay +inf):

    h_add:  c(p) = min( c(p), min_{a : p in add(a)} [ cost(a) + SUM_{q in pre(a)} c(q) ] )
    h_max:  c(p) = min( c(p), min_{a : p in add(a)} [ cost(a) + MAX_{q in pre(a)} c(q) ] )

Algebra program, one semi-naive fixpoint round (c is the state, Delta the changed propositions):

    D1       = scan(Delta, const_one)                        # (q) -> one : support of Delta
    touched  = combine(pre(a,q), D1(q), keep=(a,))            # actions with a changed precondition
    pre_t    = combine(touched(a), pre(a,q))                  # every precondition of those actions
    J        = combine(pre_t(a,q), c(q))                      # (a,q) -> c(q), reachable q only
    pre_cost = reduce(J, keep=(a,), use_mul=True)             # fold over q: + (h_add) / max (h_max)
    missing  = mask(pre_t, c, negate=True)                    # (a,q) whose q is unreachable
    bad      = reduce(missing, keep=(a,))                     # actions with >= 1 unreachable pre
    ok       = mask(pre_cost, bad, negate=True)               # keep complete groups only
    ach      = combine(add(a,p)[cost(a)], ok(a), keep=(p,))   # cost(a) (x) pre_cost(a); min over a
    c, Delta = update(c, ach)

h_add runs entirely in MINPLUS (min, +): the precondition fold and the composition with cost(a)
are both +. h_max needs max for the fold but + for the composition with cost(a); no single
semiring in algebra/ has both, so the fold runs in BOTTLENECK (min, max) and two identity scans
re-type J into BOTTLENECK and pre_cost back into MINPLUS in every round. The same program run
purely in BOTTLENECK computes the bottleneck heuristic min_a max(cost(a), max_q c(q)), which is
not h^max; `native_bottleneck` / `algebra_bottleneck` expose that variant for the test that
documents the difference (docs/decompositions/planning_hadd.md, section 7).

Golden: the queue-based generalised Dijkstra with one counter of unsatisfied preconditions per
action (how Fast Downward computes h^add and h^max), plain Python, no algebra imports.

Hotspot event (counted identically on both sides, summed over h_add and h_max):
  * one (action, precondition) accumulation: the golden decrements the action's counter and folds
    c(q) into its accumulator once per reachable precondition; the algebra folds one J tuple;
  * one (action, add-effect) relaxation when the action fires: the golden relaxes c(p) once per
    add effect of a fired action; the algebra produces one candidate of the `ach` combine.
The golden performs each event exactly once (Dijkstra order); the round-based fixpoint re-touches
an action every time one of its preconditions improves, so the algebra count is larger.
"""
from __future__ import annotations

import heapq
import math
import operator
import random

from algebra import (MINPLUS, BOTTLENECK, Relation, combine, reduce, mask, update, scan, rename,
                     fixpoint, current_trace)

NAME = "planning_hadd"
CATEGORY = "planning"
SEMIRINGS = ["minplus", "bottleneck"]
STRUCTURE = "sparse-fixpoint"
HOTSPOT_EVENT = ("action evaluations: one (action, precondition) accumulation (a counter decrement "
                 "in the queue-based golden) plus one (action, add-effect) relaxation when the "
                 "action fires; summed over h_add and h_max")
# claude: decision start — "deep" is an extra scale that draws preconditions from a rank window of
# 50 just below the add effect (long causal chains, more fixpoint rounds); it exists so that the
# sensitivity of the ratio to task depth reported in the decomposition document is reproducible
# through the harness. small/large use the plain uniform generator.
SCALES = {"small": {"P": 400, "A": 1500, "cost_max": 10, "window": None},
          "large": {"P": 4000, "A": 15000, "cost_max": 10, "window": None},
          "deep": {"P": 4000, "A": 15000, "cost_max": 10, "window": 50}}
# claude: decision end

INF = math.inf


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    P, A, cmax, window = p["P"], p["A"], p["cost_max"], p["window"]
    # claude: decision start — every proposition gets a random rank; an action's preconditions have
    # lower rank than its add effects, so the task is a random DAG in which most propositions are
    # reachable. The 5% lowest ranks are the initial state, the 5% highest ranks have no achiever
    # ("dead"), and 3% of the actions carry one dead precondition so that the complete-group check
    # (an action with an unreachable precondition must stay +inf) is exercised on every instance.
    perm = list(range(P))
    rng.shuffle(perm)
    n_init = max(1, P // 20)
    n_dead = max(1, P // 20)
    init = sorted(perm[:n_init])
    dead_ranks = range(P - n_dead, P)
    actions = []
    for _ in range(A):
        n_add = rng.randint(1, 2)
        add_ranks = rng.sample(range(n_init, P - n_dead), n_add)
        lo = min(add_ranks)
        n_pre = rng.randint(1, 3)
        pre = set()
        if rng.random() < 0.03:
            pre.add(perm[rng.choice(dead_ranks)])
        lo_min = 0 if window is None else max(0, lo - window)
        while len(pre) < n_pre:
            pre.add(perm[rng.randrange(lo_min, lo)])  # rank < lo; uniform generator reaches I
        actions.append({"pre": sorted(pre), "add": sorted(perm[r] for r in add_ranks),
                        "cost": rng.randint(1, cmax)})
    # claude: decision end
    return {"P": P, "A": A, "init": init, "actions": actions}


# ---------------------------------------------------------------------------------------------
# Golden: generalised Dijkstra with unsatisfied-precondition counters (Fast Downward style).
# ---------------------------------------------------------------------------------------------
def _native_heuristic(inst, fold, comp):
    """fold(acc, c(q)) accumulates a precondition cost into an action (operator.add for h_add,
    max for h_max); comp(cost(a), acc) composes the action cost (operator.add for h_add/h_max,
    max for the bottleneck variant). Returns (costs, counter decrements, relaxations)."""
    P, acts = inst["P"], inst["actions"]
    pre_of = [[] for _ in range(P)]
    for a, act in enumerate(acts):
        for q in act["pre"]:
            pre_of[q].append(a)
    counter = [len(act["pre"]) for act in acts]
    acc = [0] * len(acts)             # fold identity: 0 for +, and for max since costs are >= 0
    c = [INF] * P
    heap = []
    for q in inst["init"]:
        c[q] = 0
        heap.append((0, q))
    heapq.heapify(heap)
    done = [False] * P
    decrements = 0
    relaxations = 0

    def fire(a):
        nonlocal relaxations
        val = comp(acts[a]["cost"], acc[a])
        for p in acts[a]["add"]:
            relaxations += 1
            if val < c[p]:
                c[p] = val
                heapq.heappush(heap, (val, p))

    for a in range(len(acts)):        # actions without preconditions fire immediately
        if counter[a] == 0:
            fire(a)
    while heap:
        cq, q = heapq.heappop(heap)
        if done[q]:
            continue                   # stale queue entry
        done[q] = True
        for a in pre_of[q]:
            acc[a] = fold(acc[a], cq)
            counter[a] -= 1
            decrements += 1
            if counter[a] == 0:
                fire(a)
    return c, decrements, relaxations


def native(inst):
    h_add, dec_a, rel_a = _native_heuristic(inst, operator.add, operator.add)
    h_max, dec_m, rel_m = _native_heuristic(inst, max, operator.add)
    native.last_info = {"decrements_add": dec_a, "relaxations_add": rel_a,
                        "decrements_max": dec_m, "relaxations_max": rel_m}
    return {"h_add": h_add, "h_max": h_max}, dec_a + rel_a + dec_m + rel_m


def native_bottleneck(inst):
    """Diagnostic golden for the bottleneck heuristic min_a max(cost(a), max_q c(q)): what the
    algebra program computes when it is run unchanged in the BOTTLENECK semiring."""
    c, dec, rel = _native_heuristic(inst, max, max)
    return c, dec + rel


# ---------------------------------------------------------------------------------------------
# Algebra program
# ---------------------------------------------------------------------------------------------
def _algebra_heuristic(inst, fold_sr, comp_sr):
    """One heuristic. comp_sr carries c, pre, add and the composition with cost(a); fold_sr is
    the semiring whose mul folds the precondition costs (identity casts when the two differ)."""
    P, A, acts = inst["P"], inst["A"], inst["actions"]
    one = comp_sr.one
    pre = Relation(("a", "q"), comp_sr,
                   {(a, q): one for a, act in enumerate(acts) for q in act["pre"]},
                   {"a": A, "q": P})
    add_rel = Relation(("a", "p"), comp_sr,
                       {(a, p): act["cost"] for a, act in enumerate(acts) for p in act["add"]},
                       {"a": A, "p": P})
    c0 = Relation(("q",), comp_sr, {(q,): 0 for q in inst["init"]}, {"q": P})
    tr = current_trace()
    cast = fold_sr is not comp_sr

    def const_one(k, v):               # scan fn: keep the key, value -> semiring one
        return [(k, one)]

    def identity(k, v):                # scan fn: re-type a relation into another semiring
        return [(k, v)]

    def step(X, D, r):
        with tr.tag("touch"):
            D1 = scan(D, const_one, fn_name="const_one")
            touched = combine(pre, D1, keep=("a",))
        with tr.tag("gather"):
            pre_t = combine(touched, pre)
        with tr.tag("fold"):
            J = combine(pre_t, X)
            # claude: decision start — h^max folds preconditions with max but composes cost(a) with
            # +; the fold runs in BOTTLENECK and the relation is re-typed with identity scans each
            # round because combine/reduce require both operands in one semiring object.
            if cast:
                J = scan(J, identity, fn_name="cast_" + fold_sr.name, sr=fold_sr)
            pre_cost = reduce(J, keep=("a",), use_mul=True)
            if cast:
                pre_cost = scan(pre_cost, identity, fn_name="cast_" + comp_sr.name, sr=comp_sr)
            # claude: decision end
        with tr.tag("complete"):
            # claude: decision start — the complete-group check is an antijoin on the support of c
            # (a precondition outside supp(c) is unreachable) rather than a COUNTING comparison
            # with |pre(a)|: it needs no second semiring and no re-keying scan, and it costs one
            # support probe per precondition of a touched action, the same order as the fold.
            missing = mask(pre_t, X, negate=True)
            bad = reduce(missing, keep=("a",))
            ok = mask(pre_cost, bad, negate=True)
            # claude: decision end
        with tr.tag("fire"):
            ach = combine(add_rel, ok, keep=("p",))
        with tr.tag("update"):
            return update(X, rename(ach, {"p": "q"}))

    X, info = fixpoint(step, c0)
    return [X.get((p,), INF) for p in range(P)], info


def _phase_sums(records, prefix):
    out = {}
    for r in records:
        if not r.tag.startswith(prefix + "/"):
            continue
        phase = r.tag[len(prefix) + 1:]
        d = out.setdefault(phase, {"candidates": 0, "mul_ops": 0, "add_ops": 0})
        d["candidates"] += r.candidates
        d["mul_ops"] += r.mul_ops
        d["add_ops"] += r.add_ops
    return out


def algebra(inst):
    tr = current_trace()
    with tr.tag("h_add"):
        h_add, info_add = _algebra_heuristic(inst, MINPLUS, MINPLUS)
    with tr.tag("h_max"):
        h_max, info_max = _algebra_heuristic(inst, BOTTLENECK, MINPLUS)
    algebra.last_info = {
        "rounds_add": info_add["rounds"], "rounds_max": info_max["rounds"],
        "delta_sizes_add": info_add["delta_sizes"], "delta_sizes_max": info_max["delta_sizes"],
        "phases_add": _phase_sums(tr.records, "h_add"),
        "phases_max": _phase_sums(tr.records, "h_max"),
    }
    return {"h_add": h_add, "h_max": h_max}


def algebra_bottleneck(inst):
    """Diagnostic: the same program run unchanged in BOTTLENECK (min, max)."""
    costs, info = _algebra_heuristic(inst, BOTTLENECK, BOTTLENECK)
    return costs


def compare(a, b) -> bool:
    return (len(a["h_add"]) == len(b["h_add"]) and len(a["h_max"]) == len(b["h_max"])
            and all(x == y for x, y in zip(a["h_add"], b["h_add"]))
            and all(x == y for x, y in zip(a["h_max"], b["h_max"])))


def extra_info(inst, res):
    li = algebra.last_info
    ni = native.last_info
    reach = sum(1 for v in res["h_add"] if v != INF)
    pa, pm = li["phases_add"], li["phases_max"]
    return {
        "P": inst["P"], "A": inst["A"], "reachable": reach,
        "rounds_add": li["rounds_add"], "rounds_max": li["rounds_max"],
        "delta_sizes_add": li["delta_sizes_add"], "delta_sizes_max": li["delta_sizes_max"],
        "native_decrements_add": ni["decrements_add"], "native_relaxations_add": ni["relaxations_add"],
        "native_decrements_max": ni["decrements_max"], "native_relaxations_max": ni["relaxations_max"],
        # re-touch factor: (pre, Delta) pairs of the algebra per counter decrement of the golden
        "retouch_factor_add": pa["touch"]["mul_ops"] / max(1, ni["decrements_add"]),
        "retouch_factor_max": pm["touch"]["mul_ops"] / max(1, ni["decrements_max"]),
        # fire factor: ach candidates of the algebra per relaxation of the golden
        "fire_factor_add": pa["fire"]["mul_ops"] / max(1, ni["relaxations_add"]),
        "fire_factor_max": pm["fire"]["mul_ops"] / max(1, ni["relaxations_max"]),
        "phase_mul_ops_add": {k: v["mul_ops"] for k, v in pa.items()},
        "phase_candidates_add": {k: v["candidates"] for k, v in pa.items()},
        "phase_mul_ops_max": {k: v["mul_ops"] for k, v in pm.items()},
        "phase_candidates_max": {k: v["candidates"] for k, v in pm.items()},
        "control_residue": "none beyond the fixpoint termination test (|Delta| = 0 per round)",
    }
