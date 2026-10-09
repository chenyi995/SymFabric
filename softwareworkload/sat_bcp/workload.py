"""Boolean constraint propagation (unit propagation) for Q cubes over one shared clause database.

Instance: random k-SAT CNF (n variables, m clauses; k = 3 mostly, some 4- and 5-literal
clauses) and Q cubes, each a partial assignment of c literals on distinct variables. Literal
encoding: variable v -> 2v (positive) and 2v + 1 (negative); negation is l ^ 1.

Golden: a standard two-watched-literal unit propagator per cube (trail with a propagation head,
watch lists initialised from the first two literals of every clause, propagate to the fixpoint
or to the first conflict). Hotspot event: clause visits, one per watched clause inspected because
its watched literal became false. Literal visits (value look-ups inside inspected clauses) are
the secondary event.

Algebra, two programs over the same relations:
  (a) "scan" form, `algebra_scan`: every round recomputes, for every cube and every clause that
      contains a false literal, the false-literal count and the satisfied flag from asg(q, l):
        false_count(q,c,w) = combine_COUNTING(clause(c,w,l), nasg(q,l))    nasg = negated asg
        sat(q,c)           = combine_BOOLEAN(clause(c,w,l), asg(q,l))
        unit  = filter(false_count == w-1) antijoin sat;   conflict = filter(false_count == w)
        implied(q,l)       = the single non-false literal of each unit clause
        asg, Delta         = update(asg, implied minus conflicted cubes)
      The clause width w travels in the key of clause(c,w,l) so that the filter predicates are
      pure key/value tests. Iterated to the fixpoint over all Q cubes at once.
  (b) "watched" form, `algebra` (the harness's primary program): a watch relation with two
      watched literals per clause per cube; each round only the clauses touched by the newly
      falsified literals are examined,
        touched(q,l,c) = combine(Delta_false(q,l), watch(q,l,c)),
      and the clause status is computed as in (a) but restricted to touched clauses. A watch
      that fell is replaced by the smallest non-false, non-watched literal of the clause: a
      reduce under the (min,+) semiring with the literal id as the value, a grouped select-1;
      clauses whose two watches fell in the same round get a second replacement the same way.
      Watches on false literals are never removed: a literal becomes false at most once, so such
      a watch can never be touched again (the golden also leaves a watch on a false literal when
      the other watch is true). The primary program keeps one materialised relation
      watch(q,c,l) with Q * 2m initial tuples. `algebra_watched(inst, materialized=False)` is
      the factored variant: W0(c,l), the initial watches shared by all cubes, plus WMOV(q,c,l),
      the per-cube moved watches; the union of the two is watch(q,c,l). Both variants perform
      the same combine candidates (mul_ops); the factored one adds a union and one antijoin pass
      per round (about 3% more candidates) and needs Q times less state.

Correctness: per cube, golden and algebra must agree on the conflict verdict; for cubes without
a conflict they must agree on the full set of true literals. Unit propagation is confluent:
without a conflict the fixpoint assignment is unique whatever the propagation order, and the
existence of a conflict is order-independent (an all-false clause under a partial assignment
stays all-false under every extension of it), but the partial assignment at which the conflict
is detected is not, so on conflict only the verdict is compared.
"""
from __future__ import annotations

import random

from algebra import (BOOLEAN, COUNTING, MINPLUS, Relation, combine, reduce, filter as afilter,
                     mask, update, scan, fixpoint, use_trace)

NAME = "sat_bcp"
CATEGORY = "logic"
SEMIRINGS = ["boolean", "counting", "minplus"]
STRUCTURE = "frontier-fixpoint"
HOTSPOT_EVENT = ("clause visits: a watched clause inspected because one of its watched literals "
                 "became false")
# claude: decision start — cube size c raised from the plan's 12 / 20 to 34 / 140: at c = 12 / 20
# propagation was trivial (measured seeds 0-2: 0-1 conflicting cubes, 0.3-2.4 implied literals per
# cube); at 34 / 140 some cubes conflict (3-7 of 16, 13-21 of 64) and the rest propagate tens of
# literals (means 16-21 and 36-38), which is what the plan asks for.
SCALES = {"small": {"n": 300, "m": 1200, "Q": 16, "c": 34},
          "large": {"n": 2000, "m": 8000, "Q": 64, "c": 140}}
# claude: decision end


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    n, m, Q, c = p["n"], p["m"], p["Q"], p["c"]
    clauses = []
    for _ in range(m):
        # claude: decision start — clause width mixture 85% k=3, 10% k=4, 5% k=5 ("k=3 mostly,
        # some longer clauses"); literals on distinct variables with random polarity.
        r = rng.random()
        k = 3 if r < 0.85 else (4 if r < 0.95 else 5)
        # claude: decision end
        clauses.append([2 * v + rng.randrange(2) for v in rng.sample(range(n), k)])
    cubes = [[2 * v + rng.randrange(2) for v in rng.sample(range(n), c)] for _ in range(Q)]
    return {"n": n, "m": m, "Q": Q, "c": c, "clauses": clauses, "cubes": cubes}


# ---- golden: two-watched-literal propagation ---------------------------------------------------
def _propagate_2wl(n: int, clauses, cube):
    """Returns (conflict, trail, clause_visits, literal_visits). Clause literal order is mutated
    by watch moves, as in MiniSat: c[0], c[1] are the watched literals."""
    val = [None] * (2 * n)                    # truth value of every literal
    cl = [list(c) for c in clauses]
    watches = [[] for _ in range(2 * n)]      # watches[l] = clauses watching literal l
    for ci, c in enumerate(cl):
        watches[c[0]].append(ci)
        watches[c[1]].append(ci)
    trail = []
    visits = 0
    lit_visits = 0

    def assign(lit):
        val[lit] = True
        val[lit ^ 1] = False
        trail.append(lit)

    for lit in cube:                          # distinct variables by construction
        assign(lit)
    head = 0
    conflict = False
    while head < len(trail) and not conflict:
        f = trail[head] ^ 1                   # the literal that just became false
        head += 1
        ws = watches[f]
        keep = []
        i = 0
        while i < len(ws):
            ci = ws[i]
            i += 1
            visits += 1
            c = cl[ci]
            if c[0] == f:
                c[0], c[1] = c[1], c[0]
            other = c[0]
            lit_visits += 1
            if val[other] is True:            # satisfied by the other watch: keep watching f
                keep.append(ci)
                continue
            moved = False
            for k in range(2, len(c)):        # look for a non-false replacement literal
                lit_visits += 1
                if val[c[k]] is not False:
                    c[1], c[k] = c[k], c[1]
                    watches[c[1]].append(ci)
                    moved = True
                    break
            if moved:
                continue
            keep.append(ci)
            if val[other] is False:           # all literals false
                conflict = True
                keep.extend(ws[i:])
                break
            assign(other)                     # unit clause
        watches[f] = keep
    return conflict, trail, visits, lit_visits


def native(inst):
    n, clauses = inst["n"], inst["clauses"]
    res = {}
    visits = 0
    lit_visits = 0
    implied = []
    for q, cube in enumerate(inst["cubes"]):
        conflict, trail, v, lv = _propagate_2wl(n, clauses, cube)
        visits += v
        lit_visits += lv
        res[q] = (conflict, None if conflict else frozenset(trail))
        if not conflict:
            implied.append(len(trail) - len(cube))
    native.last = {"clause_visits": visits, "literal_visits": lit_visits,
                   "n_conflict": sum(1 for r in res.values() if r[0]),
                   "implied_per_cube_min": min(implied) if implied else 0,
                   "implied_per_cube_mean": (sum(implied) / len(implied)) if implied else 0.0,
                   "implied_per_cube_max": max(implied) if implied else 0}
    native.last_result = res
    return res, visits


# ---- algebra: shared relations and O(1) scan / filter functions ----------------------------------
def _relations(inst):
    n, m, Q, clauses, cubes = inst["n"], inst["m"], inst["Q"], inst["clauses"], inst["cubes"]
    dom = {"q": Q, "c": m, "w": max(len(c) for c in clauses) + 1, "l": 2 * n}
    CL = Relation.from_set(("c", "w", "l"),
                           [(ci, len(c), lit) for ci, c in enumerate(clauses) for lit in c], dom)
    ASG0 = Relation.from_set(("q", "l"), [(q, lit) for q, cube in enumerate(cubes) for lit in cube],
                             dom)
    return dom, CL, ASG0


def _neg_lit(k, v):            # (q, l) -> (q, not l), boolean
    return [((k[0], k[1] ^ 1), True)]


def _neg_lit_count(k, v):      # (q, l) -> (q, not l), counting 1
    return [((k[0], k[1] ^ 1), 1)]


def _to_cube(k, v):            # (q, ...) -> (q,)
    return [((k[0],), True)]


def _to_qc(k, v):              # (q, c, ...) -> (q, c)
    return [((k[0], k[1]), True)]


def _count_qc_from_qlc(k, v):  # (q, l, c) -> ((q, c), 1)
    return [((k[0], k[2]), 1)]


def _count_qcw(k, v):          # (q, c, w, l) -> ((q, c, w), 1)
    return [((k[0], k[1], k[2]), 1)]


def _lit_as_value(k, v):       # (q, c, w, l) -> ((q, c, l), l)   value under (min,+)
    return [((k[0], k[1], k[3]), k[3])]


def _value_to_lit(k, v):       # ((q, c), l) -> ((q, c, l), True)
    return [((k[0], k[1], v), True)]


def _is_unit(k, v):            # false_count == w - 1
    return v == k[2] - 1


def _is_all_false(k, v):       # false_count == w
    return v == k[2]


def _is_two(k, v):
    return v == 2


def _result(inst, ASG, CONFQ):
    true = {q: set() for q in range(inst["Q"])}
    for q, lit in ASG.keys():
        true[q].add(lit)
    return {q: ((q,) in CONFQ, None if (q,) in CONFQ else frozenset(true[q]))
            for q in range(inst["Q"])}


def _conflicts(conf, CONF, IMPL):
    """conf U cubes with an all-false clause U cubes with two opposite implied literals."""
    conf, _ = update(conf, scan(CONF, _to_cube, schema=("q",), fn_name="to_cube", sr=BOOLEAN))
    NIMPL = scan(IMPL, _neg_lit, schema=("q", "l"), fn_name="negate_literal")
    OPP = combine(IMPL, NIMPL)                                        # (q, l) with both l, not l
    conf, _ = update(conf, scan(OPP, _to_cube, schema=("q",), fn_name="to_cube"))
    return conf


# ---- (a) scan form -------------------------------------------------------------------------------
def algebra_scan(inst):
    dom, CL, ASG0 = _relations(inst)
    CLC = Relation(("c", "w", "l"), COUNTING, {k: 1 for k in CL.keys()}, dom)
    state = {"conf": Relation(("q",), BOOLEAN, None, dom)}

    def step(ASG, D, k):
        NASG = scan(ASG, _neg_lit_count, schema=("q", "l"), fn_name="negate_literal", sr=COUNTING)
        FC = combine(CLC, NASG, keep=("q", "c", "w"))                     # false_count
        SAT = combine(CL, ASG, keep=("q", "c"))                           # satisfied clauses
        UNIT = mask(afilter(FC, _is_unit, "false_count == w-1"), SAT, negate=True)
        CONF = afilter(FC, _is_all_false, "false_count == w")
        UNITB = scan(UNIT, _to_qc, schema=("q", "c"), fn_name="to_bool", sr=BOOLEAN)
        UCL = combine(UNITB, CL)                                          # literals of unit clauses
        IMPL = reduce(mask(UCL, NASG, negate=True), keep=("q", "l"))      # the non-false one
        state["conf"] = _conflicts(state["conf"], CONF, IMPL)
        DNEW = mask(IMPL, state["conf"], negate=True)                     # freeze conflicted cubes
        return update(ASG, DNEW)

    ASG, info = fixpoint(step, ASG0)
    algebra_scan.last_info = info
    return _result(inst, ASG, state["conf"])


# ---- (b) watched form ----------------------------------------------------------------------------
def algebra_watched(inst, materialized: bool = True):
    dom, CL, ASG0 = _relations(inst)
    Q = inst["Q"]
    W0 = Relation.from_set(("c", "l"), [(ci, lit) for ci, c in enumerate(inst["clauses"])
                                        for lit in c[:2]], dom)
    state = {"conf": Relation(("q",), BOOLEAN, None, dom),
             "nasg": scan(ASG0, _neg_lit, schema=("q", "l"), fn_name="negate_literal"),
             "wmov": Relation(("q", "c", "l"), BOOLEAN, None, dom)}
    if materialized:
        state["watch"] = Relation.from_set(("q", "c", "l"),
                                           [(q, ci, lit) for q in range(Q) for ci, lit in W0.keys()],
                                           dom)

    def new_watches(CANDL):
        """One replacement per (q, c): the smallest candidate literal (grouped select-1 as a
        reduce under (min,+) with the literal id as the value)."""
        P = reduce(scan(CANDL, _lit_as_value, schema=("q", "c", "l"), fn_name="literal_as_value",
                        sr=MINPLUS), keep=("q", "c"))
        return scan(P, _value_to_lit, schema=("q", "c", "l"), fn_name="value_to_key", sr=BOOLEAN)

    def step(ASG, D, k):
        NASG = state["nasg"]
        DF = scan(D, _neg_lit, schema=("q", "l"), fn_name="negate_literal")   # newly false literals
        # claude: decision start — the materialised watch(q,c,l) is the primary program (the
        # plan's relation); the factored W0(c,l) + WMOV(q,c,l) variant has the same touched set
        # and the same combine candidates and is reported by extra_info for the multi-query view.
        if materialized:
            T3 = combine(DF, state["watch"])                              # (q, l, c) touched
        else:
            T1 = combine(DF, W0)                                          # (q, l, c)
            T2 = combine(DF, state["wmov"])                               # (q, l, c)
            T3, _ = update(T1, T2)
        # claude: decision end
        TOUCHED = reduce(T3, keep=("q", "c"))
        KF = scan(T3, _count_qc_from_qlc, schema=("q", "c"), fn_name="count_fallen_watches",
                  sr=COUNTING)
        TCL = combine(TOUCHED, CL)                                        # (q, c, w, l)
        SAT = combine(TCL, ASG, keep=("q", "c"))
        UNS = mask(TCL, SAT, negate=True)                                 # unsatisfied touched
        FL = mask(UNS, NASG)                                              # their false literals
        FC = scan(FL, _count_qcw, schema=("q", "c", "w"), fn_name="count_false_literals",
                  sr=COUNTING)
        UNIT = afilter(FC, _is_unit, "false_count == w-1")
        CONF = afilter(FC, _is_all_false, "false_count == w")
        NF = mask(UNS, NASG, negate=True)                                 # non-false literals
        IMPL = reduce(mask(NF, UNIT), keep=("q", "l"))
        NEED = mask(mask(NF, UNIT, negate=True), CONF, negate=True)       # clauses to re-watch
        if materialized:
            CANDL = mask(NEED, state["watch"], negate=True)
        else:
            CANDL = mask(mask(NEED, W0, negate=True), state["wmov"], negate=True)
        NW1 = new_watches(CANDL)
        TWO = afilter(KF, _is_two, "fallen_watches == 2")
        NW2 = new_watches(mask(mask(CANDL, TWO), NW1, negate=True))
        NW, _ = update(NW1, NW2)
        if materialized:
            state["watch"], _ = update(state["watch"], NW)
        else:
            state["wmov"], _ = update(state["wmov"], NW)
        state["conf"] = _conflicts(state["conf"], CONF, IMPL)
        DNEW = mask(IMPL, state["conf"], negate=True)
        ASGn, Dn = update(ASG, DNEW)
        state["nasg"], _ = update(NASG, scan(Dn, _neg_lit, schema=("q", "l"),
                                             fn_name="negate_literal"))
        return ASGn, Dn

    ASG, info = fixpoint(step, ASG0)
    algebra_watched.last_info = info
    return _result(inst, ASG, state["conf"])


def algebra(inst):
    return algebra_watched(inst, materialized=True)


def compare(a, b) -> bool:
    if a.keys() != b.keys():
        return False
    for q in a:
        ca, sa = a[q]
        cb, sb = b[q]
        if ca != cb:
            return False
        if not ca and sa != sb:
            return False
    return True


def extra_info(inst, res):
    import time
    g = native.last
    info = algebra_watched.last_info
    out = {"rounds": info["rounds"], "delta_sizes": info["delta_sizes"],
           "n_conflict": g["n_conflict"], "conflict_fraction": g["n_conflict"] / inst["Q"],
           "implied_per_cube_mean": g["implied_per_cube_mean"],
           "implied_per_cube_min": g["implied_per_cube_min"],
           "implied_per_cube_max": g["implied_per_cube_max"],
           "golden_clause_visits": g["clause_visits"], "golden_literal_visits": g["literal_visits"]}
    # (a) scan form under its own trace, same golden
    t0 = time.perf_counter()
    with use_trace() as tr:
        res_scan = algebra_scan(inst)
    s = tr.summary()
    v = g["clause_visits"]
    out["scan_form"] = {"ok": compare(native.last_result, res_scan),
                        "rounds": algebra_scan.last_info["rounds"],
                        "mul_ops": s["mul_ops"], "sr_ops": s["sr_ops"], "candidates": s["candidates"],
                        "ratio_mul": s["mul_ops"] / v, "ratio_sr": s["sr_ops"] / v,
                        "ratio_cand": s["candidates"] / v, "t_s": time.perf_counter() - t0}
    # the factored W0 + WMOV variant of (b): same combine candidates, Q times less watch state
    t0 = time.perf_counter()
    with use_trace() as tr:
        res_f = algebra_watched(inst, materialized=False)
    s = tr.summary()
    out["factored_watch"] = {"ok": compare(native.last_result, res_f),
                             "rounds": algebra_watched.last_info["rounds"],
                             "mul_ops": s["mul_ops"], "sr_ops": s["sr_ops"],
                             "candidates": s["candidates"], "ratio_mul": s["mul_ops"] / v,
                             "ratio_cand": s["candidates"] / v, "t_s": time.perf_counter() - t0}
    return out
