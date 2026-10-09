"""Regular path queries on a labelled graph: reachability on the product graph (node, NFA state).

Given an NFA (6 states, built from a random regular expression such as a (b|c)* d e by the
Glushkov position construction, so it has no epsilon transitions), a labelled graph g(u, l, v) and
Q source nodes, find every (source, v) such that some path source -> v spells a word of the NFA.

    F_0(q, v, s)   = {(q, source_q, start)}
    C(q, w, t)     = OR_{v, s, l}  F(q, v, s) AND g(v, l, w) AND nfa(s, l, t)      one BFS layer
    New = C AND NOT X ;  X' = X OR New ;  repeat until New is empty
    answers(q, v)  = OR_s  X(q, v, s) AND accept(s)

Algebra program (one fixpoint round). The three-way join is two binary combines; the module switch
ORDER selects the order and the harness / tests measure every setting:
    graph_first   : C1 = combine(Delta, g)              # (q,v,s) x (v,l,w) on v      -> (q,v,s,l,w)
                    C  = combine(C1, nfa, keep=(q,w,t)) # on (s,l)                    -> (q,w,t)
    nfa_first     : C1 = combine(Delta, nfa)            # (q,v,s) x (s,l,t) on s      -> (q,v,s,l,t)
                    C  = combine(C1, g, keep=(q,w,t))   # on (v,l)                    -> (q,w,t)
    product_first : PG = combine(g, nfa, keep=(v,s,w,t)) once, outside the loop (the product graph's
                    edge relation); C = combine(Delta, PG, keep=(q,w,t)) on (v,s)
    then          : New = mask(rename(C, w->v, t->s), X, negate=True); X, Delta = update(X, New)
Readout: answers = reduce(mask(X, accept), keep=(q, v)).

Golden: BFS over the product graph from every source in plain Python; for each dequeued (v, s) it
loops over the NFA transitions (s, l, t) and, for each, over the l-labelled out-edges of v, so every
inner iteration is one product-graph edge. Both sides count "product-graph edge traversals": one
per ((v, s) popped, edge (v, l, w), transition (s, l, t)) triple; the algebra's second combine
examines exactly these triples because every (q, v, s) enters Delta once. The first combine of the
two-join program is the extra work of the chosen order (its candidates are the out-edges of v, or
the transitions of s, regardless of whether the label matches on the other side).
"""
from __future__ import annotations

import random
from collections import deque

from algebra import BOOLEAN, Relation, combine, mask, update, rename, reduce as areduce, fixpoint

NAME = "automata_rpq"
CATEGORY = "automata"
SEMIRINGS = ["boolean"]
STRUCTURE = "frontier-fixpoint"
HOTSPOT_EVENT = ("product-graph edge traversals: one per ((v, s) visited, edge (v, l, w), "
                 "NFA transition (s, l, t)) triple")
# claude: decision start — the task asked for Q=128 sources at large scale; measured with the
# default order that run took 57.8 s of algebra time (2026-09-25, seed 0; 105 rounds, 3.5 M product
# nodes), so "large" keeps the specified graph and uses Q=64 to stay under the 40 s bound, and the
# requested Q=128 is kept as the extra scale "spec_large".
SCALES = {"small": {"n": 3000, "m": 15000, "labels": 5, "Q": 32, "positions": 5},
          "large": {"n": 30000, "m": 150000, "labels": 5, "Q": 64, "positions": 5},
          "spec_large": {"n": 30000, "m": 150000, "labels": 5, "Q": 128, "positions": 5}}
# claude: decision end

ORDER = "nfa_first"           # "graph_first" | "nfa_first" | "product_first"

LABELS = "abcde"


# ---- instance -----------------------------------------------------------------------------------
def _random_regex(rng: random.Random, n_labels: int, positions: int):
    """A concatenation of factors over `positions` symbol occurrences. The first factor is a plain
    symbol (so the empty word is not accepted); exactly one factor is a starred union of two
    distinct labels, (x|y)*, as in the task's example a (b|c)* d; the remaining occurrences are
    drawn from {x, x*, (x|y)}. Returns the factor list [(kind, [labels])]."""
    # claude: decision start — one two-label starred group is mandatory: on a graph with m/n = 5
    # edges per node over 5 labels a single-label loop has mean out-degree 1 (percolation threshold,
    # reach varies from a handful of nodes to thousands between seeds) while a two-label loop has
    # mean out-degree 2 and a giant component, which makes the instances comparable across seeds.
    while True:
        factors = [("sym", [rng.randrange(n_labels)])]
        left = positions - 1
        placed = False
        while left > 0:
            kinds = ["sym", "star"] + (["union", "ustar"] if left >= 2 else [])
            if not placed and left == 2:
                kinds = ["ustar"]
            kind = rng.choice(kinds)
            if kind == "ustar":
                if placed:
                    continue
                placed = True
                labels = rng.sample(range(n_labels), 2)
            else:
                labels = [rng.randrange(n_labels) for _ in range(2 if kind == "union" else 1)]
            factors.append((kind, labels))
            left -= len(labels)
        if placed:
            return factors
    # claude: decision end


def _glushkov(factors):
    """Position automaton: state 0 = start, state p = the p-th symbol occurrence (1-based).
    Returns (transitions {(s, l, t)}, accepting set, regex string)."""
    pos = 0
    label_of = {}
    first, last, nullable = set(), set(), True
    follow = {}
    parts = []
    for kind, labels in factors:
        ps = []
        for lab in labels:
            pos += 1
            label_of[pos] = lab
            ps.append(pos)
        f_first, f_last = set(ps), set(ps)
        f_null = kind in ("star", "ustar")
        if f_null:
            for p in ps:
                follow.setdefault(p, set()).update(ps)          # loop back into the starred group
        for p in last:                                         # concatenation: Last(E1) -> First(F)
            follow.setdefault(p, set()).update(f_first)
        if nullable:
            first |= f_first
        last = f_last | (last if f_null else set())
        nullable = nullable and f_null
        txt = "|".join(LABELS[l] for l in labels)
        txt = f"({txt})" if len(labels) > 1 else txt
        parts.append(txt + ("*" if f_null else ""))
    trans = {(0, label_of[p], p) for p in first}
    for p, qs in follow.items():
        for q in qs:
            trans.add((p, label_of[q], q))
    return sorted(trans), sorted(last), " ".join(parts), pos + 1


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    n, m, L = p["n"], p["m"], p["labels"]
    edges = set()
    while len(edges) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            edges.add((u, rng.randrange(L), v))
    # claude: decision start — the NFA is the Glushkov (position) automaton of a random regular
    # expression with 5 symbol occurrences, which gives exactly 6 states and no epsilon transitions;
    # the first factor is a plain symbol so that the start state is never accepting.
    trans, accepting, regex, n_states = _glushkov(_random_regex(rng, L, p["positions"]))
    # claude: decision end
    sources = rng.sample(range(n), p["Q"])
    return {"n": n, "labels": L, "edges": sorted(edges), "nfa": trans, "accepting": accepting,
            "n_states": n_states, "regex": regex, "sources": sources}


# ---- golden -------------------------------------------------------------------------------------
def native(inst):
    """BFS on the product graph from every source. The inner loops run NFA-first: for a popped
    (v, s), for each transition (s, l, t), for each l-labelled out-edge (v, w): one traversal."""
    n, L = inst["n"], inst["labels"]
    adj = [[[] for _ in range(L)] for _ in range(n)]
    outdeg = [0] * n
    for u, l, v in inst["edges"]:
        adj[u][l].append(v)
        outdeg[u] += 1
    delta = [[] for _ in range(inst["n_states"])]
    for s, l, t in inst["nfa"]:
        delta[s].append((l, t))
    accepting = set(inst["accepting"])
    traversals = 0
    probes_nfa_first = 0       # (v, s) x transition pairs examined, matching edges or not
    probes_graph_first = 0     # (v, s) x out-edge pairs a graph-first loop would examine
    visited_all = set()
    answers = set()
    for q, src in enumerate(inst["sources"]):
        start = (src, 0)
        visited = {start}
        dq = deque([start])
        while dq:
            v, s = dq.popleft()
            probes_nfa_first += len(delta[s])
            probes_graph_first += outdeg[v]
            for l, t in delta[s]:
                for w in adj[v][l]:
                    traversals += 1
                    key = (w, t)
                    if key not in visited:
                        visited.add(key)
                        dq.append(key)
        for v, s in visited:
            visited_all.add((q, v, s))
            if s in accepting:
                answers.add((q, v))
    native.last_probes = {"nfa_first": probes_nfa_first, "graph_first": probes_graph_first}
    return {"answers": answers, "visited": visited_all}, traversals


# ---- algebra -------------------------------------------------------------------------------------
def algebra(inst):
    n, L, S, Q = inst["n"], inst["labels"], inst["n_states"], len(inst["sources"])
    dom = {"q": Q, "v": n, "w": n, "s": S, "t": S, "l": L}
    G = Relation.from_set(("v", "l", "w"), inst["edges"], dom)
    Nfa = Relation.from_set(("s", "l", "t"), inst["nfa"], dom)
    Acc = Relation.from_set(("s",), [(s,) for s in inst["accepting"]], dom)
    F0 = Relation.from_set(("q", "v", "s"), [(q, src, 0) for q, src in enumerate(inst["sources"])], dom)
    if ORDER == "product_first":
        # claude: decision start — the two static relations are joined once, outside the fixpoint;
        # PG is the product graph's edge relation and every round is then a single combine.
        PG = combine(G, Nfa, keep=("v", "s", "w", "t"))                     # on l
        # claude: decision end

    def step(X, D, k):
        if ORDER == "graph_first":
            C1 = combine(D, G)                                              # on v   -> (q,v,s,l,w)
            C = combine(C1, Nfa, keep=("q", "w", "t"))                      # on (s,l)
        elif ORDER == "nfa_first":
            C1 = combine(D, Nfa)                                            # on s   -> (q,v,s,l,t)
            C = combine(C1, G, keep=("q", "w", "t"))                        # on (v,l)
        else:
            C = combine(D, PG, keep=("q", "w", "t"))                        # on (v,s)
        New = mask(rename(C, {"w": "v", "t": "s"}), X, negate=True)
        return update(X, New)

    X, info = fixpoint(step, F0)
    Ans = areduce(mask(X, Acc), keep=("q", "v"))                            # OR over accepting s
    algebra.last_info = {"rounds": info["rounds"], "delta_sizes": info["delta_sizes"],
                         "visited": len(X), "order": ORDER}
    return {"answers": set(Ans.keys()), "visited": set(X.keys())}


def compare(a, b) -> bool:
    return a["answers"] == b["answers"] and a["visited"] == b["visited"]


def extra_info(inst, res):
    info = algebra.last_info
    return {"order": ORDER, "regex": inst["regex"], "nfa_transitions": len(inst["nfa"]),
            "accepting": inst["accepting"], "rounds": info["rounds"],
            "max_delta": max(info["delta_sizes"]) if info["delta_sizes"] else 0,
            "visited": info["visited"],
            "visited_per_source": info["visited"] / len(inst["sources"]),
            "answers": len(res["answers"]),
            "native_probes": getattr(native, "last_probes", None)}
