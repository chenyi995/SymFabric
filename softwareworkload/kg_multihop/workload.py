"""k-hop reasoning on a knowledge graph from Q source entities that share the graph.

Edges (h, r, t) carry a confidence w in (0, 1]. The score of reaching t from source q in at most
k hops under max-product is computed in the log domain, i.e. in the VITERBI (max, +) semiring on
log w:
    x_0(q, q) = 0                                   (log 1)
    x_{i+1}(q, t) = max( x_i(q, t), max_{h, r} x_i(q, h) + log w(h, r, t) )
Algebra program (semi-naive frontier fixpoint with a hop limit K = 4):
    cand      = combine(Delta(q, h), E(h, r, t), keep=(q, t))     # max over (h, r) of x + log w
    X, Delta  = update(X, cand)                                    # max-merge; Delta = improved entries
Afterwards the top-10 answers per source: the seed entries (q, q) are removed with an antijoin,
X is partitioned by q (CONTROL RESIDUE: hash partition, one pass over X, counted in extra), and one
select_k(X_q, 10) runs per source (Q selects, |X_q| candidates each).

Golden: per-source dict-based layered relaxation with the same improvement rule (a candidate
improves an entry only when it is larger by more than 1e-9 relative, the relation layer's
numeric equality; without a tolerance a 1-ulp re-ordering of the same path sum would count as an
improvement). Hotspot event: (frontier entry, edge) relaxations, one per semiring mul.

Multi-query sharing: the Q sources read the same edge relation. Every round's combine has
n_in = (|Delta|, |E|); the per-round shapes are reported in `extra`.
"""
from __future__ import annotations

import math
import random

from algebra import VITERBI, Relation, combine, mask, update, rename, select_k, fixpoint, current_trace

NAME = "kg_multihop"
CATEGORY = "graph"
SEMIRINGS = ["viterbi"]
STRUCTURE = "frontier-fixpoint (multi-query)"
HOTSPOT_EVENT = "(frontier entry, edge) relaxations: x(q,h) + log w(h,r,t) evaluated"
SCALES = {"small": {"n": 5000, "m": 30000, "R": 20, "Q": 64, "K": 4},
          "large": {"n": 50000, "m": 300000, "R": 20, "Q": 256, "K": 4}}
TOP = 10
# claude: decision start — a candidate improves a stored score only beyond 1e-9 relative, the relation layer's numeric equality, so the golden's delta sets match the algebra's update deltas
TOL = 1e-9        # relative improvement tolerance (equals Semiring.equal's default)
# claude: decision end


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    n, m, R, Q = p["n"], p["m"], p["R"], p["Q"]
    edges = {}
    while len(edges) < m:
        h, t = rng.randrange(n), rng.randrange(n)
        r = rng.randrange(R)
        if h != t and (h, r, t) not in edges:
            edges[(h, r, t)] = 1.0 - rng.random()          # confidence in (0, 1]
    sources = sorted(rng.sample(range(n), Q))
    return {"n": n, "m": m, "R": R, "Q": Q, "K": p["K"], "sources": sources,
            "edges": sorted((h, r, t, math.log(w)) for (h, r, t), w in edges.items())}


def _improves(old: float, new: float) -> bool:
    if new <= old:
        return False
    if old == -math.inf:
        return True
    return abs(new - old) > TOL * max(1.0, abs(new), abs(old))


# ---------------------------------------------------------------------------------------------
def native(inst):
    """Per-source layered relaxation over K rounds, relaxing only entries improved in the
    previous round. Counts one relaxation per (improved entry, out-edge) pair."""
    n, K = inst["n"], inst["K"]
    adj = [[] for _ in range(n)]
    for h, r, t, lw in inst["edges"]:
        adj[h].append((t, lw))
    relax = 0
    scores = {}
    top = {}
    for q in inst["sources"]:
        x = {q: 0.0}
        delta = {q: 0.0}
        for _ in range(K):
            cand = {}
            for h, xv in delta.items():
                for t, lw in adj[h]:
                    relax += 1
                    s = xv + lw
                    if s > cand.get(t, -math.inf):
                        cand[t] = s
            new_delta = {}
            for t, s in cand.items():
                if _improves(x.get(t, -math.inf), s):
                    x[t] = s
                    new_delta[t] = s
            delta = new_delta
            if not delta:
                break
        for t, s in x.items():
            scores[(q, t)] = s
        ranked = sorted(((t, s) for t, s in x.items() if t != q), key=lambda ts: (-ts[1], ts[0]))
        top[q] = ranked[:TOP]
    return {"scores": scores, "top": top}, relax


# ---------------------------------------------------------------------------------------------
def algebra(inst):
    n, K, Q = inst["n"], inst["K"], inst["Q"]
    dom = {"q": n, "h": n, "r": inst["R"], "t": n}
    E = Relation(("h", "r", "t"), VITERBI, {(h, r, t): lw for h, r, t, lw in inst["edges"]}, dom)
    X0 = Relation(("q", "h"), VITERBI, {(q, q): 0.0 for q in inst["sources"]}, dom)
    tr = current_trace()
    shapes = []

    def step(X, D, i):
        cand = combine(D, E, keep=("q", "t"))            # (q,h) x (h,r,t) -> (q,t): max_{h,r} x + log w
        rec = tr.records[-1]
        shapes.append({"round": i, "n_in": list(rec.n_in), "candidates": rec.candidates,
                       "n_out": rec.n_out, "add_ops": rec.add_ops})
        cand = rename(cand, {"t": "h"})
        return update(X, cand)

    with tr.tag("hops"):
        X, info = fixpoint(step, X0, max_rounds=K)
    with tr.tag("topk"):
        answers = mask(X, X0, negate=True)               # drop the seed entries (q, q)
        # CONTROL RESIDUE: hash-partition the answers by source (one pass over |answers| tuples);
        # the algebra has no grouped select, so one select_k per source follows.
        # claude: decision start — the per-source slice is a hash partition in control (counted in extra) rather than Q semijoins, which would stream all of X once per source
        parts: dict[int, dict] = {}
        for (q, h), v in answers.items():
            parts.setdefault(q, {})[(q, h)] = v
        partition_tuples = len(answers)
        # claude: decision end
        top = {}
        for q in inst["sources"]:
            Xq = Relation(("q", "h"), VITERBI, parts.get(q, {}), dom)
            best = select_k(Xq, TOP)                     # VITERBI order: larger score first, ties by (q, t)
            top[q] = sorted(((k[1], v) for k, v in best.items()), key=lambda ts: (-ts[1], ts[0]))
    algebra.last_info = {"rounds": info["rounds"], "delta_sizes": info["delta_sizes"],
                         "shapes": shapes, "facts": len(X), "answers": len(answers),
                         "partition_tuples": partition_tuples,
                         "select_candidates": sum(r.candidates for r in tr.records
                                                  if r.op == "select" and r.tag.startswith("topk")),
                         "hop_add_ops": sum(r.add_ops for r in tr.records if r.tag.startswith("hops"))}
    return {"scores": {(k[0], k[1]): v for k, v in X.items()}, "top": top}


def compare(a, b) -> bool:
    sa, sb = a["scores"], b["scores"]
    if sa.keys() != sb.keys():
        return False
    for k, v in sa.items():
        if not math.isclose(v, sb[k], rel_tol=1e-12, abs_tol=1e-12):
            return False
    if a["top"].keys() != b["top"].keys():
        return False
    for q, lst in a["top"].items():
        other = b["top"][q]
        if [t for t, _ in lst] != [t for t, _ in other]:
            return False
        if any(not math.isclose(s1, s2, rel_tol=1e-12, abs_tol=1e-12)
               for (_, s1), (_, s2) in zip(lst, other)):
            return False
    return True


def extra_info(inst, res):
    li = algebra.last_info
    relax = sum(s["candidates"] for s in li["shapes"])
    return {"Q": inst["Q"], "K": inst["K"], "rounds": li["rounds"], "delta_sizes": li["delta_sizes"],
            "per_round": li["shapes"], "relaxations": relax, "facts": li["facts"],
            "facts_per_source": li["facts"] / inst["Q"], "answers": li["answers"],
            "hop_add_ops": li["hop_add_ops"],
            "topk": {"selects": inst["Q"], "select_candidates": li["select_candidates"],
                     "control_residue_partition_tuples": li["partition_tuples"]}}
