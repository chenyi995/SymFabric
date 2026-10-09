#!/usr/bin/env python3
"""Derive/check pedagogical examples, not benchmark measurements.

Source: the explicit tiny fixtures inside each function below. Field accounting:
input = constructed teaching case; native = independent scalar/set calculation;
steps/result = calculation using the repository's real Relation/operators.
No runtime, resource, hardware or performance numbers are produced.

--write mechanically fills EXAMPLE blocks in both companion documents, then
reads them back. Default mode checks the blocks without changing files.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import itertools
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from algebra.relation import Relation as R
from algebra.semiring import BOOLEAN, COUNTING, PROB, REAL, MINPLUS, VITERBI, FUZZY, BOTTLENECK
from algebra.ops import combine as J, reduce as G, rename, mask, update, scan, select_k, filter as filt

# claude: decision start — tiny constructed examples explain semantics, not measured
# workloads. Use exact fractions when possible; compare independent native reasoning
# with actual primitive calls. These are not end-to-end workload regression tests.
def rel(schema, sr, data):
    return R(tuple(schema.split()), sr, data)


def boolean(schema, keys):
    return rel(schema, BOOLEAN, {tuple(k): True for k in keys})


def show(x):
    if isinstance(x, R):
        return "{" + ", ".join(f"{k}: {show(v)}" for k, v in sorted(x.items())) + "}"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {show(v)}" for k, v in sorted(x.items())) + "}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(show(v) for v in x) + "]"
    if isinstance(x, set):
        return str(sorted(x))
    return str(x)


def record(inputs, native, **steps):
    return "\n".join(["input: " + inputs, "native: " + show(native)] +
                     [f"{key}: {show(value)}" for key, value in steps.items()])


def compose(A, B):
    return J(rename(A, {"j": "m"}), rename(B, {"i": "m"}), keep=("i", "j"))


def reach(edges):
    out = set(edges)
    while True:
        more = {(a, d) for a, b in out for c, d in out if b == c}
        nxt = out | more
        if nxt == out:
            return out
        out = nxt


def datalog_tc():
    edges = {("a", "b"), ("b", "c")}
    E = boolean("i j", edges)
    C = compose(E, E)
    D = mask(C, E, negate=True)
    X, _ = update(E, D)
    assert set(X.data) == reach(edges)
    return record(f"E={show(edges)}", reach(edges), join=C, new_delta=D, result=X)


def datalog_samegen():
    up, flat, down = {("a", "p")}, {("p", "q")}, {("q", "b")}
    native = {(a, b) for a, p in up for x, y in flat for q, b in down if p == x and y == q}
    T = J(boolean("x p", up), boolean("p q", flat), keep=("x", "q"))
    C = J(T, boolean("q y", down), keep=("x", "y"))
    assert set(C.data) == native
    return record(f"Up={show(up)}; Flat={show(flat)}; Down={show(down)}", native,
                  up_join_delta=T, join_down=C, result=C)


def datalog_nonlinear_tc():
    edges = {("a", "b"), ("b", "c"), ("c", "d")}
    X = D = boolean("i j", edges)
    old = boolean("i j", [])
    steps = {}
    round_id = 0
    while D.data:
        c1, c2 = compose(D, X), compose(old, D)
        U, _ = update(c1, c2)
        C = mask(U, X, negate=True)
        old = X
        X, D = update(X, C)
        steps[f"round_{round_id}_delta_join_current"] = c1
        steps[f"round_{round_id}_old_join_delta"] = c2
        steps[f"round_{round_id}_new_delta"] = D
        round_id += 1
    assert set(X.data) == reach(edges)
    return record(f"E={show(edges)}", reach(edges), **steps, result=X)


def triangle_count():
    edges = {("a", "b"), ("b", "c"), ("a", "c")}
    E = rel("i j", COUNTING, {e: 1 for e in edges})
    W = J(E, rename(E, {"i": "j", "j": "k"}))
    closed = mask(W, rename(E, {"j": "k"}))
    total = G(closed, ())
    native = sum((a, b) in edges and (b, c) in edges and (a, c) in edges
                 for a, b, c in itertools.combinations("abc", 3))
    assert total.get(()) == native
    return record(f"oriented E={show(edges)}", native, wedges=W, closed=closed, result=total)


def sat_bcp():
    # Literal IDs are signed names, not the local bit encoding; same BCP semantics.
    clauses = {"c1": {"!a", "b"}, "c2": {"!b", "c"}}
    neg = {"a": "!a", "b": "!b", "c": "!c", "!a": "a", "!b": "b", "!c": "c"}
    known = {"a"}
    while True:
        more = set()
        for lits in clauses.values():
            if not lits & known:
                unknown = {l for l in lits if neg[l] not in known}
                if len(unknown) == 1:
                    more |= unknown
        if more <= known:
            break
        known |= more
    CL = boolean("c w l", [(c, len(ls), l) for c, ls in clauses.items() for l in ls])
    A = D = boolean("l", [("a",)])
    steps = {}
    n = 0
    while D.data:
        false = scan(A, lambda k, v: [((neg[k[0]],), v)])
        satisfied = G(J(CL, A), ("c",))
        unsat = mask(CL, satisfied, negate=True)
        fallen = mask(unsat, false)
        counts = scan(fallen, lambda k, v: [((k[0], k[1]), 1)], schema=("c", "w"), sr=COUNTING)
        units = filt(counts, lambda k, v: v == k[1] - 1)
        remaining = mask(unsat, false, negate=True)
        implied = G(mask(remaining, units), ("l",))
        A, D = update(A, implied)
        steps[f"round_{n}_false_counts"] = counts
        steps[f"round_{n}_new_assignments"] = D
        n += 1
    assert {k[0] for k in A.data} == known
    return record(f"clauses={show(clauses)}; seed=a; scan-mode teaching round", known, **steps, result=A)


def kg_multihop():
    # Use additive log scores, matching the local VITERBI/max-plus relation.
    E = rel("u v", VITERBI, {("a", "b"): -1, ("b", "c"): -2, ("a", "c"): -4})
    source = rel("u", VITERBI, {("a",): 0})
    first = rename(J(source, E, keep=("v",)), {"v": "u"})
    second = rename(J(first, E, keep=("v",)), {"v": "u"})
    X, D = update(first, second)
    native = max(E.data[("a", "c")], E.data[("a", "b")] + E.data[("b", "c")])
    assert X.get(("c",)) == native
    return record(f"log-edge weights={show(E)}; source=a; hop limit=2", native,
                  first=first, second=second, changed=D, result=X)


def bayes_net_ve():
    prior = rel("a", PROB, {(0,): F(1, 2), (1,): F(1, 2)})
    # After evidence B=1, these are P(B=1|A=a).
    likelihood = rel("a", PROB, {(0,): F(1, 4), (1,): F(3, 4)})
    joint = J(prior, likelihood)
    Z = G(joint, ())
    post = scan(joint, lambda k, v: [(k, v / Z.get(()))])
    weights = [prior.get((a,)) * likelihood.get((a,)) for a in (0, 1)]
    native = { (a,): weights[a] / sum(weights) for a in (0, 1)}
    assert post.data == native
    # Also eliminate A, not merely normalize a distribution on A.
    assert Z.get(()) == sum(weights)
    return record(f"P(A)={show(prior)}; P(B=1|A)={show(likelihood)}", native,
                  factor_product=joint, eliminate_A_evidence_mass=Z, result=post)


def prob_circuit():
    x, y, z, w1, w2 = F(1, 2), F(1, 3), F(1, 4), F(2, 5), F(3, 5)
    left = rel("n", PROB, {("xy",): x, ("xz",): x})
    right = rel("n", PROB, {("xy",): y, ("xz",): z})
    products = J(left, right)
    weights = rel("n", PROB, {("xy",): w1, ("xz",): w2})
    out = J(products, weights, keep=())
    native = w1*x*y + w2*x*z
    assert out.get(()) == native
    zero_right = rel("n", PROB, {("xy",): 0, ("xz",): z})
    safe = J(left, zero_right)
    assert ("xy",) not in safe.data
    return record(f"x={x}; y={y}; z={z}; w1={w1}; w2={w2}", native,
                  mandatory_slot_product=products, result=out, if_y_zero_safe_product=safe)


def automata_rpq():
    edges = [("a", "x", "b"), ("b", "y", "c")]
    trans = [("s0", "x", "s1"), ("s1", "y", "s2")]
    PG = J(boolean("v l w", edges), boolean("s l t", trans), keep=("v", "s", "w", "t"))
    seed = boolean("v s", [("a", "s0")])
    first = rename(J(seed, PG, keep=("w", "t")), {"w": "v", "t": "s"})
    second = rename(J(first, PG, keep=("w", "t")), {"w": "v", "t": "s"})
    accepted = G(mask(second, boolean("s", [("s2",)])), ("v",))
    native = {w for v, l, u in edges for v2, l2, w in edges if v == "a" and u == v2 and l+l2 == "xy"}
    assert {k[0] for k in accepted.data} == native
    return record(f"G={show(edges)}; automaton={show(trans)}; accept=s2; source=a", native,
                  product_edges=PG, first=first, second=second, result=accepted)


def beam_search():
    B = rel("b s", VITERBI, {("p", "x"): -1, ("q", "y"): -2})
    T = rel("s t", VITERBI, {("x", "z"): -2, ("y", "z"): -2, ("x", "w"): -5})
    C = J(B, T, keep=("t", "b"))
    best = select_k(C, 2)
    candidates = [(v+w, t, b) for (b, s), v in B.items() for (s2, t), w in T.items() if s == s2]
    native = sorted(candidates, key=lambda p: (-p[0], p[1], p[2]))[:2]
    assert best.data == {(t, b): v for v, t, b in native}
    collapsed = J(B, T, keep=("t",))
    return record(f"beam log scores={show(B)}; transition log scores={show(T)}; K=2", native,
                  candidates_with_parent=C, result=best, WRONG_collapsing_parent=collapsed)


def fuzzy_inference():
    E = rel("i j", FUZZY, {("a", "b"): F(4, 5), ("b", "c"): F(1, 2), ("a", "c"): F(1, 5)})
    C = compose(E, E)
    X, D = update(E, C)
    native = max(E.get(("a", "c")), min(E.get(("a", "b")), E.get(("b", "c"))))
    assert X.get(("a", "c")) == native and D.get(("a", "c")) == native
    return record(f"max-min edges={show(E)}", native, join=C, changed_existing_key=D, result=X)


def hmm_viterbi():
    first = {(0,): F(1, 2), (1,): F(1, 2)}
    trans = {(0, 0): F(3, 4), (0, 1): F(1, 4), (1, 0): F(1, 4), (1, 1): F(3, 4)}
    emit = {(0,): F(1, 2), (1,): F(1, 4)}
    pred = J(rel("i", PROB, first), rel("i j", PROB, trans), keep=("j",))
    alpha = J(pred, rel("j", PROB, emit))
    Z = G(alpha, ())
    paths = {(i, j): first[(i,)]*a*emit[(j,)] for (i, j), a in trans.items()}
    native = {"forward": sum(paths.values()), "best_path": max(paths, key=paths.get), "best_weight": max(paths.values())}
    assert Z.get(()) == native["forward"]
    # Retain i for this explanatory traceback comparison, then take global best.
    path_relation = J(J(rel("i", PROB, first), rel("i j", PROB, trans)), rel("j", PROB, emit))
    best = select_k(path_relation, 1)
    assert best.get(native["best_path"]) == native["best_weight"]
    return record(f"post-first-emission alpha={show(first)}; A={show(trans)}; next emission={show(emit)}", native,
                  transition_contract=pred, emission=alpha, all_path_weights=path_relation, result_forward=Z, result_best=best)


def ctrlg_dfa_hmm():
    # Single hidden state, two symbols, exactly two remaining positions; accept aa|bb.
    p = {"a": F(3, 4), "b": F(1, 4)}
    token = rel("v", PROB, {(v,): w for v, w in p.items()})
    live = boolean("v", [("a",), ("b",)])
    proxy = mask(token, live)
    beta = rel("v", PROB, {(v,): w for v, w in p.items()})
    weighted = J(token, beta)
    Z = G(weighted, ())
    exact = scan(weighted, lambda k, v: [(k, v/Z.get(()))])
    paths = {a+b: p[a]*p[b] for a in p for b in p if a == b}
    native = {(v,): paths[v+v]/sum(paths.values()) for v in p}
    assert exact.data == native and exact.data != proxy.data
    return record(f"one hidden state; P(token)={show(p)}; accept aa|bb; remaining length=2", native,
                  LOCAL_liveness_proxy=proxy, completion_beta=beta, exact_unnormalized=weighted, result_exact=exact)


def vsa_cleanup():
    # Small orthogonal teaching codebook; production generator uses Gaussian vectors.
    C = rel("k d", REAL, {("x", 0): 1, ("y", 1): 1})
    key = rel("d", REAL, {(0,): -1, (1,): 1})
    chosen = filt(C, lambda k, v: k[0] == "x")
    bound = J(chosen, key, keep=("d",))
    unbound = J(bound, key)
    sim = J(unbound, C, keep=("k",))
    best = select_k(sim, 1)
    native = {"x": 1, "y": 0}
    assert all(sim.get((k,)) == v for k, v in native.items())
    assert set(best.data) == {("x",)}
    return record(f"orthogonal teaching codebook={show(C)}; bipolar key={show(key)}; one object/slot", native,
                  bind=bound, unbind=unbound, dot_similarity=sim, result=best)


def astar_grid():
    # One expansion of a unit grid corner; h is Manhattan distance to the opposite corner.
    O = rel("u", MINPLUS, {("s",): 2})
    h = {"s": 2, "a": 1, "b": 1, "g": 0}
    edges = {("s", "a"): 1, ("s", "b"): 1}
    best = select_k(O, 1)
    g = scan(best, lambda k, v: [(k, v-h[k[0]])])
    cand_g = J(g, rel("u v", MINPLUS, edges), keep=("v",))
    cand_f = scan(cand_g, lambda k, v: [(k, v+h[k[0]])])
    rest = rename(mask(O, best, negate=True), {"u": "v"})
    next_open, _ = update(rest, cand_f)
    native = {(v,): 0+w+h[v] for (u, v), w in edges.items() if u == "s"}
    assert next_open.data == native
    return record(f"open f={show(O)}; h={show(h)}; outgoing edges={show(edges)}", native,
                  select=best, recover_g=g, neighbor_g=cand_g, result_next_open_f=next_open)


def planning_hadd():
    # At this layer r->p and r->q have already produced the prerequisite costs.
    costs = {("p",): 2, ("q",): 3}
    pre = rel("a q", MINPLUS, {("act", "p"): 0, ("act", "q"): 0})
    joined = J(pre, rel("q", MINPLUS, costs))
    addfold = G(joined, ("a",), use_mul=True)
    maxfold = G(rel("a q", BOTTLENECK, joined.data), ("a",), use_mul=True)
    action = rel("a p", MINPLUS, {("act", "g"): 1})
    hadd = J(addfold, action, keep=("p",))
    hmax = J(rel("a", MINPLUS, maxfold.data), action, keep=("p",))
    native = {"hadd(g)": sum(costs.values())+1, "hmax(g)": max(costs.values())+1}
    assert hadd.get(("g",)) == native["hadd(g)"] and hmax.get(("g",)) == native["hmax(g)"]
    return record(f"costs={show(costs)}; act: p AND q -> g; action cost=1", native,
                  join_all_preconditions=joined, sum_fold=addfold, max_fold=maxfold, result_hadd=hadd, result_hmax=hmax)


def cky_parsing():
    left = rel("i k B", PROB, {(0, 1, "A"): 1, (0, 1, "C"): 1})
    right = rel("k j C", PROB, {(1, 2, "B"): 1, (1, 2, "D"): 1})
    rules = rel("A B C", PROB, {("S", "A", "B"): F(3, 5), ("S", "C", "D"): F(2, 5)})
    pair = J(left, right, keep=("i", "j", "B", "C"))
    contributions = J(pair, rules)
    inside = G(contributions, ("i", "j", "A"))
    native = {"inside(S)": sum(rules.data.values()), "viterbi_probability(S)": max(rules.data.values())}
    assert inside.get((0, 2, "S")) == native["inside(S)"]
    assert select_k(contributions, 1).data and max(contributions.data.values()) == native["viterbi_probability(S)"]
    return record(f"sentence=a b; left lexical={show(left)}; right lexical={show(right)}; rules={show(rules)}", native,
                  child_pairs=pair, rule_matches=contributions, result_inside=inside)


def graph_bfs_sssp():
    edges = {("a", "b"): 2, ("b", "c"): 1, ("a", "c"): 5}
    E = rel("u v", MINPLUS, edges)
    source = rel("u", MINPLUS, {("a",): 0})
    first = rename(J(source, E, keep=("v",)), {"v": "u"})
    second = rename(J(first, E, keep=("v",)), {"v": "u"})
    dist, changed = update(first, second)
    # Independent exhaustive simple paths on this acyclic teaching graph.
    native_sssp = min(edges[("a", "c")], edges[("a", "b")]+edges[("b", "c")])
    frontier = J(boolean("u", [("a",)]), boolean("u v", edges), keep=("v",))
    assert dist.get(("c",)) == native_sssp and ("c",) in frontier.data
    return record(f"edges={show(edges)}; source=a", {"BFS_hops(c)": 1, "SSSP_distance(c)": native_sssp},
                  BFS_first_frontier=frontier, SSSP_first=first, SSSP_second=second, changed=changed, result=dist)


EXAMPLES = [datalog_tc, datalog_samegen, datalog_nonlinear_tc, triangle_count,
            sat_bcp, kg_multihop, bayes_net_ve, prob_circuit, automata_rpq,
            beam_search, fuzzy_inference, hmm_viterbi, ctrlg_dfa_hmm,
            vsa_cleanup, astar_grid, planning_hadd, cky_parsing, graph_bfs_sssp]
# claude: decision end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--print", action="store_true", dest="print_examples")
    args = parser.parse_args()
    outputs = {fn.__name__: fn() for fn in EXAMPLES}
    if args.print_examples:
        for name, value in outputs.items():
            print(f"\n## {name}\n{value}")
        return
    for path in (ROOT / "sym1/workload_operator_evidence.md", ROOT / "local/sym1-workload-operator-evidence.zh.md"):
        text = path.read_text()
        for name, value in outputs.items():
            start, end = f"<!-- EXAMPLE:{name} -->", f"<!-- /EXAMPLE:{name} -->"
            pattern = re.escape(start) + r".*?" + re.escape(end)
            block = f"{start}\n```text\n{value}\n```\n{end}"
            assert len(re.findall(pattern, text, flags=re.S)) == 1, (path, name, "missing/duplicate block")
            if args.write:
                text = re.sub(pattern, lambda _: block, text, flags=re.S)
            else:
                assert re.search(pattern, text, flags=re.S).group() == block, (path, name, "stale data")
        if args.write:
            path.write_text(text)
        copied = path.read_text()
        for name, value in outputs.items():
            block = f"<!-- EXAMPLE:{name} -->\n```text\n{value}\n```\n<!-- /EXAMPLE:{name} -->"
            assert block in copied, (path, name, "read-back mismatch")
    print(f"PASS: {len(outputs)} teaching cases; native/primitive assertions and both documents' data blocks agree.")


if __name__ == "__main__":
    main()
