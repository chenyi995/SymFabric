"""Algebra laws and operator semantics, checked against naive nested-loop definitions.

Goldens here are independent: they never call algebra.ops; they enumerate the mathematical
definition directly on small random relations.
"""
import itertools
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from algebra import (BOOLEAN, COUNTING, PROB, LOGPROB, VITERBI, MINPLUS, FUZZY, REAL, BOTTLENECK,
                     Relation,
                     combine, reduce, filter, mask, update, select_k, rename, scan, fixpoint,
                     use_trace)

SRS = [BOOLEAN, COUNTING, PROB, LOGPROB, VITERBI, MINPLUS, FUZZY, REAL, BOTTLENECK]


def rand_value(sr, rng):
    if sr is BOOLEAN:
        return True
    if sr is COUNTING:
        return rng.randint(1, 5)
    if sr in (LOGPROB, VITERBI):
        return -rng.random() * 3
    if sr is MINPLUS or sr is BOTTLENECK:
        return float(rng.randint(1, 9))
    if sr is REAL:
        return rng.uniform(-2, 2)
    return rng.random() * 0.9 + 0.05


def rand_rel(schema, sr, rng, n, dom=6):
    pairs = [(tuple(rng.randrange(dom) for _ in schema), rand_value(sr, rng)) for _ in range(n)]
    return Relation.from_pairs(schema, sr, pairs, {a: dom for a in schema})


@pytest.mark.parametrize("sr", SRS)
def test_semiring_laws(sr):
    rng = random.Random(1)
    for _ in range(200):
        a, b, c = (rand_value(sr, rng) for _ in range(3))
        eq = sr.equal
        assert eq(sr.add(sr.add(a, b), c), sr.add(a, sr.add(b, c)), 1e-7)
        assert eq(sr.add(a, b), sr.add(b, a), 1e-7)
        assert eq(sr.mul(sr.mul(a, b), c), sr.mul(a, sr.mul(b, c)), 1e-7)
        assert eq(sr.add(a, sr.zero), a)
        assert eq(sr.mul(a, sr.one), a)
        assert eq(sr.mul(sr.one, a), a)
        assert eq(sr.mul(a, sr.zero), sr.zero)
        # distributivity
        assert eq(sr.mul(a, sr.add(b, c)), sr.add(sr.mul(a, b), sr.mul(a, c)), 1e-7)
        if sr.idempotent:
            assert eq(sr.add(a, a), a)


@pytest.mark.parametrize("sr", SRS)
def test_combine_reduce_matches_definition(sr):
    """Y(i,k) = sum_j R(i,j) * S(j,k) computed by nested loops vs combine+reduce and fused."""
    rng = random.Random(7)
    for trial in range(10):
        R = rand_rel(("i", "j"), sr, rng, 25)
        S = rand_rel(("j", "k"), sr, rng, 25)
        golden = {}
        for i, j, k in itertools.product(range(6), repeat=3):
            if (i, j) in R and (j, k) in S:
                v = sr.mul(R.get((i, j)), S.get((j, k)))
                golden[(i, k)] = sr.add(golden.get((i, k), sr.zero), v)
        G = Relation(("i", "k"), sr, golden)
        with use_trace() as tr:
            Y1 = reduce(combine(R, S), ("i", "k"))
            Y2 = combine(R, S, keep=("i", "k"))
        assert Y1.equals(G, 1e-7), (sr.name, trial)
        assert Y2.equals(G, 1e-7), (sr.name, trial)
        # candidates must equal the number of matching pairs, not |R|*|S|
        matching = sum(1 for (i, j) in R for (j2, k) in S if j == j2)
        assert tr.records[0].candidates == matching
        assert tr.records[0].mul_ops == matching


def test_mask_is_semijoin_antijoin():
    rng = random.Random(3)
    R = rand_rel(("i", "j"), COUNTING, rng, 30)
    S = rand_rel(("j",), COUNTING, rng, 4)
    keep = mask(R, S)
    drop = mask(R, S, negate=True)
    sup = {k[0] for k in S}
    assert set(keep.keys()) == {k for k in R if k[1] in sup}
    assert set(drop.keys()) == {k for k in R if k[1] not in sup}
    assert len(keep) + len(drop) == len(R)


def test_update_returns_exact_delta_for_idempotent():
    X = Relation.from_pairs(("v",), MINPLUS, [((0,), 5.0), ((1,), 2.0)])
    D = Relation.from_pairs(("v",), MINPLUS, [((0,), 3.0), ((1,), 4.0), ((2,), 1.0)])
    Xn, ch = update(X, D)
    assert Xn.get((0,)) == 3.0 and Xn.get((1,)) == 2.0 and Xn.get((2,)) == 1.0
    assert set(ch.keys()) == {(0,), (2,)}          # (1,) did not improve


def test_select_k_orders_by_semiring():
    R = Relation.from_pairs(("v",), MINPLUS, [((i,), float(10 - i)) for i in range(10)])
    top = select_k(R, 3)
    assert set(top.keys()) == {(9,), (8,), (7,)}
    P = Relation.from_pairs(("v",), PROB, [((i,), i / 10) for i in range(10)])
    top = select_k(P, 2)
    assert set(top.keys()) == {(9,), (8,)}


def test_scan_rekey_and_merge():
    R = Relation.from_pairs(("i",), COUNTING, [((i,), 1) for i in range(10)])
    H = scan(R, lambda k, v: [((k[0] % 3,), v)], schema=("b",), fn_name="mod3")
    assert H.get((0,)) == 4 and H.get((1,)) == 3 and H.get((2,)) == 3


def test_fixpoint_transitive_closure_vs_bfs():
    rng = random.Random(11)
    n = 30
    edges = {(rng.randrange(n), rng.randrange(n)) for _ in range(60)}
    E = Relation.from_set(("i", "j"), edges, {"i": n, "j": n})
    golden = set()
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    for s in range(n):
        seen, stack = set(), list(adj.get(s, []))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj.get(x, []))
        golden |= {(s, t) for t in seen}

    def step(X, D, k):
        C = combine(rename(D, {"j": "m"}), rename(E, {"i": "m"}), keep=("i", "j"))
        return update(X, mask(C, X, negate=True))

    with use_trace():
        TC, info = fixpoint(step, E)
    assert set(TC.keys()) == golden


def test_reduce_with_mul_is_product_fold():
    R = Relation.from_pairs(("n", "c"), PROB, [((0, 0), 0.5), ((0, 1), 0.4), ((1, 0), 0.25)])
    P = reduce(R, ("n",), use_mul=True)
    assert P.get((0,)) == pytest.approx(0.2) and P.get((1,)) == pytest.approx(0.25)
    S = reduce(R, ("n",))
    assert S.get((0,)) == pytest.approx(0.9)
