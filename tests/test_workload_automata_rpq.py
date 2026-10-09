import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from algebra import use_trace
from softwareworkload import _harness as H
from softwareworkload.automata_rpq import workload as W


@pytest.fixture(autouse=True)
def _restore_order():
    order = W.ORDER
    yield
    W.ORDER = order


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    # default order: frontier joined with the NFA first, then with the graph on (v, label)
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # the second join examines exactly the product-graph edges (one per traversal); the first join
    # adds one candidate per (visited (v,s), transition of s). Measured ratio_mul 1.99-2.01 over
    # seeds 0-2 (2026-09-25).
    assert 1.5 <= r.ratio_mul <= 3.0
    assert r.ratio_cand <= 6.0                       # measured 4.16-4.38


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_graph_first_order(seed):
    W.ORDER = "graph_first"
    r = H.run(W, "small", seed)
    assert r.ok
    # every out-edge of v (5 on average) is a candidate before the label is checked: measured 3.50-3.53
    assert 2.5 <= r.ratio_mul <= 5.0
    assert r.ratio_cand <= 8.0                       # measured 5.67-5.89


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_product_first_order(seed):
    W.ORDER = "product_first"
    r = H.run(W, "small", seed)
    assert r.ok
    # the graph x NFA join is hoisted out of the loop: measured 1.19-1.29
    assert 1.0 <= r.ratio_mul <= 1.6
    assert r.ratio_cand <= 5.0                       # measured 3.35-3.67


def test_orders_agree_on_small():
    inst = W.make_instance("small", 0)
    outs = []
    for order in ("graph_first", "nfa_first", "product_first"):
        W.ORDER = order
        with use_trace():
            outs.append(W.algebra(inst))
    assert outs[0] == outs[1] == outs[2]


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    assert 1.5 <= r.ratio_mul <= 3.0                 # measured 2.004 (n=30,000, Q=64, seed 0, 2026-09-25)
    assert r.ratio_cand <= 6.0                       # measured 4.079
