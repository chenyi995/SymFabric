import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.datalog_samegen import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # join order "up_first": the two combines count exactly the semi-naive golden's two pair loops
    assert r.ratio_mul == pytest.approx(1.0)
    # measured ratio_cand 2.34-2.36 (small, seeds 0-2): one mask probe and one update per C tuple
    assert r.ratio_cand <= 3.0
    e = r.extra
    # the naive golden re-derives every fact each round: strictly more events than semi-naive
    assert e["naive_events"] > e["seminaive_events"]
    assert e["seminaive_events"] == r.native_ops
    # sg is irreflexive on a tree with sibling flat pairs: the antijoin never removes anything
    assert e["rounds"] == e["seminaive_rounds"]


def test_small_both_join_orders_agree():
    inst = W.make_instance("small", 0)
    golden, _ = W.native(inst)
    for order in ("up_first", "down_first"):
        assert W.algebra(inst, order) == golden


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    assert r.ratio_mul == pytest.approx(1.0)
    assert r.ratio_cand <= 3.0       # measured 2.57 (large, seed 0)
