import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.planning_hadd import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # measured 2026-09-25 (seeds 0-2): ratio_mul 3.90-4.21, ratio_cand 7.43-8.05; bounds with slack.
    # The ~4x comes from four passes per (action, precondition) tuple (touch, gather, join, fold)
    # times a re-touch factor of ~1.25; the golden folds each tuple once.
    assert 3.0 <= r.ratio_mul <= 5.0
    assert r.ratio_cand <= 9.5          # below the plan's 10x admission threshold
    assert 3 <= r.extra["rounds_add"] <= 12 and 3 <= r.extra["rounds_max"] <= 12


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    # measured 2026-09-25: ratio_mul 4.02, ratio_cand 7.67, 6 rounds each
    assert 3.0 <= r.ratio_mul <= 5.0
    assert r.ratio_cand <= 9.5


def test_deep_matches_golden_and_exceeds_threshold_budget_is_known():
    # long causal chains: measured 2026-09-25 ratio_mul 6.52, ratio_cand 12.42, 14 rounds each;
    # the bound below only guards against regressions, the doc reports the value itself
    r = H.run(W, "deep", 0)
    assert r.ok
    assert r.ratio_mul <= 8.0
    assert r.ratio_cand <= 15.0


def test_unreachable_propositions_stay_infinite():
    inst = W.make_instance("small", 0)
    res, _ = W.native(inst)
    achievers = {p for act in inst["actions"] for p in act["add"]}
    for p in range(inst["P"]):
        if p not in achievers and p not in inst["init"]:
            assert res["h_add"][p] == W.INF and res["h_max"][p] == W.INF
    assert sum(v == W.INF for v in res["h_add"]) >= inst["P"] // 20   # the dead propositions


def test_same_program_in_bottleneck_is_not_hmax():
    """The program run unchanged in BOTTLENECK (min, max) equals the bottleneck heuristic
    min_a max(cost(a), max_q c(q)) and differs from h^max = min_a [cost(a) + max_q c(q)]."""
    inst = W.make_instance("small", 0)
    res, _ = W.native(inst)
    gold_b, _ = W.native_bottleneck(inst)
    from algebra import use_trace
    with use_trace():
        alg_b = W.algebra_bottleneck(inst)
    assert all(x == y for x, y in zip(gold_b, alg_b))
    n_diff = sum(1 for x, y in zip(alg_b, res["h_max"]) if x != y)
    assert n_diff > 0
    # bottleneck <= h_max everywhere (max(c, m) <= c + m for c, m >= 0)
    assert all(x <= y for x, y in zip(alg_b, res["h_max"]))
