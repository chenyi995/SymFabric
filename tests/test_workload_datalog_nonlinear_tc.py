import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.datalog_nonlinear_tc import workload as W


def test_tiny_counterexample_for_single_delta():
    # a -> b -> c: the strict single-delta form (Delta |x| X_old) derives nothing beyond the edges
    inst = {"n": 3, "edges": [(0, 1), (1, 2)]}
    golden, _ = W.native(inst)
    assert golden == {(0, 1), (1, 2), (0, 2)}
    X_two, _ = W._algebra(inst, "two_term")
    assert set(X_two.keys()) == golden
    X_strict, _ = W._algebra(inst, "strict")
    assert set(X_strict.keys()) == {(0, 1), (1, 2)}          # path(a, c) is missing
    X_relaxed, _ = W._algebra(inst, "relaxed")
    assert set(X_relaxed.keys()) == golden                   # complete for TC, see the doc


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    e = r.extra
    # the non-linear program converges in fewer rounds than the linear one on the same instance
    assert e["rounds"] < e["linear_rounds"]
    assert e["linear_ok"]
    # ...but examines many more pairs: measured ratio_mul 27.1-31.3 (small, seeds 0-2)
    assert 20.0 <= r.ratio_mul <= 40.0
    assert e["relaxed_ok"] and not e["strict_ok"]


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    e = r.extra
    assert e["rounds"] < e["linear_rounds"]
    assert 55.0 <= r.ratio_mul <= 90.0      # measured 71.5 (large n=250, seed 0)
