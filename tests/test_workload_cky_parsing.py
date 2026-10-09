import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.cky_parsing import workload as W


@pytest.fixture(autouse=True)
def _restore_switches():
    order, mode = W.ORDER, W.LENGTH_MODE
    yield
    W.ORDER, W.LENGTH_MODE = order, mode


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    # default program: prescribed pair-first order, length constraint by re-keying scan
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # pair-first materializes every (B, C) pair of the two child cells; only |rules| / N^2 of them
    # carry a rule. Measured ratio_mul 3.31-3.38 over seeds 0-2 (N=12, n=20, 2026-09-25).
    assert 2.0 <= r.ratio_mul <= 5.0
    assert r.ratio_cand <= 7.0                       # measured 4.20-4.41


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_rule_first_order(seed):
    W.ORDER = "rule_first"
    r = H.run(W, "small", seed)
    assert r.ok
    # T(i,k,A,C) = sum_B span(i,k,B) rule(A,B,C) once per cell: ratio_mul 1.18-1.21 measured
    assert 1.0 <= r.ratio_mul <= 1.5
    assert r.ratio_cand <= 3.5                       # measured 2.31-2.50


def test_small_filter_length_constraint_inflates():
    W.LENGTH_MODE = "filter"
    r = H.run(W, "small", 0)
    assert r.ok
    # every adjacent pair of known spans is re-joined every round: measured ratio_mul 35.7 (seed 0)
    assert r.ratio_mul > 20.0


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    assert 3.0 <= r.ratio_mul <= 8.0                 # measured 5.98 (N=24, n=45, seed 0, 2026-09-25)
    assert r.ratio_cand <= 10.0                      # measured 6.82


def test_large_rule_first_order():
    W.ORDER = "rule_first"
    r = H.run(W, "large", 0)
    assert r.ok
    assert 1.0 <= r.ratio_mul <= 1.3                 # measured 1.065
