import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.sat_bcp import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok                                  # watched form: verdicts and implied sets per cube
    assert r.native_ops > 0
    e = r.extra
    # the instance is non-trivial: some cubes conflict, the others propagate tens of literals
    assert 0 < e["n_conflict"] < W.SCALES["small"]["Q"]
    assert e["implied_per_cube_mean"] >= 10
    # watched form: measured ratio_mul 4.7-5.2, ratio_cand 34-38 (small, seeds 0-2)
    assert 3.5 <= r.ratio_mul <= 7.0
    assert r.ratio_cand <= 50.0
    # scan form: correct but re-scans every cube's false literals each round;
    # measured ratio_mul 32-54 (small, seeds 0-2), at least 5x the watched form
    sf = e["scan_form"]
    assert sf["ok"]
    assert sf["ratio_mul"] >= 20.0
    assert sf["ratio_mul"] >= 5.0 * r.ratio_mul
    # factored watch representation: same combine work, same result
    ff = e["factored_watch"]
    assert ff["ok"]
    assert ff["mul_ops"] == r.algebra_mul_ops


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    e = r.extra
    assert 0 < e["n_conflict"] < W.SCALES["large"]["Q"]
    assert 3.5 <= r.ratio_mul <= 7.0             # measured 4.77 (large, seed 0)
    assert r.ratio_cand <= 50.0                  # measured 37.4
    assert e["scan_form"]["ok"]
    assert e["scan_form"]["ratio_mul"] >= 20.0   # measured 31.9
    assert e["factored_watch"]["ok"]
