import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.vsa_cleanup import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # measured 2026-09-25 (count is size-determined, identical for all seeds): ratio_mul 1.225,
    # ratio_sr 2.112, ratio_cand 1.237. >1 because every gather through an indicator join counts
    # one mul per element; the GEMM itself is exactly Q*16*d muls.
    assert 1.0 <= r.ratio_mul <= 1.3
    assert r.ratio_sr <= 2.2
    assert r.ratio_cand <= 1.3
    assert r.extra["gemm_mul_ops"] == r.extra["native_macs"]
    assert r.extra["accuracy_algebra"] == 1.0 and r.extra["accuracy_native"] == 1.0
    assert r.extra["U_density"] == 1.0


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    # measured 2026-09-25: ratio_mul 1.125, ratio_sr 2.046, ratio_cand 1.138
    assert 1.0 <= r.ratio_mul <= 1.2
    assert r.ratio_cand <= 1.2
    assert r.extra["gemm_mul_ops"] == r.extra["native_macs"]
