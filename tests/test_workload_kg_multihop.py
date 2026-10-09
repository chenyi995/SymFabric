import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.kg_multihop import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # the semi-naive frontier combine relaxes exactly the golden's (improved entry, edge) pairs
    assert r.ratio_mul == pytest.approx(1.0, abs=1e-3)
    # merges of candidates landing on the same (q, t) inside a round: measured ratio_sr 1.15-1.16
    assert 1.0 <= r.ratio_sr < 1.3
    # update + antijoin + per-source select: measured ratio_cand 3.55-3.58
    assert r.ratio_cand < 5.0
    assert r.extra["rounds"] == W.SCALES["small"]["K"]
    assert r.extra["topk"]["selects"] == W.SCALES["small"]["Q"]


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    assert r.ratio_mul == pytest.approx(1.0, abs=1e-3)
    assert r.ratio_cand < 5.0        # measured 3.95 (large, seed 0, 2026-09-25)
