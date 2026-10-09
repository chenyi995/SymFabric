import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from softwareworkload import _harness as H
from softwareworkload.bayes_net_ve import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tiny_two_goldens_agree(seed):
    """The dict-based VE golden is itself checked against full joint enumeration (n = 12)."""
    inst = W.make_instance("tiny", seed)
    res, _ = W.native(inst)
    bf = W.brute_force(inst)
    assert np.allclose(bf, res["marginal"], rtol=1e-9, atol=1e-12)
    assert np.allclose(np.log(bf), res["log_marginal"], rtol=0.0, atol=1e-9)
    r = H.run(W, "tiny", seed)
    assert r.ok


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # same pairwise product order on both sides: one combine candidate per product entry
    assert r.ratio_mul == pytest.approx(1.0)
    # measured (harness, small, seeds 0-2, 2026-09-25): ratio_cand 1.04-1.12 (evidence filters,
    # single-factor reduces, final normalisation), ratio_sr 1.5-1.6 (the sum-out additions)
    assert r.ratio_cand <= 1.3
    assert r.ratio_sr <= 1.8
    assert r.extra["max_factor_entries"] <= W.MAX_FACTOR_ENTRIES


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    assert r.ratio_mul == pytest.approx(1.0)
    assert r.ratio_cand <= 1.3
    assert r.extra["max_factor_entries"] <= W.MAX_FACTOR_ENTRIES
    assert r.t_algebra_s < 40.0
