import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from algebra import use_trace
from softwareworkload import _harness as H
from softwareworkload.prob_circuit import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # measured (harness, small, seeds 0-2, 2026-09-25): ratio_mul 1.097; the excess over 1 is the
    # unit-weight gather join of every product-node child slot (f_p per node, 1.5x on product layers)
    assert 1.0 <= r.ratio_mul <= 1.25
    assert r.ratio_cand == pytest.approx(r.ratio_mul)      # every candidate is one mul
    assert r.ratio_sr <= 2.0                                # a MAC is one mul and one add
    e = r.extra
    assert 1.0 <= e["ratio_mul_prob"] <= 1.25 and e["ratio_mul_prob"] == e["ratio_mul_logprob"]
    assert e["ratio_mul_counting"] <= e["ratio_mul_prob"]   # zero suppression skips 0-valued children
    assert e["logprob_vs_log_prob_max_abs"] <= 1e-8
    assert e["prob_roots_underflow_in_float32"] == 0        # no underflow at these depths


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    assert 1.0 <= r.ratio_mul <= 1.25
    assert r.t_algebra_s < 40.0


def test_fold_product_mode_is_sound_only_without_zeros():
    """reduce(use_mul=True) multiplies the children that are present; a child whose value is the
    semiring zero is absent from the relation, so the product silently drops the zero factor.
    prob / logprob leaves are strictly positive (sound); counting leaves contain zeros (unsound)."""
    inst = W.make_instance("small", 0)
    golden, _ = W.native(inst)
    with use_trace():
        res = W.algebra(inst, product_mode="fold")
    assert np.allclose(golden["prob"], res["prob"], rtol=1e-9, atol=0.0)
    assert np.allclose(golden["logprob"], res["logprob"], rtol=1e-9, atol=1e-9)
    assert np.any(inst["leaf_count"] == 0)
    assert list(golden["counting"]) != list(res["counting"])
