import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from algebra import use_trace
from softwareworkload import _harness as H
from softwareworkload.ctrlg_dfa_hmm import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # measured (harness, small, seeds 0-2, 2026-09-25): ratio_mul 1.143-1.145: emission join and
    # DFA-move gather add 2N per step to the N*N transition MACs (1 + 2/N = 1.125 at N = 16),
    # plus the live-token gather and the liveness ANDs
    assert 1.0 <= r.ratio_mul <= 1.3
    assert r.ratio_sr <= 2.3                       # a MAC is one mul and one add
    e = r.extra
    S, N = W.SCALES["small"]["S"], W.SCALES["small"]["N"]
    assert 0 < e["n_live_states"] < S              # trap states are dead, the start state is live
    assert e["fwd_T_support"] == N                 # deterministic DFA: one live DFA column
    assert e["liveness_algebra_mul_ops"] == e["liveness_native_relaxations"]
    assert e["log_scores_vs_log_max_abs"] <= 1e-8
    assert e["fwd_T_min_nonzero"] > 1e-300         # PROB forward did not underflow


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    assert 1.0 <= r.ratio_mul <= 1.2
    assert r.t_algebra_s < 40.0


def test_dense_first_scoring_matches_but_costs_more():
    """Applying the dense emission matrix before the sparse live-transition selector gives the
    same scores with N*V instead of N*|live tokens| multiplications."""
    inst = W.make_instance("small", 0)
    golden, _ = W.native(inst)
    with use_trace() as tr_sparse:
        res_sparse = W.algebra(inst, score_mode="sparse_first")
    with use_trace() as tr_dense:
        res_dense = W.algebra(inst, score_mode="dense_first")
    assert W.compare(golden, res_sparse) and W.compare(golden, res_dense)
    assert tr_dense.summary()["mul_ops"] > tr_sparse.summary()["mul_ops"]
