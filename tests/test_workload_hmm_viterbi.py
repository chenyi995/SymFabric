import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.hmm_viterbi import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    # forward + viterbi each do T*N*N transition ops; emission joins and back-tracking add O(T*N)
    assert 1.0 <= r.ratio_mul <= 1.2


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
