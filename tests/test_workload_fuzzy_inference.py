import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.fuzzy_inference import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # the round-synchronous semi-naive fixpoint re-derives a pair every time its max-min value rises;
    # widest-path Dijkstra settles it once. Measured ratio_mul 4.03-4.22 over seeds 0-2 (2026-09-25).
    assert 2.0 <= r.ratio_mul <= 7.0
    assert r.ratio_cand <= 10.0                      # measured 5.58-5.98
    assert r.extra["closure_density"] > 0.9          # the random relation is strongly connected


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    assert 2.0 <= r.ratio_mul <= 8.0                 # measured 4.84 (n=800, seed 0, 2026-09-25)
    assert r.ratio_cand <= 10.0                      # measured 6.80
