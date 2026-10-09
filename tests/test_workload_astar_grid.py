import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.astar_grid import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok                      # equal path cost (A* and the Bellman-Ford variant)
    e = r.extra
    # same (f, node) tie-breaking on both sides: expansions and relaxations agree exactly
    assert e["astar"]["expansions_ratio"] == pytest.approx(1.0)
    assert e["astar"]["ratio_relax_vs_golden"] == pytest.approx(1.0)
    # mul_ops = relaxations + one g = f + (-h) per pop + one f = g + h per open candidate:
    # measured 1.860-1.863 (small, seeds 0-2, 2026-09-25)
    assert 1.5 < r.ratio_mul < 2.2
    # the global (min,+) relaxation never does less work than A*: measured 1.16-3.06x
    assert e["bellman_ford"]["ratio_relax_vs_astar_golden"] >= 1.0


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    e = r.extra
    assert e["astar"]["expansions_ratio"] == pytest.approx(1.0)
    assert e["astar"]["ratio_relax_vs_golden"] == pytest.approx(1.0)
    assert 1.5 < r.ratio_mul < 2.2
    # measured 11.8x (large, seed 0, 2026-09-25): the heuristic's pruning is lost
    assert e["bellman_ford"]["ratio_relax_vs_astar_golden"] > 5.0
