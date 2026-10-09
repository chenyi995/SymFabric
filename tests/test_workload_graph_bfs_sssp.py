import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.graph_bfs_sssp import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    e = r.extra
    # (b) Dijkstra-with-select and (c) BFS relax exactly the golden's edges
    assert e["dijkstra"]["ratio_relax_vs_dijkstra"] == pytest.approx(1.0)
    assert e["bfs"]["ratio_relax_vs_bfs"] == pytest.approx(1.0)
    # (a) frontier relaxation re-relaxes vertices: measured 1.79-2.08 (small, seeds 0-2, 2026-09-25)
    assert 1.0 < e["frontier"]["ratio_relax_vs_dijkstra"] < 3.0
    # aggregate over the three programs: measured ratio_mul 1.26-1.36
    assert 1.0 < r.ratio_mul < 1.6
    # the pop cost of (b) (select + delete over the whole open list) is O(|open|) per vertex:
    # measured 240x the relaxations at small scale; it must stay counted
    assert 100 < e["dijkstra"]["ratio_pop_cand_vs_dijkstra"] < 500


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    e = r.extra
    assert e["dijkstra"]["ratio_relax_vs_dijkstra"] == pytest.approx(1.0)
    assert e["bfs"]["ratio_relax_vs_bfs"] == pytest.approx(1.0)
    # measured 2.38 (large, seed 0, 2026-09-25)
    assert 1.0 < e["frontier"]["ratio_relax_vs_dijkstra"] < 4.0
    assert 1.0 < r.ratio_mul < 2.0
