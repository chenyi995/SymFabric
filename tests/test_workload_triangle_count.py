import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.triangle_count import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok                      # both join forms return the golden's triangle count
    e = r.extra
    assert e["triangles"]["triangles_materialized"] == e["golden"]["triangles"]
    # the join enumerates every oriented wedge; the golden probes min(d+(i), d+(j)) per edge:
    # measured wedge / probe ratio 1.36-1.40 (small, seeds 0-2, 2026-09-25)
    assert 1.0 < e["materialized"]["ratio_mul_vs_probes"] < 1.8
    assert 1.0 < e["fused"]["ratio_mul_vs_probes"] < 1.8
    assert 1.0 < r.ratio_mul < 1.8
    # the fused form never materializes the wedges: fewer candidates than the mask form
    assert e["fused"]["candidates"] < e["materialized"]["candidates"]


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    # measured 1.60 / 1.63 (large, seed 0, 2026-09-25)
    assert 1.0 < r.ratio_mul < 2.0
