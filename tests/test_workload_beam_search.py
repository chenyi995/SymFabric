import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from softwareworkload import _harness as H
from softwareworkload.beam_search import workload as W


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_matches_golden(seed):
    r = H.run(W, "small", seed)
    assert r.ok
    assert r.native_ops > 0
    # one combine candidate per (beam entry, successor): exactly the golden's expansions
    assert r.ratio_mul == pytest.approx(1.0)
    assert r.ratio_sr == pytest.approx(1.0)          # no merges: (t, b) keys are unique
    # measured 2026-09-25: ratio_cand 2.033 = expansions + select pass + K re-index scans
    assert r.ratio_cand <= 2.1
    assert r.extra["select_calls"] == W.SCALES["small"]["L"]


def test_large_matches_golden():
    r = H.run(W, "large", 0)
    assert r.ok
    assert r.ratio_mul == pytest.approx(1.0)
    assert r.ratio_cand <= 2.1


def test_sequences_are_distinct_and_rank_ordered():
    inst = W.make_instance("small", 0)
    res, _ = W.native(inst)
    seqs = [tuple(s) for _, s in res]
    assert len(set(seqs)) == len(seqs)
    scores = [sc for sc, _ in res]
    assert scores == sorted(scores, reverse=True)
    assert all(len(s) == inst["L"] + 1 and s[0] == inst["start"] for s in seqs)
