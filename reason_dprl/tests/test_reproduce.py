"""Check artifact integrity/non-overwrite and rejection of incomplete reruns."""
import copy
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reproduce
import sources


def bundle_fixture(tmp_path, members):
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, value in members:
            member = tarfile.TarInfo(name)
            member.size = len(value)
            tf.addfile(member, io.BytesIO(value))
    return archive, {"sha256": reproduce.sha(archive), "bytes": archive.stat().st_size,
                     "file_count": len(members), "roots": ["reason_dprl/results/reference"]}


def test_restore_idempotent_and_preflights_conflict(tmp_path):
    root = tmp_path / "checkout"
    archive, spec = bundle_fixture(tmp_path, [("reason_dprl/results/reference/a", b"a"),
                                             ("reason_dprl/results/reference/b", b"b")])
    reproduce.restore(root, archive, spec)
    reproduce.restore(root, archive, spec)
    folder = root / "reason_dprl/results/reference"
    (folder / "a").unlink()
    (folder / "b").write_bytes(b"user edit")
    with pytest.raises(ValueError, match="differs"):
        reproduce.restore(root, archive, spec)
    assert not (folder / "a").exists()  # No partial writes before conflict discovery.
    assert (folder / "b").read_bytes() == b"user edit"


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "unlocked/file",
                                "reason_dprl/results/reference/../escape"])
def test_restore_rejects_bad_paths(tmp_path, name):
    archive, spec = bundle_fixture(tmp_path, [(name, b"payload")])
    with pytest.raises(ValueError, match="Invalid"):
        reproduce.restore(tmp_path / "checkout", archive, spec)


def test_restore_rejects_bad_hash_and_symlink_parent(tmp_path):
    archive, spec = bundle_fixture(tmp_path, [("reason_dprl/results/reference/a", b"a")])
    root = tmp_path / "checkout"
    with pytest.raises(ValueError, match="Content mismatch"):
        reproduce.restore(root, archive, {**spec, "sha256": "bad"})
    assert not root.exists()
    root.mkdir()
    (root / "reason_dprl").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Conflicting path"):
        reproduce.restore(root, archive, spec)


def test_source_subset_keeps_dependencies_and_only_their_links():
    lock = {"sources": [
        {"name": "a", "checkout": "src/a", "dependencies": [{"checkout": "src/a/dep"}]},
        {"name": "b", "checkout": "src/b", "dependencies": []}],
        "links": [{"entry": "a", "checkout": "src/a"},
                  {"entry": "dep", "checkout": "src/a/dep"},
                  {"entry": "b", "checkout": "src/b"}]}
    before = copy.deepcopy(lock)
    selected = sources.subset(lock, ["a"])
    assert [x["checkout"] for x in sources.all_checkouts(selected)] == ["src/a", "src/a/dep"]
    assert [x["entry"] for x in selected["links"]] == ["a", "dep"]
    assert lock == before
    with pytest.raises(ValueError, match="Unknown"):
        sources.subset(lock, ["missing"])


def test_compare_rejects_partial_and_changed_results(tmp_path):
    ref, actual = tmp_path / "ref", tmp_path / "actual"
    ref.mkdir(); actual.mkdir()
    provenance = {"dprl_commit": "commit", "checkpoint_sha256": "checkpoint",
                  "precision": {"tf32": False},
                  "config": {"max_steps": 20, "n_envs": 64, "fixed_batch_size": 64}}
    repeat = {"passed": True, "hashes": ["expected"] * 5}
    variant = {"median_ms": 1.0, "samples_ms": [1.0] * 20,
               "correctness": {"passed": True}, "repeatability": repeat,
               "output_sha256": "expected"}
    row = {"index": 0, "query_sha256": "query", "native_repeatability": repeat,
           "variants": {name: copy.deepcopy(variant) for name in reproduce.VARIANTS}}
    result = {"dataset": "family", "batch": 64, "batches": [row]}
    for folder in [ref, actual]:
        (folder / "provenance.json").write_text(json.dumps(provenance))
        (folder / "benchmark.json").write_text(json.dumps(result))
    reproduce.compare_run(actual, ref)
    result["batches"][0]["variants"].pop("fast_fused_graphs")
    (actual / "benchmark.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Incomplete set"):
        reproduce.compare_run(actual, ref)
    result = json.loads((ref / "benchmark.json").read_text())
    result["batches"][0]["query_sha256"] = "different query"
    (actual / "benchmark.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Query identity"):
        reproduce.compare_run(actual, ref)
