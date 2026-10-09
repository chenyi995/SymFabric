#!/usr/bin/env python3
"""Verify/restore frozen inputs, submit Slurm runs, and compare output identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "reason_dprl"
VARIANTS = {"native_compiled", "native_graph", "fast_policy_graphs",
            "fast_active_graphs", "fast_fused_graphs"}


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return digest(stream)


def verify_file(root, item):
    path = root / item["path"]
    if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
        raise ValueError(f"Content mismatch: {path}")


def restore(root, bundle, spec):
    """Preflight every member before writing; never replace existing content."""
    verify_file(bundle.parent, {**spec, "path": bundle.name})
    root = root.resolve()
    allowed = [PurePosixPath(p) for p in spec["roots"]]
    pending, seen = [], set()
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive.getmembers():
            rel = PurePosixPath(member.name)
            if (not member.isfile() or rel.is_absolute() or ".." in rel.parts
                    or not any(p in rel.parents for p in allowed)
                    or rel in seen):
                raise ValueError(f"Invalid/duplicate archive member: {member.name}")
            seen.add(rel)
            dest = root / rel
            for p in [dest, *dest.parents]:
                if p == root:
                    break
                if p.is_symlink() or (p != dest and p.exists() and not p.is_dir()):
                    raise ValueError(f"Conflicting path: {p}")
            if dest.exists():
                if not dest.is_file() or dest.stat().st_size != member.size:
                    raise ValueError(f"Existing artifact differs: {dest}")
                with archive.extractfile(member) as source:
                    if sha(dest) != digest(source):
                        raise ValueError(f"Existing artifact differs: {dest}")
            else:
                pending.append((member, dest))
        if len(seen) != spec["file_count"]:
            raise ValueError("Artifact file count does not match lock")
        for member, dest in pending:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, dest.open("xb") as target:
                shutil.copyfileobj(source, target)
    print(f"PASS: restored {len(pending)} files; {len(seen)-len(pending)} identical files retained")


def check(lock, datasets, environment=True):
    subprocess.run([sys.executable, str(PROJECT / "scripts/sources.py"), "check",
                    "--only", "DPrL-AAAI", "--only", "DAG_Processor"], check=True)
    for item in lock["inputs"]:
        if item["dataset"] in datasets:
            verify_file(ROOT, item)
    for dataset in datasets:
        verify_file(ROOT, lock["checkpoints"][dataset])
    expected = lock["environment"]
    if sha(ROOT / expected["requirements_file"]) != expected["requirements_sha256"]:
        raise ValueError("Requirements lock changed")
    if environment:
        # Metadata only: no PyTorch/CUDA import on the GPU-less login node.
        program = (
            "import importlib.metadata as m,json,platform,re; "
            "print(json.dumps({'python':platform.python_version(),"
            "'packages':{re.sub(r'[-_.]+','-',d.metadata['Name'].lower()):d.version "
            "for d in m.distributions()}}))"
        )
        actual = json.loads(subprocess.check_output(
            [str(PROJECT / ".venv/bin/python"), "-c", program], text=True))
        if actual["python"] != expected["python"]:
            raise ValueError(f"Python mismatch: {actual['python']}")
        for line in (ROOT / expected["requirements_file"]).read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            package, version = line.split("==")
            if actual["packages"].get(re.sub(r"[-_.]+", "-", package.lower())) != version:
                raise ValueError(f"Package mismatch: {line}")
    print("PASS: pinned datasets, frozen checkpoints, and requested environment checks")


def submit(lock, args):
    datasets = args.dataset or ["family", "wn18rr"]
    check(lock, datasets)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Use a new output directory: {output}")
    if args.smoke:
        batches, repeat = 1, 3
    else:
        batches, repeat = 3, 20
    snapshot = output / "harness.tar.gz"
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)
        subprocess.run([sys.executable, str(PROJECT / "scripts/snapshot_harness.py"),
                        str(snapshot)], check=True)
    manifest = {"smoke": args.smoke, "batches": batches, "repeat": repeat, "jobs": []}
    for dataset in datasets:
        # Unique node-local caches avoid stale/removed shared-memory caches.
        key = hashlib.sha256(str(output).encode()).hexdigest()[:12]
        env_updates = {
            "SYMFABRIC_CACHE_BASE": f"/tmp/symfabric-repro-{os.getuid()}-{key}-{dataset}",
            "SYMFABRIC_HARNESS_ARCHIVE": str(snapshot),
        }
        command = ["sbatch", "--parsable", f"--partition={args.partition}",
                   f"--nodelist={args.node}", f"--output={output}/{dataset}.slurm.log",
                   str(PROJECT / "scripts/run_gpu.sbatch"),
                   str(PROJECT / "scripts/optimize_kg_fast.py"), "--dataset", dataset,
                   "--checkpoint", str(ROOT / lock["checkpoints"][dataset]["path"]),
                   "--output", str(output / dataset), "--batch", "64",
                   "--batches", str(batches), "--repeat", str(repeat)]
        if args.dry_run:
            print(shlex.join(["env", *(f"{k}={v}" for k, v in env_updates.items()), *command]))
            continue
        job = subprocess.check_output(command, cwd=ROOT,
                                      env={**os.environ, **env_updates}, text=True).strip()
        manifest["jobs"].append({"dataset": dataset, "job": job,
                                  "command": command, "environment": env_updates})
        (output / "submission.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"SUBMITTED: {dataset} job={job}; log={output}/{dataset}.slurm.log")


def compare_run(actual_dir, reference_dir, smoke=False):
    actual = json.loads((actual_dir / "benchmark.json").read_text())
    ref = json.loads((reference_dir / "benchmark.json").read_text())
    ap = json.loads((actual_dir / "provenance.json").read_text())
    rp = json.loads((reference_dir / "provenance.json").read_text())
    for key in ("dprl_commit", "checkpoint_sha256", "precision"):
        if ap[key] != rp[key]:
            raise ValueError(f"Provenance mismatch: {key}")
    for key in ("max_steps", "n_envs", "fixed_batch_size"):
        if ap["config"][key] != rp["config"][key]:
            raise ValueError(f"Configuration mismatch: {key}")
    if actual["dataset"] != ref["dataset"] or actual["batch"] != ref["batch"]:
        raise ValueError("Dataset/batch mismatch")
    n = 1 if smoke else len(ref["batches"])
    if len(actual["batches"]) != n:
        raise ValueError(f"Incomplete run: expected {n} batches")
    for a, b in zip(actual["batches"], ref["batches"]):
        if a["index"] != b["index"] or a["query_sha256"] != b["query_sha256"]:
            raise ValueError("Query identity/order mismatch")
        expected = b["native_repeatability"]["hashes"][0]
        native = a["native_repeatability"]
        if not native["passed"] or len(native["hashes"]) != 5 or set(native["hashes"]) != {expected}:
            raise ValueError("Native output identity/repeatability mismatch")
        if set(a["variants"]) != VARIANTS:
            raise ValueError("Incomplete set of variants")
        for name, result in a["variants"].items():
            if len(result["samples_ms"]) != (3 if smoke else 20):
                raise ValueError(f"Incomplete timing samples: {name}")
            if name.startswith("fast_"):
                repeat = result["repeatability"]
                if (not result["correctness"]["passed"] or not repeat["passed"]
                        or len(repeat["hashes"]) != 5 or set(repeat["hashes"]) != {expected}
                        or result["output_sha256"] != expected):
                    raise ValueError(f"Optimized correctness/output mismatch: {name}")
    print(f"PASS: {actual['dataset']}, {n} batches; query/output identities and in-run audits")
    for name in sorted(VARIANTS):
        measured = statistics.mean(b["variants"][name]["median_ms"] for b in actual["batches"])
        reference = statistics.mean(b["variants"][name]["median_ms"] for b in ref["batches"][:n])
        print(f"  {name}: {measured:.3f} ms; reference {reference:.3f} ms (not an equality target)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("--include-mnist", action="store_true")
    p.add_argument("--skip-env", action="store_true")
    p = sub.add_parser("restore")
    p.add_argument("bundle", type=Path)
    p = sub.add_parser("submit")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", action="append", choices=["family", "wn18rr"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--partition", default="athena")
    p.add_argument("--node", default="node2")
    p = sub.add_parser("compare")
    p.add_argument("output", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dataset", action="append", choices=["family", "wn18rr"])
    args = parser.parse_args()
    lock = json.loads((PROJECT / "repro.lock.json").read_text())
    try:
        if args.command == "restore":
            restore(ROOT, args.bundle.resolve(), lock["bundle"])
        elif args.command == "check":
            check(lock, ["family", "wn18rr"] + (["mnist"] if args.include_mnist else []),
                  environment=not args.skip_env)
        elif args.command == "submit":
            submit(lock, args)
        else:
            for dataset in args.dataset or ["family", "wn18rr"]:
                compare_run(args.output / dataset, ROOT / lock["reference_runs"][dataset], args.smoke)
    except (OSError, ValueError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
