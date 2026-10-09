#!/usr/bin/env python3
"""Normalize public SNAP inputs or run unchanged GAPBS kernels and their verifiers."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "sym1/data"
NAMES = ("ca-GrQc", "email-Enron", "roadNet-CA")


def prepare(out):
    import numpy as np
    rows = []
    for name in NAMES:
        source = DATA / f"{name}.txt.gz"
        edges = set()
        raw_lines = loops = 0
        with gzip.open(source, "rt") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                u, v = map(int, line.split()[:2])
                raw_lines += 1
                if u == v:
                    loops += 1
                    continue
                edges.add((min(u, v), max(u, v)))
        nodes = sorted({x for edge in edges for x in edge})
        remap = {u: i for i, u in enumerate(nodes)}
        arr = np.array(sorted((remap[u], remap[v]) for u, v in edges), dtype=np.int64)
        degree = np.bincount(arr.ravel(), minlength=len(nodes))
        target = DATA / f"{name}.el"
        with target.open("x") as f:
            np.savetxt(f, arr, fmt="%d", delimiter=" ")
        np.save(DATA / f"{name}.original_ids.npy", np.asarray(nodes))
        row = {"name": name, "url": f"https://snap.stanford.edu/data/{name}.txt.gz",
               "sha256_gz": hashlib.sha256(source.read_bytes()).hexdigest(),
               "sha256_el": hashlib.sha256(target.read_bytes()).hexdigest(),
               "raw_lines": raw_lines, "self_loop_lines_removed": loops,
               "nodes": len(nodes), "undirected_edges": len(arr), "source": int(degree.argmax()),
               "max_degree": int(degree.max()), "degree_median": float(np.median(degree)),
               "degree_p99": float(np.quantile(degree, .99)),
               "csr_min_bytes_64bit_offsets_32bit_neighbors": 8*(len(nodes)+1)+4*2*len(arr),
               "semantics": "simple undirected graph; sorted compact IDs; weights absent"}
        (DATA / f"{name}.json").write_text(json.dumps(row, indent=2)+"\n")
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "datasets.json").write_text(json.dumps(rows, indent=2)+"\n")


def run(out):
    rows = []
    revision = subprocess.check_output(["git", "-C", str(ROOT/"softwareworkload/upstream/gapbs"),
                                        "rev-parse", "HEAD"], text=True).strip()
    for name in NAMES:
        meta = json.loads((DATA/f"{name}.json").read_text())
        for kernel in ("bfs", "sssp", "tc"):
            # claude: decision start — compare fixed inputs at one, eight and 32 threads.
            # Experimental controls, not sourced hardware latency constants.
            for threads in (1, 8, 32):
                cmd = [str(ROOT/"softwareworkload/upstream/gapbs"/kernel),
                       "-f", str(DATA/f"{name}.el"), "-s", "-n", "5", "-v", "-a"]
                if kernel != "tc":
                    cmd += ["-r", str(meta["source"])]
                if kernel == "sssp":
                    # source: GAPBS benchmark/bench.mk uses delta=2 for non-road synthetic weights.
                    # SNAP roadNet has no distances; all three inputs use GAPBS synthetic weights.
                    cmd += ["-d", "2"]
                env = dict(os.environ, OMP_NUM_THREADS=str(threads))
                label = f"{name}-{kernel}-t{threads}"
                start = time.perf_counter()
                result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=600)
                elapsed = time.perf_counter()-start
                (out/f"{label}.log").write_text(result.stdout)
                trials = [float(x) for x in re.findall(r"Trial Time:\s+([0-9.]+)", result.stdout)]
                verified = result.stdout.count("PASS")
                row = {"dataset": name, "kernel": kernel, "threads": threads,
                       "revision": revision, "command": cmd, "exit_code": result.returncode,
                       "trial_s": trials, "process_wall_s": elapsed, "verified_trials": verified,
                       "pass": result.returncode == 0 and verified == 5 and len(trials) == 5,
                       "log": str(out/f"{label}.log"), "provenance": "measured CPU",
                       "weights": "GAPBS deterministic generated weights" if kernel == "sssp" else "unweighted"}
                rows.append(row)
                (out/"graph.json").write_text(json.dumps(rows, indent=2)+"\n")
                print(json.dumps(row), flush=True)
                if not row["pass"]:
                    raise RuntimeError(f"baseline or verification failed: {label}")
            # claude: decision end


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "run"))
    args = p.parse_args()
    out = Path(os.environ["SYMFABRIC_RUN_DIR"])
    (prepare if args.action == "prepare" else run)(out)
