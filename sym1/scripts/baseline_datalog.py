#!/usr/bin/env python3
"""Compiled Souffle reachability on real graphs, independently checked with BFS."""
from collections import deque
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT/"sym1/results/build-souffle-local/src"


def command(args, target):
    start = time.perf_counter()
    result = subprocess.run(list(map(str, args)), text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=600)
    target.write_text(result.stdout)
    assert result.returncode == 0, (args, result.stdout[-4000:])
    return time.perf_counter()-start


def main():
    out = Path(os.environ["SYMFABRIC_RUN_DIR"])
    # claude: decision start — bounded source reachability preserves fixpoint behavior without all-pairs expansion.
    # Query counts, seeds, threads and repetitions below are diagnostic controls, not paper configurations.
    program = out/"reach.dl"
    program.write_text(""".decl edge(u:number,v:number)
.input edge
.decl seed(q:number,v:number)
.input seed
.decl reach(q:number,v:number)
reach(q,v) :- seed(q,v).
reach(q,v) :- reach(q,u),edge(u,v).
.output reach
""")
    source = out/"reach.cpp"
    command([BUILD/"souffle", "-j", "32", "-g", source, "-p", out/"default-profile.json", program],
            out/"generate.log")
    command([sys.executable, BUILD/"souffle-compile.py", "-v", "-o", out/"reach", source], out/"compile.log")
    rows = []
    for name in ("ca-GrQc", "email-Enron"):
        meta = json.loads((ROOT/f"sym1/data/{name}.json").read_text())
        adj = [[] for _ in range(meta["nodes"])]
        dataset = out/name
        dataset.mkdir()
        with (ROOT/f"sym1/data/{name}.el").open() as f, (dataset/"edge.facts").open("x") as g:
            for line in f:
                u,v = map(int, line.split())
                adj[u].append(v)
                adj[v].append(u)
                g.write(f"{u}\t{v}\n{v}\t{u}\n")
        for seed in (0, 1):
            sources = [meta["source"]]+random.Random(seed).sample(
                [i for i in range(len(adj)) if i != meta["source"]], 63)
            for qcount in (1, 8, 64):
                facts = dataset/f"seed{seed}-q{qcount}"
                facts.mkdir()
                (facts/"edge.facts").symlink_to(dataset/"edge.facts")
                (facts/"seed.facts").write_text("".join(f"{q}\t{v}\n" for q,v in enumerate(sources[:qcount])))
                golden = set()
                bfs_work = []
                for q,src in enumerate(sources[:qcount]):
                    seen = {src}
                    queue = deque([src])
                    levels, frontiers, edge_visits = 0, [], 0
                    while queue:
                        frontiers.append(len(queue))
                        levels += 1
                        for _ in range(len(queue)):
                            u = queue.popleft()
                            edge_visits += len(adj[u])
                            for v in adj[u]:
                                if v not in seen:
                                    seen.add(v)
                                    queue.append(v)
                    golden.update((q,v) for v in seen)
                    bfs_work.append(dict(source=src, levels=levels, frontiers=frontiers,
                                         reached=len(seen), edge_visits=edge_visits))
                for threads in (1, 8, 32):
                    for trial in range(3):
                        run = facts/f"t{threads}-r{trial}"
                        run.mkdir()
                        cmd = [out/"reach", "-F", facts, "-D", run, "-j", threads, "-p", run/"profile.json"]
                        wall = command(cmd, run/"stdout.log")
                        with (run/"reach.csv").open() as f:
                            actual = {tuple(map(int, line.split())) for line in f}
                        assert actual == golden, (name, seed, qcount, threads, trial)
                        command([BUILD/"souffleprof", run/"profile.json", "-c", "rel"], run/"relations.txt")
                        row = dict(dataset=name, seed=seed, queries=qcount, threads=threads,
                                   trial=trial, wall_s=wall, output_tuples=len(actual), pass_check=True,
                                   bfs_work=bfs_work, profile=str(run/"profile.json"),
                                   relations=str(run/"relations.txt"), command=list(map(str,cmd)),
                                   provenance="measured compiled CPU runtime; wall includes fact I/O; profile separates phases")
                        rows.append(row)
                        (out/"datalog.json").write_text(json.dumps(rows, indent=2)+"\n")
                    print(json.dumps({k:row[k] for k in ("dataset","seed","queries","threads","wall_s","output_tuples")}), flush=True)
    # claude: decision end


if __name__ == "__main__":
    main()
