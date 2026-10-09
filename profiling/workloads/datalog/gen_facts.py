#!/usr/bin/env python3
"""Random EDB generator for the Datalog workload (Souffle); writes TSV fact files into --out.
  --nodes N --edges M   edge.facts for tc.dl (transitive closure)
  --vars V --stmts K    alloc/assign/load/store.facts for andersen.dl (points-to; heap objects H = V/10)
Prints the EDB tuple count; run.sh adds the IDB tuples to derive the W_bytes lower bound."""
import argparse, random  # noqa: E401
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--nodes", type=int); ap.add_argument("--edges", type=int)
ap.add_argument("--vars", type=int); ap.add_argument("--stmts", type=int)
a = ap.parse_args(); rng = random.Random(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def write(name, rows):
    (out / f"{name}.facts").write_text("".join(f"{x}\t{y}\n" for x, y in sorted(rows)))
    return len(rows)


n = 0
if a.nodes:
    n += write("edge", {(rng.randrange(a.nodes), rng.randrange(a.nodes)) for _ in range(a.edges)})
if a.vars:
    V, H = a.vars, max(1, a.vars // 10)
    kinds = {k: set() for k in ("alloc", "assign", "load", "store")}
    for _ in range(a.stmts):  # statement kinds drawn uniformly
        k = rng.choice(list(kinds)); kinds[k].add((rng.randrange(V), rng.randrange(H if k == "alloc" else V)))
    for k, rows in kinds.items(): n += write(k, rows)
print(f"edb_tuples {n}")
