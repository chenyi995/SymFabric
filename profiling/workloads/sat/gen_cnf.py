#!/usr/bin/env python3
"""Random k-SAT DIMACS generator (fallback when no SAT Competition instance is available) and clause-DB size.
  gen_cnf.py --n 350 --ratio 4.26 --out inst.cnf   uniform random 3-SAT (hardness peaks near ratio 4.26)
  gen_cnf.py --stats inst.cnf                       prints 4 x (literal occurrences) = W_bytes lower bound
4 B per literal is the 32-bit literal encoding; clause headers, watch lists and learned clauses are not counted,
so the value is a lower bound (Kissat's maximum-resident-set-size statistic is the upper bound)."""
import argparse, random, sys  # noqa: E401
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int); ap.add_argument("--ratio", type=float, default=4.26); ap.add_argument("--k", type=int, default=3)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out"); ap.add_argument("--stats")
a = ap.parse_args()
if a.stats:
    lits = clauses = 0
    with open(a.stats, errors="replace") as f:
        for line in f:
            toks = line.split()
            if not toks or toks[0] in ("c", "p", "%"): continue
            lits += sum(1 for t in toks if t != "0"); clauses += toks.count("0")
    print(4 * lits)
    print(f"clauses={clauses} literal_occurrences={lits}", file=sys.stderr)
else:
    rng = random.Random(a.seed); m = int(a.ratio * a.n)
    with open(a.out, "w") as f:
        f.write(f"c random {a.k}-SAT n={a.n} m={m} seed={a.seed}\np cnf {a.n} {m}\n")
        for _ in range(m):
            f.write(" ".join(str(v if rng.random() < 0.5 else -v) for v in rng.sample(range(1, a.n + 1), a.k)) + " 0\n")
