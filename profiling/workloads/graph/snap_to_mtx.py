#!/usr/bin/env python3
"""SNAP edge list (src<TAB>dst, '#' comments, .gz allowed) -> Matrix Market coordinate file for the LAGraph demos:
1-based compacted ids; --symmetric keeps one copy per undirected edge in the lower triangle; --weights writes
random integer weights 1..255 (for sssp_demo).  Also writes '<dst>.stats' with the node and edge counts."""
import argparse, gzip, random  # noqa: E401
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--symmetric", action="store_true"); ap.add_argument("--weights", action="store_true")
a = ap.parse_args(); rng = random.Random(0); ids, edges = {}, set()
with (gzip.open if a.src.endswith(".gz") else open)(a.src, "rt") as f:
    for ln in f:
        if ln[:1] in "#\n": continue
        u, v = ln.split()[:2]
        u, v = ids.setdefault(u, len(ids) + 1), ids.setdefault(v, len(ids) + 1)
        if a.symmetric:
            if u == v: continue
            u, v = max(u, v), min(u, v)          # lower triangle: row >= column
        edges.add((u, v))
with open(a.dst, "w") as f:
    f.write(f"%%MatrixMarket matrix coordinate {'integer' if a.weights else 'pattern'} "
            f"{'symmetric' if a.symmetric else 'general'}\n{len(ids)} {len(ids)} {len(edges)}\n")
    for u, v in sorted(edges):
        f.write(f"{u} {v} {rng.randint(1, 255)}\n" if a.weights else f"{u} {v}\n")
open(a.dst + ".stats", "w").write(f"nodes {len(ids)} edges {len(edges)}\n")
print(f"nodes {len(ids)} edges {len(edges)}")
