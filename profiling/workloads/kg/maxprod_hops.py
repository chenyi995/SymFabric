#!/usr/bin/env python3
"""Stand-in for NBFNet's multi-hop reasoning: L-hop (max, x) propagation x_{k+1} = max(x_k, A (x) x_k) over a
weighted graph, batched over B source entities sharing the graph (Q = B).  --semiring boolean (OR, AND) and
sumprod (+, x) give reachability and path counting.  Edge list: random or a SNAP txt(.gz).  Untested (no GPU)."""
import argparse, gzip, time, torch  # noqa: E401
ap = argparse.ArgumentParser()
ap.add_argument("--nodes", type=int, default=100_000)
ap.add_argument("--edges", type=int, default=1_000_000)
ap.add_argument("--edges-file", default=None, help="SNAP edge list (src<TAB>dst, '#' comments), .gz allowed")
ap.add_argument("--hops", type=int, default=4)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--semiring", choices=["maxprod", "boolean", "sumprod"], default="maxprod")
ap.add_argument("--device", default="cuda")
a = ap.parse_args(); dev = torch.device(a.device); g = torch.Generator().manual_seed(0)
if a.edges_file:
    with (gzip.open if a.edges_file.endswith(".gz") else open)(a.edges_file, "rt") as f:
        e = torch.tensor([tuple(map(int, ln.split()[:2])) for ln in f if ln[:1] not in "#\n"], dtype=torch.long)
    _, e = torch.unique(e, return_inverse=True)                     # compact ids
    src, dst, n = e[:, 0], e[:, 1], int(e.max()) + 1
else:
    n = a.nodes; src, dst = (torch.randint(0, n, (a.edges,), generator=g) for _ in range(2))
w = torch.rand(src.numel(), generator=g) * 0.9 + 0.1                 # weights in (0.1, 1]; 1 for Boolean
if a.semiring == "boolean": w = torch.ones_like(w)
src, dst, w = src.to(dev), dst.to(dev), w.to(dev)
B = a.batch
x0 = torch.zeros(n, B, device=dev)
x0[torch.randint(0, n, (B,), generator=g), torch.arange(B)] = 1.0   # one-hot source per query
idx = dst[:, None].expand(-1, B)
red = "sum" if a.semiring == "sumprod" else "amax"


def hops():
    x = x0
    for _ in range(a.hops):
        msg = x[src] * w[:, None]                                    # (x): gather along edges, M x B
        y = torch.zeros_like(x).scatter_reduce_(0, idx, msg, reduce=red, include_self=True)  # (+): reduce by dst
        x = x + y if a.semiring == "sumprod" else torch.maximum(x, y)   # closure up to L hops
    return x


for _ in range(2): hops()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(a.iters):
    torch.cuda.nvtx.range_push("kg_hops"); hops(); torch.cuda.nvtx.range_pop()
torch.cuda.synchronize(); dt = time.perf_counter() - t0
print(f"SYMFABRIC_METRIC W_bytes {src.numel() * (8 + 8 + 4) + n * B * 4}")   # int64 src, dst; f32 w; state N x B
print(f"SYMFABRIC_METRIC Q {B}\nSYMFABRIC_METRIC nodes {n}\nSYMFABRIC_METRIC edges {src.numel()}")
print(f"SYMFABRIC_METRIC hot_time_s {dt:.4f}\nSYMFABRIC_METRIC semiring_ops {a.iters * a.hops * src.numel() * B}")
