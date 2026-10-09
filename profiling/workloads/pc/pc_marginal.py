#!/usr/bin/env python3
"""pyjuice marginal inference ((+, x) sum-product) on an HCLT circuit learned from random 28x28-style data.
Calls follow the pyjuice README (juice.structures.HCLT, juice.compile, pc(x)); the API was not checked on this
host: on failure compare with $SYMFABRIC_EXT/pyjuice/examples.  Untested here (no GPU).  Prints SYMFABRIC_METRIC
lines for parse_*.py; the NVTX range 'pc_marginal' brackets each timed forward pass."""
import argparse, time, torch  # noqa: E401
ap = argparse.ArgumentParser()
ap.add_argument("--num-latents", type=int, default=64)
ap.add_argument("--vars", type=int, default=784)
ap.add_argument("--cats", type=int, default=256)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--samples", type=int, default=4096, help="random samples the HCLT structure is learned from")
ap.add_argument("--missing", type=float, default=0.5, help="fraction of variables marginalised out per query")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--device", default="cuda")
a = ap.parse_args(); dev = torch.device(a.device); torch.manual_seed(0)
import pyjuice as juice  # noqa: E402
data = torch.randint(0, a.cats, (a.samples, a.vars), dtype=torch.long)
ns = juice.structures.HCLT(data.float().to(dev), num_bins=32, sigma=0.5 / 32, num_latents=a.num_latents, chunk_size=32)
pc = juice.compile(ns) if hasattr(juice, "compile") else juice.TensorCircuit(ns)
pc.to(dev)
x = data[: a.batch].to(dev)
mask = torch.rand(x.shape, device=dev) < a.missing
try:  # marginal query: masked variables are summed out (keyword believed to be missing_mask; verify)
    pc(x, missing_mask=mask); mode, run = "marginal_missing_mask", (lambda: pc(x, missing_mask=mask))
except TypeError:
    mode, run = "full_evidence_likelihood", (lambda: pc(x))
for _ in range(3): run()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(a.iters):
    torch.cuda.nvtx.range_push("pc_marginal"); run(); torch.cuda.nvtx.range_pop()
torch.cuda.synchronize(); dt = time.perf_counter() - t0
W = sum(p.numel() * p.element_size() for p in pc.parameters()) + sum(b.numel() * b.element_size() for b in pc.buffers())
print(f"SYMFABRIC_METRIC W_bytes {W}")   # parameters + compiled layer buffers (edge / index tables)
print(f"SYMFABRIC_METRIC Q {a.batch}")    # queries evaluated against the same circuit
print(f"SYMFABRIC_METRIC mode {mode}\nSYMFABRIC_METRIC hot_time_s {dt:.4f}\nSYMFABRIC_METRIC num_latents {a.num_latents}")
