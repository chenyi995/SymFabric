#!/usr/bin/env python3
"""VSA cleanup memory and resonator network on d-dimensional bipolar (MAP) hypervectors.  torchhd (pip install
torch-hd) generates the vectors when installed, plain torch otherwise; the hot loops are written out so the
profile does not depend on torchhd internals.  Q = batch of queries sharing the codebooks.  Untested (no GPU)."""
import argparse, time, torch  # noqa: E401
ap = argparse.ArgumentParser()
ap.add_argument("--dim", type=int, default=10_000)
ap.add_argument("--items", type=int, default=10_000, help="cleanup memory size M")
ap.add_argument("--factors", type=int, default=100, help="codebook size F per factor (3 factors)")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--noise", type=float, default=0.2, help="fraction of flipped components per query")
ap.add_argument("--steps", type=int, default=20, help="resonator iterations")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--device", default="cuda")
a = ap.parse_args(); dev = torch.device(a.device); torch.manual_seed(0)
try:
    import torchhd
    rand = lambda n, d: torchhd.random(n, d, "MAP").as_subclass(torch.Tensor).to(dev)  # noqa: E731  components in {-1, +1}
    vecs = f"torchhd_{torchhd.__version__}"
except ImportError:
    rand = lambda n, d: torch.randint(0, 2, (n, d), device=dev).float() * 2 - 1  # noqa: E731
    vecs = "plain_torch"
mem = rand(a.items, a.dim); A, Bk, C = (rand(a.factors, a.dim) for _ in range(3)); B = a.batch
tgt = torch.randint(0, a.items, (B,), device=dev)
q = mem[tgt] * ((torch.rand(B, a.dim, device=dev) < a.noise).float() * -2 + 1)   # noisy copies of stored items
ia, ib, ic = (torch.randint(0, a.factors, (B,), device=dev) for _ in range(3))
s = A[ia] * Bk[ib] * C[ic]                                                       # composite = bind of one item per factor


def cleanup():      # (+, x) inner products against the memory, then argmax (max selection)
    return (q @ mem.T).argmax(1)


def resonator():    # per factor: unbind the other estimates, project onto the codebook (two GEMMs), sign
    ea, eb, ec = (cb.sum(0, keepdim=True).sign().expand(B, -1) for cb in (A, Bk, C))
    for _ in range(a.steps):
        ea = ((s * eb * ec) @ A.T @ A).sign(); eb = ((s * ea * ec) @ Bk.T @ Bk).sign(); ec = ((s * ea * eb) @ C.T @ C).sign()
    return (ea @ A.T).argmax(1), (eb @ Bk.T).argmax(1), (ec @ C.T).argmax(1)


for _ in range(2): cleanup(); resonator()
torch.cuda.synchronize(); t = {}
for name, fn in (("vsa_cleanup", cleanup), ("vsa_resonator", resonator)):
    t0 = time.perf_counter()
    for _ in range(a.iters):
        torch.cuda.nvtx.range_push(name); fn(); torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(); t[name] = time.perf_counter() - t0
ra, rb, rc = resonator()
print(f"SYMFABRIC_METRIC W_bytes {(mem.numel() + 3 * A.numel()) * mem.element_size()}\nSYMFABRIC_METRIC Q {B}")
print(f"SYMFABRIC_METRIC cleanup_time_s {t['vsa_cleanup']:.4f}\nSYMFABRIC_METRIC resonator_time_s {t['vsa_resonator']:.4f}")
print(f"SYMFABRIC_METRIC cleanup_accuracy {(cleanup() == tgt).float().mean().item():.3f}")   # sanity, expect ~1.0
print(f"SYMFABRIC_METRIC resonator_accuracy {((ra == ia) & (rb == ib) & (rc == ic)).float().mean().item():.3f}")
print(f"SYMFABRIC_METRIC vectors_from {vecs}")
