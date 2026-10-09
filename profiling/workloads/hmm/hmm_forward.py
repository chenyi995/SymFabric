#!/usr/bin/env python3
"""Stand-in for Ctrl-G's HMM: dense HMM forward pass in PyTorch, batched over B sequences sharing the transition
matrix A (K x K) and emission table E (K x V): alpha_{t+1} = (alpha_t A) * e_{t+1}, renormalised per step.
--semiring maxplus runs the Viterbi (max, +) recursion in log space instead.  Q = B.  Untested (no GPU)."""
import argparse, time, torch  # noqa: E401
ap = argparse.ArgumentParser()
ap.add_argument("--states", type=int, default=256)
ap.add_argument("--vocab", type=int, default=50257)
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--semiring", choices=["sumprod", "maxplus"], default="sumprod")
ap.add_argument("--device", default="cuda")
a = ap.parse_args(); dev = torch.device(a.device); g = torch.Generator().manual_seed(0)
A = torch.rand(a.states, a.states, generator=g); A /= A.sum(1, keepdim=True)
E = torch.rand(a.states, a.vocab, generator=g); E /= E.sum(1, keepdim=True)
obs = torch.randint(0, a.vocab, (a.batch, a.steps), generator=g)
pi = torch.full((a.states,), 1.0 / a.states)
A, E, obs, pi = (x.to(dev) for x in (A, E, obs, pi))
sp = a.semiring == "sumprod"
if not sp:
    A, E, pi = A.log(), E.log(), pi.log()


def forward():
    e0 = E[:, obs[:, 0]].T                                   # B x K gather of emission columns (random access into E)
    alpha = pi[None, :] * e0 if sp else pi[None, :] + e0
    logz = torch.zeros(a.batch, device=dev)
    for t in range(1, a.steps):
        e = E[:, obs[:, t]].T
        if sp:
            alpha = (alpha @ A) * e                          # (+, x): one GEMM per step + elementwise product
            s = alpha.sum(1, keepdim=True); logz += s.log().squeeze(1); alpha = alpha / s
        else:
            alpha = (alpha[:, :, None] + A[None]).amax(1) + e   # (max, +) over a B x K x K temporary
    return logz + (alpha.sum(1).log() if sp else alpha.amax(1))


for _ in range(3): forward()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(a.iters):
    torch.cuda.nvtx.range_push("hmm_forward"); forward(); torch.cuda.nvtx.range_pop()
torch.cuda.synchronize(); dt = time.perf_counter() - t0
print(f"SYMFABRIC_METRIC W_bytes {(A.numel() + E.numel() + a.batch * a.states) * 4}")   # A + E + alpha, float32
print(f"SYMFABRIC_METRIC Q {a.batch}\nSYMFABRIC_METRIC hot_time_s {dt:.4f}")
print(f"SYMFABRIC_METRIC semiring_ops {a.iters * (a.steps - 1) * a.batch * a.states * a.states}")  # (mul, add) pairs
