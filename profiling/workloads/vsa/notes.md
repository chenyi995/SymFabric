# VSA / hyperdimensional computing (torchhd) — notes

**Hot loop.** Cleanup: sim = Q Mem^T ((+, x) inner products of B queries against M item vectors, d = 10 000)
followed by argmax per query (max selection): a dense GEMM feeding a SELECT. Resonator network: factorises
s = a (x) b (x) c by iterating, per factor, unbind (elementwise x with the other two estimates), projection onto
the codebook (two GEMMs) and sign (bundling threshold). Q = batch sharing memory and codebooks. Structure: dense,
static, high arithmetic intensity at large B — the workload closest to a plain systolic array.

**Software counters.** `vsa_cleanup.py` prints W (memory + codebooks), Q, per-phase times, cleanup and
factorisation accuracy (sanity: near 1.0 at 20 % noise; otherwise the vectors or sign/unbind steps are wrong,
not the profile).

**What to look for.** nsys: GEMM vs argmax/reduce kernels per NVTX range; one row per phase and scale. ncu:
high sm_throughput_pct and low dram_bw_util_pct at M = 10 000 (memory stays in L2); at M = 50 000 the 2 GB memory
streams from DRAM on every cleanup, so arithmetic intensity is bounded by 2B flop per 4-byte element — the
transition that decides whether the fabric's dense mode earns its area.
