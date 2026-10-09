# HMM / constrained decoding (Ctrl-G stand-in) — notes

**Hot loop.** Forward recursion alpha_{t+1}(j) = e_{t+1}(j) * sum_i alpha_t(i) A(i, j): a dense (+, x)
matrix-vector product per step (a B x K by K x K GEMM over the batch) followed by an elementwise product with the
gathered emission column. Viterbi replaces (+, x) by (max, +) in log space; the stand-in materialises a
B x K x K temporary reduced by amax (memory-bound by construction, hence the smaller batch at the large scale).
Ctrl-G runs such HMM passes per generated token to score whether a DFA constraint can still be met, interleaved
with the LLM's decoding step. Q = batch of sequences sharing A and E.

**Software counters.** `hmm_forward.py` prints W (A + E + alpha, float32), Q, hot time and the number of
(mul, add) pairs (`semiring_ops`; with `hot_time_s` this is a derived throughput, not a measurement).

**What to look for.** nsys: `hmm_forward` range vs wall (S); GEMM vs gather/elementwise kernels (`--share`).
ncu: at K = 256 the model fits in L2 (high l2_hit_pct, low DRAM utilisation); at K = 4096 A is 64 MB and E about
800 MB float32, so DRAM utilisation and the step kernel's arithmetic intensity move to the memory-bound regime —
direct evidence for H7. Option A (Ctrl-G) additionally shows the LLM's share of the timeline.
