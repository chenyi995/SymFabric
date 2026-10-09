# Probabilistic circuits (pyjuice) — notes

**Hot loop.** One bottom-up pass over a layered smooth, decomposable circuit: product nodes multiply their
children, sum nodes take weighted sums (parameters theta). Algebra: (+, x) over probabilities, executed in log
space (log-sum-exp for +). pyjuice compiles the DAG into layers of block-sparse matrix operations (the plan's
`h_{l+1} = W_l (x)_S h_l`): static sparsity, shared across the batch of queries (Q = batch). Marginal inference
sets the masked input distributions to 1 and runs the same pass.

**Software counters.** `pc_marginal.py` prints W (parameter + buffer bytes), Q, `mode` (whether the
`missing_mask` marginal path was accepted) and the timed forward time. Node/edge counts live in the structure
object `ns` (accessor per the pyjuice docs; not printed because the API was not checked).

**What to look for.** nsys: `pc_marginal` NVTX range vs wall (S); sum/product layer kernels vs elementwise
glue. ncu: DRAM utilisation and L2 hit rate of the layer kernels — REASON reports 60.8 % DRAM-BW utilisation and
42.4 % L1 hit rate for its marginal kernel (paper note, Table II); the two scales show whether the compiled
parameter tables outgrow L2 (W column vs the GPU's L2 size).
