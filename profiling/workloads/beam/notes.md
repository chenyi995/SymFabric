# Beam / tree search over an LLM (Hugging Face beam search stand-in) — notes

**Hot loop.** Per decoding step: (neural) one transformer forward over batch x beams candidates; (symbolic)
log-softmax of the logits, beam score = parent score + token log-prob for all batch x beams x vocab extensions,
top-2k selection per batch item, gather/reorder of the KV cache by the selected parents. Algebra of the
selection: (max, +) over log-probabilities with a k-best SELECT instead of a plain max. Tree-of-thought's Q
candidates sharing one verifier correspond to the Q = batch x beams candidates sharing weights and cache prefixes.

**Software counters.** `beam_generate.py` prints W (weights + KV cache at final length), Q, decode time, tokens
per second, output shape; transformers exposes `output_scores` / `return_dict_in_generate` for per-step scores.

**What to look for.** nsys `stats_cuda_gpu_kern_sum.csv`: share of topk / sort / gather / index_select
(selection) vs gemm / cutlass (matmul) — both shares land in the row's notes. The small run (beams 4, batch 1) is
launch-bound (many tiny kernels); the large run (beams 16, batch 8) shows whether selection time grows with Q.
ncu on the selection kernels: sectors per request of the cache-reorder gather, DRAM utilisation of top-k.
tree-of-thought-llm (princeton-nlp) itself was judged too heavy for this kit; its search loop has the same shape.
