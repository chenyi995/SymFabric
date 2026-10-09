# beam_search — beam search decoding over a sparse Markov language model

**Category**: search · **Semiring(s)**: viterbi (max,+) · **Execution structure**: select-loop
**Module**: `softwareworkload/beam_search/workload.py` · **Test**: `tests/test_workload_beam_search.py`

## 1. Problem and reference algorithm
A Markov LM over V tokens with a transition relation T(s,t) = log P(t | s) that keeps 32
successors per token (top-k sparsified; generated directly with Gaussian logits, log-softmax
normalised over the 32). Decoding starts at token 0, runs L = 20 steps with beam width K, and
returns the K best hypotheses (score, token sequence). Hypotheses are never merged: two beams
that end in the same token are different prefixes. Ties are broken by score descending, then
token ascending, then parent beam ascending, on both sides. `native()` is plain Python beam
search with `heapq.nsmallest` on the key (-score, token, parent beam) and counts one candidate
expansion per (beam entry, successor) pair per step. Scales: small V=200, K=8; large V=2,000,
K=32 (32 successors per token, L = 20 at both).

## 2. Mathematical decomposition
    cand(t, b) = beam(b, s) ⊗ T(s, t)          (max,+): score + log P(t|s); s is a function of b, so
                                               the ⊕ over s never merges two candidates
    surv       = top_K over (t, b) of cand      by value, ties by key (t, b)
    beam'(b', t) = cand(t, b) for b' = rank(t, b) among the survivors
Fixed-count iteration over L steps; the ⊕ of the semiring is never exercised — pruning is done by
SELECT, not by ⊕ (the plan's class K).

## 3. Algebra program
One step of `iterate(L, step, beam_0)`, beam_0 = {(0, start): 0.0}:

    cand  = combine(beam(b,s), T(s,t), keep=(t,b))     (b,s)x(s,t)->(t,b)   score + log-prob
    surv  = select_k(cand, K)                          (t,b)->(t,b)         K best, ties by (t,b)
    rank  = sorted(surv, key=(-score, t, b))           CONTROL RESIDUE      dense ids, parent map
    beam' = scan(surv, reindex_by_rank)                (t,b)->(b',t)        b' = rank[(t,b)]
After L steps the sequences are read back through the parent maps (CONTROL RESIDUE).

## 4. Hotspot event and work efficiency
Event: one candidate expansion = one (beam entry, successor) pair scored. The combine produces
exactly one candidate per matching (beam, T) pair, so the ⊗ count equals the golden's expansions.

Measured (harness, 2026-09-25; counts are size-determined, identical for seeds 0–2):

| scale | native_ops (expansions) | ratio_mul | ratio_sr | ratio_cand | select calls / candidates / outputs | t_native / t_algebra (s) |
|---|---|---|---|---|---|---|
| small (seed 0) | 4,896 | 1.000 | 1.000 | 2.033 | 20 / 4,896 / 160 | 0.001 / 0.041 |
| large (seed 0) | 19,488 | 1.000 | 1.000 | 2.033 | 20 / 19,488 / 640 | 0.008 / 0.494 |

ratio_mul = ratio_sr = 1 exactly: one ⊗ per expansion and no ⊕ (keys (t,b) are unique).
ratio_cand ≈ 2: every candidate is examined once more by `select_k` (select candidates =
expansions), plus K re-index scan tuples per step (160 small / 640 large,
the remaining 0.03). Select cost: 20 calls per run, each over K·32 candidates
(32 at step 1), returning K.

## 5. scan() functions used
- `reindex_by_rank((t, b), v) -> [((rank[(t, b)], t), v)]`: one dict lookup per survivor; O(1).
  `rank` is computed in plain Python before the scan (section 6).

## 6. Control residue
- Rank of the K survivors by (score desc, token asc, parent beam asc): one sort of K items per
  step, K log K compares; 160 items sorted in total at small, 640 at large.
- Parent map stores: 160 small / 640 large (one per survivor per step).
- Back-pointer reads at the end: 160 small / 640 large (K sequences × L).
All three are O(K) per step against K·32 expansions per step: 3·K / (K·32) = 9.4% of the
hotspot in item counts at both K (derived); the sort's K·log2(K) compares are 9% of the
expansions at K=8 and 16% at K=32 (derived).

## 7. Where the algebra is insufficient or awkward
1. `select_k` returns a set of tuples, not an ordered list. Assigning dense beam ids therefore
   needs a rank computed outside the operators (a K log K sort per step). A SELECT that emits its
   output in rank order — which a systolic priority queue produces anyway — would make the
   re-index a pure scan with no residue.
2. Back-pointers live outside the algebra, as in hmm_viterbi: neither `select_k` nor the fused
   reduce exposes the key of the winning candidate. Here it matters less because the (t, b) key
   itself is the back-pointer; only the per-step maps must be kept.
3. The semiring ⊕ is never used: keys are unique by construction, and pruning is a SELECT. This is
   the plan's section 4.5 point in the other direction — a (max,+) fixpoint over tokens (Viterbi)
   would merge hypotheses by last token and compute a different quantity (the best path per
   token), not beam search.
4. `select_k` is one pass over K·32 candidates; the golden's heap is also one pass (nsmallest).
   The candidate count is the same order; the difference is log K compares per candidate in the
   heap versus a full ranking — invisible in candidate counts.

## 8. Shape seen by the hardware
T is a sparse V×V relation with 32 non-zeros per row (density 16% small, 1.6% large), static
across steps — the shared operand; beam is a K-entry vector; cand has K·32 entries per step;
select reduces K·32 → K; 20 steps. Batched decoding of Q sequences would add a Q dimension that
shares T (multi-query). No output growth: the working set is K·32 tuples throughout. The select
dominates the non-⊗ work (half of all candidates).
