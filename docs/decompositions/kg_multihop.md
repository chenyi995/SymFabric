# kg_multihop — k-hop max-product reasoning from Q sources on a shared knowledge graph

**Category**: graph · **Semiring(s)**: viterbi (log-domain (max,+)) · **Execution structure**: frontier-fixpoint, multi-query
**Module**: `softwareworkload/kg_multihop/workload.py` · **Test**: `tests/test_workload_kg_multihop.py`

## 1. Problem and reference algorithm
Knowledge graph with n entities and m distinct edges (h, r, t), 20 relation types, confidence
w ∈ (0, 1] per edge (n = 5000, m = 30000, Q = 64 small; n = 50000, m = 300000, Q = 256 large).
For every source q of the Q sources and every entity t: the best max-product score of a path of
at most K = 4 hops from q to t, in the log domain, plus the 10 best answers t ≠ q per source.
`native()` runs, per source, K rounds of layered relaxation that relax only the entries improved
in the previous round, with plain dictionaries; it counts one relaxation per (improved entry,
out-edge) pair. Both sides treat a candidate as an improvement only when it exceeds the stored
score by more than 1e-9 relative (the relation layer's numeric equality): without a tolerance a
1-ulp re-ordering of the same path sum would count as an improvement and make the delta sets
diverge.

## 2. Mathematical decomposition
x_0(q, q) = 0 (log 1).  x_{i+1}(q, t) = x_i(q, t) ⊕ ⊕_{h, r} x_i(q, h) ⊗ E(h, r, t)
with (⊕, ⊗) = (max, +) on log w, i.e. x_{i+1}(q, t) = max(x_i(q, t), max_{h,r} x_i(q, h) + log w(h, r, t)).
Semi-naive form: only entries that improved in round i (Δ_i) are extended in round i + 1; exact
because (max,+) is idempotent and selective. Hop limit: K rounds. Answers: top-10 of
{x_K(q, t) : t ≠ q} per q.

## 3. Algebra program
    E          : (h, r, t) -> log w                              # shared by all Q sources
    X_0 = Δ_0  : (q, h) -> 0 for every source q
    round i:
      cand     = combine(Δ(q, h), E(h, r, t), keep=(q, t))    # max over (h, r) of x + log w, reduce fused
      X, Δ     = update(X, rename(cand, t -> h))              # max-merge; Δ = improved entries
    fixpoint with max_rounds = K
    answers    = mask(X, X_0, negate=True)                    # drop the seed entries (q, q)
    per source q: top_q = select_k(answers_q, 10)             # answers_q = the (q, *) slice of answers
The slice by q is a hash partition of `answers` done in control (one pass, |answers| tuples;
CONTROL RESIDUE, counted in `extra`), because the algebra has no grouped select.

## 4. Hotspot event and work efficiency
Event: one relaxation = one (improved entry (q, h), out-edge (h, r, t)) pair, i.e. one (max,+)
mul x(q, h) + log w. Measured (harness, 2026-09-25):

| scale/seed | Q | relaxations (golden = combine candidates) | rounds | Δ sizes per round | facts in X | facts per source | hop add_ops | top-k selects | select cand. (= answers) | ratio_mul | ratio_sr | ratio_cand | t_native s | t_algebra s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small/0 | 64 | 89975 | 4 | 354, 2153, 12434, 62602 | 76214 | 1191 | 13825 | 64 | 76150 | 1.000 | 1.154 | 3.583 | 0.07 | 0.27 |
| small/1 | 64 | 100585 | 4 | 387, 2385, 13876, 69067 | 84150 | 1315 | 16499 | 64 | 84086 | 1.000 | 1.164 | 3.554 | 0.08 | 0.35 |
| small/2 | 64 | 98204 | 4 | 387, 2344, 13502, 67827 | 82602 | 1291 | 15666 | 64 | 82538 | 1.000 | 1.160 | 3.567 | 0.09 | 0.32 |
| large/0 | 256 | 394115 | 4 | 1527, 9144, 54744, 322239 | 387177 | 1512 | 7194 | 256 | 386921 | 1.000 | 1.018 | 3.951 | 0.60 | 2.28 |

ratio_mul is exactly 1.000: the semi-naive combine relaxes the same (entry, edge) pairs as the
per-source golden, because both extend exactly the entries improved in the previous round and
apply the same improvement rule. ratio_sr > 1 counts the max-merges of candidates that land on the
same (q, t) inside one round (fused reduce) and the max-merges of update: ratio_sr is
1.154-1.164 at small scale and 1.018 at large, where the entities reached per source (1512) are a far
smaller fraction of n = 50000 than at small scale (1191 of 5000), so fewer paths collide on the
same (q, t). ratio_cand ≈ 4.0 adds the update probe of every distinct candidate (q, t), the
antijoin over X and the per-source selects (|answers| candidates in total).

Per-round shape of the shared combine, large/0 (measured): n_in = (|Δ|, |E|), candidates, n_out
(distinct (q, t) reached), add_ops (in-round merges):

| round | n_in | candidates | n_out | add_ops |
|---|---|---|---|---|
| 0 | 256, 300000 | 1527 | 1527 | 0 |
| 1 | 1527, 300000 | 9150 | 9145 | 5 |
| 2 | 9144, 300000 | 54904 | 54776 | 128 |
| 3 | 54744, 300000 | 328534 | 323584 | 4950 |

## 5. scan() functions used
none.

## 6. Control residue
The hop limit (fixpoint max_rounds = K, a round counter). The hash partition of `answers` by
source before the per-source selects: one pass over 386921 tuples at large (measured, `extra`),
the same order as one update round and far below the 394115 relaxations. Sorting the ≤ 10
selected answers per source for the output (Q × 10 log 10).

## 7. Where the algebra is insufficient or awkward
1. No grouped select. "Top-10 per source" is Q separate select_k calls over the Q slices of one
   relation; slicing by key prefix is not an operator, so it is done as a hash partition in
   control. A SELECT unit that keeps one queue per key prefix (or a partition operator) would
   express it in one pass over 386921 tuples — which is what the Q selects cost together anyway,
   so the residue is a matter of expression, not of work.
2. The improvement test inside update uses a numeric tolerance; it is part of the semantics of
   "changed" for a floating-point (max,+) fixpoint and the golden had to adopt the same rule to
   reproduce the delta sets. This is an interface subtlety, not missing capability.
3. Nothing else: the multi-query k-hop reasoning is one combine per round with the sources as an
   extra key attribute, and the semi-naive delta makes it work-optimal (ratio_mul 1.000).

## 8. Shape seen by the hardware
Sparse E with m = 300000 tuples at large (keys h, r, t: 16 + 5 + 16 bits, 32-bit log weight),
read once per round for all Q sources together: the combine's n_in is (|Δ|, |E|) with |Δ|
growing from Q = 256 to 54744 over the four rounds (table above), so the build side is Δ
and E is the streamed operand in every round. Multi-query sharing: one streaming pass over E per
round serves all Q sources — 1200000 edge reads in total (derived: K × m); executing the Q
sources one at a time would read E once per source and round, 307200000 edge reads (derived:
Q × K × m), Q = 256 × more. A memory-side multi-query scan that carries the Q-keyed Δ in
the die and streams E from HBM therefore amortises the edge reads over Q; the arithmetic
(394115 relaxations) is unchanged. Output growth: X reaches 387177 facts, about 1512 per
source, after 4 hops on a graph with average out-degree 6; the last round produces
328534 of the 394115 candidates. The dominant operator is the round-3 combine.
