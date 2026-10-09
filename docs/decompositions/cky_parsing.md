# cky_parsing — PCFG inside algorithm (CKY) in PROB and VITERBI

**Category**: parsing · **Semiring(s)**: prob (+, ×), viterbi (max, +) on log weights · **Execution structure**: dag
**Module**: `softwareworkload/cky_parsing/workload.py` · **Test**: `tests/test_workload_cky_parsing.py`

## 1. Problem and reference algorithm
A random PCFG in Chomsky normal form (N non-terminals, 3-5 binary rules A → B C per non-terminal with
distinct right-hand sides, 3 lexical rules A → w per non-terminal over a vocabulary of N words,
weights normalized per left-hand side) and a sentence of length n generated from it (a uniformly
random binary bracketing labelled top-down from the start symbol by sampling the grammar's rules).
Two passes over the same chart recursion: PROB gives the inside probability of every span and of the
root; VITERBI, on log weights, gives the best-parse score (Goodman's semiring parsing).
`native()` is the textbook CKY triple loop (span length, start, split point) with the rules indexed
by their left child B: for every B in the left cell and every rule A → B C it looks C up in the right
cell. It counts one (i, k, j, rule) combination for every rule application whose two children are
both present (one ternary product), separately in each pass; the look-ups with a missing right child
are reported in `extra` (29,160 at small scale) but not counted.

## 2. Mathematical decomposition
span(i, i+1, A) = lex(A, w_i)
span(i, j, A)   = ⊕_{k, B, C} rule(A, B, C) ⊗ span(i, k, B) ⊗ span(k, j, C),   i < k < j
with (⊕, ⊗) = (+, ×) or (max, +). Fixed-count iteration over the span length ℓ = 2..n; every span of
length ℓ depends only on shorter spans, so each round is one DAG layer. The three-way product is two
binary joins; the two orders are
    pair-first: P(i, j, B, C) = ⊕_k span(i, k, B) ⊗ span(k, j, C);   span(i, j, A) = ⊕_{B,C} P ⊗ rule(A, B, C)
    rule-first: T(i, k, A, C) = ⊕_B span(i, k, B) ⊗ rule(A, B, C);   span(i, j, A) = ⊕_{k,C} T ⊗ span(k, j, C)
T is built once per completed cell (it does not depend on j), which is the binarized ("dotted rule")
form of the grammar.

## 3. Algebra program
Default (the prescribed pair-first order; the length constraint by re-keying), one round per ℓ:

    words(i, j, w) ⋈ lex(w, A)                       spans = combine(words, lex, keep=(i, j, A))          # ℓ = 1
    L(i, k, B)      = rename(spans, j->k, A->B)      # metadata only
    Rr(k, i, C)     = scan(spans, shift_end_ℓ)       # (i_r, j_r, C) -> (k=i_r, i=j_r-ℓ, C), only if j_r ≥ ℓ
    P(i, B, C)      = combine(L, Rr, keep=(i, B, C)) # natural join on (i, k): exactly the adjacent pairs of
                                                     # total length ℓ; fused ⊕ over k
    cell(i, A)      = combine(P, rules(A, B, C), keep=(i, A))        # join on (B, C); fused ⊕ over (B, C)
    new(i, j, A)    = scan(cell, add_end_ℓ)          # (i, A) -> (i, i+ℓ, A)
    spans           = update(spans, new)             # union on disjoint keys

Rule-first (`ORDER = "rule_first"`): T(i, k, A, C) = combine(rename(new, j->k, A->B), rules,
keep=(i, k, A, C)) after every round (accumulated with update on disjoint keys), and the recursion is
the single combine(T, Rr, keep=(i, A)) on (i, k, C).
Filter variant (`LENGTH_MODE = "filter"`, measured only): L ⋈ rename(spans, i->k, A->C) on k over
every known span, then filter(j − i = ℓ), then the rule join.

**Decision on the length constraint** (claude): the join can only equate attributes, so j − i = ℓ
is carried by re-keying the right operand with the round's constant ℓ (an O(1) scan); the natural
join then matches i_left = j_right − ℓ together with k and produces no candidate that is discarded.
Per-length relations were the alternative; they need a union over the ℓ − 1 split points, and the
only union operator is `update`, whose change test uses `Semiring.equal` with an effectively
absolute 1e-9 tolerance for values below 1 — inside probabilities are 7.2e-25 (small) and
1.2e-62 (large) at the root, so every merge would be judged "unchanged" and dropped (verified on
a two-tuple example: update({0: 3e-20}, {0: 2e-20}) returned 3e-20 and an empty change set). The
fused ⊕ inside combine has no tolerance test, so the sum over k is done there.

## 4. Hotspot event and work efficiency
Event: one (i, k, j, rule) combination with both children present, per pass. Algebra muls: pair-first
does one mul per (B, C) child pair (all N² combinations of two dense cells, whether or not a rule has
that right-hand side) plus one per (P tuple, rule) match; rule-first does one per (T entry, right
child) match plus |cell| × rules-per-B per completed cell.

Measured (harness, small N=12 n=20, seed 0, 2026-09-25), grammar with 46 binary rules over
38 distinct right-hand sides, mean cell fill 0.79 of N:

| program | ratio_mul | ratio_sr | ratio_cand | algebra time |
|---|---|---|---|---|
| pair-first, re-key (default) | 3.305 | 5.905 | 4.201 | 0.24 s |
| rule-first, re-key | 1.189 | 2.074 | 2.312 | 0.17 s |
| pair-first, filter | 35.718 | 62.142 | 45.010 | 2.33 s |
| rule-first, filter | 10.921 | 20.617 | 12.199 | 0.69 s |

Seeds 0-2 (small): pair-first ratio_mul 3.305-3.383, ratio_cand 4.201-4.410; rule-first
ratio_mul 1.177-1.212, ratio_cand 2.312-2.499. native 0.02 s.
Measured (harness, large N=24 n=45, seed 0, 2026-09-25), 99 rules over 84 right-hand sides,
cell fill 0.76: pair-first ratio_mul 5.980, ratio_sr 11.332, ratio_cand 6.824
(algebra 7.9 s, native 0.40 s); rule-first ratio_mul 1.065, ratio_sr 1.983,
ratio_cand 2.025 (algebra 3.5 s).

Explanation. Pair-first: with nearly full cells the pair join costs N² per (i, k, j) while the
reference applies only the ~4N rules, so ratio_mul ≈ N / (rules per non-terminal) ≈ 3 (N=12) and 6
(N=24), which is what was measured; the rule join adds only n²/2 × |rules| (it runs on P, already
reduced over k). Rule-first: the recursion's mul count is at most the reference's (rules sharing
(A, C) are merged into one T entry), and the T construction adds O(n² · rules) — 1 + 3/n in the dense
limit, measured 1.189 (n=20) and 1.065 (n=45). ratio_sr ≈ 2 × ratio_mul − (outputs):
every candidate of a fused combine is one ⊗ and, except the first per key, one ⊕. ratio_cand adds the
two scans per round (20% of all candidates for the default program at small scale) and
the update writes. The filter variant re-joins every adjacent pair of already known spans every round
and discards all but one length: measured inflation 10.8 × in ratio_mul at n=20; the
candidate count grows as Σ_ℓ n ℓ² N² ≈ n⁴ N² / 3 against n³ N² / 6 for the exact program, i.e. ~2n ×
(derived; not run at n=45).

## 5. scan() functions used
- `shift_end_ℓ`: (i_r, j_r, C), v → ((k = i_r, i = j_r − ℓ, C), v) if j_r ≥ ℓ, else nothing. One
  subtraction and one comparison per tuple.
- `add_end_ℓ`: (i, A), v → ((i, i + ℓ, A), v). One addition per tuple.
- (filter variant only) predicate `len_eq_ℓ`: j − i = ℓ, one subtraction per tuple.

## 6. Control residue
The ℓ loop (n − 1 rounds) and the round constant captured by the two scan closures; reading the root
value `spans.get((0, n, S))`. Nothing else is computed outside the operators; the two semiring passes
share the program text and differ only in the relation semiring and in `rule_val` (p or log p).

## 7. Where the algebra is insufficient or awkward
1. **Join order is the whole story.** The prescribed pair-first order costs N / (rules per
   non-terminal) × the reference (measured 3.305 × and 5.980 ×); the rule-first order
   (T = binarized grammar, built once per cell) is within 19% / 6.5% of the
   reference. Both are expressible; the algebra does not choose. A planner (or the workload author)
   must pick the order in which the sparse operand — here the rule table, |rules| / N² =
   0.32 of the (B, C) space — is joined before the dense × dense product is formed.
2. **Arithmetic constraints between attributes** (j − i = ℓ) have no operator. The re-keying scan is
   an O(1) map, but it re-keys every known span every round (20% of candidates at small
   scale, 12% at large), and it hides part of the program's structure in the scan
   function. A hardware join with an affine key offset (probe key = build key + constant) would
   remove both scans.
3. **`update` cannot accumulate a non-idempotent numeric semiring at small magnitudes** (§3). This
   is the reason the sum over split points is fused into combine rather than being a union of
   per-length relations; the exemplar hmm_viterbi does not hit it because LOGPROB values are
   O(−100), where the 1e-9 relative tolerance is meaningful. Either `Semiring.equal` needs an
   absolute-vs-relative rule that follows the semiring, or `update` needs an exact-merge mode for
   non-idempotent semirings.
4. **A ternary product is two muls.** One reference event is rule ⊗ left ⊗ right; the algebra's
   binary combine spends one mul in each join. For rule-first the first mul is amortized over j (T is
   per cell), so it costs 3/n, not 1; for pair-first both muls are paid per (i, k, j).

## 8. Shape seen by the hardware
All relations are small and dense-ish: at large scale 18,843 span tuples in each pass (chart of
1,035 cells × N=24, fill 0.76), 99 rules, and per round a P relation of at most
(n − ℓ + 1) × N² tuples. n − 1 = 44 rounds of shrinking size (DAG layers); no fixpoint, no delta. The
rule table is the stationary operand (weight-stationary candidate); the T table of the rule-first
order is the stationary operand per cell and is the analogue of hmm_viterbi's transition matrix. The
two passes are the same dataflow with a different PE (⊕, ⊗) selection. Combine dominates
(79% of candidates for the default program at small scale; scans 20%,
update 2%).
