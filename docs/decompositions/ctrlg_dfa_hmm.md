# ctrlg_dfa_hmm — Ctrl-G constrained-decoding step: HMM × DFA forward and next-token scores

**Category**: probabilistic · **Semiring(s)**: prob (+,×), logprob (LSE,+), boolean · **Execution structure**: dense-iterate (forward) with a sparse-fixpoint prelude (DFA liveness)
**Module**: `softwareworkload/ctrlg_dfa_hmm/workload.py` · **Test**: `tests/test_workload_ctrlg_dfa_hmm.py`

## 1. Problem and reference algorithm
An HMM (N states, vocabulary V; Dirichlet(1) transition A, emission B, initial π) and a partial
DFA over tokens with S states: each non-trap state has 3 explicitly labelled transitions to other
states, every state has a self-loop on each of the V/8 default tokens, every other token has no
transition (the constraint is violated), and S/4 non-accepting trap states have only the
self-loops (dead). 3 (small) / 8 (large) accepting states; a prefix of T = 32 / 48 tokens sampled
from the HMM under the constraint. Small: N = 16, V = 64, S = 12 (123 transitions); large:
N = 48, V = 256, S = 32 (1,096 transitions).
Computed: live(s) (an accepting state is reachable), fwd_T(h, s) over the product space after the
prefix, and for every token score(tok) = Σ_{h,s,h'} fwd_T(h, s) A(h, h') B(h', tok) [δ(s, tok) ∈ live].
`native()`: liveness by BFS on the reversed DFA; forward with explicit loops — the DFA is
deterministic, so the golden tracks the single DFA state of the prefix and runs the N-state HMM
forward; pred = fwd·A (N² MACs) and one N-MAC dot product per live token; the same in the log
domain with log-sum-exp. It counts MACs.

## 2. Mathematical decomposition
live  = lfp( X = F ∪ { s : ∃ tok, s' ∈ X . δ(s, tok, s') } )                       BOOLEAN, semi-naive
fwd_{t+1}(h', s') = ⊕_{h,s} fwd_t(h, s) ⊗ A(h, h') ⊗ B(h', tok_t) ⊗ δ(s, tok_t, s')   three joins per step:
                    dense on h (N × N), elementwise on h' (emission column), sparse on s (DFA move)
score(tok) = ⊕_{h',s} [ ⊕_h fwd_T(h, s) ⊗ A(h, h') ] ⊗ LiveST(s, tok) ⊗ B(h', tok)

## 3. Algebra program
    liveness (BOOLEAN):
        Rev(s2, s)     = reduce(Delta(s, tok, s2), keep=(s2, s))                  # OR over tok
        step:  New = mask(rename(combine(D, Rev, keep=(s,)), s -> s2), X, negate=True); X, D = update(X, New)
        Live           = fixpoint(step, Accepting)
        LiveST(s, tok) = reduce(mask(Delta, Live), keep=(s, tok))                # [exists live successor]
    forward (PROB, then LOGPROB), token 0: f = combine(combine(init(h2, s), B_tok0(h2)), Delta_tok0(s, s2), keep=(h2, s2))
        step t:  p = combine(f(h, s), A(h, h2), keep=(h2, s))                     # N·N·|live s| MACs
                 p = combine(p, B_tok(h2))                                        # emission column, N ⊗
                 p = combine(p, Delta_tok(s, s2), keep=(h2, s2))                  # DFA move, unit weights, N ⊗
                 f = rename(p, h2 -> h, s2 -> s);   iterate(T − 1)
    scores:  pred = combine(f_T, A, keep=(h2, s))                                 # N·N MACs
             LT   = scan(LiveST, (s, tok) -> ((s, tok), one))                     # BOOLEAN -> unit weights
             pl   = combine(pred, LT)                                             # (h2, s, tok), live tokens only
             sc   = combine(pl, B(h2, tok), keep=(tok,))                          # N·|live tokens| MACs
B_tok and Delta_tok are the emission matrix and the transition relation indexed by token (built
once, a layout step as in hmm_viterbi).

## 4. Hotspot event and work efficiency
Native MACs per semiring: (T − 1)·N² forward + N² prediction + N·|live tokens| scores, two
semirings: 16,736 (small seed 0), 224,352 (large seed 0). The liveness BFS relaxations (28 small,
78 large) are not MACs and are excluded from the denominator; the algebra's liveness ANDs (28 and
78, equal to the BFS relaxations) are included in the numerator.
Measured (harness, small, seed 0, 2026-09-25): ratio_mul 1.145, ratio_sr 2.088, ratio_cand 1.178;
seeds 1 and 2: 1.145 / 2.088 / 1.178 and 1.143 / 2.086 / 1.171. Large (seed 0): ratio_mul 1.056,
ratio_sr 2.039, ratio_cand 1.077. Per numeric semiring (prob = logprob): ratio_mul 1.143 (small),
1.055 (large); the remainder is liveness.
Why above 1: each step costs N² transition MACs plus N emission ⊗ plus N unit-weight ⊗ for the
DFA move (a gather): 1 + 2/N = 1.125 at N = 16 and 1.042 at N = 48. The score step adds a
unit-weight gather of N·|live tokens| tuples (176 small, 1,584 large) before the same number of
MACs. ratio_sr ≈ 2: one ⊕ per MAC. ratio_cand: the liveness mask / reduce / update candidates
(394 small, 3,169 large — mostly the two |Delta|-sized passes) over 3 / 4 semi-naive rounds.
Dense-first scoring alternative (pred ⊗ B for all V tokens, then mask by LiveST, then reduce):
ratio_mul 1.225 (small), 1.137 (large) against 1.145 / 1.056 sparse-first — N·V = 1,024 instead
of N·|live| = 176 MACs at small scale. Sparse-first is used; it is also what the golden does.
Underflow: minimum non-zero PROB forward entry 1.2e-62 (small seed 0), 4.0e-120 (large seed 0),
far from the float64 minimum at these T. Derived from the measured decay (120 decades in 48
steps at V = 256, i.e. 2.5 decades per token), the PROB run would cross 1e-308 near T = 125, so
the LOGPROB run is the one a decoder must use for realistic prefix lengths.
LOGPROB agreed with log(PROB) on the scores to 5.7e-14 (small) and 1.1e-13 (large) absolute.
Timing (harness, seed 0): algebra 0.030 s small, 0.340 s large; native 0.004 s / 0.015 s.

## 5. scan() functions used
`bool_to_prob` / `bool_to_logprob`: (s, tok) -> ((s, tok), one) — converts the BOOLEAN
live-transition selector into a unit-weight relation of the numeric semiring so that it can be
joined; O(1) per tuple.

## 6. Control residue
Indexing the DFA transitions and the emission matrix by token (a dictionary of relations built
once: |Delta| plus N × distinct-prefix-tokens tuples) — the layout step hmm_viterbi also performs
for its emission columns; the T-step loop; reading back the N × S and V-sized results. Instance
generation samples the prefix under the constraint with a plain BFS, outside the measured work.

## 7. Where the algebra is insufficient or awkward
1. Semiring mismatch between the BOOLEAN liveness result and the numeric forward: `combine`
   requires equal semirings, so the boolean selector has to be converted with a scan (unit
   weights) before it can be joined, and the join then costs one ⊗ per selected tuple (the
   N·|live tokens| gather). `mask` applies a boolean support to a numeric relation without ⊗, but
   only as a post-filter, which forces the dense-first order measured above as more expensive
   because mask cannot re-key. A semijoin that re-keys (or a combine that recognises a
   unit-weight side) would remove the gather cost.
2. Degeneracy: for a deterministic DFA and a fixed prefix the product-state forward has exactly
   one non-empty DFA column (support N × 1 measured at both scales). Zero suppression makes the
   algebra as cheap as the state-tracking golden, but the product space is not exercised in
   width; that happens in Ctrl-G's backward pass (β over all (h, s) pairs), which this workload
   does not include.
3. Trace accounting caveat: combine records matched pairs only. When the small live-transition
   table is the build side and the dense B (N × V) is probed, the N·V hash probes (1,024 small,
   12,288 large) do not appear in `candidates`.

## 8. Shape seen by the hardware
A (N × N dense, 256 / 2,304 tuples) is reused at every step and would be reused by every DFA
column: the weight-stationary operand; fwd (one live column of N values) streams through it. The
emission column B(·, tok_t) and the DFA move δ(s_T, tok_t, ·) are per-step gathers of N and at
most S tuples. Delta is sparse (123 / 1,096 tuples of S·V = 768 / 8,192, 16% / 13%); in the score
step it selects 11 / 33 live tokens of 64 / 256, so the dense B (1,024 / 12,288 tuples) should be
indexed by the sparse selector rather than streamed — the sparse operand is the stationary or
indexed side there, the reverse of the transition step where the dense A is stationary. Liveness:
3 / 4 semi-naive rounds with deltas [5, 1, 0] / [8, 5, 3, 0] over a reversed graph of 35 / 102
edges — negligible work. The two numeric semirings run the identical dataflow.
