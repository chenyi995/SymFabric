# hmm_viterbi — HMM forward filtering and Viterbi decoding

**Category**: probabilistic · **Semiring(s)**: logprob (LSE,+), viterbi_bp (custom (max,+) with back-pointer) · **Execution structure**: dense-iterate
**Module**: `softwareworkload/hmm_viterbi/workload.py` · **Test**: `tests/test_workload_hmm_viterbi.py`

## 1. Problem and reference algorithm
Forward: a_t(j) = b_j(o_t) Σ_i a_{t-1}(i) A_ij, in the log domain. Viterbi: same with max and
back-pointers, path recovered by back-tracking. `native()` is the textbook numpy loop and counts
one transition multiply-accumulate per (t, i, j).

## 2. Mathematical decomposition
a_t(j) = ⊕_i [ a_{t-1}(i) ⊗ A(i,j) ] ⊗ B(j, o_t)   with (⊕,⊗) = (LSE,+) or (max,+).
Fixed-count iteration over t (no fixpoint).

## 3. Algebra program
    s   = combine(a, A, keep=(j,))      # (i) x (i,j) -> (j): ⊕_i a(i) ⊗ A(i,j)   [dense, N x N]
    a'  = combine(s, B_t)               # (j) x (j) -> (j): elementwise ⊗ with the emission column
    rename j -> i; iterate(T-1).
Viterbi uses the value type (score, bp) with add = max-by-score and mul = (s1+s2, bp of the
operand that carries one); the transition relation stores (log A_ij, i), so the reduce over i keeps
the arg-max predecessor. Back-tracking: T filter(layers[t], key == cur) calls, O(N) each.

## 4. Hotspot event and work efficiency
Event: transition MAC, T·N·N per pass, two passes (forward + Viterbi).
Measured (harness, small N=16, T=40, seed 0, 2026-09-25): ratio_mul 1.062, ratio_sr 2.000,
ratio_cand 1.094. Large (N=48, T=120): ratio_mul 1.031.
ratio_mul > 1: the emission join adds N ⊗ per step (1/N of the transition work). ratio_sr = 2
because a MAC is one ⊗ and one ⊕ and the native count is MACs.

## 5. scan() functions used
none (the emission multiply is a join on j).

## 6. Control residue
Argmax over the final layer (N compares) and the back-tracking loop control; O(T + N).

## 7. Where the algebra is insufficient or awkward
The back-pointer had to be smuggled into the value type. A hardware reduce for selective
semirings should expose "arg" (the key of the winning candidate) directly, which is what a
comparator tree does anyway. This is an interface convenience, not a missing capability.

## 8. Shape seen by the hardware
Dense N×N transition operand reused every step (weight-stationary candidate); a is a length-N
vector; batch of sequences would add a Q dimension sharing A. Two semirings on identical
dataflow: the only difference is the PE's (⊕,⊗) selection.
