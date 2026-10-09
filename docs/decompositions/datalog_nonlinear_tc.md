# datalog_nonlinear_tc — transitive closure with the non-linear rule, semi-naive

**Category**: logic · **Semiring(s)**: boolean · **Execution structure**: sparse-fixpoint
**Module**: `softwareworkload/datalog_nonlinear_tc/workload.py` · **Test**: `tests/test_workload_datalog_nonlinear_tc.py`

## 1. Problem and reference algorithm
    path(x, z) :- edge(x, z).      path(x, z) :- path(x, y), path(y, z).
Same instance generator as datalog_tc (random digraph with m edges plus a chain through the first
n // 8 vertices), at smaller sizes: small n = 100, m = 300; large n = 250, m = 750. `native()` is
the BFS closure from every source vertex, as in datalog_tc, and counts one relaxation per out-edge
scanned from a vertex already known reachable. The algebra evaluates the non-linear program;
`extra_info` also runs datalog_tc's linear program on the same instance for the comparison.

## 2. Mathematical decomposition
X_0 = Δ_0 = E. The rule is quadratic in path, so the semi-naive delta has two terms (Ullman):
    Δ_{i+1} = ( Δ_i ⋈ X_i  ∪  X_{i-1} ⋈ Δ_i ) − X_i,      X_i = X_{i-1} ∪ Δ_i,
where the first term covers Δ ⋈ Δ and Δ ⋈ X_old and the second covers X_old ⋈ Δ. Every pair of
facts not both present before round i is examined exactly once. Round r derives paths of length
up to 2^(r+1) - 1, so the number of rounds is O(log diameter) instead of O(diameter).

Why one delta is not enough. The linear rewriting joins Δ with the other body atom as it stood
before the round: Δ' = (Δ ⋈ X_old) − X. Counterexample, edges a→b, b→c: X_0 = Δ_0 = {ab, bc},
X_old = ∅, so Δ_0 ⋈ X_old = ∅, the fixpoint stops after one round and path(a, c) is never derived
(`test_tiny_counterexample_for_single_delta`; on the random instances this "strict" form returns
the edge set: 300 / 750 facts instead of 9,217 / 54,290, measured seed 0). The "relaxed" form
Δ ⋈ X_new (X_new includes Δ, so Δ ⋈ Δ is covered) is a different case: the plan expected it to be
wrong too, and that expectation is wrong for this program. It is complete for transitive
closure: every path x→…→w→z of shortest length L ≥ 2 has the prefix (x, w) of length L − 1, which
enters some Δ_r, and the edge (w, z) is in X_r ⊇ E, so (x, z) is derived in round r (induction on
L). Measured: the relaxed form agrees with the golden on every instance (seeds 0-2 small, seed 0
large) with 30% fewer candidates than the two-term form (600,644 vs 857,184 mul_ops at small seed
0) in the same 5 rounds. It is nevertheless not a semi-naive evaluation of the rule: it relies on
E ⊆ X, that is, it contains the right-linear program Δ ⋈ E and adds redundant Δ ⋈ (X ∖ E) work;
for a non-linear rule whose derivations do not all have a base-fact suffix it is incomplete. The
two-term form is kept as the primary program because it is the rewriting that is correct for any
non-linear rule.

## 3. Algebra program
    C1   = combine(rename(Δ, j->m), rename(X, i->m), keep=(i, j))        # Δ ⋈ X_new
    C2   = combine(rename(X_old, j->m), rename(Δ, i->m), keep=(i, j))    # X_old ⋈ Δ
    U, _ = update(C1, C2)                                                 # OR-union of the two
    New  = mask(U, X, negate=True)                                        # antijoin on (i, j)
    X, Δ = update(X, New)
    fixpoint until Δ is empty; X_old is the input X of the previous round, kept in a closure.

## 4. Hotspot event and work efficiency
Event: one pair of facts whose middle attribute matches = one AND: (path(s,x), edge(x,y)) in the
BFS golden, (path(x,y), path(y,z)) in the algebra.
Measured (harness, small, seed 0, 2026-09-25): native_ops 27,938; ratio_mul 30.68, ratio_sr
60.42, ratio_cand 32.68; 5 rounds, Δ sizes [861, 4640, 3401, 15, 0]; |TC| = 9,217 (92% of n²).
Seeds 1, 2: ratio_mul 31.33, 27.15. Large (n = 250, seed 0): native_ops 164,033; ratio_mul
71.49, ratio_sr 142.01, ratio_cand 73.56; 5 rounds, Δ sizes [2325, 19157, 31821, 237, 0]; |TC|
= 54,290. Timing: small 0.76 s algebra; large 11.1 s algebra, 19.3 s including the three extra
diagnostic runs of `extra_info`.

Linear program of datalog_tc on the same instance (measured, seed 0): small 10 rounds, 27,938
mul_ops (ratio 1.000 to BFS), 57,721 candidates; large 11 rounds, 164,033 mul_ops, 338,500
candidates. So the non-linear program halves the number of rounds (5 vs 10 / 11) and pays 30.7×
(small) / 71.5× (large) the join work.

Per-round combine candidates (C1 = Δ ⋈ X_new, C2 = X_old ⋈ Δ), measured seed 0:

| round | small: Δ in | C1 | C2 | large: Δ in | C1 | C2 |
|---|---|---|---|---|---|---|
| 0 | 300 (E) | 958 | 0 | 750 (E) | 2,409 | 0 |
| 1 | 861 | 10,508 | 2,695 | 2,325 | 30,756 | 7,479 |
| 2 | 4,640 | 277,458 | 56,388 | 19,157 | 1,749,423 | 244,244 |
| 3 | 3,401 | 310,280 | 196,112 | 31,821 | 6,789,077 | 2,806,550 |
| 4 | 15 | 1,440 | 1,345 | 237 | 46,692 | 50,169 |

Growth with n (same generator, m = 3n, seed 0, measured 2026-09-25):

| n | |TC| | BFS relaxations | non-linear mul_ops | ratio_mul | rounds (non-linear / linear) |
|---|---|---|---|---|---|
| 100 | 9,217 | 27,938 | 857,184 | 30.7 | 5 / 10 |
| 200 | 36,292 | 110,209 | 6,568,871 | 59.6 | 5 / 11 |
| 250 | 54,290 | 164,033 | 11,726,799 | 71.5 | 5 / 11 |
| 300 | 80,091 | 241,677 | 21,384,344 | 88.5 | 5 / 12 |

The ratio grows linearly in n (about 0.3 n). The closure is about 90% dense, so a Δ fact (x, y)
joins with about n out-facts of y: the non-linear program does Θ(|TC| · n) = Θ(n³) ANDs against
Θ(n · m) = Θ(n²) for BFS (m = 3n). ratio_sr ≈ 2 × ratio_mul because 95% of the candidates land
on an existing key and are OR-merged (small seed 0: 810,672 add_ops for 857,184 candidates and
46,512 outputs): every path is re-derived once per intermediate vertex.

## 5. scan() functions used
none.

## 6. Control residue
Loop termination and one closure variable holding the previous state X_old (no operations; X_old
= X − Δ as an antijoin would cost a full pass over X every round). Union of the two join outputs
is an `update` (its candidates are counted).

## 7. Where the algebra is insufficient or awkward
The algebra expresses the program fully; the finding is about the program, not the operators:
the clean non-linear form is asymptotically worse than the reference, ratio_mul 30.7 at n = 100
rising to 88.5 at n = 300 (measured), with ratio_cand of the same size. The plan's case "fewer
rounds, more work per round" is confirmed and quantified: 2× fewer rounds for Θ(n)× more
candidates on a dense closure. Awkward points: (i) the fixpoint driver passes only (X, Δ); the
two-term rewriting needs X_old, which had to be kept outside the operators; (ii) there is no
n-ary union, so the two joins are merged with an `update` pass (the union and the state
update together are 29,620 candidates at small seed 0, 3% of the total); (iii) the relaxed
single-delta form is 30% cheaper and complete here, but only because E ⊆ X, a fact the algebra
cannot check or exploit by itself.

## 8. Shape seen by the hardware
Sparse Boolean relations that become dense: |TC| ≈ 0.9 n². Rounds 2 and 3 carry about 98% of
the candidates (98.0% small, 98.8% large, derived); the probe side of C1 is X (up to 54,290
tuples at large) and of C2 is X_old. The antijoin probe table is X. Δ peaks at 31,821 tuples (large) and collapses in one round. Five
rounds at every size in the sweep, against 10-12 for the linear program. No multi-query sharing.
