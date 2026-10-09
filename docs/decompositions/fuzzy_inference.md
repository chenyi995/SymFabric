# fuzzy_inference — max-min fuzzy relational inference (fuzzy transitive closure + queries)

**Category**: logic · **Semiring(s)**: fuzzy (max, min) · **Execution structure**: sparse-fixpoint
**Module**: `softwareworkload/fuzzy_inference/workload.py` · **Test**: `tests/test_workload_fuzzy_inference.py`

## 1. Problem and reference algorithm
A fuzzy relation R : U × U → [0, 1] over n elements (about 6 non-zero memberships per row) is
composed with itself in the (max, min) semiring, (R ∘ S)(x, z) = max_y min(R(x, y), S(y, z)), until the
fuzzy transitive closure R* = max_{k ≥ 1} R^k stops changing. Then Q fuzzy queries q (sparse fuzzy
sets, 4 non-zero memberships each) are answered by one more composition, out_q(z) = max_x min(q(x),
R*(x, z)). This is the inference step of max-min fuzzy rule bases and of the Overmind-style
neuro-symbolic accelerators; with Boolean memberships it is Datalog transitive closure.

`native()` computes R* by a widest-path (max-min) Dijkstra from every element: the element y with
the largest tentative value is settled, and every out-edge (y, z) is relaxed once with
min(value(y), R(y, z)). The queries are two nested plain loops over the non-zeros of q and of the
R* row. It counts one (x, y, z) min-evaluation per relaxation and one per (q(x), R*(x, z)) pair. The
source's own out-edges are base facts (X_0 = R on the algebra side) and are not counted, exactly as
in datalog_tc.

## 2. Mathematical decomposition
X_0 = Δ_0 = R. Each round: C(x, z) = max_y min(Δ(x, y), R(y, z)); X'(x, z) = max(X(x, z), C(x, z));
Δ' = {(x, z) : X'(x, z) > X(x, z)} (with the new value). Stop when Δ' = ∅.
out(qid, z) = max_x min(q(qid, x), X*(x, z)).
The single-delta rule is exact: if (x, y) did not change in the previous round, every candidate
min(X(x, y), R(y, z)) it can produce was already merged into X. Termination follows from
idempotence (max) and the finite set of membership values; the fixpoint is R^(k) for the first k with
R^(k+1) = R^(k), which is the transitive closure of a max-min relation.

## 3. Algebra program
It is the datalog_tc chain with the semiring swapped. Side by side, what changed:

    datalog_tc (BOOLEAN)                                      fuzzy_inference (FUZZY)
    E    = Relation.from_set((i, j), edges)                   R    = Relation((x, y), FUZZY, {(x, y): μ})
    C    = combine(rename(Δ, j->m), rename(E, i->m),          C    = combine(rename(Δ, y->m), rename(R, x->m),
                   keep=(i, j))         # OR-merged                          keep=(x, y))         # max-merged, mul = min
    New  = mask(C, X, negate=True)      # antijoin            (removed: a pair already in X can still improve)
    X, Δ = update(X, New)               # Δ = new facts       X, Δ = update(X, C)                 # Δ = pairs whose value rose
    fixpoint(step, E)                                         fixpoint(step, R)
                                                              out  = combine(q(qid, x), rename(X, y->z), keep=(qid, z))

Schemas: R, X, Δ, C : (x, y); the join is on m; q : (qid, x); out : (qid, z). Three changes: the
value semiring, the removal of the antijoin (only valid when a stored value cannot change, i.e. for
OR), and the query phase, which is one more composition sharing X across the Q queries.

## 4. Hotspot event and work efficiency
Event: one (x, y, z) min-evaluation. Reference: one per out-edge relaxed from a settled element in
the per-source Dijkstra, i.e. Σ_{(x, y) ∈ R*} outdeg(y), plus the query pairs. Algebra: one per
matching (Δ(x, y), R(y, z)) pair in the combine, i.e. Σ_rounds Σ_{(x, y) ∈ Δ_k} outdeg(y), plus the
same query pairs. The query phase has ratio 1 by construction (both sides touch exactly the
(q(x) > 0, R*(x, z) > 0) pairs).

Measured (harness, small n=300 d=6 Q=32, seed 0, 2026-09-25): ratio_mul 4.223, ratio_sr 8.277,
ratio_cand 5.934; 41 rounds; |R*| = 89,700 of n² = 90,000 (density
0.997); Σ_k |Δ_k| = 399,381, re-derivation factor Σ_k |Δ_k| / |R*| = 4.45;
native 0.18 s, algebra 3.36 s. Seeds 0-2 (small): ratio_mul 4.028-4.223, ratio_cand
5.576-5.984, rounds 25-43.
Measured (harness, large n=800 d=6 Q=128, seed 0, 2026-09-25): ratio_mul 4.841, ratio_sr 9.509,
ratio_cand 6.800; 48 rounds; |R*| = 638,400 (density 0.998);
re-derivation factor 5.25; native 1.42 s, algebra 31.7 s.
Measured (harness, scale "spec_large" n=2000 d=6 Q=128 — the size the task specified, seed 0, 2026-09-25):
ratio_mul 5.650, ratio_sr 11.131, ratio_cand 7.933; 72 rounds; |R*| = 3,998,000;
re-derivation factor 5.85; native 9.7 s, algebra 252.3 s. This run exceeds the 40 s
budget by a wide margin, which is why "large" is n=800 (decision marker in `SCALES`).

ratio_mul > 1 is the re-derivation factor: the fixpoint is round-synchronous (Bellman-Ford-like), so
a pair (x, z) re-enters Δ every time a longer path with a larger bottleneck value is found, and is
re-joined with all of row z each time; Dijkstra settles each (x, z) once. The factor grows with n
(4.45 → 5.25 → 5.85) because the widest paths in the top membership levels
are long: the α-cut of a random relation with 6 non-zeros per row at level α keeps 6(1 − α) edges per
row, which drops to one edge per row at α = 5/6 (derived), so the widest paths of most pairs run
through a subgraph near the percolation threshold, and the round count (41 / 48 /
72) is the length of the longest such path. ratio_sr ≈ 2 × ratio_mul: a candidate that
lands on an existing (x, z) key of the fused reduce costs one min and one max (59%
of the combine candidates at small scale, 1,438,834 of 2,434,558) and every C tuple is
max-merged into X by update (898,256 adds). ratio_cand adds the update inputs (|C| per round,
29% of all candidates).

## 5. scan() functions used
none.

## 6. Control residue
Loop termination (|Δ| = 0), one scalar per round; construction of the R and q relations from the
instance. Nothing is computed outside the operators.

## 7. Where the algebra is insufficient or awkward
1. **Schedule, not expressiveness.** The operators express the closure exactly, but the only
   iteration structure that fits all-pairs closure is the round-synchronous fixpoint, and in a
   non-Boolean idempotent semiring it re-derives values (measured 4.223 ×, 4.841 ×, 5.650 ×
   at the three sizes; the trend is upward with n). The reference beats it by settling elements in
   value order. Expressing that order in the algebra needs a priority-ordered frontier: `select_k`
   provides selection, but a per-source best-first loop is a select-loop of O(n) tiny rounds per
   source, O(n²) operator invocations in total, which is not a hardware-shaped program. A rewrite that
   stays round-synchronous is the threshold decomposition: process the membership levels from high to
   low, and at each level run the Boolean semi-naive closure of the newly admitted edges over the
   current cut (values are then constant within a level, so no pair is re-derived inside a level, and
   a pair never re-enters Δ at a lower level because max cannot lower it). Its combine candidates are
   bounded by Σ_{(x, y) ∈ R*} outdeg_{≥ level(x, y)}(y) ≤ the reference count for the right-extension
   step, plus the left-extension step of the same size (derived bound, not measured; not implemented
   here). Cost: one filter per level (100 levels) and a nested fixpoint.
2. **The antijoin of datalog_tc does not transfer.** `mask(C, X, negate=True)` is correct only when
   the semiring add cannot change a stored value. For FUZZY the change detection has to be `update`
   itself, which also means every C tuple is written through to update (ratio_cand 5.934 instead
   of datalog_tc's ~2).
3. **`update` detects change with `Semiring.equal`**, whose tolerance is absolute 1e-9 for values
   below 1. Membership degrees were quantized to 1/100 steps so that two distinct values are
   never within tolerance; with continuous random memberships an improvement below 1e-9 would be
   silently dropped (harmless here, fatal for PROB accumulation — see cky_parsing §7).
4. **Pure-Python budget.** The specified large size (n=2000) runs correctly but takes 252.3 s for
   the algebra program (measured, above); "large" was reduced to n=800 (31.7 s).

## 8. Shape seen by the hardware
R is sparse (n × 6 non-zeros, 1,800 tuples at small scale) and stationary across rounds; X grows to
a dense n² relation (density 0.997 small, 0.998 large) and is the probe table of
update. Δ starts at |R|, peaks at 64,169 tuples (small; 453,551 large) around round 3-4
and then decays over 41 (small) / 48 (large) rounds — a long tail of small
rounds. The combine dominates the candidates (71% at small scale, the rest is update).
The query phase is multi-query by construction: Q × 4 query tuples probe the dense closure once
(38,272 candidates at small scale, 1% of the total), so the closure is the
shared operand and the queries are the small side of the hash join. Values are 32-bit memberships
with mul = min and add = max: the PE needs two comparators and no multiplier.
