# datalog_tc — transitive closure by semi-naive Datalog

**Category**: logic · **Semiring(s)**: boolean · **Execution structure**: sparse-fixpoint
**Module**: `softwareworkload/datalog_tc/workload.py` · **Test**: `tests/test_workload_datalog_tc.py`

## 1. Problem and reference algorithm
    path(i, j) :- edge(i, j).      path(i, k) :- path(i, j), edge(j, k).
`native()` runs a BFS from every source vertex and counts one relaxation per out-edge scanned
from a vertex already known reachable (the source's own out-edges are base facts).

## 2. Mathematical decomposition
X_0 = Δ_0 = E.  Each round:  C(i,k) = ⋁_j Δ(i,j) ∧ E(j,k);  Δ' = C ∧ ¬X;  X' = X ∨ Δ'.
Stop when Δ' = ∅. Linear recursion, so the single-delta rule is exact (Codex plan §5.1).

## 3. Algebra program
    C        = combine(rename(Δ, j->m), rename(E, i->m), keep=(i, j))    # (i,m),(m,j) -> (i,j), OR-merged
    New      = mask(C, X, negate=True)                                     # antijoin on (i,j)
    X, Δ     = update(X, New)                                              # OR-merge; Δ = new facts
    fixpoint over the three lines until Δ is empty.

## 4. Hotspot event and work efficiency
Event: one (path(i,j), edge(j,k)) pair examined = one AND. The algebra's combine candidates equal
the BFS relaxations exactly because every path fact enters Δ exactly once.
Measured (harness, small, seed 0, 2026-09-25): ratio_mul 1.000, ratio_sr 1.264, ratio_cand 2.060,
11 rounds, |TC| = 36,292 from 600 edges on 200 vertices. Large (2500 vertices, 10,000 edges):
ratio_mul 1.000, 12 rounds, |TC| = 6,007,409.
ratio_sr > 1: duplicate derivations of the same (i,j) inside one round are OR-merged (26% of
candidates at small scale). ratio_cand ≈ 2: every C tuple is probed once against X (mask) and
every new fact is written once (update).

## 5. scan() functions used
none.

## 6. Control residue
Loop termination test (|Δ| = 0), one scalar per round.

## 7. Where the algebra is insufficient or awkward
Nothing missing. The dense-matrix formulation (Boolean X ← X ∨ X·E) would cost O(n³); the
semi-naive delta form is what makes ratio_mul = 1.

## 8. Shape seen by the hardware
Sparse Boolean relations; Δ shrinks after the first rounds (delta sizes recorded in `extra`).
X grows to |TC| which can be O(n²) — 6 M facts at the large scale — so the antijoin probe table
is the large operand, not the edge relation. No multi-query sharing in this formulation; the
"Q sources sharing E" view appears in kg_multihop.
