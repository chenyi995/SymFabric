# datalog_samegen — same generation on a random rooted tree by semi-naive Datalog

**Category**: logic · **Semiring(s)**: boolean · **Execution structure**: sparse-fixpoint
**Module**: `softwareworkload/datalog_samegen/workload.py` · **Test**: `tests/test_workload_datalog_samegen.py`

## 1. Problem and reference algorithm
    sg(x, y) :- flat(x, y).
    sg(x, y) :- up(x, x1), sg(x1, y1), down(y1, y).
Instance: a random rooted tree with n nodes built level by level (level k holds min(b^k, remaining)
nodes, every node picks a uniformly random parent in the previous level); up = child -> parent,
down = parent -> child, flat = random ordered sibling pairs (same parent, x != y). The flat pairs
are drawn per depth with a quota proportional to the number of sibling pairs at that depth
(stratified sampling): one flat pair at depth k induces about b^(2(d-k)) sg facts, so a plain
uniform draw would make |sg| depend on whether a single near-root pair was drawn. Small: n = 400,
b = 3, 120 flat pairs, depth 6. Large: n = 5000, b = 4, 800 flat pairs, depth 6.

`native()` runs two goldens with Python sets and asserts that they agree:
* naive: X_{r+1} = flat ∪ body(X_r) from X_0 = flat until nothing changes; the whole rule body is
  re-evaluated on the full X_r every round;
* semi-naive: the body is evaluated on the previous round's new facts only, new = body(Δ) − X.

Both count the same hotspot event, in the loop nesting `for (x1,y1) in Δ: for x in children(x1):
+1; for y in children(y1): +1`: one event per (up, sg) pair combined plus one per (partial, down)
pair combined. native_ops is the semi-naive count (the reference that matters); the naive count
is reported in `extra`.

## 2. Mathematical decomposition
X_0 = Δ_0 = flat. Each round, with Δ(x1, y1) the new facts of the previous round:
    T(x, y1) = ⋁_{x1} up(x, x1) ∧ Δ(x1, y1)
    C(x, y)  = ⋁_{y1} T(x, y1) ∧ down(y1, y)
    Δ' = C ∧ ¬X;   X' = X ∨ Δ'.
The rule is linear in sg, so the single-delta form is exact. up and down are functions (every
node has one parent), so C has no duplicate derivations: (x, y) comes only from (parent(x),
parent(y)). Stop when Δ' = ∅.

## 3. Algebra program
    D1  = rename(Δ, x->x1, y->y1)                                # (x1, y1), no record cost
    T   = combine(UP(x, x1), D1(x1, y1), keep=(x, y1))           # join on x1: (up, sg) pairs
    C   = combine(T(x, y1), DOWN(y1, y), keep=(x, y))            # join on y1: (partial, down) pairs
    New = mask(C, X, negate=True)                                # antijoin on (x, y)
    X, Δ = update(X, New)                                        # OR-merge; Δ = new facts
    fixpoint over the five lines until Δ is empty.
This is join order "up_first". Order "down_first" is T = combine(D1, DOWN, keep=(x1, y)),
C = combine(UP, T, keep=(x, y)); both are implemented (`algebra(inst, order)`).

Join order choice, measured (small, seeds 0-9, 2026-09-25; combine candidates, join1 + join2):

| seed | up_first | down_first | seed | up_first | down_first |
|---|---|---|---|---|---|
| 0 | 4472 + 9438 = 13910 | 3894 + 9438 = 13332 | 5 | 3523 + 8137 = 11660 | 4351 + 8137 = 12488 |
| 1 | 8027 + 16251 = 24278 | 6892 + 16251 = 23143 | 6 | 4324 + 9109 = 13433 | 4067 + 9109 = 13176 |
| 2 | 3305 + 7099 = 10404 | 3021 + 7099 = 10120 | 7 | 1439 + 3090 = 4529 | 1457 + 3090 = 4547 |
| 3 | 3067 + 6844 = 9911 | 3212 + 6844 = 10056 | 8 | 7845 + 17980 = 25825 | 8802 + 17980 = 26782 |
| 4 | 6394 + 15521 = 21915 | 7300 + 15521 = 22821 | 9 | 2907 + 6775 = 9682 | 3207 + 6775 = 9982 |

The second join costs Σ_Δ c(x1)·c(y1) in both orders (identical column). The first join costs
Σ_Δ c(x1) (up_first) or Σ_Δ c(y1) (down_first); flat is not symmetric, so the sign of the
difference is random: up_first is smaller in 6 of 10 seeds, the spread is −4.7% to +7.1% of the
total. There is no work-based reason to prefer one order; up_first is the default because its two
candidate counts are exactly the two loop counts of the semi-naive golden, which makes ratio_mul
= 1 an exact identity rather than an approximation.

## 4. Hotspot event and work efficiency
Event: one (up, sg) pair combined or one (partial, down) pair combined = one AND.
Measured (harness, small, seed 0, 2026-09-25): native_ops 13,910 (semi-naive golden), naive golden
46,402 events (3.34× the semi-naive count, derived); ratio_mul 1.000, ratio_sr 1.000, ratio_cand
2.357; 6 rounds, Δ sizes [381, 596, 1863, 6498, 100, 0], |sg| = 9,558 = 120 flat + 9,438 derived.
Seeds 1 and 2: ratio_mul 1.000, ratio_cand 2.339 and 2.365, |sg| 16,372 and 7,219.
Large (seed 0): native_ops 383,247, naive golden 986,784 (2.57×, derived); ratio_mul 1.000,
ratio_sr 1.000, ratio_cand 2.568; 6 rounds, Δ sizes [3360, 11050, 34470, 57558, 194081, 0],
|sg| = 301,319. Timing: small 0.02 s algebra / 0.01 s both goldens; large 0.84 s / 0.48 s.
ratio_sr = 1: add_ops = 0, no duplicate derivations (up and down are functions). ratio_cand > 2:
mask and update each touch every C tuple once (small seed 0: 13,910 + 9,438 + 9,438 = 32,786).
Against the naive golden the algebra would look 3.3× (small) / 2.6× (large) better than the
reference; that ratio is not reported as the work-efficiency ratio, the semi-naive one is.

## 5. scan() functions used
none.

## 6. Control residue
Loop termination (|Δ| = 0), one scalar per round. Instance construction (tree, flat) is not part
of either side's count.

## 7. Where the algebra is insufficient or awkward
Nothing is missing; ratio_mul = 1 exactly. Two awkward points:
* The antijoin is provably a no-op on this instance family (sg is irreflexive because flat has
  x != y and parents are unique, so a derived pair is never a flat pair and is derived only once):
  the mask output equalled its input in every round (small seed 0: [381, 596, 1863, 6498, 100,
  0] in and out). It costs 29% of all candidates (9,438 of 32,786 at small seed 0). An
  engine that knew the functional dependencies could skip it; the algebra has no way to state
  them.
* There is no three-way join: the programmer sequences two binary combines and the intermediate
  T(x, y1) is materialised (4,472 tuples at small seed 0). A pipelined n-ary join would avoid the
  intermediate; its candidate count would be the same.

## 8. Shape seen by the hardware
Sparse Boolean relations. UP and DOWN are functional relations of n − 1 tuples (399 / 4,999);
FLAT has 120 / 800 tuples; X grows to 9,558 / 301,319. Δ grows by about b per round and collapses
at the leaves (Δ sizes above); 6 rounds at both scales (depth 6: five productive rounds after the
depth-1 flat pair plus one empty round). The second combine dominates (join2 / join1 ≈ 2.1 at
small seed 0, the mean number of children of y1). The probe table of the antijoin is X, the
largest operand. No multi-query sharing.
