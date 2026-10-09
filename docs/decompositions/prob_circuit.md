# prob_circuit — layered probabilistic-circuit evaluation under three semirings

**Category**: probabilistic · **Semiring(s)**: prob (+,×), logprob (LSE,+), counting (+,× over integers) · **Execution structure**: dag
**Module**: `softwareworkload/prob_circuit/workload.py` · **Test**: `tests/test_workload_prob_circuit.py`

## 1. Problem and reference algorithm
A random layered probabilistic circuit in the pyjuice / RAT-SPN shape: W leaves, then L internal
layers alternating product (odd) and sum (even), the last sum layer being a single root. Every
node draws its children without replacement from the previous layer; sum weights are
Dirichlet-normalised; product fan-in f_p = 2, sum fan-in f_s = 8, root fan-in W. Small: L = 6,
W = 64, B = 16 (1,472 edges); large: L = 8, W = 256, B = 32 (8,448 edges). The same DAG is
evaluated for a batch of B inputs under three parametrisations:
prob — leaves are Bernoulli(θ_l) densities over variable v(l) = l mod W/2, evaluated at B complete
assignments (likelihood); logprob — the same in the log domain; counting — leaves are the literal
indicators [X_v(l) = pol(l)] evaluated at B partial assignments (W/8 observed variables, so 12.5%
of leaf values are 0), every weight is 1: the number of complete sub-circuits consistent with the
evidence, which equals the model count when the circuit is deterministic and decomposable. The
generator guarantees only the layered structure ("decomposable-ish"), so the counting run is the
counting-semiring evaluation of the DAG, not a certified #SAT answer.
`native()` is layer-wise numpy: gather the child values into a (B, width, fan-in) block, then a
weighted sum / log-sum-exp / product along the fan-in axis (Python integers for counting). It
counts one edge evaluation per batch element and per edge, densely, for each of the three runs.

## 2. Mathematical decomposition
Sum node:      S(b, n) = ⊕_c W(n, c) ⊗ V(b, c)          a semiring join on c fused with a reduce onto (b, n)
Product node:  P(b, n) = ⊗_{k < f_p} V(b, child_k(n))    a mul-fold: no ⊕ over c, every child must be present
Layers are evaluated bottom-up in a fixed order (no fixpoint). The batch index b is an extra key
attribute carried by every operator, so the B inputs share the circuit operands (Q dimension).

## 3. Algebra program
    V_0(b, c) = Relation.dense(leaves)                                            # B x W
    sum layer:     V'(b, n)  = combine(W(n, c), V(b, c), keep=(b, n))                # f_s·width·B candidates, fused ⊕
    product layer: P_k(b, n) = combine(E_k(n, c), V(b, c), keep=(b, n))   k < f_p    # slot-k gather, unit weights, no ⊕
                   V'(b, n)  = combine(P_0, P_1) ... combine(·, P_{f_p-1})           # natural join on (b, n): ⊗
    V(b, c) = rename(V', n -> c)
    fold alternative (measured in sec. 4): V' = reduce(combine(E(n, c), V(b, c)), keep=(b, n), use_mul=True)
Run three times: (PROB, weights), (LOGPROB, log weights), (COUNTING, unit weights).

## 4. Hotspot event and work efficiency
Event: one edge evaluation per batch element — a child value multiplied by its weight and
accumulated into a sum node, or a child value folded into a product node. Native count:
3 · B · |edges| = 70,656 (small), 811,008 (large).
Measured (harness, small, seed 0, 2026-09-25): ratio_mul 1.097, ratio_sr 1.722, ratio_cand 1.097;
seeds 1 and 2: ratio_mul 1.093 and 1.100. Large (seed 0): ratio_mul 1.097, ratio_sr 1.745,
ratio_cand 1.097. Per semiring (small seed 0 / large seed 0): prob and logprob 1.130 / 1.121;
counting 1.030 / 1.050 (small seeds 1, 2: 1.018, 1.038).
Why above 1: a sum layer costs exactly one ⊗ per edge (fused MAC). A product layer costs
2·f_p − 1 = 3 ⊗ per node instead of f_p = 2 edge evaluations, because moving a child value to
its parent's key is a join and a join multiplies (by the unit weight) before the real ⊗ between
the slots. Small scale: (2·512 + 3·(3·64) + 64) / 1,472 = 1.130, the measured prob ratio.
ratio_cand equals ratio_mul: every candidate is one ⊗ and nothing else is touched. ratio_sr
1.72–1.79: a MAC is one ⊗ and one ⊕ while the native count is edge evaluations.
Counting below prob: the relation never holds zero-valued tuples, so a 0 leaf (12.5% of leaves)
removes the edges above it from the joins, while the golden evaluates them densely. No root was 0
at either scale; the largest root count was 8,581,309 (24 bits, small seed 0) and
249,713,559,286,470 (48 bits, large seed 0).
Fold alternative (measured, seed 0): small ratio_mul 1.122, ratio_cand 1.252; large 1.115, 1.236.
The ⊗ count is the same as the chain (f_p gathers plus f_p − 1 folds) except that the fold does
not shrink with zeros, and the separate reduce pass re-touches every gathered tuple. Its counting
result is wrong for 16/16 (small) and 32/32 (large) batch elements (sec. 7). The chain is used.
Underflow: minimum root likelihood 6.9e-4 (small seed 0), 2.4e-6 (large seed 0); no root below
the float32 normal minimum, and a float32 evaluation of the same circuit produced no zero root.
logprob agreed with log(prob) to 2.7e-15 (small) and 1.6e-14 (large) absolute. The random circuit
is not decomposable, so each root value multiplies only about 2^(L/2) leaf densities; a
decomposable circuit over all W/2 = 128 variables would produce likelihoods near 2^-128, below the
float32 minimum 1.2e-38, which is where the (LSE,+) run stops being a cross-check and becomes
the only usable one.

## 5. scan() functions used
none.

## 6. Control residue
The layer loop (L steps); building the weight and slot-edge relations from the instance (a layout
step, |edges| tuples per semiring, done once); reading the B root values. No per-tuple Python work
outside the operators.

## 7. Where the algebra is insufficient or awkward
1. Product nodes are a mul-fold, not a semiring aggregation. `reduce(use_mul=True)` folds the
   tuples that are present, so a child whose value is the semiring zero (absent under zero
   suppression) is silently skipped: the counting run is wrong for every batch element (16/16 and
   32/32 measured). A hardware mul-reduce needs either a completeness guarantee (dense groups) or
   an explicit "missing child = zero" rule; the join chain provides the latter because a natural
   join drops keys absent on either side. This is a semantic gap of `reduce(use_mul=True)`, not a
   missing operator.
2. The gather (re-keying a child value to its parent) costs one ⊗ per edge although it is data
   movement in hardware; ratio_mul overstates product-layer arithmetic by (2·f_p − 1)/f_p (1.5 for
   binary products). A combine whose unit-weight side is recognised as a permutation would bring
   ratio_mul to 1.0 and leave ratio_cand (data movement) unchanged.
3. The chain needs the edges split by child slot (E_0, E_1, ...). That is the tensorised layout
   PC libraries already use (a width × f_p child-index tensor), so it is a layout requirement,
   not a residue, but it means the algebra program depends on a fixed fan-in per layer.

## 8. Shape seen by the hardware
Sparse-structured but complete operands: W(n, c) has f_s·W tuples (512 small, 2,048 large), each
E_k has W tuples (64 / 256), V(b, c) is dense B × W (1,024 / 8,192 tuples). Every layer is a
small sparse matrix (weights or slot edges) applied to a dense value matrix with B columns: the
weights are the stationary operand and the batch streams through, so B is the multi-query
dimension. 18 (small) / 24 (large) operator records per semiring, all ⊗ in combine. A sum layer is
a fan-in-8 reduction tree per (b, n); a product layer is an f_p-way elementwise multiply of
gathered vectors. The three semirings run the identical dataflow; only the PE's (⊕, ⊗) and the
value width change (float64, and integers that reach 48 bits at the large scale — the 64-bit
COUNTING width of the semiring table would overflow one or two layers deeper).
