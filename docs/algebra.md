# The SymFabric operator algebra (v1, as implemented in `algebra/`)

## 1. Types

- **Semiring** (K, ⊕, ⊗, 0̄, 1̄) with three flags that matter for execution: `idempotent`
  (a ⊕ a = a: OR, min, max — deltas are exact), `selective` (a ⊕ b ∈ {a, b} — an arg-selection
  exists), `numeric` (floats — equality needs a tolerance). Implemented: boolean, counting, prob
  (+,×), real (+,×), logprob (LSE,+), viterbi (max,+), minplus (min,+), bottleneck (min,max),
  fuzzy (max,min). Workloads may define their own (hmm_viterbi's `viterbi_bp` carries a back-pointer
  in the value).
- **Relation** R : D₁ × … × Dₙ → K with a schema of attribute names, stored zero-suppressed
  (only values ≠ 0̄), with optional domain sizes per attribute (used for key widths and density in
  the trace, never for semantics).

## 2. Operators and what they count

| operator | definition | counted as |
|---|---|---|
| `rename(R, map)` | attribute renaming, no data movement | 0 (alias record) |
| `combine(R, S, keep=None)` | natural join on shared attributes, value = R ⊗ S; with `keep`, fused with the reduce onto `keep` | one candidate and one ⊗ per matching pair; one ⊕ per duplicate output key |
| `reduce(R, keep, use_mul=False)` | group by `keep`, fold the rest with ⊕ (or ⊗) | one candidate per input tuple, one fold per merge |
| `filter(R, pred)` | keep tuples with pred(key, value) | one candidate per input tuple |
| `mask(R, S, negate)` | semijoin / antijoin on the support of S (set membership, not subtraction) | one candidate per probe |
| `update(X, D)` | X'(k) = X(k) ⊕ D(k); returns X' and the changed keys with new values | one candidate per delta tuple; one ⊕ per existing key |
| `select_k(R, k)` | the k best values under the semiring's order | one candidate per input tuple |
| `scan(R, f)` | elementwise O(1) map, may re-key; duplicates merged with ⊕ | one candidate per input tuple |
| `fixpoint(step, X0)` / `iterate(n, step, X0)` | monotone semi-naive loop / fixed count | loop control only |

Counting convention for work efficiency: one hotspot event of the reference algorithm (a MAC, an
edge relaxation, a clause visit) corresponds to one ⊗ of the algebra program (`ratio_mul`); the
merges (`ratio_sr`) and every tuple touched by any pass (`ratio_cand`) are reported separately.

## 3. Semantic rules learned from the workloads (2026-09-25)

1. **Deltas are exact only for idempotent ⊕.** `update` reports a key as changed when its value
   changed; for boolean/min/max that is the semi-naive delta. For (+,×) every touched key changes
   and contribution counting is the program's responsibility.
2. **Antijoin-based novelty (`mask(C, X, negate=True)`) is only correct when ⊕ cannot change a
   stored value** (boolean). For fuzzy/min-plus closures the change test must be `update` itself
   (fuzzy_inference §7).
3. **`reduce(use_mul=True)` is unsound under zero suppression**: a group member equal to 0̄ is
   absent and is skipped by the fold (prob_circuit §7, counting semiring wrong on every batch
   element). Use chains of binary `combine`, or count members and mask incomplete groups.
4. **`update`'s change test uses `Semiring.equal`** (relative 1e-9, absolute for |x| < 1).
   Accumulating small probabilities through `update` can silently drop contributions
   (cky_parsing §7); fuse the sum into `combine(keep=…)` instead, or give non-idempotent semirings an
   exact-merge mode.
5. **Non-linear recursion** needs Δ ⋈ X_new ∪ X_old ⋈ Δ (or, for transitive closure specifically,
   the provably complete Δ ⋈ X_new); the driver only passes (X, Δ), so X_old lives in the program
   (datalog_nonlinear_tc §7).
6. **h^max is not "h^add in another semiring"**: it folds preconditions with max and composes
   action cost with +, two different operations; the (min,max) semiring computes something else
   (planning_hadd §7, locked in a test).

## 4. What the workloads asked for that v1 does not have

Collected from the section-7 entries of the 18 decompositions; each item names the workloads that
paid for its absence and the measured price.

| missing capability | who paid | measured price | v2 proposal |
|---|---|---|---|
| **INDEX / gather** (keyed point access without a ⊗) | astar_grid (heuristic lookup), vsa_cleanup, planning_hadd (three identity joins per tuple), ctrlg_dfa_hmm, prob_circuit (unit-weight moves) | 1.1–1.5× extra ⊗ (vsa 1.12–1.22, ctrlg 1.06–1.15, prob_circuit 1.10); 4× in planning | `gather(T, keys)` counted as candidates, not ⊗ |
| **grouped select** (k best per group key) | kg_multihop (Q selects), vsa_cleanup (Q argmax filters), sat_bcp (watch replacement as a (min,+) reduce), beam_search (rank) | Q² probes in vsa (1–1.2 % of work), 256 select calls in kg | `select_k(R, k, by=group_attrs)` |
| **persistent priority queue with O(1) pop and point delete** | graph_bfs_sssp (b), astar_grid | select over the whole open list: 240× / 2433× (sssp), 45–197× (A*) of the relaxations | a SELECT unit that keeps state across steps (systolic PQ); in the model `systolicarch` already prices it incrementally |
| **early-exit gather ("first match")** | sat_bcp watched form | the join reads 3.1 literals per touched clause, the propagator 1.9–2.0: ⊗ 5× | per-group state machine over touched clauses |
| **fused combine → antijoin → update** | every fixpoint (datalog_tc, samegen, rpq, fuzzy, kg) | ratio_cand ≈ 2–2.6 where ratio_mul = 1 | one fused round operator; `pim.model.fused_round` already prices it |
| **join-order / cardinality planning** | cky_parsing (pair-first 3.3–6.0× vs rule-first 1.07–1.19×), automata_rpq (graph-first 3.5× vs product-first 1.13–1.29×), triangle_count (wedge join 1.4–1.6× vs intersection) | up to 6× | planner with fan-out statistics; loop-invariant join hoisting; intersection-driven multiway join for cyclic queries |
| **affine key constraints** (j − i = ℓ) | cky_parsing | re-keying scans, 12–20 % of candidates; filter form 10.8× | probe key = build key + constant in the join unit |
| **functional-dependency knowledge** | datalog_samegen | a provably empty antijoin costs 29 % of candidates | optional; a planner hint |
| **priority-ordered frontier (Dijkstra order) inside a round-synchronous loop** | fuzzy_inference (4.2–5.7×), planning_hadd (4–6.5×, 12.4× deep), graph_bfs_sssp (a) (1.8–2.4×) | re-derivation of the same key at better values | threshold decomposition (levels), delta-stepping buckets = grouped select |

Nothing in this table needed a new value algebra: the semirings covered every workload. The gaps
are in indexing, selection, fusion and planning — the "structure algebra" half of the plan.
