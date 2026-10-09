# automata_rpq — regular path queries as product-graph reachability

**Category**: automata · **Semiring(s)**: boolean · **Execution structure**: frontier-fixpoint
**Module**: `softwareworkload/automata_rpq/workload.py` · **Test**: `tests/test_workload_automata_rpq.py`

## 1. Problem and reference algorithm
A labelled directed graph g(u, l, v) (n nodes, m = 5n edges, 5 labels, so one edge per label per node
on average), an NFA with 6 states and no ε-transitions — the Glushkov position automaton of a random
regular expression with 5 symbol occurrences that contains exactly one two-label starred group, e.g.
`e (e|a) (c|e)*` (seed 0) — and Q source nodes. Answer: every (source, v) such that some path from the
source to v spells a word of the NFA's language. This is reachability from (source, start) in the
product graph whose nodes are (graph node, NFA state).
`native()` runs a BFS on the product graph from each source; for a dequeued (v, s) it loops over the
NFA transitions (s, l, t) and, for each, over the l-labelled out-edges of v, so every inner iteration
is one product-graph edge; a node's answers are the visited (v, s) with s accepting. It counts one
product-graph edge traversal per ((v, s), edge (v, l, w), transition (s, l, t)) triple, Σ over the
visited product nodes; the (v, s) × transition pairs it examines (`native_probes.nfa_first`) and the
(v, s) × out-edge pairs a graph-first loop would examine (`native_probes.graph_first`) are reported
in `extra` for the join-order discussion.

## 2. Mathematical decomposition
F_0(q, v, s) = [v = source_q ∧ s = start].   Each round:
C(q, w, t) = ⋁_{v, s, l} Δ(q, v, s) ∧ g(v, l, w) ∧ nfa(s, l, t);   Δ' = C ∧ ¬X;   X' = X ∨ Δ'.
Stop when Δ' = ∅ (the round index is the BFS depth in the product graph).
answers(q, v) = ⋁_s X*(q, v, s) ∧ accept(s).
Linear recursion over Δ, so the single-delta rule is exact; Boolean OR is idempotent, so each product
node enters Δ once and the antijoin is valid.

## 3. Algebra program
    G(v, l, w) = Relation.from_set(edges);  Nfa(s, l, t);  Acc(s);  F0(q, v, s)
    one round, two-join orders (module switch ORDER):
      graph_first : C1 = combine(Δ, G)                # (q,v,s) ⋈ (v,l,w) on v      -> (q,v,s,l,w)
                    C  = combine(C1, Nfa, keep=(q,w,t))   # on (s,l)               -> (q,w,t)
      nfa_first   : C1 = combine(Δ, Nfa)              # (q,v,s) ⋈ (s,l,t) on s      -> (q,v,s,l,t)
                    C  = combine(C1, G, keep=(q,w,t))     # on (v,l)               -> (q,w,t)
      product_first (rewrite, measured): PG = combine(G, Nfa, keep=(v,s,w,t)) once before the loop;
                    C  = combine(Δ, PG, keep=(q,w,t))     # on (v,s)
      New   = mask(rename(C, w->v, t->s), X, negate=True)                          # antijoin on (q,v,s)
      X, Δ  = update(X, New)
    fixpoint(step, F0);   answers = reduce(mask(X, Acc), keep=(q, v))               # OR over accepting s
Default order: nfa_first (the better of the two prescribed orders, measured below).

## 4. Hotspot event and work efficiency
Event: one product-graph edge traversal. The algebra's second combine (or the single combine of
product_first) examines exactly the reference's triples — every (q, v, s) enters Δ once and is joined
with all its product edges — so ratio_mul − 1 is the first join's candidate count divided by the
traversals: Σ_{visited} outdeg(v) for graph_first (every out-edge of v, whatever its label), Σ_{visited}
|δ(s)| for nfa_first (every transition of s, whether or not v has an edge with that label), and
|PG| / traversals for product_first (the one-off product join, amortized over Q sources and all rounds).

Measured (harness, small n=3,000 m=15,000 Q=32, seed 0, regex `e (e|a) (c|e)*`, 11 transitions,
accepting [2, 3, 4, 5], 2026-09-25); 23 rounds, 57,034 product nodes visited
(1,782 per source), 41,581 answers, native 0.06 s:

| order | ratio_mul | ratio_sr | ratio_cand | algebra time |
|---|---|---|---|---|
| graph_first | 3.502 | 3.738 | 5.887 | 0.68 s |
| nfa_first (default) | 1.992 | 2.228 | 4.377 | 0.57 s |
| product_first | 1.288 | 1.524 | 3.673 | 0.56 s |

Seeds 0-2 (small; regexes `e (e|a) (c|e)*`, `e (e|b)* d e*`, `c (a|b)* (d|b)`): graph_first ratio_mul
3.502-3.532, nfa_first 1.992-2.005, product_first 1.193-1.288; ratio_cand 5.667-5.887 /
4.159-4.377 / 3.348-3.673.
Measured (harness, large n=30,000 m=150,000 Q=64, seed 0, regex `a c* (d|b)* d`, 2026-09-25); 44
rounds, 1,779,629 product nodes visited (27,806 per source), 593,506 answers, native
3.1 s: graph_first ratio_mul 3.505 / ratio_sr 3.601 / ratio_cand 5.580
(29.6 s); nfa_first 2.004 / 2.100 / 4.079 (25.6 s); product_first
1.127 / 1.223 / 3.202 (22.0 s).
Measured (harness, scale "spec_large" n=30,000 m=150,000 Q=128 — the source count the task specified,
seed 0, nfa_first, 2026-09-25): ratio_mul 2.004, ratio_sr 2.100, ratio_cand 4.079; 105
rounds, 3,511,153 product nodes visited; native 6.2 s, algebra 57.4 s. The ratios are
those of Q=64 (the per-source work is identical); the time is over the 40 s budget, which is why
"large" uses Q=64 (decision marker in `SCALES`).

Explanation. The reference's probe counts confirm the accounting: at small scale it examined
114,036 (v, s) × transition pairs against 114,994 traversals, and a graph-first loop would
have examined 287,690 (v, s) × out-edge pairs — exactly the first-join candidates of the two
orders (nfa_first ratio_mul − 1 = 0.992, graph_first 2.502). nfa_first wins
because a state has 2-3 transitions while a node has 5 out-edges, and with one edge per label per node
each transition matches one edge on average, so the NFA-first waste equals the useful work
(ratio_mul ≈ 2). product_first pays only the one-off join of the two static relations
(|PG| = Σ_l m_l × |δ_l|). ratio_sr ≈ ratio_mul: Boolean OR-merges are rare in the fused reduce
because most (q, w, t) are produced once per round. ratio_cand adds the antijoin probe of every C
tuple against X and the update write of every new product node (mask + update = 43%
of all candidates for nfa_first at small scale) and the readout (mask + reduce over X, once).

## 5. scan() functions used
none.

## 6. Control residue
Loop termination (|Δ| = 0); construction of the four input relations from the instance; the NFA
construction (Glushkov) is instance generation, shared by both sides. Nothing else.

## 7. Where the algebra is insufficient or awkward
1. **Join order, again.** The two prescribed orders differ by 1.76 × in ratio_mul at small
   scale (1.75 × at large) for the same result; the algebra states the three-way join but
   not its order. The rule is the usual one: join the frontier with the operand whose per-key fan-out
   is smaller (|δ(s)| ≈ 2-3 versus outdeg(v) = 5 here); with a denser graph or a sparser NFA the
   winner flips, so the choice needs cardinality information at planning time.
2. **The static join can be hoisted.** G ⋈ Nfa on the label does not depend on the frontier; joining
   it once (product_first) turns every round into a single combine whose candidates equal the
   reference's traversals, at the price of a one-off join that examines 33,136 (edge, transition)
   pairs (small) and of storing the product-edge relation instead of the m = 15,000 graph edges. This is what "reachability on the product graph" means literally, and it is the
   efficient program (ratio_mul 1.288 small, 1.127 large). Nothing in the algebra is
   missing; a compiler pass (loop-invariant join hoisting) is.
3. **Antijoin + update.** As in datalog_tc, every C tuple is probed against X and every new tuple is
   written once; ratio_cand ≈ ratio_mul + 2 × (|C| / traversals) is the price of the semi-naive
   frontier and cannot be removed inside the operator set (a fused "combine-antijoin-update" would
   count once).
4. **Instance sensitivity, not an algebra issue.** With m/n = 5 edges per node over 5 labels, a
   single-label loop x* has mean out-degree 1 in its sub-graph (percolation threshold) and the reach
   per source varies by orders of magnitude between seeds; the generator therefore requires one
   two-label starred group, which gives mean out-degree 2 and a giant component (decision marker in
   `_random_regex`).
5. **Pure-Python budget.** The specified Q=128 at n=30,000 runs correctly but takes 57.4 s of
   algebra time (the reference interpreter's `update` copies the X dictionary, 3,511,153 keys at
   the end, in every one of the 105 rounds); "large" keeps the graph and halves Q (25.6 s).

## 8. Shape seen by the hardware
G is sparse and stationary (m tuples, 150,000 at large scale), Nfa is tiny (11 tuples) and
stationary, X grows to 57,034 (small) / 1,779,629 (large) product-node tuples and is the
antijoin probe table. Rounds = BFS depth = 23 (small) / 44 (large); Δ peaks at
9,575 (small) / 270,496 (large) tuples then decays. The Q sources are batched in one
relation (attribute q), so all queries share G and Nfa in every round (the multi-query sharing that
the datalog_tc document defers to kg_multihop). Combine dominates (46% of candidates for nfa_first at
small scale; mask 32%, update 11%); values are 1-bit, so the PE is a hash
probe plus AND/OR, and the cost is entirely in key handling.
