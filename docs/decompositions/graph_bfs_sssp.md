# graph_bfs_sssp — single-source shortest paths and BFS hop distance

**Category**: graph · **Semiring(s)**: minplus, boolean · **Execution structure**: frontier-fixpoint (programs a, c) and select-loop (program b)
**Module**: `softwareworkload/graph_bfs_sssp/workload.py` · **Test**: `tests/test_workload_graph_bfs_sssp.py`

## 1. Problem and reference algorithm
Random sparse digraph with m distinct edges on n vertices, integer weights 1..10, one random
source (n = 2000, m = 10000 small; n = 20000, m = 100000 large). Computed for every reachable
vertex v: dist(v), the weighted shortest-path distance, and hops(v), the BFS hop distance.
`native()` is heapq Dijkstra with lazy deletion plus a queue BFS; both count one edge relaxation
per out-edge scanned from a settled (Dijkstra) or dequeued (BFS) vertex.
Three algebra programs solve the same instance and `algebra()` returns all three results:
(a) frontier relaxation in (min,+) (semi-naive Bellman-Ford), (b) Dijkstra with SELECT, one
select per settled vertex, and (c) BFS levels in the Boolean semiring with a level counter.

## 2. Mathematical decomposition
(a) dist_0 = Δ_0 = {s ↦ 0}. Round k: cand(v) = ⊕_u Δ(u) ⊗ E(u, v) = min_u (Δ(u) + w(u, v));
dist' = dist ⊕ cand (elementwise min); Δ' = {v : dist'(v) < dist(v)}. Stop when Δ' = ∅.
Bellman-Ford restricted to the vertices improved in the previous round; exact because (min,+) is
idempotent and selective.
(b) open : V → tentative distance. Step: u* = argmin open (ties: smallest id); dist(u*) = open(u*);
open' = (open \ {u*}) ⊕ {v ↦ open(u*) + w(u*, v) : (u*, v) ∈ E, v not settled}. One SELECT per
settled vertex; the sequential dependence between steps is inherent (Dijkstra's invariant).
(c) visited_0 = Δ_0 = {s}; cand = ⋁_u Δ_k(u) ∧ E(u, v); Δ_{k+1} = cand ∧ ¬visited_k;
visited_{k+1} = visited_k ∨ Δ_{k+1}; level(v) = k + 1 for v ∈ Δ_{k+1}.

## 3. Algebra program
(a), one fixpoint round:
    cand      = combine(Δ(u), E(u, v), keep=(v,))          # (u) x (u,v) -> (v), min-merged
    cand      = rename(cand, v -> u)
    X, Δ      = update(X, cand)                              # X' = min(X, cand); Δ = improved vertices
    fixpoint until Δ = ∅
(b), one step; E is stored as one row relation per source vertex (a CSR row), selected by the
popped vertex id:
    best      = select_k(open, 1)                            # pop: (min,+) order, ties by vertex id
    open      = mask(open, best, negate=True)                # point delete of the popped vertex
    dist      = update(dist, best)                           # settle
    cand      = combine(best, E_row[u], keep=(v,))          # relax the out-edges of u
    cand      = mask(rename(cand, v -> u), dist, negate=True)   # drop candidates into settled vertices
    open      = update(open, cand)                           # min-merge into the open list
    loop until open = ∅
(c), one fixpoint round:
    cand      = combine(Δ(u), E_bool(u, v), keep=(v,))     # (u) x (u,v) -> (v), OR-merged
    new       = mask(rename(cand, v -> u), visited, negate=True)   # visited test
    visited, Δ = update(visited, new)                        # Δ = newly visited = next frontier
    level     = update(level, scan(Δ, const_level(k + 1)))  # level counter, MINPLUS relation
    fixpoint until Δ = ∅

## 4. Hotspot event and work efficiency
Event: one edge relaxation = one out-edge scanned from a settled or dequeued vertex (golden) =
one matching (frontier tuple, edge tuple) pair inside a combine (algebra, one (min,+) or (OR,AND)
mul). The trace holds three programs, two of which recompute the SSSP result, so native_ops =
2 × Dijkstra relaxations + BFS relaxations; the harness ratios are the work-weighted mean of the
per-program ratios. Per-program figures, measured (harness, 2026-09-25):

| scale/seed | D = Dijkstra relax. | B = BFS relax. | (a) relax. | (a)/D | (a) rounds | (b) relax./D | (b) Σ open (select cand.) | (b) pop cand. (select + delete)/D | (b) avg / max open | (c) relax./B | (c) rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small/0 | 9909 | 9909 | 19076 | 1.93 | 13 | 1.000 | 1188068 | 240 | 599 / 937 | 1.000 | 9 |
| small/1 | 9927 | 9927 | 20612 | 2.08 | 15 | 1.000 | 1190427 | 240 | 599 / 946 | 1.000 | 9 |
| small/2 | 9928 | 9928 | 17738 | 1.79 | 12 | 1.000 | 1191854 | 240 | 600 / 945 | 1.000 | 8 |
| large/0 | 99322 | 99322 | 236148 | 2.38 | 17 | 1.000 | 120842582 | 2433 | 6084 / 9556 | 1.000 | 11 |

Harness aggregate over the three programs, measured (harness, 2026-09-25):

| scale/seed | native_ops (2D + B) | mul_ops | ratio_mul | ratio_sr | ratio_cand | t_native s | t_algebra s |
|---|---|---|---|---|---|---|---|
| small/0 | 29727 | 38894 | 1.308 | 2.137 | 82.5 | 0.00 | 0.70 |
| small/1 | 29781 | 40466 | 1.359 | 2.243 | 82.6 | 0.00 | 0.70 |
| small/2 | 29784 | 37594 | 1.262 | 2.051 | 82.5 | 0.00 | 0.71 |
| large/0 | 297966 | 434792 | 1.459 | 2.445 | 813.9 | 0.12 | 53.06 |

(a) relaxes 1.79-2.08× (small, seeds 0-2) and
2.38× (large) the edges Dijkstra relaxes: a vertex whose distance improves again
re-enters the frontier and its out-edges are rescanned. The ratio grows with the instance because
longer shortest paths (more rounds: 12-15 small, 17 large) give more
opportunities to improve a vertex after its first relaxation. Its ratio_cand adds the update probe
of every candidate (3.54× D at large).
(b) relaxes exactly Dijkstra's edges (ratio 1.000 at every seed and scale): every reachable vertex
is popped once and its row is scanned once. Its cost is elsewhere: select_k streams the whole open
relation (|open| candidates per pop) and the point delete streams it again, so the pop cost is
2 Σ|open| = 241685164 candidates at large, 2433× the relaxations
(240× at small): the open list holds on average 0.30 n vertices on this
random graph, so Σ|open| grows as n². This term alone puts the harness ratio_cand at
814 (large), far above the plan's 10× admission line; ratio_mul and ratio_sr do not see it
because select and mask perform no semiring arithmetic. With a SELECT unit that pops and deletes
in O(1) (the systolic priority queue of the plan, section 5) the pop cost would be 2 ops per
settled vertex, 39722 at large, i.e. 0.40× D (derived from the measured pop count).
(c) relaxes exactly the BFS edges (ratio 1.000): the mask against visited makes every vertex
enter the frontier once. ratio_cand ≈ 2.13 (small/0) because every candidate is probed once by the
visited mask and the new vertices are written by update and the level scan.

## 5. scan() functions used
`const_level(key, v) -> [(key, k + 1)]` in (c): writes the current fixpoint round as the level of
a newly visited vertex; one output per input, O(1). None in (a) and (b).

## 6. Control residue
(a) the fixpoint termination test (|Δ| = 0). (b) the loop over the open list, reading the popped
vertex id from the one-tuple select result, and the row lookup E_row[u] (one dictionary access
per pop, the CSR row-pointer read). (c) the round counter k passed to the level scan. All O(1)
per step; no arithmetic outside the operators.

## 7. Where the algebra is insufficient or awkward
1. No point delete. The pop of a priority queue is SELECT plus the removal of the selected
   tuple; the algebra expresses the removal as an antijoin, which streams the whole relation. Both
   halves cost |open| per pop: Σ|open| = 1188068 (small) and 120842582 (large)
   select candidates, doubled by the delete (measured). This is the hardware argument for a
   SELECT unit: a systolic priority queue keeps the order incrementally and pops in O(1), which
   removes the n² term (derived: 2 ops per pop, ratio 0.40× D at large).
2. No keyed index access. The interpreter's combine streams the larger operand, so
   combine(best, E) would read all m edges per pop (one pass over the m = 100000 edges for
   each of the 19861 pops at large). Program (b) therefore stores E as one relation per source vertex and selects the row of
   the popped vertex by key; the algebra has no operator for that (rename is metadata only), it is
   an INDEX access done in control. The recorded candidates are unaffected (matching pairs only).
3. Re-relaxation in (a) is inherent to the frontier form. Avoiding it (delta-stepping, Dial's
   buckets) needs a grouped select — pop all open tuples whose value lies in the current bucket —
   which the algebra does not offer; select_k(open, 1) gives Dijkstra's exact order at the cost
   of item 1.
4. The visited mask in (c) is redundant with update in an idempotent semiring (update's delta is
   exactly the set of new vertices); it is kept to mirror the BFS visited test and costs one probe
   per candidate.
5. Interpreter only: update copies the state relation on every call (O(|X|) per pop in (b)); this
   is wall-clock, not recorded work, and is why the large instance takes 53 s.

## 8. Shape seen by the hardware
E is sparse: m = 100000 tuples at large, key (u, v) with 15 + 15 bits, weight 32 bits. (a) and (c)
stream E once per round: (a) 17 rounds, (c) 11 rounds at large (E streamed
1700000 and 1100000 tuple reads, derived: rounds × m), against frontier sizes per round of
4, 28, 140, 675, 3091, 9248, 11618, 8483, 5771, 3825, 2281, 1163, 493, 215, 78, 23, 0 (a) and 4, 28, 140, 673, 3048, 8546, 6665, 721, 34, 1, 0 (c). A memory-side frontier expansion that reads only the
rows of the frontier would touch Σ deg(frontier) = the relaxation count instead. (b) reads E row
by row (99322 tuples in total) but needs an ordered structure of average size
6084 and peak 9556 entries (large) — the capacity a SELECT unit must hold. Single source:
no multi-query sharing. Output: distances for 19861 of 20000 vertices. The dominant
operator is combine in (a) and (c) and select/mask in (b).
