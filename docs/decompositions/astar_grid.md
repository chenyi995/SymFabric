# astar_grid — A* on a 2-D grid with random obstacles

**Category**: search · **Semiring(s)**: minplus · **Execution structure**: select-loop
**Module**: `softwareworkload/astar_grid/workload.py` · **Test**: `tests/test_workload_astar_grid.py`

## 1. Problem and reference algorithm
H × W grid (60 × 60 small, 250 × 250 large), each cell blocked with probability 0.3,
4-connected, unit edge costs, Manhattan heuristic h (admissible and consistent). Start and goal
lie in the largest connected component: the start is its top-left-most cell, the goal its cell
farthest from the start by Manhattan distance, so a path exists and is long. Computed: the cost
of a shortest start-goal path. `native()` is heapq A* with (f, node) ordering, lazy deletion and
a closed set (no reopening, valid for a consistent heuristic); it counts node expansions and edge
relaxations (every free neighbour scanned from an expanded node). The hotspot event and
native_ops are the relaxations; expansions are reported on both sides.

## 2. Mathematical decomposition
open : cells → g (cost so far), f(n) = g(n) ⊗ h(n) = g(n) + h(n) in (min,+).
Step: n* = argmin_n f(n) over open (ties: smallest cell id); stop if n* is the goal;
closed ← closed ∪ {n*}; for every grid edge (n*, m) with m ∉ closed:
g'(m) = g(n*) ⊗ w(n*, m) = g(n*) + 1, open(m) ← open(m) ⊕ g'(m) = min(open(m), g'(m)).
f is kept incrementally: f(m) = g'(m) + h(m) is formed for the candidates of one expansion only,
and the g of the popped node is recovered from its f with the inverse of the (min,+) mul,
g = f + (−h). One SELECT per expansion; the sequence of expansions is inherently sequential.
Counter-example (plan, section 4.5): dist = fixpoint of dist ⊕ (Δ ⊗ E), the global (min,+)
relaxation of graph_bfs_sssp program (a), reaches every cell of the component and never uses h.

## 3. Algebra program
E is stored as one row relation per cell (its ≤ 4 grid edges), selected by the popped cell id.
    best_f    = select_k(f_open, 1)                          # pop the open node with the smallest f
    stop if best_f is the goal; path cost = f(goal) (h(goal) = 0)
    f_open    = mask(f_open, best_f, negate=True)            # point delete
    best_g    = combine(best_f, scan(best_f, neg_h))         # g = f + (−h): (min,+) mul with the looked-up heuristic
    closed    = update(closed, best_g)                       # expanded set
    cand_g    = combine(best_g, E_row[u], keep=(m,))        # relaxations: g + 1 for every free neighbour
    cand_g    = mask(rename(cand_g, m -> n), closed, negate=True)   # never reopen an expanded node
    f_cand    = combine(cand_g, scan(cand_g, lookup_h))      # f = g + h on the changed candidates only
    f_open    = update(f_open, f_cand)                       # min-merge into the open list
Bellman-Ford variant on the same grid (nested trace, reported in `extra` only):
    rows      = scan(Δ, gather_row)                          # the frontier's edge rows (index access)
    cand      = combine(Δ(n), rows(n, m), keep=(m,))
    X, Δ      = update(X, rename(cand, m -> n));  fixpoint until Δ = ∅

## 4. Hotspot event and work efficiency
Event: one edge relaxation = one free neighbour scanned from an expanded node (golden) = one
matching (popped node, grid edge) pair in the expansion combine (algebra). native_ops = golden
relaxations. Measured (harness, 2026-09-25):

| scale/seed | component cells | golden expansions | golden relax. | algebra expansions | algebra relax. | Σ open (select cand.) | pop cand. (select + delete)/relax. | avg / max open | heuristic muls | ratio_mul | ratio_sr | ratio_cand | t_native s | t_algebra s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small/0 | 2460 | 759 | 2283 | 759 | 2283 | 51363 | 44.9 | 68 / 121 | 1206 | 1.860 | 2.004 | 49.5 | 0.00 | 0.13 |
| small/1 | 2473 | 2114 | 5954 | 2114 | 5954 | 213910 | 71.8 | 101 / 163 | 3007 | 1.860 | 2.001 | 76.4 | 0.00 | 0.35 |
| small/2 | 2483 | 882 | 2672 | 882 | 2672 | 84105 | 62.9 | 95 / 165 | 1425 | 1.863 | 2.005 | 67.5 | 0.00 | 0.16 |
| large/0 | 42827 | 3438 | 10257 | 3438 | 10257 | 1009158 | 196.7 | 294 / 590 | 5437 | 1.865 | 2.003 | 201.3 | 0.01 | 1.38 |

Path cost is equal on both sides at every run (checked by `compare`). Expansions and relaxations
are identical (ratio 1.000) because both sides order the open list by (f, cell id) and never
reopen a closed node; the tie-breaking rule is the same, so no difference appears.
ratio_mul ≈ 1.87: besides the relaxations, every pop performs one (min,+) mul to recover g from
f, and every candidate that survives the closed mask performs one mul to add h (5437 at
large: the relaxations whose target is not yet closed). ratio_sr ≈ 2 adds the min-merges
of update. ratio_cand is dominated by the pop: select streams the open list and the delete streams
it again, 197× the relaxations at large (avg |open| 294, peak 590); with an O(1)
pop-and-delete the pop cost would be 2 per expansion, 0.67× the relaxations (derived from the
measured expansion count). The grid's open list is a wavefront of O(√cells) cells, so this term
is far smaller than for the random graph of graph_bfs_sssp (197× against 2433× at large/0).

Global (min,+) relaxation instead of SELECT (the plan's section 4.5 counter-example), measured
on the same grids (harness, 2026-09-25):

| scale/seed | A* golden relax. | Bellman-Ford relax. | BF / A* | BF rounds | cells reached by BF | A* expansions / component cells | cost (both) |
|---|---|---|---|---|---|---|---|
| small/0 | 2283 | 6992 | 3.06 | 106 | 2460 | 0.31 | 105 |
| small/1 | 5954 | 6880 | 1.16 | 135 | 2473 | 0.85 | 126 |
| small/2 | 2672 | 7004 | 2.62 | 119 | 2483 | 0.36 | 117 |
| large/0 | 10257 | 120886 | 11.79 | 501 | 42827 | 0.08 | 498 |

Why the global relaxation loses the heuristic's pruning: A* expands a node only when its
f = g + h is the smallest in the open list, and with a consistent heuristic it expands only nodes
with f ≤ C* (the optimal cost); the frontier fixpoint has no notion of f — every improved cell is
relaxed in the next round regardless of how far it is from the goal — so it settles the whole
reachable component (42827 cells at large, of which A* expands 3438). On this unit-cost grid
every cell enters the frontier once (the fixpoint degenerates to BFS), so the loss is exactly the
ratio of reached edges to A*-expanded edges: 1.16-3.06× at small scale (the heuristic
prunes little when the start-goal path must detour around obstacles: seed 1 expands 0.85 of the
component) and 11.8× at large scale. With non-unit weights the fixpoint would also re-relax
cells (graph_bfs_sssp program (a): 1.8-2.4× measured there), multiplying the two losses. Hence the plan's
conclusion: the K (selection) class must exist; the F (frontier) class cannot replace it.

## 5. scan() functions used
`neg_h(key, v) -> [(key, −h(key))]`: looks up the heuristic of the popped cell (Manhattan
formula from the cell id, O(1)); its output is the right operand of the (min,+) mul that recovers
g from f.
`lookup_h(key, v) -> [(key, h(key))]`: the heuristic of a candidate cell, O(1); right operand of
the mul that forms f = g + h.
`gather_row(key, v) -> the ≤ 4 edge tuples of the frontier cell` (Bellman-Ford variant only):
keyed row access into the edge relation, at most 4 outputs per input.

## 6. Control residue
The select loop; the goal test on the popped cell id (one comparison per pop); reading the row
E_row[u] of the popped cell (one dictionary access per pop); the Manhattan formula inside the two
heuristic scans. All O(1) per step.

## 7. Where the algebra is insufficient or awkward
1. f = g + h "as combine of open with a heuristic relation" cannot be applied to the whole open
   list every step: it would cost |open| muls per expansion (Σ|open| = 1009158 at large, the same
   n²-like term as the select) and, in this interpreter, stream the whole heuristic relation
   (43675 free cells) per step. The program therefore adds h only to the candidates changed by an
   expansion (≤ 4 per step) and looks h up by key with a scan: an INDEX access, since the
   interpreter's combine streams the larger operand and the algebra has no keyed lookup.
2. Recovering g from f uses the inverse of ⊗ (g = f + (−h)); this exists in (min,+) but not in a
   general semiring. The alternative is to keep two open relations (g and f) and delete the popped
   node from both, doubling the delete cost.
3. Pop = select + point delete, both streaming the open list (197× the relaxations at
   large); the systolic priority-queue argument of graph_bfs_sssp, section 7, applies unchanged.
4. The frontier fixpoint is not a substitute for the select loop: 11.8× the relaxations at
   large (measured, table above).

## 8. Shape seen by the hardware
Grid edges are implicit (≤ 4 per cell) and read row by row; the edge relation has 121708 tuples
at large (16 + 16 bit keys) but only 10257 are ever read by A*. The open list is a wavefront:
average 294, peak 590 entries at large — a SELECT unit of a few thousand entries holds it.
3438 sequential steps at large, each touching ≤ 4 candidates: the workload is latency-bound
on the pop, not throughput-bound. No multi-query sharing (single start-goal pair). The
Bellman-Ford variant instead runs 501 rounds with frontier sizes in the hundreds and reads every
edge of the component once.
