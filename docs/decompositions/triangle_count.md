# triangle_count — triangle counting by semiring join on a power-law graph

**Category**: graph · **Semiring(s)**: counting · **Execution structure**: sparse join (one shot, no fixpoint)
**Module**: `softwareworkload/triangle_count/workload.py` · **Test**: `tests/test_workload_triangle_count.py`

## 1. Problem and reference algorithm
Undirected graph with a power-law-like degree sequence: the expected degree of the rank-i
vertex is proportional to (i + 1)^(−1/(γ−1)) with γ = 2.5 (Chung-Lu style; endpoints are sampled
proportionally to the expected degrees until m distinct edges exist). n = 3000, m = 15000 small;
n = 20000, m = 120000 large; the largest degree is 584 (small/0) and 2470 (large/0).
Edges are oriented from the lower to the higher (degree, id) rank, so every triangle {a ≺ b ≺ c}
is the wedge a → b → c closed by the edge a → c and is counted exactly once; after orientation
the largest out-degree is only 15 / 27. Computed: the number of triangles.
`native()` is the edge-iterator algorithm: for every oriented edge (i, j) it probes the smaller of
the out-neighbour sets N+(i), N+(j) into the larger (hash intersection) and counts one probe per
adjacency entry examined, min(d+(i), d+(j)) per edge.

## 2. Mathematical decomposition
count = Σ_{i,j,k} E(i, j) ⊗ E(j, k) ⊗ E(i, k) in the counting semiring (⊕, ⊗) = (+, ×), E ∈ {0, 1}.
Two evaluation orders of the same triple join:
materialized: W(i, j, k) = E(i, j) ⊗ E(j, k) (every 2-path), then W ∧ E(i, k), then Σ over all;
fused: W(i, k) = Σ_j E(i, j) ⊗ E(j, k) (2-path counts), then W(i, k) ⊗ E(i, k), then Σ over all.
The golden evaluates the same triple join in the intersection order: for each (i, j) ∈ E,
|N+(i) ∩ N+(j)|.

## 3. Algebra program
    E          : (i, j) -> 1, oriented
    materialized:
      wedges   = combine(E(i, j), rename(E, (j, k)))             # (i,j) x (j,k) -> (i, j, k)
      closed   = mask(wedges, rename(E, (i, k)))                 # semijoin on (i, k)
      count    = reduce(closed, keep=())                         # Σ over every attribute
    fused:
      W        = combine(E(i, j), rename(E, (j, k)), keep=(i, k))   # wedge counts per (i, k), reduce fused
      T        = combine(W, rename(E, (i, k)))                   # elementwise: count × 1 on closed pairs
      count    = reduce(T, keep=())
Both forms run on the same instance; `algebra()` returns both counts and `compare` checks both
against the golden.

## 4. Hotspot event and work efficiency
Event: one candidate third vertex examined — an adjacency entry probed by the golden, a wedge
(i, j, k) produced by the join (one counting mul). The trace holds both forms, so native_ops =
2 × golden probes and the harness ratio is the mean of the two per-form ratios. Measured
(harness, 2026-09-25):

| scale/seed | golden probes | triangles (golden = both forms) | oriented wedges | closed wedges | mat. mul / probes | mat. cand. / probes | fused mul / probes | fused cand. / probes | fused (i,k) pairs | ratio_mul | ratio_sr | ratio_cand | t_native s | t_algebra s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small/0 | 73661 | 7525 | 100571 | 7525 | 1.365 | 2.833 | 1.407 | 1.449 | 74192 | 1.386 | 1.637 | 2.141 | 0.01 | 0.26 |
| small/1 | 73340 | 7125 | 99608 | 7125 | 1.358 | 2.813 | 1.400 | 1.442 | 74089 | 1.379 | 1.622 | 2.128 | 0.01 | 0.25 |
| small/2 | 73462 | 7256 | 101483 | 7256 | 1.381 | 2.862 | 1.424 | 1.466 | 75670 | 1.403 | 1.649 | 2.164 | 0.01 | 0.26 |
| large/0 | 786114 | 55097 | 1261490 | 55097 | 1.605 | 3.280 | 1.628 | 1.651 | 961834 | 1.616 | 1.853 | 2.465 | 0.10 | 3.81 |

The join enumerates every oriented wedge, Σ_j d−(j)·d+(j) = 1261490 at large, while the golden
probes Σ_(i,j)∈E min(d+(i), d+(j)) = 786114 entries: the wedge / probe ratio is 1.36-1.38× at
small scale and 1.60× at large. The fused form performs the same muls plus the mul of W with E
(1279662 against 1261490 at large) but roughly halves the candidates: it never materializes the
1261490 wedge tuples that the mask form must probe again (mat. cand. / probes 3.28 against
1.65 fused). ratio_sr adds the merges: 317827 adds in the fused form (2-paths merged onto the same
(i, k) pair, plus the final reduce) and the 55096 adds of the final counting reduce in the
materialized form.
Total triangles: 55097 at large/0 and 7525 / 7125 / 7256 at small/0-2, identical on both
sides and for both forms.

## 5. scan() functions used
none.

## 6. Control residue
none beyond reading the single count out of the final relation. The degree-rank orientation is
instance preprocessing shared by both sides.

## 7. Where the algebra is insufficient or awkward
1. Join order versus worst-case-optimal join. The binary combine produces every 2-path before the
   closing edge is checked, so its work is the wedge count Σ_j d−(j)·d+(j); the intersection
   algorithm checks, per edge, only min(d+(i), d+(j)) candidates. This is the classical gap
   between a pairwise join plan and a worst-case-optimal (intersection-driven, Leapfrog / Generic
   Join style) evaluation of the triangle query: matching the golden would need a combine that
   intersects the two adjacency rows on the third attribute instead of enumerating one row and
   probing the other — a multiway or semijoin-first join primitive, not one of the six operators.
   Measured gap after the degree orientation: 1.36-1.60×, within the plan's admission line.
2. The orientation is what keeps the gap small on a power-law graph. Without it the wedge count
   would be Σ_j C(d(j), 2) = 714400 (small/0) and 12976106 (large/0), derived from the degree
   sequence, 10.3× the oriented wedges at large: the hubs of degree 2470 dominate. The
   orientation is a preprocessing step outside the algebra (a sort by degree and a rename of
   each edge), not expressible with the six operators; both sides share it.
3. The materialized form's intermediate is the full wedge relation (1261490 tuples of
   3 × 15 bits at large) — output growth that a hardware pipeline should avoid by fusing the
   closing semijoin into the join (the fused form, or a mask-inside-combine).

## 8. Shape seen by the hardware
Sparse Boolean-valued counting relation E with m tuples (15 + 15 bit keys at large), used three
times in one join (as (i, j), (j, k) and (i, k)); build side E (m tuples) and probe side E; no
fixpoint, no multi-query. Output growth: 1261490 wedge tuples (materialized) or 961834 (i, k)
pairs (fused) at large, against 55097 triangles. The dominant operator is the wedge combine;
the closing mask / elementwise combine and the final reduce are one pass over its output.
