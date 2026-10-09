# bayes_net_ve — variable elimination for a marginal query in a discrete Bayesian network

**Category**: probabilistic · **Semiring(s)**: prob (+,×), logprob (LSE,+) · **Execution structure**: dag
**Module**: `softwareworkload/bayes_net_ve/workload.py` · **Test**: `tests/test_workload_bayes_net_ve.py`

## 1. Problem and reference algorithm
A Bayesian network over n variables with domain sizes 2 or 3; every variable draws 1–3 parents
(P(second parent) = 0.9, P(third) = 0.8) from a window of the 10 (small) / 12 (large) preceding
variables, so the moral graph has bounded bandwidth; CPTs are Dirichlet(1). Query: P(X_q | e)
for one variable q with hard evidence on 3 (small) / 5 (large) other variables. Small n = 25,
large n = 60; a tiny scale (n = 12) exists for the brute-force cross-check. An instance whose
min-fill elimination would create a factor product above 2^14 entries is regenerated (one
regeneration at large seed 2, none otherwise).
`native()` is textbook VE over dense Python-dict factor tables: restrict to the evidence,
eliminate in min-fill order (multiply every factor containing v pairwise, sum v out), multiply the
remaining factors, normalise; the same in the log domain. A second golden, `brute_force()`, builds
the full joint by numpy broadcasting; the two goldens agree at the tiny scale (seeds 0–2) to a
maximum absolute difference of 1.1e-16 (measured 2026-09-25). The small-scale joint has up to
3^25 entries and cannot be enumerated.
Hotspot: one multiplication per output entry of a pairwise factor product.

## 2. Mathematical decomposition
P(x_q | e) ∝ Σ_{x ∖ {q} ∖ e} ∏_i CPT_i(x_i | x_pa(i)) with the evidence variables fixed.
VE step for v:  F_new(rest) = ⊕_{x_v} ⊗_{F ∋ x_v} F(x_v, rest_F) — a join of all factors that
mention v, fused with a reduce over v. The elimination order fixes a tree of joins (the
elimination tree); no fixpoint and no iteration.

## 3. Algebra program
    F_i(x_pa(i), x_i) = Relation.dense(CPT_i)                                  # n factors
    F_i = filter(F_i, x_e == e)     for every evidence attribute in F_i         # attribute kept, one value
    for v in min_fill_order:                                                    # CONTROL RESIDUE
        P     = combine(F_1, F_2); P = combine(P, F_3); ...                     # natural joins
        F_new = combine(P, F_last, keep = union of the schemas minus x_v)       # fused ⊕ over x_v
        (a single involved factor: F_new = reduce(F_1, keep))
    M(x_q) = combine(...) over the remaining factors; reduce(keep=(x_q,))       # drops evidence attributes
    Z      = reduce(M, keep=())
    answer = scan(M, v -> v / Z)                                                # LOGPROB: v - log Z
Run in PROB and LOGPROB (log CPTs); exp(log answer) is checked against the PROB answer (≤ 1e-8).

## 4. Hotspot event and work efficiency
Native product entries over the two semirings: 41,196 (small seed 0; seeds 1, 2: 8,736 and
23,844), 102,982 (large seed 0; seeds 1, 2: 134,886 and 83,866).
Measured (harness, small, seed 0, 2026-09-25): ratio_mul 1.000, ratio_sr 1.588, ratio_cand 1.049;
seeds 1 and 2: ratio_mul 1.000, ratio_sr 1.525 and 1.542, ratio_cand 1.117 and 1.042. Large
(seed 0): ratio_mul 1.000, ratio_sr 1.653, ratio_cand 1.116; seeds 1 and 2: ratio_mul 1.000,
ratio_sr 1.622 and 1.584, ratio_cand 1.135 and 1.070.
ratio_mul = 1 exactly: both sides multiply the same factors in the same order, and every combine
candidate is one entry of the dense product because the factors have full support (Dirichlet
CPTs have no zeros; evidence only restricts the domain). ratio_sr above 1 is the sum-out
additions (entries minus groups per elimination). ratio_cand above 1: the evidence filters (|F|
candidates each; 14 filter records at small seed 0, 46 at large), the reduce passes when a single
factor contains the eliminated variable, the final reduce, and the two normalising scans.
Largest factor product: 4,374 entries (small seed 0; 864 and 3,888 for seeds 1, 2), 5,832 (large
seed 0; 10,368 and 2,916 for seeds 1, 2); all within the 2^14 = 16,384 bound by construction.
Timing (harness, seed 0): algebra 0.067 s small, 0.177 s large; native 0.065 s / 0.161 s.

## 5. scan() functions used
`div_Z` (PROB): v -> v / Z; `sub_logZ` (LOGPROB): v -> v − log Z, with Z the scalar produced by
reduce(M, keep=()). O(1) per tuple; card(q) ≤ 3 tuples.

## 6. Control residue
Min-fill ordering in plain Python, shared by both sides: 4,827 / 4,615 / 5,435 neighbour-pair
checks at the small scale (seeds 0–2) and 36,111 at large seed 0 — 12% (small seed 0) and 35%
(large seed 0) of the native multiplication count, so it is not negligible when the factors are
this small. Also: selecting the factors that contain the eliminated variable (schema membership
tests, O(#factors) per step) and assembling the `keep` list.

## 7. Where the algebra is insufficient or awkward
1. The elimination order is a graph heuristic (greedy min-fill with fill counting), a select-loop
   over an interaction graph that the algebra has no operator for. Every VE engine has this
   phase before the numeric one; its cost is reported above.
2. Normalisation is a division, outside the semiring; it is expressed as a scan with a captured
   scalar (sec. 5).
3. Evidence handling costs one pass over every factor that mentions an evidence variable; an
   indexed filter would touch only the matching tuples, which the algebra does not model.
4. Nothing is missing for the numeric phase: VE is combine/reduce by construction and the work
   matches the reference multiplication for multiplication.

## 8. Shape seen by the hardware
Dense factors with 2^a·3^b entries; 48 (small) / 118 (large) combine records, the largest with
4,374 (small seed 0) to 10,368 (large seed 1) candidates, most joins sharing 1–3 attributes. The
elimination tree is a DAG of small dense tensor contractions whose shapes change at every step —
a poor fit for a fixed-shape array unless factors are padded to a common width. No multi-query
sharing (one query); two semirings on identical dataflow.
