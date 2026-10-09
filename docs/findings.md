# Findings (exploration, 2026-09-25)

Every number below is labelled **measured** (harness or model run, with the results directory),
**derived** (formula from sourced constants), or **cited** (paper and location). Model numbers are
outputs of first-order analytical models (`systolicarch/`, `pim/`), not silicon or cycle-accurate
simulation. Instance sizes are bounded by the pure-Python reference interpreter (the largest
operands are 1–25 MB), which matters for section 4.

## 1. Starting point (cited)

- Symbolic/probabilistic kernels take 34.8–63.9 % of end-to-end time in REASON's six workloads
  (REASON, Sec. III-A, p. 4) and are memory-bound on GPUs: DRAM bandwidth utilisation 57–70 %,
  ALU utilisation 29–53 % (REASON, Table II, p. 5).
- A TPU-like fixed-MAC systolic array is 74.7–109.2x slower than REASON's tree engine on the
  symbolic-only kernels (REASON, Fig. 13, p. 13).
- KVChime's bank-level multi-query PIM keeps 8 query operands per bank and moves the PIM-vs-GPU
  crossover from 54–55 to 512–545 concurrent queries (KVChime, Sec. 5.5, Fig. 8); its 16-lane
  FP16 bank datapath is 6,205–7,343 um2 at 0.666–1.3 GHz in ASAP7 (KV-PIM RTL components table).
- The NeSy tutorial and CoCoSys only profile VSA and soft-logic models in detail; search, SAT,
  Datalog, KG, planning and PC kernels are named but not measured there (notes in `ref/notes/`).

## 2. Does one small algebra cover the workloads? (measured, harness)

Eighteen workloads across logic, probabilistic, graph, search, planning, VSA, automata, fuzzy and
parsing were written twice (independent golden + algebra program). All eighteen agree with their
goldens at every scale and seed tested (131 tests, `python3 -m pytest -q tests/`; 131 passed on 2026-09-26). The full table
with operator counts and ratios is generated in `docs/coverage.md`; the work-efficiency picture
(ratio_mul = semiring multiplications / hotspot events of the reference algorithm, large scale,
seed 0):

| class | workloads | ratio_mul | what the excess is |
|---|---|---|---|
| exact or near-exact (≤ 1.2) | datalog_tc 1.00, datalog_samegen 1.00, kg_multihop 1.00, bayes_net_ve 1.00, beam_search 1.00, hmm_viterbi 1.02, ctrlg_dfa_hmm 1.06, cky (rule-first) 1.07, prob_circuit 1.10, vsa_cleanup 1.12, automata_rpq (product-first) 1.13, BFS 1.00 | 1.00–1.13 | unit-weight gathers written as joins; emission joins; layout |
| join-order dependent | automata_rpq (nfa-first) 2.0 / (graph-first) 3.5; cky (pair-first) 6.0; triangle_count 1.6 | 1.6–6.0 | the algebra states the join but not its order; wedge join vs set intersection |
| round-synchronous fixpoint vs priority-ordered algorithm | graph_bfs_sssp frontier 2.4 (vs Dijkstra), fuzzy_inference 4.8 (5.7 at n=2000), planning_hadd 4.0 (6.5 on deep tasks) | 2.4–6.5 | keys re-derived at better values; needs a priority-ordered or level-ordered schedule |
| exact hotspot but the queue is the cost | astar_grid 1.9 (expansions identical to heapq A*), Dijkstra-with-select 1.0 | — | select over the whole open list costs 45–2,433x the relaxations in the interpreter; a persistent priority queue makes it O(1) per pop (section 3) |
| boundary cases | sat_bcp watched form 4.8 (ratio_cand 36); sat_bcp scan form 32–54; datalog_nonlinear_tc 31–88 (≈ 0.3 n) | 5–88 | the clean mathematical form re-scans all clauses / re-derives Θ(n)x; only the incremental (watched, linear-delta) programs keep per-round work proportional to the reference |

Three conclusions:

1. **The value algebra is not the problem.** Nine semirings covered everything; h^max needed two
   operations (max-fold, +-compose) and is not a semiring swap (planning_hadd §7).
2. **Indexing, selection and fusion are.** Every excess above 1.2 traces back to a missing
   structural capability, listed with its measured price in `docs/algebra.md` §4: keyed gather,
   grouped select, a persistent priority queue with delete, early-exit gather, fused
   combine→antijoin→update, join-order planning, affine key offsets.
3. **The plan's section 4.5 warnings are confirmed and quantified**: BCP as a full scan is
   32–54x, Bellman-Ford instead of A* on a grid is 11.8x, non-linear closure is Θ(n)x.

## 3. Where does the time go on a reconfigurable array? (measured, model)

Source: `eval/results/20260925-203019-final6-large` (18 workloads, large scale, seed 0; six
configurations; `bottleneck.md` and `plots/` in that directory; `docs/coverage.md` table 3).

**3.1 A fixed-MAC array is not a symbolic accelerator.** Executing the same traces with dense
semiring GEMMs over the full index domains (`tpu_like`, 128x128, classical fill/drain) costs, relative
to the reconfigurable `dc_hbm3` array of the same size and memory: 30x (datalog_tc), 8.6 thousand x
(datalog_samegen), 11 thousand x (sat_bcp), 135 thousand x (automata_rpq), 0.37 million x (SSSP),
1.3 million x (triangle counting), 1.5 million x (KG multi-hop) and 2.8 million x (A*). Only the genuinely dense or nearly dense kernels stay within
a small factor (HMM 1x, prob_circuit 3x, CKY 4x, fuzzy closure 8x) and the 90 %-dense non-linear
closure is actually faster dense. REASON's 75–109x (Fig. 13) is the mild end of this range because
its DAGs are small; the plan's sparse, memory-resident workloads are the severe end.

**3.2 With one merge/probe lane per column, the array is idle.** In the base configurations the
dense mode carries 0.3 % (edge28) and 4.6 % (dc_hbm3) of all modelled cycles; merge/join lanes carry
24 % / 53 %, probe-and-update of state tables 53 % / 19 %, the priority queue 22 % / 23 %. Median PE
utilisation (multiplications per PE-cycle) is 0.006 (edge28) and 0.002 (dc_hbm3); the maximum is
0.06 / 0.016. The PEs exist for the 0.3–4.6 %.

**3.3 Reconfigurable PEs recover the array.** Letting every PE act as a merge / probe / stream lane
(`*_allpe`; Kung-Lehman systolic relational cells; no other change) speeds up the compute-bound
workloads by 5–122x on `dc_hbm3` (CKY 85x, fuzzy 122x, samegen 83x, rpq 88x, triangle 29x, VSA 24x,
sat_bcp 21x, planning 18x, non-linear TC 26x, ctrl-g 9x, prob_circuit 6.5x) and 2.6–31x on `edge28`;
median PE utilisation rises to 0.03 (max 0.59–0.68). Two families do not move: priority-queue-bound
search (SSSP 1.2x, KG multi-hop 1.0x, beam 1.5x on `dc_hbm3` — a pop is one element per cycle
whatever the lane count) and the already memory-bound edge workloads (datalog_tc, rpq, fuzzy,
triangle: 1.0–1.1x).

**3.4 After that, the chip is memory- or queue-bound.** On `dc_hbm3_allpe` the share of
memory-bound cycles is 0.96 (datalog_tc), 0.83 (triangle), 0.80 (sat_bcp), 0.77 (planning), 0.72
(VSA), 0.27–0.33 (samegen, rpq, ctrl-g), and the priority queue becomes the largest family at 63 %
of all cycles (probe/update 17 %, dense 13 %, merge 7 %). In other words, once the lanes are there,
a symbolic accelerator is limited by (a) bandwidth to state tables that do not fit on chip and (b)
the serial pop rate of selection — the two things the plan reserves for the memory side and for the
SELECT unit.

**3.5 Batching is what makes the dense mode worth having.** For the HMM the dense GEMM per step has
M = 1 row; on a 32x32 array the step costs 191 cycles (about 6D) at 1.2 % utilisation and on 128x128 it costs
383 cycles, slower than on 32x32. With Q = 64 sequences sharing the transition matrix (`hmm_viterbi` scale
`batched`, `eval/results/20260925-202255-hmmbatched-batched`) utilisation rises to 8.2 % (edge28)
and the 128x128 array finally beats the 32x32 one (0.068 vs 0.409 ms). Fracturable strips
(Tessera) are what let a 48-state model use a 128-wide array at all.

## 4. When is memory the bottleneck, and does PIM help? (measured and derived, model)

**4.1 Measured at the interpreter's scale.** On the edge platform (1.25 MB SRAM, 104 GB/s LPDDR5;
`edge28`) the four closure-type fixpoints are memory-bound at the "large" instances (share of
memory-bound cycles: datalog_tc 0.80, automata_rpq 0.81, fuzzy closure 0.77, triangle counting
1.00; sat_bcp 0.54 and VSA 0.23 mixed), and the dominant operators are the probe of the state
table (`mask`, partitioned pass, 35–54 %) and its update (`update`, random read-modify-write or
stream merge, 29–53 %). On the datacenter platform (24 MiB buffer, 1.2 TB/s HBM3E) every "large"
instance fits on chip and nothing is memory-bound (datalog_tc 0.05, all others 0.00) — with one lane
per column. With reconfigurable PEs (`dc_hbm3_allpe`, section 3.4) the same instances become
memory-bound because the compute side is 20–120x faster and the inputs still have to be read once.

**4.2 Memory-side execution, apples to apples (measured, model; `pim/`).** Edge chip + LPDDR5-PIM
(128 banks, 16-lane bank PEs at 0.666 GHz, 8 query slots): the placement selector moves the fused
fixpoint rounds (combine → antijoin → update) into the banks and the modelled time drops 7.6x
(datalog_tc), 10.7x (automata_rpq), 1.7–2.0x (fuzzy), 1.6x (triangle), 1.8x (sat_bcp, multi-query
clause scan), 1.4–1.8x (beam, samegen); everything else stays on the chip (1.0x). Datacenter chip
+ HBM3-PIM (512 banks): 2.4x (datalog_tc), 1.5x (rpq), 1.2–1.3x (sat_bcp, samegen), 1.0x otherwise;
against the all-PE chip only datalog_tc 1.19x, because that chip is already bandwidth-bound and the
PIM shuffle over the buffer die is assumed to run at the same bandwidth (TBD[src]).

**4.3 Derived: scaling the working set** (`eval/scale_traces.py`,
`eval/results/20260925-203156-scaled-20260925-202252-final-large`; every count in a trace multiplied
by 4–256, algorithmic shape unchanged — an extrapolation, not a measurement). Once the largest
operand exceeds the 24 MiB buffer (factor 4: 96 MB for datalog_tc, 28 MB for rpq, 71 MB for
triangle; fuzzy reaches 72 MB only at factor 16) `dc_hbm3` becomes memory-bound (share of
memory-bound cycles 0.69–0.75 datalog_tc, 0.66–0.71 rpq, 0.77–1.00 triangle, 0.69–0.70 fuzzy from factor 16;
fuzzy at factor 4 is 18 MB and 0.30) and stays there up to 6 GB;
HBM3-PIM then gives a stable 5.3–5.7x (datalog_tc), 4.9–6.5x (rpq), 1.4x (fuzzy from factor 16, triangle from factor 64; at most 1.01x below),
1.2–1.4x (sat_bcp). Against `dc_hbm3_allpe` the gains are 1.3x (datalog_tc, sat_bcp) and 2.1–3.4x
(rpq). Workloads that never move: kg_multihop (priority-queue-bound), vsa_cleanup and prob_circuit
(dense GEMM), CKY, A* (small operands). So the memory side pays for the class the plan calls
"multi-query x large shared knowledge structure with a fixpoint" — and the size of the gain is set
by two unsourced parameters (buffer-die shuffle bandwidth, bank-PE lanes/frequency), see 4.5.

**4.4 HBM-to-HBM.** With 16 stacks in a 2-D UCIe mesh (AMMA's geometry: 15 ns/hop, 1.5 TB/s per
link) the fused round scales with the bank count (16x rows and PE lanes) and pays (S−1)/S of its
shuffle over the mesh plus the diameter latency per round: datalog_tc 23x (dc_hbm3) / 6.4x
(dc_hbm3_allpe) at the measured scale, rpq 2.1x — but this compares one chip against sixteen
stacks of PIM logic; it says that the shuffle, not the link, is the limiter (the 6 hops of latency
per round are negligible against the round time), consistent with AMMA's own finding that latency
is insensitive to D2D bandwidth (AMMA Sec. 7.6).

**4.5 Sensitivity (TBD[src] parameters; `eval/results/20260925-203158-sens-large`).** Each
unsourced parameter was swept on the large traces; the table gives the spread (max/min of the
modelled time over the sweep) — median over the 18 workloads and the worst workload.

| side | parameter (sweep) | median spread | worst | what it says |
|---|---|---|---|---|
| chip | `op_launch_cycles` 0 / 32 / 256 | 1.8x (dc_hbm3), 13.2x (dc_hbm3_allpe) | 235x (A*), also SSSP, beam, KG | select-loop workloads issue thousands of tiny operators; any per-operator control cost dominates them. A statically scheduled (LPU-like) pipeline with no per-operator launch is a requirement, not a nicety. |
| chip | `sram_bytes` x1 / x4 / x16 (edge28) | 1.0x | 4.3x (rpq), 1.0x elsewhere | on-chip capacity flips the closure workloads between memory- and compute-bound |
| chip | `freq_hz` 500 MHz–1 GHz | 1.4–1.9x | 2.0x | proportional, as expected |
| chip | `random_access_bytes` 32 / 64 / 128 | 1.0x | 1.09x (fuzzy) | the partitioned-pass alternative caps the damage of random access |
| PIM | `fabric_bw_Bps` 0.5x / 1x / 4x of the interface | 1.0x | 3.4x (datalog_tc, both platforms) | the buffer-die shuffle bandwidth sets the PIM gain on join-heavy fixpoints; it is the first constant to source (NDPBridge-style measurements) |
| PIM | `global_reduce_s` 0.1 / 1 / 10 us | 1.03x | 5.2x (rpq, 44 rounds, edge) | round-count-heavy fixpoints need a fast termination flag |
| PIM | `cmd_latency_s` 0.1 / 0.2 / 1 us | 1.0x | 2.1x (beam, small groups) | matters only for tiny groups |
| PIM | bank PE `pe_lanes` 16–64, `pe_freq_hz` 0.666–1.3 GHz, `query_slots` 8–128 | 1.0x | 1.2x (sat_bcp, beam) | the bank PE is not the limiter for these traces; the row stream and the shuffle are |

So the qualitative conclusions of 4.1–4.3 (which workloads are memory-bound, which move to the
memory side) hold across the sweeps; the magnitude of the PIM gain on the closure fixpoints is
uncertain by about 3x until the buffer-die shuffle is sourced or measured.

## 5. What did not fit, and what a v2 algebra needs

- **Priority-ordered work** (Dijkstra, A*, delta-stepping, threshold decomposition of fuzzy/min-plus
  closures): the algebra expresses it only as select-loops of tiny rounds; the systolic priority
  queue turns each pop into O(1) but the loop remains serial per pop (18–95 % of modelled cycles in
  sssp / kg_multihop at large scale, first run).
- **Search control** (CDCL conflict analysis, backjumping, clause deletion) was not attempted;
  sat_bcp covers propagation only, as the plan intended.
- **Reduce-with-mul under zero suppression** is unsound (prob_circuit §7); product nodes must be
  binary joins or complete-group folds.
- **Numeric change detection** in `update` (relative 1e-9) drops small contributions for
  non-idempotent semirings (cky §7); v2 needs an exact-merge mode.
- **Instance scale**: the interpreter caps operands at a few tens of MB, so the memory-boundness
  of the datacenter configuration is shown by derived scaling (section 4), not by measurement.
