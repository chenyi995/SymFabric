# Session record — 2026-09-25 — exploration of the SymFabric direction

Account: chenyi9. Agent: claude. No commit or push was requested in this session; this record is
written so that the work is documented if the user decides to commit it (rules section 28).

## What was built

- Repository skeleton: `git init`, remote `origin` set to git@github.com:chenyi995/SymFabric.git
  (remote empty as of 2026-09-25), `aimd/` rules cloned and git-ignored, `CLAUDE.md`, `.gitignore`,
  `README.md`, `local/` for Chinese drafts.
- `ref/notes/`: eight one-page notes of the papers under `ref/` (REASON, NeSy tutorial, CoCoSys,
  KVChime, AMMA, Tessera, piPE-SA, agentic workflows), every reused number with its location.
- `algebra/`: semirings, weighted relations, seven operators and two control structures, with a
  work-counting trace (`algebra/trace.py`); 24 core tests.
- `softwareworkload/`: 18 workloads, each with an independent golden, an algebra program, a test
  file and a decomposition document under `docs/decompositions/`; harness `_harness.py`.
- `systolicarch/`: analytical model of a reconfigurable semiring systolic array (dense GEMM,
  merge-join, probe/update, systolic priority queue, stream modes); six configurations; 7 tests.
- `pim/`: bank-level PIM model (multi-query scan, fused fixpoint round, bank stream/probe),
  multi-stack (HBM-to-HBM mesh) extension, placement selector; five configurations; 5 tests.
- `eval/`: `run_arch.py`, `run_sensitivity.py`, `make_coverage.py`, `analyze_bottleneck.py`,
  `plot_results.py`, `scale_traces.py`.
- `profiling/`: GPU/CPU profiling kit (43 files) with run instructions; nothing executed here.
- `docs/`: `algebra.md`, `arch-systolic.md`, `arch-pim.md`, `coverage.md` (generated),
  `findings.md`, `decompositions/*.md`.

## Decisions (agent, claude)

- Repository layout keeps the user's folders (`plan/`, `ref/`, `softwareworkload/`, `systolicarch/`,
  `pim/`) and adds `algebra/`, `docs/`, `eval/`, `tests/`, `profiling/`, `local/`.
- Reference PDFs are git-ignored; `ref/README.md` keeps their SHA-256 prefixes.
- Work-efficiency convention: one hotspot event of the reference algorithm = one semiring
  multiplication (`ratio_mul`); merges and all touched tuples reported separately.
- Architecture constants: every value carries a `source:` comment; three parameters had no source
  (`random_access_bytes`, `op_launch_cycles`, `fabric_bw_Bps`/`global_reduce_s`) and are marked
  `TBD[src]` and swept by `eval/run_sensitivity.py`.
- The 28 nm clock of REASON is not stated in the paper; 500 MHz (its scaled-node footnote) is used
  and flagged.
- `reduce(use_mul=True)` kept but documented as unsound under zero suppression (found by the
  prob_circuit workload).
- Two analytical-model corrections made after the first full run: a persistent priority queue
  prices `select` incrementally and the following pop-delete as one pop; PIM command groups pay a
  200 ns round trip (AMMA Sec. 4.2). The first-run directory `eval/results/20260925-201800-full-large`
  predates both corrections and is kept for comparison.
- "All-PE" configurations (`edge28_allpe`, `dc_hbm3_allpe`) added to isolate the value of
  reconfigurable PEs (every PE usable as a merge/probe lane).

## Checks run

- `python3 -m pytest -q tests/`: 129 passed (2026-09-25, ~5 min).
- `python3 -m softwareworkload._harness --scale small|large`: 18/18 workloads agree with goldens.
- Results directories (all under `eval/results/`, never overwritten):
  `20260925-194839-probe-large` (2 workloads, first model), `20260925-201801-full-small`,
  `20260925-201800-full-large` (pre-correction models), `20260925-202254-final-small`,
  `20260925-202252-final-large` (with traces), `20260925-202255-hmmbatched-batched`,
  `20260925-202446-allpe-small`, `20260925-202804-allpe-large`,
  `20260925-203019-final6-large` (six configurations; source of `docs/coverage.md`),
  `20260925-203200-final6-small`, `20260925-203156-scaled-20260925-202252-final-large` (derived
  working-set extrapolation), `20260925-203158-sens-large` (sensitivity of the TBD[src] parameters).

## Archived

Nothing archived: no file was superseded during the session (rules section 22).

## Missing evidence and open items

- No synthesis of a semiring PE, a comparator/merge cell, or a symbolic bank PE (plan's E4).
- No cycle-level simulation; all architecture numbers are first-order model outputs.
- Instance sizes are bounded by the Python interpreter (largest operand about 25 MB); the
  datacenter-scale memory-boundness is shown by scaled traces (derived), not measured.
- GPU profiling kit untested on a GPU host; `ncu` metric names marked "verify".
- `graph_bfs_sssp` large takes 52 s (over the 40 s guideline); `fuzzy_inference` and
  `automata_rpq` large sizes were reduced by their authors to fit, with the specified sizes kept as
  `spec_large` scales.
