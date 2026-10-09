# REASON DPrL project delivery

Account: chenyi9. Date: 2026-10-08 in America/Los_Angeles.

## Authorized work and decisions

chenyi9 requested a separate project directory for the REASON and DPrL
research direction, a clear GPU-versus-REASON comparison, shared simulator
checkouts exposed through symlinks, and a commit and push of the repository.

Codex placed the project in `reason_dprl/`, reused the existing shared upstream
stores, and added a pinned DPrL checkout to the software store. Relative symlinks
provide project entry points; a lock and source helper restore the same
revisions on another checkout. The proposed dynamic mechanisms remain research
hypotheses. REASON's existing SAT feedback and conflict handling are preserved
in the baseline description.

The repository and remote had no commits before this delivery. The existing
research source tree was imported as a separate initial commit. The project
source wiring and its comparison documents are separate subsequent commits.

## Commits covered by this delivery

| Commit | Change |
|---|---|
| `e1efe2039e6cea2027c049759974d2360edf62b3` | Import the existing SymFabric research workspace |
| `bf1e69529bbe2c37bc9d0a491729d26fb83c2cbf` | Add pinned shared sources and simulator links for REASON DPrL |
| `4641ec331b75a42f7f7024bbdb1da6bcd9cd3b60` | Document the REASON DPrL project and comparison protocol |

This session record is committed separately as the final commit of the same
delivery, before pushing the branch to origin.

## Delivered content

- `reason_dprl/README.md`: research scope, workloads, baseline matrix, source
  entry points, setup commands, and current execution prerequisites.
- `reason_dprl/COMPARISON.md`: matched methods and platforms, correctness,
  trajectory replay versus closed-loop execution, profiling, simulation
  inputs, complete-query accounting, and attribution requirements.
- `reason_dprl/scripts/sources.py` and `sources.lock.json`: source recording,
  missing-checkout restoration, revision verification, and non-destructive
  symlink setup. The lock is generated from local Git metadata and read back
  for verification.
- Relative links to DPrL and the available GPU, DAG, sparse, systolic, memory,
  and FPGA-reference source trees. Source presence is distinguished from a
  built or calibrated simulation backend.

Existing ignored PDFs, third-party checkouts, generated results, local Chinese
drafts, and model/data directories retain their existing storage policy. No
existing source was archived or removed.

## Validation performed

- PASS: full existing regression with
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q tests/`.
  Test output was captured in the session execution transcript.
- PASS: `python3 sym1/scripts/check_evidence_docs.py`.
- PASS: source revision recording and read-back, link setup, and
  `python3 reason_dprl/scripts/sources.py check`.
- PASS: isolated local-Git fixtures exercised pinned revision restoration,
  nested dependency restoration, idempotent linking, whole-workspace relocation,
  revision mismatch rejection, and preservation of a conflicting ordinary file.
  The fixture remains under `/tmp/symfabric-source-wiring-dqy4k16s/`.
- PASS: new project Python syntax and relative Markdown link destinations.
- PASS: new project staged whitespace checks and Git symlink modes; the
  committed source inventory excludes downloaded upstream trees and PDFs.

The imported historical documents contain pre-existing trailing-whitespace
warnings in `sym1/README.md` and an extra ending blank line in
`sym1/baseline_report.md`; the import preserved their contents. The Chinese
string literals in the existing bilingual evidence checker are matching keys
for local documents, not new repository prose, and were preserved with the
validated checker.

## Remaining evidence

An actual REASON-specific compiler and simulator has not been located in this
checkout. DPU-v2 and generic Accel-Sim are labelled as supporting tools rather
than a completed REASON implementation. The current host's NVIDIA driver query
fails, so measured DPrL GPU profiles still require a working GPU environment.
Datasets, checkpoints, workload lowering, integrated timing validation, and
accelerator performance measurements remain execution prerequisites. This
delivery establishes the project and source wiring; it reports no accelerator
speedup.
