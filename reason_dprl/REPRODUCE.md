# Start here: reproduce and extend the DPrL investigation

Run commands from the **SymFabric repository root**. This is the handoff entry
point for the 2026-10-09 work. Read [proposal/README.md](proposal/README.md) for
the corrected research question and [proposal/PRIOR_ART.md](proposal/PRIOR_ART.md)
for overlap with CogSys/CoCoSys. The literature's broad 36-paper list is not a
count of DPrL-equivalent dynamic workloads.

For the subsequent binding/state hypothesis tests and the faster GPU adapter,
read [proposal/DYNAMIC_REPRODUCE.md](proposal/DYNAMIC_REPRODUCE.md). That guide
adds the exact descriptor-stage experiment, CTP/DPLA* probes, and a separate
artifact bundle; it does not replace the frozen baseline bundle below.

## Same-server quick path

The login host has no usable GPU. Submit to Slurm `athena/node2`; the measured
platform is an RTX A5000 (24 GB), driver 580, PyTorch 2.9.0+cu130, Python 3.12.13.
The following commands verify existing source, input, checkpoint and package
identities without importing CUDA on the login host:

```bash
python3 reason_dprl/scripts/reproduce.py check

# New directory required. This submits Family and WN18RR, one GPU job each.
python3 reason_dprl/scripts/reproduce.py submit \
  --output reason_dprl/results/NEW-smoke --smoke
```

The submitter prints job IDs. Inspect `submission.json` and
`family.slurm.log` / `wn18rr.slurm.log` in the chosen directory. Use `squeue -u
"$USER"` for running jobs and `sacct -j JOB_IDS --format=JobID,State,ExitCode,Elapsed`
for completion. Require `COMPLETED`, exit `0:0`, and `OPTIMIZATION_COMPLETE` in
both logs. A present `benchmark.json` alone can be an incomplete run.

```bash
# After both jobs complete: checks queries, checkpoint, output hashes, in-run
# correctness audits, all five variants, and timing sample counts.
python3 reason_dprl/scripts/reproduce.py compare \
  reason_dprl/results/NEW-smoke --smoke

# Full experiment: 3 disjoint batches of 64, 20 repetitions each, 20 proof steps.
python3 reason_dprl/scripts/reproduce.py submit \
  --output reason_dprl/results/NEW-full
# After completion:
python3 reason_dprl/scripts/reproduce.py compare reason_dprl/results/NEW-full
reason_dprl/.venv/bin/python reason_dprl/scripts/summarize_optimizations.py \
  reason_dprl/results/NEW-full/family reason_dprl/results/NEW-full/wn18rr \
  --output reason_dprl/results/NEW-full-analysis
```

`--smoke` uses one batch and three timing repetitions; it verifies execution,
not the published three-batch aggregate. `--dataset family` or `--dataset wn18rr`
selects one job (also pass it to `compare`). `--dry-run` prints the submission
commands after checking prerequisites and creates no output directory/jobs.
`--partition` and `--node` can override cluster placement; another GPU/driver
is a new performance platform. A complete rerun took roughly seven minutes per
primary job after environment setup; queueing and cold compilation vary.

Each submission snapshots the Python harness and uses a distinct `/tmp` cache
per run/dataset. Do not edit a source checkout while its jobs are running. The
compiled workload stays in the pinned upstream checkout; provenance records
its commit, the harness hashes, checkpoint hash, GPU, command and configuration.
The helper changes orchestration only, not the measured inference code.

## Fresh checkout: source, environment, then frozen artifacts

```bash
git clone git@github.com:chenyi995/SymFabric.git
cd SymFabric
# Only these two upstream repositories are needed for this investigation.
python3 reason_dprl/scripts/sources.py fetch \
  --only DPrL-AAAI --only DAG_Processor

# Use an installed uv; create a new environment, preserving any existing one.
uv venv --python 3.12.13 reason_dprl/.venv
uv pip install --python reason_dprl/.venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match -r reason_dprl/requirements-gpu.lock.txt
```

[sources.lock.json](sources.lock.json) pins DPrL to
`cfc66233c7aca3677cfb4ae74a71fdf9559ed89c` and DPU-v2 to
`cefe56ec57a9794e1ae2263f2998da65f1e93340`. The source helper refuses revision
conflicts and tracked edits; it does not silently reset an existing checkout.
Family/WN18RR data text files are already tracked in the pinned DPrL source.
You do not need to fetch every GPU/DRAM simulator in the full source catalog.

Frozen profiling checkpoints, MNIST inputs, raw GPU profiles, benchmark JSON,
training logs, generated kernel sources and historical harness snapshots are
in one external bundle. Large run outputs and datasets remain ignored by Git.
[repro.lock.json](repro.lock.json) records the exact file identities:

| Property | Value |
|---|---|
| Bundle | `symfabric-dprl-20261009-v1.tar.gz` |
| Server | `athena.egr.duke.edu` |
| Server path | `/zpool-00/home/cw636/chenyi/SymFabric-artifacts/20261009-dprl/symfabric-dprl-20261009-v1.tar.gz` |
| Size / regular files | 83,773,410 bytes / 247 |
| SHA-256 | `62b784cd53ba7668988615714821a8fb9f4d7b8a7357a096f6792386b5e5676b` |

On this server, restore directly:

```bash
python3 reason_dprl/scripts/reproduce.py restore \
  /zpool-00/home/cw636/chenyi/SymFabric-artifacts/20261009-dprl/symfabric-dprl-20261009-v1.tar.gz
python3 reason_dprl/scripts/reproduce.py check --include-mnist
```

On another host with existing cluster access, first copy the bundle:

```bash
scp cw636@athena.egr.duke.edu:/zpool-00/home/cw636/chenyi/SymFabric-artifacts/20261009-dprl/symfabric-dprl-20261009-v1.tar.gz /tmp/
python3 reason_dprl/scripts/reproduce.py restore /tmp/symfabric-dprl-20261009-v1.tar.gz
python3 reason_dprl/scripts/reproduce.py check --include-mnist
```

The bundle is **not hosted on GitHub**. Exact output reproduction on a different
machine requires access to this frozen bundle. Restoring verifies its SHA and
size, validates all paths and existing-file conflicts before writing, and
retains identical files. It refuses to overwrite changed artifacts or follow
symlinked output parents. `check --skip-env` permits checking source and input
identity before installing the GPU environment; it is not an execution check.

The package list pins versions, not wheel hashes. Environment installation on
another platform and performance equivalence are not guaranteed by the lock.
Do not mistake successful source restoration for completed GPU reproduction.

## What the artifacts mean

| Artifact | Meaning |
|---|---|
| `results/20261009-gpu-opt/family_v3/` and `wn18rr_v3/` | Primary unprofiled optimized GPU results, jobs 220687 / 220688 |
| `results/20261009-gpu-opt/analysis/summary.json` | Aggregate latency table; raw per-batch samples remain in benchmark files |
| `results/20261009-gpu-opt/analysis/profile_attribution.json` | Nsight kernel-to-launch-to-NVTX attribution |
| `results/20261009-gpu-opt/analysis/profiles_with_counters.json` | Selected Nsight Compute samples, not all-kernel counters |
| `results/20261009-gpu-opt/nsys_family_fast/` and `nsys_wn18rr_fast/` | Raw Nsight captures, SQLite and exported tables |
| `results/20261009-gpu-opt/kernel_sources/` | Generated kernels used to verify source-level bottleneck attribution |
| `results/20261009-a5000/` | Earlier GPU/hybrid runs, profiling checkpoint training, and mapping ablations |
| `proposal/GPU_RESULTS.md` | Primary measured results and profiler instructions |
| `RESULTS_20261009.md` | Earlier experiment, with measured/modelled boundaries and later corrections |
| `RECONSTRUCTION.md` | Functional DPU-v2-based reconstruction and its assumptions |

Expected primary mean-of-batch-median latencies (milliseconds):

| Variant | Family | WN18RR |
|---|---:|---:|
| Corrected compiled | 52.396 | 97.978 |
| Complete CUDA Graph | 46.092 | 95.280 |
| Best tested compact path | 14.975 | 24.623 |

These are reference measurements, not exact timing acceptance thresholds.
Correctness is checked against the corrected native execution within each run:
active actions, retained state each round, final success/depth and repeatability.
Valid logits allow `atol=2e-4, rtol=2e-5` for FP32 batching roundoff. `compare`
additionally checks frozen-run query and final-output hashes. It does not store
and compare a full historical state trace or certify all future inputs.

The best variant differs: `fast_policy_graphs` for Family and
`fast_fused_graphs` for WN18RR. Both use stable first-K candidates. The original
scatter race affects **both datasets**; the old WN18RR ~112 ms trace is not the
denominator for the new corrected experiment. Do not disable the adapter to
claim a faster run. The optimized entry point installs it automatically.

## Profiling and the REASON reconstruction

Follow [GPU_RESULTS.md](proposal/GPU_RESULTS.md#reproduce) for `nsys` / `ncu`
submission. Use separate fresh output directories and keep profiler timings
out of primary performance aggregates. CUPTI captures resolve asynchronous
CUDA Graph launches via correlation IDs; empty PyTorch CUDA tables are not
proof of idle hardware. Phase spans, summed kernel durations, and unprofiled
end-to-end latency are different quantities. Source-level hotspot checks and
counter sample limits are documented in that report.

For GPU + reconstructed REASON, use `bench_kg.py bench --stable-candidates`
with the corresponding frozen checkpoint and the runner in RECONSTRUCTION.md.
The command executes functional comparisons and emits **uncalibrated** hybrid
estimates. KG offload currently covers final policy dot products only, leaving
unification and state construction on the GPU. No original REASON RTL, timing
calibration, or authors' simulator is present. The source checker prints that
REASON-specific artifacts are unavailable even though our reconstruction runs.
Do not describe this as a measured hardware speedup/slowdown or as a new
first-order symbolic engine.

## Retraining is a separate experiment

The checkpoints are short profiling runs, not paper-converged accuracy:
Family and WN18RR used 32,768 training transitions, MNIST 1,000 iterations.
The historical Family training used stable candidates; the historical WN18RR
training did not. Both optimized inference baselines now use the correction.
For new work, train both with the correction and label the results as new
checkpoints. A seed alone does not recreate the archived weights bit for bit.
The training commands in RECONSTRUCTION.md use fresh outputs; use `--dataset
wn18rr --stable-candidates` for a corrected WN18RR training experiment.

Without the bundle you can fetch datasets and train new weights, but cannot
claim reproduction of the frozen numbers. Every backend in a new comparison
must share the same weights, queries, candidate ordering, budgets and numerical
contract. No accuracy or hardware generality claims follow from two KG datasets.

## Handoff validation

The new entry point was exercised on this server: jobs **220700 / 220701**
completed successfully and `compare --smoke` passed for both datasets. The
combined test suite passed **21 tests**. An isolated directory also fetched
both pinned public repositories, restored the full artifact bundle, and passed
all input hashes. This validates source/artifact recovery and the working
environment, not a second clean installation of all GPU packages.

## Files to change next

- `scripts/fast_candidates.py`: exact stable candidate merging; it currently
  compacts after the unifier has created dense intermediate fact states.
- `scripts/optimize_kg_fast.py`: combined five-variant benchmark and correctness
  audits. Preserve the corrected native and full-graph controls.
- `scripts/optimize_kg_graphs.py`: live compaction, graph buckets, state packing.
- Pinned upstream `kge_experiments/unification.py`: inspect the semantics, but
  keep upstream tracked files clean; add adapters in our scripts as needed.
- `proposal/LITERATURE.md` and `PRIOR_ART.md`: narrow admission criteria and
  required prior-art comparisons. The next proposed transfer workloads are
  DPLA* and ACTP; they have not been implemented or profiled here.

Validate local checks with `reason_dprl/.venv/bin/python -m pytest -q
reason_dprl/tests`. The GPU packing check is `scripts/check_fast_candidates.py`
through `run_gpu.sbatch`; its original validation was job 220686. None of this
requires changing the unrelated core algebra or architecture-model packages.
