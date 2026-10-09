# Reproduce the dynamic-state hypothesis tests

Run from the SymFabric repository root. Read
[DYNAMIC_VALIDATION.md](DYNAMIC_VALIDATION.md) for conclusions and limitations.
The earlier baseline/checkpoint recovery instructions in
[REPRODUCE.md](../REPRODUCE.md) are prerequisites for a fresh machine.

## Recover the exact evidence

```bash
# Verify the original DPrL sources, data, checkpoints and GPU environment.
python3 reason_dprl/scripts/reproduce.py check

# Restore missing CTP/DeepProbLog checkouts at their pinned commits.
# Existing checkouts with another revision or tracked edits are rejected.
python3 reason_dprl/scripts/dynamic_artifacts.py sources --fetch

# Restore the separate follow-up bundle from its same-server default path.
# Verifies bundle SHA/size and all member paths before writing; conflicts fail.
python3 reason_dprl/scripts/dynamic_artifacts.py restore
python3 reason_dprl/scripts/dynamic_artifacts.py check
```

[dynamic.lock.json](../dynamic.lock.json) records upstream pins, bundle identity,
all archived artifact hashes, and primary job identities. The bundle is stored
under `/zpool-00/home/cw636/chenyi/SymFabric-artifacts/20261009-dprl/` on
`athena.egr.duke.edu`; it is **not on GitHub**. Its name is
`symfabric-dynamic-validation-20261009-v1.tar.gz`. It contains results, raw
profiles, compiler-function snapshots, harness snapshots and the small Ubuntu
SWI-Prolog runtime packages. It does not duplicate the original checkpoint
bundle. On another host, copy it with existing cluster access and pass the
local archive path to `dynamic_artifacts.py restore`.

Important artifacts under `reason_dprl/results/20261009-dynamic-validation/`:

| Artifact | Role |
|---|---|
| `family-full/`, `wn18rr-full/` | Primary complete comparisons; jobs 220709 / 220713 |
| `summary.json` | Validated three-batch aggregates |
| `nsys-family/`, `nsys-wn18rr/` | Raw graph-node traces; jobs 220714 / 220717 |
| `profile-summary.json` | Launch-correlated kernel attribution |
| `ctp-v2/` | Official CTP mechanism probe and fixed-input graph control; job 220711 |
| `dpla/` | Official DPLA* engine semantic probe, pins and runtime packages |
| `harness-v1.tar.gz`, `harness-final.tar.gz` | Source snapshots; primary provenance contains exact script hashes |
| `slurm-accounting.txt` | Completion and exit codes |

The archived failed setup attempts and earlier `ctp/` probe are not primary
performance results. `profile-family/` and `profile-wn18rr/` stop at the capture
variant and intentionally lack a complete three-variant comparison. Their
instrumented timings must not replace `*-full/`.

## Rerun DPrL on Slurm

The environment is the original Python 3.12.13 / PyTorch 2.9.0+cu130 lock.
The login host has no GPU. Submit separate jobs to `athena/node2`:

```bash
mkdir -p reason_dprl/results/NEW-dynamic
sbatch --parsable \
  --export=ALL,SYMFABRIC_CACHE_BASE=/tmp/NEW-dynamic-family \
  --output=reason_dprl/results/NEW-dynamic/family.log \
  reason_dprl/scripts/run_gpu.sbatch reason_dprl/scripts/validate_dynamic.py \
  --dataset family \
  --checkpoint reason_dprl/results/20261009-a5000/family_train_stable_shm/checkpoint.pt \
  --output reason_dprl/results/NEW-dynamic/family

sbatch --parsable \
  --export=ALL,SYMFABRIC_CACHE_BASE=/tmp/NEW-dynamic-wn18rr \
  --output=reason_dprl/results/NEW-dynamic/wn18rr.log \
  reason_dprl/scripts/run_gpu.sbatch reason_dprl/scripts/validate_dynamic.py \
  --dataset wn18rr \
  --checkpoint reason_dprl/results/20261009-a5000/wn18rr_train_a5000/checkpoint.pt \
  --output reason_dprl/results/NEW-dynamic/wn18rr
```

Use new output paths. Do not edit the scripts or upstream source while jobs
run. For a quick functional smoke, append `--batches 1 --repeat 3` to each job;
this does not reproduce the three-batch timing aggregate. Primary jobs took
about five and nine minutes, including compilation; queue/cache conditions vary.
Require Slurm `COMPLETED`, exit `0:0`, and `VALIDATION_COMPLETE` in both logs.

```bash
python3 reason_dprl/scripts/summarize_dynamic.py \
  reason_dprl/results/NEW-dynamic/family \
  reason_dprl/results/NEW-dynamic/wn18rr \
  --output reason_dprl/results/NEW-dynamic/summary.json
```

For smoke results add `--batches 1 --repeat 3` to the summarizer. It rejects
partial comparisons, missing edge tests, failed state/action audits, different
output hashes or incorrect repetition counts. The main benchmark compares
against a fresh corrected native trace; neither future states nor actions are
cached in the timed optimized path. Index construction, compilation and graph
capture are startup costs reported separately from steady-state latency.

The adapter supports only the recorded ground-fact/non-parity configuration.
It retains first-K order and inspects overflow proofs before truncation.
Changing these semantics invalidates the comparison even if final answers
happen to agree on a small batch.

## Nsight attribution

Use the existing wrapper, with a new profile/output directory. For Family:

```bash
sbatch --export=ALL,SYMFABRIC_CACHE_BASE=/tmp/NEW-dynamic-family \
  --output=reason_dprl/results/NEW-dynamic/nsys-family.log \
  reason_dprl/scripts/profile_optimized.sbatch nsys \
  reason_dprl/results/NEW-dynamic/nsys-family \
  reason_dprl/scripts/validate_dynamic.py --dataset family \
  --checkpoint reason_dprl/results/20261009-a5000/family_train_stable_shm/checkpoint.pt \
  --output reason_dprl/results/NEW-dynamic/profile-family \
  --batches 1 --repeat 3 --capture-only compact

python3 reason_dprl/scripts/summarize_dynamic_profile.py \
  reason_dprl/results/NEW-dynamic/nsys-family/native.sqlite \
  --output reason_dprl/results/NEW-dynamic/profile-summary.json
```

For WN18RR change the dataset, checkpoint and output/cache paths. Pass both
SQLite files to the summarizer for a combined report. Capture uses
`dynamic_validation`, not the older analyzer's `native_query` range. Profiler
windows contain substantial overhead; use the uninstrumented full runs for
latency. This experiment did not collect new Nsight Compute hardware counters.

## CTP execution probe

```bash
sbatch --export=ALL,SYMFABRIC_CACHE_BASE=/tmp/NEW-ctp \
  --output=reason_dprl/results/NEW-ctp.log \
  reason_dprl/scripts/run_gpu.sbatch reason_dprl/scripts/probe_ctp.py \
  --output reason_dprl/results/NEW-ctp
```

Require `CTP_PROBE_COMPLETE`. The probe imports the pinned official classes,
uses a bundled 32-fact prefix of Kinship and seed-42 random frozen weights,
counts operation shapes for two different queries, and compares eager versus
fixed-input graph execution. This is **not ACTP or trained CTP accuracy**.
The CUDA event omission in the recorded depth-zero profiler trace is disclosed
in the results report; do not interpret missing events as zero GPU work.

## DPLA* semantic probe

On a system with SWI-Prolog installed, run:

```bash
swipl -q -s reason_dprl/scripts/probe_dpla.pl -- \
  softwareworkload/upstream/deepproblog/src/deepproblog/engines/prolog_engine/prolog_files/engine_heap.pl
```

The JSON must contain `passed: true`, with two versus one fact successors and
matching ground heuristic arguments. The harness consults the official engine;
its clauses and heuristic values are synthetic. It proves the execution
contract, not application accuracy, neural inference latency or a bottleneck.

For the recorded Ubuntu 22.04 runtime without system installation:

```bash
mkdir -p reason_dprl/results/NEW-dpla-runtime
for package in reason_dprl/results/20261009-dynamic-validation/dpla/runtime_debs/*.deb; do
  dpkg-deb -x "$package" reason_dprl/results/NEW-dpla-runtime
done
dpla_runtime="$PWD/reason_dprl/results/NEW-dpla-runtime"
SWI_HOME_DIR="$dpla_runtime/usr/lib/swi-prolog" \
LD_LIBRARY_PATH="$dpla_runtime/usr/lib:$dpla_runtime/usr/lib/x86_64-linux-gnu" \
  "$dpla_runtime/usr/bin/swipl" -q -s reason_dprl/scripts/probe_dpla.pl -- \
  softwareworkload/upstream/deepproblog/src/deepproblog/engines/prolog_engine/prolog_files/engine_heap.pl
```

The packaged runtime is Linux x86-64 and is not a portability guarantee for
other operating systems. Its package versions and hashes are in the DPLA*
provenance and follow-up artifact lock.
