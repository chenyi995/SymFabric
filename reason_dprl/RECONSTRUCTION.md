# REASON reconstruction and GPU experiments

This implementation runs pinned DeepProofLog on an allocated Slurm GPU and
executes a functional REASON reconstruction on the same inputs. Accelerator
latency comes from an **uncalibrated cycle-based model**. It is not a measurement
of REASON hardware or a reproduction of the authors' Accel-Sim integration.

For the latest corrected controls and portable input recovery, start with
[REPRODUCE.md](REPRODUCE.md). This document also preserves the earlier experiment
contract; the optimized GPU comparison is in [GPU_RESULTS.md](proposal/GPU_RESULTS.md).

## Source and implemented mechanisms

`scripts/reason_model.py` imports the public DPU-v2 `GraphClass` and its binary
tree constructor from the fresh checkout at
`systolicarch/upstream/DAG_Processor`, commit
`cefe56ec57a9794e1ae2263f2998da65f1e93340`. The MIT-licensed upstream checkout is
kept intact. REASON uses this architecture as its implementation template
([paper, Section VII-A](https://arxiv.org/html/2601.20784v1#S7.SS1)).

The added compiler groups binary arithmetic nodes into single-output blocks
with bounded depth, keeps shared values between blocks, and produces a
deterministic schedule. The timing model accounts for dependencies, tree issue,
bank reads and writes, finite registers, lowest-free register allocation,
spills and reloads, finite SRAM, external transfers, output buffering,
configuration and GPU handoff. Its static scheduling is conservative and does
not reproduce the authors' compiler heuristics or establish optimal throughput.

A separate two-watched-literal implementation checks implication queues,
watch relocation and priority conflict flushing against an independent clause
scan. It is a functional SAT control reference. SAT timing, CDCL search and
first-order unification are not implemented by that component.

| Parameter | Value | Provenance |
|---|---:|---|
| Trees and aggregate nodes | 12 and 80 | REASON Table III |
| Maximum tree depth | 3 | Section V-F |
| Register banks and registers per bank | 64 and 32 | Section V-F; global scope is an assumption |
| SRAM | 1.25 MB | Table III; decimal MB interpretation |
| Nominal clock | 500 MHz | Scaled 12/8 nm rows, Table III footnote; not the unspecified 28 nm clock |
| Data words | FP32, 4 bytes | Reconstruction choice, verified numerically |
| Bank read and write ports | 1 each | Assumption |
| SRAM and external access latency | 2 and 100 cycles | Assumptions |
| External transfer rate | 64 bytes/cycle | Assumption, not measured GPU L2 bandwidth |
| Handoff | 1 microsecond per direction | Assumption |
| Output queue | 16 entries | Assumption |
| Instruction size | 16 bytes | Assumption |

The reported 80 nodes and 12 depth-three trees do not specify a complete
physical topology. This implementation limits aggregate issued arithmetic
work to 80 nodes per cycle; it does not assert a particular 80-node layout.
Sensitivity cases vary clock, handoff, ports, register capacity, memory latency
and interface bandwidth. `mapping_sweep.py` also varies the number of independent
queries kept live at once (1, 4, 8, 16, or the entire batch). Each tile runs
serially inside one accelerator invocation, preserving every operation and
the complete-batch interface payload. This exposes compiler-induced register
spilling instead of treating one conservative mapping as a hardware limit.
Simulated delay counters can overlap compute and
must not be summed as independent end-to-end speedups.

## Workload contracts

DPrL source is pinned to `cfc66233c7aca3677cfb4ae74a71fdf9559ed89c`. Data files,
checkpoints and GPU query tensors are hashed in each run. Locally trained
checkpoints are frozen for inference and shared by both backends; their
training budget and achieved quality are reported without claiming convergence
or reproducing the paper's accuracy.

| Workload | GPU baseline | REASON mapping | GPU/CPU work retained |
|---|---|---|---|
| Family and WN18RR | Native compiled `PPO.evaluate_policy`, plus sampled ranking | Policy's final dot products in the paper's sum-product/SpMSpM mode | Embeddings, MLP, normalization, action selection, first-order unification, variable renaming, candidate/state/history management |
| MNIST addition | Shipped PI marginal scoring, eager and compiled | Static probability-space sum-product DAG | CNN, softmax, logarithms and target gathering |

KG uses the upstream depth, atom and candidate budgets, including WN18RR's
fact-pair cap. It disables optional KGE score fusion because the public source
does not ship those separately trained KGE weights. Sampled ranking is labelled
with its query and negative counts; it is not exhaustive ranking. The native
proof API runs the entire depth budget even after a query completes. The
stop-when-finished experiment is a separate work-reduction ablation and checks
that terminal outputs are identical.

KG offload uses SRAM-sized tiles of eight candidates. Query vectors are shared
within a tile, while the conservative mapping resets register allocation
between tiles. Interface payload counts the query vector once per step.
Functional simulator values feed the real GPU environment in a closed loop;
actions and terminal outputs must agree before a timing estimate is emitted.
The performance estimate uses a validated fixed-trajectory replay to measure
retained GPU/host work, then adds accelerator and interface time at each serial
dependency. CPU interpreter and PCIe emulation time are recorded separately.
Offloading the final KG dot products is an explicit supported-operator
experiment. A symbolic-only dispatcher has no implemented KG lowering and
falls back entirely to the GPU; there is no demonstrated KG symbolic speedup.

### Upstream candidate reproducibility correction (both KG datasets)

The pinned fast-mode `unification.pack_results` clamps invalid and overflowing
scatter destinations to the last retained candidate. CUDA writes collide there;
the source comment incorrectly assumes ordered writes. Eight repeated native
proof calls produced eight terminal-output hashes. `--stable-candidates`
installs `stable_candidates.py` at runtime on both backends. It retains the
first K valid entries in the same fact-then-rule order and sends all discarded
writes to a separate sink. The downloaded source remains intact. Corrected
Family training and inference are labelled separately from the untouched
upstream diagnostic. The first historical WN18RR batch passed repeatability without the adapter,
but expanded testing exposed the same race on the second batch. All current
optimized inference comparisons use the correction on both datasets. The
historical WN18RR training checkpoint remains frozen; its training did not use
the correction. See the optimized report for the valid comparison denominator.
The duplicate-index behavior is also documented by
[PyTorch 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.scatter_.html).

The public MNIST `MNISTEvalMixin.evaluate` predicts digits by neural argmax; it
does not call symbolic PI scoring. The control experiment here explicitly
measures **PI marginal scoring**, which does exercise that recurrence. The
shipped recurrence propagates unconditional carry distributions and adds
per-digit log marginals. This is preserved exactly within numerical tolerance;
it must not be described as exact joint probability of the complete target sum.
The test input digit order is reversed for PI, matching upstream training.
The compact GPU control removes impossible mask entries while retaining the
same arithmetic contract; its gain is a software/work-reduction gain.
All reported inference uses strict FP32 with TF32 disabled. Fixed resident
inputs are also tested with a complete CUDA Graph capture. Graph controls check
outputs before contributing to the best-GPU comparison. For MNIST, GPU prefix
and suffix graphs provide the retained-work term in the graph-based hybrid
estimate; the two measured segment completions and the modeled handoff are
charged explicitly. These measurements do not cover dynamic input loading.

## Timing and reproducibility

`run_gpu.sbatch` targets `node2` in `athena` because it has a CUDA 13 compatible
driver. Each job requests one GPU, eight CPUs and 64 GB host memory. Generated
compiler caches originally used executable node-local `/dev/shm`. For new jobs,
use a unique `/tmp` directory via `SYMFABRIC_CACHE_BASE`; the reproduction helper
does this automatically after shared-memory cache files disappeared in later runs. Shared NFS imports and later local-disk cache
renames stalled cold compilation. They are outside steady-state inference
timing. `snapshot_harness.py` creates an immutable source bundle; setting
`SYMFABRIC_HARNESS_ARCHIVE` makes Slurm extract it locally to avoid stale NFS
source/bytecode and records its file hashes in each run.
The Python environment is project-local under `.venv`.

The recorded dependency set is [requirements-gpu.lock.txt](requirements-gpu.lock.txt)
(Python 3.12; PyTorch 2.9.0 CUDA 13 wheels). With `uv` installed:

```bash
uv venv --python 3.12.13 reason_dprl/.venv
uv pip install --python reason_dprl/.venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match -r reason_dprl/requirements-gpu.lock.txt
mkdir -p reason_dprl/results
python3 reason_dprl/scripts/snapshot_harness.py reason_dprl/results/harness.tar.gz
export SYMFABRIC_HARNESS_ARCHIVE=reason_dprl/results/harness.tar.gz
export SYMFABRIC_CACHE_BASE=/tmp/symfabric-manual-reconstruction
```

```bash
# Run from the repository root. Choose fresh output directories for every run.
sbatch --output=reason_dprl/results/train-family-%j.log \
  reason_dprl/scripts/run_gpu.sbatch reason_dprl/scripts/bench_kg.py train \
  --dataset family --stable-candidates --batch 64 --steps 32768 --output reason_dprl/results/family-train

sbatch --output=reason_dprl/results/bench-family-%j.log \
  reason_dprl/scripts/run_gpu.sbatch reason_dprl/scripts/bench_kg.py bench \
  --dataset family --stable-candidates --batch 64 --checkpoint reason_dprl/results/family-train/checkpoint.pt \
  --output reason_dprl/results/family-bench

sbatch --output=reason_dprl/results/train-mnist-%j.log \
  reason_dprl/scripts/run_gpu.sbatch reason_dprl/scripts/bench_mnist.py train \
  --iterations 1000 --output reason_dprl/results/mnist-train

sbatch --output=reason_dprl/results/bench-mnist-%j.log \
  reason_dprl/scripts/run_gpu.sbatch reason_dprl/scripts/bench_mnist.py bench \
  --batch 64 --checkpoint reason_dprl/results/mnist-train/checkpoint.pt \
  --output reason_dprl/results/mnist-bench

reason_dprl/.venv/bin/python -m pytest -q reason_dprl/tests
```

Unprofiled timings include GPU completion. Warm-up/compilation, allocations,
latency samples, query/checkpoint hashes and software versions are saved.
PyTorch timelines and operator tables are captured in separate runs. On this
installation those traces contained CPU activities only despite requesting
CUDA activities. GPU attribution therefore uses actual CUPTI kernels from
Nsight Systems, rather than interpreting empty traces as idle GPU time.
`profile_gpu.sbatch` supports Nsight Systems and selective Nsight Compute;
`--capture-only` excludes compilation through CUDA profiler API or NVTX ranges.
Hardware counter access succeeded on node2. `NCU_KERNEL_REGEX` selects hotspot
launches; counters from the first few reset kernels are not treated as a
whole-workload sample. Nsight capture windows can incur substantial overhead,
so their busy fractions are not unprofiled utilization estimates.
The profiler wrapper builds reports and SQLite databases locally, then copies
completed artifacts to the result directory to avoid NFS export stalls.

The simulated hybrid is a serial, warm, inference-only estimate on the measured
GPU platform. Cold mapping/configuration is reported separately. No shared-L2
contention, RTL calibration, power, area, energy or overlapping multi-batch
throughput claims are made. Those require further integration and validation.
