# Dynamic proof execution with REASON and DeepProofLog

The Slurm GPU experiment harness and DPU-v2-backed REASON reconstruction are
documented in [RECONSTRUCTION.md](RECONSTRUCTION.md). They add executable
measurements and an explicitly uncalibrated accelerator model to the comparison
design below. Generated run records remain under the ignored `results/` tree.

This project evaluates DeepProofLog (DPrL) on a GPU and on a GPU coupled to
REASON, then isolates the benefit of extending REASON for dynamic proof
execution. The research question is whether processing variable candidate sets
and maintaining proof state near the execution units reduces complete query
latency or increases throughput at matched answer quality.

The proposed mechanisms are research hypotheses, not established performance
results. The comparison separates algorithmic work reduction from faster
execution of the same proof process.

The [comparison protocol](COMPARISON.md) defines baselines, timing boundaries,
profiling, simulation inputs, correctness, and the evidence required for a
speedup claim. [sources.lock.json](sources.lock.json) records the inspected
source revisions; [scripts/sources.py](scripts/sources.py) restores and checks
the source links without building or running workloads.

## Research scope

DPrL selects logic-resolution actions using a neural policy conditioned on the
current goal. Applying rules and substitutions creates successor goals at
runtime. Its manageable-state variant uses exact dynamic programming; its
large proof-space variant uses policy-gradient learning. The initial hardware
comparison concerns inference with a frozen checkpoint. Training is a separate
experiment with separate forward, backward, rollout, and optimizer accounting.

REASON already supports dynamic SAT propagation, watched-literal indexing,
feedback queues, and conflict handling. Its probabilistic path uses offline DAG
construction and mapping. The proposed extension concerns goal-dependent proof
expansion, variable-size candidate sets, and persistent query state. It does
not claim that REASON has no dynamic control, nor that watched-literal handling
already implements first-order unification. See REASON sections IV-C and V-C–E.

DPrL assigns probabilities to derivations, rather than possible worlds. Keep
that semantics on every execution backend. A Boolean closure, a maximum-score
KG path, and DPrL query success probability are different outputs.

## Comparison matrix

| Configuration | Execution | Evidence sought |
|---|---|---|
| GPU DPrL | Pinned DPrL GPU implementation and its required host work | Measured performance and bottlenecks |
| GPU with original REASON | Same DPrL method; only supported operations offloaded; remaining work on GPU or CPU | Benefit and limitations of existing REASON |
| GPU with dynamic REASON | Same method and platform, with explicitly modeled dynamic mechanisms | Incremental benefit and cost of the proposed extension |

Keep the GPU, host, memory platform, checkpoint, query set, numerical contract,
and search budget aligned. A reconstructed REASON model must be labelled as a
reconstruction. The existing SymFabric analytical model and DPU-v2 are not
substitutes for the original REASON baseline.

## Workloads and hypotheses

| Workload | Role in the investigation |
|---|---|
| DPrL KG completion on Family and WN18RR | Goal-dependent candidates, unification, substitutions, history checks, and varying proof length |
| DPrL MNIST addition with dynamic programming | Regular, manageable-state control case for arithmetic and reduction reuse |
| Concurrent KG queries sharing a knowledge base and policy | State residency, work scheduling, and cross-query data reuse |

Dataset files and trained checkpoints must be acquired and identified separately
from source code. The public DPrL revision has GPU-vectorized unification,
fixed-shape candidate tensors, masks, and compiled execution paths. Compare
against those paths when supported; a Python-only implementation is not a
competitive GPU baseline. Candidate padding, state copying, and uneven query
progress are hypotheses to measure, not predetermined bottlenecks.

Possible extensions include indexed candidate generation, compact candidate
storage, queues carrying goal descriptors, and reuse of rule templates. A
candidate architecture could compile reusable computation templates offline
and submit descriptors at runtime. Recompiling a full DAG for every new goal
must be accounted for if that is the implementation used.

## Shared source layout

Source trees are stored once in `softwareworkload/upstream/` and
`systolicarch/upstream/`. Git tracks the relative symlinks below and the lock
file, not the downloaded source trees. Moving the whole SymFabric checkout
preserves the links after restoring its upstream directories.

| Entry point | Shared source | Role |
|---|---|---|
| [workloads/dprl](workloads/dprl) | `softwareworkload/upstream/DPrL-AAAI` | DPrL workload source |
| [simulators/accel-sim](simulators/accel-sim) | `systolicarch/upstream/accel-sim-framework` | GPU timing and integration framework |
| [simulators/gpgpu-sim](simulators/gpgpu-sim) | Accel-Sim's `gpu-simulator/gpgpu-sim` checkout | Underlying GPU model |
| [simulators/dpu-v2](simulators/dpu-v2) | `systolicarch/upstream/DAG_Processor` | Separate DAG compiler and RTL reference |
| [simulators/sam](simulators/sam) | `systolicarch/upstream/sam` | Sparse coordinate and streaming reference |
| [simulators/scale-sim](simulators/scale-sim) | `systolicarch/upstream/SCALE-Sim` | Dense systolic primitive model |
| [simulators/stonne](simulators/stonne) | `systolicarch/upstream/stonne` | Dense and sparse accelerator reference |
| [simulators/ramulator2](simulators/ramulator2) | `systolicarch/upstream/ramulator2` | Optional detailed DRAM modeling |
| [simulators/graphlily](simulators/graphlily) | `systolicarch/upstream/GraphLily` | FPGA semiring source reference, not a CPU timing simulator |

These are alternative supporting tools, not an already integrated simulation
stack. The added reconstruction imports DPU-v2 graph construction and supplies
its own arithmetic lowering and timing model; see [RECONSTRUCTION.md](RECONSTRUCTION.md).
It does not establish calibrated REASON execution or turn DPU-v2 into the
authors' artifact. No `simulators/reason` link points to an unrelated simulator.

From the SymFabric root:

```bash
# Verify the current checkouts, recorded revisions, and symlink destinations.
python3 reason_dprl/scripts/sources.py check

# Create missing links to already available, matching checkouts.
python3 reason_dprl/scripts/sources.py link

# On a fresh checkout, download missing pinned sources and then create links.
# Requires network access; does not change existing checkouts or build anything.
python3 reason_dprl/scripts/sources.py fetch
```

The helper refuses to overwrite an existing directory or conflicting link, or
silently update a source revision. `record` captures initial source metadata
from local Git checkouts into a new lock file and refuses to overwrite one.
The historical dependency inventory used by that command is
[`sym1/upstream_lock.json`](../sym1/upstream_lock.json).

## Execution requirements and current readiness

CPU execution can collect algorithmic event counts and support functional
validation, subject to DPrL dependency and device compatibility. It cannot
measure GPU bottlenecks. A real GPU is needed for the measured GPU baseline
and new GPU instruction traces. Existing matching traces can be replayed on a
CPU server. REASON simulation can also run offline on the CPU once its backend
and workload mapping exist.

The login host has no usable GPU, but Slurm `athena/node2` provides RTX A5000
GPUs. The project-specific scripts now train frozen profiling checkpoints,
measure compiled and CUDA Graph inference, capture Nsight profiles, and execute
the functional arithmetic reconstruction. An opt-in stable candidate adapter
corrects a reproducibility problem found in upstream Family inference.
See [the run guide](RECONSTRUCTION.md) and [the measured results](RESULTS_20261009.md).
Accelerator timings remain uncalibrated model estimates; GPU timings are measured.

## Sources

- [REASON](https://arxiv.org/abs/2601.20784), especially sections IV–VII:
  architecture, mapping, runtime control, and measured-GPU/simulated-system evaluation.
- [DeepProofLog](https://arxiv.org/abs/2511.08581), sections III–VI and appendices
  B–C: semantics, proof environment, and workloads.
- [DPrL source](https://github.com/DeepProofLog/DPrL-AAAI): the lock file records
  the inspected implementation; it is not assumed identical to the paper artifact.
- [Architectural Implications of Agentic AI Workflows](https://arxiv.org/abs/2608.04458):
  dependency chains, bursts, task locality, and end-to-end accounting.
- [Efficient Processing of Neuro-Symbolic AI](https://proceedings.mlr.press/v288/wan25a.html):
  workload-to-operator characterization and cross-layer co-design. Its VSA case
  study does not establish DPrL's bottleneck.
