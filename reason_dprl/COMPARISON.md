# Comparing DPrL execution on GPUs and REASON

The comparison keeps DPrL's reasoning method fixed while changing the execution
backend. It must identify both the change in complete-query performance and
the mechanism responsible. Algorithmic search reduction and faster execution
of the same search are reported separately.

## Baseline contracts

| Contract | GPU DPrL | Original REASON system | Dynamic REASON system |
|---|---|---|---|
| Method | Frozen DPrL policy and specified inference procedure | Same | Same for the architectural comparison |
| Neural execution | Pinned GPU software path | Same GPU path unless explicitly mapped and priced | Same, with any changes isolated |
| Symbolic execution | DPrL GPU path plus required host work | Supported operations mapped to REASON; residual operations retained on GPU or CPU | Explicitly specified dynamic extensions |
| Memory and communication | Real platform | Matched GPU/host/memory model and defined REASON attachment | Same attachment unless treated as a separate experiment |
| Evidence label | Measured | Simulated, with model validation recorded | Simulated, with added mechanisms recorded |
| Output contract | Query scores, proof validity and task quality | Same | Same |

If the actual REASON artifact is unavailable, name the backend a REASON
reconstruction. Calibrate the mechanisms that are available and document
unvalidated ones. Do not replace it with `systolicarch/model.py`, SAM, or DPU-v2
while retaining the REASON label. A generic Accel-Sim checkout does not include
REASON's custom datapath, compiler, or GPU integration automatically.

Original REASON must not receive unpriced support for first-order unification,
dynamic allocation, variable renaming, sampling, or history management. Maintain
an operation-to-device map with evidence for every offloaded operation and an
explicit fallback for the remainder. Mapping overhead belongs in the result.

## Workload identity and correctness

Each comparison records source commits, dependency versions, dataset and split
checksums, rule/fact checksums, checkpoint checksum, query order, batch policy,
precision, random seeds, and the exact inference procedure. Record limits on
proof depth, goal length, candidate capacity, fact sampling, and rollout budget.
Changing a limit or sampling rule can change the answer and work, not just speed.

For KG completion, state whether evaluation ranks against sampled negatives or
the complete entity set, and preserve that choice across backends. Record MRR,
Hits metrics and proof success/validity alongside latency. For MNIST addition,
record sum accuracy and the numerical probability contract. An inference-time
comparison uses a trained checkpoint; training to convergence is a separate
study and cannot be mixed with per-query latency.

Use two complementary execution modes:

- **Matched trajectory:** record the actions for each query under the reference
  policy. Execute those same actions and candidate semantics on each backend.
  Check candidate sets, substitutions, state transitions, and terminal results.
  This isolates implementation cost without letting stochastic exploration
  change the workload. It is an attribution experiment, not a substitute for
  validating the accelerated policy path.
- **Closed loop:** each backend produces its own scores and choices under the
  declared numerical and random-stream contract. Check valid proofs and task
  quality, and report expansions and trajectory lengths. If floating-point
  differences change action selection, report their effect rather than assuming
  identical work. Scheduling that changes search or stopping needs this mode.

Check small instances against an independent functional implementation or
exhaustive enumeration where tractable. DPrL derivation probabilities cannot
be replaced by Boolean reachability or maximum-path scores. History-dependent
action masks make a goal atom alone an insufficient state identity. Any state
sharing must preserve the relevant full goal, substitutions, history, and
policy inputs. Sum-product aggregation must not count a contribution twice.

## GPU measurement

Measure the optimized, supported DPrL GPU execution path, including its
vectorized environment and compiled kernels where applicable. Record eager
and compiled variants explicitly if both are evaluated. Do not assume a
configuration uses compilation merely because compilation helpers exist.

Separate setup from steady-state execution: data/model loading, index creation,
compilation, warm-up, and trace capture are recorded separately. Complete-query
timing starts with ready query inputs and ends when answers are available to
the caller, with device completion included. Also report cold-start cost if
it matters to the deployment being compared.

Add NVTX ranges at the following semantic boundaries before profiling. This
table specifies the instrumentation to implement; it is not a claim that the
current upstream source emits these ranges.

| Range | Structural counters | Timing and resource questions |
|---|---|---|
| Candidate lookup and unification | Rule/fact probes, attempted matches, valid matches, substitutions | Indexed access cost; wasted comparisons; memory stalls |
| Successor construction and filtering | Valid candidates, allocated candidate slots, atoms copied, history probes | Padding; temporary allocations; compaction and scatter traffic |
| Policy evaluation and selection | Embedding gathers, scored candidates, action masks, selections | Batch efficiency; neural compute; normalization and selection cost |
| State and history update | Changed atoms, history reads/writes, terminated queries | State movement; history checks; inactive batch lanes |
| Query orchestration | Stage transitions, ready/waiting queries, launches, synchronizations | Host overhead; gaps; serial dependencies; queueing |

Use Nsight Systems for the CPU/GPU timeline, launches, synchronization and data
movement. Use Nsight Compute selectively on dominant kernels for achieved
throughput, memory traffic, cache behavior, occupancy and stall reasons, with
metric names checked against the installed version and GPU. Obtain benchmark
latency from separate runs without kernel-counter replay. Record profiler
overhead and capture boundaries. The existing profiling wrappers are starting
points and require DPrL-specific integration and validation.

Distinguish logical tensor bytes, allocated capacity, and measured memory
traffic. Low compute utilization does not by itself establish a bandwidth
bottleneck: dependencies, memory latency, small kernels, and masked work are
different causes. A CPU profile does not establish a GPU time breakdown.

## Simulation input and backend requirements

Maintain two different trace types:

- GPU instruction and memory traces for the pinned GPU baseline or retained GPU
  kernels. Accel-Sim's trace-driven path collects these on GPU hardware and
  replays them in its timing model.
- DPrL semantic events for mapping onto REASON: queries, goals, candidate
  generation, unification, substitutions, policy requests, actions, state
  writes, dependencies and terminal events. GPU SASS instructions are not a
  REASON instruction stream.

A semantic trace must identify `query_id`, `step_id`, `event_id`, predecessor
event IDs, operation, input/output state IDs, shapes, valid versus allocated
counts, selected action, and logical read/write sizes. Record full data or a
replayable reference to reconstruct functional transitions. Include the
address/layout mapping when using it to predict cache or bank behavior.
Logical sizes alone cannot establish physical memory stalls.

Replay is valid for an unchanged semantic trajectory. If allocation, batching,
policy selection, search order, or time-based termination changes future work,
couple the timing simulator to functional state progression. Keep
algorithm-dependent feedback in that loop instead of prescribing it with an
old trace.

The REASON backend needs explicitly modeled tree execution, instruction issue,
mapping/configuration costs, banked storage, buffering, backpressure, scalar
control, and the GPU interface. Price unsupported operations on their fallback
device. Record numerical units and throughput rather than treating all
operations as unit-cost semiring arithmetic. Constants must cite REASON's
applicable configuration or measured calibration; unknown values remain
`TBD[src]` and sensitivity parameters, not claimed measurements.

Use the supporting simulators only for the mechanisms they implement:

- Accel-Sim/GPGPU-Sim: GPU kernels and modeled integration.
- DPU-v2: a separate DAG execution/compiler reference.
- SAM: coordinate processing, sparse streams and backpressure experiments.
- SCALE-Sim or STONNE: selected dense/sparse compute primitives.
- Ramulator2: detailed memory behavior if the selected memory model is supported
  and integrated. Adding its timing to another model requires one owner for each
  memory delay, so the same access is not charged twice.
- GraphLily: an FPGA semiring implementation reference with its own toolchain.

Linking these repositories does not connect their clocks, queues, or memory
models. A system-level comparison requires an explicit integration contract.

## End-to-end timing and attribution

For a strictly serial execution with no overlap, a derived accounting model is:

```text
T_GPU = T_policy + T_symbolic + T_host
T_hybrid = T_retained_GPU_CPU + T_REASON + T_handoff + T_mapping
speedup = T_GPU / T_hybrid
```

Each term covers disjoint time. In an overlapping pipeline, use dependency-aware
event timing and total makespan instead of adding overlapping durations or
subtracting aggregate kernel time from wall time. GPU/REASON shared-memory
contention can change retained GPU time; isolated GPU measurements alone do
not capture it.

An optional optimistic bound, derived for the serial case with all other work
unchanged and zero added overhead, is `speedup_max = 1 / (1 - f)`, where `f` is
the measured fraction of baseline time that could be removed. State those
assumptions; it is not a predicted accelerator result. A weak bound indicates
that accelerating the selected stages cannot substantially improve that case.

Report complete-query latency distributions, throughput, answer quality, work
counts, and state footprint at the same query load. Record the batch or
concurrency setting and distinguish queueing from service time. A hardware
speedup claim needs agreement on work and semantics; a search optimization needs
quality-versus-cost results as well.

Use controlled ablations for compact candidates, indexed matching, persistent
state, and cross-query scheduling when those mechanisms exist. Include the
cost of each mechanism. Simulated stall attribution should distinguish operand
readiness, tree underutilization, SRAM bank conflicts, memory requests, full
queues, configuration, fallback work, and handoff waits. Overlapping stall
reasons must not be summed as independent time savings.

## Execution record and completion criteria

Every actual run uses a new ignored directory under `results/` or `traces/`.
Keep its command, environment, source/configuration hashes, input/checkpoint
checksums, output digest, correctness result, profiler/simulator logs, and
measurement repetitions. Do not commit raw traces, datasets, model weights,
or generated benchmark results. Derive reported tables from retained records.

The comparison is ready to support a performance claim only when:

- The GPU baseline completes functional checks and has measured timing with
  identified hot stages.
- The original REASON mapping has a supported-operation inventory, priced
  fallbacks, configuration provenance, and a calibrated timing backend.
- The extended backend preserves the output contract and accounts for added
  state, control, and interface costs.
- End-to-end timing includes communication, setup/amortization, and contention
  according to the stated deployment; kernel-only gains are labelled as such.
- Speedups, slowdowns, and changes in proof work or quality are all reported.

At project creation, source wiring and this protocol are available. The missing
REASON backend, GPU execution environment, workload mapping, datasets and
checkpoints are execution prerequisites, not completed experiments.
