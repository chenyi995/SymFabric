# Optimized DPrL GPU inference — 2026-10-09

The strongest tested strict-FP32 GPU variants reduce complete batch latency
from **46.09 to 14.98 ms for Family (3.08x)** and from **95.28 to 24.62 ms for
WN18RR (3.87x)** relative to the matched complete-CUDA-Graph controls. Latencies
are measured; speedups are derived ratios. These improvements use software
work compaction, graph replay, and corrected candidate-packing kernels.

## Measurement contract

- Slurm `athena/node2`, one NVIDIA RTX A5000 24 GB per job, PyTorch
  `2.9.0+cu130`, CUDA 13, eight allocated CPUs, four PyTorch compute threads.
- Frozen checkpoint, inference only, strict FP32, TF32 disabled, 64 resident
  queries, unchanged 20-step greedy rollout budget. No candidate pruning,
  precision reduction, altered search order, or precomputed query trajectory.
- Three disjoint test batches per dataset, 20 warm completed runs per variant
  per batch. Main tables use the **mean of the three per-batch medians**; this
  is not a pooled latency percentile or a confidence interval.
- Timed scope: resident queries through completed output tensors, including
  online index discovery, host shape synchronization, graph input copies, and
  final device synchronization. Data loading, checkpoint loading, compilation,
  graph creation, and disk output are excluded. Cold-start inference and request
  arrival/queueing are not measured here.
- Both native and optimized paths use deterministic, stable first-K candidate
  semantics. WN18RR retains the upstream configured 8,000 fact-pair cap and 262
  candidate slots per query; Family retains 130 slots. Neither is a claim of
  exhaustive unrestricted theorem search.
- The checkpoints were trained for profiling (32,768 transitions), not to
  reproduce the paper's converged accuracy. All comparisons keep their weights
  fixed. Extrapolation to other checkpoints, GPUs, or search distributions needs
  additional measurements.

Primary jobs are `220687` (WN18RR) and `220688` (Family). Full sample arrays,
configuration, source/checkpoint hashes, query hashes, per-round work counts,
and correctness checks are in
[WN18RR records](../results/20261009-gpu-opt/wn18rr_v3/) and
[Family records](../results/20261009-gpu-opt/family_v3/).

## End-to-end results

| Variant | Family, measured ms | WN18RR, measured ms |
|---|---:|---:|
| Corrected compiled native GPU | 52.396 | 97.978 |
| Same computation in a complete CUDA Graph | 46.092 | 95.280 |
| Compact neural inputs + fast candidate packing + per-shape graphs | **14.975** | 26.649 |
| Also compact active queries with eager state gathers | 23.900 | 28.867 |
| Also compile/capture the active-query state gathers | 19.590 | **24.623** |

Relative to compiled native inference, the best variants improve latency by
3.50x and 3.98x respectively. Relative to the stronger complete-graph controls,
the gains are 3.08x and 3.87x. One fixed variant is selected for each dataset;
there is no per-query oracle choosing the fastest implementation.

![Measured latency comparison](../results/20261009-gpu-opt/analysis/latency.png)

| Dataset and chosen variant | Batch 0 median / p95, ms | Batch 1 median / p95, ms | Batch 2 median / p95, ms |
|---|---:|---:|---:|
| Family, compact neural inputs with fixed query batch | 14.771 / 14.821 | 15.344 / 16.194 | 14.811 / 14.892 |
| WN18RR, compact queries with compiled state gathers | 25.407 / 25.996 | 22.731 / 23.251 | 25.732 / 26.542 |

The individual WN18RR speedups over matched graph controls are 3.74x, 4.19x,
and 3.71x. The reported 3.87x is the ratio of mean batch medians, not the best
single-batch speedup. Twenty timing samples per batch give only a preliminary
tail-latency estimate.

## What changed

1. **Neural work compaction.** Discover valid candidates for active queries from
   live GPU state. Gather only those rows and round their count to a power-of-two
   bucket. Combine current and candidate states in the shared embedding/MLP
   path, then scatter their logits back to the original action positions.
   Candidate order and masking semantics are retained.
2. **Graph replay for changing shapes.** Cache CUDA Graphs by tensor shape and
   stride. Copy current inputs into static buffers for every invocation. Query
   and candidate counts remain online decisions; their synchronization cost is
   included. This is not replay of cached proof answers.
3. **Candidate-packing kernel.** Compute valid-entry prefix ranks, write only
   uniquely owned retained positions, and gather directly from fact/rule state
   tensors. Invalid entries do not contend on an atomic sink; large concatenated
   state intermediates are avoided. The output is exactly the stable first-K
   reference result.
4. **Active-query compaction and its cost.** Bucket unfinished queries, carry
   their original IDs, and preserve completed outputs. This reduces environment
   work but adds index discovery, state gathering, and copying. Compiling these
   gathers recovers part of that overhead. Family still prefers the fixed query
   batch; WN18RR benefits from the fully optimized compact-query variant.

The first isolated neural-compaction experiment on Family reduced its graph
control from 46.370 to 35.245 ms (mean of three batch medians, measured). This
alone was a 1.32x gain. The later table combines additional mechanisms and is
the primary comparison. Not every incremental optimization improves complete
latency, even when it reduces the number of arithmetic operations.

## Correctness and repeatability

Every optimized variant in the primary jobs passed all of the following:

- Every active argmax action equals the corrected native action at every round.
- Every retained real query's current/derived states, counts, next-variable
  index, history hashes/count, termination, success, and depth match exactly.
- Final outputs match, and five additional repeated output hashes per batch
  and variant agree with the corrected native result.
- Valid logits satisfy `atol=2e-4, rtol=2e-5`; the maximum observed absolute
  difference was `0.000152587890625`. FP32 batching can change rounding, while
  the tested action and proof trajectories remained identical.

The new packing kernel also passed eight GPU cases, eager and compiled, with
five repetitions each: empty inputs, no valid entries, complete validity,
overflow, differing atom dimensions, noncontiguous strides, and a 64-query
8,000-fact input. See job `220686` and
[the checks](../scripts/check_fast_candidates.py). The existing 13 functional
tests passed with the project Python environment. These checks do not establish
accuracy parity on the entire dataset or on converged checkpoints.

Expanded WN18RR testing exposed the same upstream colliding-scatter issue as
Family. Earlier uncorrected WN18RR numbers around 112 ms are therefore **not the
control for the new speedup claim**. The corrected control is approximately
95 ms in these jobs. The failed exploratory run is retained rather than used
as a favorable baseline.

## Where time remains: phase diagnostics

The following are means of separately instrumented CUDA-event phase spans over
the three batches, **not additive components of the uninstrumented medians**.
They include scheduling gaps and diagnostic event overhead. In particular,
“compact policy” includes index discovery, shape synchronization, input copies,
and graph dispatch as well as actual neural kernels.

| Phase, measured diagnostic span | Family best variant, ms | WN18RR best variant, ms |
|---|---:|---:|
| Reset and initial candidate generation | 0.430 | 1.018 |
| Active-index discovery | Not performed | 2.628 |
| Active proof-state packing | Not performed | 4.295 |
| Compact policy path | 6.074 | 7.149 |
| Argmax selection | 0.152 | 0.223 |
| Environment transitions and next candidates | 9.499 | 11.789 |
| Completed-output update | Included in fixed state | 0.146 |

The environment remains important after removing dense neural padding. For
WN18RR, index discovery plus state packing costs about 6.9 ms in the instrumented
path, which explains why simply dropping finished queries was insufficient.
These spans alone cannot prove that those milliseconds require new hardware;
further kernel fusion, indexing, and persistent GPU execution remain plausible.

## Nsight: remaining device work versus orchestration

Nsight Systems jobs `220690` and `220691` captured the selected Family and
WN18RR variants on the first batch. CUDA Graph **node tracing** was enabled.
Kernels are attributed to the NVTX phase enclosing their launch API using CUDA
correlation IDs, so asynchronous graph execution is not lost when it outlives
the CPU phase. See [attribution analysis](../scripts/analyze_optimized_profile.py)
and [raw derived records](../results/20261009-gpu-opt/analysis/profile_attribution.json).

| Captured quantity, measured | Family | WN18RR |
|---|---:|---:|
| GPU kernel count | 1,735 | 2,028 |
| Kernels shorter than 10 microseconds | 1,435 | 1,825 |
| Sum of kernel durations | 13.083 ms | 15.251 ms |
| Environment kernels | 8.115 ms / 62.0% | 10.393 ms / 68.1% |
| Compact-policy kernels, including index processing | 4.458 ms / 34.1% | 3.234 ms / 21.2% |
| SGEMMs alone, subset of policy | 2.299 ms / 17.6% | 1.388 ms / 9.1% |
| Device-to-host shape-count copies | 20 / 80 bytes | 40 / 160 bytes |
| Device-to-device graph-input copies | 290 / 24.479 MB | 764 / 47.845 MB |

Percentages use **summed kernel time**, not complete inference time. The capture
windows were 23.68 and 77.22 ms, compared with unprofiled first-batch medians of
14.77 and 25.41 ms. The tracer and diagnostic events materially perturb CPU
dispatch, particularly for WN18RR. Capture-window idle fractions and CUDA API
durations must not be presented as unprofiled GPU utilization or added to the
primary benchmark latency.

The source-level sequence behind the largest remaining environment kernels is:

1. Enumerate predicate-matching fact/rule pairs and perform unification.
2. Materialize substituted remaining goals for the fact-pair capacity, then
   overwrite invalid candidates with padding. The fused `index_put` kernel
   takes **1.187 ms in Family and 1.699 ms in WN18RR** across 20 transitions.
3. Check whether remaining atoms form a valid ground-fact proof, excluding the
   original query. The corresponding fused reduction takes **0.944 / 1.376 ms**.
4. Materialize a True-state replacement for proven candidates. This fused
   conditional write takes **1.148 / 1.636 ms**.
5. Merge/pack fact and rule candidates, update variable/state metadata, and
   perform history checks before the next policy decision. The new packing
   implementation accelerates the merge; it does **not** yet eliminate the
   earlier padded fact-state materialization.

The first four actions are visible in the pinned
[unification source](../../softwareworkload/upstream/DPrL-AAAI/kge_experiments/unification.py)
(`unify_with_facts`, `all_atoms_are_ground_facts`, and `_derive_step_impl`).
[Saved generated kernels](../results/20261009-gpu-opt/kernel_sources/) support
the correspondence. These three named hotspots are not an exhaustive
decomposition of the environment.

Nsight Compute jobs `220692` and `220693` sampled 12 selected launches per
dataset. The counter means below describe **those launches only**; replay changes
cache/clock conditions. The Family masking row has only one sample.

| Selected kernel | Dataset / sampled launches | SM throughput | DRAM throughput | Achieved occupancy |
|---|---|---:|---:|---:|
| Invalid fact-state padding write | Family / 1 | 15.3% | 89.1% | 81.7% |
| Invalid fact-state padding write | WN18RR / 6 | 9.8% | 91.6% | 85.5% |
| Ground-fact proof reduction | WN18RR / 6 | 15.9% | 78.4% | 33.3% |
| Compact-policy `32x32` SGEMM | Family / 11 | 39.3% | 5.5% | 18.7% |

The padding write is bandwidth intensive even at high occupancy. Its input
volume is still an implementation choice: indexing and compacting earlier in
unification could reduce it on a GPU too. The compact MLP now has smaller GEMMs
and lower occupancy; the sampled GEMM has 0.75 waves per SM. Removing padding
can decrease utilization while improving complete latency.

WN18RR's captured active-state gather kernels total only 0.153 ms, while its
separately instrumented packing span is about 4.3 ms. Input copying and host
dispatch therefore matter in addition to gather arithmetic. The graph-input
copies in the table are **device-to-device**, not proof states transferred to
the CPU. Tiny shape-count transfers still synchronize execution. A persistent
GPU runtime could remove much of this adapter overhead; it is not a fundamental
property of GPU hardware.

The remaining optimization priorities are consequently: **earlier indexed or
compact candidate generation; fewer state materializations; and fewer host
round trips**. More neural matrix throughput alone targets a substantially
smaller fraction of the optimized workload.

## Reproduce

For a fresh checkout, frozen-checkpoint recovery, and automatic verification,
start with [REPRODUCE.md](../REPRODUCE.md). The commands below expose the
individual benchmark and profiler entry points.

Run from the repository root with the existing project virtual environment and
the recorded checkpoints. Choose fresh output directories. Dataset-specific
local caches avoid the disappearing shared `/dev/shm` cache files encountered
in exploratory jobs. Those compilation failures are excluded from timing.

```bash
sbatch --export=ALL,SYMFABRIC_HARNESS_ARCHIVE=,SYMFABRIC_CACHE_BASE=/tmp/symfabric-opt3-family \
  --output=reason_dprl/results/NEW-family.log \
  reason_dprl/scripts/run_gpu.sbatch reason_dprl/scripts/optimize_kg_fast.py \
  --dataset family \
  --checkpoint reason_dprl/results/20261009-a5000/family_train_stable_shm/checkpoint.pt \
  --output reason_dprl/results/NEW-family --batches 3 --repeat 20

sbatch --export=ALL,SYMFABRIC_HARNESS_ARCHIVE=,SYMFABRIC_CACHE_BASE=/tmp/symfabric-opt3-wn \
  --output=reason_dprl/results/NEW-wn.log \
  reason_dprl/scripts/run_gpu.sbatch reason_dprl/scripts/optimize_kg_fast.py \
  --dataset wn18rr \
  --checkpoint reason_dprl/results/20261009-a5000/wn18rr_train_a5000/checkpoint.pt \
  --output reason_dprl/results/NEW-wn --batches 3 --repeat 20

python3 reason_dprl/scripts/summarize_optimizations.py \
  reason_dprl/results/NEW-family reason_dprl/results/NEW-wn \
  --output reason_dprl/results/NEW-analysis
```

To profile the selected WN18RR path (use `fast_policy_graphs` for Family):

```bash
sbatch --export=ALL,SYMFABRIC_CACHE_BASE=/tmp/symfabric-opt3-wn \
  --output=reason_dprl/results/NEW-nsys.log \
  reason_dprl/scripts/profile_optimized.sbatch nsys reason_dprl/results/NEW-nsys \
  reason_dprl/scripts/optimize_kg_fast.py --dataset wn18rr \
  --checkpoint reason_dprl/results/20261009-a5000/wn18rr_train_a5000/checkpoint.pt \
  --output reason_dprl/results/NEW-profile-run --batches 1 --repeat 3 \
  --capture-only fast_fused_graphs
```

Use `ncu` instead of `nsys` and set `NCU_KERNEL_REGEX` to sample a chosen kernel;
the runner limits collection to 12 launches in `native_query/`. Profiler-run
benchmark timings are not the primary performance results. Counter summaries
are saved in [profiles_with_counters.json](../results/20261009-gpu-opt/analysis/profiles_with_counters.json).

Source snapshots are retained as `harness-v1.tar.gz`, `harness-v2.tar.gz`, and
`harness-v3.tar.gz` under the [experiment root](../results/20261009-gpu-opt/).
Primary jobs ran the repository scripts directly; their exact file hashes are
recorded in provenance. Large run outputs and plots remain ignored by Git.

The next software comparisons should include argument-indexed candidate
generation, a persistent GPU loop that avoids host shape synchronization, and
lower precision with application-level quality checks. **The best tested
variant is not a demonstrated GPU ceiling.**
