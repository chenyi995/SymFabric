# Testing the binding-dependent state hypothesis

2026-10-09. This follows the narrowed question in [PRIOR_ART.md](PRIOR_ART.md).

**Decision:** binding-dependent successor construction is real, but the current
evidence does not establish a common hardware bottleneck or justify a dynamic
REASON extension. GPU software removes substantial DPrL work while preserving
its outputs. A DPLA* execution probe confirms related semantics, whereas CTP
uses a materially different execution representation. These are different
levels of evidence, not three equivalent accelerator workloads.

## What was tested

The new GPU adapter changes the fact stage of the pinned DPrL KG engine:

1. Inspect fact IDs and bindings, without expanding every fact into a complete
   successor tensor.
2. Test whether the remaining goals constitute a proof in registers. Check
   **every fact in the original capped domain**, including matches beyond the
   first-K retention limit. DPrL treats such an overflow proof as success.
3. Keep the same first-K matching fact IDs in the same original order.
4. Materialize only those retained successors. Reuse the existing rule stage,
   ground-fact pruning, variable renaming, history, and policy code.

`compact` scans the original predicate prefix. `indexed` additionally builds a
fixed-KB lookup for predicate and bound arguments, preserving the original
shuffled/capped prefix and order. The index does not contain future decisions.
Its construction cost is reported separately; every online lookup, match,
proof check, state write, policy call and completion synchronization is timed.

This is an inference adapter for the recorded non-parity, ground-fact, zero-pad
configuration. It does **not** implement a persistent binding-delta store,
incremental embeddings, a hardware engine, or a REASON timing model. Those
unimplemented mechanisms must not receive credit for its measured speedup.

The source remains at `cfc66233c7aca3677cfb4ae74a71fdf9559ed89c`. The adapter
replaces one section of the pinned function at runtime and saves the generated
function and hash. It leaves the upstream checkout unchanged. This deliberate
dependency on the exact upstream function is part of the reproduction contract.

## DPrL measurements

Real Slurm RTX A5000, batch 64, 20 proof rounds, strict FP32/TF32 off, unchanged
short profiling checkpoints, three disjoint frozen query batches, 20 completed
timing repetitions per batch. Values below are **measured means of batch
medians**, not single-query latency. The previous best implementation is rerun
in the same experiment. Its query and answer hashes also match the previous
artifact. These are performance experiments, not converged-accuracy claims.

| Dataset | Previous best GPU | Descriptor scan | Argument index + descriptors | Best additional speedup |
|---|---:|---:|---:|---:|
| Family | 14.933 ms | **9.926 ms** | 10.215 ms | **1.50x** |
| WN18RR | 24.530 ms | **11.457 ms** | 11.669 ms | **2.14x** |

Family's argument index is slightly slower than the descriptor scan. Reducing
lookup candidates does not automatically reduce complete latency: the lookup
has a cost, and the scan already avoids the large dense state writes. WN18RR
also gains no measured benefit from the argument index in this comparison.

Family is a direct fact-stage ablation: both controls keep a fixed query batch.
The WN18RR comparison is **best previous configuration versus best new
configuration**: the previous runner compacts active queries, while both new
variants use a fixed batch. WN18RR's 2.14x therefore includes removing active
query packing; it must not be attributed entirely to fact materialization.
The preceding fixed-query WN18RR baseline was 26.649 ms in a separate run,
recorded in [GPU_RESULTS.md](GPU_RESULTS.md), not this paired timing table.

The argument-index build takes 0.0619 / 1.0315 seconds for Family / WN18RR and
stores 1,572,048 / 3,297,736 bytes in its lookup arrays, in addition to the
identity map and temporary descriptors. These measured build times include
host construction and transfer; compilation/capture and startup are outside
the steady-state table. No index-update performance for a changing KB is
claimed.

All active actions and retained persistent proof-state tensors match the
corrected native reference at every tested transition. Valid logits use
`atol=2e-4, rtol=2e-5`; maximum observed error for either dataset is 0.000152587890625, the
same scale as the preceding neural compaction. Five repeated output hashes per
batch/variant agree. Separate GPU cases cover proof beyond K, exclusion,
repeated variables, padding arguments, no match, rule application, and terminal
states, with and without excluded queries. This is tested equivalence on the
recorded inputs and edge cases, not a formal proof for all configurations.

The old logical fact-state tensor is `[B,F,A-1,3]` int64. The replacement is
`[B,K,A-1,3]`, plus fact flags/IDs and register computations. Family reduces
that one tensor from 19,530,240 to 998,400 bytes per derivation; WN18RR reduces
it from 61,440,000 to 2,012,160 bytes. These are
**derived logical sizes**, not measured DRAM traffic or total memory savings;
the retained rule/state pipeline still uses padded tensors.

### What remains after the change

Separate Nsight Systems captures of the winning descriptor-scan configuration
attribute kernels to the NVTX phase of their launch API, including graph
nodes. These are **measured summed GPU kernel times**, not complete latency:

| Captured kernel work | Family | WN18RR |
|---|---:|---:|
| Environment transitions | 3.260 ms (40.9%) | 5.625 ms (58.6%) |
| Compact policy phase | 4.429 ms (55.6%) | 3.575 ms (37.2%) |
| All kernels | 7.962 ms | 9.603 ms |
| Kernels shorter than 10 microseconds | 1,585 / 1,757 | 1,626 / 1,820 |

The policy phase includes feature gathers, graph-input copies, compaction and
MLP work; it is not an MLP-only number. The environment still performs fact
membership/pruning, masks, variable renaming and retained-state/history
updates. Removing attempted full states does not remove these stages.

Each capture includes 20 four-byte device-to-host count transfers for online
neural compaction and 290 device-to-device copies (24.48 / 48.79 MB). The
resident proof state is not sent to the CPU each round. Small launches,
shape synchronization, graph staging and retained padded-state processing
remain software targets. No new hardware-counter study proves a bandwidth
ceiling for this version.

Instrumentation stretches the capture windows to 42.33 / 41.35 ms; these are
**not** the performance results. The uninstrumented table above is the latency
evidence. Kernel-time fractions are not replaceable fractions of end-to-end
time and cannot directly be inserted into an accelerator Amdahl estimate.

## Does a small state delta imply reusable neural features?

Across active pre-selection states, only **3.47% / 0.162%** of Family / WN18RR
predicate-prefix fact attempts pass argument matching before exclusion.
This explains why constructing all attempted successors wastes work.

Only **0.574% / 1.079%** of the retained non-padding child atoms are
byte-identical to an atom in their parent in Family / WN18RR, counting multiset
intersection. Padding is excluded.
Bindings and upstream variable standardization change embedding indices.
The embedding implementation directly indexes those variable IDs; renaming
cannot simply be ignored while claiming the same frozen policy.

This weakens the proposed *direct parent-atom feature cache*. It does not
disprove predicate/constant feature reuse, sibling sharing, a representation
with explicit renaming deltas, or a redesigned policy. We have not measured an
incremental feature implementation. The raw goal-length distributions and
match/reuse counts are retained in each `benchmark.json`.

## Independent implementation checks

### DPLA*: the semantic mechanism transfers, the bottleneck remains unmeasured

Cloned [official DeepProbLog](https://github.com/ML-KULeuven/deepproblog) at
`f64181f66a3a789a96a47d7c2d3fae4d56178990`. Executed its actual
[`get_node_child` / `get_all_node_children`](https://github.com/ML-KULeuven/deepproblog/blob/f64181f66a3a789a96a47d7c2d3fae4d56178990/src/deepproblog/engines/prolog_engine/prolog_files/engine_heap.pl)
using SWI-Prolog 8.4.2, synthetic clauses, and a deterministic test heuristic.
The probe does not replace the engine's unification, `findall`, or callback
placement. It does not reproduce the paper's trained neural heuristic.

For `path(X,Y) :- edge(X,Z), edge(Z,Y)`, the test KB contains
`edge(a,b), edge(a,d), edge(b,c), edge(d,e)`:

| Query | Bound successor goals after the next fact step | Heuristic inputs |
|---|---|---|
| `path(a,c)` | `[edge(b,c)]`, `[edge(d,c)]` | `edge(b,c)`, `edge(d,c)` |
| `path(b,c)` | `[edge(c,c)]` | `edge(c,c)` |

The instantiated goals, branch count, and neural-interface arguments depend
on live bindings. This supports the narrow semantic class described by the
[DPLA* paper](https://proceedings.kr.org/2021/45/kr2021-0045-manhaeve-et-al.pdf).
However, the implementation uses Prolog terms and `findall`, not DPrL's dense
fact-by-goal tensor. No DPLA* application profile establishes that copying or
binding updates dominate end-to-end latency. This is a **semantic probe**, not
a second measured GPU bottleneck or a hardware speedup experiment.

### CTP: data-dependent values with a largely fixed expansion schedule

Cloned [official CTP](https://github.com/uclnlp/ctp) at
`8cd7e639024f2aa28bdae4783e940cad784580ae`. Ran its unmodified `BatchHoppy` and
`BatchNeuralKB` on an A5000, using 32 bundled Kinship facts, batch 16, embedding
width 32, top-k 2, and random frozen weights (seed 42). This is a mechanism
probe, not CLUTRR accuracy reproduction or an ACTP implementation.

| Depth / reformulators | Eager latency | Fixed-query CUDA Graph | Fact score / fact forward / reformulator calls |
|---|---:|---:|---:|
| 0 / 1 | 0.300 ms | 0.048 ms | 1 / 0 / 0 |
| 1 / 1 | 1.885 ms | 0.309 ms | 2 / 1 / 1 |
| 2 / 1 | 6.310 ms | 1.522 ms | 4 / 5 / 4 |
| 2 / 2 | 18.498 ms | 4.910 ms | 9 / 16 / 12 |

These are measured medians of 20 repetitions. Captured outputs exactly match
the eager output. The graph control has fixed resident inputs, so it is not a
complete streaming-serving comparison. Changing query values leaves the
recorded operation count/shape sequence unchanged for all four configurations.
Values and top-k entity identities remain data dependent; this does not mean
all semantic reasoning is static.

The source propagates subgoal embeddings, top-k entity choices, and dense RBF
comparisons. It repeats/concatenates fact and embedding tensors. At depth 2 /
two reformulators, the instrumented trace contains 699 GPU kernels and 4.926 ms
of summed kernel time. CPU operator attribution assigns about 1.050 ms of GPU
work to `copy_` and 0.784 ms to `cat`; neither is evidence of DPrL-style exact
substitution-state construction. Profiler overhead is excluded from the timing
table. The depth-zero profiler file has no CUDA events and cannot support a
kernel-time claim.

### ACTP: a relevant paper, not an executed implementation here

No identifiable official ACTP implementation was recovered in this search.
Do not call the CTP probe an ACTP reproduction. The
[ACTP paper, Appendix C](https://ceur-ws.org/Vol-3212/paper5.pdf) reports that
copying/masking/selection overhead makes ACTP-8C3 slower than CTP-3, although
it beats CTP-8. Its evidence concerns the cost of selecting fewer expansion
paths; it does not isolate exact first-order successor-state materialization.

## Consequence for the proposal

The narrow semantic phenomenon appears in DPrL and DPLA*. There is currently
only **one end-to-end profiled implementation** of the proposed exact-state
bottleneck. Two DPrL datasets do not increase that count. CTP/ACTP should remain
comparisons with different representations, rather than evidence of identical
hardware demand. This experiment establishes neither prevalence nor a recent
growth trend for the narrow workload.

The successful GPU change is delayed materialization and exact work reduction.
Indexing, filtering, fusion, and avoiding unnecessary copies are existing
software techniques. This result does not establish novelty against
CogSys/CoCoSys, nor demonstrate that their programmable designs cannot perform
the operation. See the [prior-art boundary audit](PRIOR_ART.md).

**Proceed with GPU runtime work and workload validation before hardware.**
A hardware proposal needs a second real application with trained models,
measured binding/state costs, and an advantage remaining after equivalent
GPU indexing, fusion and persistent execution. Direct parent-atom caching
also needs a different reuse argument than the one tested here. A finite
state/binding engine remains a research possibility; the current evidence is
insufficient to make it the chosen architecture.

Reproduction commands and artifact identities are in
[DYNAMIC_REPRODUCE.md](DYNAMIC_REPRODUCE.md).
