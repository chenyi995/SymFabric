# Prior-art overlap and the remaining hypothesis

Audit date: 2026-10-09. The earlier broad proposal overlaps substantially with
existing work. A different benchmark name, or the word "dynamic", is not a
hardware contribution.

## CogSys: a concrete boundary, with real overlap

[CogSys](https://arxiv.org/html/2503.01162v1), HPCA 2025, targets VSA workloads.
Its §V describes GEMM/circular-convolution modes and vector SIMD support.
Crucially, **§VI-B explicitly implements adSCH offline in host software**:
the operation graph and resource mapping are computed from known operations,
sizes, dependencies, and iteration counts. It also interleaves independent
neural/symbolic tasks and partitions array resources.

Our inference: an online compact DPrL execution would need to discover work
from live bindings rather than receive its future compact work graph offline.
The paper does not demonstrate that first-order state path. This is a boundary
of the published evaluation, **not proof that its programmable hardware cannot
run such a path**. A padded DPrL schedule can itself be fixed. A fair comparison
must allow template reuse and charge runtime work discovery on every backend.

Cross-query overlap, adaptive partitioning, and a shared neural/symbolic engine
are already covered ideas. They cannot be the headline novelty of this project.

## CoCoSys: broader than VSA, and a direct warning against our old framing

The public [CoCoSys author manuscript](https://zishenwan.github.io/publication/IEEE_Micro_2026_CoCoSys.pdf)
is a center-wide overview, not one accelerator. The retrieved version has 16
pages and a Nov/Dec 2026 issue label; as of this audit it is an available author
version. Pages 7–10 discuss compositional workloads, nested reasoning pipelines,
speculative symbolic execution, operator fusion, coordinated caching, CogSys,
REASON, and heterogeneous silicon. Figure 5 includes AlphaGeometry.

Thus it already addresses more than static vector operations. Its cited
[ASPLOS 2026 study](https://doi.org/10.1145/3760250.3762235), *Compositional AI
Beyond LLMs*, is a mandatory comparison. The ACM full text was inaccessible
during this audit; statements above about those optimizations come from the
accessible CoCoSys manuscript, not a claimed independent inspection of ASPLOS
implementation details. The [authors' center summary](https://cocosys.ece.gatech.edu/2026/04/28/cocosys-spotlight-best-2025-q3-publication/)
also identifies complex control and data movement as characterization targets.

We cannot establish novelty by claiming CoCoSys omitted dynamic reasoning.
An experiment must isolate a concrete state-generation mechanism beyond the
already proposed fusion, speculation, caching, and heterogeneous execution.

## REASON and software precedents

[REASON §§V-D/V-E](https://arxiv.org/html/2601.20784v1) already handles SAT
watch-list traversal, propagation queues, overlapping memory access, and
conflict cancellation. SAT has evolving execution state. Calling all REASON
workloads "static DAGs" is incorrect. Exact first-order goal substitution is a
different operation; the existence of that difference is not a novelty proof.

[Atos](https://arxiv.org/abs/2112.00132) is a GPU dynamic-task precedent.
[TreeThink](https://arxiv.org/abs/2607.11258) is a 2026 preprint concerning
asynchronous proof search. Neither software queues nor asynchronous neural
batching can be assumed unavailable to a GPU baseline. Prolog indexing and
variable environments, database joins, and copy-on-write state are additional
prior-art families that must be screened before an architectural novelty claim.

## Claim-by-claim disposition

| Candidate claim | Disposition | What would make a stronger, testable claim |
|---|---|---|
| Neural and symbolic kernels have different resource needs | Already established; overlaps CogSys/CoCoSys | No novelty claim |
| Interleave independent queries or add queues | Existing scheduling idea | Treat as baseline functionality |
| Keep state near compute; reduce copying | Broad existing design principle | Specify state format, ownership, updates, finite capacity, and cost |
| Support runtime work creation | REASON SAT and GPU runtimes already do this | Identify the cost of binding-dependent successor construction specifically |
| Build successors as compact descriptors and binding deltas | Research hypothesis; indexing and delta storage themselves are not new | Prove exact semantics and incremental feature computation; measure saved bytes versus extra indirections |
| Hardware is necessary because GPU is slow | Unsupported by current data | Beat indexed/persistent GPU execution at a specified latency, throughput, energy, and area budget |

## Proposed experiment, before choosing hardware

Use a bounded descriptor such as `(parent_context, rule_or_fact, substitution)`
instead of eagerly copying every candidate's complete goal tensor. Match via
predicate/argument indexes, construct only required changes, and reuse atom
features when their instantiated arguments are unchanged. Commit the selected
state, preserving candidate order, history semantics, and the proof budget.

The [follow-up experiment](DYNAMIC_VALIDATION.md) now implements delayed
fact-state materialization and an optional argument index on the GPU, with
unchanged outputs. It does not implement a persistent binding-delta store or
incremental neural features. The complete hypothesis's novelty and
transferability remain unproven.
Bindings may alter many atoms; caching may add more traffic than it saves;
numerical aggregation may change rounding and hence action selection. Charge
all of these costs. Do not replace DPrL's history hashing with a different
algorithm or hide a changed candidate set behind an approximate policy.

Evaluate the same representation first on the GPU, then in a finite-resource
symbolic state engine. Measure input indexes, context allocation, update reads
and writes, neural feature materialization, and full completion latency.
The follow-up executes a semantic probe of DPLA*'s official Prolog engine and
an official CTP GPU execution probe. DPLA* confirms live goal/binding changes;
CTP uses embedding propagation and largely fixed expansion shapes in the tested
configurations. Neither establishes DPrL's materialization cost as a shared
end-to-end bottleneck. ACTP has paper evidence only. Family and WN18RR remain
two datasets of one implementation, not two independent systems.

The initial GPU profile established expensive dense intermediate writes after
downstream compaction. The follow-up removes much of that work in GPU software,
improving previous best batch latency by 1.50x / 2.14x. Its very low direct
parent-atom reuse also weakens the simplest incremental-feature hypothesis.
Pursue the software result and measure transfer before hardware. Hardware needs
an additional measured residual and a calibrated benefit; it does not follow
from membership in NeSy or from our uncalibrated REASON reconstruction.
