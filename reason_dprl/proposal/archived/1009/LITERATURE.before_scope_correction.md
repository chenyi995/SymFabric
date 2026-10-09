# Literature scope and evidence — searched 2026-10-09

The relevant class is **neural-guided search with explicit symbolic state
transitions**. DPrL's knowledge-graph prover is an instance: a current goal
determines valid derivations, a neural policy selects an action, and symbolic
execution creates the next proof state. Related systems differ substantially
in the symbolic operation and in how often they invoke a neural model.

## How many papers, and what does the count mean?

The [auditable inventory](papers.tsv) contains **36 distinct papers**:

| Category | Count | Interpretation |
|---|---:|---|
| Neural-guided proof methods | 18 | Direct evidence for a recurring neural/search/formal-execution loop |
| Adjacent search and generation workloads | 6 | Related control structure in knowledge graphs, planning, synthesis, and constrained SQL |
| Neighboring methods / controls | 4 | Relevant NeSy or formal-proving work that should not automatically be assigned the same runtime |
| Systems, runtimes, and architecture studies | 6 | Prior art and required software/hardware comparisons |
| Surveys | 2 | Broader context; not additional target workloads |

Thus this search identified **24 concrete method/workload papers in the direct
or adjacent groups**, of which 18 concern proof construction. The 36-paper total
is not a claim that 36 systems execute DPrL's unification kernel. It is also not
a census or an estimate of the total literature.

Searches used DPrL and REASON as seeds, then neural theorem proving,
policy-guided search, proof-assistant feedback, execution-guided synthesis,
neuro-symbolic GPU runtimes, and citations from the retrieved papers. Entries
were checked against author manuscripts, proceedings, or publisher pages.
Preprints and venue versions are counted once; HOList/DeepHOL and
LeanDojo/ReProver each contribute one paper. Versioned systems with separate
papers, such as AlphaGeometry/AlphaGeometry2, contribute two. Years in the
inventory refer to the first public paper, not a later conference year.

The 18 direct proof papers have the following first-public-year distribution:

| Period | Papers in this selected inventory |
|---|---:|
| 2016-2020 | 3 |
| 2021-2022 | 3 |
| 2023 | 3 |
| 2024 | 4 |
| 2025 | 5 |

Twelve of the 18 appeared from 2023 onward. This is evidence of sustained recent
activity in the selected lineage, **not a statistically unbiased growth rate**.
DPrL first appeared in 2025 and was accepted at AAAI 2026. TreeThink is a July
2026 systems preprint, not an established conference result. Sources and dates
are linked in the inventory.

## Is this an important direction?

It is a well-established and active approach **within automated reasoning**,
with recent systems in major ML venues and substantial formal-mathematics
results. Representative milestones include
[TacticZero](https://proceedings.neurips.cc/paper/2021/hash/4dea382d82666332fb564f2e711cbc71-Abstract.html),
[LeanDojo](https://arxiv.org/abs/2306.15626),
[AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5), and
[AlphaProof](https://www.nature.com/articles/s41586-025-09833-y).
These establish research relevance; they do not establish that stochastic
logic programming dominates NeSy or that the class dominates AI compute.

A broader systematic review screened 1,428 candidate NeSy papers and included
167 published in 2020-2024 under its relevance, peer-review, and code-availability
criteria. That supports a substantial surrounding research community. **167
must not be presented as the number of DPrL-like dynamic proof workloads.**
[Systematic review](https://arxiv.org/abs/2501.05435).

The recent change is not the invention of search. Neural policies have become
stronger, formal feedback is used repeatedly during search, and search budgets
have become a system design concern. AlphaProof describes saved formal states,
batched neural requests, parallel prover threads, and dynamically expanded
proof trees. TreeThink explicitly offers asynchronous search and batched model
execution. These are direct signs that execution and orchestration deserve
attention, while also demonstrating substantial software prior art.
[AlphaProof methods](https://www.nature.com/articles/s41586-025-09833-y),
[TreeThink](https://arxiv.org/abs/2607.11258).

## Shared structure, and limits to generalization

| Shared property | DPrL instance | Broader instance | Architectural implication to test |
|---|---|---|---|
| State-dependent actions | Applicable facts and rules depend on the current goal and bindings | Valid tactics, retrieved premises, legal graph edges | Online indexing and compact work queues |
| Neural/symbolic feedback | Score a derivation, then resolve the selected goal | Propose a tactic, execute it, observe new goals | Frequent boundary crossings; independent requests can overlap |
| Uneven work | Variable candidate count and proof depth | Uneven tactic cost and search branching | Throughput and tail latency need different batching policies |
| Persistent symbolic state | Goals, substitutions, history and termination | Proof contexts, tree frontiers, caches and certificates | State locality and reclamation can matter more than arithmetic |
| Conditional progress | Next query state depends on the previous action | Children depend on verified parent transitions | A single path remains serial; concurrency comes from independent work |

These implications are **our synthesis**, not measurements for every listed
system. An SLD unifier, a Lean elaborator, a knowledge-graph adjacency lookup,
and a geometric deduction engine are not interchangeable hardware kernels.
The common scheduling interface may generalize more widely than the data path.
LLM-heavy provers may remain dominated by neural generation; a small DPrL MLP
cannot establish their symbolic fraction.

DeepProbLog, DeepStochLog, and Scallop are useful controls because compiled
circuits, tabling, or fixed-point evaluation can avoid a fine-grained
neural-policy feedback loop. DeepSeek-Prover-V2's training-time subgoal
decomposition alone does not establish such a loop at inference. Plain
chain-of-thought generation without an executing symbolic environment is
outside the proposed target class.

## Prior art that constrains the proposed contribution

REASON already supports dynamic SAT behavior: watched-literal traversal,
implication queues, pipelined memory fetch, and conflict cancellation. A proposal
called only “make REASON dynamic” would misstate this prior art. The unproven
extension is efficient **goal-dependent proof-state creation, binding, and
neural-guided continuation**, rather than the existence of a queue.
[REASON, Sections V-D/V-E](https://arxiv.org/html/2601.20784v1).

Atos already implements dynamic irregular task scheduling on GPUs. TreeThink
implements asynchronous formal proof search in software. These must be treated
as design precedents and comparison points, not reasons to assume GPUs cannot
execute the proposed workload.
[Atos](https://arxiv.org/abs/2112.00132),
[TreeThink](https://arxiv.org/abs/2607.11258).

The architecture characterization literature supports investigating heterogeneous
control and memory behavior, but does not prove that every symbolic kernel is
bandwidth-bound. Our own profiles must determine the relevant limit.
[Cross-platform characterization](https://arxiv.org/abs/2409.13153).
