# Literature screen: what is specifically dynamic about DPrL?

Primary sources checked on 2026-10-09. This replaces the earlier broad
neural/symbolic-feedback taxonomy; that version is preserved in
[archived/1009](archived/1009/LITERATURE.before_scope_correction.md).

**Finding:** there is an established lineage of goal-directed neural logic
provers. Our targeted screen identifies six mechanism-near papers including
DPrL, with substantial differences between them. It does **not** establish a
recent boom in DPrL-equivalent workloads, common measured bottlenecks, or a new
hardware contribution. CogSys and CoCoSys are material prior art, not exclusions
from the comparison. See the [overlap audit](PRIOR_ART.md).

## Admission criteria

The initial target is **online construction and selection of first-order proof
successors**, with three properties:

1. State includes goals and variable substitutions, not just a vector of scores
   over a previously instantiated circuit or an existing graph vertex.
2. Resolving a goal instantiates new subgoals or substitutions during the proof.
   Their values influence subsequent matching and state access.
3. Learned scores, a learned selector, or a proof heuristic affect which of
   those derivations are explored. The location and frequency of neural calls
   must be recorded; these need not equal DPrL's successor-scoring policy.

These are screening criteria, not a claim that the underlying ideas are new.
Prolog-style unification, indexes, and state management have long histories.
Variable candidate *cardinality*, variable goal *content*, and a variable
physical kernel graph are separate properties. Fixed-width templates can have
dynamic content; a padded GPU program can execute dynamic proof semantics with
a fixed kernel graph. DPrL's measured 20-round implementation is such a program.

## The narrow lineage

The machine-readable [focused inventory](focused_papers.tsv) records the
mechanism, limits, source location, and code where identified. Venue years are
shown here; the older broad inventory uses first-public years.

| Paper | Mechanism relevant to the screen | Difference that prevents treating it as identical to DPrL |
|---|---|---|
| **DPrL**, AAAI 2026 | Exact SLD successor goals are embedded and scored in context | Our implementation is a bounded greedy rollout; no measured tree frontier or branch rollback |
| **DPLA***, KR 2021 | A*-like search over partial SLD proofs; neural fact probabilities and a parametric heuristic | Closest exact-resolution relative, but best-proof approximation and probabilistic semantics differ from DPrL's derivation distribution |
| **GNTP**, AAAI 2020 | Retrieved promising matches drive recursive proof construction and substitutions | Soft unification and a configured top-k; not DPrL's exact legal candidate set |
| **CTP**, ICML 2020 | Goal-conditioned selectors produce rules/subgoals recursively | Continuous predicates and fixed reformulation templates; dynamic content does not demonstrate variable physical fan-out |
| **ACTP**, NeSy 2022 workshop | A learned policy selects reformulators during recursive proof expansion | Fixed action menu and selected-path budget; especially close in control, less close in exact candidate semantics |
| **NLProlog**, ACL 2019 | External backward-chaining prover with substitutions and similarity-based proof pruning | Neural similarity/weak unification, not a newly called DPrL policy for every successor set |

Primary sources, respectively: [DPrL §§2–4](https://arxiv.org/html/2511.08581v2),
[DPLA* §3](https://proceedings.kr.org/2021/45/kr2021-0045-manhaeve-et-al.pdf),
[GNTP, proving and efficient-inference sections](https://ojs.aaai.org/index.php/AAAI/article/view/5962/5818),
[CTP §3, Algorithm 2](https://arxiv.org/html/2007.06477v3),
[ACTP §§2–3](https://ceur-ws.org/Vol-3212/paper5.pdf), and
[NLProlog §§3–4](https://aclanthology.org/P19-1618.pdf).

**Count conservatively:** two exact-resolution examples including DPrL, plus
four soft/weak-unification or reformulation relatives. These six are not six
implementations of the same dynamic data path. The original
[NTP, NeurIPS 2017](https://papers.nips.cc/paper/6969-end-to-end-differentiable-proving)
is a seventh, foundational entry, kept outside the six: recursive construction
and symbolic bindings alone do not establish selective online execution, since
the original method enumerates proof alternatives.

Candidate systems need application profiles before claiming shared hardware
needs. The [follow-up verification](DYNAMIC_VALIDATION.md) now runs DPLA*'s
official successor/heuristic path on synthetic clauses and profiles CTP's
official GPU execution with frozen random weights. DPLA* confirms related
binding semantics, while CTP shows a different, largely fixed expansion
schedule. Neither establishes a second shared end-to-end state-construction
bottleneck; ACTP remains paper evidence only. Published algorithms are evidence
for a possible recurring mechanism, not evidence for its fraction of runtime.

## Exclusions and boundary cases

| Example | What is shared | Why it cannot establish the same hardware bottleneck |
|---|---|---|
| CogSys VSA workloads | Iteration, neural/symbolic dependence, changing values | Vector-symbolic binding is not an exact logical substitution environment |
| DeepProbLog exact compiled inference | Neural facts and logic | Circuit evaluation differs from DPLA*'s proof-search path; specify inference mode |
| DeepStochLog / DPrL MNIST control | Logic derivations and neural probabilities | Tabling or manageable-state dynamic programming can amortize proof construction |
| MINERVA / DeepPath | State-dependent choice of graph edges | Usually traverses an existing KG adjacency structure; not rule-driven construction of residual goals and substitutions |
| LeanDojo, HTPS, AlphaProof | New proof obligations and online search decisions | Elaborators, tactic execution, and model sizes differ; shared scheduling does not imply shared unification cost |
| AlphaGeometry | New constructions and evolving symbolic deductions | A real structural-dynamism relative, but a different deduction engine; also already in CoCoSys's characterization scope |
| Plain LLM chain of thought | Autoregressive dependence | Does not by itself execute an explicit symbolic state transition |

These distinctions classify execution paths, not entire frameworks forever.
They should be revised if an implementation supplies contrary evidence. In
particular, an exact-inference control and an approximate-search target can
come from the same DeepProbLog project.

## Breadth and trend: claims we can support

The original [36-paper inventory](papers.tsv) remains useful as **broad context
only**. Its proof-method category means neural-guided proof research, not
membership in the narrow target above. The previous 24-method subtotal and
recent-year histogram must not be used as evidence that DPrL-style dynamic
unification is widespread. Nor does a count of all NeSy papers establish that.

The selected close lineage spans 2019–2026 venue years, with a 2017 predecessor.
It demonstrates that DPrL is not an isolated algorithm. It does not show that
this narrow family is a dominant direction or that its publication rate has
recently accelerated. Recent formal-proving activity provides motivation to
test transfer, not permission to assume it.

Search procedure: start from DPrL's resolution contract; search neural theorem
proving, goal-conditioned rule selection, adaptive proof-path selection, and
approximate DeepProbLog inference; inspect algorithms rather than abstracts
alone. Separately inspect CogSys §VI-B and the public CoCoSys author manuscript.
Count conference/preprint versions once. This is a targeted screen, not a
systematic census or a novelty clearance.

## What must be measured across admitted workloads

Record (a) attempted matches versus retained successors, (b) goal atoms and
substitution updates per transition, (c) bytes materialized versus bytes whose
values actually change, (d) independent ready contexts, and (e) latency in
matching, state construction, neural scoring, and synchronization. Include
configured top-k, depth limits, and approximation semantics.

Our proposed common unit is a **goal plus a binding-dependent state update**.
Whether that unit is useful for hardware is an experimental question. If the
other systems mostly spend time in embedding similarity, external Prolog, or
large neural generation, DPrL's measured dense-state writes do not generalize.
