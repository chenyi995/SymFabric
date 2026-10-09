# sym1 — workload and operator-decomposition evidence handbook

Companion to the [research plan](README.md), complementing its step-by-step interface description. [Complete Chinese counterpart](../local/sym1-workload-operator-evidence.zh.md).

The handbook connects **workload literature → native code and its work → a native example** with **decomposition literature → lowering code and its work → the same example's operator trace**. Each workload has both chains. A mathematical mapping is not evidence of a hardware speedup.

## Reading and evidence boundaries

This is written for a reader encountering these papers for the first time: identify the input and answer, define terms, follow the computation, then read a small example. Descriptive names replace advanced notation.

Sources have four distinct roles: **paper basis** for established definitions/mechanisms; **local mapping/code inspection** for this implementation; **teaching derivation** for constructed inputs checked by a script; **bottleneck hypothesis** for a claim needing measurement. Every lowering step names its source; paper locators are at the workload opening or the D entry. Local limitations and teaching examples are not presented as published results.

Vocabulary: a join pairs records with matching fields; a reduction combines a group; a frontier is the set to expand in this round; host code controls loops, grouping and readout outside the primitives; a golden is an independently computed reference answer; a query is an independent request; a token is a sequence element; dynamic programming stores and reuses answers to smaller problems. Each workload supplies the specific meaning and sources.

- “Direct” means a source gives the relevant recurrence or execution mechanism; “local derivation” means we apply that foundation to this repository's schemas/operators. No single cited paper establishes every operator in this interface.
- Source links below pin upstream code to the revisions in [the lock file](upstream_lock.json). They establish inspected code, not that the paper's original artifact revision has been reproduced or its benchmark run.
- Native code blocks are original explanatory pseudocode; lowering is written as a step-by-step interface mapping. Neither is a verbatim paper/code excerpt. Real function links are separate. For/each means visit each item, while means continue under a condition, return names the answer, and brackets select an entry. Papers support mechanisms; linked code defines exact fields, ties and state protocols.
- All example inputs are **constructed teaching fixtures**, not collected datasets. Their output values are **derived**, using [documentation_examples.py](scripts/documentation_examples.py), independent scalar/set calculations and actual local algebra primitives. Some examples expose only a key round; their scope is stated. They are not full-workload regressions or benchmark results.
- The script fills and rechecks identical data blocks in both language versions. Run the command below for a read-only check; add --write to refresh generated blocks. Neither command runs simulators, upstream baselines or GPUs.

~~~bash
python3 sym1/scripts/documentation_examples.py
~~~

## A. Literature that actually supports operator decomposition

Read the locator, not just the title. In the workload sections, D references identify the exact foundation used.

<a id="d1"></a>

### D1. [Green, Karvounarakis & Tannen, Provenance Semirings](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf)

Locator: §3, Definitions 3.1–3.2; §5. Annotated tuples; join multiplies, projection/union adds. Positive algebra does not by itself define our antijoin, top-k or mutable state.

Plain-language meaning and limits: Think of each database row as carrying a number or a truth value. Matching two rows combines their values; several matches contributing to the same answer must then be combined again. The paper supplies that meaning, but it does not decide how a hash table, queue or hardware lane is built. [D1 · Provenance Semirings](#d1).

<a id="d2"></a>

### D2. [Aji & McEliece, The Generalized Distributive Law](https://www.cs.ubc.ca/~murphyk/Teaching/Papers/GDL.pdf)

Locator: §II–IV. Marginalize a product of functions through distributivity. Exactness needs the stated structure/schedule conditions.

Plain-language meaning and limits: A factor is a table assigning a weight to a combination of variable values. Many queries multiply matching factors and add over values that are not returned. Distributivity lets shared partial results be reused. The message-passing schedule must satisfy the paper's conditions; arbitrary cyclic iteration is not guaranteed exact. [D2 · The Generalized Distributive Law](#d2).

<a id="d3"></a>

### D3. [Abo Khamis, Ngo & Rudra, FAQ: Questions Asked Frequently](https://arxiv.org/abs/1504.04044)

Locator: FAQ definition; InsideOut; equivalent variable orderings. Eliminate variables by join and aggregation; not all mixed-aggregate reorderings are legal.

Plain-language meaning and limits: A variable can be removed by grouping records on the remaining variables and combining its possible values. Choosing which variable to remove first changes the size of the temporary tables. When some groups use addition and others use maximum, order changes need an explicit legality argument. [D3 · FAQ: Questions Asked Frequently](#d3).

<a id="d4"></a>

### D4. [Goodman, Semiring Parsing](https://aclanthology.org/J99-4004.pdf)

Locator: §1, Figures 1–3; semiring framework. Deduction antecedents multiply; alternative proofs add; inside versus best derivation changes the value algebra.

Plain-language meaning and limits: A parsing rule combines several smaller pieces into a larger piece. Multiply the contributions needed together; add the contributions of alternative parses, or choose the best if the requested answer is a best parse. Recording the chosen parse still needs additional identity and traceback information. [D4 · Semiring Parsing](#d4).

<a id="d5"></a>

### D5. [Eisner, Goldlust & Smith, Compiling Comp Ling: Practical Weighted Dynamic Programming and the Dyna Language](https://aclanthology.org/H05-1036.pdf)

Locator: §4.2 and Figure 3. Indexed chart/agenda, change propagation, and ordered old/delta/current contributions. Its increment is not automatically our changed-total delta.

Plain-language meaning and limits: A chart stores partial answers; an agenda lists partial answers whose change may affect other answers. Propagate a change only to relevant rules. The paper gives a useful incremental-evaluation foundation, but its increments must not be confused with this repository's table of changed total values. [D5 · Dyna: Practical Weighted Dynamic Programming](#d5).

<a id="d6"></a>

### D6. [Mohri, Semiring Frameworks and Algorithms for Shortest-Distance Problems](https://cs.nyu.edu/~mohri/pub/jalc.pdf)

Locator: Generic-Single-Source-Shortest-Distance; topological variant. Path extension, aggregation and queue discipline are separate; termination depends on graph/algebra assumptions.

Plain-language meaning and limits: Separate what a path score means from which pending item is processed next. Adding edge costs and choosing the minimum is one instance; other value rules are possible under their assumptions. Queue order, termination and handling of improved values remain part of the algorithm. [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6).

<a id="d7"></a>

### D7. [Abo Khamis et al., Convergence of Datalog over (Pre-) Semirings](https://arxiv.org/abs/2105.14435)

Locator: Datalog-over-semiring framework; stability/convergence results. Justifies auditing convergence assumptions; it is not a universal finite-termination theorem.

Plain-language meaning and limits: Having reusable add/combine rules does not by itself make a recursive program finish. The value domain and the allowed improvements matter. In particular, setting an iteration cap is a runtime policy, not a mathematical proof that a final answer has been reached. [D7 · Convergence of Datalog over (Pre-) Semirings](#d7).

<a id="d8"></a>

### D8. [GraphBLAS C API Specification](https://graphblas.org/docs/GraphBLAS_API_C_v2.1.0.pdf)

Locator: `GrB_mxm`, `GrB_vxm`, reduction, mask/accumulator semantics. Sparse contraction, structural masks and accumulation have an established interface; our arbitrary-arity relations are a local generalization.

Plain-language meaning and limits: This item is an API standard, not a research paper. It is a concrete precedent for sparse matrices, declared value operations, masks and accumulation. A structural mask asks whether an entry is present, not how large its value is. Named tables with arbitrary fields still require the local mapping described here. [D8 · GraphBLAS C API Specification (standard)](#d8).

<a id="d9"></a>

### D9. [Jordan, Scholz & Subotić, Soufflé: On Synthesis of Program Analyzers](https://souffle-lang.github.io/pdf/cav16.pdf)

Locator: §2 and Figure 1. Datalog is lowered through relational algebra into executable code; local join order still needs a work comparison.

Plain-language meaning and limits: A rule-based program is translated into a relational execution plan, then into executable code. This supports the claim that rules can be decomposed into concrete data operations. It does not establish that every join order is equally efficient or that the local Python interpreter is an optimized Soufflé baseline. [D9 · Soufflé: On Synthesis of Program Analyzers](#d9).

<a id="d10"></a>

### D10. [Hsu et al., The Sparse Abstract Machine](https://arxiv.org/html/2208.14610v2)

Locator: primitive definitions; sparse tensor/dataflow examples. Physical lowering into coordinate streams, intersections, value operations and reductions; not a proof of all sym1 control/state support.

Plain-language meaning and limits: Sparse inputs contain coordinates as well as values. Coordinate-processing units discover which entries meet; arithmetic units operate on those values; reduction units combine contributions belonging to one output. End-of-group markers preserve grouping. This is a physical decomposition precedent, not an implementation of all our stateful control. [D10 · The Sparse Abstract Machine](#d10).

<a id="d11"></a>

### D11. [Huang & Chiang, Better k-best Parsing](https://aclanthology.org/W05-1506.pdf)

Locator: k-best derivation algorithms. Derivation identity and structured candidate generation matter. Our batch select is not their complete algorithm.

Plain-language meaning and limits: Finding several best derivations requires preserving which choices produced each derivation and generating the right candidates. Sorting a batch of scores is only one piece. A fixed-width beam may discard a prefix needed by a later globally best answer, so beam search is not this exact algorithm. [D11 · Better k-best Parsing](#d11).

<a id="d12"></a>

### D12. [Felzenszwalb & McAllester, The Generalized A* Architecture](https://arxiv.org/abs/1110.2216)

Locator: weighted deduction and generalized A*. Connects prioritized search and inference; our grid mapping additionally relies on its own consistency/tie/state assumptions.

Plain-language meaning and limits: A prioritized inference system chooses what to expand according to accumulated cost and a remaining-cost estimate. It supplies a bridge between search and deduction. The actual grid heuristic, tie rule and settled-state policy still need to be checked in the local implementation. [D12 · The Generalized A* Architecture](#d12).

<a id="d13"></a>

### D13. [Budiu et al., DBSP: Automatic Incremental View Maintenance for Rich Query Languages](https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf)

Locator: incremental/differential operators. A contrast for additive differences and repeated computation. Do not identify DBSP differences with update's new total values.

Plain-language meaning and limits: An additive difference describes how much an output changes when an input changes. It can include cancellation and removal. A table containing each changed row's new total answers a different question. Citing this paper therefore helps state a boundary; it does not turn the local update into DBSP. [D13 · DBSP: Automatic Incremental View Maintenance](#d13).

## B. Individual operators, explained without abstract notation

A relation is a table. Each row has identifying fields, called its key, and a value. A schema lists those field names. Support means the rows that are present. A missing row does not always mean the number zero: for shortest paths it means unreachable; a stored zero-cost path can be valid.  (Mechanism sources: [D1 · Provenance Semirings](#d1), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6); exact behavior: [relation.py](../algebra/relation.py).)

The [plan's operator section](README.md) gives all implementation qualifications. The key code is [ops.py](../algebra/ops.py), [relation.py](../algebra/relation.py) and [control.py](../algebra/control.py).

| Code name | What it does | Literature supporting the mechanism |
|---|---|---|
| combine | Match shared fields, combine matching values, then combine contributions with equal output keys | [D1 · Provenance Semirings](#d1), [D2 · The Generalized Distributive Law](#d2), [D3 · FAQ: Questions Asked Frequently](#d3) |
| reduce / project | Group by retained fields and combine values within each group | [D1 · Provenance Semirings](#d1), [D3 · FAQ: Questions Asked Frequently](#d3) |
| filter / rename | Keep rows satisfying a condition / change field names without changing values | [D1 · Provenance Semirings](#d1) |
| mask | Keep rows with a matching key in another table; negate=True keeps unmatched rows | [D8 · GraphBLAS C API Specification (standard)](#d8); arbitrary-field matching is a local extension |
| scan | Apply a small, bounded transformation to each row and combine duplicate output keys | Local row-map interface; not Torch-Struct's parallel prefix scan |
| update | Merge candidates into saved state and return the actual changed rows | [D5 · Dyna: Practical Weighted Dynamic Programming](#d5), [D7 · Convergence of Datalog over (Pre-) Semirings](#d7), [D9 · Soufflé: On Synthesis of Program Analyzers](#d9); our returned values are new totals |
| select_k | Keep k best rows; k is the requested row count | Ordered-search context: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6), [D11 · Better k-best Parsing](#d11), [D12 · The Generalized A* Architecture](#d12); the exact primitive is local |
| iterate / fixpoint | Repeat a fixed number of times / until no valid new change remains | [D5 · Dyna: Practical Weighted Dynamic Programming](#d5), [D7 · Convergence of Datalog over (Pre-) Semirings](#d7), [D9 · Soufflé: On Synthesis of Program Analyzers](#d9) |

**How combine works.** Imagine a table of known paths and a table of edges. Match each path's end with an edge's start and form a longer path. For reachability, existence is enough. For shortest paths, add the component costs and keep the smallest candidate for each destination. For probability, multiply compatible contributions and add alternatives under the model's probability rules. The semiring terminology in the papers refers to these paired calculation rules; it is not an extra operator.  (Mechanism sources: [D1 · Provenance Semirings](#d1), [D2 · The Generalized Distributive Law](#d2), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6); exact behavior: [ops.py](../algebra/ops.py).)

The implementation hashes matching fields, visits compatible row pairs, calculates a value and merges it into the output group. With no matching fields, every row pairs with every row on the other side. A different algorithm, such as sorted-list intersection, may implement the same answer with different work.  (Mechanism sources: local interface definition; exact behavior: [ops.py](../algebra/ops.py).)

**Important differences between the operators.**

- Mask checks existence, not whether a new score is better, and does not multiply by the second table's weights. Update can accept an improved score for a key that already exists.  (Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](#d8); exact behavior: [ops.py](../algebra/ops.py).)

- Update's changed table, also called delta, stores the new total value rather than the difference from the old one. Floating-point comparisons use a tolerance. Cancellation can leave a saved zero without producing a deletion notification. See [D13](#d13) for a different additive-difference framework.  (Mechanism sources: [D13 · DBSP: Automatic Incremental View Maintenance](#d13); exact behavior: [ops.py](../algebra/ops.py).)

- A product of mandatory children must include every child. A reduce with use_mul=True visits only stored children; a missing zero child can be incorrectly skipped. The circuit example uses separately gathered child slots to avoid that mistake.  (Mechanism sources: [D2 · The Generalized Distributive Law](#d2); exact behavior: [ops.py](../algebra/ops.py).)

- Scan changes one row at a time with bounded work and bounded output. It cannot hide a global sort or an arbitrary search. Rename must avoid colliding field names.  (Mechanism sources: local interface definition; exact behavior: [ops.py](../algebra/ops.py).)

- Select_k is global batch selection. Per-query grouping, rank calculation, a persistent priority queue and path reconstruction require additional work. Score ties use the full identifying key.  (Mechanism sources: [D11 · Better k-best Parsing](#d11); exact behavior: [ops.py](../algebra/ops.py).)

- A fixed iteration limit is not a proof of convergence. Probability and counting must not repeatedly add old contributions.  (Mechanism sources: [D7 · Convergence of Datalog over (Pre-) Semirings](#d7); exact behavior: [control.py](../algebra/control.py).)

**How a workload is assembled.** Name its input tables and saved state; list each operator's inputs and outputs; say whether to execute once, repeat a fixed number of steps, repeat until no change, or repeatedly process the best pending item; finally name the returned answer. A DAG is an operation-dependency graph without a cycle. Feedback means feeding changed rows into the next round. Ordered search has a separate declared stop condition.  (Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](#d5), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6), [D9 · Soufflé: On Synthesis of Program Analyzers](#d9); exact behavior: [control.py](../algebra/control.py).)

**Reading the examples.** Input is the constructed teaching input; native is the independently calculated answer; result is the operator result. Other lines show intermediate tables. Parentheses hold a row's identifying fields; the value after the colon is its attached value. True means a fact exists; an empty table contains no rows; an empty key () stores one scalar answer. The field order and remaining short names are explained beside each example. Fractions are exact derived values, not measured performance. Source and check: [documentation_examples.py](../sym1/scripts/documentation_examples.py).

### Lowering further to physical primitives

Combine is a semantic contraction, not one indivisible hardware operator. [D10 · The Sparse Abstract Machine](#d10) provides a concrete sparse lowering precedent:


[SAM `Intersect2.update`](https://github.com/weiya711/sam/blob/6e477d142668fa0a769235365f5c77a049440297/sam/sim/src/joiner.py) → [SAM `Multiply2.update`](https://github.com/weiya711/sam/blob/6e477d142668fa0a769235365f5c77a049440297/sam/sim/src/compute.py) → [SAM `Reduce.update`](https://github.com/weiya711/sam/blob/6e477d142668fa0a769235365f5c77a049440297/sam/sim/src/accumulator.py).

For a sparse dot product, align matching coordinates, multiply matched values, reduce within a fiber (one coordinate group), and emit a token marking its boundary. Missing coordinates produce no product; reductions still need correct empty-fiber semantics. These primitives do not supply hash construction, persistent priority queues, conflict handling, or arbitrary fixpoint termination for free. They are a code-level precedent for physical decomposition, not a completed sym1 datapath.

## C. Per-workload evidence chains

- [datalog_tc: Linear transitive closure](#datalog_tc)
- [datalog_samegen: Same-generation recursive join](#datalog_samegen)
- [datalog_nonlinear_tc: Nonlinear transitive closure](#datalog_nonlinear_tc)
- [triangle_count: Triangle counting](#triangle_count)
- [sat_bcp: SAT Boolean constraint propagation](#sat_bcp)
- [kg_multihop: Weighted knowledge-graph paths](#kg_multihop)
- [bayes_net_ve: Bayesian-network variable elimination](#bayes_net_ve)
- [prob_circuit: Arithmetic / probabilistic circuit evaluation](#prob_circuit)
- [automata_rpq: Regular path query](#automata_rpq)
- [beam_search: Beam expansion and pruning](#beam_search)
- [fuzzy_inference: Max-min fuzzy relational closure](#fuzzy_inference)
- [hmm_viterbi: HMM forward and Viterbi inference](#hmm_viterbi)
- [ctrlg_dfa_hmm: DFA-constrained HMM guidance](#ctrlg_dfa_hmm)
- [vsa_cleanup: VSA bind, bundle, unbind and cleanup](#vsa_cleanup)
- [astar_grid: A* on a grid](#astar_grid)
- [planning_hadd: Relaxed-planning h-add and h-max](#planning_hadd)
- [cky_parsing: CKY inside / best-score parsing](#cky_parsing)
- [graph_bfs_sssp: BFS and single-source shortest paths](#graph_bfs_sssp)

<a id="datalog_tc"></a>

### C.1. datalog_tc — Linear transitive closure

#### Workload literature and mathematical task

**Papers and reading locations:** [Sun et al., Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf) — §1 reachability rules; §2 semi-naive evaluation and Figure 1.

**Input, calculation and answer.** Input: directed graph edges. Output: every start/end pair connected by a nonempty path. A vertex reaches itself only if a cycle provides such a path. Definition checked against [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf); local scope: [Local native](../softwareworkload/datalog_tc/workload.py#L43).

**Terms, conventions and pitfalls.** A recursive rule reuses its own earlier answers. Closure means that applying the rule again produces no new reachable pairs. This is semi-naive evaluation: reuse previous answers and expand only newly discovered paths. Rounds depend on previous rounds; the independent native code checks the answer by graph traversal. Source roles: mathematical background: [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf); implementation details and limits: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

#### Native key code and work

[Local native](../softwareworkload/datalog_tc/workload.py#L43). `native` performs independent graph traversals from each source; `algebra` starts from edges and extends only newly discovered pairs. [GDlog `tc.dl`](https://github.com/harp-lab/gdlog/blob/65a6ee960ced8d04bc725ccdfa68f004f8479226/test/souffle/tc.dl) actually contains a nonlinear path–path rule, so it is not the same execution schedule as this linear local variant.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
For each source vertex:
    traverse the graph from its outgoing neighbors
    record every reached endpoint for that source
```

#### Workload example

The native chain traversal and join both discover the indirect pair. The displayed result is the complete non-reflexive closure for this acyclic input. Start at a and follow its outgoing edge to b. The edge leaving b reaches c, so the join discovers the indirect pair a-to-c. The update adds this pair to the original edges. There is no edge leaving c, so the next round adds nothing; result includes direct and indirect reachability.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf).

#### Decomposition literature and justification

- [D1 · Provenance Semirings](#d1) (§3, Definitions 3.1–3.2; §5): Annotated tuples; join multiplies, projection/union adds. Positive algebra does not by itself define our antijoin, top-k or mutable state.

- [D9 · Soufflé: On Synthesis of Program Analyzers](#d9) (§2 and Figure 1): Datalog is lowered through relational algebra into executable code; local join order still needs a work comparison.

- [D7 · Convergence of Datalog over (Pre-) Semirings](#d7) (Datalog-over-semiring framework; stability/convergence results): Justifies auditing convergence assumptions; it is not a universal finite-termination theorem.

**How the foundations connect to this program.** The rule body is an equijoin on the intermediate vertex; existential projection is Boolean reduction. Semi-naive evaluation limits the recursive input to the changed table, then removes known support. These mechanisms are established; the named-attribute mapping below is local. The precise fields, schedule and state operations below are a local mapping, checked against [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

#### Key lowering code and work

[Local algebra](../softwareworkload/datalog_tc/workload.py#L71); [detailed decomposition](../docs/decompositions/datalog_tc.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Initialize both the accumulated path table and the new-path table with the original edges. Mechanism sources: [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf), [D9 · Soufflé: On Synthesis of Program Analyzers](#d9). Interface mapping / exact execution: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

2. combine: match each new path's end with an original edge's start; retain the original path start and the edge's end. Mechanism sources: [D1 · Provenance Semirings](#d1). Interface mapping / exact execution: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

3. mask with negate=True: remove start/end pairs already in the accumulated table. Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](#d8), [D9 · Soufflé: On Synthesis of Program Analyzers](#d9). Interface mapping / exact execution: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

4. update: save genuinely new pairs and use them as the next new-path table. Mechanism sources: [D9 · Soufflé: On Synthesis of Program Analyzers](#d9). Interface mapping / exact execution: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

5. Stop when that new-path table is empty. Return the accumulated paths. Mechanism sources: [D7 · Convergence of Datalog over (Pre-) Semirings](#d7). Interface mapping / exact execution: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** E means the edge table. Letters a, b and c name vertices. A pair lists start then end; True means reachable. join is the candidate table and new_delta is the genuinely new subset.

<!-- EXAMPLE:datalog_tc -->
```text
input: E=[('a', 'b'), ('b', 'c')]
native: [('a', 'b'), ('a', 'c'), ('b', 'c')]
join: {('a', 'c'): True}
new_delta: {('a', 'c'): True}
result: {('a', 'b'): True, ('a', 'c'): True, ('b', 'c'): True}
```
<!-- /EXAMPLE:datalog_tc -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): Full all-pairs TC is not source-batched reachability with the same output size. Frontier size and duplicate candidates remain separate profiling targets. [Local algebra](../softwareworkload/datalog_tc/workload.py#L71); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="datalog_samegen"></a>

### C.2. datalog_samegen — Same-generation recursive join

#### Workload literature and mathematical task

**Papers and reading locations:** [Sun et al., Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf) — §2 same-generation rules and Figure 1.

**Input, calculation and answer.** Input: seed same-generation pairs and links between parents and children. Output: all pairs obtained by repeatedly pairing children whose parents are already known to be of the same generation. Definition checked against [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf); local scope: [Local _golden_seminaive](../softwareworkload/datalog_samegen/workload.py#L114).

**Terms, conventions and pitfalls.** A seed is a fact supplied before recursion. Same-generation here names the relation defined by these rules, rather than an independently inferred real-world family tree. Changing the join order preserves the Boolean answer, not necessarily the amount of work. Both naive and semi-naive native paths check the generated cases. The paper derives sibling seeds from shared parents; the local workload supplies Flat directly. They describe the same instance only when seeds agree. Source roles: mathematical background: [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf); implementation details and limits: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

#### Native key code and work

[Local _golden_seminaive](../softwareworkload/datalog_samegen/workload.py#L114). [GDlog `sg.dl`](https://github.com/harp-lab/gdlog/blob/65a6ee960ced8d04bc725ccdfa68f004f8479226/test/souffle/sg.dl) defines the sibling seed and recursive rule. Local `_golden_naive` / `_golden_seminaive` enumerate children; `_algebra` exposes two binary join orders, starting from the local `flat` seed.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
For each newly known same-generation parent pair:
    read the children of the first parent
    read the children of the second parent
    emit each matching child pair and discard already known pairs
```

#### Workload example

This example shows one propagation round, not an invented complete family tree. Add the emitted pair to the existing seed and continue if further descendants exist. Flat says that p and q already belong to the same generation. Looking up the child of p produces the temporary child/other-parent pair shown in up_join_delta. Looking up the child of q then replaces that remaining parent with b. result is the newly derived child pair, not the complete closure including seeds.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf).

#### Decomposition literature and justification

- [D1 · Provenance Semirings](#d1) (§3, Definitions 3.1–3.2; §5): Annotated tuples; join multiplies, projection/union adds. Positive algebra does not by itself define our antijoin, top-k or mutable state.

- [D3 · FAQ: Questions Asked Frequently](#d3) (FAQ definition; InsideOut; equivalent variable orderings): Eliminate variables by join and aggregation; not all mixed-aggregate reorderings are legal.

- [D9 · Soufflé: On Synthesis of Program Analyzers](#d9) (§2 and Figure 1): Datalog is lowered through relational algebra into executable code; local join order still needs a work comparison.

**How the foundations connect to this program.** The conjunction becomes two joins and the internal parent keys are projected away. Associativity permits this Boolean regrouping; it does not make intermediate cardinalities equal. [Soufflé `generateClauseVersions` / `generateRecursiveStratum`](https://github.com/souffle-lang/souffle/blob/a1303be3c0166400dee3d1f36f0d96abe03e6901/src/ast2ram/seminaive/UnitTranslator.cpp) is an upstream recursive-lowering reference. The precise fields, schedule and state operations below are a local mapping, checked against [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

#### Key lowering code and work

[Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147); [detailed decomposition](../docs/decompositions/datalog_samegen.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Initialize saved and newly discovered same-generation pairs from the seed table. Mechanism sources: [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

2. combine: pair each newly related parent pair with the children of its first parent. Mechanism sources: [D1 · Provenance Semirings](#d1). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

3. combine: match that intermediate table with the children of its second parent; retain the two children. Mechanism sources: [D1 · Provenance Semirings](#d1). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

4. Remove child pairs already saved, then update the saved and new-pair tables. Mechanism sources: [D9 · Soufflé: On Synthesis of Program Analyzers](#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

5. Repeat until no new pair is produced. Alternatively, join the second parent's children first; compare intermediate sizes. Mechanism sources: [D3 · FAQ: Questions Asked Frequently](#d3), [D9 · Soufflé: On Synthesis of Program Analyzers](#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** Up lists child then parent; Down lists parent then child; Flat contains seed same-generation pairs. The example's p and q are parents, a and b their children. up_join_delta retains the first child and the other parent.

<!-- EXAMPLE:datalog_samegen -->
```text
input: Up=[('a', 'p')]; Flat=[('p', 'q')]; Down=[('q', 'b')]
native: [('a', 'b')]
up_join_delta: {('a', 'q'): True}
join_down: {('a', 'b'): True}
result: {('a', 'b'): True}
```
<!-- /EXAMPLE:datalog_samegen -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): Measure the intermediate T and skewed child fan-out. Arithmetic sharing cannot remove combinatorial pair generation required by this rule. [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="datalog_nonlinear_tc"></a>

### C.3. datalog_nonlinear_tc — Nonlinear transitive closure

#### Workload literature and mathematical task

**Papers and reading locations:** [Sun et al., Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf) — §2 semi-naive evaluation; pinned GDlog tc.dl supplies the nonlinear rule.

**Input, calculation and answer.** Input and output are the same reachability question as linear transitive closure. The computation instead joins an already known path to another known path, so both sides can change between rounds. Definition checked against [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf); local scope: [Local native](../softwareworkload/datalog_nonlinear_tc/workload.py#L68).

**Terms, conventions and pitfalls.** Nonlinear here means that the same changing path relation occurs twice in a rule; it does not mean a neural-network nonlinearity. The first join already includes new paths on both sides. The general two-occurrence recursive rule needs both delta placements, even if a particular graph happens to yield the same pair from each. Boolean deduplication is not ordinary arithmetic counting. Source roles: mathematical background: [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf); implementation details and limits: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

#### Native key code and work

[Local native](../softwareworkload/datalog_nonlinear_tc/workload.py#L68). [GDlog `tc.dl`](https://github.com/harp-lab/gdlog/blob/65a6ee960ced8d04bc725ccdfa68f004f8479226/test/souffle/tc.dl) contains `path(from,mid), path(mid,to)`. Local `native` computes reachability independently; `_algebra(variant='two_term')` retains old state and expands both delta placements.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
For each source vertex:
    start a queue from its outgoing neighbors
    traverse each previously unseen reachable vertex
    record source and reached endpoint in the closure
```

#### Workload example

Both delta placements can produce the same output pair through different intermediate vertices. The trace explicitly shows that deduplication step and the empty terminating round. The first round joins input edges into longer paths. In the next round, a-to-d can be produced by either delta position. Both copies refer to the same reachability fact, so they are stored once. Empty new_delta ends the process; the trace keeps both expansion terms visible even where their answers overlap.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf).

#### Decomposition literature and justification

- [D5 · Dyna: Practical Weighted Dynamic Programming](#d5) (§4.2 and Figure 3): Indexed chart/agenda, change propagation, and ordered old/delta/current contributions. Its increment is not automatically our changed-total delta.

- [D9 · Soufflé: On Synthesis of Program Analyzers](#d9) (§2 and Figure 1): Datalog is lowered through relational algebra into executable code; local join order still needs a work comparison.

- [D7 · Convergence of Datalog over (Pre-) Semirings](#d7) (Datalog-over-semiring framework; stability/convergence results): Justifies auditing convergence assumptions; it is not a universal finite-termination theorem.

**How the foundations connect to this program.** The old/delta/current split comes from incremental rule evaluation, not from calling a whole closure one scan. [Soufflé `generateClauseVersions` / `generateRecursiveStratum`](https://github.com/souffle-lang/souffle/blob/a1303be3c0166400dee3d1f36f0d96abe03e6901/src/ast2ram/seminaive/UnitTranslator.cpp) generates rule versions for recursive occurrences. Local Boolean union safely absorbs repeated outputs; this cannot be transferred unchanged to ordinary counting. The precise fields, schedule and state operations below are a local mapping, checked against [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

#### Key lowering code and work

[Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92); [detailed decomposition](../docs/decompositions/datalog_nonlinear_tc.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Start the accumulated and new-path tables from the edges; start the old snapshot empty. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](#d5), [D9 · Soufflé: On Synthesis of Program Analyzers](#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

2. First combine: new paths on the left with current accumulated paths on the right. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](#d5), [D9 · Soufflé: On Synthesis of Program Analyzers](#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

3. Second combine: the old accumulated snapshot on the left with new paths on the right. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](#d5), [D9 · Soufflé: On Synthesis of Program Analyzers](#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

4. Merge those candidate tables using existence rather than counting; remove pairs already known. Mechanism sources: [D1 · Provenance Semirings](#d1). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

5. Save the current accumulated table as the old snapshot before updating it. Save genuinely new pairs for the next round. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](#d5), [D9 · Soufflé: On Synthesis of Program Analyzers](#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

6. Repeat until there are no new pairs; return all accumulated paths. Mechanism sources: [D7 · Convergence of Datalog over (Pre-) Semirings](#d7). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** E is the edge table. round is an iteration label. delta means newly discovered paths, current the accumulated paths now, and old the snapshot before those additions. Each pair is start then end.

<!-- EXAMPLE:datalog_nonlinear_tc -->
```text
input: E=[('a', 'b'), ('b', 'c'), ('c', 'd')]
native: [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')]
round_0_delta_join_current: {('a', 'c'): True, ('b', 'd'): True}
round_0_old_join_delta: {}
round_0_new_delta: {('a', 'c'): True, ('b', 'd'): True}
round_1_delta_join_current: {('a', 'd'): True}
round_1_old_join_delta: {('a', 'd'): True}
round_1_new_delta: {('a', 'd'): True}
round_2_delta_join_current: {}
round_2_old_join_delta: {}
round_2_new_delta: {}
result: {('a', 'b'): True, ('a', 'c'): True, ('a', 'd'): True, ('b', 'c'): True, ('b', 'd'): True, ('c', 'd'): True}
```
<!-- /EXAMPLE:datalog_nonlinear_tc -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): Do not equate fewer rounds with less work. Report candidate joins, deduplication and stored X separately. [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="triangle_count"></a>

### C.4. triangle_count — Triangle counting

#### Workload literature and mathematical task

**Papers and reading locations:** [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619) — benchmark-kernel definitions: triangle counting.

**Input, calculation and answer.** Input: an undirected graph without self-loops or duplicate edges. Output: the number of triangles. Order vertices consistently, then orient edges by that order so each triangle is counted once. Definition checked against [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619); local scope: [Local native](../softwareworkload/triangle_count/workload.py#L62).

**Terms, conventions and pitfalls.** A neighbor is a vertex joined by an edge. Intersection keeps vertices appearing in both neighbor lists. Orientation chooses a consistent direction only for counting; it does not change the original undirected task. The independent native algorithm intersects oriented neighbor lists and can avoid materializing wedges. A scalar result is represented by a row with no identifying fields. Source roles: mathematical background: [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619); implementation details and limits: [Local algebra](../softwareworkload/triangle_count/workload.py#L95).

#### Native key code and work

[Local native](../softwareworkload/triangle_count/workload.py#L62). [GAPBS `OrderedCount` / `Hybrid`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/tc.cc) intersects ordered adjacency lists; local `native` independently intersects oriented neighbor sets. Local `algebra` compares materialized wedges with fused endpoint aggregation.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
For each edge directed by the fixed vertex order:
    intersect the forward-neighbor sets of its endpoints
    add the number of common neighbors to the triangle count
```

#### Workload example

A single ordered triple survives the closing-edge mask. With several middle vertices, aggregation must preserve their count rather than Boolean support. The ordered edges allow the wedge a-b-c. Its endpoints a and c also have an edge, so the closing-edge mask retains it. The final reduction adds the retained counts. It is the fixed orientation, not Boolean deduplication alone, that prevents counting the same undirected triangle under every vertex order.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619).

#### Decomposition literature and justification

- [D1 · Provenance Semirings](#d1) (§3, Definitions 3.1–3.2; §5): Annotated tuples; join multiplies, projection/union adds. Positive algebra does not by itself define our antijoin, top-k or mutable state.

- [D3 · FAQ: Questions Asked Frequently](#d3) (FAQ definition; InsideOut; equivalent variable orderings): Eliminate variables by join and aggregation; not all mixed-aggregate reorderings are legal.

- [D8 · GraphBLAS C API Specification (standard)](#d8) (`GrB_mxm`, `GrB_vxm`, reduction, mask/accumulator semantics): Sparse contraction, structural masks and accumulation have an established interface; our arbitrary-arity relations are a local generalization.

- [D10 · The Sparse Abstract Machine](#d10) (primitive definitions; sparse tensor/dataflow examples): Physical lowering into coordinate streams, intersections, value operations and reductions; not a proof of all sym1 control/state support.

**How the foundations connect to this program.** The expression is a cyclic join with final counting aggregation; a closing-edge support mask may filter wedges. [LAGraph `LAGr_TriangleCount`](https://github.com/GraphBLAS/LAGraph/blob/7543de1f310371c540c23ff5da82705aa4e94ea0/src/algorithm/LAGr_TriangleCount.c) supplies an upstream sparse-algebra implementation with alternative triangle formulations. Our particular orientation/schema/fusion remains a local choice. The precise fields, schedule and state operations below are a local mapping, checked against [Local algebra](../softwareworkload/triangle_count/workload.py#L95).

#### Key lowering code and work

[Local algebra](../softwareworkload/triangle_count/workload.py#L95); [detailed decomposition](../docs/decompositions/triangle_count.md); primitive implementations: [ops.py](../algebra/ops.py).

1. combine: join two directed edges sharing a middle vertex; this creates a two-edge path, called a wedge. Mechanism sources: [D1 · Provenance Semirings](#d1), [D3 · FAQ: Questions Asked Frequently](#d3). Interface mapping / exact execution: [Local algebra](../softwareworkload/triangle_count/workload.py#L95).

2. mask: retain a wedge only if an edge also joins its start directly to its end. Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](#d8). Interface mapping / exact execution: [Local algebra](../softwareworkload/triangle_count/workload.py#L95).

3. reduce: add the surviving counts to obtain the triangle total. Mechanism sources: [D1 · Provenance Semirings](#d1). Interface mapping / exact execution: [Local algebra](../softwareworkload/triangle_count/workload.py#L95).

4. A fused alternative first counts wedges by their start/end pair, matches the closing edge and sums. It must retain counts, not replace them with existence flags. Mechanism sources: [D3 · FAQ: Questions Asked Frequently](#d3). Interface mapping / exact execution: [Local algebra](../softwareworkload/triangle_count/workload.py#L95).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** E is the oriented edge table. A wedge triple lists start, middle and end. closed contains wedges with a closing edge. The empty key () stores the final scalar count.

<!-- EXAMPLE:triangle_count -->
```text
input: oriented E=[('a', 'b'), ('a', 'c'), ('b', 'c')]
native: 1
wedges: {('a', 'b', 'c'): 1}
closed: {('a', 'b', 'c'): 1}
result: {(): 1}
```
<!-- /EXAMPLE:triangle_count -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): A materialized-wedge implementation can do more work than adjacency intersection. Fusion and work-efficient enumeration need separate comparisons. [Local algebra](../softwareworkload/triangle_count/workload.py#L95); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="sat_bcp"></a>

### C.5. sat_bcp — SAT Boolean constraint propagation

#### Workload literature and mathematical task

**Papers and reading locations:** [Moskewicz et al., Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf) — BCP section: two watched literals.

**Input, calculation and answer.** Input: clauses joined by AND, with alternatives inside each clause joined by OR, plus a partial truth assignment for each query. Output: forced assignments and whether a contradiction is found. If a clause is not already true and all but one of its literals are false, its remaining literal must be true. Definition checked against [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf); local scope: [Local _propagate_2wl](../softwareworkload/sat_bcp/workload.py#L86).

**Terms, conventions and pitfalls.** A literal is a variable or its negation. A unit clause has only one not-yet-false literal, which must become true; a conflict means no remaining assignment can satisfy a clause under the current decisions. A query is one independently propagated partial assignment. A watch is a watched clause literal used to avoid inspecting unaffected clauses. Static clauses can be shared; assignments, propagation history and watches are query-private. The local monotone program retains obsolete false watches; it is not a backtracking deletion protocol. Generated clauses have at least two literals and consistent initial assignments. General empty/unit input clauses need explicit initialization. Source roles: mathematical background: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf); implementation details and limits: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

#### Native key code and work

[Local _propagate_2wl](../softwareworkload/sat_bcp/workload.py#L86). [CaDiCaL `Internal::propagate`](https://github.com/arminbiere/cadical/blob/c60730422e758ef1cebe7aeddf2dda31c996bf04/src/propagate.cpp) follows the assignment trail and lazy watches. Local `_propagate_2wl` is the independent golden; `algebra_scan` is full-clause scanning; `algebra_watched` first identifies touched clauses, then rescans their literals.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
While there are unprocessed assignments:
    take the next true literal
    inspect clauses watching its now-false opposite
    move a watch, force the remaining literal, or report conflict
```

#### Workload example

The executable example exposes the full-scan count/filter core on binary clauses. In watched mode the first implication touches the next clause; this tiny case does not exercise relocating a watch to another non-false literal. The seed makes a true and therefore makes !a false. Clause c1 can now be satisfied only by b; making b true in turn makes !b false, so c2 forces c. The false-count rows explain why each implication fires. The final empty changes end propagation. This small case has no conflict and does not exercise watch relocation.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf).

#### Decomposition literature and justification

- [D1 · Provenance Semirings](#d1) (§3, Definitions 3.1–3.2; §5): Annotated tuples; join multiplies, projection/union adds. Positive algebra does not by itself define our antijoin, top-k or mutable state.

- [D9 · Soufflé: On Synthesis of Program Analyzers](#d9) (§2 and Figure 1): Datalog is lowered through relational algebra into executable code; local join order still needs a work comparison.

**How the foundations connect to this program.** These sources support joins, projection and relational execution, not a published theorem that Chaff equals our operators. The count/filter/anti decomposition is a local derivation of the unit-clause predicate. Watch relocation, conflicts and query freezing require explicit state/control. The precise fields, schedule and state operations below are a local mapping, checked against [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

#### Key lowering code and work

[Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265); [detailed decomposition](../docs/decompositions/sat_bcp.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Initialize true and false literal tables from the input assignment. A literal is a variable or its negation. Keep the initial two watched literals per query/clause and start with no conflicts. Mechanism sources: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

2. scan flips newly true literals to identify their newly false opposites. combine with the watch table and reduce by query/clause to locate touched clauses; also count how many watches fell false. Mechanism sources: [D1 · Provenance Semirings](#d1), [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

3. Read the literals of touched clauses. Exclude satisfied clauses, identify false literals with mask, and count them per clause. Mechanism sources: [D1 · Provenance Semirings](#d1). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

4. filter identifies unit clauses and all-false clauses. A unit clause has exactly one remaining non-false literal; project that literal as a forced assignment. Mechanism sources: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

5. For a touched clause needing replacement watches, exclude existing watches and select the smallest available literal identifier. If both watches fell false, choose another distinct replacement. Mechanism sources: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

6. Record conflicts from an all-false clause or contradictory new implications. Freeze conflicted queries; update other assignments and feed only new assignments into the next round. Mechanism sources: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

7. Stop when no new assignment remains. Return assignments and per-query conflict flags. Mechanism sources: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** c1 and c2 name clauses; !a means not a. false_counts is keyed by clause name and clause length, with the number of false literals as its value. new_assignments lists newly forced true literals. The example exercises scan-mode propagation, not watch relocation.

<!-- EXAMPLE:sat_bcp -->
```text
input: clauses={c1: ['!a', 'b'], c2: ['!b', 'c']}; seed=a; scan-mode teaching round
native: ['a', 'b', 'c']
round_0_false_counts: {('c1', 2): 1}
round_0_new_assignments: {('b',): True}
round_1_false_counts: {('c2', 2): 1}
round_1_new_assignments: {('c',): True}
round_2_false_counts: {}
round_2_new_assignments: {}
result: {('a',): True, ('b',): True, ('c',): True}
```
<!-- /EXAMPLE:sat_bcp -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): BCP is not #SAT/FAQ model counting. Initial empty/unit clauses, backtracking and complete CDCL need separate coverage; touched-clause rescanning need not match CaDiCaL's work. [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="kg_multihop"></a>

### C.6. kg_multihop — Weighted knowledge-graph paths

#### Workload literature and mathematical task

**Papers and reading locations:** [Zhu et al., Neural Bellman-Ford Networks](https://arxiv.org/abs/2106.06935) — §3.1 path formulation; Appendix A.5 most reliable path; §3.2 learned operators.

**Input, calculation and answer.** Input: weighted graph edges with relation labels, query sources, a maximum path length and a requested answer count. Output: the best allowed path score for each destination, then the requested best destinations. Multiply edge probabilities along a path and choose the largest product among paths reaching the same destination. Definition checked against [Neural Bellman-Ford Networks](https://arxiv.org/abs/2106.06935); local scope: [Local native](../softwareworkload/kg_multihop/workload.py#L67).

**Terms, conventions and pitfalls.** A knowledge-graph edge labels the kind of relationship as well as its endpoints. A hop is one edge traversal. Log-scores preserve probability ordering: a larger log-score still means a more probable path. The maximum hop count and requested output count are different settings. Floating-point improvement checks must match the independent golden. This is scalar best-path scoring, not the full learned NBFNet computation. Source roles: mathematical background: [Neural Bellman-Ford Networks](https://arxiv.org/abs/2106.06935); implementation details and limits: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

#### Native key code and work

[Local native](../softwareworkload/kg_multihop/workload.py#L67). [NBFNet `bellmanford`](https://github.com/DeepGraphLearning/NBFNet/blob/eb49ff2770d1629edb04e96c9a6d7fee809097f2/nbfnet/model.py) propagates layer states; [NBFNet `message` / `aggregate` / `combine`](https://github.com/DeepGraphLearning/NBFNet/blob/eb49ff2770d1629edb04e96c9a6d7fee809097f2/nbfnet/layer.py) implements learned message/aggregation functions. Local `native` instead propagates scalar log path scores and keeps best values per destination.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
Repeat within the allowed hop budget:
    for each query and improved vertex score:
        visit its allowed outgoing edges
        candidate = current_log_score + edge_log_score
        keep the best score for each destination
```

#### Workload example

A later indirect path improves an already present destination. Values below are log-scores, not probabilities or measured model accuracy. Compare the direct a-to-c score with the score obtained by adding the a-to-b and b-to-c log-scores. The indirect route has the larger log-score and replaces c's previous value. changed stores that new complete score; subtracting the old score would answer a different question.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Neural Bellman-Ford Networks](https://arxiv.org/abs/2106.06935).

#### Decomposition literature and justification

- [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6) (Generic-Single-Source-Shortest-Distance; topological variant): Path extension, aggregation and queue discipline are separate; termination depends on graph/algebra assumptions.

- [D8 · GraphBLAS C API Specification (standard)](#d8) (`GrB_mxm`, `GrB_vxm`, reduction, mask/accumulator semantics): Sparse contraction, structural masks and accumulation have an established interface; our arbitrary-arity relations are a local generalization.

**How the foundations connect to this program.** For the scalar best-path instance, edge extension is max-plus contraction and improvement is a state update. Learned PNA/nonlinear modules are not justified by that semiring identity. The precise fields, schedule and state operations below are a local mapping, checked against [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

#### Key lowering code and work

[Local algebra](../softwareworkload/kg_multihop/workload.py#L104); [detailed decomposition](../docs/decompositions/kg_multihop.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Store scores as logarithms of probabilities. Extending a path then adds its edge log-score instead of multiplying probabilities. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6), [Neural Bellman-Ford Networks](https://arxiv.org/abs/2106.06935). Interface mapping / exact execution: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

2. Seed the source with the score of an empty path. Each query has its own saved scores. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

3. combine: extend newly improved paths with matching outgoing edges, group by query and destination, and keep the largest log-score. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6), [D8 · GraphBLAS C API Specification (standard)](#d8). Interface mapping / exact execution: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

4. update saved scores and feed actual improvements into the next round. Stop at the hop limit or when there are no improvements. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

5. Exclude the source as required by the local answer definition. Group candidates by query and use select_k for the requested answer count. Interface mapping / exact execution: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** Each edge key names its start and end; values are log-scores. first and second are successive expansion results. changed reports an improved destination and its new total score, not a score difference.

<!-- EXAMPLE:kg_multihop -->
```text
input: log-edge weights={('a', 'b'): -1, ('a', 'c'): -4, ('b', 'c'): -2}; source=a; hop limit=2
native: -3
first: {('b',): -1, ('c',): -4}
second: {('c',): -3}
changed: {('c',): -3}
result: {('b',): -1, ('c',): -3}
```
<!-- /EXAMPLE:kg_multihop -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): Top-k is partitioned per query on the host. A neural NBFNet baseline cannot be claimed from this scalar implementation. [Local algebra](../softwareworkload/kg_multihop/workload.py#L104); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="bayes_net_ve"></a>

### C.7. bayes_net_ve — Bayesian-network variable elimination

#### Workload literature and mathematical task

**Papers and reading locations:** [Dechter, Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594) — belief assessment and bucket-elimination algorithm.

**Input, calculation and answer.** Input: probability tables for a discrete Bayesian network, observed variable values and the variable to ask about. Output: the probability of each query value after accounting for the observations. For each possible assignment, multiply its compatible factor entries; add over hidden assignments, then normalize. Definition checked against [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594); local scope: [Local native](../softwareworkload/bayes_net_ve/workload.py#L233).

**Terms, conventions and pitfalls.** Evidence means an observed variable value; prior means probability before considering that observation; posterior means probability after conditioning on it. Normalizing divides weights by their total so that they form a probability distribution. A bucket with only one factor still needs reduction. Choosing the order and managing factor lists are host work, not scan callbacks. Probability-space and log-space evaluations are separate. The independent golden uses factor dictionaries and small-instance enumeration. Source roles: mathematical background: [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594); implementation details and limits: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

#### Native key code and work

[Local native](../softwareworkload/bayes_net_ve/workload.py#L233). [pgmpy `VariableElimination._variable_elimination`](https://github.com/pgmpy/pgmpy/blob/fa42ee20e09488b10f9f556abed8ad415658ed05/pgmpy/inference/ExactInference.py) executes factor product, `marginalize`/`maximize`, and normalization. Local `_DictVE.product` / `sum_out` and `native` enumerate factor-table assignments; `brute_force` independently enumerates assignments.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
For each variable in the elimination order:
    collect all factors containing that variable
    multiply matching entries and sum over that variable's values
    replace the collected factors with the resulting factor
Combine remaining factors and normalize the query weights.
```

#### Workload example

This smallest evidence example shows product, elimination to evidence mass, and posterior normalization. A hidden-variable bucket uses the same reduction while retaining its other variables. The observation is that B takes the shown observed value; the question is how this changes the probabilities of A. For each A value, multiply its prior by the likelihood of the observation to get factor_product. Summing those products gives the probability of the evidence. Divide each product by that total to obtain result. The temporary elimination used for the denominator does not discard the query table.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594).

#### Decomposition literature and justification

- [D2 · The Generalized Distributive Law](#d2) (§II–IV): Marginalize a product of functions through distributivity. Exactness needs the stated structure/schedule conditions.

- [D3 · FAQ: Questions Asked Frequently](#d3) (FAQ definition; InsideOut; equivalent variable orderings): Eliminate variables by join and aggregation; not all mixed-aggregate reorderings are legal.

**How the foundations connect to this program.** Factor multiplication is a join on shared variables; marginalization is projection/reduction. InsideOut explains legal elimination reordering and its width cost. Normalization is a final division/log subtraction, not another semiring sum. The precise fields, schedule and state operations below are a local mapping, checked against [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

#### Key lowering code and work

[Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269); [detailed decomposition](../docs/decompositions/bayes_net_ve.md); primitive implementations: [ops.py](../algebra/ops.py).

1. filter each factor table to entries compatible with observed evidence. A factor is a table that assigns a weight to a combination of variable values. Mechanism sources: [D1 · Provenance Semirings](#d1), [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

2. Choose a hidden variable to eliminate. Collect the current factors containing that variable; this collection is its bucket. Mechanism sources: [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594), [D3 · FAQ: Questions Asked Frequently](#d3). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

3. combine those factors on shared variable values, multiplying their weights. reduce by the other variables to sum over the eliminated one. Mechanism sources: [D2 · The Generalized Distributive Law](#d2), [D3 · FAQ: Questions Asked Frequently](#d3). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

4. Replace that bucket with the resulting smaller factor. Repeat in the declared elimination order. Mechanism sources: [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594), [D3 · FAQ: Questions Asked Frequently](#d3). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

5. Combine remaining factors and keep the query variable. Sum its unnormalized weights, then divide each weight by that sum. Mechanism sources: [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

6. If the sum is zero, the evidence has zero probability and a conditional answer is undefined. The log-space version performs the equivalent operations using log-sum-exp. Mechanism sources: [D2 · The Generalized Distributive Law](#d2), [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** A and B name random variables. P(A) is the prior distribution; P(B=1|A) gives the likelihood of the stated observation for each A value. factor_product holds unnormalized posterior weights; evidence_mass is their sum; result is the normalized posterior. The numbers in the names denote the observed value, not a hardware parameter.

<!-- EXAMPLE:bayes_net_ve -->
```text
input: P(A)={(0,): 1/2, (1,): 1/2}; P(B=1|A)={(0,): 1/4, (1,): 3/4}
native: {(0,): 1/4, (1,): 3/4}
factor_product: {(0,): 1/8, (1,): 3/8}
eliminate_A_evidence_mass: {(): 1/2}
result: {(0,): 1/4, (1,): 3/4}
```
<!-- /EXAMPLE:bayes_net_ve -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): Elimination order is host-selected. Intermediate factor width and bytes, not merely multiply throughput, can dominate; MAP cannot freely exchange max and sum. [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="prob_circuit"></a>

### C.8. prob_circuit — Arithmetic / probabilistic circuit evaluation

#### Workload literature and mathematical task

**Papers and reading locations:** [Liu et al., Scaling Tractable Probabilistic Circuits: A Systems Perspective (PyJuice)](https://arxiv.org/abs/2406.00766) — probabilistic-circuit definitions and forward inference.

**Input, calculation and answer.** Input: a directed computation graph of sum/product nodes, leaf values and weights. Output: the root value for each input item. A sum node adds weighted child values; a product node multiplies all required child values. Definition checked against [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766); local scope: [Local _native_eval](../softwareworkload/prob_circuit/workload.py#L114).

**Terms, conventions and pitfalls.** A node represents a calculation. Leaves supply input values; internal nodes consume child results. A topological order means that every child is computed before the parent that needs it. Log-space products become addition; log-space sums use log-sum-exp. The counting run uses true/false leaf indicators and unit sum weights to count arithmetic derivations. A generic multiply-fold over only stored children is unsafe when required zeros were suppressed. The random DAG is not automatically a certified probabilistic model. Source roles: mathematical background: [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766); implementation details and limits: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

#### Native key code and work

[Local _native_eval](../softwareworkload/prob_circuit/workload.py#L114). [PyJuice `SumLayer.forward` / `_forward_pytorch_kernel`](https://github.com/Tractables/pyjuice/blob/00e8e770ef79538514331ccb965a34ed530f1da1/src/pyjuice/layer/sum_layer.py) evaluates weighted sums in a stable log representation; [PyJuice `ProdLayer.forward`](https://github.com/Tractables/pyjuice/blob/00e8e770ef79538514331ccb965a34ed530f1da1/src/pyjuice/layer/prod_layer.py) sums child log-values for products. Local `_native_eval` walks a synthetic layered arithmetic DAG.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
Visit layers after their children are ready.
For a sum node: multiply each child value by its edge weight, then add.
For a product node: multiply the values of all required children.
Return each sample's root value.
```

#### Workload example

The expression contains shared input x and alternative products. The final extra line replaces y with zero to expose why a required-slot join is safer than multiplying only stored members. Compute the product of x with y and the product of x with z separately. Weight the first branch by w1 and the second by w2, then add the contributions for the root. The final check changes y to zero: the entire xy product must disappear as a zero result, while xz remains. Multiplying only the stored nonzero children would incorrectly keep x in the first branch.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766).

#### Decomposition literature and justification

- [D1 · Provenance Semirings](#d1) (§3, Definitions 3.1–3.2; §5): Annotated tuples; join multiplies, projection/union adds. Positive algebra does not by itself define our antijoin, top-k or mutable state.

- [D2 · The Generalized Distributive Law](#d2) (§II–IV): Marginalize a product of functions through distributivity. Exactness needs the stated structure/schedule conditions.

- [D3 · FAQ: Questions Asked Frequently](#d3) (FAQ definition; InsideOut; equivalent variable orderings): Eliminate variables by join and aggregation; not all mixed-aggregate reorderings are legal.

**How the foundations connect to this program.** The sum is a join followed by aggregation. A product over mandatory children is a different fold: local code gathers each child slot and chains joins on the parent key, so an absent zero child kills the product. No cited generic aggregation rule licenses silently dropping required zero children. The precise fields, schedule and state operations below are a local mapping, checked against [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

#### Key lowering code and work

[Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143); [detailed decomposition](../docs/decompositions/prob_circuit.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Process layers only after the child values they need are available; independent nodes and input items can be processed together. Mechanism sources: [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766). Interface mapping / exact execution: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

2. For a sum layer, combine the child-value table with the edge-weight table and group by sample and parent node. Mechanism sources: [D1 · Provenance Semirings](#d1), [D2 · The Generalized Distributive Law](#d2). Interface mapping / exact execution: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

3. For a product layer, separately gather each mandatory child slot, then combine the gathered tables on sample and parent. A missing zero child must make the entire product zero. Mechanism sources: [D2 · The Generalized Distributive Law](#d2), [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766). Interface mapping / exact execution: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

4. rename the output node field for the next layer, then continue through the finite layer list. Mechanism sources: [D1 · Provenance Semirings](#d1). Interface mapping / exact execution: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

5. Read the designated root for each input item. Use ordinary probability, log probability or integer counting consistently within an execution. Mechanism sources: [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766). Interface mapping / exact execution: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** x, y and z are leaf values; w1 and w2 weight alternative product branches. xy and xz are names for child products, not unexplained variables. mandatory_slot_product shows those products; the final line checks what happens if y is zero.

<!-- EXAMPLE:prob_circuit -->
```text
input: x=1/2; y=1/3; z=1/4; w1=2/5; w2=3/5
native: 17/120
mandatory_slot_product: {('xy',): 1/6, ('xz',): 1/8}
result: {(): 17/120}
if_y_zero_safe_product: {('xz',): 1/8}
```
<!-- /EXAMPLE:prob_circuit -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): The local random DAG is not automatically a valid probabilistic circuit, WMC instance, or general MAP evaluator. No GPU PyJuice timing was obtained by inspecting this code. [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="automata_rpq"></a>

### C.9. automata_rpq — Regular path query

#### Workload literature and mathematical task

**Papers and reading locations:** [cuRPQ: A High-Performance GPU-Based Framework for Processing Regular and Conjunctive Regular Path Queries](https://arxiv.org/abs/2602.20748) — regular-path-query definition and product-graph construction/traversal.

**Input, calculation and answer.** Input: a graph with edge labels, an automaton describing allowed label sequences and query sources. Output: destinations reachable by a path whose label sequence the automaton accepts. Track both the graph position and the automaton state while moving. Definition checked against [cuRPQ](https://arxiv.org/abs/2602.20748); local scope: [Local native](../softwareworkload/automata_rpq/workload.py#L146).

**Terms, conventions and pitfalls.** An automaton is a finite-state checker for a sequence of labels. An accepting state means the labels read so far form an allowed sequence. A nondeterministic automaton may retain several possible states, so the state must remain part of each visited key. Graph-first, automaton-first and precomputed-product joins are alternative schedules, not stages to run consecutively. A precomputed product table is built outside the loop. The local automaton has no transitions that consume no label; such transitions would need additional handling. Source roles: mathematical background: [cuRPQ](https://arxiv.org/abs/2602.20748); implementation details and limits: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

#### Native key code and work

[Local native](../softwareworkload/automata_rpq/workload.py#L146). Local `_glushkov` constructs an epsilon-free NFA; `native` traverses graph-vertex/automaton-state pairs with a queue. There is no pinned cuRPQ author-code entry in the current clone manifest; local code is an independent implementation, not that artifact.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
Start with the graph source and the automaton's initial state.
While the queue is not empty:
    read a graph edge and an automaton transition with the same label
    enqueue the destination vertex/state pair if it is new
Return graph vertices paired with accepting automaton states.
```

#### Workload example

Graph and automaton must advance on the same label. The intermediate pair identifies both where traversal is and which language prefix has matched. The graph and automaton both allow an x-labeled move first and a y-labeled move next. product_edges pairs only transitions carrying the same label. first and second advance the graph vertex and automaton state together. Only the latter pair has an accepting state, so the result keeps its graph endpoint and drops the automaton-state field.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [cuRPQ](https://arxiv.org/abs/2602.20748).

#### Decomposition literature and justification

- [D1 · Provenance Semirings](#d1) (§3, Definitions 3.1–3.2; §5): Annotated tuples; join multiplies, projection/union adds. Positive algebra does not by itself define our antijoin, top-k or mutable state.

- [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6) (Generic-Single-Source-Shortest-Distance; topological variant): Path extension, aggregation and queue discipline are separate; termination depends on graph/algebra assumptions.

- [D8 · GraphBLAS C API Specification (standard)](#d8) (`GrB_mxm`, `GrB_vxm`, reduction, mask/accumulator semantics): Sparse contraction, structural masks and accumulation have an established interface; our arbitrary-arity relations are a local generalization.

**How the foundations connect to this program.** Equality of edge labels is a join; existentially eliminating the label yields product edges; Boolean closure and an accepting-state mask finish the query. Local code additionally compares graph-first, NFA-first and product-first join orders. The precise fields, schedule and state operations below are a local mapping, checked against [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

#### Key lowering code and work

[Local algebra](../softwareworkload/automata_rpq/workload.py#L188); [detailed decomposition](../docs/decompositions/automata_rpq.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Initialize each query at its source vertex and the automaton's start state. Mechanism sources: [cuRPQ](https://arxiv.org/abs/2602.20748). Interface mapping / exact execution: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

2. combine graph edges with automaton transitions that have the same label. This may be done on demand or precomputed into a product-edge table. Mechanism sources: [D1 · Provenance Semirings](#d1), [cuRPQ](https://arxiv.org/abs/2602.20748). Interface mapping / exact execution: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

3. Extend each new graph/state pair through a compatible product edge. Keep query identity, new vertex and new automaton state. Mechanism sources: [D1 · Provenance Semirings](#d1), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

4. mask out already visited pairs and update the saved and new-pair tables. Repeat until no new pair remains. Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](#d8). Interface mapping / exact execution: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

5. mask by accepting automaton states and reduce away the automaton state; return query/destination pairs. Mechanism sources: [D1 · Provenance Semirings](#d1), [D8 · GraphBLAS C API Specification (standard)](#d8). Interface mapping / exact execution: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** G rows list graph start, edge label and graph end. Automaton rows list old state, label and new state. s0/s1/s2 are state names. A product edge lists old vertex, old state, new vertex and new state.

<!-- EXAMPLE:automata_rpq -->
```text
input: G=[[a, x, b], [b, y, c]]; automaton=[[s0, x, s1], [s1, y, s2]]; accept=s2; source=a
native: ['c']
product_edges: {('a', 's0', 'b', 's1'): True, ('b', 's1', 'c', 's2'): True}
first: {('b', 's1'): True}
second: {('c', 's2'): True}
result: {('c',): True}
```
<!-- /EXAMPLE:automata_rpq -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): Precomputing the product graph can expand memory dramatically. Epsilon transitions would need extra closure handling; all-pairs and source-specific queries are different outputs. [Local algebra](../softwareworkload/automata_rpq/workload.py#L188); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="beam_search"></a>

### C.10. beam_search — Beam expansion and pruning

#### Workload literature and mathematical task

**Papers and reading locations:** [Freitag & Al-Onaizan, Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/) — §3 Original Beam Search; §4 is a set of additional strategies.

**Input, calculation and answer.** Input: token-transition scores, an initial token, the number of hypotheses to retain and a length limit. Output: the surviving token sequences and scores. Extend every surviving sequence, score each extension, and retain only the requested number of best extensions. Definition checked against [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/); local scope: [Local native](../softwareworkload/beam_search/workload.py#L54).

**Terms, conventions and pitfalls.** A prefix is the sequence generated so far. Pruning drops candidates now, without a guarantee that all globally best complete sequences remain. Parent identifiers say which earlier prefix produced a new candidate. Beam means the currently retained set of candidate sequences. Top-k means the best k items, where k is the requested count. Rank calculation and parent storage are explicit control work. This local Markov toy conditions on the last token; a language model also needs the full prefix and model state. Source roles: mathematical background: [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/); implementation details and limits: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

#### Native key code and work

[Local native](../softwareworkload/beam_search/workload.py#L54). [Transformers `_beam_search` / `_get_top_k_continuations`](https://github.com/huggingface/transformers/blob/96331a9f93b72697f160a958d2883d4b49a56739/src/transformers/generation/utils.py) accumulates token scores, selects continuations and maintains generation state. Local `native` uses a Markov transition table, a candidate list and a heap; there is no transformer/KV computation.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
For each retained prefix and allowed next token:
    candidate_score = prefix_log_score + next_token_log_score
    keep the next token and the parent-prefix identifier
Retain the requested number of best candidates with the declared tie rule.
```

#### Workload example

The best surviving hypotheses can share a last token but have different parents. The WRONG line deliberately removes that key to show the lost hypothesis; it is not the implemented result. Add each parent's score to each allowed next-token score, keeping the parent identifier in every candidate. Both retained candidates can end in z because they represent different sequences. WRONG_collapsing_parent shows why grouping only by the last token prematurely erases a valid beam candidate.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/).

#### Decomposition literature and justification

- [D1 · Provenance Semirings](#d1) (§3, Definitions 3.1–3.2; §5): Annotated tuples; join multiplies, projection/union adds. Positive algebra does not by itself define our antijoin, top-k or mutable state.

- [D11 · Better k-best Parsing](#d11) (k-best derivation algorithms): Derivation identity and structured candidate generation matter. Our batch select is not their complete algorithm.

**How the foundations connect to this program.** The transition expansion is a local weighted join; ordering/pruning is an extra operation. The k-best literature clarifies derivation identity, not an equivalence between beam search and exact k-best parsing. Our retained key must include the parent beam. The precise fields, schedule and state operations below are a local mapping, checked against [Local algebra](../softwareworkload/beam_search/workload.py#L69).

#### Key lowering code and work

[Local algebra](../softwareworkload/beam_search/workload.py#L69); [detailed decomposition](../docs/decompositions/beam_search.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Keep a distinct identifier for every current sequence, even if several end with the same token. Mechanism sources: [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/), [D11 · Better k-best Parsing](#d11). Interface mapping / exact execution: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

2. combine the sequence table with allowed transitions on the last token. Add the transition log-score; retain both the new token and the parent-sequence identifier. Mechanism sources: [D1 · Provenance Semirings](#d1), [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/). Interface mapping / exact execution: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

3. select_k retains the requested number of best candidates globally for this beam. Mechanism sources: [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/). Interface mapping / exact execution: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

4. Host code ranks survivors by decreasing score, then increasing token and parent identifier; save parent links and scan to assign the new ranks. Interface mapping / exact execution: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

5. Repeat to the length limit, then follow saved parent links backward to reconstruct each sequence. Mechanism sources: [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/). Interface mapping / exact execution: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** p and q identify parent sequences; x and y are their last tokens; z and w are proposed next tokens. K is the retention count. Candidate keys retain next token and parent. WRONG_collapsing_parent deliberately shows the incorrect result after dropping parent identity.

<!-- EXAMPLE:beam_search -->
```text
input: beam log scores={('p', 'x'): -1, ('q', 'y'): -2}; transition log scores={('x', 'w'): -5, ('x', 'z'): -2, ('y', 'z'): -2}; K=2
native: [[-3, z, p], [-4, z, q]]
candidates_with_parent: {('w', 'p'): -6, ('z', 'p'): -3, ('z', 'q'): -4}
result: {('z', 'p'): -3, ('z', 'q'): -4}
WRONG_collapsing_parent: {('w',): -6, ('z',): -3}
```
<!-- /EXAMPLE:beam_search -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): `select_k` is global batch selection, not a persistent priority queue or grouped beam operator. Ranking, EOS handling and backpointers are not free scan work. [Local algebra](../softwareworkload/beam_search/workload.py#L69); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="fuzzy_inference"></a>

### C.11. fuzzy_inference — Max-min fuzzy relational closure

#### Workload literature and mathematical task

**Papers and reading locations:** [Zadeh, Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051) — max-min composition and transitivity definitions.

**Input, calculation and answer.** Input: relation strengths between items and query strengths. Output: the strongest indirect relation for each pair, then each query's strength to each answer. A path is as strong as its weakest edge; among alternative paths choose the strongest. Definition checked against [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051); local scope: [Local native](../softwareworkload/fuzzy_inference/workload.py#L68).

**Terms, conventions and pitfalls.** Relation strength lies between zero and one, with a larger value meaning a stronger relation. It is not a probability of mutually exclusive events; the weakest-link/strongest-path rule is the requested fuzzy relation operation. Do not remove candidates merely because the key is already present. Generated cases use a finite set of possible strengths, which bounds the number of improvements. The independent golden uses ordered widest-path propagation. Source roles: mathematical background: [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051); implementation details and limits: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

#### Native key code and work

[Local native](../softwareworkload/fuzzy_inference/workload.py#L68). Local `native` computes widest-path values with a priority queue and evaluates the query relation. There is no separate pinned Zadeh author implementation; the scalar definition and independent local golden are the evidence.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
For every possible start and end:
    a path has the strength of its weakest edge
    choose the strongest path between these endpoints
For a query:
    extend query strengths through the resulting relation
    use the same weakest-link / strongest-alternative rules
```

#### Workload example

The direct relation is weaker than an indirect path. A support antijoin would incorrectly discard the improvement demonstrated below. For the indirect route through b, take the weaker of the two edge strengths. Compare this value with the direct a-to-c strength and keep the stronger route. The endpoint pair already exists, but its value changes, so a support-only visited filter would incorrectly discard the improvement.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051).

#### Decomposition literature and justification

- [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6) (Generic-Single-Source-Shortest-Distance; topological variant): Path extension, aggregation and queue discipline are separate; termination depends on graph/algebra assumptions.

- [D8 · GraphBLAS C API Specification (standard)](#d8) (`GrB_mxm`, `GrB_vxm`, reduction, mask/accumulator semantics): Sparse contraction, structural masks and accumulation have an established interface; our arbitrary-arity relations are a local generalization.

**How the foundations connect to this program.** Use max as alternative aggregation and min as path extension. The local named-relation contraction is derived from this recurrence; changed values must be propagated even when support already existed. The precise fields, schedule and state operations below are a local mapping, checked against [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

#### Key lowering code and work

[Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115); [detailed decomposition](../docs/decompositions/fuzzy_inference.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Seed saved and newly improved relations with the input strengths. Mechanism sources: [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051). Interface mapping / exact execution: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

2. combine newly improved paths with another relation edge: take the smaller strength for each extension, then the largest strength among alternatives to the same pair. Mechanism sources: [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

3. update saved strengths and propagate genuinely improved values, including improvements to pairs that already existed. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

4. Repeat until no strength improves. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

5. combine query strengths with the completed relation using the same weaker-link/strongest-alternative rules. Mechanism sources: [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051). Interface mapping / exact execution: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** A pair lists start and end; its value is relation strength. join is an indirect candidate. changed_existing_key demonstrates a stronger value for a pair already in the input.

<!-- EXAMPLE:fuzzy_inference -->
```text
input: max-min edges={('a', 'b'): 4/5, ('a', 'c'): 1/5, ('b', 'c'): 1/2}
native: 1/2
join: {('a', 'c'): 1/2}
changed_existing_key: {('a', 'c'): 1/2}
result: {('a', 'b'): 4/5, ('a', 'c'): 1/2, ('b', 'c'): 1/2}
```
<!-- /EXAMPLE:fuzzy_inference -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): Shared support is not equivalent to settled value. Queue/dependency effects differ from Boolean reachability despite a similar join shape. [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="hmm_viterbi"></a>

### C.12. hmm_viterbi — HMM forward and Viterbi inference

#### Workload literature and mathematical task

**Papers and reading locations:** [Rush, Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/) — semiring dynamic programming and parallel linear-chain implementation.

**Input, calculation and answer.** Input: initial hidden-state probabilities, probabilities of moving between hidden states, probabilities of observations in each state, and an observation sequence. Forward inference returns the total probability of the observations. Viterbi returns the single most likely hidden-state path and its score. Definition checked against [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/); local scope: [Local _native_one](../softwareworkload/hmm_viterbi/workload.py#L71).

**Terms, conventions and pitfalls.** A hidden state is the model's unobserved condition; an observation is the data actually seen. Transition probabilities connect consecutive hidden states, while emission probabilities connect a state to a possible observation. The implementation stores log probabilities. LSE, short for log-sum-exp, adds the represented probabilities and returns the logarithm of their sum, without explicitly converting very small probabilities. Tagged Viterbi values contain a score and a predecessor. Combining them adds scores and uses the right operand's valid predecessor, otherwise the left one's; alternatives choose higher score then smaller predecessor identifier. Emission tags mean “no new predecessor.” Preserve operand order, save every layer, and use score then state identifier for final ties. Readout/traceback is explicit host work, with per-layer filtering in the current code. These tagged values and tie rules are a specific local definition; arbitrary tagged values are not claimed to satisfy semiring laws or allow operand swapping. Source roles: mathematical background: [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/); implementation details and limits: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

#### Native key code and work

[Local _native_one](../softwareworkload/hmm_viterbi/workload.py#L71). [Torch-Struct `LinearChain.logpartition` / `hmm`](https://github.com/harvardnlp/pytorch-struct/blob/7146de5659ff17ad7be53023c025ffd099866412/torch_struct/linearchain.py) performs a matrix-based parallel scan; local `_native_one` uses sequential vector recurrences and stores Viterbi predecessors. They are different schedules, not interchangeable work counts.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
For each next hidden state:
    form one candidate from each previous state and its transition
    forward_value = add all candidate probabilities
    best_value, parent = choose the highest-probability candidate
    apply the next observation likelihood to both values
Save parents for the final backward reconstruction.
```

#### Workload example

The example starts after the first emission and shows one transition/emission step. Exact probability-space fractions expose all paths; production code uses log-space and tagged predecessors instead of enumerating them. For forward inference, multiply each accumulated state probability by the outgoing transition probability, add contributions arriving at the same state, and apply the next observation likelihood. The final total sums the resulting state values. all_path_weights instead keeps every path separately: Viterbi chooses its largest entry, whereas forward adds all of them. A best-path pointer must preserve which path achieved that entry.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/).

#### Decomposition literature and justification

- [D2 · The Generalized Distributive Law](#d2) (§II–IV): Marginalize a product of functions through distributivity. Exactness needs the stated structure/schedule conditions.

- [D4 · Semiring Parsing](#d4) (§1, Figures 1–3; semiring framework): Deduction antecedents multiply; alternative proofs add; inside versus best derivation changes the value algebra.

- [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6) (Generic-Single-Source-Shortest-Distance; topological variant): Path extension, aggregation and queue discipline are separate; termination depends on graph/algebra assumptions.

**How the foundations connect to this program.** Transition contraction eliminates the previous state, then multiplies the emission. Local LOGPROB implements stable forward inference; tagged max-plus carries a predecessor for Viterbi. The tag/tie/traceback contract is additional to scalar semiring arithmetic. The precise fields, schedule and state operations below are a local mapping, checked against [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

#### Key lowering code and work

[Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110); [detailed decomposition](../docs/decompositions/hmm_viterbi.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Initialize each state from its initial probability and the probability of the first observed item in that state. Mechanism sources: [D2 · The Generalized Distributive Law](#d2), [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/). Interface mapping / exact execution: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

2. combine previous-state scores with the transition table, matching the previous state and retaining the next state. Mechanism sources: [D2 · The Generalized Distributive Law](#d2). Interface mapping / exact execution: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

3. For forward inference, add contributions arriving at the same next state. For Viterbi, keep only the best contribution and its previous-state identifier. Mechanism sources: [D4 · Semiring Parsing](#d4). Interface mapping / exact execution: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

4. Apply the next observation's probability, rename next state to current state, and repeat along the observation sequence. Mechanism sources: [D2 · The Generalized Distributive Law](#d2). Interface mapping / exact execution: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

5. Forward readout combines final-state probabilities. Viterbi selects the best final state for each query and follows saved predecessor identifiers backward. Mechanism sources: [D4 · Semiring Parsing](#d4). Interface mapping / exact execution: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** alpha is the probability already accumulated after the first observation. A is the state-transition table. Emission means the likelihood of the next observed item in a state. A path key lists previous and next states. forward sums path weights; best_path identifies the largest-weight path.

<!-- EXAMPLE:hmm_viterbi -->
```text
input: post-first-emission alpha={(0,): 1/2, (1,): 1/2}; A={(0, 0): 3/4, (0, 1): 1/4, (1, 0): 1/4, (1, 1): 3/4}; next emission={(0,): 1/2, (1,): 1/4}
native: {best_path: [0, 0], best_weight: 3/16, forward: 3/8}
transition_contract: {(0,): 1/2, (1,): 1/2}
emission: {(0,): 1/4, (1,): 1/8}
all_path_weights: {(0, 0): 3/16, (0, 1): 1/32, (1, 0): 1/16, (1, 1): 3/32}
result_forward: {(): 3/8}
result_best: {(0, 0): 3/16}
```
<!-- /EXAMPLE:hmm_viterbi -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): More parallel depth can introduce more arithmetic work. Backpointer traffic and state dimension need profiling independently of LSE throughput. [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="ctrlg_dfa_hmm"></a>

### C.13. ctrlg_dfa_hmm — DFA-constrained HMM guidance

#### Workload literature and mathematical task

**Papers and reading locations:** [Zhang et al., Ctrl-G: Adaptable Logical Control for Large Language Models](https://arxiv.org/abs/2406.13892) — §2 HMM and guidance formulation; §§3.2–3.3 probability versus logical feasibility.

**Input, calculation and answer.** Input: an HMM probability model, a deterministic automaton describing allowed output sequences and an observed prefix. Output in the local proxy: next-token scores restricted to tokens that can still lead to acceptance. Full constrained generation additionally weights how probable a valid completion is within the remaining length. Definition checked against [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892); local scope: [Local native](../softwareworkload/ctrlg_dfa_hmm/workload.py#L132).

**Terms, conventions and pitfalls.** The automaton checks sequence legality; the HMM assigns sequence probabilities. A deterministic automaton has at most one successor for a given state and token. Acceptance is a Boolean condition, not a probability value. HMM is a hidden Markov model; DFA is a deterministic finite automaton. Liveness means only that some accepted continuation exists, not its probability. For a known prefix, a deterministic automaton has one active state; the entire state-product table need not be occupied. Use compatible initialization for transition-then-emission calculations. Invalid transitions contribute nothing, normalization requires a positive total, and variable lengths/end-of-sequence need an explicit termination rule. Full Ctrl-G uses HMM-estimated constraint-satisfaction probabilities to reweight the language model's next-token distribution. The teaching case checks the constrained HMM distribution alone, not the full language-model generation pipeline. The default sparse_first schedule filters through the sparse allowed-next-step table before gathering token probabilities; dense_first computes all vocabulary contributions before filtering and is a comparison schedule. Source roles: mathematical background: [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892); implementation details and limits: [Local _forward_and_scores](../softwareworkload/ctrlg_dfa_hmm/workload.py#L184).

#### Native key code and work

[Local native](../softwareworkload/ctrlg_dfa_hmm/workload.py#L132). [Ctrl-G `ConstraintLogitsProcessor.compute_logits`](https://github.com/joshuacnf/Ctrl-G/blob/379463249c368739f81c46c1e71b8673bfcedf7c/ctrlg/utils.py) combines cached prefix state with remaining-length completion caches. Local `native` / `_forward_and_scores` implement only prefix HMM×DFA propagation and a liveness-filtered next-token score.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
Local proxy:
    advance the prefix through the HMM and automaton
    score each next token and keep successors that can still accept
Full constrained HMM calculation:
    additionally multiply by remaining valid-completion probability
    normalize the candidate weights
```

#### Workload example

Both next tokens remain feasible, but their valid completion masses differ. The local proxy and exact conditional result are intentionally shown side by side; only the explanatory script computes the extra exact recurrence. Starting with a allows only another a; starting with b allows only another b. Both starts are logically feasible, so the local liveness mask retains the original token probabilities. Exact constrained HMM inference also multiplies each candidate by the probability of its required continuation, then normalizes. The different result is a semantic difference, not a timing result.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892).

#### Decomposition literature and justification

- [D2 · The Generalized Distributive Law](#d2) (§II–IV): Marginalize a product of functions through distributivity. Exactness needs the stated structure/schedule conditions.

- [D3 · FAQ: Questions Asked Frequently](#d3) (FAQ definition; InsideOut; equivalent variable orderings): Eliminate variables by join and aggregation; not all mixed-aggregate reorderings are legal.

- [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6) (Generic-Single-Source-Shortest-Distance; topological variant): Path extension, aggregation and queue discipline are separate; termination depends on graph/algebra assumptions.

**How the foundations connect to this program.** Product-state dynamic programming contracts hidden states and uses DFA support joins. The displayed backward valid-completion probability recurrence is the full finite-horizon ingredient; replacing valid-completion probability by Boolean liveness is a different computation. The code mapping is local and does not claim a faithful full Ctrl-G reproduction. The precise fields, schedule and state operations below are a local mapping, checked against [Local _forward_and_scores](../softwareworkload/ctrlg_dfa_hmm/workload.py#L184).

#### Key lowering code and work

[Local _forward_and_scores](../softwareworkload/ctrlg_dfa_hmm/workload.py#L184); [detailed decomposition](../docs/decompositions/ctrlg_dfa_hmm.md); primitive implementations: [ops.py](../algebra/ops.py).

1. First reduce away token labels to obtain the edges needed for backward traversal. Starting from accepting states, use combine, rename, mask and update to mark every state that can still reach acceptance. Mechanism source: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6); exact execution: [Local algebra](../softwareworkload/ctrlg_dfa_hmm/workload.py#L231).

2. mask retains automaton transitions whose successors appear in that live-state table. reduce drops the successor field, leaving an allowed-next-step table keyed by current state and token. Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](#d8), [D1 · Provenance Semirings](#d1); exact execution: [Local algebra](../softwareworkload/ctrlg_dfa_hmm/workload.py#L231).

3. Process the observed prefix: combine hidden-state probabilities with state transitions, apply the observed token likelihood and match its automaton transition. Initialize the first observation consistently with the model. Mechanism sources: [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892), [D2 · The Generalized Distributive Law](#d2). Interface mapping / exact execution: [Local _forward_and_scores](../softwareworkload/ctrlg_dfa_hmm/workload.py#L184).

4. For each proposed next token, combine current probabilities with hidden-state transitions and token likelihoods; mask out automaton successors that cannot reach acceptance. This is the local proxy. Interface mapping / exact execution: [Local _forward_and_scores](../softwareworkload/ctrlg_dfa_hmm/workload.py#L184).

5. For full fixed-length conditioning, build a backward table of valid-completion probability. With no emissions remaining, only accepting states succeed. For each larger remaining length, enumerate a hidden-state transition and emitted token, multiply their probabilities by the previous completion-table entry, and add alternatives. Paper basis: [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892) (§§3.2–3.3); this explains the missing computation and is not implemented in the local function above.

6. Weight next-token candidates by that remaining-completion probability, then divide each candidate's weight by the total. This additional backward calculation is not in the local proxy. Paper basis: [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892) (§§3.2–3.3); this explains the missing computation and is not implemented in the local function above.

#### Decomposition example: derived intermediate values

**Field and short-name guide.** a and b are token names; aa|bb means that only the sequence aa or bb is accepted. LOCAL_liveness_proxy is the current simplified answer. completion_beta means remaining valid-completion probability. exact_unnormalized includes this weight; result_exact normalizes it.

<!-- EXAMPLE:ctrlg_dfa_hmm -->
```text
input: one hidden state; P(token)={a: 3/4, b: 1/4}; accept aa|bb; remaining length=2
native: {('a',): 9/10, ('b',): 1/10}
LOCAL_liveness_proxy: {('a',): 3/4, ('b',): 1/4}
completion_beta: {('a',): 3/4, ('b',): 1/4}
exact_unnormalized: {('a',): 9/16, ('b',): 1/16}
result_exact: {('a',): 9/10, ('b',): 1/10}
```
<!-- /EXAMPLE:ctrlg_dfa_hmm -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): Adding full valid-completion probability changes both work and state size. An accelerator claim for full Ctrl-G must include that missing work and the surrounding model/guidance boundary. [Local _forward_and_scores](../softwareworkload/ctrlg_dfa_hmm/workload.py#L184); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="vsa_cleanup"></a>

### C.14. vsa_cleanup — VSA bind, bundle, unbind and cleanup

#### Workload literature and mathematical task

**Papers and reading locations:** [Heddes et al., Torchhd](https://www.jmlr.org/papers/v24/23-0300.html); [CogSys](https://zishenwan.github.io/publication/HPCA25_CogSys.pdf) — Torchhd: representations and bind/bundle/similarity operations; CogSys: VSA workload characterization.

**Input, calculation and answer.** Input: vectors representing attribute values, attribute keys, slot keys, object descriptions and queries. Output: the codebook entry most similar to each recovered query vector. Bind by multiplying corresponding coordinates, bundle by adding vectors, unbind using the queried keys, and compare against the codebook. Definition checked against [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html); local scope: [Local native](../softwareworkload/vsa_cleanup/workload.py#L65).

**Terms, conventions and pitfalls.** A codebook is a dictionary storing a vector for each candidate value. A key vector identifies which attribute or position a value occupies; it is different from a relational table's identifying key. A vector is a list of numbers; a coordinate is one position in that list; a dot product multiplies corresponding positions and adds them. The generated codebook is Gaussian-valued; only attribute/slot keys are bipolar, containing positive or negative one, so multiplying a key twice cancels its effect. Input-presence tables use neutral weights. Encoding and cleanup are both included. This is elementwise binding, not circular convolution. Source roles: mathematical background: [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html); implementation details and limits: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

#### Native key code and work

[Local native](../softwareworkload/vsa_cleanup/workload.py#L65). [Torchhd `bind` / `bundle` / `dot_similarity`](https://github.com/hyperdimensional-computing/torchhd/blob/9d73e1bd1f169d718340dfd40aa1aa976faa3851/torchhd/functional.py) dispatches by representation; [Torchhd `MAPTensor.bind` / `bundle`](https://github.com/hyperdimensional-computing/torchhd/blob/9d73e1bd1f169d718340dfd40aa1aa976faa3851/torchhd/tensors/map.py) gives elementwise multiplication and additive bundling. Local `native` uses NumPy elementwise products, sums and dot products; the generated codebook is Gaussian, keys/slots are bipolar.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
object_vector = add attribute vectors after binding each to its attribute key
scene_vector = add object vectors after binding each to its position key
query_vector = unbind the requested position and attribute from the scene
answer = codebook candidate with highest dot-product similarity
```

#### Workload example

A deliberately tiny orthogonal codebook makes bind/unbind/cleanup hand-checkable. One object and one slot make the sums singleton; this is not a sample from the Gaussian generator and says nothing about high-dimensional retrieval accuracy. The teaching codebook places x and y in different vector coordinates. Binding multiplies the selected vector by the key coordinate by coordinate; multiplying by that same bipolar key again removes the key. Dot products then compare the recovered vector with the codebook. A missing y similarity is a zero, not an unknown value. This tiny case does not measure multi-object interference.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html).

#### Decomposition literature and justification

- [D1 · Provenance Semirings](#d1) (§3, Definitions 3.1–3.2; §5): Annotated tuples; join multiplies, projection/union adds. Positive algebra does not by itself define our antijoin, top-k or mutable state.

- [D3 · FAQ: Questions Asked Frequently](#d3) (FAQ definition; InsideOut; equivalent variable orderings): Eliminate variables by join and aggregation; not all mixed-aggregate reorderings are legal.

- [D10 · The Sparse Abstract Machine](#d10) (primitive definitions; sparse tensor/dataflow examples): Physical lowering into coordinate streams, intersections, value operations and reductions; not a proof of all sym1 control/state support.

**How the foundations connect to this program.** Coordinate-matched products become joins; bundling and dot products eliminate object/attribute/dimension indices with addition; cleanup adds selection. This is our tensor-to-relation derivation, not a claim that Torchhd or CogSys uses this API. The precise fields, schedule and state operations below are a local mapping, checked against [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

#### Key lowering code and work

[Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92); [detailed decomposition](../docs/decompositions/vsa_cleanup.md); primitive implementations: [ops.py](../algebra/ops.py).

1. combine each object's attribute choices with the codebook and the corresponding attribute key; add contributions by object and vector coordinate. Mechanism sources: [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html), [D1 · Provenance Semirings](#d1). Interface mapping / exact execution: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

2. combine object vectors with scene slot assignments and slot keys; add contributions by scene and coordinate. Mechanism sources: [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html), [D1 · Provenance Semirings](#d1). Interface mapping / exact execution: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

3. For each query, gather its scene vector and multiply corresponding coordinates by its requested slot and attribute keys. Mechanism sources: [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html). Interface mapping / exact execution: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

4. combine the recovered vector with codebook candidates on attribute and coordinate; multiply matching coordinates and reduce by query and candidate to form dot-product similarities. Mechanism sources: [D3 · FAQ: Questions Asked Frequently](#d3), [D10 · The Sparse Abstract Machine](#d10). Interface mapping / exact execution: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

5. Filter rows for each query and select its best candidate, with deterministic ties. Interface mapping / exact execution: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** x and y name codebook candidates; each key also includes its vector-coordinate index. bind and unbind show vectors before and after removing the key. dot_similarity gives each candidate's dot product. Omitted zero coordinates still mean zero. The teaching codebook is orthogonal, unlike the Gaussian generator.

<!-- EXAMPLE:vsa_cleanup -->
```text
input: orthogonal teaching codebook={('x', 0): 1, ('y', 1): 1}; bipolar key={(0,): -1, (1,): 1}; one object/slot
native: {x: 1, y: 0}
bind: {(0,): -1}
unbind: {(0,): 1}
dot_similarity: {('x',): 1}
result: {('x',): 1}
```
<!-- /EXAMPLE:vsa_cleanup -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): Interference, normalization and representation choice affect accuracy. Dense dot products may favor conventional GEMM; shared sparse operators are not automatically faster. [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="astar_grid"></a>

### C.15. astar_grid — A* on a grid

#### Workload literature and mathematical task

**Papers and reading locations:** [Hart, Nilsson & Raphael, A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf) — A* search procedure and heuristic conditions.

**Input, calculation and answer.** Input: a grid with obstacles, a start and a goal, and equal-cost orthogonal moves. Output: a shortest path cost. Among pending vertices choose the smallest “cost already spent plus estimated cost still needed,” then relax its neighbors. Definition checked against [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf); local scope: [Local native](../softwareworkload/astar_grid/workload.py#L85).

**Terms, conventions and pitfalls.** The open table contains candidates waiting to be expanded; the closed/settled table contains vertices already finalized under the heuristic assumptions. The heuristic is an estimate, not distance that has already been traveled. The heuristic is Manhattan distance: the horizontal and vertical separation from the goal. On this grid it is consistent, meaning the estimate at a vertex never exceeds an outgoing move's cost plus the next vertex's estimate; this supports the local no-reopen rule. In the code, a scan constructs the correction value and a min-plus combine adds it. Row fetching, stop checks and stale-entry semantics remain explicit. Source roles: mathematical background: [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf); implementation details and limits: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

#### Native key code and work

[Local native](../softwareworkload/astar_grid/workload.py#L85). Local `native` uses a heap, g-score map and closed set. `_astar` expresses one extraction/expansion using operators; `_bellman_ford_frontier` is a distinct control baseline. No pinned author artifact of the original A* paper is claimed.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
Choose the pending vertex with smallest actual-plus-estimated cost.
Stop if it is the goal; otherwise mark it settled.
For each neighbor:
    candidate_cost = actual_cost_to_current + edge_cost
    improve the neighbor if this candidate is cheaper
```

#### Workload example

One corner expansion shows g recovery, edge relaxation and f reconstruction. It checks a key iteration, not a complete run or an optimal-expansion theorem. The selected start is stored with estimated total cost. recover_g subtracts the estimated remaining cost to recover what has actually been spent. neighbor_g then adds each outgoing edge cost. Finally add each neighbor's own remaining estimate to obtain its pending priority. result_next_open_f is the next pending table, not the final shortest-path answer.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf).

#### Decomposition literature and justification

- [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6) (Generic-Single-Source-Shortest-Distance; topological variant): Path extension, aggregation and queue discipline are separate; termination depends on graph/algebra assumptions.

- [D12 · The Generalized A* Architecture](#d12) (weighted deduction and generalized A*): Connects prioritized search and inference; our grid mapping additionally relies on its own consistency/tie/state assumptions.

**How the foundations connect to this program.** Ordered search is supported by queue/deduction literature; the exact select→scan→join→mask→update mapping is local. Crucially, stored f must be converted back to g before adding an edge. The precise fields, schedule and state operations below are a local mapping, checked against [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

#### Key lowering code and work

[Local _astar](../softwareworkload/astar_grid/workload.py#L138); [detailed decomposition](../docs/decompositions/astar_grid.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Initialize the pending table with the start and its estimated total cost. The settled table is initially empty. Mechanism sources: [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

2. select_k picks the best pending vertex; a full-key mask removes it. If it is the goal, return its path cost; an empty pending table without reaching the goal means failure. Mechanism sources: [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf), [D12 · The Generalized A* Architecture](#d12). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

3. Recover the actual cost already spent by subtracting the heuristic estimate. Save that cost in the settled table. Mechanism sources: [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

4. Fetch the selected vertex's adjacency row and combine it with the actual cost; add each edge cost to form neighbor candidates. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

5. Exclude settled neighbors. Add each remaining neighbor's heuristic estimate to obtain its estimated total cost. Mechanism sources: [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

6. update pending scores with improvements and repeat in the declared priority order. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6), [D12 · The Generalized A* Architecture](#d12). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** s is the start, g the goal, and a/b are neighboring vertices. h is estimated remaining cost; g-score means cost already spent; f-score is their sum. The vertex named g is distinct from the term g-score. The example shows one expansion, not the complete search.

<!-- EXAMPLE:astar_grid -->
```text
input: open f={('s',): 2}; h={a: 1, b: 1, g: 0, s: 2}; outgoing edges={('s', 'a'): 1, ('s', 'b'): 1}
native: {('a',): 2, ('b',): 2}
select: {('s',): 2}
recover_g: {('s',): 0}
neighbor_g: {('a',): 1, ('b',): 1}
result_next_open_f: {('a',): 2, ('b',): 2}
```
<!-- /EXAMPLE:astar_grid -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): The host fetches an adjacency row; global batch selection is not a hardware persistent queue. A true serial dependency cannot be erased by operator sharing. [Local _astar](../softwareworkload/astar_grid/workload.py#L138); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="planning_hadd"></a>

### C.16. planning_hadd — Relaxed-planning h-add and h-max

#### Workload literature and mathematical task

**Papers and reading locations:** [Bonet & Geffner, Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf) — §4 additive and max heuristics; weighted action costs also checked against Fast Downward.

**Input, calculation and answer.** Input: initially true facts and actions with preconditions, effects and costs. Output: estimated cost to reach each fact after ignoring delete effects. An action is usable only when all its prerequisites are reachable; add its own cost to the combined prerequisite cost, then choose the cheapest action reaching each effect. Definition checked against [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf); local scope: [Local _native_heuristic](../softwareworkload/planning_hadd/workload.py#L107).

**Terms, conventions and pitfalls.** Delete relaxation ignores an action's effects that would make facts false. It makes reachability easier to estimate. h-add can charge shared supporting work more than once, so its value is not generally a guaranteed lower bound on a real plan's cost. h-max means maximum across prerequisites followed by addition of action cost, not taking the maximum everywhere. Helper rows indicating a prerequisite exists use a neutral additive cost, so numeric zero is a valid stored row. Generated actions have nonempty prerequisite lists. A general action with no prerequisites needs separate initialization with no prerequisite cost. The current result is not a complete plan. Source roles: mathematical background: [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf); implementation details and limits: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

#### Native key code and work

[Local _native_heuristic](../softwareworkload/planning_hadd/workload.py#L107). [Fast Downward `AdditiveHeuristic::relaxed_exploration`](https://github.com/aibasel/downward/blob/9b81c7e422fdf7be9f73b96e9a7a969c483dd5d4/src/search/heuristics/additive_heuristic.cc) sums reached precondition costs and waits for all prerequisites; [Fast Downward `HSPMaxHeuristic::relaxed_exploration`](https://github.com/aibasel/downward/blob/9b81c7e422fdf7be9f73b96e9a7a969c483dd5d4/src/search/heuristics/max_heuristic.cc) takes their maximum. Local `_native_heuristic` independently relaxes fact costs; it is heuristic evaluation, not a whole planner.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
Initial facts have no additional cost.
For each action whose prerequisites are all reachable:
    prerequisite_cost = sum (h-add) or maximum (h-max) of their costs
    candidate_effect_cost = prerequisite_cost + action_cost
Keep the cheapest candidate for each effect fact.
```

#### Workload example

The prerequisite costs have already been reached. The same action produces different additive and max heuristic values while charging its cost in both cases. The action requires both p and q. The join must recover both costs before the action is usable. h-add sums them; h-max keeps the larger prerequisite cost. Both then add the action cost. The outputs are estimates for the effect g; neither is an executed plan.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf).

#### Decomposition literature and justification

- [D5 · Dyna: Practical Weighted Dynamic Programming](#d5) (§4.2 and Figure 3): Indexed chart/agenda, change propagation, and ordered old/delta/current contributions. Its increment is not automatically our changed-total delta.

- [D3 · FAQ: Questions Asked Frequently](#d3) (FAQ definition; InsideOut; equivalent variable orderings): Eliminate variables by join and aggregation; not all mixed-aggregate reorderings are legal.

**How the foundations connect to this program.** The relational gather and group fold are locally derived from the heuristic recurrence. Mixed max-then-add is explicit; h-max is not a uniform min/max path semiring. Completeness masking prevents an action from firing with missing prerequisites. The precise fields, schedule and state operations below are a local mapping, checked against [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

#### Key lowering code and work

[Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172); [detailed decomposition](../docs/decompositions/planning_hadd.md); primitive implementations: [ops.py](../algebra/ops.py).

1. Initialize true facts with no added cost; other facts are not yet reachable. Mechanism sources: [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

2. combine changed facts with the action-precondition table to find touched actions; then read all prerequisites of each touched action. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](#d5), [D3 · FAQ: Questions Asked Frequently](#d3). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

3. Join current fact costs to those prerequisite rows. Separately identify any action with a missing prerequisite and prevent it from firing. Mechanism sources: [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf), [D3 · FAQ: Questions Asked Frequently](#d3). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

4. For h-add, add prerequisite costs. For h-max, take the largest prerequisite cost. These are alternative evaluations. Mechanism sources: [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

5. Add each action's own cost, group its effects by resulting fact, and retain the lowest candidate cost. Mechanism sources: [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

6. update saved fact costs and repeat only from real improvements. Return proposition costs; a full state heuristic would additionally combine goal costs. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](#d5). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** p and q are prerequisite facts; g is the effect fact; act names an action. sum_fold adds prerequisite costs and max_fold takes their maximum. result_hadd/result_hmax include the action's own cost.

<!-- EXAMPLE:planning_hadd -->
```text
input: costs={('p',): 2, ('q',): 3}; act: p AND q -> g; action cost=1
native: {hadd(g): 6, hmax(g): 4}
join_all_preconditions: {('act', 'p'): 2, ('act', 'q'): 3}
sum_fold: {('act',): 5}
max_fold: {('act',): 3}
result_hadd: {('g',): 6}
result_hmax: {('g',): 4}
```
<!-- /EXAMPLE:planning_hadd -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): This does not return an optimal plan; h-add is not generally admissible. Local generated actions have nonempty prerequisites; empty-precondition actions require separate initialization. [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="cky_parsing"></a>

### C.17. cky_parsing — CKY inside / best-score parsing

#### Workload literature and mathematical task

**Papers and reading locations:** [Goodman, Semiring Parsing](https://aclanthology.org/J99-4004.pdf); [Torch-Struct](https://aclanthology.org/2020.acl-demos.38/) — Goodman §1, Figures 1–3; Torch-Struct CKY implementation.

**Input, calculation and answer.** Input: a sentence and weighted grammar rules that either produce a token or combine two child categories. Output: total parse probability and best parse score. Build answers for short spans before longer spans; consider every permitted split and matching grammar rule. Definition checked against [Semiring Parsing](https://aclanthology.org/J99-4004.pdf); local scope: [Local _cky](../softwareworkload/cky_parsing/workload.py#L116).

**Terms, conventions and pitfalls.** A grammar category names a type of phrase; a binary rule says which two child categories can form a parent. The start category is the category required for the complete sentence. Inside probability adds all compatible parses, not just the most likely tree. A span is a contiguous sentence interval; a chart is the saved table of span/category answers. The optional rule-first schedule first joins left children with rules and caches that table, then joins right children; newly completed spans extend the cache. Inserting only fresh span keys avoids recounting old probabilities. The best-score output is not a reconstructed parse tree or an exact k-best list. Source roles: mathematical background: [Semiring Parsing](https://aclanthology.org/J99-4004.pdf); implementation details and limits: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

#### Native key code and work

[Local _cky](../softwareworkload/cky_parsing/workload.py#L116). [Torch-Struct `CKY.logpartition`](https://github.com/harvardnlp/pytorch-struct/blob/7146de5659ff17ad7be53023c025ffd099866412/torch_struct/cky.py) combines left/right spans with rule tensors in increasing width. Local `_cky` independently loops over span length, start, split and compatible rules.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
Process sentence fragments from short to long.
For every allowed split and compatible grammar rule:
    parse_contribution = left_weight * right_weight * rule_weight
    add it for inside probability, or keep the best parse contribution
Read the start-category score for the complete sentence.
```

#### Workload example

Lexical facts permit several child pairs, but only pairs present in the grammar survive the rule join. Probability aggregation and best-derivation aggregation then give different root scores. Each token admits more than one lexical category. child_pairs first considers their category combinations, but rule_matches keeps only pairs allowed by a binary grammar rule. Multiplying child weights and rule weight gives a complete-parse contribution. Inside adds all valid contributions; best-parse scoring retains the largest. The trace stores scores, not a reconstructed tree.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [Semiring Parsing](https://aclanthology.org/J99-4004.pdf).

#### Decomposition literature and justification

- [D4 · Semiring Parsing](#d4) (§1, Figures 1–3; semiring framework): Deduction antecedents multiply; alternative proofs add; inside versus best derivation changes the value algebra.

- [D5 · Dyna: Practical Weighted Dynamic Programming](#d5) (§4.2 and Figure 3): Indexed chart/agenda, change propagation, and ordered old/delta/current contributions. Its increment is not automatically our changed-total delta.

- [D3 · FAQ: Questions Asked Frequently](#d3) (FAQ definition; InsideOut; equivalent variable orderings): Eliminate variables by join and aggregation; not all mixed-aggregate reorderings are legal.

**How the foundations connect to this program.** This is a direct literature recurrence; replacing its contractions by named joins is local lowering. The implementation's endpoint re-keying enforces the current span length before joining; rule-first and pair-first orders trade intermediate size. The precise fields, schedule and state operations below are a local mapping, checked against [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

#### Key lowering code and work

[Local _parse](../softwareworkload/cky_parsing/workload.py#L192); [detailed decomposition](../docs/decompositions/cky_parsing.md); primitive implementations: [ops.py](../algebra/ops.py).

1. combine sentence tokens with lexical rules to initialize all single-token spans. Mechanism sources: [D4 · Semiring Parsing](#d4). Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

2. For each target span length, rename the left child's end as the split point and retain its start and category. Mechanism sources: [D4 · Semiring Parsing](#d4). Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

3. scan the right-child table: use its start as the split point and subtract the target length from its end to compute the required parent start; discard impossible endpoints. Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

4. combine left and right children on both parent start and split point. This enforces adjacency and target length before producing child-category pairs. Mechanism sources: [D1 · Provenance Semirings](#d1), [D4 · Semiring Parsing](#d4). Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

5. combine those pairs with compatible binary grammar rules. Add contributions for total probability or keep the best log-score for the best-score evaluation. Mechanism sources: [D4 · Semiring Parsing](#d4). Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

6. scan attaches the parent's end; update inserts only newly computed spans of this length. Finish by reading the sentence-wide start category. Mechanism sources: [D4 · Semiring Parsing](#d4), [D5 · Dyna: Practical Weighted Dynamic Programming](#d5). Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** A/B/C/D are grammar categories and S is the start category; a/b are sentence tokens. A lexical row gives interval start, interval end and category. A rule key gives parent, left child and right child. child_pairs includes all considered pairs; rule_matches retains compatible rules; inside is the summed parse probability.

<!-- EXAMPLE:cky_parsing -->
```text
input: sentence=a b; left lexical={(0, 1, 'A'): 1, (0, 1, 'C'): 1}; right lexical={(1, 2, 'B'): 1, (1, 2, 'D'): 1}; rules={('S', 'A', 'B'): 3/5, ('S', 'C', 'D'): 2/5}
native: {inside(S): 1, viterbi_probability(S): 3/5}
child_pairs: {(0, 2, 'A', 'B'): 1, (0, 2, 'A', 'D'): 1, (0, 2, 'C', 'B'): 1, (0, 2, 'C', 'D'): 1}
rule_matches: {(0, 2, 'A', 'B', 'S'): 3/5, (0, 2, 'C', 'D', 'S'): 2/5}
result_inside: {(0, 2, 'S'): 1}
```
<!-- /EXAMPLE:cky_parsing -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): The local default reports chart/root scores, not a reconstructed parse tree. Unary cycles, epsilon rules and exact k-best derivations need additional machinery. [Local _parse](../softwareworkload/cky_parsing/workload.py#L192); existing measurements are separately reported in [baseline report](baseline_report.md).

<a id="graph_bfs_sssp"></a>

### C.18. graph_bfs_sssp — BFS and single-source shortest paths

#### Workload literature and mathematical task

**Papers and reading locations:** [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619) — benchmark-kernel definitions: BFS and SSSP.

**Input, calculation and answer.** Input: a graph and source; weighted shortest paths also use nonnegative edge costs. BFS returns minimum hop counts, while SSSP returns minimum total edge cost. A path with fewer hops need not have a smaller weighted cost. Definition checked against [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619); local scope: [Local native](../softwareworkload/graph_bfs_sssp/workload.py#L58).

**Terms, conventions and pitfalls.** Relaxing an edge means checking whether reaching its source and traversing that edge gives a cheaper destination distance, and updating the distance if so. A heap is a data structure for repeatedly retrieving a smallest pending score. Frontier means the vertices or distances to expand in the current round. BFS levels and weighted distances are different answers. Dijkstra's settled-state rule relies on nonnegative edges. The independent native code uses a BFS queue and a Dijkstra heap. Source roles: mathematical background: [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619); implementation details and limits: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

#### Native key code and work

[Local native](../softwareworkload/graph_bfs_sssp/workload.py#L58). [GAPBS `DOBFS` / `TDStep` / `BUStep`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/bfs.cc) switches traversal direction; [GAPBS `DeltaStep`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/sssp.cc) uses delta-stepping buckets. Local `native` uses queue BFS and heap Dijkstra; `_sssp_frontier` and `_sssp_dijkstra_select` are distinct operator schedules.

The following is a reading outline of the native algorithm, not runnable source; the preceding paragraph links real local functions and pinned upstream code.

```text
BFS: visit newly discovered neighbors with a queue and record their levels.
Dijkstra: remove the smallest pending distance from a heap,
          then try to improve the distances of outgoing neighbors.
```

#### Workload example

A direct edge is best by hop count but worse by weighted distance. The second SSSP round improves an existing key; Boolean anti-visited logic would lose that improvement. BFS discovers c immediately because a direct edge exists; it does not inspect that edge's cost when counting hops. Weighted SSSP first records the direct-edge cost, then improves c through b. changed must contain the improved distance even though c was already discovered. A BFS visited mask cannot be reused as the SSSP improvement test.

The example is locally constructed and script-derived, not copied experimental data: [documentation_examples.py](../sym1/scripts/documentation_examples.py). The underlying task basis is [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619).

#### Decomposition literature and justification

- [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6) (Generic-Single-Source-Shortest-Distance; topological variant): Path extension, aggregation and queue discipline are separate; termination depends on graph/algebra assumptions.

- [D8 · GraphBLAS C API Specification (standard)](#d8) (`GrB_mxm`, `GrB_vxm`, reduction, mask/accumulator semantics): Sparse contraction, structural masks and accumulation have an established interface; our arbitrary-arity relations are a local generalization.

**How the foundations connect to this program.** Boolean frontier expansion and min-plus relaxation are established sparse contractions. BFS masks visited support; SSSP must preserve numerical improvements. Dijkstra additionally selects a minimum and maintains settled state under nonnegative edge weights. The precise fields, schedule and state operations below are a local mapping, checked against [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

#### Key lowering code and work

[Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111); [detailed decomposition](../docs/decompositions/graph_bfs_sssp.md); primitive implementations: [ops.py](../algebra/ops.py).

1. BFS: initialize the new-vertex and visited tables with the source. combine new vertices with outgoing edges, mask out visited vertices, update visited state and record the discovery round as hop count. Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](#d8), [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619). Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

2. Repeat BFS until no new vertex is found. Each vertex's first discovery determines its level. Mechanism sources: [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619). Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

3. Frontier SSSP: start the source at no added cost. combine improved distances with edges, add edge costs and keep the lowest candidate for each destination. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6), [D8 · GraphBLAS C API Specification (standard)](#d8). Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

4. update all genuinely better distances and repeat. Do not filter merely by previously visited support, because an existing distance can improve. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6). Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

5. Dijkstra is an alternative SSSP schedule: select the lowest pending distance, remove it, settle that vertex, expand its adjacency row, exclude settled destinations and update pending distances. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](#d6). Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

6. Run each chosen schedule with its own state and stopping condition; they are alternatives, not stages to concatenate. Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

#### Decomposition example: derived intermediate values

**Field and short-name guide.** An edge key lists start and end and its value is edge cost. BFS_hops counts traversed edges; SSSP_distance sums their costs. first/second name successive relaxation rounds; changed contains the new total distance for an improved vertex.

<!-- EXAMPLE:graph_bfs_sssp -->
```text
input: edges={('a', 'b'): 2, ('a', 'c'): 5, ('b', 'c'): 1}; source=a
native: {BFS_hops(c): 1, SSSP_distance(c): 3}
BFS_first_frontier: {('b',): True, ('c',): True}
SSSP_first: {('b',): 2, ('c',): 5}
SSSP_second: {('c',): 3}
changed: {('c',): 3}
result: {('b',): 2, ('c',): 3}
```
<!-- /EXAMPLE:graph_bfs_sssp -->

Data and verification source: [documentation_examples.py](../sym1/scripts/documentation_examples.py); input is constructed, native is independently computed, and the other calculation rows use local primitives. The preceding example explanation states how the intermediates arise.

Boundary / work to profile (local analysis, not a paper's measured result): Native direction optimization and bucket size can dominate work and synchronization. This decomposition alone does not explain all CPU scaling or prove shared-hardware benefit. [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111); existing measurements are separately reported in [baseline report](baseline_report.md).

## D. What this closes, and what remains

The two chains above cover the current workload inventory and distinguish direct recurrences, derived relational mappings and physical lowering precedents. The examples check teaching-case semantics, not the completeness of every implementation branch.

Still required for the paper: full Ctrl-G completion inference if claimed; an actual learned NBFNet workload if claimed; circuit validity for probabilistic/WMC claims; matched optimized native algorithms; all relevant control and state costs; and cycle/RTL evidence for sharing. No new performance or hardware claim follows from this handbook. Use the [validated baseline report](baseline_report.md) for measured results.
