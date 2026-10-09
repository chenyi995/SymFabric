# sym1 — shared hardware and spatial execution for weighted-relation operators

Project code name: `sym1`. First paper of the SymFabric line; target venue: MICRO.

chenyi9 ruled (2026-09-26): take operator-decomposition mathematics as established background; investigate shared hardware and spatial execution. This remains the project direction. The memory-side discussion motivates bottleneck placement.

This revision corrects the original plan against primary papers, upstream code and local workload implementations. It separates a research hypothesis from completed evidence. The [literature audit](literature_review.md), [validated baseline report](baseline_report.md), [source inventory](upstream_status.md) and [pinned revisions](upstream_lock.json) supply the evidence trail. The complete local Chinese counterpart is [sym1-plan.zh.md](../local/sym1-plan.zh.md).

Reading guide: §0–2 give the claim and interface; §3 reviews prior work; §4 specifies every current workload mathematically; §5 connects profiling to bottlenecks; §6 defines fair evaluation; §7 records implementation status and corrections.

For the two complete evidence chains—workload literature/native code/example and decomposition literature/lowering code/operator example—read the [workload and operator-decomposition handbook](workload_operator_evidence.md) ([Chinese](../local/sym1-workload-operator-evidence.zh.md)). It links pinned upstream functions, separates published mechanisms from local derivations, and provides script-checked teaching traces for every workload below. The examples are derived illustrations, not additional baseline measurements.

Reading convention: established literature mechanisms, local code behavior, teaching derivations and untested hardware hypotheses are labelled separately. Calculation descriptions avoid abstract symbols; retained code/example names are explained in the companion.

## 0. Research hypothesis and evidence boundary

**Hypothesis, not an established result.** A spatial engine that shares arithmetic, comparison, buffering and state-access resources across weighted joins, reductions, masks, updates and ordered selection may use a fixed area budget more effectively than a fixed mixture of specialized engines. Fusing these operators may additionally reduce intermediate materialization and repeated software dispatch. Those are different mechanisms and need separate experiments.

Weighted relational computation, semiring inference, hardware relational pipelines, runtime-configurable sparse engines, feedback execution and query batching all have precedents (§3). The novelty must therefore be an implemented, quantitatively justified combination of resource sharing and stateful spatial execution, rather than the first common algebra, first relational interface, or first accelerator with dynamic control.

The existing repository establishes executable decompositions and independent reference paths. Its closed-form architecture model is an exploratory model, not cycle-level validation. Upstream CPU runs and primitive simulations identify several bottlenecks (§5), but do not establish sym1 speedup, area efficiency, frequency, power or energy.

Proposed contributions, conditional on the required evidence:

1. A precisely specified mapping interface, with value semantics, state ownership, work accounting and coverage against independent goldens.
2. A realizable shared datapath and memory/control organization for fused operator chains, including feedback, contention and capacity handling.
3. Matched-algorithm comparisons that isolate physical sharing from fusion, scheduling and avoided work, with cycle-level validation and synthesized hardware costs.

## 1. Paper frame and argument

### 1.1 Motivation without overstating the measurements

Compositional neural/symbolic pipelines can spend substantial time in symbolic or probabilistic stages; this motivates measuring the complete pipeline rather than assuming neural acceleration solves the application. The relevant primary characterization is [Towards Cognitive AI Systems](https://research.ibm.com/publications/towards-cognitive-ai-systems-workload-and-characterization-of-neuro-symbolic-ai), [CogSys](https://zishenwan.github.io/publication/HPCA25_CogSys.pdf) and [REASON](https://arxiv.org/html/2601.20784v1). Their workload populations and timing boundaries differ; their percentages must not be pooled or transferred to this repository.

For an actually measured symbolic fraction f and symbolic speedup s, Amdahl's law gives the **derived** bound:

```text
end-to-end speedup = 1 / ((1 - f) + f / s)
```

This expression assumes the remaining execution time is unchanged. Transfers, queueing and additional control can invalidate that assumption.

Low CPU scaling is a symptom, not a universal root cause. Narrow frontiers, barriers, excess algorithmic work, indirect accesses, uneven degrees and sequential dependencies produce different interventions. The local triangle-counting baseline scales across CPU threads, while road-graph SSSP suffers from waiting and fine-grained rounds. Neither observation proves a general property of all symbolic workloads.

A shared engine is useful only when its flexibility outweighs muxing, routing, register, queue, bank and scheduling costs. Sharing does not by itself shorten a true dependency chain or repair an unnecessarily expensive join order. Conversely, an optimized CPU or GPU may already exploit the relevant locality and parallelism.

### 1.2 Paper organization

| Paper section | Content | Evidence required |
|---|---|---|
| Introduction | Application motivation and falsifiable sharing hypothesis | Matched pipeline profile; no borrowed universal bottleneck percentages |
| Background | Weighted relations, inference, recursion and ordered search | Established theory and explicit semantic conditions |
| Characterization | Per-workload work, latency, memory, synchronization and shape | Native optimized baselines and clearly labelled proxies |
| Interface | Operators, value types, state and accounting | Executable semantics and independent goldens |
| Architecture | Shared resources, lanes, banks, queues, feedback and reconfiguration | Cycle model and RTL rather than an operator sketch |
| Mapping | Join/elimination order, sparsity/layout, fusion and agenda scheduling | Legal transformations and work-preservation checks |
| Methodology | Baselines, resource matching, measurement boundaries | Reproducible inputs, versions, configuration and raw logs |
| Evaluation | Sharing, fusion, scheduling, area/frequency/energy and coverage | Ablations, negative results and genuinely held-out mappings |
| Related work | Theory, hardware, optimized software and system context | Primary-source comparisons below |

### 1.3 Questions the evidence must answer

A shared versus dedicated comparison must hold the algorithm, output semantics, fusion capability and memory budget constant. A SAM-like configuration cannot stand in for Revet merely by disabling a queue; a simplified arithmetic-DAG model cannot stand in for REASON while omitting its BCP feedback and batch interface. When a faithful reproduction is unavailable, name the comparator by the capabilities actually modeled.

The generality tax includes value width, nonlinear numerical units, key handling, persistent state and timing closure. A single scalar back-pointer is not the same cost or semantics as a proof set. Query batching must report single-query latency and replicated state as well as throughput.

## 2. Interface, algebra and work accounting

### 2.1 A common calculation pattern and its limits

Many workloads use the same three steps: match compatible records, calculate a contribution for each match, then combine contributions belonging to the same answer. The matching fields decide which inputs can work together; the chosen value rules decide what the answer means.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2), [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3); exact behavior: [ops.py](../algebra/ops.py).)

| Calculation | Extend one candidate | Combine alternative candidates | Meaning of a missing row | Sources |
|---|---|---|---|---|
| Reachability | Both pieces must exist | Keep the answer if any candidate exists | False | [D1 · Provenance Semirings](workload_operator_evidence.md#d1) |
| Counting | Multiply the component counts | Add candidate counts | Zero | [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [D4 · Semiring Parsing](workload_operator_evidence.md#d4) |
| Ordinary probability | Multiply component probabilities | Add alternative contributions under a valid factorization | Zero probability | [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2) |
| Real-valued dot product | Multiply matching coordinates | Add products | Zero | [D8 · GraphBLAS C API Specification (standard)](workload_operator_evidence.md#d8) |
| Log probability | Add log probabilities | Use log-sum-exp | Negative infinity, representing zero probability | [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2), [D4 · Semiring Parsing](workload_operator_evidence.md#d4) |
| Best log-score / Viterbi | Add component log-scores | Keep the largest score | Negative infinity | [D4 · Semiring Parsing](workload_operator_evidence.md#d4), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6) |
| Shortest distance / min-plus | Add path costs | Keep the smallest cost | Positive infinity, meaning unreachable | [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6) |
| Minimum bottleneck cost | Keep the larger component cost along one path | Keep the smallest path cost among alternatives | Positive infinity | [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6) |
| Fuzzy path strength | Keep the weaker link along one path | Keep the strongest alternative path | Zero strength | [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6) |

Log-sum-exp, abbreviated LSE, means adding the probabilities represented by log values and returning the logarithm of that sum; a stable implementation avoids explicitly reconstructing very small probabilities. The paired rules are often called a **semiring**. This existing mathematical framework is background, not a sym1 invention. A workload may still need filters, mutable state, ordering and several different calculation rules.  (Mechanism sources: [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2), [D4 · Semiring Parsing](workload_operator_evidence.md#d4); exact behavior: [ops.py](../algebra/ops.py).)

A backpointer records which earlier choice produced a result; a proof tag records supporting evidence. Neither is specified by a scalar score alone. Selecting a few rows is not automatically exact k-best derivation search. Planning h-max takes the maximum prerequisite cost and then adds the action cost, so it does not simply use the bottleneck row throughout. [D11 · Better k-best Parsing](workload_operator_evidence.md#d11); [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf) (§4); [HMM tagged values](../softwareworkload/hmm_viterbi/workload.py).

For repetition, keep three meanings separate: newly present keys, improved values at existing keys, and numerical differences. Finite Boolean reachability with only new facts added eventually stops. Even if merging a value with itself leaves it unchanged, an unlimited sequence of strictly improving values need not end. Shortest-path convergence depends on the graph and weights; bounded dynamic programming stops because its length is fixed. Repeatedly adding previously counted contributions overcounts ordinary probabilities or counts. Such recursive calculations require explicit contribution/difference accounting and a termination argument. See [semiring Datalog convergence](https://arxiv.org/abs/2105.14435) and [DBSP](https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf).  (Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6), [D7 · Convergence of Datalog over (Pre-) Semirings](workload_operator_evidence.md#d7), [D13 · DBSP: Automatic Incremental View Maintenance](workload_operator_evidence.md#d13); exact behavior: [control.py](../algebra/control.py).)

### 2.2 Operators and implementation obligations

| Operator | Mathematical contract | Candidate implementation; cost that must be included |
|---|---|---|
| `combine` | Natural join on common keys, multiply values, aggregate onto retained keys | Dense contraction, sorted intersection or indexed probe; ordering/index construction and output aggregation |
| `reduce` | Group and fold with a declared operation | Segmented/tree reduction; group boundaries, arbitrary fan-in and zero semantics |
| `filter` / `scan` | Predicate / elementwise transform, possibly re-keying | ALU and routing; re-keying may invalidate ordering |
| `mask` | Semijoin / antijoin using support | Merge, bitmap or hash lookup; support versus value comparison must be explicit |
| `update` | Combine values and return genuinely changed keys | Atomic or ordered read-modify-write; bank conflicts, duplicates and change detection |
| `select_k` | Ordered selection under an explicit score and tie order | Reduction, heap or queue; persistent insertion/deletion is stronger than batch selection |
| `fixpoint` / `iterate` | Convergent feedback / bounded iteration | State ownership, delta buffers, feedback credits and termination detection |

A comparator, arithmetic unit and FIFO may be shared, but this does not establish that merge join, hash-table updates and a persistent priority queue are the same cell with one mode bit. Dense mode also needs a feeding/reuse strategy; LSE needs numerical support beyond a multiplier-adder. Queue spill, state migration and switching latency remain design parameters requiring evidence.

### 2.3 Comparable work counts

Report a vector of semantic events: key probes/comparisons, successful join pairs, value products, value accumulations, duplicate candidates, changed outputs, scanned coordinates, queue operations and bytes at each measured memory level. Instrument each event once at its semantic boundary.

For one matched event type, the **derived** work-inflation ratio is:

```text
work-inflation ratio for an event type = mapped implementation's event count / reference algorithm's event count
```

State the event and reference algorithm. If the denominator is zero or algorithms expose incomparable events, report raw counts rather than a ratio. A Python hash lookup, a sparse-coordinate comparison and a semiring multiplication are not interchangeable units. There is no universal acceptable inflation threshold.

Separate correctness goldens from performance baselines. An independent Python golden can establish the answer but not competitive execution speed. Integer/Boolean outputs can be exact; floating-point output comparison requires a declared numerical contract. Changes in reduction order, precision and tie-breaking must be visible.



### 2.4 What each operator reads, does and returns

The names below match [ops.py](../algebra/ops.py), [relation.py](../algebra/relation.py) and [control.py](../algebra/control.py). They describe software behavior, not a proposed physical circuit.

**Start with a table.** A relation is a table whose rows have named identifying fields and an attached value. The identifying fields are the **key**; their ordered names are the **schema**. **Support** means the keys currently represented. For example, a path row has a start vertex, an end vertex and a value saying whether the path exists or how much it costs.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1); exact behavior: [relation.py](../algebra/relation.py).)

An omitted row uses the selected calculation's “no result” value. This is false for reachability, zero for ordinary probability, positive infinity for shortest distances, and negative infinity for log probabilities. Positive infinity means no finite route is known; negative infinity is the log representation of zero probability. Thus a stored cost of zero can be a real, valid row. A **semiring** is the mathematical name for a compatible pair of rules: how to extend one candidate and how to combine alternatives. The operator descriptions below spell out those rules instead of using special symbols.  (Mechanism sources: [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6); exact behavior: [ops.py](../algebra/ops.py).)

**combine: match rows, compute each contribution, combine equal answers.**

1. Find rows whose shared named fields agree.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1); exact behavior: [ops.py](../algebra/ops.py).)

2. Combine the values of each matching pair using the selected candidate-extension rule.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1); exact behavior: [ops.py](../algebra/ops.py).)

3. Keep the requested identifying fields. Rows with the same retained key contribute to the same answer.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1); exact behavior: [ops.py](../algebra/ops.py).)

4. Combine those contributions using the selected alternative rule.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1); exact behavior: [ops.py](../algebra/ops.py).)

For a path example, match the end of a known path with the start of an edge. Keep the original start and the edge's end. Reachability asks whether any match exists; shortest paths add the two costs and keep the smallest candidate cost; probability calculations multiply contributions and add alternative contributions. Both input tables must use the same value rules. The left/right operand order is preserved. If there are no shared fields, every row on one side pairs with every row on the other; this can be expensive.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6); exact behavior: [ops.py](../algebra/ops.py).)

**reduce / project: group rows and combine their values.** Keep selected identifying fields, collect rows with the same retained key, then apply the declared rule, such as addition, minimum or maximum. In this API, project is another name for the default reduce. It is not merely deleting columns without combining duplicates.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3); exact behavior: [ops.py](../algebra/ops.py).)

The special option use_mul=True applies the candidate-extension rule within each group. It visits only stored members: it neither creates empty groups nor recovers missing zero-valued children. A product requiring every child must check that all children are present or join separately gathered child slots. Otherwise a missing zero child could incorrectly be ignored.  (Mechanism sources: local interface definition; exact behavior: [ops.py](../algebra/ops.py).)

**filter: keep rows meeting a condition.** The condition may inspect both the identifying fields and the value. Retained rows keep their value.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1); exact behavior: [ops.py](../algebra/ops.py).)

**rename: change field names, not the data.** Renaming the end field to middle, for example, lets the next combine match it with another table's middle field. Renaming is simultaneous, must not collide field names, and performs no value aggregation.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1); exact behavior: [ops.py](../algebra/ops.py).)

**mask: keep rows with a matching key in a second table.** Only existence in the second table matters; its weights do not multiply the first table's values. By default the common fields define the match; on= explicitly names the matching fields. With negate=True, keep rows that have no match; this is an **antijoin**. It can filter already visited vertices, but it cannot decide whether a new shortest-path cost improves an existing one. It also does not implement a constant-cost persistent queue deletion.  (Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](workload_operator_evidence.md#d8); exact behavior: [ops.py](../algebra/ops.py).)

**scan: perform a small change on each row.** A callback is the small function applied to a row. It can change a value, rename/rebuild a key or emit a bounded number of rows. Duplicate output keys are combined with the output table's declared rule. Literal negation, attaching a discovery level and adjusting an endpoint are examples. An entire sort, variable-length search or unbounded group calculation may not be hidden in the callback. Changing the value rules must be explicit; keeping the same numeric values does not make arbitrary reorderings safe. This scan is a row-wise map, not a parallel prefix scan.  (Mechanism sources: local interface definition; exact behavior: [ops.py](../algebra/ops.py).)

**update: merge candidates into saved state and report actual changes.** Saved and candidate tables must have the same identifying fields and value rules. For each candidate, combine its value with the saved value; insert a new key or save an actual change. Return both the updated table and a table containing only changed keys. The changed table is also called **delta**. Its values are the new totals, not the numerical difference between the new and old values.  (Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9); exact behavior: [ops.py](../algebra/ops.py).)

The implementation uses a tolerance when comparing floating-point values. A value judged unchanged is not written back, so very small additive contributions can be dropped. Boolean discovery, improved minimum/maximum scores and additive differences therefore need different update reasoning. Repeatedly adding old probability/count totals is not a valid generic recursive calculation. After cancellation, the current update can leave an explicit zero in saved state while omitting it from the returned changed table; this is not a general deletion notification.  (Mechanism sources: [D13 · DBSP: Automatic Incremental View Maintenance](workload_operator_evidence.md#d13); exact behavior: [ops.py](../algebra/ops.py).)

**select_k: retain a requested number of best rows.** The parameter k is the number of rows to keep. The caller specifies whether larger or smaller scores are better. Equal scores are resolved by the complete key in a deterministic order; values are unchanged. The implementation selects globally across its input. It does not directly return ranks, select separately within groups or maintain a persistent priority queue.  (Mechanism sources: [D11 · Better k-best Parsing](workload_operator_evidence.md#d11), [D12 · The Generalized A* Architecture](workload_operator_evidence.md#d12); exact behavior: [ops.py](../algebra/ops.py).)

To select separately for each query, the caller first partitions or filters rows by query and invokes select_k on each group. KG currently partitions in host code; VSA filters per query. Sorting ranks, saving parents and following parent pointers are separate work.  (Mechanism sources: local interface definition; exact behavior: [ops.py](../algebra/ops.py).) Call sites: [KG](../softwareworkload/kg_multihop/workload.py), [VSA](../softwareworkload/vsa_cleanup/workload.py).

**iterate: repeat a declared step a fixed number of times.** With no iterations, return the initial state. Each step must state its inputs, outputs and auxiliary state.  (Mechanism sources: local interface definition; exact behavior: [control.py](../algebra/control.py).)

**fixpoint: repeat until nothing changes.** Each step reads accumulated state and the last changed table, then produces the next state and changed table. Stop when the changed table is empty, provided the update rule correctly represents progress. Hitting max_rounds, the configured iteration limit, is not a convergence proof; the current driver can return at that limit. Watches, queues, old-state snapshots and parsing tables remain explicit state.  (Mechanism sources: [D7 · Convergence of Datalog over (Pre-) Semirings](workload_operator_evidence.md#d7), [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9); exact behavior: [control.py](../algebra/control.py).)

**Support-only helper tables.** Sometimes only row existence should matter during combine. Assign each helper row the selected extension rule's “do not change the other value” value: one for ordinary multiplication, zero for addition of costs/log-scores. Do not assume that every presence marker must be a nonzero numeric value. Boolean table union and insertion of fresh, nonoverlapping keys are safe uses of update; repeatedly merging an old probability result is a different operation.  (Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2); exact behavior: [ops.py](../algebra/ops.py).)

### 2.5 How those steps form a workload

Describe a workload as an executable recipe, with named tables instead of a long formula:

1. **Initialize:** name every input table, seed row and initial saved state.  (Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6); exact behavior: [control.py](../algebra/control.py).)

2. **Compute a step:** state which table each operator reads, which fields it matches or groups, which value calculation it performs, and where the result is stored.  (Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6); exact behavior: [control.py](../algebra/control.py).)

3. **Choose control:** execute once in dependency order; repeat for a fixed length; repeat until no rows change; or repeatedly choose the best pending item.  (Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6); exact behavior: [control.py](../algebra/control.py).)

4. **Read the answer:** identify the returned rows, values, path or conflict flag.  (Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6); exact behavior: [control.py](../algebra/control.py).)

A dependency-ordered program is often called a DAG: a graph of operations with no cycle. A feedback program sends new changes into another iteration. Ordered search instead stops under its declared condition, such as reaching the goal or emptying its pending queue.  (Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6); exact behavior: [control.py](../algebra/control.py).)

These are not interchangeable controls. A join reads multiple tables; update returns multiple tables; a filter reads one table. Keep those inputs and outputs explicit. Static inputs can be reused between iterations, but reuse and storage costs are not free. Section 4 spells out each workload's recipe rather than hiding its computation inside the word “iteration.”  (Mechanism sources: local interface definition; exact behavior: [ops.py](../algebra/ops.py).)

## 3. Related work and implications for sym1

The review covers the comparison axes required by this project. “Relevant” does not mean reproduced, and a paper's omitted primitive does not prove that its architecture cannot support an extension. Links identify primary papers or official project artifacts. Publication years below are bibliographic metadata, not experimental measurements.

### 3.1 Mathematical and language foundations

| Work | Established contribution | Consequence for this plan |
|---|---|---|
| [Aji–McEliece, generalized distributive law](https://authors.library.caltech.edu/records/sw1pm-bwj40) | Distributive factor computations unify inference and decoding | Sum/product factor contraction is background; elimination legality matters |
| [Dechter, bucket elimination](https://www.sciencedirect.com/science/article/pii/S0004370299000594) | Common elimination framework for probabilistic and deterministic reasoning | Factor growth and elimination order precede hardware optimization |
| [FAQ / InsideOut](https://arxiv.org/abs/1504.04044), [AJAR](https://dl.acm.org/doi/10.1145/2902251.2902293) | Aggregate-join planning with algebraic ordering constraints | Compare good join/elimination orders, not only pairwise materialization |
| [LMFAO](https://arxiv.org/abs/1906.08687) and factorized evaluation | Share work across aggregates and avoid repeated materialization | Multi-query reuse is an existing software idea; physical resource sharing is a separate claim |
| [Dyna](https://aclanthology.org/H05-1036/), [finite arithmetic-circuit solver](https://www.cs.jhu.edu/~jason/papers/filardo%2Beisner.iclp12.pdf) | Weighted deduction, agenda execution and maintained derived values | Weighted rules and agenda control are precedents |
| [Goodman, semiring parsing](https://aclanthology.org/J99-4004/), [Huang–Chiang, k-best parsing](https://aclanthology.org/W05-1506/) | Algebraic parsing and ranked derivation extraction | CKY scalar scores, back-pointers and k-best structures are different contracts |
| [Green–Karvounarakis–Tannen, provenance semirings](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf), [first-order provenance](https://arxiv.org/abs/1712.01980) | Annotated relations track how answers are derived | A scalar confidence cannot replace provenance with conjunction/conflict semantics |
| [Datalog over (pre-)semirings](https://arxiv.org/abs/2105.14435) | Convergence and semi-naive evaluation depend on algebraic conditions | State the conditions per recursive workload |
| [DBSP](https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf), [Differential Dataflow](https://github.com/TimelyDataflow/differential-dataflow) | Explicit incremental differences and maintained computations | Changed-key feedback is not general non-monotone incremental evaluation |
| [GraphBLAS](https://graphblas.org/), [Semiring Programming](https://arxiv.org/abs/1609.06954), [Tensor Logic](https://arxiv.org/abs/2510.12269) | Algebraic graph computation and unification of inference/rules | Algebra/interface unification alone is not a new hardware contribution |

[Factor graphs and the sum-product algorithm](https://www.isiweb.ee.ethz.ch/papers/arch/aloe-2001-1.pdf) provide the message-passing view; [generalized A*](https://arxiv.org/abs/1110.2216) supplies ordered inference over AND/OR structure; [TensorLog](https://www.cs.cmu.edu/~wcohen/pubs-g.html) compiles logical inference into differentiable factor/message computations. Junction trees, valuation algebras, generalized Dijkstra and weighted automata form the broader lineage. Their role is factorization, legal elimination, ordered propagation or derivation extraction; this plan does not assign an unverified hardware capability to that lineage.

### 3.2 Closest hardware

| Work | What the primary source establishes | Comparison that matters |
|---|---|---|
| [REASON, HPCA 2026](https://arxiv.org/html/2601.20784v1) | Combined probabilistic/logical execution; BCP feedback FIFO and batching; clause access can reach DRAM | Same probabilistic/logical tasks, feedback state, clause traffic and batch semantics; not a static acyclic-only strawman |
| [PIU, ISSCC 2021](https://ieeexplore.ieee.org/document/9366061), [DPU-v2, MICRO 2022](https://arxiv.org/abs/2210.13184) | Irregular inference/arithmetic-DAG specialization; DPU-v2 exploits static connectivity with compiler scheduling | Dependency depth, issued useful nodes, register/bank pressure and numerical precision |
| [CogSys, HPCA 2025](https://zishenwan.github.io/publication/HPCA25_CogSys.pdf) | VSA-oriented characterization and co-designed PE/data movement; GEMM and circular-convolution modes | Reuse its profiling method; compare actual supported computations and codebook movement |
| [NSFlow, DAC 2025](https://arxiv.org/abs/2504.19323) | FPGA neuro-symbolic dataflow framework, reconfigurable compute and memory organization | Generator/runtime capabilities and mixed-precision assumptions; distinct from CogSys |
| [SIMD², ISCA 2022](https://arxiv.org/abs/2205.01252) | Matrix instructions beyond ordinary multiply-add | Dense shape utilization and actual value-operation hardware |
| [GraphLily, ICCAD 2021](https://github.com/cornell-zhang/GraphLily) | HBM FPGA GraphBLAS overlay; semiring operation passed at runtime | Sparse/dense vector switching, masking, state traffic and overlay mode selection |
| [SAM, ASPLOS 2023](https://arxiv.org/html/2208.14610v2), [Stardust, CGO 2025](https://compilers.stanford.edu/publications/cgo25stardust/) | Sparse token streams, scanners, intersections, reduction and compilation to reconfigurable dataflow | Coordinate work, backpressure, fusion and mapping; local SAM primitive runs are not Onyx silicon measurements |
| [Capstan, MICRO 2021](https://arxiv.org/abs/2104.12760), [Revet, HPCA 2024](https://arxiv.org/html/2302.06124v2) | Sparse-access machinery and hierarchical dataflow threads with branches/loops; Revet discusses Aurochs hash-join/index precedents | Dynamic control, sparse scheduling, live-state communication and memory issue constraints |
| [Flexagon, ASPLOS 2023](https://arxiv.org/abs/2301.10852), [Trapezoid, ISCA 2024](https://yang-yifan.github.io/papers/isca24_trapezoid.pdf), [ExTensor, MICRO 2019](https://www.kartikhegde.com/papers/extensor) | Adaptive sparse dataflows, dense/sparse datapath reuse and hierarchical intersection | Strong precedents for resource/dataflow flexibility and skipped work; match sparsity, layout and area |
| [Q100, ASPLOS 2014](https://arcade.cs.columbia.edu/q100-asplos14.pdf) | Relational operator tiles and pipelined spatial execution | Direct precedent for a relational hardware interface; compare specialization versus shared resources |
| [HWPQ, OSCAR 2025 artifact](https://github.com/realise-lab/hwpq) | Hardware priority-queue architectures and implementations | Queue area, depth, insertion/removal rate and persistent-state semantics |
| [PolyGraph, ISCA 2021](https://web.cs.ucla.edu/~tjn/papers/isca2021-polygraph.pdf) | Graph algorithm/schedule flexibility with taskflow execution | Work efficiency versus throughput and phase-dependent scheduling |
| [RAP, ISCA 2025 paper](https://par.nsf.gov/servlets/purl/10626450) | Reconfigurable automata-processing architecture | Automata representation and utilization; comparison details require the paper configuration |

Historical systolic relational cells and priority queues are precedents for the individual operations, not evidence that all modes share economically. A fixed topology can still have data-dependent execution time and feedback. FPGA partial reconfiguration is different from selecting an already configured overlay mode; ICAP latency cannot rule out runtime operator switching.

### 3.3 Optimized software and workload provenance

| Family and primary references | Native mathematical/execution structure | Local mapping and fair-baseline condition |
|---|---|---|
| [Souffle](https://souffle-lang.github.io/profiler), [GPUlog](https://sidharthkumar.io/publications/asplos2025.pdf), [VFLog](https://ojs.aaai.org/index.php/AAAI/article/view/33665) | Indexed recursive relations, delta joins and duplicate elimination | Datalog kernels; paper GPUlog and checkout `gdlog` are not independent baselines |
| [SRDatalog](https://arxiv.org/html/2604.20073v1), [FlowLog](https://arxiv.org/abs/2511.00865) | Worst-case-optimal GPU joins; incremental dataflow with structured query planning | Include binary-intermediate growth, skew, index/delta costs and subplan reuse; identify exact artifact revision |
| [Scallop](https://arxiv.org/abs/2304.04812), [Lobster](https://arxiv.org/html/2503.21937v2), [DOLPHIN](https://proceedings.mlr.press/v267/naik25a.html) | Provenance-valued reasoning and differentiable execution | Proof-set processing, neural predicates and gradients must be matched; DOLPHIN can retain complex symbolic execution on CPU while batching numerical work |
| [PyJuice](https://arxiv.org/abs/2406.00766), [KLay](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3ba82362eb0aa75487069f19fde794fe-Abstract-Conference.html), [Einsum Networks](https://arxiv.org/abs/2004.06231) | Block/layered probabilistic circuits and compiled sparse arithmetic circuits | Match circuit validity, precision, inference/backward pass, batch and compilation amortization |
| [Torch-Struct](https://aclanthology.org/2020.acl-demos.38/) | Semiring dynamic programs, parallel scans and parsing | HMM/CKY; compare matrix-scan and vector recurrence at equal requested outputs |
| [Ctrl-G](https://arxiv.org/abs/2406.13892), [GeLaTo](https://proceedings.mlr.press/v202/zhang23g.html) | Tractable probabilities of constrained continuations | Full remaining-horizon backward probabilities, not only DFA liveness |
| [XGrammar](https://arxiv.org/abs/2411.15100), [llguidance](https://github.com/guidance-ai/llguidance), [Outlines](https://github.com/dottxt-ai/outlines) | Grammar/token compatibility and per-prefix state | Token masks, tokenization and persistent stack cost; DFA-only kernels do not reproduce general CFG execution |
| [GAPBS](https://arxiv.org/abs/1508.03619), [GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS), [GraphBLAST](https://github.com/gunrock/graphblast), [Gunrock](https://github.com/gunrock/gunrock), [cuGraph](https://github.com/rapidsai/cugraph) | Optimized graph traversal, sparse algebra and intersections | Direction-optimized BFS, delta-stepping/Dijkstra, oriented TC; same graph semantics and output |
| [cuRPQ](https://arxiv.org/abs/2602.20748) | Labelled-graph × automaton reachability with optimized traversal/state | Compare traversal, visited-state maintenance and materialization rather than assuming full product construction |
| [Kissat](https://github.com/arminbiere/kissat), [CaDiCaL](https://github.com/arminbiere/cadical) | Watched-literal propagation within CDCL SAT solving | BCP closure/conflict is a kernel, not complete CDCL; cube-private trail/watch state matters |
| [Fast Downward](https://www.fast-downward.org/latest/documentation/search/Evaluator/) | Relaxed planning heuristics and search | h-add/h-max use counter/agenda algorithms; full planning additionally needs search and state generation |
| [NBFNet](https://arxiv.org/abs/2106.06935) | Learned generalized Bellman–Ford message/aggregate functions | Scalar max-product path scoring is only a proxy; learned aggregators need not satisfy semiring laws |
| [Transformers](https://github.com/huggingface/transformers) | Autoregressive decoding with hypothesis state | Score selection plus model execution, prefix/KV state and reorder cost; local Markov beam is a restricted kernel |
| [torchhd](https://github.com/hyperdimensional-computing/torchhd) | Hypervector encoding, binding, bundling and cleanup | Specify the representation; bipolar elementwise binding is not circular convolution |
| [pgmpy](https://github.com/pgmpy/pgmpy) | Discrete graphical-model inference | Same factors, evidence, elimination order and output marginal |
| [XLOG, recent preprint](https://arxiv.org/abs/2609.27203) | Integrated GPU reasoning including Datalog and weighted model counting | Track the inspected version; no local performance reproduction established |

Lobster supports richer provenance than a scalar top-one score. Proof conjunctions, alternative proofs and conflict checks cannot be replaced by one Viterbi predecessor. Similarly, “k-max” structured inference needs ranked derivation state and combination logic, not just grouped selection of final scalar values. [Juice.jl](https://github.com/Tractables/Juice.jl) is an additional official circuit-software reference; distinguish its logic/probabilistic subpackages and Julia implementation from PyJuice rather than merging their performance claims.

### 3.4 Memory-side and system context

[Moctopus](https://arxiv.org/abs/2403.10051) and [PimBeam](https://venero.github.io/publication/2025-TPDS) study PIM regular-path queries, including partitioning, batching and skew. [CoGraf](https://uu.diva-portal.org/smash/record.jsf?pid=diva2%3A2040671) studies graph processing near memory. [JSPIM](https://arxiv.org/abs/2508.08503) is a relational PIM preprint; the authors now provide an artifact under the [SPARQ name](https://github.com/Tajdari-S/SPARQ-reproduced-results). Keep paper/artifact versions separate when interpreting results. The [author project list for PIM-FW and GenDRAM](https://peter90301.github.io/) distinguishes published/accepted work from submitted work; a listed project is not a calibrated local simulator.

System motivation also comes from [Towards Efficient Neuro-Symbolic AI](https://arxiv.org/html/2409.13153v1) and the [CoCoSys research program](https://cocosys.ece.gatech.edu/research/). These sources motivate whole-pipeline accounting; they do not establish that every sym1 kernel is bandwidth-bound. CPU sampled cycles, logical allocation totals and simulator bubbles cannot replace measured memory traffic.

### 3.5 Artifact and citation policy

The authoritative clone/build/run distinction is [upstream_status.md](upstream_status.md) with [upstream_lock.json](upstream_lock.json). A source checkout does not imply a reproduced paper. FlowLog, cuRPQ and hardware queue references newly discussed here are literature entries unless the inventory records an artifact.

Exact benchmark assignments in the old draft were partly unverified. §6 retains the benchmark families as proposed coverage, not as claims that every cited system used every named suite. Check dataset license, revision, semantics and paper configuration before an experiment. Unverified publication-specific counts or performance constants remain `TBD[src]`.

## 4. Workload inputs, calculations and outputs

This section covers every current softwareworkload/*/workload.py. It explains the input, each calculation step and the returned answer. These are checked local definitions/mappings, not new performance measurements. Paper titles and reading locations accompany each task, and every lowering step names its mechanism source and local function. A paper's arithmetic rules do not imply that it implements this entire API.

For a first reading, use the companion's terminology, native-code reading outline and walked-through example. Tables have descriptive names. Delta means a changed table: either new records or improved total values as declared. Host means code outside the primitives that manages loops, grouping, sorting and readout.

Bottleneck evidence: **M** = a local measurement of the stated kernel; **P** = a related primitive/proxy measurement, not the full workload; **H** = an implementation/theory-based hypothesis requiring profiling. The [baseline report](baseline_report.md) is the numerical source of truth. Paper citations explain mechanisms and do not replace local measurements.

### 4.1 `datalog_tc` — transitive closure

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#datalog_tc).

**Papers and reading locations:** [Sun et al., Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf) — §1 reachability rules; §2 semi-naive evaluation and Figure 1.

**Input, calculation and output.** Input: directed graph edges. Output: every start/end pair connected by a nonempty path. A vertex reaches itself only if a cycle provides such a path. [Local native](../softwareworkload/datalog_tc/workload.py#L43).

**How the operators form the computation.** 

1. Initialize both the accumulated path table and the new-path table with the original edges. Mechanism sources: [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf), [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9). Interface mapping / exact execution: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

2. combine: match each new path's end with an original edge's start; retain the original path start and the edge's end. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1). Interface mapping / exact execution: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

3. mask with negate=True: remove start/end pairs already in the accumulated table. Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](workload_operator_evidence.md#d8), [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9). Interface mapping / exact execution: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

4. update: save genuinely new pairs and use them as the next new-path table. Mechanism sources: [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9). Interface mapping / exact execution: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

5. Stop when that new-path table is empty. Return the accumulated paths. Mechanism sources: [D7 · Convergence of Datalog over (Pre-) Semirings](workload_operator_evidence.md#d7). Interface mapping / exact execution: [Local algebra](../softwareworkload/datalog_tc/workload.py#L71).

**Terms, state and implementation limits.** A recursive rule reuses its own earlier answers. Closure means that applying the rule again produces no new reachable pairs. This is semi-naive evaluation: reuse previous answers and expand only newly discovered paths. Rounds depend on previous rounds; the independent native code checks the answer by graph traversal. [Local algebra](../softwareworkload/datalog_tc/workload.py#L71); [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf).

**Bottleneck/fit [P,H].** Measure examined joins, duplicates, index accesses, frontier sizes and rounds. Fused membership checks and resident state may avoid materialization; they do not remove duplicate-producing graph structure. Local Souffle runs measure source-batched reachability, including source seeds, not this complete all-pairs nonempty-path closure.

Attribution sources: the local code above and [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.2 `datalog_samegen` — recursive same-generation relation

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#datalog_samegen).

**Papers and reading locations:** [Sun et al., Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf) — §2 same-generation rules and Figure 1.

**Input, calculation and output.** Input: seed same-generation pairs and links between parents and children. Output: all pairs obtained by repeatedly pairing children whose parents are already known to be of the same generation. [Local _golden_seminaive](../softwareworkload/datalog_samegen/workload.py#L114).

**How the operators form the computation.** 

1. Initialize saved and newly discovered same-generation pairs from the seed table. Mechanism sources: [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

2. combine: pair each newly related parent pair with the children of its first parent. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

3. combine: match that intermediate table with the children of its second parent; retain the two children. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

4. Remove child pairs already saved, then update the saved and new-pair tables. Mechanism sources: [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

5. Repeat until no new pair is produced. Alternatively, join the second parent's children first; compare intermediate sizes. Mechanism sources: [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3), [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147).

**Terms, state and implementation limits.** A seed is a fact supplied before recursion. Same-generation here names the relation defined by these rules, rather than an independently inferred real-world family tree. Changing the join order preserves the Boolean answer, not necessarily the amount of work. Both naive and semi-naive native paths check the generated cases. The paper derives sibling seeds from shared parents; the local workload supplies Flat directly. They describe the same instance only when seeds agree. [Local _algebra](../softwareworkload/datalog_samegen/workload.py#L147); [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf).

**Bottleneck/fit [H].** Fan-out and join ordering can create large intermediates before deduplication. Probe reuse and fused projection may help, but hardware sharing cannot rescue an inappropriate join order. Profile both orders with identical final SG and separate intermediate counts.

Attribution sources: the local code above and [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.3 `datalog_nonlinear_tc` — nonlinear recursive closure

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#datalog_nonlinear_tc).

**Papers and reading locations:** [Sun et al., Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf) — §2 semi-naive evaluation; pinned GDlog tc.dl supplies the nonlinear rule.

**Input, calculation and output.** Input and output are the same reachability question as linear transitive closure. The computation instead joins an already known path to another known path, so both sides can change between rounds. [Local native](../softwareworkload/datalog_nonlinear_tc/workload.py#L68).

**How the operators form the computation.** 

1. Start the accumulated and new-path tables from the edges; start the old snapshot empty. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

2. First combine: new paths on the left with current accumulated paths on the right. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

3. Second combine: the old accumulated snapshot on the left with new paths on the right. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

4. Merge those candidate tables using existence rather than counting; remove pairs already known. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

5. Save the current accumulated table as the old snapshot before updating it. Save genuinely new pairs for the next round. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D9 · Soufflé: On Synthesis of Program Analyzers](workload_operator_evidence.md#d9). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

6. Repeat until there are no new pairs; return all accumulated paths. Mechanism sources: [D7 · Convergence of Datalog over (Pre-) Semirings](workload_operator_evidence.md#d7). Interface mapping / exact execution: [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92).

**Terms, state and implementation limits.** Nonlinear here means that the same changing path relation occurs twice in a rule; it does not mean a neural-network nonlinearity. The first join already includes new paths on both sides. The general two-occurrence recursive rule needs both delta placements, even if a particular graph happens to yield the same pair from each. Boolean deduplication is not ordinary arithmetic counting. [Local _algebra](../softwareworkload/datalog_nonlinear_tc/workload.py#L92); [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf).

**Bottleneck/fit [H].** Growing recursive relations, duplicate paths and update pressure can dominate. Persistent indexes and fused update are plausible targets. Fewer rounds need not mean less total work; compare a good linear closure algorithm as well as the matched nonlinear schedule.

Attribution sources: the local code above and [Optimizing Datalog for the GPU (GPUlog)](https://sidharthkumar.io/publications/asplos2025.pdf) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.4 `triangle_count` — oriented triangle intersections

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#triangle_count).

**Papers and reading locations:** [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619) — benchmark-kernel definitions: triangle counting.

**Input, calculation and output.** Input: an undirected graph without self-loops or duplicate edges. Output: the number of triangles. Order vertices consistently, then orient edges by that order so each triangle is counted once. [Local native](../softwareworkload/triangle_count/workload.py#L62).

**How the operators form the computation.** 

1. combine: join two directed edges sharing a middle vertex; this creates a two-edge path, called a wedge. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3). Interface mapping / exact execution: [Local algebra](../softwareworkload/triangle_count/workload.py#L95).

2. mask: retain a wedge only if an edge also joins its start directly to its end. Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](workload_operator_evidence.md#d8). Interface mapping / exact execution: [Local algebra](../softwareworkload/triangle_count/workload.py#L95).

3. reduce: add the surviving counts to obtain the triangle total. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1). Interface mapping / exact execution: [Local algebra](../softwareworkload/triangle_count/workload.py#L95).

4. A fused alternative first counts wedges by their start/end pair, matches the closing edge and sums. It must retain counts, not replace them with existence flags. Mechanism sources: [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3). Interface mapping / exact execution: [Local algebra](../softwareworkload/triangle_count/workload.py#L95).

**Terms, state and implementation limits.** A neighbor is a vertex joined by an edge. Intersection keeps vertices appearing in both neighbor lists. Orientation chooses a consistent direction only for counting; it does not change the original undirected task. The independent native algorithm intersects oriented neighbor lists and can avoid materializing wedges. A scalar result is represented by a row with no identifying fields. [Local algebra](../softwareworkload/triangle_count/workload.py#L95); [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619).

**Bottleneck/fit [M,P].** GAPBS provides CPU kernel measurements; SAM measures isolated forward-neighbor intersections. Coordinate scans, list imbalance and skip efficiency matter. Intersection/segmented reduction is a good operator fit, but SAM's existing skip capability is prior art. Local CPU scaling contradicts a blanket “CPU has no parallel efficiency” explanation.

Attribution sources: the local code above and [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.5 `sat_bcp` — Boolean constraint propagation

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#sat_bcp).

**Papers and reading locations:** [Moskewicz et al., Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf) — BCP section: two watched literals.

**Input, calculation and output.** Input: clauses joined by AND, with alternatives inside each clause joined by OR, plus a partial truth assignment for each query. Output: forced assignments and whether a contradiction is found. If a clause is not already true and all but one of its literals are false, its remaining literal must be true. [Local _propagate_2wl](../softwareworkload/sat_bcp/workload.py#L86).

**How the operators form the computation.** 

1. Initialize true and false literal tables from the input assignment. A literal is a variable or its negation. Keep the initial two watched literals per query/clause and start with no conflicts. Mechanism sources: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

2. scan flips newly true literals to identify their newly false opposites. combine with the watch table and reduce by query/clause to locate touched clauses; also count how many watches fell false. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

3. Read the literals of touched clauses. Exclude satisfied clauses, identify false literals with mask, and count them per clause. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

4. filter identifies unit clauses and all-false clauses. A unit clause has exactly one remaining non-false literal; project that literal as a forced assignment. Mechanism sources: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

5. For a touched clause needing replacement watches, exclude existing watches and select the smallest available literal identifier. If both watches fell false, choose another distinct replacement. Mechanism sources: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

6. Record conflicts from an all-false clause or contradictory new implications. Freeze conflicted queries; update other assignments and feed only new assignments into the next round. Mechanism sources: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

7. Stop when no new assignment remains. Return assignments and per-query conflict flags. Mechanism sources: [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf). Interface mapping / exact execution: [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265).

**Terms, state and implementation limits.** A literal is a variable or its negation. A unit clause has only one not-yet-false literal, which must become true; a conflict means no remaining assignment can satisfy a clause under the current decisions. A query is one independently propagated partial assignment. A watch is a watched clause literal used to avoid inspecting unaffected clauses. Static clauses can be shared; assignments, propagation history and watches are query-private. The local monotone program retains obsolete false watches; it is not a backtracking deletion protocol. Generated clauses have at least two literals and consistent initial assignments. General empty/unit input clauses need explicit initialization. [Local algebra_watched](../softwareworkload/sat_bcp/workload.py#L265); [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf).

**Bottleneck/fit [H].** Irregular clause reads, long propagation chains and state updates are candidates. Compare an efficient two-watched-literal golden with early conflict exit. For a nonconflicting cube compare closure; for a conflicting cube compare the conflict verdict without requiring identical partially completed trails. BCP alone is not full CDCL learning/backjumping.

Attribution sources: the local code above and [Chaff: Engineering an Efficient SAT Solver](https://www.princeton.edu/~chaff/publication/DAC2001v56.pdf) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.6 `kg_multihop` — bounded max-product path scoring

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#kg_multihop).

**Papers and reading locations:** [Zhu et al., Neural Bellman-Ford Networks](https://arxiv.org/abs/2106.06935) — §3.1 path formulation; Appendix A.5 most reliable path; §3.2 learned operators.

**Input, calculation and output.** Input: weighted graph edges with relation labels, query sources, a maximum path length and a requested answer count. Output: the best allowed path score for each destination, then the requested best destinations. Multiply edge probabilities along a path and choose the largest product among paths reaching the same destination. [Local native](../softwareworkload/kg_multihop/workload.py#L67).

**How the operators form the computation.** 

1. Store scores as logarithms of probabilities. Extending a path then adds its edge log-score instead of multiplying probabilities. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6), [Neural Bellman-Ford Networks](https://arxiv.org/abs/2106.06935). Interface mapping / exact execution: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

2. Seed the source with the score of an empty path. Each query has its own saved scores. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

3. combine: extend newly improved paths with matching outgoing edges, group by query and destination, and keep the largest log-score. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6), [D8 · GraphBLAS C API Specification (standard)](workload_operator_evidence.md#d8). Interface mapping / exact execution: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

4. update saved scores and feed actual improvements into the next round. Stop at the hop limit or when there are no improvements. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

5. Exclude the source as required by the local answer definition. Group candidates by query and use select_k for the requested answer count. Interface mapping / exact execution: [Local algebra](../softwareworkload/kg_multihop/workload.py#L104).

**Terms, state and implementation limits.** A knowledge-graph edge labels the kind of relationship as well as its endpoints. A hop is one edge traversal. Log-scores preserve probability ordering: a larger log-score still means a more probable path. The maximum hop count and requested output count are different settings. Floating-point improvement checks must match the independent golden. This is scalar best-path scoring, not the full learned NBFNet computation. [Local algebra](../softwareworkload/kg_multihop/workload.py#L104); [Neural Bellman-Ford Networks](https://arxiv.org/abs/2106.06935).

**Bottleneck/fit [H].** Sparse expansion, uneven degree and repeated state access are candidates; batching may reuse edges. The scalar recurrence is not complete NBFNet: learned message/aggregate functions and training require additional semantics. Profile query-count-dependent reuse and state footprint rather than assuming batching always helps.

Attribution sources: the local code above and [Neural Bellman-Ford Networks](https://arxiv.org/abs/2106.06935) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.7 `bayes_net_ve` — posterior by variable elimination

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#bayes_net_ve).

**Papers and reading locations:** [Dechter, Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594) — belief assessment and bucket-elimination algorithm.

**Input, calculation and output.** Input: probability tables for a discrete Bayesian network, observed variable values and the variable to ask about. Output: the probability of each query value after accounting for the observations. For each possible assignment, multiply its compatible factor entries; add over hidden assignments, then normalize. [Local native](../softwareworkload/bayes_net_ve/workload.py#L233).

**How the operators form the computation.** 

1. filter each factor table to entries compatible with observed evidence. A factor is a table that assigns a weight to a combination of variable values. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

2. Choose a hidden variable to eliminate. Collect the current factors containing that variable; this collection is its bucket. Mechanism sources: [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594), [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

3. combine those factors on shared variable values, multiplying their weights. reduce by the other variables to sum over the eliminated one. Mechanism sources: [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2), [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

4. Replace that bucket with the resulting smaller factor. Repeat in the declared elimination order. Mechanism sources: [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594), [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

5. Combine remaining factors and keep the query variable. Sum its unnormalized weights, then divide each weight by that sum. Mechanism sources: [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

6. If the sum is zero, the evidence has zero probability and a conditional answer is undefined. The log-space version performs the equivalent operations using log-sum-exp. Mechanism sources: [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2), [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594). Interface mapping / exact execution: [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269).

**Terms, state and implementation limits.** Evidence means an observed variable value; prior means probability before considering that observation; posterior means probability after conditioning on it. Normalizing divides weights by their total so that they form a probability distribution. A bucket with only one factor still needs reduction. Choosing the order and managing factor lists are host work, not scan callbacks. Probability-space and log-space evaluations are separate. The independent golden uses factor dictionaries and small-instance enumeration. [Local _ve_algebra](../softwareworkload/bayes_net_ve/workload.py#L269); [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594).

**Bottleneck/fit [H].** Induced factor size, not simply sparse arithmetic rate, can dominate. Compare the same elimination order first, then report order optimization separately. The local generator bounds factor growth; it does not represent unrestricted high-treewidth inference. The golden uses independent factor dictionaries, with small-instance enumeration checks.

Attribution sources: the local code above and [Bucket Elimination: A Unifying Framework for Reasoning](https://www.sciencedirect.com/science/article/pii/S0004370299000594) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.8 `prob_circuit` — layered arithmetic-circuit evaluation

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#prob_circuit).

**Papers and reading locations:** [Liu et al., Scaling Tractable Probabilistic Circuits: A Systems Perspective (PyJuice)](https://arxiv.org/abs/2406.00766) — probabilistic-circuit definitions and forward inference.

**Input, calculation and output.** Input: a directed computation graph of sum/product nodes, leaf values and weights. Output: the root value for each input item. A sum node adds weighted child values; a product node multiplies all required child values. [Local _native_eval](../softwareworkload/prob_circuit/workload.py#L114).

**How the operators form the computation.** 

1. Process layers only after the child values they need are available; independent nodes and input items can be processed together. Mechanism sources: [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766). Interface mapping / exact execution: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

2. For a sum layer, combine the child-value table with the edge-weight table and group by sample and parent node. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2). Interface mapping / exact execution: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

3. For a product layer, separately gather each mandatory child slot, then combine the gathered tables on sample and parent. A missing zero child must make the entire product zero. Mechanism sources: [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2), [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766). Interface mapping / exact execution: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

4. rename the output node field for the next layer, then continue through the finite layer list. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1). Interface mapping / exact execution: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

5. Read the designated root for each input item. Use ordinary probability, log probability or integer counting consistently within an execution. Mechanism sources: [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766). Interface mapping / exact execution: [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143).

**Terms, state and implementation limits.** A node represents a calculation. Leaves supply input values; internal nodes consume child results. A topological order means that every child is computed before the parent that needs it. Log-space products become addition; log-space sums use log-sum-exp. The counting run uses true/false leaf indicators and unit sum weights to count arithmetic derivations. A generic multiply-fold over only stored children is unsafe when required zeros were suppressed. The random DAG is not automatically a certified probabilistic model. [Local _eval_circuit](../softwareworkload/prob_circuit/workload.py#L143); [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766).

**Bottleneck/fit [H].** Layer width, child gathers, batch shape and repeated intermediate traffic are candidates; persistent fused layers may help. The generated random arithmetic DAG is not certified smooth, decomposable or deterministic. Consequently, its root is not automatically a valid probability or weighted model count. Replacing sum by max is not a general MAP solver. Compare PyJuice/KLay only on matched valid circuits and inference semantics.

Attribution sources: the local code above and [Scaling Tractable Probabilistic Circuits: A Systems Perspective](https://arxiv.org/abs/2406.00766) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.9 `automata_rpq` — graph × NFA reachability

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#automata_rpq).

**Papers and reading locations:** [cuRPQ: A High-Performance GPU-Based Framework for Processing Regular and Conjunctive Regular Path Queries](https://arxiv.org/abs/2602.20748) — regular-path-query definition and product-graph construction/traversal.

**Input, calculation and output.** Input: a graph with edge labels, an automaton describing allowed label sequences and query sources. Output: destinations reachable by a path whose label sequence the automaton accepts. Track both the graph position and the automaton state while moving. [Local native](../softwareworkload/automata_rpq/workload.py#L146).

**How the operators form the computation.** 

1. Initialize each query at its source vertex and the automaton's start state. Mechanism sources: [cuRPQ](https://arxiv.org/abs/2602.20748). Interface mapping / exact execution: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

2. combine graph edges with automaton transitions that have the same label. This may be done on demand or precomputed into a product-edge table. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [cuRPQ](https://arxiv.org/abs/2602.20748). Interface mapping / exact execution: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

3. Extend each new graph/state pair through a compatible product edge. Keep query identity, new vertex and new automaton state. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

4. mask out already visited pairs and update the saved and new-pair tables. Repeat until no new pair remains. Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](workload_operator_evidence.md#d8). Interface mapping / exact execution: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

5. mask by accepting automaton states and reduce away the automaton state; return query/destination pairs. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [D8 · GraphBLAS C API Specification (standard)](workload_operator_evidence.md#d8). Interface mapping / exact execution: [Local algebra](../softwareworkload/automata_rpq/workload.py#L188).

**Terms, state and implementation limits.** An automaton is a finite-state checker for a sequence of labels. An accepting state means the labels read so far form an allowed sequence. A nondeterministic automaton may retain several possible states, so the state must remain part of each visited key. Graph-first, automaton-first and precomputed-product joins are alternative schedules, not stages to run consecutively. A precomputed product table is built outside the loop. The local automaton has no transitions that consume no label; such transitions would need additional handling. [Local algebra](../softwareworkload/automata_rpq/workload.py#L188); [cuRPQ](https://arxiv.org/abs/2602.20748).

**Bottleneck/fit [H].** Label selectivity, automaton nondeterminism, product-state growth and visited-set traffic govern work. Shared graph reads and fused membership updates may help; a materialized product can instead enlarge memory demand. Epsilon transitions would require an additional closure treatment.

Attribution sources: the local code above and [cuRPQ](https://arxiv.org/abs/2602.20748) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.10 `beam_search` — top-K prefix expansion

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#beam_search).

**Papers and reading locations:** [Freitag & Al-Onaizan, Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/) — §3 Original Beam Search; §4 is a set of additional strategies.

**Input, calculation and output.** Input: token-transition scores, an initial token, the number of hypotheses to retain and a length limit. Output: the surviving token sequences and scores. Extend every surviving sequence, score each extension, and retain only the requested number of best extensions. [Local native](../softwareworkload/beam_search/workload.py#L54).

**How the operators form the computation.** 

1. Keep a distinct identifier for every current sequence, even if several end with the same token. Mechanism sources: [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/), [D11 · Better k-best Parsing](workload_operator_evidence.md#d11). Interface mapping / exact execution: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

2. combine the sequence table with allowed transitions on the last token. Add the transition log-score; retain both the new token and the parent-sequence identifier. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/). Interface mapping / exact execution: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

3. select_k retains the requested number of best candidates globally for this beam. Mechanism sources: [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/). Interface mapping / exact execution: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

4. Host code ranks survivors by decreasing score, then increasing token and parent identifier; save parent links and scan to assign the new ranks. Interface mapping / exact execution: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

5. Repeat to the length limit, then follow saved parent links backward to reconstruct each sequence. Mechanism sources: [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/). Interface mapping / exact execution: [Local algebra](../softwareworkload/beam_search/workload.py#L69).

**Terms, state and implementation limits.** A prefix is the sequence generated so far. Pruning drops candidates now, without a guarantee that all globally best complete sequences remain. Parent identifiers say which earlier prefix produced a new candidate. Beam means the currently retained set of candidate sequences. Top-k means the best k items, where k is the requested count. Rank calculation and parent storage are explicit control work. This local Markov toy conditions on the last token; a language model also needs the full prefix and model state. [Local algebra](../softwareworkload/beam_search/workload.py#L69); [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/).

**Bottleneck/fit [H].** Candidate generation, ordered selection and parent/state movement are candidates; a persistent selection structure may help. The time steps remain sequential. This is a sparse Markov toy, whereas language-model beam search conditions on the entire prefix and additionally moves/reorders model state and KV caches.

Attribution sources: the local code above and [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.11 `fuzzy_inference` — max-min closure and query composition

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#fuzzy_inference).

**Papers and reading locations:** [Zadeh, Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051) — max-min composition and transitivity definitions.

**Input, calculation and output.** Input: relation strengths between items and query strengths. Output: the strongest indirect relation for each pair, then each query's strength to each answer. A path is as strong as its weakest edge; among alternative paths choose the strongest. [Local native](../softwareworkload/fuzzy_inference/workload.py#L68).

**How the operators form the computation.** 

1. Seed saved and newly improved relations with the input strengths. Mechanism sources: [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051). Interface mapping / exact execution: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

2. combine newly improved paths with another relation edge: take the smaller strength for each extension, then the largest strength among alternatives to the same pair. Mechanism sources: [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051), [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

3. update saved strengths and propagate genuinely improved values, including improvements to pairs that already existed. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

4. Repeat until no strength improves. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6). Interface mapping / exact execution: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

5. combine query strengths with the completed relation using the same weaker-link/strongest-alternative rules. Mechanism sources: [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051). Interface mapping / exact execution: [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115).

**Terms, state and implementation limits.** Relation strength lies between zero and one, with a larger value meaning a stronger relation. It is not a probability of mutually exclusive events; the weakest-link/strongest-path rule is the requested fuzzy relation operation. Do not remove candidates merely because the key is already present. Generated cases use a finite set of possible strengths, which bounds the number of improvements. The independent golden uses ordered widest-path propagation. [Local algebra](../softwareworkload/fuzzy_inference/workload.py#L115); [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051).

**Bottleneck/fit [H].** Repeated improvements and unordered rounds may do more work than an ordered schedule. Min/max units and fused state updates fit naturally, but the priority policy and work count must be matched before attributing gains to hardware.

Attribution sources: the local code above and [Similarity Relations and Fuzzy Orderings](https://www.sciencedirect.com/science/article/pii/S0020025571800051) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.12 `hmm_viterbi` — forward likelihood and best state sequence

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#hmm_viterbi).

**Papers and reading locations:** [Rush, Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/) — semiring dynamic programming and parallel linear-chain implementation.

**Input, calculation and output.** Input: initial hidden-state probabilities, probabilities of moving between hidden states, probabilities of observations in each state, and an observation sequence. Forward inference returns the total probability of the observations. Viterbi returns the single most likely hidden-state path and its score. [Local _native_one](../softwareworkload/hmm_viterbi/workload.py#L71).

**How the operators form the computation.** 

1. Initialize each state from its initial probability and the probability of the first observed item in that state. Mechanism sources: [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2), [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/). Interface mapping / exact execution: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

2. combine previous-state scores with the transition table, matching the previous state and retaining the next state. Mechanism sources: [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2). Interface mapping / exact execution: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

3. For forward inference, add contributions arriving at the same next state. For Viterbi, keep only the best contribution and its previous-state identifier. Mechanism sources: [D4 · Semiring Parsing](workload_operator_evidence.md#d4). Interface mapping / exact execution: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

4. Apply the next observation's probability, rename next state to current state, and repeat along the observation sequence. Mechanism sources: [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2). Interface mapping / exact execution: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

5. Forward readout combines final-state probabilities. Viterbi selects the best final state for each query and follows saved predecessor identifiers backward. Mechanism sources: [D4 · Semiring Parsing](workload_operator_evidence.md#d4). Interface mapping / exact execution: [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110).

**Terms, state and implementation limits.** A hidden state is the model's unobserved condition; an observation is the data actually seen. Transition probabilities connect consecutive hidden states, while emission probabilities connect a state to a possible observation. The implementation stores log probabilities. LSE, short for log-sum-exp, adds the represented probabilities and returns the logarithm of their sum, without explicitly converting very small probabilities. Tagged Viterbi values contain a score and a predecessor. Combining them adds scores and uses the right operand's valid predecessor, otherwise the left one's; alternatives choose higher score then smaller predecessor identifier. Emission tags mean “no new predecessor.” Preserve operand order, save every layer, and use score then state identifier for final ties. Readout/traceback is explicit host work, with per-layer filtering in the current code. These tagged values and tie rules are a specific local definition; arbitrary tagged values are not claimed to satisfy semiring laws or allow operand swapping. [Local algebra](../softwareworkload/hmm_viterbi/workload.py#L110); [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/).

**Bottleneck/fit [P,H].** Torch-Struct chain experiments show that scheduling and temporary tensors matter. Dense vector recurrence enumerates previous/next state pairs; naive matrix composition additionally enumerates an intermediate state. Work is therefore **derived** to grow with sequence length times state count twice for the former, and potentially sequence length times state count three times for the latter. Matrix composition may shorten parallel dependency depth at the cost of more arithmetic; these are not measured rates. LSE needs numerical support. The upstream score experiments are not full traceback benchmarks, and the local backtrace scans whole layers.

Attribution sources: the local code above and [Torch-Struct: Deep Structured Prediction Library](https://aclanthology.org/2020.acl-demos.38/) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.13 `ctrlg_dfa_hmm` — prefix filtering versus constrained completion

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#ctrlg_dfa_hmm).

**Papers and reading locations:** [Zhang et al., Ctrl-G: Adaptable Logical Control for Large Language Models](https://arxiv.org/abs/2406.13892) — §2 HMM and guidance formulation; §§3.2–3.3 probability versus logical feasibility.

**Input, calculation and output.** Input: an HMM probability model, a deterministic automaton describing allowed output sequences and an observed prefix. Output in the local proxy: next-token scores restricted to tokens that can still lead to acceptance. Full constrained generation additionally weights how probable a valid completion is within the remaining length. [Local native](../softwareworkload/ctrlg_dfa_hmm/workload.py#L132).

**How the operators form the computation.** 

1. First reduce away token labels to obtain the edges needed for backward traversal. Starting from accepting states, use combine, rename, mask and update to mark every state that can still reach acceptance. Mechanism source: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6); exact execution: [Local algebra](../softwareworkload/ctrlg_dfa_hmm/workload.py#L231).

2. mask retains automaton transitions whose successors appear in that live-state table. reduce drops the successor field, leaving an allowed-next-step table keyed by current state and token. Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](workload_operator_evidence.md#d8), [D1 · Provenance Semirings](workload_operator_evidence.md#d1); exact execution: [Local algebra](../softwareworkload/ctrlg_dfa_hmm/workload.py#L231).

3. Process the observed prefix: combine hidden-state probabilities with state transitions, apply the observed token likelihood and match its automaton transition. Initialize the first observation consistently with the model. Mechanism sources: [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892), [D2 · The Generalized Distributive Law](workload_operator_evidence.md#d2). Interface mapping / exact execution: [Local _forward_and_scores](../softwareworkload/ctrlg_dfa_hmm/workload.py#L184).

4. For each proposed next token, combine current probabilities with hidden-state transitions and token likelihoods; mask out automaton successors that cannot reach acceptance. This is the local proxy. Interface mapping / exact execution: [Local _forward_and_scores](../softwareworkload/ctrlg_dfa_hmm/workload.py#L184).

5. For full fixed-length conditioning, build a backward table of valid-completion probability. With no emissions remaining, only accepting states succeed. For each larger remaining length, enumerate a hidden-state transition and emitted token, multiply their probabilities by the previous completion-table entry, and add alternatives. Paper basis: [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892) (§§3.2–3.3); this explains the missing computation and is not implemented in the local function above.

6. Weight next-token candidates by that remaining-completion probability, then divide each candidate's weight by the total. This additional backward calculation is not in the local proxy. Paper basis: [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892) (§§3.2–3.3); this explains the missing computation and is not implemented in the local function above.

**Terms, state and implementation limits.** The automaton checks sequence legality; the HMM assigns sequence probabilities. A deterministic automaton has at most one successor for a given state and token. Acceptance is a Boolean condition, not a probability value. HMM is a hidden Markov model; DFA is a deterministic finite automaton. Liveness means only that some accepted continuation exists, not its probability. For a known prefix, a deterministic automaton has one active state; the entire state-product table need not be occupied. Use compatible initialization for transition-then-emission calculations. Invalid transitions contribute nothing, normalization requires a positive total, and variable lengths/end-of-sequence need an explicit termination rule. Full Ctrl-G uses HMM-estimated constraint-satisfaction probabilities to reweight the language model's next-token distribution. The teaching case checks the constrained HMM distribution alone, not the full language-model generation pipeline. The default sparse_first schedule filters through the sparse allowed-next-step table before gathering token probabilities; dense_first computes all vocabulary contributions before filtering and is a comparison schedule. [Local _forward_and_scores](../softwareworkload/ctrlg_dfa_hmm/workload.py#L184); [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892).

**Bottleneck/fit [H].** Product transitions, vocabulary contraction, backward tables and repeated decoding steps are candidates. The local Boolean liveness plus forward `combine` kernel does not reproduce full Ctrl-G/GeLaTo constrained continuation probabilities. Measure the complete backward/online boundary when making that application claim.

Attribution sources: the local code above and [Adaptable Logical Control for Large Language Models (Ctrl-G)](https://arxiv.org/abs/2406.13892) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.14 `vsa_cleanup` — bind, bundle, unbind and nearest-codeword search

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#vsa_cleanup).

**Papers and reading locations:** [Heddes et al., Torchhd](https://www.jmlr.org/papers/v24/23-0300.html); [CogSys](https://zishenwan.github.io/publication/HPCA25_CogSys.pdf) — Torchhd: representations and bind/bundle/similarity operations; CogSys: VSA workload characterization.

**Input, calculation and output.** Input: vectors representing attribute values, attribute keys, slot keys, object descriptions and queries. Output: the codebook entry most similar to each recovered query vector. Bind by multiplying corresponding coordinates, bundle by adding vectors, unbind using the queried keys, and compare against the codebook. [Local native](../softwareworkload/vsa_cleanup/workload.py#L65).

**How the operators form the computation.** 

1. combine each object's attribute choices with the codebook and the corresponding attribute key; add contributions by object and vector coordinate. Mechanism sources: [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html), [D1 · Provenance Semirings](workload_operator_evidence.md#d1). Interface mapping / exact execution: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

2. combine object vectors with scene slot assignments and slot keys; add contributions by scene and coordinate. Mechanism sources: [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html), [D1 · Provenance Semirings](workload_operator_evidence.md#d1). Interface mapping / exact execution: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

3. For each query, gather its scene vector and multiply corresponding coordinates by its requested slot and attribute keys. Mechanism sources: [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html). Interface mapping / exact execution: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

4. combine the recovered vector with codebook candidates on attribute and coordinate; multiply matching coordinates and reduce by query and candidate to form dot-product similarities. Mechanism sources: [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3), [D10 · The Sparse Abstract Machine](workload_operator_evidence.md#d10). Interface mapping / exact execution: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

5. Filter rows for each query and select its best candidate, with deterministic ties. Interface mapping / exact execution: [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92).

**Terms, state and implementation limits.** A codebook is a dictionary storing a vector for each candidate value. A key vector identifies which attribute or position a value occupies; it is different from a relational table's identifying key. A vector is a list of numbers; a coordinate is one position in that list; a dot product multiplies corresponding positions and adds them. The generated codebook is Gaussian-valued; only attribute/slot keys are bipolar, containing positive or negative one, so multiplying a key twice cancels its effect. Input-presence tables use neutral weights. Encoding and cleanup are both included. This is elementwise binding, not circular convolution. [Local algebra](../softwareworkload/vsa_cleanup/workload.py#L92); [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html).

**Bottleneck/fit [H].** Codebook traffic, batch shape and reduction/selection are candidates; grouping queries by attribute may improve reuse. This binding is not the circular convolution evaluated by CogSys. Superposition introduces crosstalk: agreement with the independent numerical golden and recovery of the originally encoded attribute are separate correctness/accuracy measures.

Attribution sources: the local code above and [Torchhd](https://www.jmlr.org/papers/v24/23-0300.html) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.15 `astar_grid` — heuristic ordered shortest-path search

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#astar_grid).

**Papers and reading locations:** [Hart, Nilsson & Raphael, A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf) — A* search procedure and heuristic conditions.

**Input, calculation and output.** Input: a grid with obstacles, a start and a goal, and equal-cost orthogonal moves. Output: a shortest path cost. Among pending vertices choose the smallest “cost already spent plus estimated cost still needed,” then relax its neighbors. [Local native](../softwareworkload/astar_grid/workload.py#L85).

**How the operators form the computation.** 

1. Initialize the pending table with the start and its estimated total cost. The settled table is initially empty. Mechanism sources: [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

2. select_k picks the best pending vertex; a full-key mask removes it. If it is the goal, return its path cost; an empty pending table without reaching the goal means failure. Mechanism sources: [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf), [D12 · The Generalized A* Architecture](workload_operator_evidence.md#d12). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

3. Recover the actual cost already spent by subtracting the heuristic estimate. Save that cost in the settled table. Mechanism sources: [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

4. Fetch the selected vertex's adjacency row and combine it with the actual cost; add each edge cost to form neighbor candidates. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

5. Exclude settled neighbors. Add each remaining neighbor's heuristic estimate to obtain its estimated total cost. Mechanism sources: [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

6. update pending scores with improvements and repeat in the declared priority order. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6), [D12 · The Generalized A* Architecture](workload_operator_evidence.md#d12). Interface mapping / exact execution: [Local _astar](../softwareworkload/astar_grid/workload.py#L138).

**Terms, state and implementation limits.** The open table contains candidates waiting to be expanded; the closed/settled table contains vertices already finalized under the heuristic assumptions. The heuristic is an estimate, not distance that has already been traveled. The heuristic is Manhattan distance: the horizontal and vertical separation from the goal. On this grid it is consistent, meaning the estimate at a vertex never exceeds an outgoing move's cost plus the next vertex's estimate; this supports the local no-reopen rule. In the code, a scan constructs the correction value and a min-plus combine adds it. Row fetching, stop checks and stale-entry semantics remain explicit. [Local _astar](../softwareworkload/astar_grid/workload.py#L138); [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf).

**Bottleneck/fit [H].** Repeatedly scanning a relation to find/delete the minimum adds work absent from a persistent queue. Queue hardware may remove this overhead, but it does not remove the pop→relax→push dependency. Global relaxation is a different algorithm and can lose heuristic pruning. Independent queries trade more state for available concurrency.

Attribution sources: the local code above and [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.16 `planning_hadd` — delete-relaxed proposition costs

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#planning_hadd).

**Papers and reading locations:** [Bonet & Geffner, Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf) — §4 additive and max heuristics; weighted action costs also checked against Fast Downward.

**Input, calculation and output.** Input: initially true facts and actions with preconditions, effects and costs. Output: estimated cost to reach each fact after ignoring delete effects. An action is usable only when all its prerequisites are reachable; add its own cost to the combined prerequisite cost, then choose the cheapest action reaching each effect. [Local _native_heuristic](../softwareworkload/planning_hadd/workload.py#L107).

**How the operators form the computation.** 

1. Initialize true facts with no added cost; other facts are not yet reachable. Mechanism sources: [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

2. combine changed facts with the action-precondition table to find touched actions; then read all prerequisites of each touched action. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5), [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

3. Join current fact costs to those prerequisite rows. Separately identify any action with a missing prerequisite and prevent it from firing. Mechanism sources: [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf), [D3 · FAQ: Questions Asked Frequently](workload_operator_evidence.md#d3). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

4. For h-add, add prerequisite costs. For h-max, take the largest prerequisite cost. These are alternative evaluations. Mechanism sources: [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

5. Add each action's own cost, group its effects by resulting fact, and retain the lowest candidate cost. Mechanism sources: [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

6. update saved fact costs and repeat only from real improvements. Return proposition costs; a full state heuristic would additionally combine goal costs. Mechanism sources: [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5). Interface mapping / exact execution: [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172).

**Terms, state and implementation limits.** Delete relaxation ignores an action's effects that would make facts false. It makes reachability easier to estimate. h-add can charge shared supporting work more than once, so its value is not generally a guaranteed lower bound on a real plan's cost. h-max means maximum across prerequisites followed by addition of action cost, not taking the maximum everywhere. Helper rows indicating a prerequisite exists use a neutral additive cost, so numeric zero is a valid stored row. Generated actions have nonempty prerequisite lists. A general action with no prerequisites needs separate initialization with no prerequisite cost. The current result is not a complete plan. [Local _algebra_heuristic](../softwareworkload/planning_hadd/workload.py#L172); [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf).

**Bottleneck/fit [H].** Re-reading all prerequisites of touched actions may be more expensive than the golden's generalized Dijkstra with unsatisfied-precondition counters. Incremental counters and persistent agenda state are plausible targets; compare equal propagation work or report the algorithm change separately. Full planning also includes search and repeated heuristic calls.

Attribution sources: the local code above and [Planning as Heuristic Search](https://www.cs.toronto.edu/~sheila/2542/s14/A1/bonetgeffner-heusearch-aij01.pdf) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.17 `cky_parsing` — inside probability and best parse score

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#cky_parsing).

**Papers and reading locations:** [Goodman, Semiring Parsing](https://aclanthology.org/J99-4004.pdf); [Torch-Struct](https://aclanthology.org/2020.acl-demos.38/) — Goodman §1, Figures 1–3; Torch-Struct CKY implementation.

**Input, calculation and output.** Input: a sentence and weighted grammar rules that either produce a token or combine two child categories. Output: total parse probability and best parse score. Build answers for short spans before longer spans; consider every permitted split and matching grammar rule. [Local _cky](../softwareworkload/cky_parsing/workload.py#L116).

**How the operators form the computation.** 

1. combine sentence tokens with lexical rules to initialize all single-token spans. Mechanism sources: [D4 · Semiring Parsing](workload_operator_evidence.md#d4). Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

2. For each target span length, rename the left child's end as the split point and retain its start and category. Mechanism sources: [D4 · Semiring Parsing](workload_operator_evidence.md#d4). Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

3. scan the right-child table: use its start as the split point and subtract the target length from its end to compute the required parent start; discard impossible endpoints. Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

4. combine left and right children on both parent start and split point. This enforces adjacency and target length before producing child-category pairs. Mechanism sources: [D1 · Provenance Semirings](workload_operator_evidence.md#d1), [D4 · Semiring Parsing](workload_operator_evidence.md#d4). Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

5. combine those pairs with compatible binary grammar rules. Add contributions for total probability or keep the best log-score for the best-score evaluation. Mechanism sources: [D4 · Semiring Parsing](workload_operator_evidence.md#d4). Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

6. scan attaches the parent's end; update inserts only newly computed spans of this length. Finish by reading the sentence-wide start category. Mechanism sources: [D4 · Semiring Parsing](workload_operator_evidence.md#d4), [D5 · Dyna: Practical Weighted Dynamic Programming](workload_operator_evidence.md#d5). Interface mapping / exact execution: [Local _parse](../softwareworkload/cky_parsing/workload.py#L192).

**Terms, state and implementation limits.** A grammar category names a type of phrase; a binary rule says which two child categories can form a parent. The start category is the category required for the complete sentence. Inside probability adds all compatible parses, not just the most likely tree. A span is a contiguous sentence interval; a chart is the saved table of span/category answers. The optional rule-first schedule first joins left children with rules and caches that table, then joins right children; newly completed spans extend the cache. Inserting only fresh span keys avoids recounting old probabilities. The best-score output is not a reconstructed parse tree or an exact k-best list. [Local _parse](../softwareworkload/cky_parsing/workload.py#L192); [Semiring Parsing](https://aclanthology.org/J99-4004.pdf).

**Bottleneck/fit [H].** Pair-first joins can enumerate nonterminal pairs without a matching rule. Rule-first indexing, split constraints and early span-length filtering reduce work before hardware sharing matters. Chart residency and segmented reductions may reduce materialization after the algorithm is matched.

Attribution sources: the local code above and [Semiring Parsing](https://aclanthology.org/J99-4004.pdf) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

### 4.18 `graph_bfs_sssp` — frontier BFS and weighted shortest paths

[Walked-through example, native code and lowering evidence](workload_operator_evidence.md#graph_bfs_sssp).

**Papers and reading locations:** [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619) — benchmark-kernel definitions: BFS and SSSP.

**Input, calculation and output.** Input: a graph and source; weighted shortest paths also use nonnegative edge costs. BFS returns minimum hop counts, while SSSP returns minimum total edge cost. A path with fewer hops need not have a smaller weighted cost. [Local native](../softwareworkload/graph_bfs_sssp/workload.py#L58).

**How the operators form the computation.** 

1. BFS: initialize the new-vertex and visited tables with the source. combine new vertices with outgoing edges, mask out visited vertices, update visited state and record the discovery round as hop count. Mechanism sources: [D8 · GraphBLAS C API Specification (standard)](workload_operator_evidence.md#d8), [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619). Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

2. Repeat BFS until no new vertex is found. Each vertex's first discovery determines its level. Mechanism sources: [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619). Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

3. Frontier SSSP: start the source at no added cost. combine improved distances with edges, add edge costs and keep the lowest candidate for each destination. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6), [D8 · GraphBLAS C API Specification (standard)](workload_operator_evidence.md#d8). Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

4. update all genuinely better distances and repeat. Do not filter merely by previously visited support, because an existing distance can improve. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6). Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

5. Dijkstra is an alternative SSSP schedule: select the lowest pending distance, remove it, settle that vertex, expand its adjacency row, exclude settled destinations and update pending distances. Mechanism sources: [D6 · Semiring Frameworks and Algorithms for Shortest-Distance Problems](workload_operator_evidence.md#d6). Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

6. Run each chosen schedule with its own state and stopping condition; they are alternatives, not stages to concatenate. Interface mapping / exact execution: [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111).

**Terms, state and implementation limits.** Relaxing an edge means checking whether reaching its source and traversing that edge gives a cheaper destination distance, and updating the distance if so. A heap is a data structure for repeatedly retrieving a smallest pending score. Frontier means the vertices or distances to expand in the current round. BFS levels and weighted distances are different answers. Dijkstra's settled-state rule relies on nonnegative edges. The independent native code uses a BFS queue and a Dijkstra heap. [Local _sssp_frontier / _sssp_dijkstra_select / _bfs_levels](../softwareworkload/graph_bfs_sssp/workload.py#L111); [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619).

**Bottleneck/fit [M,H].** GAPBS measurements expose graph-dependent behavior: narrow road frontiers and many bucket rounds interact with barriers; wider graph kernels can scale. Its delta-stepping and direction-optimized traversal are not the same schedule as a simple relational frontier. Fused state access and better scheduling may help, but CPU waiting is not proof of saturated DRAM. Generated edge weights are not geographic road distances.

Attribution sources: the local code above and [The GAP Benchmark Suite](https://arxiv.org/abs/1508.03619) for the algorithmic mechanism; facts labelled M/P are checked against [baseline report](baseline_report.md); whether sharing mitigates these effects remains a local hypothesis, not a sym1 speedup established by the cited paper.

## 5. CogSys-style profiling and bottleneck attribution

### 5.1 Measurement protocol

Use [CogSys](https://zishenwan.github.io/publication/HPCA25_CogSys.pdf) as a method: connect application stages, operator mix, memory behavior and scaling to a specific execution mechanism. Do not transfer its VSA bottleneck or numerical ratios to unrelated workloads.

For each workload, record:

1. **Contract and timing boundary:** requested answer, preprocessing/indexing, compilation, kernel, feedback/control, transfer and postprocessing. Provide kernel and complete-path latency separately.
2. **Algorithmic work and critical path:** §2.3 event counts, rounds, frontier-width distribution, fan-in/out, join intermediates, queue operations and true dependency depth.
3. **Resource behavior:** CPU useful instructions, IPC, cache/TLB behavior, actual memory bandwidth, synchronization/lock time and NUMA placement; GPU occupancy, active lanes, memory stalls and launches when GPU measurements exist.
4. **Storage and movement:** resident state, index/queue capacity, allocation volume, intermediate spill and actual transferred bytes. These are distinct quantities.
5. **Controlled scaling:** thread count, query batch, graph/circuit structure, sparsity and available memory/bandwidth; preserve correctness and identify changes in algorithm.
6. **Causal ablation:** perturb the suspected cause while holding others fixed; report noise, host interference, warm-up and repeated samples.

A conventional FLOP roofline is suitable only for compatible arithmetic. For joins and queues, also use operation-specific rates and bytes per useful event. Low multiply-active time in a sparse stream can mean few coordinate matches rather than memory saturation.

### 5.2 What current evidence establishes

Numerical values, commands, configurations and raw-run links remain in the generated [baseline report](baseline_report.md), avoiding independently maintained copies in this plan.

| Evidence | Supported interpretation | Boundary |
|---|---|---|
| GAPBS road SSSP and CPU samples | Fine-grained bucket rounds and synchronization/waiting are important on this host | Sampled cycle share is not wall-time fraction; shared-host load affects waiting |
| GAPBS triangle counting | CPU parallelism can be effective for the measured graph/kernel | Does not characterize all graph inputs or all symbolic workloads |
| Torch-Struct and schedule ablation | Matrix composition can add arithmetic and temporary tensors compared with inference-only vector recurrence | A software schedule gain is not a sym1 hardware gain; positive allocation sums are not peak RSS or DRAM traffic |
| Compiled Souffle reachability | Indexed recursive-rule work and relation maintenance can be separated from I/O | Source-batched reachability is not complete all-pairs TC; copying is not established as the universal dominant cost |
| SAM adjacency intersections | Coordinate work, skips and pipeline fill/drain affect primitive-model cycles | Resident primitive model has no calibrated end-to-end graph/DRAM timing |
| SCALE-Sim shape studies | Narrow dimensions can underfill a dense array even without compute-window memory stalls | Prefetch is separately accounted; ordinary GEMM timing does not model LSE circuitry |
| Existing algebra traces / analytical model | Expose possible work inflation and execution-mode diversity | Neither CPU/GPU profiles nor cycle-accurate hardware proof |

The remaining workload-specific entries in §4 are hypotheses, not “all workloads profiled.” Source inspection identifies what to measure; it does not establish which term dominates native runtime.

### 5.3 Can shared operators solve the bottleneck?

| Cause | Mechanism to test | What sharing alone cannot establish |
|---|---|---|
| Excess join/DP/agenda work | Better join order, sparse intersections, vector recurrence, ordered/counter-based updates | Avoided work is an algorithm/compiler gain unless isolated separately |
| Intermediate materialization and repeated dispatch | Fused producer/consumer paths with resident state | Physical PE sharing is not required for fusion; a dedicated fused baseline is necessary |
| Idle resources from changing operator mix | Reassign shared PE resources across phases | Net benefit after routing, numerical units, switching and clock loss |
| Narrow frontiers or causal search chains | Efficient local feedback and independent-query concurrency | Single-query dependence is still present; batching changes latency/state trade-offs |
| Random memory and bank conflicts | Better layout/indexing, buffering and scheduling | A shared ALU does not create memory bandwidth or locality |
| Persistent queue/state overhead | Queue and table support with bounded capacity and spill handling | Batch top-k is not a persistent queue; state capacity can dominate area |

The central distinction is **shared mathematical operators**, **fused execution**, and **shared physical resources**. They are related design layers, not interchangeable explanations of performance.

## 6. Benchmark and evaluation plan

### 6.1 Benchmark coverage

These are proposed benchmark families, not a record of completed application runs. Assignments to main results, regression and held-out groups remain proposals until ruled.

| Domain | Candidate workload/data family | Required semantic check |
|---|---|---|
| Datalog / program analysis | Author Datalog suites, graph closure, same-generation, points-to facts | Same rules, indexes, output materialization and closure semantics |
| Differentiable reasoning | Scallop/Lobster task suites | Equal provenance, proof retention and training/inference mode |
| Probabilistic circuits | PyJuice/Einsum circuit settings; KLay compiled circuits | Circuit validity, normalization, evidence and marginal/MPE/WMC distinction |
| Sequence DP / parsing | Torch-Struct chain and grammar tasks; annotated sequence/tree corpora | Scores versus paths/trees; forward versus backward pass |
| Constrained decoding | Ctrl-G/GeLaTo generation tasks; grammar/schema workloads | Full completion probability versus legal-token mask; tokenizer and EOS |
| Graph analytics | GAP, Graphalytics, SuiteSparse and public graph collections | Directedness, duplicate/self-edge normalization and weight construction |
| RPQ | Labelled graph/query suites from cuRPQ or PIM RPQ papers | Regex semantics, epsilon handling, sources and duplicate-answer policy |
| SAT | SAT Competition / SATLIB CNF, recorded cube assignments | BCP kernel versus full solver, conflict exit, watched-state ownership |
| Planning | IPC PDDL translated with a pinned planner | Cost semantics and heuristic-only versus full search |
| Knowledge graphs | Link-prediction and rule-reasoning datasets | Scalar paths versus learned messages; query relations and ranking protocol |
| Ordered search | Grid maps and road-graph search | Edge costs, heuristic validity, tie order and termination |
| Beam decoding | Translation/speech or language-generation tasks | Prefix identity, model scores, length penalty and state reorder |
| VSA | torchhd/NVSA-style tasks and codebooks | Binding representation, encoding cost and recovery accuracy |
| Bayesian networks | UAI / bnlearn-style discrete networks | Same factors, evidence, induced width and output marginal |
| End-to-end composition | REASON's AlphaGeometry, R2-Guard, GeLaTo, Ctrl-G, NeuroPC and LINC pipelines | Full application boundaries and remaining symbolic/control work |
| Held-out candidates | Previously unused application families, such as graph pattern queries | Freeze selection before mapping; an already tuned in-repo workload cannot be held out |

Real datasets already used by local graph baselines are distinct from the synthetic inputs in the algebra workload suite. An operator-level golden does not establish task-level model accuracy or full application coverage.

### 6.2 Baselines and sharing ablations

Use native optimized software for each family (§3.3), and independent goldens for correctness. Compare the same algorithm on shared and dedicated hardware first; separately compare the best supported algorithm on each platform.

The hardware comparison requires:

- A fixed mixture of specialized resources and a shared-resource design under equal total area and equal memory/bandwidth assumptions.
- Both designs with matched fusion and state-management capability, plus an explicit unfused ablation.
- Matched algorithms/schedules, followed by independently labelled join-order, priority-policy and batching ablations.
- Synthesized area, achievable frequency and power for arithmetic/value variants, comparators, key routing, registers, queues, banks and interconnect.
- Full cycle behavior for feed/drain, contention, feedback, termination, switching, state spill and backpressure.
- Faithful reference-engine capabilities; generic models labelled as such instead of being named after incompletely reproduced papers.

For one operation type, ideal service time is **derived** by dividing the number of operations by achieved operations per second. Dependencies, movement and overlap also affect execution. Sharing may change the processing rate or resource allocation; an algorithm change may change the operation count. Report both. This is local evaluation accounting, not a speedup established by a cited paper. Diversity of operation-mode shares in the exploratory model is not evidence of area-normalized benefit.

### 6.3 Correctness, fairness and held-out evaluation

Correctness regression must pass. Performance regression is an observation to report honestly, including slowdowns; “must never be slower than REASON” is not a valid benchmark policy.

Pin inputs, revisions, precision, output semantics, thread affinity and timing boundaries. Retain preprocessing and full-path costs even when also reporting steady-state kernels. On the shared host, separate structural work evidence from noisy timing evidence. Enforce the user-approved resource budget recorded by the existing runner metadata.

Genuine held-out evaluation requires an untouched workload or family selected before design/mapping decisions. All currently mapped workloads, including VSA, have already informed the design and are coverage tests rather than held-out evidence.

## 7. Repository status, corrections and conventions

### 7.1 Evidence inventory

| Artifact | What exists | Interpretation |
|---|---|---|
| `algebra/`, `softwareworkload/`, `tests/` | Operator interpreter, workload programs and independent goldens | Functional decomposition evidence; see §4 for application boundaries |
| [baseline_report.md](baseline_report.md), [scripts](scripts/) | CPU GAPBS/Souffle/Torch-Struct and SAM/SCALE experiments with checks and raw logs | Measured CPU or primitive-model evidence only |
| [upstream_status.md](upstream_status.md), [lock](upstream_lock.json) | Pinned workload and simulator source inventory | Distinguishes checkout, build, runnable flow and successful experiment |
| [systolicarch/model.py](../systolicarch/model.py) | Closed-form architecture estimates | Analytical sensitivity, not cycle-level execution or synthesized PPA |
| [docs/findings.md](../docs/findings.md), [coverage](../docs/coverage.md) | Earlier decomposition/model analyses | Historical analytical claims require their original accounting and audit |
| GPU comparisons | Missing on the current host because the NVIDIA driver is unavailable | CPU/simulator measurements do not fill this evidence gap |
| sym1 cycle simulator and shared-PE RTL | Required to establish the hardware claim | Latency/area/frequency/power values remain `TBD[src]` |

### 7.2 Corrections incorporated from the previous draft

| Previous issue | Corrected interpretation |
|---|---|
| “No hardware takes relations as its interface” | Q100 and earlier relational hardware are direct precedents |
| REASON described as only a static acyclic DAG | Its BCP feedback, batch interface and off-chip clause access are acknowledged |
| Revet assigned to ASPLOS and treated as lacking dynamic control | HPCA 2024; dataflow threads, branches/loops and Aurochs precedents included |
| Partial-bitstream cost used to dismiss FPGA mode switching | Overlay runtime selection is distinct from circuit reconfiguration |
| SAM static topology equated with static cycles or impossible extensions | Runtime token flow/backpressure and artifact boundaries are explicit |
| All operators claimed to share one cheap merge cell; value algebra “free” | Physical sharing and numerical/state costs are hypotheses requiring RTL evidence |
| Arbitrary semiring fixpoints and proof tags collapsed to scalar values | Convergence, delta semantics and provenance contracts are explicit |
| Ctrl-G liveness presented as full conditioned generation | Prefix proxy and finite-horizon backward probability are separately specified |
| Random PC DAG called valid likelihood/WMC; max pass called general MAP | Circuit structural conditions and requested inference task must be checked |
| NBFNet, beam, VSA and planning proxies equated with whole applications | Actual local mathematics and full-application boundaries are documented |
| All CPU kernels described as lacking parallel efficiency | Workload-specific work, synchronization, shape and memory causes are distinguished |
| Analytical speedups or primitive activity interpreted as hardware proof | Evidence types and calibrated-model requirements are separated |
| Already used VSA labelled held-out; no-slowdown benchmark rule | Untouched held-out selection and honest reporting of regressions |

The earlier [literature review](literature_review.md) remains the audit trail; its original section references refer to the pre-revision plan.

### 7.3 Documentation and reporting conventions

Follow [project conventions](../CLAUDE.md) and `aimd/AI-RULES.en.md`. Repository documentation is English; the Chinese working copy stays under ignored `local/`. Experimental numbers require measured/derived/cited provenance, exact accounting and script-checked transfer. This plan links the validated numerical report instead of duplicating its tables. Unestablished hardware constants use `TBD[src]`.

Run outputs remain append-only. Statements about inspected source, local measurements, primitive simulation, analytical estimates and future design evidence must remain distinguishable.
