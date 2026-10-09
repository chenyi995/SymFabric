# sym1 literature review and claim audit

Review date: 2026-09-26. This document records the audit of the pre-revision proposal. Its claim corrections are now incorporated into the [revised plan](README.md), which also expands the related-work coverage and specifies every local workload mathematically. Original section numbers and the assessment below refer to the earlier draft, not to unresolved errors in the revised plan. Source revisions are recorded in `upstream_lock.json`; source acquisition and execution status are different claims.

The expanded [workload/operator evidence handbook](workload_operator_evidence.md) now supplies a separate decomposition bibliography, individual-operator mathematics/code, and a native-versus-lowered teaching example for each current workload. It distinguishes literature-established mechanisms from local derivations; its examples are not performance data.

## Assessment

**Assessment of the earlier draft: conditional go for baseline characterization; its novelty argument could not pass unchanged.** Shared hardware for relation matching, value algebra, state updates and ordered work remains a testable architecture hypothesis. The inspected literature does not establish that this exact combination is already solved, but neither does a search establish its absence. The defensible contribution would be a measured implementation advantage at equal semantics and work, not the mathematical unification, the existence of data-dependent control, or fusion alone.

The highest-risk comparison is a capable sparse/dataflow architecture with dynamic control, followed by optimized semiring-relational software. A restricted tensor-stream model must not be renamed Revet, and an arithmetic DAG model must not be renamed REASON. CPU measurements and upstream scheduling simulations can locate candidate bottlenecks; they cannot establish GPU speedup, silicon efficiency or generality tax.

## Corrections to the proposal

| Proposal claim/location | Primary-source finding | Consequence |
|---|---|---|
| Sections 0 and 3.1 treat REASON as only a static acyclic DAG and lacking batching | REASON describes hardware watched-literal traversal, a BCP FIFO and feedback, conflict handling, and an API with batch identifiers and batch size. See sections V-D and V-E and Listing 1. | Compare relational state semantics and actual queue/control costs, not a claimed absence of feedback or batches. |
| Section 3.3 infers all REASON working sets fit its SRAM | REASON section V-D explicitly allows clauses in remote scratchpad/DRAM with selected clauses cached on chip. | SRAM capacity is not evidence that complete datasets fit on chip. |
| Sections 1.1 and 3.1 classify Revet as static and cite the wrong venue | The author page identifies HPCA 2024. Sections II-C and III of the paper describe dataflow threads, nested parallel patterns and data-dependent loops; the Aurochs predecessor already mapped hash joins and index traversal. | “Dynamic control” and “hash joins” cannot serve as standalone differentiators. |
| Section 1.1 uses partial-bitstream latency to rule out FPGA operator switching | GraphLily is an overlay; its public host code passes the semiring operation as a kernel argument. | Distinguish changing a configured circuit from selecting a mode inside that circuit. The ICAP argument does not establish the claimed limitation. |
| Sections 1.1 and 3.1 interpret SAM's operator list as a hardware impossibility | SAM Definition 3.6 specifies its ALU operations. Sparse scanners, token streams and backpressure still react to runtime data. | An omitted primitive is not proof that the architecture cannot be extended; static topology is not a fixed cycle schedule. |
| Section 3.3 equates Lobster's proof provenance with a scalar Viterbi back-pointer | Lobster section 3.5 describes multiple provenances. Its top-one proof carries a conjunction of facts and merges proof sets while checking conflicts. | A scalar-score experiment does not cover proof-set maintenance or differentiable provenance. |
| Section 2 presents one accumulator/fixpoint expression across all value domains | Semiring convergence and valid semi-naive evaluation require explicit algebraic conditions. | Idempotent Boolean reachability is a sound initial test; counting/probability recursion needs derivation accounting and termination conditions. |

Sources for the audit: [REASON](https://arxiv.org/html/2601.20784v1), [Revet paper](https://arxiv.org/html/2302.06124v2), [Revet venue](https://fredrikbk.com/hpca24.html), [GraphLily](https://github.com/cornell-zhang/GraphLily), [SAM](https://arxiv.org/html/2208.14610v2), [Lobster](https://arxiv.org/html/2503.21937v2), [semiring convergence](https://arxiv.org/abs/2105.14435).

The GraphLily code evidence is `graphlily/module/spmv_module.h`, `set_semiring` and the `kernel_.setArg` call carrying `semiring_.op`, plus `graphlily/hw/overlay.h`. This is a checked code-level counterexample to requiring a partial bitstream for every operation change; it does not establish the cost of every possible FPGA datapath.

## Mathematical and semantic foundations

| Work | Relevant established result | Requirement for sym1 |
|---|---|---|
| [Generalized distributive law](https://authors.library.caltech.edu/records/sw1pm-bwj40) | Sum-of-products computations over appropriate algebraic domains unify inference and decoding algorithms. | Treat the common expression as background. |
| [FAQ / InsideOut](https://arxiv.org/abs/1504.04044) | Functional aggregate queries combine variable elimination and joins; legal variable order affects semantics and intermediate complexity. | Preserve aggregation order restrictions and report intermediate work, not only useful multiplications. |
| [Dyna](https://aclanthology.org/H05-1036/) | Weighted deductive programs compile dynamic programming over algebraic values. | Weighted rules and agenda execution are precedents, not new interface semantics. |
| [Datalog over (pre-)semirings](https://arxiv.org/abs/2105.14435) | Least-fixpoint convergence and semi-naive evaluation depend on the ordered algebra. | Specify which value domains support which recursion and delta rules. |
| [DBSP](https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf) | Incremental relational computations extend to aggregation and non-monotone recursion through an explicit difference semantics. | A changed-key bitmap alone is not a general replacement for weighted differences. |
| [Scallop](https://arxiv.org/abs/2304.04812) | Relational programs combine recursion, aggregation, negation and provenance-based differentiable reasoning. | Distinguish scalar semiring kernels from the full language and training pipeline. |

**Semantic inference.** Repeatedly accumulating the entire right-hand side under ordinary addition can recount old derivations. The proposed runtime must distinguish newly discovered support from newly contributed weight. Likewise, unbounded counting on a reachable cycle need not terminate. A finite-domain idempotent test does not validate arbitrary weighted recursion. Floating-point reduction order also matters: bit-exact integer/Boolean checking and tolerance-based floating-point checking are different contracts.

The same distinction applies to control. A priority queue does not remove the dependency from a popped item to its relaxations and new insertions. Independent queries can expose additional work, but query batching must be reported together with single-query latency, replicated state and service-time trade-offs. These are deductions from the execution dependencies, not measured sym1 improvements.

## Closest hardware and execution models

| System and primary source | Interface and mechanisms actually described | Fair comparison axis | Local artifact boundary |
|---|---|---|---|
| [REASON](https://arxiv.org/html/2601.20784v1) | Combined probabilistic and logical engine, arithmetic/Boolean tree processing and specialized BCP memory/control | Equivalent probabilistic/logical kernels, clause-access traffic and control state | No public implementation link located in inspected material; generic Accel-Sim is not its implementation |
| [DPU-v2](https://arxiv.org/abs/2210.13184) | Static-connectivity irregular arithmetic DAG execution with compiler scheduling | Dependency depth, register/bank pressure, useful-node issue rate | Author RTL/compiler/input DAGs cloned; reproduction needs its VCS flow |
| [SAM](https://arxiv.org/html/2208.14610v2) | Sparse scanners, coordinate intersections/unions, repeat, reduction and arithmetic streams | Coordinate work, empty fibers, skip support, FIFOs and backpressure | Public Python primitive simulator available; this is not full Onyx silicon |
| [Capstan](https://arxiv.org/abs/2104.12760) | Sparse vector reconfigurable dataflow architecture and memory machinery | Sparse-access scheduling, lane utilization and memory issue constraints | Paper-level comparator; no matched simulator located in inspected sources |
| [Revet](https://arxiv.org/html/2302.06124v2) | Hierarchical dataflow threads with branch and loop control | Dynamic-region synchronization and live-state communication | Paper-level comparator; cannot emulate it merely by removing sym1 queue mode |
| [SIMD²](https://research.nvidia.com/publication/2022-06_simd2-generalized-matrix-instruction-set-accelerating-tensor-computation-beyond) | Generalized matrix operations beyond ordinary multiply-add | Dense shape utilization and value-operation support | SCALE-Sim measures dense scheduling, not SIMD²'s complete ISA or semiring circuitry |
| [GraphLily](https://github.com/cornell-zhang/GraphLily) | HBM FPGA GraphBLAS overlay and runtime middleware | Sparse-vector density, memory layout, masking, repeated graph iterations | Source cloned; requires FPGA platform/toolchain for hardware reproduction |
| [Q100](https://users.cs.duke.edu/~lkw34/papers/wu-q100-asplos2014.pdf) | Relational operator tiles with stream/pipeline execution | Shared versus dedicated operator resources and intermediate spilling | Strong interface/fusion precedent; not a ready interchangeable simulator |

**Architecture inference.** A comparison can still investigate whether shared PE resources reduce idle specialization, whether fused delta/state processing reduces materialization, and whether sparse and dense modes can share control/storage economically. It must account for banking, key width, variable value width, table capacity, output bursts and feedback deadlock. A comparator, register and FIFO sketch is not yet evidence that probe/update and persistent priority ordering have negligible area or timing cost.

For SAM specifically, per-block statistics are useful only with the complete token path and an independent answer check. A low multiply-active fraction can be caused by few coordinate matches even without any modeled DRAM stalls. It must not be relabelled as measured memory bandwidth saturation.

## Software baselines and what each stresses

| Family | Primary sources and cloned implementation | Native workload / structure | Bottleneck evidence required |
|---|---|---|---|
| Relational fixpoint | [Souffle](https://souffle-lang.github.io/profiler), [GDlog](https://github.com/harp-lab/gdlog), [VFLog](https://ojs.aaai.org/index.php/AAAI/article/view/33665) | Indexed relations, recursive joins, delta merge and duplicate elimination; graph reachability and program analysis | Join time versus relation/index maintenance; round count; tuples examined and retained; input/output excluded from kernel timing |
| Multiway Datalog | [SRDatalog](https://arxiv.org/html/2604.20073v1), author-group `srdatalog-python` checkout | Attribute-at-a-time intersections for cyclic queries, skew handling and deterministic materialization | Binary intermediate growth versus intersection work; key skew; count/materialize and delta/index overhead |
| Provenance reasoning | [Lobster](https://arxiv.org/html/2503.21937v2), Scallop | GPU relational operations plus value tags, recursion and differentiable provenance | Tuple structure cost and tag-processing cost measured separately; compare equal provenance semantics |
| Probabilistic circuits | [PyJuice](https://arxiv.org/abs/2406.00766), [KLay](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3ba82362eb0aa75487069f19fde794fe-Abstract-Conference.html) | Layered/block-structured probabilistic computation versus compiled arithmetic-circuit layers | Layer widths, batch size, gather/scatter, layout, backward pass and compilation amortization |
| Semiring sequence DP | [Torch-Struct](https://aclanthology.org/2020.acl-demos.38/), `pytorch-struct` | Parallel chain scan and structured dynamic programming | Matrix-scan versus vector-recurrence work, temporary tensors, reductions and numerical operators |
| Graph traversal | [GAPBS](https://arxiv.org/abs/1508.03619), [GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS), GraphBLAST/Gunrock/cuGraph | Direction-optimized BFS, delta-stepping SSSP, oriented triangle intersections | Frontier width, barrier cost, skew, direction/ordering choices and bytes moved |
| Integrated GPU reasoning | [XLOG](https://arxiv.org/abs/2609.27203) | Newly posted CUDA-native reasoning engine including Datalog and weighted-model-counting components | Track as a recent preprint, not established reproducible performance evidence |

SRDatalog's source checkout is an author-group implementation, not yet a verified paper-artifact revision. Its preprint is relevant because it explicitly distinguishes join work explosion from low hardware utilization. A binary join can be a poor algorithmic baseline even if its accelerator implementation is efficient. Similarly, default Torch-Struct scan is a real software baseline but need not be the strongest CPU schedule for inference-only chains.

Lobster paper and repository capabilities must be matched at the locked revision. Reporting a limited CPU scalar-value kernel as a reproduction of its differentiable tasks would be incorrect. Published speedups from different paper versions, datasets, GPUs or training/inference modes are not pooled in this review.

Additional cloned workloads cover SAT (`kissat`, `cadical`), planning (`downward` and benchmark inputs), constrained decoding (`Ctrl-G`, `xgrammar`), knowledge-graph inference (`NBFNet`), vector-symbolic computation (`torchhd`), beam search (`transformers`) and Bayesian inference (`pgmpy`). Their presence is source coverage only. The CPU graph and DP experiments below do not stand in for end-to-end SAT solving, neural decoding or training.

## Baseline methodology and interpretation

The runnable local characterization uses unchanged upstream implementations wherever possible. Dedicated scripts under `scripts/` supply inputs, compare independent goldens, record exact commands and preserve timestamped raw outputs. The user-approved resource ceiling is enforced by the logged runner. The source inventory pins every checkout; failed setup attempts remain separate from successful runs.

| Baseline | Input and checking | What the result can establish |
|---|---|---|
| GAPBS CPU | Public SNAP graphs normalized to simple undirected graphs; native independent verifiers | Kernel latency and thread scaling on this host, plus separate sampled CPU hotspots |
| Compiled Souffle CPU | Source-indexed reachability on the same graph data; independent deque BFS checks complete result sets | Recursive-rule, copy/merge and I/O time; an indexed relational baseline, not whole-closure performance |
| Torch-Struct CPU | Seeded synthetic potentials; independent NumPy recurrence | Semiring DP execution costs and schedule effects; not task accuracy on a language dataset |
| SAM simulator | Adjacency-list intersections drawn from real graphs; set-intersection coordinates and reduction checked | Primitive-model cycles, token overhead and skip/FIFO sensitivity; not full triangle-kernel latency |
| SCALE-Sim | Explicit GEMM shapes, upstream configuration and native reports | Dense scheduling utilization and supplied-bandwidth sensitivity; not nonlinear semiring latency |

All hardware constants used for these baselines come from upstream configurations, with diagnostic shape/seed/thread choices explicitly distinguished from hardware measurements. The dense model's operation latency is not substituted for log-sum-exp, and its memory interface is not called a calibrated HBM system. CPU performance counters are not GPU utilization counters.

## Evidence needed for the project claim

The baseline characterization can identify where work, synchronization, sparse matching and temporary materialization dominate. A positive result would narrow the hardware problem, not automatically validate the proposed PE. Establishing the paper claim additionally requires a cycle-level sym1 implementation with state/feedback, matched-capability comparators, and synthesized area/frequency/power for shared and dedicated cells. The existing closed-form model in `systolicarch/model.py` does not provide those measurements.

The decisive distinction is between **avoiding unnecessary work**, **executing the same work more efficiently**, and **moving less data between operators**. Each has a different denominator. Software schedule gains must remain software gains; isolated primitive utilization must remain a primitive result; an end-to-end gain needs the complete pipeline including CPU control and data preparation.

Measured local findings and raw-run links are generated separately in `baseline_report.md`. GPU baseline results remain missing because this host has no working NVIDIA driver, as established in the environment checks. No CPU or abstract-simulator result is used to fill that gap.
