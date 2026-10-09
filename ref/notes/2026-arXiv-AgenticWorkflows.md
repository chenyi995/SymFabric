# Architectural Implications of Agentic AI Workflows (arXiv 2026)

Jirong Yang, Peizhe Liu, Chaojie Zhang, Jovan Stojkovic; UT Austin / Microsoft Azure. arXiv:2608.04458v1 [cs.AI], 5 Aug 2026 (header, p. 1).
Characterization paper (Azure fleet study + controlled study of four open-source frameworks) plus Agora, a server-runtime prototype. No accelerator is designed; the hardware is a commodity CPU+GPU server.

- **Workload**: Agentic AI = an LLM inside a control loop that invokes tools (web search, code execution, retrieval) and hands off to other agents (Sec. II, p. 2). (a) Fleet: a 24-hour trace of production agentic requests on Microsoft Azure, with time, tool activity and CPU metrics attributed to three host roles: schedulers, orchestrators, runners (Sec. IV, p. 4). (b) Controlled: SWE-Agent (single-agent, sequential, tool-intensive coding), Trae (sequential coding, long tool chains, three model endpoints), CORAL (four-agent, parallel, homogeneous-model evolutionary discovery), Owl (heterogeneous workforce: coordinator, six text roles, vision and speech models; run on GAIA) (Sec. IV, p. 4; Sec. V-B, p. 11). Taxonomy axes: orchestration (host- vs LLM-orchestrated), execution structure (sequential vs parallel), model composition (homogeneous vs heterogeneous) (Table I, p. 3).
- **Operators**: The paper treats tools as opaque CPU/network phases and measures nothing inside them. Our mapping (not the paper's): orchestrator dependency tracking over the workflow DAG = INDEX + SELECT over a step-dependency relation, iterated until the ready frontier is empty (boolean semiring); "tool search"/tool discovery (Fig. 2, p. 4), search & retrieval and data/query tools (Fig. 11, p. 6) = INDEX + FILTER + COMBINE over weighted relations (top-k or join); code execution, builds, tests and "eval" tools (Fig. 3 legend: bash, read, write, edit, eval, other, p. 4) are general-purpose CPU compute with no SymFabric operator. Agora's MILP placement (Sec. V-A-2, p. 9) is off the critical path.
- **Data scale**: fleet trace 24 hours, request count not given (Sec. IV, p. 4). One production request spans nearly a minute with multiple LLM calls, agent hand-offs, 3 tool-discovery operations and 3 tool executions across 2 tools (Sec. IV-A, Fig. 2, p. 4). One CORAL run = 580 LLM calls interleaved with 552 tool invocations, from millisecond file-system operations to multi-second compilations and evaluations (Sec. IV-A, Fig. 3, p. 4). Concurrent tasks Q = 1 to 32 per server (Sec. IV, p. 4); C in {1, 8, 32} in the pooling study (Sec. V-B, p. 10). Sparsity is not discussed.
- **Hardware mechanism**: Agora, a software runtime on commodity servers. (1) CPU harvesting of idle runner cores with role-based isolation, orchestrator hints (workflow graph and compute-bound vs I/O-bound tool type) and grace-guarded adaptive retreat (Sec. V-A-1, Fig. 16, p. 7-8). (2) GPU harvesting: consolidate agents onto shared vLLM instances, oversubscribe GPU memory with host-memory staging and orchestrator-driven prefetch of the next agent's state, MILP placement over profiling data (Sec. V-A-2, Fig. 18, p. 8-9). (3) Role-aware core pools (scheduler / orchestrator / runner) and per-task pinning inside the runner pool (Sec. V-B, Fig. 19, p. 9-11). Future-hardware asks: a dedicated agent-scheduling / context-switch engine; way-partitioned or agent-tagged caches, TLBs and branch predictors; heterogeneous or morphable cores (Sec. V-C, p. 11).
- **Simulation / implementation method**: measurement only, no simulator. Fleet: time/tool/CPU attribution plus a host-wide PMU snapshot from production nodes (Sec. IV, p. 4; Sec. IV-B, p. 6). Controlled study and Agora: one server with a 96-core AMD EPYC 7V12 and eight NVIDIA A100 GPUs, one vLLM instance per agent role (Sec. IV, p. 4); DCPerf as the co-located harvester (Sec. V-A-1, p. 8).
- **Limitations**: fleet distributions (Figs. 4, 9, 10, 11, 12) are plots only; the text quotes a single fleet number (tool time comparable to or exceeding LLM time for more than 27% of requests, Sec. IV-A, p. 4), so category fractions and per-request tool-call counts cannot be reused. Tool time is split only into compute-bound vs I/O-bound, never into symbolic vs generic compute. Single server, four frameworks, A100-class GPUs; no accelerator-side tool offload is evaluated; energy appears only as GPU energy-efficiency deltas (Table III, p. 9). Agora is a prototype; no code release is mentioned.
- **Numbers worth reusing**: see the three tables below (every value carries its location).
- **Implications for SymFabric**: see "Implications for a symbolic accelerator" below.
- **Open source**: none stated for Agora or the fleet trace. The frameworks used (SWE-Agent, Trae, CORAL, Owl) and the harvester (DCPerf) are existing open-source projects (Sec. IV, p. 4; Sec. V-A-1, p. 8).

## Where the time goes

| Workflow | Stage | Fraction of latency or resources | Location |
|---|---|---|---|
| Azure fleet, 24-h trace | Tool execution vs LLM inference | Tool time is heavy-tailed; comparable to or exceeding LLM inference time for more than 27% of requests | Sec. IV-A, Fig. 4, p. 4 |
| Azure fleet, one representative request | Whole lifecycle | Nearly 1 minute; multiple LLM calls, agent hand-offs, 3 tool-discovery ops, 3 tool executions across 2 tools; tool phases are "a substantial fraction" of end-to-end latency | Sec. IV-A, Fig. 2, p. 4 |
| Azure fleet | Tool-call mix by functional category | Code / API execution is the single largest category; then GitHub/DevOps, search & retrieval, data/query (databases), communication/productivity, business/workflow (percentages only in the figure) | Sec. IV-A, Fig. 11, p. 5-6 |
| Azure fleet | Tool calls per active request (avg and P99) | "Many" per request on average; P99 far heavier and category-dependent (values only in the figure) | Sec. IV-A, Fig. 12, p. 5-6 |
| Azure fleet, per host role | Scheduler / orchestrator / runner signature (CPU, network I/O, context switching, memory pressure, disk ops) | A CPU-heavy role is light on network; a high-dispatch-rate role is light on disk (values only in the figure) | Sec. IV-B, Fig. 9, p. 6 |
| CORAL, one run (4 agents, parallel) | LLM calls vs tool invocations | 580 LLM calls, 552 tool invocations (bash/read/write/edit/eval/other); ms file-system ops to multi-second compilations and evaluations; hundreds of CPU-GPU crossings | Sec. IV-A, Fig. 3, p. 4 |
| CORAL, research vs algorithmic task sets | Median host CPU vs GPU | Research: CPU 99%, GPU 44%; Algorithms: CPU 31%, GPU 55% | Sec. IV-A, Fig. 8, p. 5 |
| Trae (staged sequential coding) | Host CPU in sequential portions vs stage boundaries (parallel builds and tests) | 11% median, bursts to nearly 100% | Sec. IV-A, Fig. 5, p. 5 |
| Four frameworks, one server | Median host CPU / GPU SM activity / GPU power | CPU 6% (Owl) to 31% (CORAL); GPU SM below 55%; GPU power up to 50-70% of peak | Sec. IV-B, Fig. 13, p. 6 |
| Owl (heterogeneous, 8 GPUs, high concurrency) | Per-GPU power | 4 GPUs (busiest text and vision roles) repeatedly near peak; remaining GPUs near idle ~20% | Sec. IV-A, Fig. 7, p. 5 |
| Owl on GAIA | Tool time | Mostly waiting on network I/O, low baseline CPU demand; tool-agnostic retreat keeps only 37.7% harvester throughput vs 80% with tool-type hints | Sec. V-A-1, p. 8; Sec. V-B, p. 11 |
| All frameworks, serving plane (scheduler role) | Host CPU cores | 10.3 cores total across control processes; an 11-core pool keeps 98.5% of serving throughput; 16 or more cores add almost nothing | Sec. V-B, p. 10 |
| Runner (tool) role, high concurrency | Host CPU cores | 15.6 -> 11.5 cores after compaction; Owl (C=32) peak runner usage 70.1 -> 14.1 cores when the budget shrinks from 96 to 24 cores | Sec. V-B, Table IV, p. 11 |
| SWE-Agent, concurrency 1 -> 32 | Coordination overhead (involuntary context switches) | 71 -> 660 per second | Sec. IV-B, Fig. 15, p. 7 |
| SWE-Agent, CORAL, Owl (Trae split frontend/backend) | Host pipeline slots | 43-47% lost to backend stalls; IPC 1.2-1.6; L1-D MPKI 14-20 | Sec. IV-B, Fig. 14, p. 7 |

## Sharing and concurrency patterns

- Multiple agents may share the same model backend; independent subtasks run concurrently while dependencies and iterative feedback serialize others (Sec. II, p. 2).
- Homogeneous model composition (the default in most frameworks) lets one resident model be shared across agents with cross-agent batching and only per-agent context stored separately; heterogeneous composition keeps several models resident, raising GPU-memory pressure and fragmenting batches (Table I and Sec. III, p. 3).
- Consolidation: collocated agents running the same model share a single copy of the weights, freeing memory for a larger KV cache; with cross-agent prefix reuse the KV-cache hit rate rises by +68% (Owl, consolidation) and +75% (Owl, harvesting), +8% (CORAL, consolidation), -17% (CORAL, harvesting) (Table III, p. 9).
- Time-sharing: agents never active at the same time (sequential workflows) share a device without contending; Agora pairs a hot agent with a cold one (Sec. V-A-2, p. 9). Owl: 33% of GPUs freed with +82% throughput, +22% tasks/h, 2.5x lower tail latency, +51% energy efficiency; CORAL (all agents concurrently active): harvesting 50% of GPUs gives -71% throughput and 16x worse tail latency, so parallel workflows must not be harvested (Table III, p. 9).
- Prefetch from the DAG: in host-orchestrated workflows the orchestrator knows the next agent and Agora prefetches its state from host memory before it is needed (Sec. V-A-2, p. 9); the same knowledge yields burst and tool-type hints for CPU harvesting, while LLM-orchestrated workflows give no hints (Sec. V-A-1, p. 8).
- Task working-set locality: a task (one agent's assignment) runs a sequence of tools against the same files and repository state (edit, shell, test); pinning by task keeps that sequence on one core set, while pinning by tool type gives no benefit (Sec. V-B, p. 10-11). Best layout, 2 tasks per pool: tail tool time -12.8%, tool CPU-seconds -34%, batch completion +4.7% (Sec. V-B, Table V, p. 11); fully private cores are worse than the shared pool on batch time (-1.01%) (Table V, p. 11).
- Multiplexing penalty: many similar agents on shared cores evict one another's cache lines and branch history (Sec. IV-B, p. 7; Sec. V-C, p. 11); involuntary context switches grow 71 -> 660 per second for SWE-Agent from 1 to 32 tasks (Fig. 15, p. 7).
- Burst structure: parallel workflows create concurrent inference and tool bursts across agents; within one agent a stage fans out into multiple tool invocations (Table I, p. 3; Sec. IV-A, p. 4-5).
- Repeated calls per request: 580 LLM calls and 552 tool calls in one CORAL run (Fig. 3, p. 4); 3 tool-discovery operations and 3 tool executions across 2 tools in one fleet request (Fig. 2, p. 4).
- Tensor-parallel sharing: for CORAL, TP8 does not outperform the role-dedicated baseline while two TP4 instances do (Sec. V-A-2, p. 9).
- Related sharing work cited (pointers only): Tokendance collective KV-cache sharing [10], DeLM shared context [45], Pythia workflow predictability [75] (References, p. 12-13).

## Hardware constants (for reuse with source)

| Quantity | Value | Unit | Location | Conditions |
|---|---|---|---|---|
| Host CPU | AMD EPYC 7V12, 96 | cores | Sec. IV, p. 4 | controlled-study server |
| GPUs | 8 x NVIDIA A100 | devices | Sec. IV, p. 4 | one vLLM instance per agent role |
| NUMA node | 24 | cores | Sec. V-B, p. 10 | C=1 and C=8 on one node; C=32 on 48 cores across two nodes |
| Concurrency | 1-32 | tasks | Sec. IV, p. 4 | controlled study |
| Fleet trace | 24 | hours | Sec. IV, p. 4 | Azure production agentic requests |
| Example request duration | nearly 1 | minute | Sec. IV-A, Fig. 2, p. 4 | one production request |
| Tool-dominated requests | >27 | % of requests with tool time >= LLM time | Sec. IV-A, Fig. 4, p. 4 | fleet |
| CORAL run | 580 / 552 | LLM calls / tool invocations | Sec. IV-A, Fig. 3, p. 4 | one run, 4 agents |
| Tool latency range | milliseconds to multi-second | s | Sec. IV-A, p. 4 | file-system ops vs compilations/evaluations |
| Median host CPU utilization | 6-31 | % | Sec. IV-B, Fig. 13, p. 6 | Owl low end, CORAL high end |
| Median GPU SM activity | <55 | % | Sec. IV-B, Fig. 13, p. 6 | four frameworks |
| Median GPU power | up to 50-70 | % of peak | Sec. IV-B, Fig. 13, p. 6 | four frameworks |
| Idle-GPU power | ~20 | % of peak | Sec. IV-A, Fig. 7, p. 5 | Owl, lightly loaded roles |
| Trae host CPU | 11 median / nearly 100 peak | % | Sec. IV-A, Fig. 5, p. 5 | staged execution |
| CORAL CPU / GPU | 99 / 44 (research); 31 / 55 (algorithms) | % median | Sec. IV-A, Fig. 8, p. 5 | same framework, different task sets |
| Host IPC | 1.2-1.6 | instr/cycle | Sec. IV-B, Fig. 14, p. 7 | four frameworks |
| Backend-stall slots | 43-47 | % of pipeline slots | Sec. IV-B, Fig. 14, p. 7 | SWE-Agent, CORAL, Owl |
| L1-D MPKI | 14-20 | misses per kilo-instruction | Sec. IV-B, Fig. 14, p. 7 | four frameworks |
| Involuntary context switches | 71 -> 660 | per second | Sec. IV-B, Fig. 15, p. 7 | SWE-Agent, 1 -> 32 tasks |
| Scheduler-plane footprint | 10.3 | cores | Sec. V-B, p. 10 | across frameworks; 11 cores keep 98.5% throughput |
| Runner usage after compaction | 15.6 -> 11.5 | cores | Sec. V-B, p. 11 | high concurrency |
| Owl peak runner usage | 70.1 -> 14.1 | cores | Sec. V-B, p. 11 | C=32, budget 96 -> 24 cores; average runner usage -4.8% |
| CPU harvest yield | 95.0 (low load) / 53.7 (high load) / 74 (average) | % of standalone harvester throughput | Table II, p. 8; Sec. V-A-1, p. 8 | agent slowdown 2.8% / 3.9%; = 0.74 server-equivalents |
| Host CPU utilization gain | +31 | % | Sec. V-A-1, p. 8 | +3.3% agent latency (Sec. I rounds to 30% / <3%, p. 2) |
| Per-framework harvest (low / high load) | SWE-Agent 101.5 / 63.6; Trae 100.1 / 73.3; CORAL 94.7 / 1.0; Owl 83.6 / 76.9 | % | Table II, p. 8 | slowdown 2.5/1.2; 1.4/4.3; 1.0/9.9; 6.2/0.2 % |
| Owl tool-type hint effect | 37.7 -> 80 | % of standalone harvester throughput | Sec. V-A-1, p. 8 | 0.2% average agent slowdown with hints |
| GPU harvest, Owl | 33 freed; +82 throughput; +22 tasks/h; 2.5x tail; +51 energy eff.; +75 KV hit | % / x | Table III, p. 9 | high load, vs role-dedicated |
| GPU consolidation, Owl | +106 throughput; +36 tasks/h; 3.8x tail; +42 energy eff.; +68 KV hit | % / x | Table III, p. 9 | fixed GPU count |
| GPU consolidation, CORAL | +15 throughput; +28 tasks/h; 5.3x tail; +20 energy eff.; +8 KV hit | % / x | Table III, p. 9 | fixed GPU count |
| GPU harvest, CORAL | 50 freed; -71 throughput; -55 tasks/h; 16x worse tail; -53 energy eff.; -17 KV hit | % / x | Table III, p. 9 | parallel workflow, not worth harvesting |
| Tool CPU-seconds saved by compaction | 25-46 | % | Table IV, p. 11 | C=8: 40.1 / 44.2 / 46.0; C=32: 25.2 / 29.7 / 33.7 (Shared / Split / Isolate) |
| Serving CPU change under compaction | <2 | % | Sec. V-B, Table IV, p. 11 | batch completion changes under 1% |
| Best pinning, 2 tasks/pool | 4.73 batch; 3.30 / 11.58 full lane; 3.24 / 12.81 tool sum | % faster (median / max) | Table V, p. 11 | C=32, vs wide shared runner pool |

## Implications for a symbolic accelerator

- Upper bound on the non-LLM opportunity: tool time equals or exceeds LLM time for more than 27% of fleet requests (Fig. 4, p. 4) and one CORAL run has almost 1:1 LLM-to-tool calls (580 vs 552, Fig. 3, p. 4). Non-LLM time is a first-class latency term, but the paper never says how much of it is symbolic; SymFabric must produce that breakdown itself.
- Symbolic and plausibly offloadable: search & retrieval and data/query tool categories (Fig. 11, p. 5-6); tool discovery ("Tool Search", Fig. 2, p. 4); orchestrator dependency tracking, for which the authors themselves ask for "dedicated hardware or runtime support for agent scheduling and context management" (Sec. IV-B, p. 7; Sec. V-C, p. 11). These are INDEX / FILTER / COMBINE / SELECT over relations and a DAG. The "eval" tools in CORAL and the test runs in Trae and SWE-Agent are verification-flavored, but the paper describes them as compilations and test executions (Sec. IV-A, p. 4-5), i.e. general-purpose CPU work; only symbolic sub-steps inside them (constraint checks, candidate search) would map.
- Not offloadable: I/O-bound tools. Owl on GAIA spends most tool time waiting on network I/O (Sec. V-B, p. 11); GitHub/DevOps, communication/productivity and business/workflow categories (Fig. 11, p. 6) and disk-heavy roles (Fig. 9, p. 6) are network or storage bound. Reuse the paper's compute-bound vs I/O-bound tool-type hint (Sec. V-A-1, p. 8) as the offload classifier.
- Code execution is the largest tool category (Fig. 11, p. 5-6) and compute-bound, but it is compiler/interpreter work; SymFabric helps only when the executed program is itself a solver, planner or verifier.
- Boundary-crossing cost: the paper's core finding is that hundreds of CPU-GPU crossings per run (Sec. IV-A, p. 4) put the host on the critical path. A symbolic accelerator is a third device; offload is worthwhile for multi-second tools (compilations, evaluations) or if the accelerator also runs the dependency loop, while millisecond file-op-class tools should stay on the CPU.
- Pooled, not per-agent: tool demand is bursty (Trae 11% median to nearly 100%, Fig. 5, p. 5) and role-dedicated devices strand capacity (Owl idle GPUs at ~20% of peak power, Fig. 7, p. 5). A per-agent symbolic unit would idle the same way; SymFabric should be a shared multi-tenant resource that batches Q concurrent agents' queries, mirroring the consolidation that gave Owl +82% throughput on 33% fewer GPUs (Table III, p. 9). Sequential workflows time-share well, parallel ones do not (Table III, CORAL, p. 9).
- Sharing model: homogeneous agents share weights and prompt prefixes (KV hit +68% to +75%, Table III, p. 9). The SymFabric analog is many agents sharing one knowledge base, index or rule set: keep shared relations resident once and only per-agent state (frontier, bindings) separately, the same weights-vs-context split the paper exploits. Host-orchestrated workflows expose the DAG for prefetching (Sec. V-A-2, p. 9); LLM-orchestrated ones do not (Table I, p. 3).
- Locality: pin by task (same files/repo) rather than by tool type (Sec. V-B, p. 10-11), so schedule symbolic queries by task working set, not by operator type, and keep a task's relation slices resident across its tool sequence.
- Memory-system signal: host cores are backend-bound (43-47% stall slots, IPC 1.2-1.6, L1-D MPKI 14-20, Fig. 14, p. 7; fleet MPKI high at the deeper cache levels, Fig. 10, p. 6). Host-side tool and orchestration work is memory-latency bound, consistent with pointer-chasing symbolic workloads; this supports giving a symbolic accelerator its own memory path, but the paper cannot settle whether PIM is warranted.
