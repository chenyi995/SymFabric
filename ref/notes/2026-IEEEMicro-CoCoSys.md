# CoCoSys: The Co-Design of Cognitive AI from Algorithms to Systems (IEEE Micro 2026)

Z. Wan, Y. Cao, S. K. Gupta, L. Heck, T. Krishna, Y. Lin, A. Naeemi, B. Olshausen, P. Panda, J. Rabaey, V. Raghunathan, P. Raina, T. Rosing, K. Roy, J.-S. Seo, N. Shanbhag, J. Tenenbaum, A. Raghunathan, A. Raychowdhury. IEEE Micro Special Issue on JUMP 2.0, Nov/Dec 2026, DOI 10.1109/MM.2026.3734155. Overview article of the SRC/DARPA JUMP 2.0 CoCoSys center; 14 pp. in the author's accepted version (page footers). The article has no numbered sections, so locations below are given as section title + page + figure.

Text-extraction caveat: the pdftotext dump interleaves both columns and, on pp. 4-6, embeds text of an older proposal document that sits inside the images of Figs. 2-4 (e.g. "Theme 1 ...", "initial team of 10 PIs", "target >10X reduction"). That embedded text is NOT the article body and is not cited here. The bar labels of Fig. 5(a) are only partly legible; the bar-to-benchmark grouping used below is inferred from bar order and is marked "inferred".

- **Workload**: Not a single workload; a center portfolio. (i) Neuro-symbolic AI (NSAI) is the end-to-end case study: six LLM/DNN + symbolic pipelines, AlphaGeometry, R2-Guard, GeLaTo, Ctrl-G, NeuroPC, LINC (Fig. 5 top, p. 10), plus seven "classical" NSAI models profiled earlier in [5] (Thrust II, p. 6; not enumerated in this article). (ii) Vector-symbolic / hyperdimensional (HD) computing: Residue HD, LifeHD, CogSys codebooks (Thrust I, p. 5; Case Study, p. 9). (iii) Probabilistic logical reasoning: first-order logic (FOL), SAT, probabilistic circuits (PCs), HMMs (REASON, Case Study, p. 9). (iv) Combinatorial optimization via Ising machines (Neuro-Ising, p. 5; TAXI, p. 8). (v) Neural side for contrast: LLM inference, SSM/Mamba, GNN, spiking nets, KANs on NPUs and CIM (Thrust II-III, pp. 6-7). (vi) Multi-agent embodied planning and federated HD learning (Thrust IV, p. 8). NSAI systems "integrate neural perception, symbolic reasoning over vector-symbolic, first-order-logic, and satisfiability representations, and probabilistic inference, composing them as pipelines and nested loops" (Case Study, p. 9).
- **Operators**: The article gives only a symbolic-vs-neural stage split (Fig. 5(a), p. 10), no per-operator breakdown. Hotspot kernels it names, with a SymFabric mapping:
  - Dense GEMM and circular convolution (VSA binding), the CogSys nsPE modes (Case Study, p. 9; Fig. 6 top, p. 11): COMBINE(x) + REDUCE(+) over the (+,x) semiring; circular convolution is the same with a cyclic INDEX on one operand. Bubble Streaming streams that cyclic shift with O(d) instead of O(d^2) on-chip memory (p. 9).
  - VSA codebook factorization / cleanup (resonator networks, Residue HD; Thrust I, p. 5; p. 9): INDEX (codebook lookup) -> COMBINE -> REDUCE (similarity) -> SELECT (argmax), iterated to a fixed point; bundling = REDUCE(+ or majority); permutation = INDEX.
  - PCs and HMMs unified as DAGs and pruned into binary trees (REASON, p. 9; "SpMSpM/DAG/DPLL Reduction", Fig. 6 middle, p. 11): tree REDUCE over (+,x) for marginals or (max,+) for MAP; HMM forward = SpMV over (+,x). REASON's node shows a x/< unit feeding a + unit (Fig. 6 middle, p. 11), i.e. a two-operator semiring cell.
  - FOL / SAT via DPLL with watched literals, BCP FIFO, implication broadcast (Fig. 6 middle, p. 11): FILTER (clause status under partial assignment) + UPDATE (implications) + SELECT (branch literal) on the Boolean semiring, iterated to a fixpoint inside a backtracking search; data-dependent control flow.
  - Sparse x sparse matrix multiply (REASON sparse-matrix mode, p. 9): INDEX (nonzero intersection) + COMBINE + REDUCE.
  - GNN gather-scatter (GraNNite, Thrust II, p. 7): INDEX + REDUCE (SpMM).
  - Ising annealing for TSP (Neuro-Ising, p. 5; TAXI, p. 8): stochastic UPDATE of spins from a local field (SpMV over (+,x)) plus a random SELECT; not a native SymFabric operator.
  - Symbolic planning (LLM -> formal task definition -> symbolic planner; Thrust IV, p. 8): search; no kernel detail given.
  Iteration structure: NSAI = "pipelines and nested loops" of neural and symbolic stages (p. 9); the symbolic stages are parameter-light yet latency-dominant, i.e. Amdahl-limited (Thrust II, p. 6).
- **Data scale**: CogSys VSA codebook 13,560 KB factorized to 190 KB, 71.4x (Case Study, p. 9). The NSAI SoC holds "the full working set of all target NSAI workloads on chip" in 5.625 MB RRAM + 0.85 MB SRAM (p. 9; Fig. 6 caption, p. 11), so the targeted NSAI working sets are below about 6.5 MB (sum of the two figures, p. 9). TAXI solves TSP instances of nearly 86,000 cities (Thrust III, p. 8). Compositional pipelines use LLaMA-3-8B as the neural stage (Fig. 5(d), p. 10), profiled on H100/H200 (Thrust II, p. 6). Concurrent queries Q: not discussed; "multi-tenant cognitive system design" appears only as a Thrust II task name (Fig. 3, p. 5). Sparsity: N:M structured sparsity for LLM CIM and CPU tiles (Thrust II-III, p. 7); zero-weight skipping in the OS-CIM macro (ref. 14 title, p. 13); no sparsity statistics for symbolic data (clauses, DAGs, codebooks) are given.
- **Hardware mechanism**: (1) CogSys: reconfigurable neuro-symbolic PE (nsPE) switching among load / dense GEMM / circular-convolution modes with a Bubble Streaming dataflow; 75.96x speedup over a TPU and up to 730x energy efficiency over an RTX GPU in 4 mm^2 at 1.48 W (Case Study, p. 9; Fig. 6 top, p. 11). (2) REASON: tree-structured PEs with probabilistic-inference / logic-evaluation / sparse-matrix modes, Benes N:N distribution crossbar over N SRAM banks, symbolic memory support (BCP FIFO, watched literals, implication broadcast), packaged as a GPU plug-in sharing the L2; 12-50x over a CPU, 310-681x energy efficiency over an RTX GPU on ten benchmarks (p. 9; Fig. 6 middle, p. 11). (3) NSAI SoC: 40 nm ULP CMOS + BEOL RRAM, ten 576-KB RRAM neural tiles + two SRAM symbolic tiles on a shared NoC, unified 128-bit VLIW interface, 10.8 TOPS/W, down to 0.73 V (p. 9; Fig. 6 bottom, p. 11). Memory-centric motifs: output-stationary SRAM CIM whose outer-product dataflow removes partial-sum traffic (Thrust III, p. 7); ADC-less analog CIM (HCiM, pp. 7-8); digital CIM with N:M sparsity for LLMs (p. 7); SOT-MRAM in-memory Ising crossbars with "no off-macro data movement" (TAXI, p. 8); RRAM for capacity-bound neural working sets, SRAM for latency-critical symbolic datapaths (pp. 8-9). Design rule: "programmable architectures that serve neural, symbolic, and probabilistic primitives within a single substrate, rather than bolting accelerators for each onto a GEMM-centric datapath" (Thrust II, p. 7).
- **Simulation / implementation method**: Mixed per component; the article reports numbers, not methods. Profiling on CPU+GPU systems, server A6000 and edge Orin (Fig. 5(a)-(c), p. 10), H100/H200 for compositional pipelines (Thrust II, p. 6). Simulation frameworks: SCALE-Sim v3 cycle-accurate systolic simulator [13], MOSAIC NPU design-space exploration [7], Voyager accelerator generation + system-level evaluation [15] (pp. 6-7). Silicon: NSAI SoC in 40 nm with TSMC (p. 9; Fig. 2, p. 4), 28 nm OS-CIM macro (p. 7), 22 nm MRAM CIM macro (Thrust I, p. 6). For CogSys (4 mm^2, 1.48 W, p. 9) and REASON the implementation level (RTL vs. simulation) and node are not stated here; see [24], [25].
- **Limitations**: Survey-style: no methodology, baselines' configurations, or error bars; results are lifted from about 25 constituent papers (reference list, pp. 13-14). Only a two-way symbolic/neural stage split is reported (Fig. 5(a), p. 10); no per-operator, per-data-structure, or bandwidth (GB/s) numbers anywhere. Fig. 5(b)-(c) latencies are legible only as minute-scale axes (0-12 min and 0-24 min gridlines, p. 10). The taxonomy is LLM + logic/PC/HMM pipelines plus VSA and Ising; Datalog, relational/knowledge-graph reasoning, weighted relations, and multi-query serving are absent. "Reconfigurability" is mode switching (three modes per PE in CogSys and REASON, p. 9), not a general operator algebra. Speedup baselines (TPU, RTX GPU, CPU) are unspecified models. The paper itself warns that macro-level CIM TOPS/W does not transfer to systems (Thrust III, p. 7).
- **Numbers worth reusing**: see the constants table below. Most useful: symbolic-stage share of end-to-end runtime 34.0-67.7% across the 14 bars of Fig. 5(a) (p. 10); software-only compositional-pipeline latency cut of 50-70% (Thrust II, p. 7; Case Study, p. 9); RRAM access energy 0.247 pJ/bit and density 4.80 Mb/mm^2 at 40 nm (p. 9); NSAI SoC 10.8 TOPS/W at 0.73 V (p. 9); CogSys 75.96x / 730x in 4 mm^2 at 1.48 W (p. 9); REASON 12-50x / 310-681x (p. 9); codebook 71.4x compression (p. 9).
- **Implications for SymFabric**:
  - External validation of the premise. A 19-author multi-institution center (author list, p. 1) concludes that symbolic/probabilistic stages dominate NSAI latency (Fig. 5(a), p. 10), are memory-bound while neural stages are compute-bound (Fig. 5(d), p. 10), and that cognitive AI needs "programmable architectures that serve neural, symbolic, and probabilistic primitives within a single substrate" (p. 7). Cite this as the motivating characterization for SymFabric's generality claim.
  - Generality gap to exploit. CogSys and REASON reach breadth via mode-switching PEs (GEMM / circular conv / tree-reduce / DPLL / SpMSpM). Every one of these is COMBINE+REDUCE over (+,x), (max,+) or Boolean semirings with a different INDEX pattern; SymFabric can present the semiring algebra as the principled unification and should demonstrate coverage of all six Fig. 5 workloads plus VSA and Ising.
  - PIM contingency will likely trigger. The paper names "memory capacity, bandwidth, and access energy" as the binding constraints (Thrust III, p. 7; Lessons, p. 12). Its RRAM/SRAM split (capacity-bound neural in RRAM, latency-critical symbolic in SRAM, pp. 8-9) is a concrete precedent; reuse the 0.247 pJ/bit figure (p. 9) and the "hundreds -> single-digit TOPS/W" macro-vs-system warning (p. 7) when arguing for or against PIM.
  - Baseline to beat. Software-level optimizations alone (speculative symbolic execution, operator fusion, cache management) already cut compositional-pipeline latency 50-70% (p. 7); SymFabric's evaluation must either compose with or exceed this.
  - Uncovered workloads are SymFabric's opening: Datalog / knowledge-graph / weighted-relation reasoning and multi-query (Q > 1) serving are not addressed anywhere in the article.
  - Reusable design points: Bubble Streaming's O(d) buffer for circular convolution on a systolic array (p. 9); REASON's Benes crossbar + N SRAM banks + x/< and + node (Fig. 6 middle, p. 11); "reconfigurable datapaths amortize their switching overhead and rival workload-specific designs" (Lessons, p. 12) supports a reconfigurable systolic array over fixed-function units.
  - Benchmarks: adopt the six Fig. 5 workloads (p. 10) and REASON's ten benchmarks (p. 9); the center plans "an open benchmark suite that jointly scores accuracy, reasoning quality, robustness, and efficiency" (Lessons, p. 12).
  - Differentiation angle: the open challenge "gradient flow through discrete symbolic operators" (Open challenges, p. 12) is natural for a semiring formulation (differentiable semirings), which CoCoSys's mode-switching hardware does not offer.
- **Open source**: None stated in the article. SCALE-Sim v3 [13], MOSAIC [7] and Voyager [15] are described as simulation/generation frameworks but no repository URLs are given.

## Workload taxonomy in the paper

| Category | Example workloads / models | Symbolic kernel (mathematical operation) | Data structure | Stated bottleneck or runtime fraction (location) |
|---|---|---|---|---|
| Math theorem proving and reasoning | AlphaGeometry: LLM + FOL, SAT solver, acyclic graph (Fig. 5 top, p. 10); benchmarks IMO, MiniF2F (Fig. 5(a)) | Deductive closure over FOL rules; SAT (Boolean semiring); DAG traversal | Clause database, proof DAG | Symbolic share 67.4% and 60.2% of runtime (bars 1-2, Fig. 5(a), p. 10; inferred grouping) |
| LLM safety / unsafety detection | R2-Guard: LLM + FOL, probabilistic circuit, HMM (Fig. 5 top, p. 10) | Probabilistic-logic inference: sum-product over PC DAG; HMM forward | PC DAG, HMM transition matrix | Symbolic share 57.9-66.8% (bars 3-5, Fig. 5(a), p. 10; inferred) |
| Constrained text generation | GeLaTo: LLM + FOL, SAT solver, HMM (Fig. 5 top, p. 10) | Constraint checking (SAT) + HMM-guided token probabilities | HMM, constraint formulas | Symbolic share 34.9-38.4% (bars 6-8, Fig. 5(a), p. 10; inferred) |
| Interactive text editing / infilling | Ctrl-G: LLM + HMM, probabilistic circuits (Fig. 5 top, p. 10) | HMM / PC marginals conditioning decoding, (+,x) semiring | HMM, PC DAG | Symbolic share 60.1-67.7% (bars 9-11, Fig. 5(a), p. 10; inferred) |
| Classification with logical constraints | NeuroPC: DNN + FOL, probabilistic circuit (Fig. 5 top, p. 10) | PC evaluation of constraint satisfaction | PC DAG | Symbolic share 50.5% (bar 12, AwA2, Fig. 5(a), p. 10; inferred) |
| Logical / deductive reasoning | LINC: LLM + FOL solver (Fig. 5 top, p. 10) | Theorem proving with an FOL solver | Logic program | Symbolic share 34.0-35.7% (bars 13-14, FOLIO / Proof..., Fig. 5(a), p. 10; inferred) |
| All six NSAI pipelines above | -- | -- | -- | Symbolic and probabilistic stages dominate CPU+GPU runtime (Fig. 5(a)); real-time targets missed as task scale grows, minute-scale latency (Fig. 5(b)); trend holds on A6000 and Orin (Fig. 5(c)); symbolic kernels memory-bound, neural LLaMA-3-8B compute-bound (Fig. 5(d)); all p. 10 |
| Vector-symbolic / HD computing | CogSys VSA codebooks (p. 9); Residue HD, LifeHD (Thrust I, p. 5) | Binding = circular convolution over (+,x); bundling = sum / majority; permutation; codebook factorization by resonator iterations; similarity search | Dense d-dim hypervectors; codebook 13,560 KB -> 190 KB (p. 9) | Codebook memory and O(d^2) convolution buffer -> O(d) (p. 9); encoding latency cut 4.1x (p. 9) |
| Probabilistic logical reasoning (unified) | REASON: FOL, SAT, PCs, HMMs under one DAG representation (p. 9) | Tree reduction (sum-product / max-product); DPLL + BCP; SpMSpM | Binary trees (pruned DAG), watched-literal lists, BCP FIFO, sparse matrices (Fig. 6 middle, p. 11) | HMMs compressed 52.6x with <1% accuracy loss (p. 9); 12-50x over CPU (p. 9) |
| Combinatorial optimization | TSP via Ising: Neuro-Ising (GNN guiding local Ising solvers, p. 5); TAXI SOT-MRAM crossbars (p. 8) | Energy minimization by stochastic spin updates; local field = SpMV | Crossbar-resident coupling matrix, hierarchically clustered subproblems | Nearly 86,000 cities, 8x faster than clustering-based Ising solvers, "no off-macro data movement" (p. 8) |
| Graph neural networks | GraNNite on edge NPUs (Thrust II, p. 7) | Gather-scatter aggregation (SpMM) | Graph adjacency | "irregular gather-scatter patterns" (p. 7) |
| Neural (for contrast) | LLM (LLaMA-3-8B), SSM/Mamba, KAN, spiking, CNN (pp. 6-7, p. 10) | Dense GEMM; recurrent scan; autoregressive decode | Weights, KV cache | Compute-bound (Fig. 5(d), p. 10); LLM decode memory-bound (p. 7); accelerators "severely underutilized on non-GEMM primitives" (p. 6) |
| Embodied multi-agent planning | ReCA; LLM translating goals to a formal task definition solved by a symbolic planner (Thrust IV, p. 8) | Hierarchical planning / search | Task definitions, dual-memory organization | "tens of minutes on desktop GPUs"; sequential LLM calls, poor scaling with agent count, brittle low-level execution (p. 8) |
| Federated / lifelong HD learning | MultimodalHD, LifeHD (p. 5, p. 8) | HD encode + bundle + similarity | Hypervectors | "bottlenecked by communication" (p. 8) |

## Hardware / system components described

- CogSys [24]: VSA-centric NSAI accelerator; nsPE with load / GEMM / circular-convolution modes, Bubble Streaming dataflow, codebook factorization; 75.96x vs TPU, up to 730x energy vs RTX GPU, 4 mm^2, 1.48 W (Case Study, p. 9; Fig. 6 top, p. 11).
- REASON [25]: probabilistic-logic accelerator as a GPU plug-in; tree PEs with three modes, Benes N:N crossbar, N SRAM banks, BCP FIFO / watched literals; 12-50x vs CPU, 310-681x energy vs RTX GPU (p. 9; Fig. 6 middle, p. 11).
- NSAI heterogeneous SoC [20]: 40 nm ULP CMOS + BEOL RRAM; ten 576-KB RRAM neural tiles + two SRAM symbolic tiles on a shared NoC; 128-bit VLIW; 10.8 TOPS/W at 0.73 V (p. 9; Fig. 6 bottom, p. 11; Fig. 2 bottom, p. 4).
- MOSAIC [7]: heterogeneous NPU design-space exploration with big / little / specialized tiles for SSMs, FFT-style long convolutions, KANs, spiking nets (Thrust II, p. 6).
- GraNNite [8], XAMBA [9], LLM-NPU [10]: mapping GNNs, SSM/Mamba scans, and memory-bound LLM decode onto commercial edge NPUs (Thrust II, p. 7).
- VEGETA [11]: flexible N:M sparse-tile extensions for CPU matrix engines, with Intel (p. 7).
- LaCache [12]: ladder-shaped KV-cache compression for long-context LLMs (p. 7).
- Compositional-pipeline system optimizations [6]: speculative symbolic execution, cross-component operator fusion, coordinated cache management; 50-70% latency cut (p. 7; p. 9).
- SCALE-Sim v3 [13]: cycle-accurate systolic-array simulator across dataflows, mappings, memory hierarchies; plus unnamed chiplet-synthesis and multi-stage distributed-inference models (p. 7).
- Output-stationary SRAM CIM macro [14]: 28 nm, 20.9-137.2 TOPS/W, structured and unstructured sparsity (Thrust III, p. 7).
- Voyager [15]: end-to-end accelerator generation and system-level CIM evaluation (p. 7).
- Digital CIM accelerator with flexible N:M sparsity for LLM inference [16] (p. 7).
- HCiM [17]: ADC-less hybrid analog-digital CIM, up to 28% macro energy reduction (pp. 7-8).
- ClipFormer [18]: memristive-crossbar write-noise mitigation for transformers (p. 8).
- TAXI [19]: SOT-MRAM crossbar in-memory Ising machine with hierarchical clustering (p. 8).
- 22 nm MRAM CIM macro with statistical error compensation [4] (Thrust I, p. 6).
- ReCA [21], MulBERRY [22], MultimodalHD [23]: multi-agent embodied, UAV-swarm, and federated-HD systems (Thrust IV, p. 8).
- Memory-side systems: no DRAM-side PIM/PNM is described; "memory-centric" here means CIM macros, on-chip RRAM capacity, and 3D stacking / chiplets with sister JUMP 2.0 centers (p. 8; p. 12).

## Hardware constants (for reuse with source)

| Quantity | Value | Unit | Location | Conditions |
|---|---|---|---|---|
| RRAM access energy | 0.247 | pJ/bit | Case Study, p. 9; Fig. 6 caption, p. 11 | 40 nm ULP CMOS, BEOL RRAM, custom edge-triggered prolonged sensing circuit |
| RRAM macro density | 4.80 | Mb/mm^2 | p. 9 | same chip; 1.67x denser than prior art at the node |
| RRAM vs SRAM cell density | 2.9 | x | p. 9 | reason neural tiles use RRAM |
| NSAI SoC energy efficiency | 10.8 | TOPS/W | p. 9; Fig. 2 bottom, p. 4 | INT8; at 0.73 V (Fig. 2) |
| NSAI SoC core voltage | 0.73-1.1 | V | Fig. 6 caption, p. 11 | operates down to 0.73 V (p. 9) |
| NSAI SoC die | 5.25 x 5.25 | mm^2 | Fig. 6 caption, p. 11 | 40 nm |
| NSAI SoC on-chip memory | 5.625 / 0.85 | MB RRAM / MB SRAM | Fig. 6 caption, p. 11 | ten 576-KB RRAM tiles + two SRAM tiles |
| Idle-tile power gating | 6.47 | x avg power reduction | p. 9 | software-defined power manager |
| VLIW interface width | 128 | bit | p. 9 | serves full NSAI paradigm space without re-fabrication |
| CogSys area / power | 4 / 1.48 | mm^2 / W | p. 9 | node not stated in article |
| CogSys speedup vs TPU | 75.96 | x | p. 9 | TPU model not stated |
| CogSys energy efficiency vs RTX GPU | up to 730 | x | p. 9 | GPU model not stated |
| CogSys codebook memory | 13,560 -> 190 | KB | p. 9 | 71.4x; encoding latency 4.1x lower |
| Bubble Streaming buffer | O(d) vs O(d^2) | on-chip memory | p. 9 | circular convolution pipelining |
| REASON speedup vs CPU | 12-50 | x | p. 9 | ten benchmarks |
| REASON energy efficiency vs RTX GPU | 310-681 | x | p. 9 | ten benchmarks |
| REASON HMM compression | 52.6 | x | p. 9 | under 1% accuracy loss |
| Symbolic share of NSAI runtime | 34.0-67.7 | % | Fig. 5(a), p. 10 | CPU+GPU system; 14 bars; labels partly garbled |
| Fig. 5(b),(c) latency axes | 0-12 and 0-24 | min | Fig. 5(b),(c), p. 10 | Small vs Large tasks; A6000 vs Orin |
| OS-CIM macro efficiency | 20.9-137.2 | TOPS/W | Thrust III, p. 7 | 28 nm SRAM CIM prototype [14] |
| Macro-to-system CIM gap | hundreds -> single-digit | TOPS/W | p. 7 | after ADC, data movement, underutilization |
| HCiM energy reduction | up to 28 | % | pp. 7-8 | vs 7-bit-ADC baseline, iso-accuracy |
| ClipFormer accuracy gain | 10-40 | % non-ideal accuracy | p. 8 | high-noise regimes, no retraining |
| MRAM CIM compute SNR | +2.7 to 6 | dB | Thrust I, p. 6 | 22 nm; ResNet-20 74.8% -> 82.0%; 0.5% overhead |
| TAXI scale / speed | ~86,000 / 8 | cities / x | p. 8 | vs clustering-based Ising solvers |
| MOSAIC savings | 46.9 / 1.5-2.4 | % iso-area energy / x vs NVDLA | Thrust II, p. 6 | 20-workload suite |
| XAMBA speedup | up to 4.8 | x | p. 7 | SSM/Mamba on a commercial AI PC NPU |
| Compositional pipeline latency cut | 50-70 | % | p. 7; p. 9 | software only, accuracy unchanged |
| ReCA | 10.2 / +4.3 | x speedup / % mission success | p. 8 | cooperative embodied agents |
| MulBERRY | 19 / 22 / 4.16 | % flight energy / % missions / x processing energy | p. 8 | UAV swarms, low-voltage operation |
| LifeHD | up to 74.8 / 34.3 | % clustering accuracy gain / x energy | Thrust I, p. 5 | vs neural lifelong-learning baselines |
| Profiling scope | 7 / 6 | models / workloads | Thrust II, p. 6 | classical NSAI [5]; LLM-integrated on H100/H200 [6] |
| Grand-challenge targets | sub-ms / >100 / >10 | latency / x energy / x data+compute | Vision, p. 3; Fig. 4, p. 6 | center-wide goals, not results |

## Claims about bottlenecks

- "they traverse sparse graphs, chase pointers, search combinatorial spaces, and bind high-dimensional vectors, producing irregular memory access, divergent control flow, and high memory intensity that map poorly onto current hardware" (Intro, p. 2).
- Gap "characterized by heterogeneous compute kernels, sparse and irregular access patterns, and high memory intensity" (Fig. 1 caption, p. 3).
- Four recurring bottlenecks: "symbolic and probabilistic operations dominate end-to-end latency; neural and symbolic kernels sit at opposite ends of the roofline, the former compute-bound and the latter memory-bound; general-purpose and ASIC accelerators are severely underutilized on non-GEMM primitives; and in compositional pipelines, parameter-light symbolic components create outsized Amdahl bottlenecks" (Thrust II, p. 6).
- "models of equal accuracy can differ several-fold in their compute cost per unit of intelligence" (Thrust II, p. 6).
- "Thrust II identifies memory capacity, bandwidth, and access energy as the binding constraints" (Thrust III, p. 7).
- "a macro reporting hundreds of TOPS/W can deliver only single-digit TOPS/W in a full workload" (Thrust III, p. 7).
- "analog-to-digital converters dominate ACIM area and energy" (Thrust III, p. 7).
- "For models whose working sets exceed on-chip SRAM, dense non-volatile memory eliminates the DRAM traffic that would otherwise dominate power" (Thrust III, p. 8).
- "a single task can take tens of minutes on desktop GPUs, limited by three bottlenecks, repeated sequential LLM calls for planning and communication, poor scaling as agents are added, and brittle low-level execution" (Thrust IV, p. 8).
- "On-device learning across many agents is bottlenecked by communication" (Thrust IV, p. 8).
- "(a) Symbolic and probabilistic stages dominate runtime on a CPU+GPU system. (b) Real-time targets are missed as task scale grows. (c) The trend holds across server (A6000) and edge (Orin) platforms. (d) Roofline analysis shows symbolic and probabilistic kernels are memory-bound while neural kernels are compute-bound" (Fig. 5 caption, p. 10).
- "Across cognitive workloads, memory capacity, bandwidth, and access energy dominate the efficiency budget" (Lessons, p. 12).
- "reconfigurable datapaths amortize their switching overhead and rival workload-specific designs" (Lessons, p. 12).
- "the compositional-pipeline work localizes the system bottlenecks but does not yet close the hardware gap" (Open challenges, p. 12).
