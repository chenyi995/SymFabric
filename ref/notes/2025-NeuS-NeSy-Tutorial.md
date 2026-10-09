# Efficient Processing of Neuro-Symbolic AI: A Tutorial and Cross-Layer Co-Design Case Study (NeuS 2025)

Wan, Liu, Yang, Raj, Raychowdhury, Krishna (Georgia Tech), PMLR 288:1-15 (page header). Tutorial that consolidates
the group's ISPASS'24 workload characterization (Wan et al., 2024c; Sec. 3), the HPCA'25 CogSys microarchitecture
(Wan et al., 2025b; Sec. 4.2), the DAC'25 NSFlow FPGA framework (Yang et al., 2025; Sec. 4.3) and a TSMC 40nm SoC
tape-out (Sec. 4.4). Read from a pdftotext extraction:
Fig. 3a, Fig. 8c and Fig. 11 labels are partly garbled (caveats marked inline); Fig. 4a was recovered and checked.

- **Workload**:
  - Five NeSy paradigms (Sec. 2.1, Fig. 1): Symbolic[Neuro] (AlphaGo/AlphaZero: MCTS + NN state evaluation);
    Neuro|Symbolic (NVSA, PrAE, VSAIT, NeuPSL, DeepProbLog, NeurASP, NSCL, NSVQA); Neuro:Symbolic->Neuro (LNN,
    deep learning for symbolic math, differentiable ILP); Neuro_Symbolic (LTN, DON); Neuro[Symbolic] (GNN with
    attention over rules, NLM, ZeroC).
  - Kernel taxonomy (Fig. 2), symbolic side: Logical Reasoning (FOL, deductive database, SAT solver, VSA); Symbolic
    Search (graph search, PDDL planning, A*); Knowledge Rep. (concept graphs, KG reasoning); Algebraic Methods
    (automata, graph algebra, symbolic algebra); Symbolic Physics (neural DEs, PINNs, differentiable physics engines);
    Probabilistic Reasoning (probabilistic circuits, MCMC, HMM, PGM). Neural side: encoders, perceivers, vision and
    language models.
  - Actually profiled: 7 workloads, LNN, LTN, NVSA, NLM, VSAIT, ZeroC, PrAE (Sec. 3.1), on Intel Xeon Silver 4114 +
    RTX 2080 Ti, Xavier NX, Jetson TX2 (Sec. 3.1). Hardware case studies target only the VSA-based class, i.e.
    NVSA-style RAVEN/RPM abductive reasoning (Sec. 4.1).
- **Operators**:
  - Hotspot: vector-symbolic circular convolution (bind) and circular correlation (unbind) plus vector-vector
    multiplications, about 80% of symbolic runtime in VSA-based NeSy (Sec. 3.5, Fig. 6). Definition
    C[n] = sum_{k=0}^{N-1} A[k] * B[(n-k) mod N] (Sec. 4.2).
  - SymFabric mapping of bind: INDEX (structured rotation (n-k) mod N, never materialized) -> COMBINE (x) -> REDUCE (+)
    over k, (+,x) semiring on reals; unbind = same with A reversed (Sec. 4.2). GPUs pay for the INDEX as explicit
    index arithmetic (Sec. 4.2); a TPU-like array pays by materializing the circulant matrix, O(d^2) (Sec. 4.2).
    The bubble-streaming dataflow is a systolic realization of the rotation INDEX by a one-cycle passing register
    (Sec. 4.2, Fig. 8b).
  - Similarity / cleanup: COMBINE (x or bit-wise) + REDUCE (+ / popcount) + SELECT (argmax over codebook). The SoC
    symbolic tile provides "bit-wise ops, CA-90 units, distance computation, and multi-precision MACs" (Sec. 4.4);
    the argmax/cleanup step itself is not named in the paper (marked + below).
  - Logic workloads (LNN, LTN, NLM): "frequent logical operations" (Sec. 3.3); vector/element-wise ops dominate the
    symbolic part (Fig. 4a). Real-valued logic = COMBINE with t-norm/t-conorm style min/max/product, REDUCE over
    conjuncts (+; the paper names neither t-norms nor semirings). Data movement + transform is a large share of LNN
    symbolic time (39.4% + 17.3%, Fig. 4a) -> INDEX/FILTER/UPDATE traffic, not arithmetic.
  - Neural parts: GEMM/Conv; nsPE runs them in GEMM mode (Sec. 4.2, Fig. 7).
  - Iteration structure: sequential neural -> symbolic phases; symbolic lies on the critical path for NVSA, VSAIT,
    PrAE, while LNN, LTN, NLM, ZeroC compile symbolic knowledge into the neural structure (Sec. 3.4, Fig. 5); the
    symbolic phase repeats per RAVEN attribute (Type, Size, Color, Number, Position) (Fig. 6); symbolic runtime
    grows "quadratically" with RPM task size (Sec. 3.2, Fig. 3c).
- **Data scale**:
  - Hypervector dimension d = 2048 and d = 20480; batches of 1 and 1000 circular convolutions (Fig. 8c legend).
  - Circulant footprint O(d^2) when bind is executed as GEMV on a TPU-like array (Sec. 4.2).
  - RAVEN/RPM task sizes from 2x2 to 3x3 (Sec. 3.2, Fig. 3c).
  - Concurrent queries Q: not quantified. adSCH interleaves "symbolic operations of other tasks" during the current
    task's neural layers and partitions the array column-wise across "parallel symbolic tasks" (Sec. 4.2, Fig. 9);
    "parallel rule query" is listed as an opportunity (Sec. 3.4). Only the 1-vs-1000 CircConv roofline points
    (Fig. 8c) quantify batching.
  - Sparsity: not reported; hypervectors are treated as dense. Working-set sizes: not given.
- **Hardware mechanism**:
  - nsPE: four registers (stationary, passing, streaming, partial sum), three modes (load, GEMM, circular
    convolution); B streams top->bottom through the passing register ("bubble") while A is stationary; partial sums
    reduce top->bottom; circular correlation by reversing A (Sec. 4.2, Fig. 7).
  - Bubble-streaming (BS) dataflow: three d=3 CircConvs finish in n+5 = 8 cycles vs 3n+15 = 24 cycles on a TPU-like
    array (n = 3 prefill cycles) (Fig. 8a); higher arithmetic intensity moves CircConv from memory-bound (GPU, TPU)
    to compute-bound (Fig. 8c).
  - adSCH: interleaved neuro/symbolic processing across tasks + adaptive cell-wise (neural vs symbolic) and
    column-wise (parallel symbolic tasks) array partitioning (Sec. 4.2, Fig. 9).
  - NSFlow (FPGA): execution trace -> dataflow graph -> Design Architecture Generator (system-config search, mapping
    selection) -> RTL -> bitstream via v++ and the XRT API; backend template = BRAM/URAM + CogSys arrays + SIMD units
    (element-wise, reduction, scalar) + control; DFS critical path, BFS same-depth parallelism, inter-loop overlap,
    runtime functions of (n, d, m, n, k), per-node memory footprint (Sec. 4.3, Fig. 10).
  - SoC: TSMC 40nm; 10 RRAM neural tiles + 2 SRAM symbolic tiles; shared memory of 12 direct-mapped SRAM banks with
    each tile writing only to its designated bank; direct-transfer bus; SECDED ECC on neural tiles; 128-bit
    instruction with five address spaces; module-level clock gating and core-level power gating; software-defined
    scheduler driven by the neural/symbolic ratio (Sec. 4.4, Fig. 11a).
  - Hybrid-memory rationale: neural kernels need capacity (RRAM), symbolic kernels need low latency (SRAM), and the
    two datapath latencies are balanced (Sec. 4.4, Fig. 11b).
- **Simulation / implementation method**:
  - Profiling with Nsight Systems and Nsight Compute on RTX 2080 Ti (Sec. 3.5); roofline on RTX 2080 Ti (Fig. 4b);
    edge measurements on Xavier NX and Jetson TX2 (Sec. 3.1, Fig. 3b).
  - CogSys: analytical cycle counts (Fig. 8a) and roofline (Fig. 8c) only; full evaluation is in HPCA'25.
  - NSFlow: RTL synthesized to an FPGA bitstream (Sec. 4.3, Fig. 10a); no performance results reproduced here.
  - SoC: silicon (die shot, Fig. 11c); power-gating gain from post-layout/APR analysis (Sec. 4.4).
- **Limitations**:
  - Hardware case studies cover one class only (VSA-based RAVEN/RPM). Search (MCTS, A*, PDDL), SAT, deductive DBs,
    KG reasoning, probabilistic circuits/MCMC/HMM and symbolic physics appear only in the taxonomy (Sec. 2.1, Fig. 2)
    and are never profiled.
  - The 7 profiled workloads run on one GPU generation; the baseline behind the 75x symbolic speedup is not stated
    in this paper (Sec. 4.1); the 10.8 TOPS/W has no stated precision or conditions (Sec. 4.1).
  - The "Other" operator category is undefined although it is 58.9% of NLM symbolic and 24.0% of LNN symbolic time
    (Fig. 4a).
  - No sparsity, working-set, baseline-energy or absolute FLOP data beyond the 19% FLOP share (Sec. 3.2).
  - Extraction caveats: Fig. 3a per-workload percentages and Fig. 8c roofline labels are positionally garbled.
- **Numbers worth reusing**: see "Hardware constants" below; headline items are NVSA 92.1% symbolic time vs 19% of
  FLOPs (Sec. 3.2), ALU <10% / L1 ~20% / L2 ~40% / DRAM BW ~90% / transfers ~50% of latency (Sec. 3.5), circular
  convolution ~80% of symbolic runtime (Sec. 3.5, Fig. 6), 8 vs 24 cycles (Fig. 8a), O(d^2) circulant (Sec. 4.2).
- **Implications for SymFabric**:
  - Generality evidence is narrower than the taxonomy: the paper measures the VSA models plus the LNN/LTN/NLM/ZeroC
    logic-style models (Sec. 3.1). For
    search, SAT, deductive DB, KG, probabilistic-circuit and planning kernels it offers names (Fig. 2) but no data.
    SymFabric must produce its own profiles there; cite this paper only for the VSA and soft-logic rows.
  - Bind/unbind must be a first-class structured INDEX (rotation) in the operator algebra, executed on the systolic
    array without materializing the O(d^2) circulant (Sec. 4.2). CogSys's BS dataflow shows a plain systolic
    array can do this (Fig. 8a-c); SymFabric's reconfigurable array should subsume the nsPE's three modes (Fig. 7).
  - Bandwidth gating for PIM: on GPUs the VSA kernels are memory-bound (Fig. 4b, Sec. 3.5), but CogSys removes the
    bottleneck by dataflow and batching, not by PIM (Fig. 8c). This supports SymFabric's "PIM only if bandwidth is
    still the bottleneck after dataflow" rule; near-memory computing is listed as future work (Sec. 4.5).
  - Logic-style workloads spend their symbolic time in data movement/transform/"other" rather than MACs (LNN
    symbolic: 39.4% + 17.3% + 24.0%, Fig. 4a): FILTER/SELECT/UPDATE and host-device orchestration are the cost, so
    SymFabric's win there must come from on-chip control and avoiding the ~50% CPU-GPU transfer overhead (Sec. 3.5).
  - Concurrency Q: adSCH's interleaving and column-wise partitioning (Sec. 4.2) plus the 1-vs-1000 CircConv roofline
    points (Fig. 8c) are the paper's version of SymFabric's many-concurrent-queries argument; batching independent
    symbolic queries is what turns the kernel compute-bound.
  - Semiring set: the paper's kernels use (+,x) on reals (VSA), bit-wise/Hamming ops (SoC tile, Sec. 4.4), and
    min/max-style soft logic (LNN, LTN). SymFabric should include (+,x), (XOR,popcount)/Hamming and (max,min) or
    Lukasiewicz-type semirings to cover these rows.
  - Memory hierarchy: the SoC's capacity-vs-latency split (dense RRAM for neural weights, SRAM for symbolic) and its
    12-bank write-ownership scheme (Sec. 4.4, Fig. 11a-b) are reusable design rationales.
  - Evaluation metrics to adopt: energy/power incl. off-chip memory, latency, throughput, hardware cost, measured on
    real NeSy workloads rather than peak (Sec. 5).
- **Open source**: Project website https://effi-nesy.github.io (Abstract). No code repository is named in the paper.

## Workload taxonomy in the paper

Notation: R = per-workload split reconstructed from bar x-positions in the extracted Fig. 3a (each symbolic/neural
pair sums to 100%; only NVSA's 92.1% is confirmed in running text, Sec. 3.2). + = mathematical characterization not
stated in the paper, taken from the cited model's own paper.

| category (paradigm) | example workloads / models named | symbolic kernel (what is computed) | data structure | where the time goes (CPU+GPU) |
|---|---|---|---|---|
| Neuro\|Symbolic, VSA abductive reasoning | NVSA, PrAE (Sec. 2.1, 3.1) | VSA bind/unbind by circular convolution/correlation, vector-vector multiply (similarity), "probabilistic abductive reasoning" over attribute distributions (Sec. 2.1, 3.5, 4.2) | dense real hypervectors, d = 2048-20480 (Fig. 8c); codebooks (+); O(d^2) circulant if done as GEMV (Sec. 4.2) | NVSA symbolic 92.1% of time, 19% of FLOPs (Sec. 3.2); PrAE symbolic 80.5% (Fig. 3a, R); circ-conv + vec-vec mult ~80% of symbolic time (Sec. 3.5, Fig. 6); NVSA symbolic ops: 49.9% vec/elem, 35.7% conv, 7.12% data movement (Fig. 4a); ALU <10%, DRAM BW ~90% (Sec. 3.5) |
| Neuro\|Symbolic, VSA for vision | VSAIT (Sec. 2.1, 3.1) | VSA hypervector transformations on image features (Sec. 3.3 groups it with NVSA/PrAE) | hypervectors | symbolic 83.7% (Fig. 3a, R); symbolic ops: 65.3% vec/elem, 20.8% data transform, 13.87% data movement (Fig. 4a) |
| Neuro\|Symbolic, probabilistic / soft logic programming | DeepProbLog, NeurASP, NeuPSL (Sec. 2.1) | probabilistic logic inference over programs (weighted model counting / ASP / PSL) (+) | logic programs, ground clauses (+) | not profiled |
| Neuro\|Symbolic, program execution for VQA | NSCL, NSVQA (Sec. 2.1) | symbolic program execution over scene graphs (+) | scene graphs, programs (+) | not profiled |
| Neuro:Symbolic->Neuro, real-valued logic networks | LNN (Sec. 2.1, 3.1); DL for symbolic math, differentiable ILP (named only) | "enforce symbolic constraints on NN outputs" (Sec. 2.1); "frequent logical operations" (Sec. 3.3); weighted real-valued logic rule evaluation with truth bounds (+) | symbolic structure compiled into the neural structure (Fig. 5); tensors of truth values (+) | symbolic 45.4% (Fig. 3a, R); symbolic ops: 39.4% data movement, 24.0% other, 19.3% vec/elem, 17.3% data transform (Fig. 4a) |
| Neuro_Symbolic, logic as soft constraints | LTN, DON (Sec. 2.1) | "logical formulas define constraints on tensor representations" (Sec. 2.1): fuzzy-logic satisfaction of FOL formulas (+) | tensor embeddings + FOL formulas (Sec. 2.1) | LTN symbolic 52.0% (Fig. 3a, R); symbolic ops: 73.1% vec/elem, 18.1% other, 6.36% data movement (Fig. 4a); LTN neural part is 62.5% MatMul (Fig. 4a) |
| Neuro[Symbolic], logic/graph reasoning inside the NN | NLM, ZeroC, GNN with attention over rules (Sec. 2.1) | "symbolic knowledge represented in graph structures" (Sec. 2.1); NLM: neural logic over predicate tensors (+) | graph structures (Sec. 2.1); predicate tensors (+) | NLM symbolic 60.6%, ZeroC symbolic 26.8% (Fig. 3a, R); NLM symbolic ops: 58.9% other, 22.9% vec/elem, 14.36% data movement; ZeroC symbolic ops: 74.9% vec/elem, 22.9% data movement (Fig. 4a) |
| Symbolic[Neuro], search with neural evaluation | AlphaGo, AlphaZero (Sec. 2.1) | Monte Carlo Tree Search with NN state evaluation (Sec. 2.1) | game tree (+) | not profiled |
| Kernel classes named but not profiled (Fig. 2) | FOL, deductive database, SAT solver; graph search, PDDL planning, A*; concept graphs, KG reasoning; automata, graph/symbolic algebra; neural DEs, PINNs, differentiable physics; probabilistic circuits, MCMC, HMM, PGM | logic rule evaluation, search, graph traversal, algebraic rewriting, probabilistic inference (names only, Fig. 2) | clause sets, graphs, circuits (implied by names) | not profiled |

## Hardware constants (for reuse with source)

| quantity | value | unit | location | conditions |
|---|---|---|---|---|
| NVSA accuracy on RAVEN-style spatial-temporal reasoning | 98.8 | % | Sec. 1 | Hersche et al. 2023; beats human, ResNet, GPT-4 |
| profiled NeSy workloads | 7 | count | Sec. 3.1 | LNN, LTN, NVSA, NLM, VSAIT, ZeroC, PrAE |
| NVSA symbolic share of inference time | 92.1 | % | Sec. 3.2, Fig. 3a | Xeon Silver 4114 + RTX 2080 Ti, RPM task |
| NVSA symbolic share of FLOPs | 19 | % | Sec. 3.2 | same |
| NVSA end-to-end latency, RTX 2080 Ti | 380 | s | Sec. 3.2, Fig. 3b | RPM task |
| NVSA end-to-end latency, Jetson TX2 | 7507 | s | Sec. 3.2, Fig. 3b | RPM task |
| NVSA symbolic share for task sizes 2x2 to 3x3 | >87 | % | Sec. 3.2, Fig. 3c | symbolic runtime "growing quadratically" |
| symbolic share per workload: LNN / LTN / NVSA / NLM / VSAIT / ZeroC / PrAE | 45.4 / 52.0 / 92.1 / 60.6 / 83.7 / 26.8 / 80.5 | % | Fig. 3a (R) | neural share = 100 minus these; assignment reconstructed, verify on PDF |
| circ-conv + vector-vector mult share of symbolic runtime | ~80 | % | Sec. 3.5, Fig. 6 | VSA-based NeSy; per RAVEN attribute Type/Size/Color/Number/Position |
| ALU utilization, NVSA symbolic ops | <10 | % | Sec. 3.5 | Nsight Systems/Compute, RTX 2080 Ti |
| L1 cache hit rate, NVSA symbolic ops | ~20 | % | Sec. 3.5 | same |
| L2 cache hit rate, NVSA symbolic ops | ~40 | % | Sec. 3.5 | same |
| DRAM bandwidth usage, NVSA symbolic ops | ~90 | % | Sec. 3.5 | "memory-bounded execution" |
| data transfers share of total latency | ~50 | % | Sec. 3.5 | NVSA |
| share of those transfers that are CPU<->GPU | >80 | % | Sec. 3.5 | NVSA |
| CogSys speedup on symbolic tasks | 75 | x | Sec. 4.1 | baseline not stated here (from Wan et al. 2025b) |
| SoC peak efficiency | 10.8 | TOPS/W | Sec. 4.1 | precision/conditions not stated |
| TPU-like array footprint for circ-conv as GEMV | O(d^2) | - | Sec. 4.2 | circulant matrix materialized; GEMVs serialized |
| TPU-like array cycles for 3 CircConvs | 3n+15 = 24 | cycles | Fig. 8a | n = 3 prefill, d = 3 |
| CogSys BS-dataflow cycles for 3 CircConvs | n+5 = 8 | cycles | Fig. 8a | same; Fig. 8b uses a 3x1 nsPE array, d = 3 |
| roofline PE count (TPU-like and CogSys) | 2^14 | PEs | Fig. 8c | printed as "214 PEs" in extraction, superscript lost |
| roofline vector dimensions / batch sizes | 2048, 20480 / 1, 1000 | d / CircConvs | Fig. 8c legend | GPU, TPU, CogSys points |
| roofline ceilings and bandwidth slopes | 130 and 23 / 15 and 2 | TFLOPS / TB/s | Fig. 8c labels | label-to-device assignment garbled in extraction; do not cite without PDF |
| nsPE registers / modes | 4 / 3 | count | Sec. 4.2, Fig. 7 | stationary, passing, streaming, partial sum / load, GEMM, circ-conv |
| SoC technology | 40 | nm | Sec. 4.4, Fig. 11c | TSMC |
| SoC neural tiles / symbolic tiles / total | 10 / 2 / 12 | count | Sec. 4.4 | RRAM-based / SRAM-based |
| RRAM per neural tile | 576 | KB | Sec. 4.4 ("576K"), Fig. 11a,c labels | |
| shared-memory banks | 12 | count | Fig. 11a label | direct-mapped SRAM banks; one write-owner tile each (Sec. 4.4) |
| instruction width / address spaces | 128 / 5 | bit / count | Sec. 4.4 | |
| die edge | 5.25 | mm | Fig. 11c label | |
| power reduction from fine-grained power gating | 5 | x | Sec. 4.4 | post-layout/APR analysis |

Fig. 4a operator breakdown of runtime, % of each part (N = neural part, S = symbolic part). Recovered from the
extraction; the vector/element-wise row was split across two lines and re-assembled in reading order; every column
sums to 100 +/- 0.1, which validates the assignment.

| workload (part) | Conv | MatMul | Vector/Elem-wise | Data Transform | Data Movement | Other |
|---|---|---|---|---|---|---|
| LNN (N) | 0.00 | 0.51 | 43.6 | 16.4 | 39.5 | 0.00 |
| LNN (S) | 0.00 | 0.00 | 19.3 | 17.3 | 39.4 | 24.0 |
| LTN (N) | 0.00 | 62.5 | 26.8 | 7.20 | 3.48 | 0.00 |
| LTN (S) | 0.00 | 0.00 | 73.1 | 2.40 | 6.36 | 18.1 |
| NVSA (N) | 30.7 | 34.8 | 22.0 | 3.11 | 9.40 | 0.00 |
| NVSA (S) | 35.7 | 0.52 | 49.9 | 6.82 | 7.12 | 0.00 |
| NLM (N) | 0.00 | 24.5 | 34.6 | 16.0 | 24.9 | 0.00 |
| NLM (S) | 0.00 | 0.00 | 22.9 | 3.85 | 14.36 | 58.9 |
| VSAIT (N) | 59.5 | 30.0 | 6.75 | 2.94 | 0.84 | 0.00 |
| VSAIT (S) | 0.00 | 0.00 | 65.3 | 20.8 | 13.87 | 0.00 |
| ZeroC (N) | 31.6 | 28.2 | 33.7 | 3.96 | 2.52 | 0.00 |
| ZeroC (S) | 0.00 | 0.00 | 74.9 | 2.13 | 22.9 | 0.00 |
| PrAE (N) | 28.6 | 36.0 | 20.1 | 4.72 | 10.6 | 0.00 |
| PrAE (S) | 28.0 | 0.91 | 56.3 | 8.11 | 6.69 | 0.00 |

The 35.7% (NVSA S) and 28.0% (PrAE S) "Conv" entries are consistent with circular convolution being counted under
Conv by the profiler (inference, not stated in the paper). Fig. 4b (roofline, RTX 2080 Ti) places the symbolic parts
of PrAE, NVSA, VSAIT, ZeroC at low operational intensity (memory-bound) and the neural parts at high intensity
(compute-bound); exact coordinates are not readable in the extraction.

## Claims about bottlenecks

- Sec. 1: "neuro-symbolic computing exhibits increased kernel heterogeneity, greater memory intensity, and irregular
  data access patterns. These characteristics create a growing mismatch with modern hardware architectures, which
  are predominantly optimized for matrix operations and convolutions".
- Sec. 3.2: "symbolic reasoning accounts for 92.1% of total inference time despite contributing only 19% of FLOPs,
  highlighting inefficient execution on GPUs."
- Sec. 3.3: symbolic operators are "Dominated by vector and scalar operations with low compute intensity and
  irregular control flow"; "NVSA relies on high-dimensional vector-symbolic transformations, which GPUs struggle to
  execute efficiently."
- Sec. 3.4 (roofline): "neural computations are compute-bound, while symbolic operations are memory-bound due to
  large streaming operations (e.g., NVSA and PrAE)."
- Sec. 3.4 (dependency graph): "The complex control results in inefficiency in CPU and GPU, and the vector-symbolic
  computation period results in low hardware utilization."
- Fig. 5 caption: "Complex control and symbolic-only phase operation result in inefficiency and low hardware
  resource utilization".
- Sec. 3.5: "NeSy workloads suffer from ALU underutilization, low cache efficiency, and excessive data transfer";
  "ALU utilization <10%, L1 and L2 cache hit rates of around 20% and 40%, respectively, and DRAM bandwidth usage
  near 90%, indicating memory-bounded execution. Data transfers contribute around 50% of total latency, with >80%
  occurring between CPU and GPU. Additionally, synchronization overheads further reduce CPU utilization."
- Sec. 3.6: "Symbolic operations tend to be memory-bounded due to large element streaming with low reuse and
  irregular access patterns."; "NeSy workloads involve complex control flow, sequential symbolic phases, and
  fine-grained dependencies between neural and symbolic, limiting hardware parallelism."
- Sec. 4.1: NeSy needs primitives for "irregular, memory-bound symbolic operations (e.g., logic, graph traversal,
  vector-symbolic ops)."
- Sec. 4.2: "TPU-like systolic array (SA) exhibits high memory footprint and low parallelism for symbolic circular
  convolution operations"; "matrix memory footprint of O(d2)"; "TPU-like SA is incapable of parallelizing multiple
  GEMV on a systolic cell and need to process them sequentially."; "GPUs require extra computations to handle the
  index calculations for the circularly shifted vector."
- Sec. 4.2 (adSCH): "(1) sequential neuro-symbolic execution leads to high latency; (2) heterogeneous kernels cause
  low utilization."
- Sec. 4.4: "neural kernels require large memory capacity, making RRAM ideal due to its density; symbolic kernels
  benefit from SRAM's lower latency for compute-heavy logic."
