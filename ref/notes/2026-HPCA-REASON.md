# REASON: Accelerating Probabilistic Logical Reasoning for Scalable Neuro-Symbolic Intelligence (HPCA 2026)

Wan, Liu, Qian, Yang, Raychowdhury, Krishna (Georgia Tech). DOI 10.1109/HPCA68181.2026.11408458 (p. 1).
Source: pdftotext -layout. Body text and tables are clean; bar-chart labels (Figs. 3a, 11, 12, 13) are readable but
their bar assignment was reconstructed from label positions and is flagged as such wherever used.

- **Workload**: Six neuro-symbolic pipelines where an LLM/DNN feeds a symbolic or probabilistic kernel (Table I, p. 4):
  AlphaGeometry (math theorem proving; FOL + SAT solver + acyclic graph), R2-Guard (LLM unsafety detection; FOL + probabilistic
  circuit + HMM), GeLaTo (constrained text generation; FOL + SAT + HMM), Ctrl-G (interactive text editing/infilling; HMM + PCs),
  NeuroPC (classification; DNN + FOL + PC), LINC (logical/deductive reasoning; FOL + solver). 10 datasets: IMO, MiniF2F,
  TwinSafety, XSTest, CommonGen, News, CoAuthor, AwA2, FOLIO, ProofWriter (Sec. VII-A, p. 11-12). Three kernel families are
  supported: FOL/SAT (DPLL/CDCL, cube-and-conquer), probabilistic circuits (sum-product), HMMs (forward-backward/decoding)
  (Sec. II-C, p. 3; Fig. 5, p. 6). The symbolic/probabilistic part takes 63.8% / 62.7% / 36.6% / 63.9% / 50.5% / 34.8% of
  end-to-end runtime for AlphaGeometry / R2-Guard / GeLaTo / Ctrl-G / NeuroPC / LINC on an A6000 (Sec. III-A, p. 4).
- **Operators**: Everything is first lowered to one DAG abstraction (Sec. IV-A, p. 6), pruned (Sec. IV-B, p. 6-7) and
  regularized into balanced two-input trees (Sec. IV-C, p. 7). GPU hotspot kernels: sparse MatVec and logic (symbolic),
  marginal and Bayesian (probabilistic) (Table II, p. 5). Mapping to SymFabric classes, per PE mode (Sec. V-B, p. 8):
  - Probabilistic mode: tree nodes execute "either addition or multiplication" -> COMBINE (product nodes) + REDUCE (sum
    nodes) over the (+, x) probability semiring. PC = one bottom-up pass; HMM = DAG unrolled over time steps with sequential
    message passing (Sec. IV-A, p. 6). No max/min, log-domain or other semiring operator is described.
  - Symbolic mode (BCP for SAT/FOL compiled to CNF): INDEX = watched-literal head-pointer table indexed by literal ID
    ("enabling O(1) access") plus next-watch pointer chasing (Sec. V-D, p. 9); FILTER = comparator classifying literals
    TRUE/FALSE/UNASSIGNED; REDUCE = adder used as counter of FALSE literals per clause, unit clause / conflict propagated up
    the reduction tree; UPDATE = decision/implication broadcast down the distribution tree, implications serialized through the
    BCP FIFO; SELECT = decision by the controller and CDCL conflict analysis on the scalar PE (Sec. V-B, V-D, V-E, p. 8-10).
    Semiring: Boolean (clause = OR over literals, formula = AND over clauses; Sec. IV-A, p. 6). Iteration: DPLL/CDCL search
    with cube-and-conquer parallel DPLL "cubes" (Sec. II-C, p. 3; Fig. 9, p. 9).
  - SpMSpM mode: leaves = multipliers, internal nodes = adders -> COMBINE + REDUCE over (+, x) for "small-scale neural or
    neural-symbolic models" (Sec. V-B, p. 8).
- **Data scale**: On-chip SRAM 1.25 MB (Fig. 10, p. 11); the clause database is meant to live off-chip: "allowing most clauses
  to reside in remote scratchpad memory or DRAM, with on-chip SRAM caching only the required clauses indexed by WLs unit"
  (Sec. V-D, p. 9). No DAG node/edge counts or clause counts are reported. Pruning cuts memory footprint by 21-43% per task,
  31.7% on average (Table IV, Sec. VII-B, p. 12). Concurrent queries Q: the API carries batch_id/batch_size (Listing 1, p. 10)
  and the GPU-REASON pipeline overlaps batch N symbolic with batch N+1 neural (Sec. VI-C, p. 11-12), but no Q value is given.
  Sparsity of symbolic/probabilistic kernels: 82% / 87% / 75% / 83% / 89% / 83% (AlphaGeometry ... LINC order) (Sec. III-B, p. 5).
- **Hardware mechanism**: GPU co-processor "plug-in" that shares L2 and the inter-SM fabric (Fig. 6a, p. 7; Sec. VII-C, p. 12).
  12 tree-based PEs / 80 tree nodes (Table III, Fig. 10, p. 11). Each PE is a Reconfigurable Tree Engine: bidirectional tree,
  Benes N:N input crossbar from the SRAM/register banks, one-bank-one-PE outputs, forwarding logic back to SRAM; a scalar PE
  issues VLIW-like instructions; a custom SIMD unit runs the "parallelable subset of symbolic solvers" (Sec. V-A, V-C, p. 8-9).
  Memory: dual-port wide-bitline SRAM banks as a banked L1, prefetcher + DMA, MMU (Sec. V-A, p. 8); banked register file with
  DSE-selected tree depth D=3, B=64 banks, R=32 registers/bank (Sec. V-F, p. 10); automatic write-address generation to the
  lowest free register, predicted by the compiler (Sec. V-C, p. 9). Symbolic support: hardware two-watched-literals unit tied
  to the N SRAM banks, linked-list watch lists with head-pointer table, BCP FIFO on the M:1 output interconnect, priority conflict
  handling that halts DMA and flushes the FIFO (Sec. V-D, V-E, p. 9-10; Fig. 6e, p. 7). Compiler: block decomposition (block
  depth <= tree depth), joint PE/register-bank mapping, tree mapping with fusion/replication, pipeline-aware reordering with
  live-range spills (Sec. V-C, p. 9; Fig. 7, p. 8). Interconnect argument: tree O(log N) root-to-leaf latency vs O(sqrt N) mesh
  vs O(N) bus (Sec. V-D, p. 10; Fig. 8, p. 9). Unified engine claimed to reach ">90% utilization with 58% lower area/power
  than decoupled designs" (Sec. V-F, p. 10). System: REASON_execute / REASON_check_status, neural_ready / symbolic_ready
  shared-memory flags, two-level (GPU-REASON and intra-REASON) pipeline (Sec. VI-B, VI-C, p. 11-12).
- **Simulation / implementation method**: RTL synthesized with Synopsys DC and placed-and-routed with Cadence Innovus at TSMC
  28 nm, using DPU-v2 [46] as the baseline template; power from Synopsys PTPX power traces (Sec. VII-A, p. 11-12). 12 nm and
  8 nm rows are DeepScaleTool projections at 0.8 V, 500 MHz (Table III footnote, p. 11). End-to-end: cycle-accurate Accel-Sim /
  GPGPU-Sim configured as Orin NX (8 SMs, 32 threads/warp, 48 KB shared memory, 128 KB L1, 2 MB L2, 128-bit LPDDR5 at
  104 GB/s) driven by memory traces captured on Orin; GPU baselines are real-hardware measurements (Sec. VII-A, p. 12).
  ML-accelerator baselines: SCALE-Sim with eight 128x128 systolic arrays (TPU-like) and MAERI with eight PEs in a 56-node tree
  (DPU-like) (Sec. VII-C, p. 12-13). Profiling: A6000 + Sapphire Rapids + DDR5, PyTorch 2.5, JAX 0.4.6, Nsight Systems/Compute,
  cProfile; Jetson Orin for edge (Sec. III, p. 4).
- **Limitations**: (1) REASON is simulated/synthesized while GPU baselines are measured; 12/8 nm rows are scaled, not built
  (Table III, p. 11). (2) The 28 nm clock frequency is never stated; 500 MHz appears only for the scaled rows. (3) Energy
  numbers are inconsistent: abstract "310-681x ... over desktop and edge GPUs" (p. 1); Sec. VII-C "681x energy efficiency
  compared to RTX GPU" in a mixed-workload long-term test (p. 12); Fig. 12b average-group labels read 838 / 681 / 310 in legend
  order Xeon CPU / Orin NX / RTX GPU and 0.87 for REASON on an axis labelled Energy (J) (p. 12). (4) No multi-query
  concurrency, no DAG/clause size, no problem-size scaling of REASON itself (Fig. 3b scales only the GPU baseline, p. 4).
  (5) Pruning slightly changes accuracy: TwinSafety AUPRC 0.758 -> 0.752, FOLIO 92% -> 91%, CoAuthor 87% -> 86% (Table IV,
  p. 12). (6) Fig. 13 LINC end-to-end labels 6.10 and 8.59 cannot be assigned to TPU-like vs DPU-like from the extraction.
- **Numbers worth reusing**: 12-50x speedup, 310-681x energy efficiency, 0.8 s per task, 6 mm2, 2.12 W (Abstract, p. 1);
  50.65x vs Orin NX and 11.98x vs RTX GPU averaged over 10 tasks (Sec. VII-C, p. 12; Fig. 11); 4.91x / 1.60x speedup and
  802x / 268x energy efficiency vs V100 / A100 (Sec. VII-C, p. 12); symbolic-only runtime of a TPU-like systolic array is
  74.71-109.24x REASON and of a DPU-like tree array 4.84-25.13x (Fig. 13 bar labels, p. 13); neuro-only TPU-like is 0.66-0.73x
  (faster than REASON) (Fig. 13, p. 13); ablation: memory layout -22%, + reconfigurable array -56%, + scheduling -73% runtime
  (Sec. VII-C, p. 13); algorithm-only on Orin NX leaves 78.3-87.0% of baseline runtime, algorithm + HW 1.94-2.08% (Table V,
  p. 13); GPU DRAM-BW utilization of symbolic/probabilistic kernels 57.4-70.3% vs ALU utilization 29.3-52.8% (Table II, p. 5).
  Full list with conditions in the table below.
- **Implications for SymFabric**:
  - The benchmark suite (6 workloads, 10 datasets, Table I / Sec. VII-A) and the Orin NX-class Accel-Sim setup (Sec. VII-A)
    are directly reusable as SymFabric's neuro-symbolic evaluation harness; symbolic share of 35-64% (Sec. III-A, p. 4)
    bounds end-to-end gains by Amdahl.
  - REASON's headline claim against systolic arrays is Fig. 13: a SCALE-Sim TPU-like array is 75-109x slower on symbolic-only
    kernels (p. 13). SymFabric must show that its operator algebra (BCP as Boolean-semiring sparse MatVec, sum-product as
    (+, x) SpMV over the DAG adjacency) removes this gap rather than running the GPU-style kernels unchanged.
  - REASON hard-wires the (+, x) semiring in probabilistic mode and Boolean BCP in symbolic mode (Sec. V-B, p. 8); a
    semiring-parametric fabric (max-plus/Viterbi, tropical, log-semiring, counting) is a differentiator the paper does not cover.
  - REASON's DAG is compiled offline into a binary (Sec. IV-C, p. 7); dynamic relations (Datalog fixpoints, runtime clause
    learning beyond the scalar PE) are not served. SymFabric's UPDATE/iteration operators should target exactly this.
  - Memory-boundedness evidence for the PIM decision: GPU DRAM-BW utilization 57-70% for symbolic/probabilistic kernels with
    L1 hit rates 37-54% (Table II, p. 5). REASON answers with 1.25 MB banked SRAM + watched-literal indexing and only 104 GB/s
    LPDDR5 (Fig. 10, p. 11), not PIM; SymFabric should quantify whether its own working sets exceed that on-chip budget.
  - Tree vs mesh broadcast argument (O(log N) vs O(sqrt N), Sec. V-D, p. 10) is aimed at exactly a systolic layout; SymFabric
    needs either a reduction/broadcast network or a formulation of BCP that is streaming-friendly (e.g., batched Q queries).
  - Baseline chip constants (Table III, p. 11) are convenient for area/power normalization at 28 nm.
- **Open source**: none stated (no repository, artifact or URL for REASON in the text).

## Hardware constants (for reuse with source)

| quantity | value | unit | location in paper | conditions (technology node, frequency, configuration) |
|---|---|---|---|---|
| technology node | 28 | nm | Table III, Fig. 10 (p. 11); Abstract (p. 1) | TSMC 28 nm, Synopsys DC + Cadence Innovus (Sec. VII-A, p. 11) |
| core VDD | 0.9 | V | Fig. 10 (p. 11) | 28 nm design |
| clock frequency | 500 | MHz | Table III footnote (p. 11) | only stated for DeepScaleTool-scaled 12 nm / 8 nm rows at 0.8 V; 28 nm frequency not stated |
| area | 6.00 | mm2 | Table III, Fig. 10 (p. 11); Abstract (p. 1) | 28 nm, post-P&R |
| area (scaled) | 1.37 / 0.51 | mm2 | Table III (p. 11) | 12 nm / 8 nm, DeepScaleTool, 0.8 V, 500 MHz |
| average power | 2.12 | W | Table III, Fig. 10 (p. 11); Sec. VII-A (p. 11-12) | 28 nm, PTPX power-trace average across workloads |
| power (scaled) | 1.21 / 0.98 | W | Table III (p. 11) | 12 nm / 8 nm |
| per-workload power (visible labels) | 1.88, 2.51 | W | Fig. 12a bar labels (p. 12) | workloads shown: TwinSafety, ComGen, XSTest, AwA2, News |
| on-chip SRAM | 1.25 | MB | Table III, Fig. 10 (p. 11) | shared local memory + N SRAM banks (M1-M4) |
| DRAM bandwidth | 104 | GB/s | Fig. 10 (p. 11); Sec. VII-A (p. 12) | 128-bit LPDDR5 peak, Orin NX-class simulator |
| number of PEs | 12 | PEs | Table III, Fig. 10 (p. 11) | tree-based PEs |
| number of tree nodes | 80 | nodes | Table III, Fig. 10 (p. 11) | total over 12 PEs |
| tree depth D | 3 | levels | Sec. V-F (p. 10) | DSE-selected on latency/energy/EDP |
| register banks B | 64 | banks | Sec. V-F (p. 10) | DSE-selected; per-PE vs total not stated |
| registers per bank R | 32 | registers | Sec. V-F (p. 10) | register width not stated |
| PE utilization (unified engine) | >90 | % | Sec. V-F (p. 10) | vs decoupled symbolic/probabilistic engines |
| area/power saving vs decoupled | 58 | % | Sec. V-F (p. 10) | same comparison |
| GPU model: SMs / threads per warp | 8 / 32 | - | Sec. VII-A (p. 12) | Accel-Sim (GPGPU-Sim) configured as Orin NX |
| GPU model: shared memory / L1 / L2 | 48 / 128 / 2048 | KB | Sec. VII-A (p. 12) | per SM / per SM / unified L2 |
| baseline Orin NX | 8 nm, 4 MB SRAM, 512 CUDA cores, 450 mm2, 15 W | - | Table III (p. 11) | as listed |
| baseline RTX A6000 | 8 nm, 16.5 MB, 10572 CUDA cores, 628 mm2, 300 W | - | Table III (p. 11) | as listed |
| baseline Xeon CPU | 10 nm, 112.5 MB, 60 cores/socket, 1600 mm2, 270 W | - | Table III (p. 11) | as listed |
| baseline TPU | 7 nm, 170 MB, 8 x 128x128 PEs, 400 mm2, 192 W | - | Table III (p. 11) | as listed |
| baseline DPU (DPU-v2) | 28 nm, 2.4 MB, 8 PEs / 56 nodes, 3.20 mm2, 1.10 W | - | Table III (p. 11) | REASON's RTL template |
| speedup vs Orin NX (avg 10 tasks) | 50.65 | x | Sec. VII-C (p. 12); Fig. 11 | end-to-end, REASON simulated, GPU measured |
| speedup vs RTX GPU (avg 10 tasks) | 11.98 | x | Sec. VII-C (p. 12); Fig. 11 | same |
| speedup vs Xeon CPU per task | 95.6-100.4 | x | Fig. 11 bar labels (p. 12) | not stated in text |
| per-task speedup vs Orin NX | 48.3, 51.5, 48.9, 50.3, 48.0, 50.2, 53.0, 51.7, 51.6, 53.0 | x | Fig. 11 bar labels (p. 12) | IMO, MiniF2F, TwinS, XSTest, ComGen, News, CoAuthor, AwA2, FOLIO, ProofWriter |
| per-task speedup vs RTX GPU | 12.4, 12.1, 11.5, 11.4, 13.8, 12.4, 10.6, 9.8, 12.7, 13.1 | x | Fig. 11 bar labels (p. 12) | same task order |
| speedup vs V100 / A100 | 4.91 / 1.60 | x | Sec. VII-C (p. 12) | conditions not detailed |
| energy efficiency vs V100 / A100 | 802 / 268 | x | Sec. VII-C (p. 12) | conditions not detailed |
| energy efficiency vs RTX GPU | 681 | x | Sec. VII-C (p. 12) | long-term mixed workloads, 15 s idle intervals |
| energy efficiency range | 310-681 | x | Abstract (p. 1) | "over desktop and edge GPUs" |
| Fig. 12b average-group labels | 838 / 681 / 310 / 0.87 | axis says J | Fig. 12b (p. 12) | legend order Xeon CPU / Orin NX / RTX GPU / REASON; assignment by label position, conflicts with text above |
| end-to-end task latency | 0.8 | s | Abstract (p. 1) | "real-time performance (<1.0 s)" Sec. VII-C (p. 12) |
| Ctrl-G baseline latency | >5 | min | Sec. I (p. 1) | single task on a desktop GPU, current implementation |
| symbolic share of runtime | 63.8 / 62.7 / 36.6 / 63.9 / 50.5 / 34.8 | % | Sec. III-A (p. 4) | AlphaGeometry / R2-Guard / GeLaTo / Ctrl-G / NeuroPC / LINC, A6000 |
| symbolic share with LLaMA-7B | 69.0 / 65.5 | % | Sec. III-A (p. 4) | GeLaTo / LINC |
| AlphaGeometry symbolic FLOPS share | 19.3 | % | Sec. III-A (p. 4) | vs 63.8% of runtime, A6000 |
| per-task symbolic share | 67.4, 60.2, 63.5, 66.8, 57.9, 36.6, 34.9, 38.4, 63.9, 60.1, 67.7, 50.5, 34.0, 35.7 | % | Fig. 3a bar labels (p. 4) | IMO, MiniF2F, TwinS, XSTest, Mod, ComGen, Review, News, ComGen, TextFl, Math, AwA2, FOLIO, Proof |
| kernel sparsity | 82 / 87 / 75 / 83 / 89 / 83 | % | Sec. III-B (p. 5) | AlphaGeometry ... LINC order |
| GPU compute throughput | 96.8 / 62.2 / 32.5 / 14.7 / 35.0 / 31.1 | % | Table II (p. 5) | MatMul / Softmax / Sparse MatVec / Logic / Marginal / Bayesian, A6000 |
| GPU ALU utilization | 98.4 / 72.0 / 43.9 / 29.3 / 48.5 / 52.8 | % | Table II (p. 5) | same kernel order |
| GPU L1 hit rate | 88.5 / 85.0 / 53.6 / 37.0 / 42.4 / 40.7 | % | Table II (p. 5) | same |
| GPU L2 hit rate | 73.4 / 66.7 / 43.9 / 32.7 / 50.2 / 47.6 | % | Table II (p. 5) | same |
| GPU DRAM BW utilization | 39.8 / 28.6 / 57.4 / 70.3 / 60.8 / 68.0 | % | Table II (p. 5) | same |
| GPU warp execution efficiency | 96.3 / 94.1 / 48.8 / 54.0 / 59.3 / 50.6 | % | Table II (p. 5) | same |
| GPU branch efficiency | 98.0 / 98.7 / 60.0 / 58.1 / 63.4 / 66.9 | % | Table II (p. 5) | same |
| GPU eligible warps per cycle | 7.2 / 7.0 / 2.4 / 2.1 / 2.8 / 2.5 | % | Table II (p. 5) | same |
| memory footprint reduction | 31.7 avg; 25, 21, 37, 30, 41, 27, 29, 43, 38, 26 | % | Sec. VII-B; Table IV (p. 12) | IMO, MiniF2F, TwinSafety, XSTest, CommonGen, News, CoAuthor, AwA2, FOLIO, ProofWriter |
| accuracy before -> after pruning | 83->83, 81->81 %; 0.758->0.752, 0.878->0.881 AUPRC; 30.3->30.2, 5.4->5.4 BLEU; 87->86 %; 87->87 %; 92->91 %; 84->84 % | - | Table IV (p. 12) | same task order |
| runtime, algorithm only on Orin NX | 84.2 / 87.0 / 78.3 / 82.9 / 86.6 | % of baseline | Table V (p. 13) | IMO / MiniF / TwinS / XSTest / ComG |
| runtime, algorithm + REASON HW | 2.07 / 1.94 / 2.04 / 1.99 / 2.08 | % of baseline | Table V (p. 13) | same tasks |
| ablation: runtime reduction | 22 / 56 / 73 | % | Sec. VII-C (p. 13) | memory layout / + reconfigurable array / + scheduling |
| TPU-like symbolic-only runtime | 81.35 / 76.10 / 109.24 / 78.48 / 74.71 / 90.89 | x REASON | Fig. 13 bar labels (p. 13) | SCALE-Sim, 8 x 128x128; AlphaG / Guard / GeLaTo / Ctrl-G / NPC / LINC |
| DPU-like symbolic-only runtime | 25.13 / 4.84 / 5.03 / 6.07 / 4.97 / 23.97 | x REASON | Fig. 13 bar labels (p. 13) | MAERI, 8 PEs / 56 nodes; same order |
| TPU-like neuro-only runtime | 0.69 / 0.71 / 0.68 / 0.66 / 0.73 / 0.68 | x REASON | Fig. 13 bar labels (p. 13) | <1 means TPU-like faster |
| DPU-like neuro-only runtime | 4.31 / 4.40 / 4.29 / 4.49 / 4.32 / 4.30 | x REASON | Fig. 13 bar labels (p. 13) | same order |
| TPU-like end-to-end runtime | 21.31 / 17.77 / 10.54 / 18.02 / 9.76 / (6.10 or 8.59) | x REASON | Fig. 13 bar labels (p. 13) | LINC pair ambiguous in extraction |
| DPU-like end-to-end runtime | 7.86 / 2.31 / 2.15 / 2.90 / 2.33 / (8.59 or 6.10) | x REASON | Fig. 13 bar labels (p. 13) | LINC pair ambiguous in extraction |
| CPU parallel efficiency (CPU+GPU) | <5 | % | Sec. VII-C (p. 12) | symbolic/probabilistic ops |
| inter-device communication overhead | >15 | % | Sec. VII-C (p. 12) | CPU+GPU neural-symbolic transfers |
| LLM-side latency reduction | 2.8-3.3 / 4-5 | x | Sec. VII-C (p. 13) | unique prompts / reused prefixes; orthogonal to REASON |
| AlphaGeometry vs CoT LLM efficiency | >2 | x | Sec. II-B, Fig. 2d (p. 3) | math reasoning tasks |

## Workload decomposition as seen by REASON

Mode assignment follows Table I kernel types (p. 4) and the mode definitions of Sec. V-B (p. 8); the paper does not
state a per-workload mode explicitly. FOL/SAT inputs are compiled to CNF DAGs (Sec. IV-A, p. 6).

| workload | symbolic kernel computes | REASON mode | data structure |
|---|---|---|---|
| AlphaGeometry (IMO, MiniF2F) | FOL deduction + SAT search over an acyclic graph of geometric facts (Table I); DPLL/CDCL traversal (Fig. 5) | Symbolic mode (BCP on tree, CDCL conflict analysis on scalar PE) | CNF DAG literal->clause->formula; linked-list watch lists + head-pointer table in SRAM, BCP FIFO (Sec. V-D) |
| R2-Guard (TwinSafety, XSTest, Mod) | FOL rules + PC marginalization ("sparse, scattered accesses for marginalization", Sec. III-B) + HMM | Probabilistic mode for PC/HMM; symbolic mode for FOL part | Sum/product DAG with weights theta, pruned by circuit flow (Sec. IV-B); banked register file + Benes crossbar (Sec. V-C) |
| GeLaTo (CommonGen, News, Review) | FOL + SAT constraint checking + HMM over token sequence (Table I) | Symbolic mode (SAT) + probabilistic mode (HMM) | CNF DAG; time-unrolled HMM DAG with transition/emission product nodes and sum nodes over prior states (Sec. IV-A) |
| Ctrl-G (CoAuthor; ComGen, TextFl, Math in Fig. 3a) | HMM inference that "iteratively reads and writes state probabilities" (Sec. III-B) + PCs | Probabilistic mode | Unrolled HMM/PC DAG, two-input regularized; registers via automatic write-address generation (Sec. V-C) |
| NeuroPC (AwA2) | FOL rules over DNN attribute outputs + PC inference (Table I) | Probabilistic mode (+ symbolic for FOL) | PC sum/product DAG pruned by cumulative flow (Sec. IV-B) |
| LINC (FOLIO, ProofWriter) | FOL theorem proving with a solver over LLM-generated formulas (Table I) | Symbolic mode | CNF DAG; watch lists; "inference rules act as graph transformation operators ... through node and edge expansion" (Sec. IV-A) |

## What REASON cannot do (stated or implied by the paper)

- No runtime graph construction: "These DAGs are constructed offline and used to generate an execution binary that is
  programmed onto REASON hardware." (Sec. IV-C, p. 7). FOL inference is described as "node and edge expansion" (Sec. IV-A,
  p. 6) but the paper does not explain how expansion is served by an offline-compiled binary.
- Pruning needs representative data and is lossy for probabilistic kernels: edges are pruned by cumulative flow "Given a
  dataset D" with a bound on the log-likelihood decrease (Sec. IV-B, p. 7); Table IV shows metric changes (p. 12).
- Only two arithmetic operators in probabilistic mode: nodes "perform arithmetic operations, either addition or
  multiplication" (Sec. V-B, p. 8). No max/min, log-domain, or general semiring operation; numeric format/precision is not stated.
- BCP is inherently serialized: "BCP must propagate them sequentially to preserve the causality chain for conflict analysis"
  (Sec. V-D, p. 10); implications queue in one FIFO, and CDCL conflict analysis runs on the scalar PE, not the tree (Sec. V-E, p. 10).
- Depth-limited blocks: block decomposition selects subgraphs "whose maximum depth does not exceed the hardware tree depth"
  (Sec. V-C, p. 9) with D=3 (Sec. V-F, p. 10); deeper DAGs pay inter-block round trips through SRAM/forwarding.
- Heuristic compiler: "The DAG-to-hardware mapping is an automated heuristic process" (Sec. V-C, p. 9); no optimality claim.
- Small on-chip capacity: 1.25 MB SRAM (Fig. 10, p. 11); large clause databases stay in "remote scratchpad memory or DRAM"
  (Sec. V-D, p. 9) and a watched-literal miss triggers an L2/DMA fetch (Sec. V-E, p. 10).
- Not a neural engine: a TPU-like array is faster on neuro-only work (0.66-0.73x REASON runtime, Fig. 13, p. 13); SpMSpM mode
  is for "small-scale neural or neural-symbolic models" (Sec. V-B, p. 8). LLM inference is assigned to the GPU (Sec. VI-A, p. 11).
- Not standalone: REASON_execute "is called after GPU SMs complete processing the neural LLM" and synchronizes through polled
  shared-memory flags (Sec. VI-B, p. 11-12).
- Kernel coverage is limited to the named families "(e.g., SAT, FOL, PC, HMM)" (Sec. V-F, p. 10); relational/Datalog-style
  joins, knowledge-graph queries, or non-Boolean logics are not discussed.
- No open-source release, no multi-query (Q) evaluation, and no stated 28 nm clock frequency (see Limitations).
