# AMMA: A Multi-Chiplet Memory-Centric Architecture for Low-Latency 1M Context Attention Serving (arXiv 2026)

arXiv:2604.26103v2 [cs.AR], 30 Apr 2026 (header, p.1). ACM-format preprint with placeholder venue "XXXX" (p.1), so no conference is named. Authors from UCSD, Columbia, Yonsei, NVIDIA and Samsung Semiconductor (p.1).

- **Workload**: Decode-phase LLM attention (QKV projection, core attention, output projection) at context lengths up to 1M tokens with small batches (1-32) (Sec. 2.1, p.3; Sec. 7, p.9). GQA models Qwen3-235B and Llama4-Maverick, MLA model DeepSeek-V3 (Sec. 7, p.9). FFN/MoE excluded: assumed disaggregated onto LPUs or GPUs (Sec. 7, p.9; Fig. 1(a), p.2). Prefill not covered.
- **Operators**:
  - QKV projection: GEMV/GEMM with M = batch (1-32) = COMBINE(x, W) + REDUCE over D_m in the (+, x) semiring at FP8 (Sec. 2.1, p.3; Sec. 4.1 P1, p.4).
  - QK^T: COMBINE + REDUCE over d_h, streamed over N = sequence; M = Q-heads-per-KV-head x batch, M = 1-16 (Sec. 4.3, p.5).
  - softmax: REDUCE-max m, FILTER-like rescale exp(s_j - m), REDUCE-sum l (Eq. 5, Sec. 6.2, p.8). The cross-cube merge (m_n, l_n, a_n) with alpha_n = e^{m_n - m} l_n / l (Eq. 6, p.8) is an associative (max, rescaled-sum) monoid, i.e. a log-sum-exp-style semiring instance.
  - PV: COMBINE + REDUCE over the sequence dimension. Output projection: COMBINE + REDUCE over H_Q d_h; it commutes with the per-row scalar alpha_n (Eq. 7, p.9), which is what allows REDUCE to be moved after COMBINE.
  - Collectives: AllGather = SELECT/broadcast; ReduceScatter/Reduce = cross-cube REDUCE (Sec. 6.1, p.8). UPDATE is only the implicit KV-cache append per token. INDEX: none, purely contiguous streaming.
  - Iteration: auto-regressive, one token per step per layer (Sec. 2.1, p.3); output-stationary systolic tiles of fixed 16x16 outputs (Sec. 4.3, p.5).
- **Data scale**: KV cache 2 H_KV S d_h per layer (Sec. 2.1, p.3); S swept 4K-1M (Fig. 10, p.10); Q (concurrent requests) = batch 1-32 (Sec. 2.1, p.3; Sec. 5.1, p.6); dense attention, no sparsity. Arithmetic intensity 32 FLOPs/byte for GQA decode (Qwen3 G = 16, FP8) vs 795 FLOPs/byte provisioned by Rubin (Sec. 1, p.1); MLA has roughly 8x higher intensity than GQA (Sec. 7.1, p.9). GQA/MLA compress KV heads 16-128x (Sec. 3.3, p.4); the worked example uses 4 KV heads (Sec. 5.2, p.6).
- **Hardware mechanism**:
  - Package: 16 HBM4 cubes in a 4x4 2D mesh joined by UCIe 3.0 D2D links; the GPU compute dies are removed entirely (Sec. 4.1, p.4; Table 1, p.9). PNM logic lives in the HBM base logic die, which from HBM4 on is fabricated at <=5 nm (Fig. 1(b), p.2; Sec. 2.2, p.3).
  - Per cube: 12 cores, one memory controller and four D2D ports on a NoC; per core: instruction front-end, DMA engine, two double-buffered input buffers, 8 systolic arrays of 16x16, output buffer, vector unit (Sec. 4.2, p.5). 96 SAs at 2 GHz give 96 FP8 TFLOPS per cube and 1536 TFLOPS per chip (Sec. 7, p.9).
  - Design principles: many small SAs (P1), LLC-free (P2), two-level crossbar for query broadcast and partial-sum collection (P3) (Sec. 4.1, p.4-5).
  - Output-stationary dataflow, "split K just enough that every SA gets one tile" rule (Eq. 2-3), continuous tiling (Eq. 4) (Sec. 4.3-4.4, p.5-6).
  - Two-level hybrid parallelism: TP over KV heads across cube groups, CP over the sequence inside a group (Sec. 5.2, p.6-7); reordered collectives ReduceScatter -> local W_O -> Reduce (Sec. 6.1, p.8).
- **Simulation / implementation method**: ScaleSim for one cube, AstraSim for the 16-cube system with a fixed 15 ns per D2D hop; 8xH100 server measured with NVML; CACTI for the GPU power breakdown; Rubin numbers projected by scaling H100 measurements (Sec. 3.2, p.4; Sec. 7, p.9). Analytical/cycle-level only: no RTL, no P&R, no silicon.
- **Limitations**: No absolute logic-die area; PNM power modeled as a flat 15 W/cube (Table 1, p.9); no energy-per-bit for D2D; no NoC/crossbar bandwidth numbers; collective algorithm inside AstraSim not stated. Internal inconsistencies: SRAM "32 KB per core ... 3 MB per cube" only works if the 32 KB is per SA (Sec. 4.2, p.5); total HBM bandwidth 44 TB/s in Table 1 (p.9) vs 40 TB/s in the text (Sec. 7.1, p.9; Sec. 7.5, p.11); "Rubin's 8 TB/s" (Sec. 7.5, p.11) vs 22 TB/s in Table 1. Rubin is projected, not measured. MoE/FFN and prefill excluded; single package only; AMMA-to-LPU/GPU hand-off link not modeled; thermal limits of PNM under DRAM discussed only qualitatively (Sec. 2.2, p.3). MLA is compute-bound on the default per-cube budget (Sec. 7.1, p.9).
- **Numbers worth reusing**: see the constants table below. Most important: 2.75 TB/s per HBM4 cube and 15 W/cube added logic (Table 1, p.9); 15 ns per D2D hop and 1500 GB/s per UCIe 3.0 link (Table 1, p.9); communication speedup 2.7x / 17.7x / 65.4x at 8K / 256K / 1M from hybrid parallelism + reordered collectives (Sec. 7.3, p.10); latency essentially flat across D2D bandwidth 0.5-2.5 TB/s (Fig. 15, p.11).
- **Implications for SymFabric**: AMMA is the closest published template for the "HBM-PNM stacks + D2D mesh, no GPU" endpoint of the SymFabric design space. Its bandwidth-matched, LLC-free logic-die sizing, the small-M array organisation, the two-level partitioning and the REDUCE-after-COMBINE collective rewrite all transfer; its dense FP8 MAC arrays, dense output tiles and static partitioning do not (details in "Reusable for SymFabric").
- **Open source**: none stated.

## Inter-chiplet / HBM-to-HBM communication model

**Topology.** 16 HBM-PNM cubes form a 4x4 2D mesh; each cube has four D2D ports and links to up to four neighbours (Sec. 4.1, p.4; Sec. 4.2, p.5). Cubes talk directly, with no host GPU as hub (Sec. 3.3, p.4). Inside a cube a NoC joins the 12 cores, the memory controller and the four D2D ports (Sec. 4.2, p.5); a two-level crossbar (local within each 8-SA core, global across the 12 cores) broadcasts the query to all SAs and collects partial sums, at O(N sqrt N) area instead of O(N^2) for a flat crossbar (Sec. 4.1 P3, p.5). The paper flags that per-cube compute and D2D link bandwidth "compete for shared NoC resources" (Sec. 1 C4, p.2) but gives no NoC bandwidth.

**Link.** UCIe 3.0 D2D: latency 15 ns per hop = adapter 4 ns (TX+RX) + PHY 10 ns (TX+RX) + channel propagation 1 ns; bandwidth 1500 GB/s (Table 1, p.9; direction convention not stated). AstraSim models the inter-cube latency as a fixed 15 ns per hop (Sec. 7, p.9). Energy per bit: not reported. For contrast, the GPU baselines use NVLink C2C at 900 ns latency and 3600 GB/s (Rubin) / 900 GB/s (H100) dual-direction bandwidth (Table 1, p.9), and a measured single 8 B GPU-to-GPU transfer takes over 12,000 ns end-to-end (Sec. 3.3, p.4).

**Partitioning across stacks (two-level hybrid parallelism, Sec. 5.2, p.6-7; Fig. 8, p.7).**
- Level 1 (TP over KV heads): the 16 cubes are divided into cube groups, one per KV head; with 4 KV heads this is 4 groups of 4 cubes, each laid out as a 2x2 sub-mesh rather than a 1x4 strip to minimise the maximum hop count within a group (Sec. 5.2, p.6-7; derived, not stated: diameter 2 vs 3 hops). Each group stores its head's projection weights and KV-cache entries and handles all Q heads of that KV head; heads are independent through QKV projection and attention, so the only cross-group traffic is one reduction at the end of the output projection (Sec. 5.2, p.7).
- Level 2 (CP inside a group): the KV cache is split along the sequence dimension across the 4 cubes (per-cube slice Seq x H_KV/16, Fig. 8(b)-(c), p.7); each cube computes a D_h/4 slice of Q, the group AllGathers Q, each cube attends over its local shard, and the group combines partial results (Sec. 5.2, p.7; Fig. 8(b), p.7). W_O is sliced per cube to H_Q x D_h/16 (Fig. 8(c), p.7).
- Property: no cross-cube communication volume scales with sequence length (Sec. 5.2, p.7), unlike naive TP16 whose 16-way AllGather/AllReduce at every stage spans the full mesh diameter with volume proportional to the sequence (Sec. 5.1, p.6; Fig. 8(a), p.7).

**What is reduced and how (Sec. 6.1, p.8; Fig. 9, p.8).**
- Default flow under hybrid parallelism: intra-group AllReduce (= ReduceScatter + AllGather) of the attention output, per-cube multiply by the local W_O^{mn[yx]} shard, then cross-group AllReduce (Sec. 6.1, p.8).
- Reordered flow: (i) intra-group ReduceScatter only, so cube n keeps a distinct slice A^{mn}; (ii) W_O re-sliced from [yx] to [yy], i.e. the Level-2 partition moves from the output to the input dimension, so each cube computes A^{mn} W_O^{mn[yy]} locally with no weight duplication; (iii) the cross-group AllReduce becomes a point-to-point Reduce to the single destination cube that forwards to the FFN accelerator, "saving roughly half the traffic". Net effect: two AllGathers removed and one AllReduce downgraded to a Reduce, with per-cube compute and memory footprint unchanged (Sec. 6.1, p.8).
- Reduction semantics: the ReduceScatter must perform the softmax-weighted sum of Eq. 6-7, sum_n alpha_n a_n W_O = (sum_n alpha_n a_n) W_O with alpha_n = e^{m_n - m} l_n / l, m = max_n m_n, l = sum_n e^{m_n - m} l_n (Eq. 6, p.8; Eq. 7, p.9). Each cube piggybacks its (m_n, l_n) on the projected output, so no extra messages are needed (Sec. 6.2, p.9). Correctness rests on FlashAttention/Ring-Attention style merging plus the commutation of a per-row scalar with the linear map W_O (Sec. 6.2, p.8-9).

**Communication overhead reported.**
- Naive TP16 vs hybrid (HP) vs hybrid + reordered (HP_RO) on AMMA: communication-only speedup 2.7x / 17.7x / 65.4x at 8K / 256K / 1M for HP_RO and 1.3x / 10.0x / 36.8x for HP alone; end-to-end latency speedup 1.1x / 1.5x / 1.6x (HP_RO) and 1.1x / 1.4x / 1.5x (HP) (Sec. 7.3, p.10; Fig. 12(a)-(b), p.10). RO's gain is a fixed latency reduction, so it matters most at short sequences (Sec. 7.3, p.10).
- Per-layer breakdown (Qwen3, Seq = 8K and 128K, BS = 1 and 4): communication-share labels on the Rubin / Rubin TP2 / TP16 / HP / HP+RO bars range from 2% to 43% (Fig. 13, p.11).
- Design-space sweep (Qwen3-235B, compute 8-256 TFLOPS/cube, D2D 0.5-2.5 TB/s): latency is almost independent of D2D bandwidth, e.g. 4.2 at 96 TFLOPS and 35 at 8 TFLOPS for all five bandwidth points at BS = 1, and 59 -> 58 at 96 TFLOPS from 0.5 to 2.5 TB/s at BS = 32 (Fig. 15, p.11; units microseconds, inferred from the axis label). The paper's explanation: the collectives are already local, and at batch 1-32 the D2D volumes are small so latency is dominated by fixed start-up delay rather than transmission time; compute is the more critical knob, with no gain beyond 96 TFLOPS/cube (Sec. 7.6, p.11). It suggests reallocating D2D area/power to compute (Sec. 7.1, p.9) or to higher HBM frequency (Sec. 7.6, p.11).
- GPU-centric counter-example: with prior PIM (AttAcc-style, 8 GPUs + 64 PIM devices over NVLink) the collective overhead is 1.5x the attention compute at 64K and 24x once compute is no longer the bottleneck, even assuming an ideal 900 ns NVLink latency (Sec. 3.3, p.4); total prior-PIM attention latency is MHA 0.8 / 3.1 / 12.3 and GQA 5.4 / 7.7 / 17.0 microseconds at 16K / 64K / 256K (Fig. 5, p.4). Rubin TP2's NVLink overhead is only amortised at 1M, where AMMA's lead narrows to 1.1x (Sec. 7.1, p.9).
- Not given: per-hop bandwidth on multi-hop routes, router/NoC latency, link width, the collective algorithm, D2D pJ/bit, and any inter-package (chip-to-chip) HBM-to-HBM model.

## Hardware constants (for reuse with source)

| quantity | value | unit | location | conditions |
|---|---|---|---|---|
| HBM generation | HBM4 + PNM | - | Table 1, p.9 | baselines: H100 HBM3, Rubin HBM4, NeuPIM HBM4+PIM |
| Per-cube (per-stack) HBM bandwidth | 2.75 | TB/s | Table 1, p.9 | HBM4; same value for Rubin cubes |
| Number of HBM cubes per chip | 16 | - | Sec. 1, p.2; Sec. 4.1, p.4; Table 1, p.9 | 4x4 2D mesh; H100 has 5 HBM3e, Rubin 8 HBM4 (Sec. 7.1, p.9) |
| Total HBM bandwidth | 44 | TB/s | Table 1, p.9 | text uses 40 TB/s, "11.9x H100" (Sec. 7.1, p.9; Sec. 7.5, p.11) |
| Chip compute | 1536 | FP8 TFLOPS | Table 1, p.9; Sec. 7, p.9 | 16 cubes; ~10% of Rubin (Sec. 7.2, p.10) |
| Per-cube compute | 96 | FP8 TFLOPS | Sec. 7, p.9 | 96 SAs of 16x16 at 2 GHz |
| Clock | 2 | GHz | Sec. 7, p.9 | |
| Cores per cube | 12 | - | Sec. 4.2, p.5 | plus memory controller and 4 D2D ports on NoC |
| SAs per core / per cube | 8 / 96 | - | Sec. 4.1, p.4; Sec. 4.2, p.5 | |
| SA size | 16x16 | PEs | Sec. 4.1, p.4 | output-stationary (Sec. 4.3, p.5) |
| PEs per cube | 24,576 | - | derived: 96 x 16 x 16 | not stated in paper |
| Logic-die process | <=5 | nm | Sec. 1, p.2; Fig. 1(b), p.2; Sec. 2.2, p.3 | HBM4 and later; HBM3 base die >=12 nm (Fig. 1(b), p.2); Duplex ~28 nm (Sec. 3.3, p.4) |
| Logic-die area | not reported | - | - | only relative: Rubin compute dies = 67% of package area, 73% of power (Sec. 1, p.1) |
| Added PNM logic power | 15 / 240 | W per cube / W per chip | Table 1, p.9 | "15W/Cube and 15Wx16=240W/Chip" |
| HBM cube + PHY power | 75 (HBM4) / 40 (HBM3) | W per cube | Table 1, p.9 | H100 compute die 500 W, Rubin compute die 800 W |
| Chip TDP | 1440 | W | Table 1, p.9 | = 16 x (75 + 15) (derived); H100 700, Rubin 2200, NeuPIM 1600/1046 |
| SRAM per core | 32 | KB | Sec. 4.2, p.5 | 2 x 6.4 KB Input Buf A + 2 x 6.4 KB Distributed Input Buf B + 6.4 KB Output Buf; double-buffered |
| SRAM per cube / per chip | 3 / 48 | MB | Sec. 4.2, p.5 | as printed; consistent only if 32 KB is per SA (96 x 32 KB) |
| SRAM vs GPUs | >2x smaller than H100, >6x smaller than Rubin | - | Sec. 4.2, p.5 | |
| No LLC | reclaims 20 | % of power budget | Sec. 4.1 P2, p.5 | |
| SA operand reuse | 16x lower SRAM read bandwidth | - | Sec. 4.1 P1, p.5 | vs vector units |
| HBM access latency | 200 | ns | Sec. 4.2, p.5 | hidden by double-buffered DMA prefetch |
| SA fill/drain overhead | 15 per end, 30 per tile | cycles | Sec. 4.4, p.5 | M_SA - 1 = 15; utilisation Eq. 2-4 |
| Continuous-tiling gain | 52% -> 67% (n = 2), 81% (n = 4) | per-SA efficiency | Sec. 4.4, p.6 | N = 3072, k = 32 |
| D2D protocol | UCIe 3.0 | - | Table 1, p.9; Sec. 7, p.9 | |
| D2D latency per hop | 15 | ns | Table 1, p.9; Sec. 7, p.9 | adapter 4 ns (TX+RX) + PHY 10 ns (TX+RX) + channel 1 ns |
| D2D link bandwidth | 1500 | GB/s | Table 1, p.9 | direction convention not stated |
| D2D energy per bit | not reported | - | - | |
| D2D bandwidth sweep | 0.5-2.5 | TB/s | Sec. 7.6, p.11; Fig. 15, p.11 | |
| Compute sweep | 8-256 | TFLOPS per cube | Sec. 7.6, p.11 | plus 512 TFLOPS variant "Ours 512T" (Sec. 7.1, p.9; Fig. 10, p.10) |
| NVLink C2C (baselines) | 900 ns; 3600 (Rubin) / 900 (H100) | ns; GB/s dual-direction | Table 1, p.9 | |
| Measured GPU 8 B transfer | >12,000 | ns | Sec. 3.3, p.4 | end-to-end on real GPU |
| H100 profile | 693 W avg; 700 W TDP; 125 W static; <200 W HBM; ~370 W NoC/LLC/SM; 130 W LLC | W | Sec. 3.2, p.4 | Qwen3-235B attention, batch 1 |
| H100 utilisation | >90 (HBM BW) / <5 (compute) | % | Sec. 3.1, p.3; Fig. 3, p.3 | |
| Arithmetic intensity | 32 (GQA decode) / 795 (Rubin) | FLOPs per byte | Sec. 1, p.1; Sec. 2.1, p.3 | Qwen3 G = 16, FP8; MLA ~8x GQA (Sec. 7.1, p.9) |
| NeuPIM bank-PIM bandwidth | 198 total / 24.75 per cube | TB/s | Table 1, p.9 | 9x HBM interface (Sec. 7, p.9); 4.5x AMMA (Sec. 7.1, p.9) |
| Latency vs H100 | 15.5x | - | Abstract, p.1 | 12.0-16.3x at BS = 1, 13-20x at BS = 32 (Sec. 7.1, p.9) |
| Latency vs Rubin | 1.8-2.5x | - | Sec. 7.1, p.9 | vs Rubin TP2 1.5-2.4x, 1.1x at 1M |
| Latency vs NeuPIMs | 3.4x / 1.4x | - | Sec. 7.1, p.9 | Qwen3-235B / Llama4-Maverick |
| MLA (DeepSeek-V3) | 1.9x faster at 4K; up to 2.9x slower at long seq; 1.8-2.1x faster with 512 TFLOPS/cube | vs Rubin | Sec. 7.1, p.9 | |
| Energy efficiency | 6.9x | Token/J vs H100 | Abstract, p.1 | 5.6-6.6x (Sec. 7.2, p.9); 2.6-3.1x vs Rubin; 2.8-4.8x vs Rubin TP2 (Sec. 7.2, p.10) |
| Chip power in evaluation | <1500 | W | Sec. 7.2, p.10 | 1.4x below Rubin, 2.6x below Rubin TP2 |
| Ablation, latency | 1.1 / 1.5 / 1.6x | vs TP16 | Sec. 7.3, p.10; Fig. 12(a) | HP_RO at 8K / 256K / 1M; HP 1.1 / 1.4 / 1.5x |
| Ablation, communication only | 2.7 / 17.7 / 65.4x | vs TP16 | Sec. 7.3, p.10; Fig. 12(b) | HP_RO; HP alone 1.3 / 10.0 / 36.8x |
| Throughput vs batch | 0.223 -> 0.478 | tok/us | Sec. 7.4, p.10; Fig. 14, p.11 | Qwen3-235B, seq 64K, BS 1 -> 32 (2.14x); latency 30x worse; compute saturates at BS = 16 |
| Throughput vs H100 | 14-16x | - | Sec. 7.4, p.11; Fig. 12(c), p.10 | at comparable or lower power |
| Time breakdown | 85 (proj) / 15 (attn) | % | Sec. 7.5, p.11 | BS = 1, 8K; at 128K attention is 60% (BS = 1), 86% (BS = 4) |
| Per-op speedup vs Rubin | 1.76 / 2.57 / 3.89x | - | Sec. 7.5, p.11 | Proj QKV / Proj O / attention |
| Fig. 15 latency vs D2D BW | 4.2 (96T) and 35 (8T) flat over 0.5-2.5 TB/s at BS = 1; 59 -> 58 (96T) at BS = 32 | us (inferred) | Fig. 15, p.11 | Qwen3-235B |
| Prior-PIM attention latency | MHA 0.8 / 3.1 / 12.3; GQA 5.4 / 7.7 / 17.0 | us | Fig. 5, p.4 | 16K / 64K / 256K, 64 Q heads; comm 1.5x compute at 64K, 24x with more compute (Sec. 3.3, p.4) |

## Reusable for SymFabric

**Would host symbolic operators well.**
- The sizing philosophy: compute matched to raw HBM bandwidth, no LLC, tiny double-buffered SRAM with DMA streaming (Sec. 4.1 P1-P2, Sec. 4.2, p.4-5). Symbolic scan / match / set operators are single-pass streams over relations with reuse bounded by the number of concurrent queries Q, exactly AMMA's small-M regime (M = 1-32). Eq. 2-4 (Sec. 4.4, p.5-6) is a ready-made utilisation model with M_SA replaced by SymFabric's Q dimension.
- Many small output-stationary arrays with a fixed output tile: for a semiring product between Q query vectors and a long relation (Boolean and/or for match, min-plus for shortest path, counting semiring for aggregates) the streamed dimension is the relation, and the fixed 16x16 output tile keeps the cross-array collection cost independent of relation length (Sec. 4.3, p.5). This is SymFabric's COMBINE + REDUCE if the PE is made semiring-generic.
- The two-level crossbar (query broadcast in, partial sums out, Sec. 4.1 P3, p.5) is exactly the SELECT/broadcast plus REDUCE fabric a systolic symbolic core needs; the vector unit on the output buffer (Sec. 4.2, p.5) is the natural home for FILTER / UPDATE post-processing (thresholding, normalisation, top-k preparation).
- Two-level hybrid partitioning (Sec. 5.2, p.6-7): Level 1 by an independent key (relation, predicate, head) so no communication until the final merge; Level 2 by the streamed dimension (tuples) so each group does one gather and one reduce whose volume does not grow with relation size. This is a sound default multi-stack layout for SymFabric, and the 2x2 sub-mesh-per-group rule bounds hop count.
- The reordered collective (Sec. 6.1, p.8) is an algebraic rewrite: when the subsequent COMBINE is linear in the reduced operand and commutes with a per-row scalar (Eq. 7, p.9), push the projection before the cross-stack REDUCE, turning AllReduce + AllGather into ReduceScatter + Reduce. In semiring terms this is distributivity of (x) over (+), so SymFabric's operator algebra can apply it generically. The softmax merge (Eq. 6, p.8) with piggybacked (m_n, l_n) statistics is a template for carrying reduction metadata across stacks.
- Constants for the SymFabric cost model: 15 ns per hop and 1500 GB/s per D2D link, 200 ns HBM latency, 2.75 TB/s per stack, 15 W/cube added logic, 75 W/cube HBM4 (Table 1, p.9; Sec. 4.2, p.5). The DSE conclusion that at small Q hop count and synchronisation rounds dominate, not link width (Sec. 7.6, p.11), directly informs whether SymFabric needs wide D2D or just few collective rounds. The ScaleSim + AstraSim methodology (Sec. 7, p.9) is reusable.

**Would not fit without redesign.**
- PEs are FP8 multiply-accumulate; symbolic semirings need compare / select / AND / OR / min / max datapaths and often integer or bit-vector operands. The SA has fixed-latency control with no data-dependent early exit.
- Dense fixed 16x16 output tiles assume dense output. FILTER, set operations and match produce sparse, variable-cardinality output; a stream-compaction stage and output-buffer management are absent.
- No INDEX support: the DMA streams contiguous KV tiles (Sec. 4.2, p.5); there is no gather/scatter, hashing or tree lookup, and 3 MB of SRAM per cube (Sec. 4.2, p.5, as printed) cannot hold indices or hash tables. SymFabric's INDEX operator needs a different memory-side engine.
- UPDATE is limited to KV-cache append; random writes to relations are not addressed.
- Communication assumes static, workload-independent partitions (KV heads, sequence). Symbolic joins need data-dependent repartitioning (all-to-all shuffles across the mesh), where the multi-hop fixed 15 ns per hop and NoC contention matter far more than in AMMA's neighbour-confined collectives; Fig. 15's flat latency vs D2D bandwidth (p.11) would not carry over.
- Single package only: no inter-package HBM-to-HBM link, no AMMA-to-LPU/GPU hand-off model, no D2D pJ/bit, no ISA or compiler description beyond an "instruction front-end" (Sec. 4.2, p.5).
