# piPE-SA: Unlocking Systolic Array Efficiency via Deeply Pipelined Processing Elements (2026-submission)

Venue and year are not printed in the text; the newest reference access dates are "Mar. 2026" ([27], p. 7) and "2026-04-11" ([10], p. 7), so it is filed as a 2026 submission. Own-group systolic-array PE paper. Page numbers refer to the 7-page pdftotext layout; "Sec. IV-B.1/2/3" are the numbered paragraphs "Max Frequency", "Area and Power Efficiency", "Energy per FLOP".

- **Workload**: Dense GEMM on systolic arrays (SAs) for LLM training/inference with BF16/FP16 inputs and FP32 accumulation (Abstract; Sec. I, p. 1). Nine BF16 GEMM shapes from LLaMA-2/3, GPT-3, PaLM, BERT, OPT, GPT-Neo, Mistral (attention QK^T / PV, output and QKV projections, FFN) with arithmetic intensity (AI) from 62 to 1024 FLOP/B (Table IV, p. 4). Cycle validation uses square INT8/INT32 GEMMs of size 8 to 64 in steps of 8 (Sec. IV-A, p. 4).
- **Operators**: One hotspot operator, the fused multiply-add C += A*B: COMBINE (x) fused with REDUCE (+) over the ordinary (+,x) semiring, BF16 inputs, FP32 accumulator (Fig. 2(c), p. 2). Iteration structure is a 2-D wavefront: Weight-Stationary (WS) holds one operand in the PE and reduces down a column through the partial-sum chain; Output-Stationary (OS) reduces in place through an accumulator feedback loop (Sec. II-A, Fig. 2(b), p. 2). No INDEX / FILTER / SELECT / UPDATE (dense, static schedule). The reduction order inside each output is preserved (Sec. III-A, p. 3), which matters for non-associative FP (+) but is unnecessary for idempotent semirings.
- **Data scale**: r x r tiles on arrays of r = 8 (implemented), 32 (1,024 PEs, taken to GDS) and 128 (simulated) (Sec. IV-A, p. 4; Sec. IV-E, p. 6); n = ceil(N/r)ceil(K/r) tiles for WS and ceil(M/r)ceil(N/r) for OS (Table II, p. 2); 320 GB/s memory bandwidth in the 128 x 128 model (Sec. IV-A, p. 4). The SymFabric "concurrent queries Q" axis appears as the number of independent output tiles in flight: OS needs at least s independent tiles to fill its s accumulation slots (Sec. III-A, p. 3; Sec. III-C, p. 4). Dense; no sparsity.
- **Hardware mechanism**: (1) Pipeline the FMA inside the PE into s = 1..5 stages; the pipelined FMA still accepts one new independent operation per cycle (Sec. III, p. 3). (2) WS: replace the one-cycle boundary skew with s-cycle shift registers per row so activations meet partial sums that now arrive s cycles later; horizontal forwarding and output alignment registers unchanged; same method for Input-Stationary (Sec. III-B, Fig. 4, p. 3-4). (3) OS: interleave s independent output tiles across s accumulation slots (the accumulator plus the s-1 feedback-path pipeline registers); slot j = t mod s owns the issue port in cycle t; tile launches are spaced >= r cycles so the single drain path (one row per cycle, r cycles per tile) never carries two tiles (Sec. III-A, Fig. 3, Algorithm 1, p. 3-4). (4) Treat (array size r, depth s, frequency f) as a joint design space (Sec. IV-C, p. 5-6).
- **Simulation / implementation method**: RTL. BF16xBF16+FP32 PE built on the FPnew FMA [13]; 8 x 8 WS array with its input skew registers in TSMC N28 HPC+; Genus sweep of about 100 timing targets per depth; Innovus place-and-route of selected points; all five 32 x 32 arrays taken to GDS; Joules gate-level power under GEMM switching activity, one FMA = 2 FLOPs (Sec. IV-A, p. 4). Cycle-level validation: piPE-SA integrated into Gemmini (native WS and buffered OS backend, INT8/INT32), Verilator, 8 x 8, s = 1-5 (Sec. IV-A, p. 4; Sec. IV-D, p. 6). Roofline: COCOSSim + DRAMSim3, 128 x 128 WS array, frequencies extrapolated from post-route results (Sec. IV-A, p. 4; Sec. IV-E, p. 6).
- **Limitations**: Only the (+,x) semiring and a single FP format per PE. OS array PPA is not measured; it is asserted to follow WS because OS PEs "differ only at the periphery" (Sec. IV-A, p. 4). 128 x 128 frequencies are extrapolated (Sec. IV-E, p. 6); the DSE scales 8 x 8 area and power linearly with PE count (Sec. IV-C, p. 5). Absolute array area/power exist only as curves in Figs. 5-6 (not readable in the text). No statement of where the pipeline registers sit inside the FMA, no register/wire/clock area split, no supply voltage or corner. Algorithm 1 assumes a uniform K per tile (p. 4); OS cycle formulas assume K >= r (Sec. III-C, p. 4).
- **Numbers worth reusing**: see "PE timing and area table" (Table I, p. 1) and "Hardware constants" below. Headline: post-route 8 x 8 f_max 0.76 / 1.39 / 1.75 / 2.26 / 2.62 GHz for s = 1..5, 3.45x at s = 5 (Sec. IV-B.1, Fig. 6(a), p. 4-5); best operating point of s = 4 gives 1.72x GFLOPS/mm2 and 1.38x GFLOPS/W over s = 1, s = 2 gives 1.43x and 1.30x (Sec. IV-B.2, p. 5); minimum energy of every depth 2.1-2.4 pJ/FLOP (Sec. IV-B.3, p. 5).
- **Implications for SymFabric**: see "Reusable for SymFabric" below.
- **Open source**: none stated for piPE-SA itself. Building blocks are open: FPnew [13], Gemmini [5], COCOSSim [17], DRAMSim3 [18] (References, p. 7).

## PE timing and area table

Table I (p. 1), "Combinational MAC/FMA critical-path delay and area at minimum delay in TSMC N28". All cells are legible in the text extraction. There is no FP32-input row; FP32 appears only as the accumulation type. Each row is a single combinational (unpipelined) multiply-accumulate unit synthesized at its minimum achievable delay, so the area column is the aggressive-timing end of the area-vs-delay curve (for comparison, the same BF16 FMA is 2,149 um2 when synthesized at relaxed timing, Sec. I / Fig. 1(a), p. 1).

| Datatype (inputs) | Accumulation | Min delay (ps) | Area at min delay (um2) | Location |
|---|---|---|---|---|
| INT8  | INT32 | 260   | 1,199 | Table I, p. 1 |
| INT16 | INT32 | 326   | 3,085 | Table I, p. 1 |
| FP16  | FP32  | 938   | 6,991 | Table I, p. 1 |
| BF16  | FP32  | 1,034 | 7,013 | Table I, p. 1 |

Text framing of the same data: BF16/FP16 FMAs with FP32 accumulation have minimum critical paths "exceeding 930 ps", "3-4x longer" than INT8/INT16 MACs at 260-326 ps synthesized with the same flow (Sec. I, p. 1). The BF16->FP32 FMA in Fig. 2(c) (p. 2) is labelled with an 8-bit mantissa multiplier, an exponent-difference unit, a 68-bit alignment shifter and a 44-bit adder, then normalization and rounding; the highlighted critical path runs through exponent alignment, mantissa addition, normalization and rounding, i.e. through the accumulation side, not the multiplier (Sec. II-A, p. 2; Sec. II-B, p. 3).

## Pipelining method

- Depth: s = 1 (baseline) to 5 stages inside the FMA; s equals the PE latency in cycles and the pipelined FMA accepts a new independent operation every cycle (Sec. III, p. 3; Sec. IV-A, p. 4). The register positions inside the FPnew datapath are not stated.
- Why the classical schedule breaks: in WS the one-cycle inter-row skew pairs activations with partial sums of a different output vector once partial sums arrive s cycles late; in OS the accumulator feedback becomes an s-cycle loop-carried dependency, so one output can be updated only every s cycles; pipelining only the multiplier does not help because the critical path is in the accumulation logic (Sec. II-B, Fig. 2(c), p. 2-3). Retiming cannot create stages inside the OS feedback loop (Sec. V, p. 6).
- WS fix: s-cycle shift registers per row at the left boundary; output alignment registers and II unchanged; only input skew grows with s (Sec. III-B, Fig. 4, p. 3-4).
- OS fix: s accumulation slots per PE, round-robin issue (slot j = t mod s); results copied out before a slot is reused; launches spaced >= r cycles so drains never overlap and output-buffer bandwidth is unchanged (Sec. III-A, Fig. 3(c), Algorithm 1, p. 3-4; Sec. V, p. 6).
- Cycle consequences (Table III, p. 4; M, N output dims, K reduction length, r array dim):

| Dataflow | Metric | s = 1 | s stages | Location |
|---|---|---|---|---|
| OS | tile latency | K + 3r - 2 | sK + 3r - 2 | Table III, p. 4 |
| OS | II | K | sK (same slot); >= r (between slots) | Table III, p. 4 |
| WS | tile latency | M + 3r - 2 | M + (s+2)r - 2 | Table III, p. 4 |
| WS | II | M | M | Table III, p. 4 |

- Throughput is preserved: in steady state one OS tile still starts every K cycles; WS keeps II = M and only adds startup latency (Sec. III-C, p. 4). Totals: T_WS(s) = L_WS(s) + (n-1)M; T_OS(s) = L_OS(s) + (s-1)r + sK(ceil(n/s)-1) for K >= r, otherwise r replaces K (Sec. III-C, p. 4). Fixed startup cost (s-1)r cycles for WS and (s-1)(K+r) for OS; speedup S(s) = f(s)/f(1) * T(1)/T(s), which tends to f(s)/f(1) for many tiles (Eq. 1, Sec. III-C, p. 4). OS additionally needs enough independent tiles to fill s slots and enough output bandwidth (Sec. III-C, p. 4).
- Validation: Verilator-measured Gemmini cycles match Table III exactly for every size and depth, WS and OS; WS startup overhead below 6% at GEMM size 32 and below 1% at 64; OS overhead within 3.5% at 64 and non-monotonic in s because s*ceil(n/s) - n slot positions go unused; wall-clock speedup reaches 3.4x at s = 5 for both dataflows (Sec. IV-D, Fig. 8, p. 6).
- Frequency: 8 x 8 post-route 0.76 / 1.39 / 1.75 / 2.26 / 2.62 GHz for s = 1..5, 3.45x at s = 5 and clearly sub-linear; synthesis predicts 0.79-2.83 GHz, so P&R plus clock tree cost 4-10% (Sec. IV-B.1, Fig. 6(a), p. 4-5). 32 x 32 post-route 0.71 / 1.30 / 1.70 / 1.95 / 2.32 GHz, 86-97% of the 8 x 8 values at 16x the PE count, 3.27x at s = 5 (Sec. IV-E, p. 6).
- Area and power: at aggressive timing, pipelining avoids the logic-optimization bloat; at a relaxed 2-2.5 ns period s = 5 costs about 40% more area than s = 1 because register overhead remains (Sec. IV-B.2, Fig. 5, p. 5). Best operating points: s = 4 gives 1.72x GFLOPS/mm2 and 1.38x GFLOPS/W; s = 2 gives 1.43x and 1.30x (Sec. IV-B.2, p. 5). Energy: the minimum of every depth lies within 2.1-2.4 pJ/FLOP across the sweep (a 2-stage pipeline lowers the minimum by 13%, Abstract, p. 1); at each depth's max-frequency post-route point energy is 6.0 (s=1), 3.4 (s=3), 3.8 (s=4), 5.0 (s=5) pJ/FLOP as the register-plus-clock share of power grows from 6% to 67%; depths must be compared at the same throughput target (Sec. IV-B.3, Fig. 6(c,d), p. 5).
- DSE: with r in {4, 8, 16, 32}, the power-optimal map mostly selects s = 4 and the area-optimal map mostly s = 2; single-stage PEs are rarely optimal; pipelining does not always reduce array size (Sec. IV-C, Fig. 7, p. 5-6).
- System effect: 128 x 128 compute ceiling 21.6 TFLOPS (s=1, 0.66 GHz) to 67.4 TFLOPS (s=5, 2.06 GHz), ridge point ~68 to ~211 FLOP/B; a memory-bound kernel (AI 62) gains only ~1.10x, transitional kernels (AI 114 / 205) ~1.3x / ~1.9x, compute-bound kernels (AI > 680) 2.6-2.8x against a 3.1x frequency ratio (Sec. IV-E, Fig. 9, p. 6).

## Hardware constants (for reuse with source)

| quantity | value | unit | location | conditions |
|---|---|---|---|---|
| INT8 x INT8 + INT32 MAC, min combinational delay / area | 260 / 1,199 | ps / um2 | Table I, p. 1 | TSMC N28, unpipelined, synthesized at minimum delay |
| INT16 x INT16 + INT32 MAC, min delay / area | 326 / 3,085 | ps / um2 | Table I, p. 1 | same |
| FP16 x FP16 + FP32 FMA, min delay / area | 938 / 6,991 | ps / um2 | Table I, p. 1 | same |
| BF16 x BF16 + FP32 FMA, min delay / area | 1,034 / 7,013 | ps / um2 | Table I, p. 1 | same; equals the 1.03 ns point of Fig. 1 |
| BF16 FMA area at relaxed timing | 2,149 | um2 | Sec. I, Fig. 1(a), p. 1 | combinational FMA alone; relaxed end of a ~1-5 ns period sweep (Fig. 1 x-axis) |
| BF16 FMA area inflation, relaxed -> 1.03 ns | 3.26 | x | Sec. I, p. 1 | combinational FMA alone |
| BF16 FMA energy per FLOP, relaxed -> 1.03 ns | 3.2 -> 14.6 | pJ/FLOP | Sec. I, Fig. 1(b), p. 1 | combinational FMA alone |
| Per-stage slack of a 2-stage FMA at 1.03 ns | ~50 | % of period | Sec. I, p. 1 | stated estimate |
| BF16 FMA datapath widths | 8 / 68 / 44 | bits (mantissa mult / align shifter / adder) | Fig. 2(c), p. 2 | figure labels only |
| Post-route f_max, 8 x 8 WS array, s = 1..5 | 0.76 / 1.39 / 1.75 / 2.26 / 2.62 | GHz | Sec. IV-B.1, Fig. 6(a), p. 4-5 | N28 HPC+, FPnew BF16 FMA, incl. input skew registers, Innovus |
| Synthesis f_max range, s = 1..5 | 0.79-2.83 | GHz | Sec. IV-B.1, p. 5 | Genus |
| Frequency lost to P&R and clock tree | 4-10 | % | Sec. IV-B.1, p. 5 | 8 x 8 |
| Frequency gain s = 5 vs s = 1 | 3.45 (8 x 8) / 3.27 (32 x 32) | x | Sec. IV-B.1, p. 4-5; Sec. VI, p. 6; Sec. IV-E, p. 6 | post-route |
| Post-route f_max, 32 x 32 (1,024 PEs), s = 1..5 | 0.71 / 1.30 / 1.70 / 1.95 / 2.32 | GHz | Sec. IV-E, p. 6 | taken to GDS; 86-97% of 8 x 8 values |
| Extrapolated f, 128 x 128, s = 1..5 | 0.66 / 1.21 / 1.65 / 1.68 / 2.06 | GHz | Sec. IV-E, p. 6 | extrapolation, not implemented |
| Area overhead of s = 5 vs s = 1 at relaxed timing | ~40 | % | Sec. IV-B.2, p. 5 | 8 x 8 WS synthesis, 2-2.5 ns period |
| Best-point efficiency gain, s = 4 vs s = 1 | 1.72 / 1.38 | x GFLOPS/mm2 / x GFLOPS/W | Sec. IV-B.2, p. 5; Abstract | each depth at its own best operating point |
| Best-point efficiency gain, s = 2 vs s = 1 | 1.43 / 1.30 | x GFLOPS/mm2 / x GFLOPS/W | Sec. IV-B.2, p. 5 | same |
| Minimum energy per FLOP, any depth | 2.1-2.4 | pJ/FLOP | Sec. IV-B.3, p. 5 | 8 x 8 WS synthesis sweep, Joules, GEMM switching activity, 1 FMA = 2 FLOP |
| Reduction of minimum energy per FLOP, s = 2 vs s = 1 | 13 | % | Abstract, p. 1 | same sweep |
| Energy per FLOP at max-frequency post-route points, s = 1 / 3 / 4 / 5 | 6.0 / 3.4 / 3.8 / 5.0 | pJ/FLOP | Sec. IV-B.3, Fig. 6(d), p. 5 | s = 2 value not given in text |
| Register + clock share of power, s = 1 -> s = 5 | 6 -> 67 | % | Sec. IV-B.3, Fig. 6(c), p. 5 | at max-frequency post-route points |
| 128 x 128 compute ceiling, s = 1 / s = 5 | 21.6 / 67.4 | TFLOPS | Sec. IV-E, p. 6 | 0.66 / 2.06 GHz, 320 GB/s (Sec. IV-A, p. 4) |
| Roofline ridge point, s = 1 -> s = 5 | ~68 -> ~211 | FLOP/B | Sec. IV-E, p. 6 | same model |
| Pipelining startup cost | (s-1)r (WS) / (s-1)(K+r) (OS) | cycles | Sec. III-C, p. 4 | per GEMM, K >= r for OS |
| Synthesis sweep density | ~100 | timing targets per depth | Sec. IV-A, p. 4 | Genus |

## Reusable for SymFabric

What the numbers imply for a PE that switches among (+,x), (max,+), (min,+), (OR,AND) and log-sum-exp (LSE):

- The (+,x) FP mode sets the single-cycle floor. Any mode-switchable PE that includes an FP16/BF16 (+,x) path inherits a combinational critical path of 938-1,034 ps in N28 (Table I, p. 1), about 3-4x the INT8/INT16 MAC (Sec. I, p. 1), and that path is the FP accumulate (align, add, normalize, round), not the multiplier (Sec. II-B, p. 3). Pipelining rather than logic pushing is the right lever: the same FMA is 2,149 um2 relaxed vs 7,013 um2 at 1.03 ns and 3.2 vs 14.6 pJ/FLOP (Sec. I, Fig. 1, p. 1).
- Tropical modes move the expensive part out of the loop. In (max,+)/(min,+) the "multiply" is an adder and the "accumulate" is a compare-select. For FP operands the alignment/normalization/rounding cost moves into the feed-forward combine stage, and the OS feedback loop holds only a comparator and mux, so the loop-carried hazard that forces slot interleaving in the paper (Sec. II-B, p. 3) is much milder; the registers can go in the feed-forward path, where the cheap WS skew fix (Sec. III-B, p. 3-4) is all that is needed. For integer operands the whole PE is adder plus comparator, in the regime of the INT MAC rows (260-326 ps, Table I, p. 1) or below. The paper has no comparator or max/min numbers, so this is qualitative.
- Idempotent modes can drop the order constraint. piPE-SA preserves the reduction order inside each output (Sec. III-A, p. 3) because FP (+) is non-associative. max, min and OR are exact and order-free, so their accumulations can be split across slots, tree-reduced, or merged late without changing results, giving SymFabric scheduling freedom the paper never needed.
- (OR,AND) is register-bound, not logic-bound. A 1-bit AND/OR datapath is far below the register-and-clock overhead the paper measures (register plus clock share rising from 6% to 67% of power from s = 1 to s = 5, Sec. IV-B.3, p. 5), so pipelining the Boolean datapath is pointless; its speed will be set by the shared PE's slowest mode or by array wiring. The natural design is to bit-slice many Boolean lanes into the FP-width datapath, but the paper gives no packing or SIMD-lane numbers.
- LSE is the worst case and the strongest argument for the OS slot machinery. LSE(a,b) = max(a,b) + log1p(exp(-|a-b|)) puts an FP subtract, a compare, a transcendental (LUT or polynomial) and an FP add into the accumulate loop, deeper than the FMA's accumulate. Sustaining one issue per cycle then requires s_LSE independent accumulations in flight, exactly the paper's s slots (Sec. III-A, p. 3), which in SymFabric terms means Q >= s_LSE independent queries or output tiles; WS needs s_LSE-deep skew registers per row (Sec. III-B, p. 3-4). Table III (p. 4) gives latency and II as closed forms in s and can be reused directly with s set per mode.
- Make s a per-mode parameter. The WS skew depth and the OS slot count are both parameterized by s, and Gemmini cycle counts match the formulas exactly for s = 1-5 (Sec. IV-D, p. 6), so one controller can serve several semirings with a mode-dependent s. Physical register positions are fixed, so cheap modes either pay the FP mode's latency or need bypass muxes; the paper does not quantify bypass cost.
- Do not over-pipeline. Frequency gain is sub-linear (3.45x at s = 5 for 8 x 8, 3.27x for 32 x 32; Sec. IV-B.1, p. 4-5; Sec. IV-E, p. 6), energy at f_max is minimized at s = 3 (3.4 pJ/FLOP) and rises again at s = 4-5 (Sec. IV-B.3, p. 5), s = 5 costs ~40% more area than s = 1 at a relaxed clock (Sec. IV-B.2, p. 5), and the DSE prefers s = 4 for power and s = 2 for area (Sec. IV-C, p. 5-6). A programmable-semiring PE should target 2-4 stages and compare modes at iso-throughput, as the paper does (Sec. IV-B.3, p. 5).
- Short reductions amortize badly. Startup cost is (s-1)r for WS and (s-1)(K+r) for OS, and the OS formulas assume K >= r (Sec. III-C, p. 4); overhead is largest for small GEMMs (Sec. IV-D, p. 6). Symbolic workloads with short or irregular reduction lengths and few tiles are exactly the regime where deep PE pipelines pay least, and the roofline ridge moving from ~68 to ~211 FLOP/B (Sec. IV-E, p. 6) says the operand-fetch side must scale with PE frequency.
- Methodology to copy: Genus sweep of ~100 timing targets per depth, Innovus on selected points, Joules power under realistic switching activity, 32 x 32 to GDS to confirm the frequency gain survives routing and clock distribution (Sec. IV-A, p. 4); report area-vs-period curves per mode (Fig. 1, Fig. 5) and pick the knee.

What the paper does not tell us:

- No timing, area or power for any operator other than integer MAC and FP FMA: no comparator, max/min, LUT/exp/log, Boolean, or mode-select mux numbers; no cost of an operator-selectable datapath.
- No FP32 x FP32 FMA row (FP32 is only the accumulation type, Table I, p. 1); no INT8 array-level PPA (INT8 appears only in Table I and in the Gemmini cycle validation, Sec. IV-A, p. 4).
- No statement of where the s-1 register sets sit inside the FPnew FMA or how the ~1 ns path splits per stage; no PE-level area or power for pipelined PEs separate from the 8 x 8 array; no register / wire / clock-tree area split; absolute array area and power exist only as unreadable curves (Figs. 5-6, p. 5).
- No supply voltage, corner, threshold flavor or temperature beyond "TSMC N28 HPC+" (Sec. IV-A, p. 4).
- No OS array measurements (asserted to follow WS trends, Sec. IV-A, p. 4); 128 x 128 frequencies are extrapolations (Sec. IV-E, p. 6).
- No support for variable K per tile, early termination, or data-dependent control (Algorithm 1 assumes a fixed K, p. 4), so nothing about FILTER / SELECT style operators, sparsity, or non-uniform reductions; no multi-format or mode switching although FPnew itself is multiformat [13].
