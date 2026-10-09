# Tessera: Fine-Grained Fracturable Systolic Array for Adaptive Tensor Shapes (HPCA 2027 submission #1832, confidential draft)

> Group-internal draft ("Confidential Draft, Do NOT Distribute", title page p.1). Locations below are section / figure / algorithm plus the page number of the draft.

- **Workload**: dense GEMM (C = A×B, M×K×N) from LLM serving — attention projections, attention logits/context, FFN, unembedding (Table I p.1); a DNN inference suite (the Planaria network suite, 259 layers — §V-B-3 p.9, §VI-B p.10); continuous-batching LLM traces (Llama-2-7B, Llama-3-8B, Llama-2-70B TP=4, Phi-2 on the Splitwise conversation/code traces and an arXiv summarization trace, simulated by Vidur — §V-B-3 p.9); multi-tenant QoS serving (Planaria workloads A/B/C — §VI-E p.11). Focus: small, tall, wide and non-array-aligned GEMMs that underutilize a monolithic weight-stationary (WS) D×D array, D commonly 128 or 256 (§I p.1; Table I p.1; §II-A p.2-3).
- **Operators**: GEMM only, over the (+,×) semiring, BF16 inputs with FP32 accumulate (§V-A-1 p.8). SymFabric mapping: COMBINE = the PE multiply, REDUCE = the partial-sum chain along the horizontal Psum route (§III p.3; Fig. 5a p.6); INDEX = the row-rotated stationary weight layout plus the column-wise permutation placement (Algorithm 1 p.4); UPDATE of the stationary operand at runtime = in-array weight load by a one-bit lock pulse (§III-B p.4), needed because attention's QK^T / PV stationary operands are produced at runtime (§III-B p.4). Iteration structure: a "fold" = one weight-matrix load + stream of M activation rows + drain (§III-C p.5); a GEMM = R rounds of D×D packed weight matrices (Algorithm 2 p.8; §IV-D p.8). No FILTER / SELECT, no sparsity.
- **Data scale**: 128×128 PE array (§V-B-2 p.9) run as 16 strips of 128×8 (Abstract p.1; §V-B-2 p.9); RTL at 32×32 = 1024 PEs (§V-A-1 p.8). Utilization study: Q = 1 … 1024 back-to-back tiled GEMMs, tile height M/D from 1/128 to 16 at D=128 (Fig. 2 p.3) — here Q means back-to-back GEMMs, not concurrent queries. Serving traces: opening 300 requests of the Splitwise trace at 1 QPS (§V-B-3 p.9). Dense only.
- **Hardware mechanism**: (1) Twist-torus systolic array (TTSA): WS array whose activation route is the diagonal torus link (x,y)→(x+1,(y−1) mod D) and whose partial-sum route is horizontal (x,y)→(x+1,y) (§III-A p.3), so all operands travel in one direction and the boundary skew-register triangle disappears (Fig. 1 p.2; §III-C p.5); a column-wise permutation placement keeps every torus hop on a same-row or adjacent-row physical neighbor for any D (Algorithm 1 p.4). (2) In-array weight load: weights stream row-major on the activation route and a one-bit lock pulse on the Psum route latches them in twist order (§III-B p.4; Fig. 3b p.4). (3) Vertical strips of D×d PEs with a strip memory interleaved between adjacent strips; two boundary-PE types (TBPE at d-row block edges, SBPE at strip edges) implement a one-bit partition/merge choice per boundary (§IV p.5; §IV-B p.6; Fig. 4 p.5; Fig. 5 p.6). (4) Mirror-chain mapping of tall/wide weight matrices with partial-sum reuse (tall) or activation reuse (wide) across merged boundaries (§IV-C p.6-7; Fig. 6 p.7; Fig. 7 p.8) and a per-GEMM EDP-optimal (k,n) choice with first-fit packing into D×D rounds (Algorithm 2 p.8).
- **Simulation / implementation method**: RTL + Cadence Genus synthesis at 700 MHz in TSMC 28 nm for four 32×32 tiers (d = 16, 8, 4, 2), gate-level power with Cadence Joules (§V-A p.8-9); cycle-level model extending SCALE-Sim for 128×128 (§V-B-1 p.9). No P&R, no silicon.
- **Limitations**: fixed WS dataflow — WS/OS/IS switching is future work (§VII p.11); dense GEMM, one number format; mapping is EDP-optimal per layer but not QoS-aware, so Planaria wins the B-S and B-M multi-tenant settings (§VI-E p.11); RTL only at 32×32, the 128×128 results come from the cycle model plus per-op energy scaled from the 32×32 gate-level runs (§V-A p.8-9; §V-B-1 p.9); controller area excluded from totals (§VI-A p.10); fine granularity is expensive at chip level — up to 97% area overhead at d=2, dominated by strip-memory SRAM at N28 (§VI-A p.10); no absolute PE area, SRAM capacity, or absolute 128×128 power is reported; no cycle-count validation of the SCALE-Sim extension against RTL is stated.
- **Numbers worth reusing**: see "Hardware constants" below.
- **Implications for SymFabric**: see "Reusable for SymFabric" below.
- **Open source**: none stated for Tessera; the Planaria baseline is rebuilt from Planaria's released code (§V-A-1 p.8).

## Reconfiguration mechanism

**Granularity.** The array is cut along columns into vertical strips of D×d PEs, each built from d×d PE blocks stacked along the column, with a strip memory between every pair of adjacent strips (§IV p.5; Fig. 4a p.5). Partition points exist in both directions — between adjacent strips and between stacked blocks — so the unit of reconfiguration is a flex compute region (FCR): a rectangle bounded by cuts at d granularity in both directions, rows n1·d to n2·d of a contiguous strip group (§IV p.5). Evaluated point: D=128, d=8, i.e. 16 strips of 128×8 and a minimum region of 8×8 (§V-B-2 p.9; §VI-A p.10; Table II footnote d p.11). Candidate weight-matrix sides are S = {d·2^n, n = 0 … log2(D/d)} (Algorithm 2 p.8). d=8 was chosen because the geomean EDP over 259 layers stays within 1.01× of every layer's own optimum (§VI-B p.10; Fig. 10 p.9); boundary cost "roughly doubles per halving" of d (§VI-A p.10).

**What is switched.** Not the dataflow: every region runs the same twist-torus WS dataflow (§III p.3). What changes is (a) the region shape, through the boundary interconnect, and (b) the weight-load order, through the lock-pulse schedule:
- Twist boundary PE (TBPE) at every d-row block boundary, in odd- and even-column forms because the twist's crossing links alternate sides with column parity (§IV-B p.6; Fig. 5c,d p.6). Merge: the mux selects the across-boundary activation, so blocks above and below run as one taller region. Partition: the mux selects the in-strip predecessor and restarts the chain, opening a new region edge. Either choice preserves the torus-link timing (§IV-B p.6).
- Strip boundary PE (SBPE) at the end of each PE row on the strip's memory-facing edge (§IV-B p.6; Fig. 5b p.6). Merge: the partial sum goes to a pipeline flop forwarding to the next strip, and the two strips run as one wider array. Partition: it is written directly into the adjacent strip memory, which then serves the right strip's operand reads (§IV-B p.6).
- Each boundary is one binary decision; the same one-bit enable drives every mux along it; a strip-boundary partition is a row of multiplexers each with an optional pipeline flop; per-PE datapath, per-strip accumulator and in-strip torus links are untouched, so boundary cost grows only with the number of boundaries — one mux row per strip boundary and per block edge — independent of array side (§IV-B p.6).
- Mirror-chain: a K×N weight matrix pads to an s×l rectangle (s a multiple of d, l a multiple of s), splits into s×s blocks laid across strips; the leading block loads in twist permutation order and the trailing block in reversed order, so partial sums (tall, split along K) or activations (wide, split along N) cross the merged boundary with in-array reuse intact, with no extra hardware beyond the boundary PEs (§IV-C p.6-7; Fig. 6 p.7; Fig. 7 p.8).
- Strip memory: two single-port R/W banks in double-buffered ping-pong; each bank has D/d lanes, lane y wired to rows [yd, yd+d); one lane delivers d 16-bit operands or d 32-bit partial sums per SRAM access, so the array reads D operands and drains D partial sums per cycle from D/d disjoint macros without bank conflicts; bandwidth scales with strip count (§IV-A p.5).

**Reconfiguration cost.**
- Area (32×32 RTL, array-only, TSMC 28 nm): 0.9% to 12% over the TTSA monolith for d = 16 → 2, versus Planaria's 12.6% to 51% (Abstract p.1; §VI-A p.10; §VIII p.11); 2.6% at d=8 versus Planaria's 21.1% (§VI-A p.10). Full chip including strip-memory SRAM: 6.6% to 97% (Planaria 13.5% to 121%) (§VI-A p.10).
- Power / energy: array power 24.8% below Planaria at d=2 (§VI-A p.10); 4.41 pJ/op for Tessera, equal to the WS monolith's 4.41 pJ/op, versus Planaria's 4.54 pJ/op (§VI-A p.10; Fig. 9 p.9). Energy per op rises at finer granularity because partitioned regions re-read shared activations and detour partial sums through strip memory (§IV-B p.6).
- Time: no separate reconfiguration latency is quantified. Regions are "reformed between GEMMs" at "reform points" by setting the enables (§IV p.5). The quantified per-GEMM costs are fold timing: first weight column ready after D−1 cycles (§III-B p.4); fold latency M+2D−1 cycles vs classical M+3D−2 (§III-C p.5); steady-state II = max(M,D), with D−M bubbles per fold when M<D (§III-C p.5); strip-memory refill hidden when M ≥ d (§IV-A p.5); T_GEMM = H + R·max(M,H) + W − 1 (§IV-D p.8).
- Register state: the skew-register triangle is removed and only one weight-bypass register plus a one-bit lock register are added per PE, so the fabric holds fewer registers than the classical WS array (§III p.3; §III-C p.5).

**Control scheme.** A controller sets the partition/merge choice per boundary and both enables (TBPE, SBPE) per region (§IV p.5; §IV-B p.6); controllers are "instruction-driven" and excluded from area totals (§VI-A p.10). A region is reserved for one workload between two reform points; multiple regions coexist (§IV p.5). A runtime algorithm (Abstract p.1) — Algorithm 2 (p.8) — picks per GEMM the (k,n) ∈ S² minimizing EDP, splits K into h=⌈K/k⌉ blocks and N into w=⌈N/n⌉ strips, and first-fits weight matrices into D×D rounds, leaving a partly filled round open for the next independent GEMM (Algorithm 2 p.8; §IV-D p.8). Under continuous batching only admitted, mutually independent GEMMs are packed (§VI-D p.10-11).

## Simulation methodology
- **Simulator**: cycle-level model extending SCALE-Sim [41] with the twist-torus fold timing of §III-C and Algorithm 2 (§V-B-1 p.9); the abstract calls it "cycle-accurate" (Abstract p.1). Per tiled GEMM, the fold latency and II of §III-C are the inputs to the cycle model, applied per region (§III-C p.5). DRAM bandwidth swept at 1 GHz over TPU v5e (800 GiB/s), HBM3E (1.2 TB/s) and HBM4 (2.8 TB/s) (§V-B-1 p.9); headline results at HBM4 (§VI-B, §VI-C p.10).
- **Energy in the model**: gate-level array energy per op from the 32×32 RTL plus per-access SRAM dynamic energy from TSMC N28 macro datasheets (§V-B-1 p.9); the EDP objective sums computation energy and SRAM read/write energy and excludes DRAM, arguing a TPU-class chip holds a whole layer on chip (§VI-B p.10).
- **Validation**: matched Tessera and Planaria RTL at every granularity tier with the same PE count, d and on-chip SRAM (§V-A-1 p.8); the classical WS and unpartitioned TTSA monoliths "calibrated the array level" (§V-A-1 p.8); Planaria implemented from its released code, and the 16-sub-array tier reproduces the Planaria paper's configuration (§V-A-1 p.8). No cycle-count cross-check of the cycle model against RTL is stated.
- **RTL / synthesis flow**: Cadence Genus at 700 MHz, TSMC 28 nm; SRAM macros from the TSMC 28 nm memory compiler; each 32-scale chip runs one 128×32×32 GEMM in gate-level simulation; Cadence Joules integrates the per-cycle power timeline (§V-A-2 p.9). Designs include strip / sub-array / pod memory (Planaria: one pod memory per 4 sub-arrays), the main buffer and off-chip channel models, accounted separately (§V-A-1 p.8). "Array level" = FMA + array logic only, controllers excluded (§VI-A p.10).
- **Compared designs (all 128×128 PEs)**: monolithic WS; Planaria (sixteen 32×32); SOSA (sixteen partitioned 32×32); FlexSA (four 64×64); SISA (eight output-stationary 128×16 slabs); Tessera (sixteen 128×8 strips); Ideal = TTSA fully partitioned into 256 independent 8×8 sub-arrays, a reference upper bound (§V-B-2 p.9). Granularity study sweeps d = 128 → 2 (§V-B-2 p.9).
- **Workloads**: Planaria network suite, 259 layers (§V-B-3 p.9; §VI-B p.10); Vidur-simulated LLM traces (§V-B-3 p.9); multi-tenant QoS with Poisson arrivals and deadlines of 1×, 1/4×, 1/16× the MLPerf server latency target (§VI-E p.11).
- **Metrics**: area and power normalized to the WS monolith (Fig. 8 p.9); array-only dynamic power trace over one GEMM pass and pJ/op (Fig. 9 p.9); geomean EDP normalized to each layer's own optimum (Fig. 10 p.9); speed-up over WS per network (Fig. 11 p.9) and per trace vs DRAM bandwidth (Fig. 12 p.10); maximum QPS meeting the MLPerf SLA (Fig. 13 p.11); analytic utilization U = QM/(QM+3D−2) (§II-B p.3).

## Hardware constants (for reuse with source)

| Quantity | Value | Unit | Location | Conditions |
|---|---|---|---|---|
| Technology node | TSMC 28 nm (N28) | — | §V-A-2 p.9; §VIII p.11 | RTL/synthesis of 32×32 tiers |
| Synthesis clock | 700 | MHz | §V-A-2 p.9 | Cadence Genus |
| Cycle-model clock | 1 | GHz | §V-B-1 p.9 | SCALE-Sim extension, 128×128 |
| Tools | Cadence Genus (synthesis), Cadence Joules (power), TSMC 28 nm memory compiler (SRAM) | — | §V-A-2 p.9 | gate-level sim of one 128×32×32 GEMM |
| PE datapath | BF16 in, FP32 accumulate FMA + weight-bypass register + 1-bit lock register | — | §V-A-1 p.8; §III p.3; Fig. 5a p.6 | all compared designs BF16/FP32 |
| PE area (absolute) | not reported | — | Fig. 8 p.9 gives only normalized array totals | — |
| RTL array size | 32×32 = 1024 PEs; tiers d = 16, 8, 4, 2 | PEs | §V-A-1 p.8 | matched Tessera vs Planaria |
| Evaluated array size | 128×128 | PEs | §V-B-2 p.9 | cycle level, all designs matched |
| Tessera strip shape | 16 strips of 128×8 | PEs | Abstract p.1; §V-B-2 p.9 | D=128, d=8 |
| Minimum partition | 8×8 (d=8) | PEs | §VI-A p.10; §VI-B p.10; Table II p.11 | chosen by EDP sweep |
| Candidate weight-matrix sides | S = {d·2^n}, n = 0 … log2(D/d) | PEs | Algorithm 2 p.8 | per-GEMM (k,n) ∈ S² |
| TTSA monolith array area | 1.01 | × WS-monolith array | Fig. 8 p.9 | 32×32, d=32, array-only |
| Tessera array area, d=16/8/4/2 | 1.02 / 1.04 / 1.07 / 1.13 | × WS-monolith array | Fig. 8 p.9 | array-only (FMA + array logic), controller excluded (§VI-A p.10) |
| Planaria array area, d=16/8/4/2 | 1.13 / 1.21 / 1.28 / 1.51 | × WS-monolith array | Fig. 8 p.9 | same |
| Array area overhead of fracturing | 0.9 to 12 (Tessera) vs 12.6 to 51 (Planaria) | % over own monolith | Abstract p.1; §VI-A p.10; §VIII p.11 | d = 16 → 2, 32×32 |
| Array area overhead at d=8 | 2.6 (Tessera) vs 21.1 (Planaria) | % | §VI-A p.10 | 32×32 |
| Array area saving vs Planaria | 9.5 to 25.1 | % | Abstract p.1; §I p.2 | same granularity |
| Full-chip area (+SRAM), Tessera d=32/16/8/4/2 | 1.70 / 1.81 / 2.03 / 2.47 / 3.35 | × WS-monolith array | Fig. 8 p.9 | "TTSA+M" / "Tes+M", includes strip-memory SRAM |
| Full-chip area (+SRAM), Planaria d=32/16/8/4/2 | 1.69 / 1.92 / 2.21 / 2.69 / 3.73 | × WS-monolith array | Fig. 8 p.9 | "WS+M" / "Pla+M" |
| Full-chip area overhead | 6.6 to 97 (Tessera) vs 13.5 to 121 (Planaria) | % | §VI-A p.10; §VIII p.11 | dominated by strip-memory SRAM at N28 |
| Array power, Tessera d=32/16/8/4/2 | 1.02 / 1.03 / 1.04 / 1.06 / 1.09 | × WS-monolith array power | Fig. 8 p.9 | gate-level, 128×32×32 GEMM |
| Array power, Planaria d=16/8/4/2 | 0.99 / 1.07 / 1.28 / 1.45 | × WS-monolith array power | Fig. 8 p.9 | same |
| Full-chip power (+SRAM), Tessera d=32/16/8/4/2 | 1.07 / 1.11 / 1.19 / 1.33 / 1.59 | × WS-monolith array power | Fig. 8 p.9 | same |
| Full-chip power (+SRAM), Planaria d=32/16/8/4/2 | 1.05 / 1.08 / 1.22 / 1.55 / 1.96 | × WS-monolith array power | Fig. 8 p.9 | same |
| Array power vs Planaria | 24.8 below | % | §VI-A p.10 | d=2 |
| Array energy per op | 4.41 (Tessera) = 4.41 (WS); 4.54 (Planaria) | pJ/op | §VI-A p.10; Fig. 9 p.9 | gate-level, array-only, 32×32 |
| Strip memory organization | 2 single-port R/W banks (ping-pong); D/d lanes per bank; lane = d × 16-bit operands or d × 32-bit partial sums per access | — | §IV-A p.5 | absolute capacity not reported |
| Array-side SRAM bandwidth | D operands read + D partial sums drained per cycle from D/d disjoint macros | words/cycle | §IV-A p.5 | refill hidden when M ≥ d |
| DRAM bandwidth assumed | 800 GiB/s (TPU v5e) / 1.2 TB/s (HBM3E) / 2.8 TB/s (HBM4) | — | §V-B-1 p.9; Fig. 12 p.10 | headline results at HBM4 |
| Boundary hardware | 1 mux row per strip boundary and per block edge; 1-bit enable; optional pipeline flop | — | §IV-B p.6 | independent of array side |
| Classical WS utilization | U = QM/(QM+3D−2) | — | §II-B p.3 | Q back-to-back tiled GEMMs, M ≥ D |
| Utilization with a short dimension | min(M,N,K)/D | — | §II-A p.3 | classical D×D WS |
| TTSA utilization when M < D | M/D (D−M bubbles per fold) | — | §III-C p.5 | per region, D → region side |
| Speed-up on an underutilized shape | padded/useful MACs = 1 / (min(M,N,K)/D) | — | §IV-C p.8 | D-aligned shapes perform equally |
| Pipeline fill | D (TTSA) vs 2D−1 (classical) | cycles | §III-B p.5 | — |
| Fold latency | M+2D−1 (TTSA) vs M+3D−2 (classical) | cycles | §III-C p.5 | double buffering |
| Steady-state initiation interval | max(M, D) | cycles | §III-C p.5 | — |
| Drain tail removed | D−1 (full array); d−1 (sub-array side d) | cycles | §II-B p.3; §III-C p.5; §VI-C p.10 | — |
| First weight column ready | D−1 | cycles | §III-B p.4 | in-array weight load |
| GEMM time | T_GEMM = H + R·max(M,H) + W − 1 | cycles | §IV-D p.8 | H×W padded weight, R rounds |
| Speed-up over WS monolith (overall range) | 1.24 to 23 | × | Abstract p.1; §I p.2 | 128×128, HBM4 |
| DNN suite geomean over WS | 3.5 | × | §VI-C p.10 | HBM4 |
| GNMT encoder/decoder over WS / over Planaria | 23 / 4.2 | × | §VI-C p.10 | small-M layers |
| MobileNet-v1, SSD-MobileNet-v1 over WS | up to 3.0 | × | §VI-C p.10 | — |
| Over Planaria outside GNMT | up to 2.3 | × | §VI-C p.10 | — |
| LLM traces over WS (HBM4) | 1.24 (code) / 8.7 (Phi-2 conv) / 6.7 (arXiv) / 5.4 (Splitwise conv) | × | §VI-D p.11 | continuous batching |
| LLM traces over Planaria | up to 2.2; 1.58 on Splitwise conv | × | §VI-D p.11 | — |
| Multi-tenant over Planaria | up to 3.45; wins 5 of 7 decided settings | × | §VI-E p.11 | deadlines 1×, 1/4×, 1/16× MLPerf |
| Geomean EDP vs per-layer optimum, d=2/4/8/16/32/64/128 | 1.00 / 1.00 / 1.01 / 1.06 / 1.17 / 1.37 / 1.91 | × | Fig. 10 p.9; §VI-B p.10 | 259 layers, HBM4 |
| GNMT-encoder EDP at d=128 | 11.8 | × optimum | §VI-B p.10 | — |
| Preferred weight-matrix sizes | K ∈ {16, 32, 64}, N ∈ {8, 16, 32} | PEs | §VI-B p.10 | most layers |

## Reusable for SymFabric

**Carries a programmable-semiring PE (min/+, max/×, OR/AND, log-sum-exp):**
- Twist-torus routes and the permutation placement (Algorithm 1 p.4): pure interconnect that forwards an "activation" and a "partial sum" one hop per cycle; nothing depends on × or +. The timing formulas — D-cycle fill (§III-B p.5), fold latency M+2D−1 and II = max(M,D) (§III-C p.5) — hold for any PE whose ⊗-then-⊕ completes within one hop on both routes. For a multi-cycle ⊕ (log-sum-exp) both routes must be pipelined equally to keep the diagonal alignment; the paper does not treat this case.
- In-array stationary-operand load by lock pulse (§III-B p.4): latches whatever word streams past, so it works for min/+ weights, Boolean relation tiles or log-domain weights. It is the mechanism SymFabric needs when the stationary operand is a runtime intermediate — the QK^T / PV case in the paper (§III-B p.4).
- TBPE/SBPE boundary muxes with one-bit enables (§IV-B p.6): semiring-agnostic; the partition/merge choice only decides where the ⊕-chain terminates and where the "activation" chain restarts.
- Mirror-chain mapping (§IV-C p.6-7): partial-sum chaining across a K-split needs only associativity of ⊕ (true for +, min, max, OR, log-sum-exp); activation forwarding across an N-split is semiring-free. Caveat: the "padded zeros" that fill a fixed square (§IV-C p.7) must become the semiring's ⊕-identity / ⊗-annihilator (+∞ for min/+, −∞ for max/× in log domain, FALSE for OR/AND, −∞ for log-sum-exp), so the PE must encode that value.
- Strip-memory organization — D/d lanes, ping-pong banks, one lane per d-row block (§IV-A p.5): reusable; lane widths (16-bit operands, 32-bit partial sums, §IV-A p.5) must be re-sized to the semiring's value type (1-bit for OR/AND, FP32 for log-sum-exp).
- Algorithm 2 (EDP-optimal (k,n) plus first-fit rounds, p.8) and the granularity-selection method (EDP vs d sweep, Fig. 10 p.9): reusable structure; the energy terms must be re-characterized for the semiring PE.
- Methodology: SCALE-Sim extension with per-region fold timing (§V-B-1 p.9), 32×32 RTL tiers in TSMC 28 nm at 700 MHz with Genus/Joules (§V-A p.8-9), and the WS-monolith-normalized area/power reporting (Fig. 8 p.9) can be used unchanged for SymFabric's array.

**Assumes a MAC (do not reuse numerically):**
- The PE datapath (FMA, BF16 in / FP32 accumulate — Fig. 5a p.6; §V-A-1 p.8), the 4.41 pJ/op (§VI-A p.10), and every normalized ratio in Fig. 8 (p.9) are relative to an FMA PE. A cheaper PE (OR/AND, integer min/+) makes the same boundary muxes and strip memory a larger fraction; a costlier PE (log-sum-exp with exp/log units) makes them smaller. The 0.9% to 12% array overhead (§VI-A p.10) therefore does not transfer as a number, only as a mechanism.
- The EDP optimum d=8 (§VI-B p.10) and the preferred K/N sizes (§VI-B p.10) are specific to GEMM shapes of the Planaria suite and LLM layers.
- Dataflow is fixed WS; SymFabric operators that want a stationary output (join-then-aggregate) need the WS/OS/IS switching the paper leaves as future work (§VII p.11).
- Bandwidth: finer partitioning trades in-array reuse for SRAM traffic (§II-C p.3; §IV-B p.6), and Tessera is the only design whose speed-up still grows from HBM3E to HBM4 "because packing fills the array until bandwidth is the limit" (§VI-D p.11). A fine-grained fracturable 128×128 array is thus bandwidth-limited at 1.2 TB/s on these traces (§VI-D p.11; Fig. 12 p.10) — the datum to feed SymFabric's "processing-in-memory only if bandwidth-bound" decision, measurable with the same SCALE-Sim extension.
