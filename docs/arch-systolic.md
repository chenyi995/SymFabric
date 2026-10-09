# SymFabric compute side: a reconfigurable semiring systolic array (SSA)

Status: design sketch backed by the analytical model in `systolicarch/`. Numbers below are either
cited (paper, section) or marked *design choice*; nothing here has been synthesized.

## 1. Why a plain GEMM systolic array is not enough

REASON measured a TPU-like array (SCALE-Sim, 8 x 128x128) at 74.7x–109.2x the runtime of its tree
engine on the symbolic-only kernels of six neuro-symbolic workloads, while being 1.4x faster on the
neural parts (REASON, Fig. 13, p. 13). The reason is structural, not a matter of tuning: symbolic
hot loops are joins, masks, fixpoints and selections over sparse relations, and a fixed-MAC array can
only execute them as dense (+,x) products over the full index domains. Our own model reproduces the
effect: for the semi-naive transitive closure at the "large" scale the dense execution performs
n^3 multiplications per round where the sparse join performs |Delta join E| (see `eval/results`).

Three things therefore have to be reconfigurable, in this order of importance:

1. **the dataflow** (dense weight-stationary vs sorted-stream merge vs priority queue vs bitmap),
2. **the array shape** (a 128-wide array runs a 16-wide codebook or a 12-state DFA at 12% utilisation
   unless it can be fractured into strips),
3. **the semiring** (add/mul, max/+, min/+, OR/AND, LSE/+), which is the cheapest of the three: a mux
   in front of the adder and the multiplier.

## 2. Execution modes

| Mode | Operators | Data | Cycle model (`systolicarch/model.py`) | Source of the formula |
|---|---|---|---|---|
| dense_gemm | combine(keep=...) on dense operands | stationary K x N operand, M streaming rows, batch = Q | tiles·max(M,D) + 2D−1 (twist) or + 3D−2 (classical); strips of width d run tiles in parallel | Tessera §II-B, §III-C, §IV (fold latency, II, fracturing) |
| dense_elementwise | combine without keep on dense operands | broadcast product | M·K·N / PEs | — (design) |
| merge_join | combine on sparse operands | two coordinate streams sorted on the join key | (n_left + n_right + candidates) / merge_lanes | design; one comparator-emit unit per column (Kung & Lehman 1980 systolic relational ops; ExTensor-style intersection) |
| segmented_reduce | reduce | keyed stream | n_in / stream_lanes (+ one partition pass if accumulators exceed SRAM) | design |
| onchip_probe / random_probe / partitioned_probe | mask (semijoin / antijoin) | probe stream vs table | n_probe / lanes if the table is on chip; else min(random 64 B accesses, streaming partition pass) | design; random-access granularity is TBD[src] |
| onchip_update / random_rmw / stream_merge | update | state + delta | n_delta / lanes on chip; else min(RMW per key, stream-merge of the state) | design |
| systolic_pq | select_k | persistent queue | inserts + pops; capacity = one cell per PE | Leiserson 1979 systolic priority queue |
| stream | filter, scan | stream | n_in / stream_lanes | design |

The PE datapath needed by all modes: one multiplier-class unit (mul / add / min / max / AND), one
adder-class unit (add / max / min / OR / LSE-approx), one comparator on keys, a valid bit, a key
register and a value register, plus the twist-torus links and the strip boundary muxes of Tessera.
piPE-SA gives the N28 cost of the arithmetic core: an INT8 MAC is 260 ps / 1,199 um2 and a BF16
FMA 1,034 ps / 7,013 um2 at minimum delay (piPE-SA Table I, p. 1); the comparator, the mode muxes
and the LSE approximation are not in that table and are the first thing E4 (semiring-node
synthesis) has to measure.

## 3. Configurations modelled

See `systolicarch/config.py` for every constant and its source. `edge28` follows REASON's platform
(500 MHz assumed, 1.25 MB SRAM, 104 GB/s LPDDR5; array 32x32 from Tessera's RTL scale). `dc_hbm3`
follows Tessera's cycle model (128x128, strips of 8, 1 GHz, 1.2 TB/s HBM3E) with a 24 MiB buffer
(TPU v1). `tpu_like` is `dc_hbm3` with a fixed-MAC datapath: dense mode only, monolithic, classical
fill/drain. `dc_hbm4` swaps in one HBM4 cube (2.75 TB/s, AMMA Table 1).

## 4. What the model does not capture

- Bank conflicts and NoC contention inside the chip; the merge units are assumed fed at line rate.
- The cost of sorting: merge_join assumes both streams arrive sorted on the join key. Relations
  produced by a previous operator in a different key order need a sort or a hash partition; the
  mapper charges a partition pass only when a probe table does not fit on chip.
- Reconfiguration latency between operators (`op_launch_cycles`, default 0, swept).
- Skew: candidates are spread evenly over lanes.
These are the items a cycle-level model (SCALE-Sim extension or an event-driven simulator) would add.
