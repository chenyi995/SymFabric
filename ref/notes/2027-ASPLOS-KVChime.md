# KVChime: Enabling Shared-KV Attention in Processing-in-Memory Systems for Multi-Agent LLM Inference (ASPLOS 2027 submission draft; venue/year NOT stated in the text -- "Anonymous Author(s)", references reach Aug-2026 arXiv IDs)

- **Workload**: Multi-agent LLM serving under software KV reuse. Many agents/requests attend to a shared, retained KV pool (documents, task instructions, conversation histories) plus a small request-private KV (recomputed replacements + newly appended tokens) (Sec. 1, Sec. 2.1). Two phases: selective prefill (a few recomputed/new query tokens attend to a long retained context) and decode (one query token per agent per step) (Sec. 2.1, Sec. 3.3). Five KV-reuse workloads (Table 2) x five models 7B-76B (Table 1). Upstream reuse policy (which tokens are retained/recomputed) is an input, not a contribution (Sec. 2.1, Sec. 3).
- **Operators**: hotspot = attention over reused KV = "thin GEMM": Q.K^T, softmax, P.V with few queries scanning a long KV (Sec. 2.2). Mapping to SymFabric classes:
  - INDEX: logical-view translation -- the KV address translator maps every logical token position to exactly one shared or private physical entry using base addresses + diff mask + logical/reference positions (Sec. 3.2 step 1, Fig. 3(a)).
  - COMBINE + REDUCE over the real (+,x) semiring: Q.K^T and P.V as bank-level MACs; partial results reduced by bank-group (BG) accumulators, then across banks on the logic die via TSVs (Sec. 3, Fig. 2(b)).
  - SELECT / FILTER: per-request diff mask replaces each diff-marked shared score with the private score at the same logical position; causal mask; probability routing zeroes the non-selected value path (Sec. 3.2 steps 3-4, Fig. 3(b)).
  - REDUCE (normalization): one global softmax over the complete merged logical view, so shared and private entries share one normalization domain (Sec. 3.2 step 3).
  - UPDATE: private-KV writes (replacements, appended entries) overlap the start of the scan (Sec. 4.2, Eq. 6). Shared KV is immutable (Sec. 3.2).
  - Iteration: per layer, per KV head; a column-streaming loop over KV positions in which each streamed column is applied to M resident queries before advancing (Sec. 3, Sec. 3.3). Operands FP16 (Sec. 5.1); no non-standard semiring.
- **Data scale**: retained context 6,741-31,846 tokens per reader; recomputed 48-1,283; private input 10-80; readers 1-4 (Table 2). Selective-prefill query count = recomputed + private input (Sec. 5.1): 58 (Document 8k) to 1,341 (MuSiQue) per reader (derived from Table 2). Decode Q = one token per ready request, one vector per query head, plus GQA heads sharing a KV head (Sec. 3.3). D = 128 for all models; 32-80 Q/KV heads; 32-80 layers (Table 1). Dense arithmetic; the only "sparsity" is structural (which version of each position is live).
- **Hardware mechanism**: (1) Query-side position alignment: fold the consumer-to-reference shift Delta into the query's own RoPE angle, Q' = Q R(Delta), so resident post-RoPE shared keys are consumed unmodified (Sec. 3.1, Eqs. 1-3). (2) Replacement-aware in-memory attention over a request-specific logical KV view (shared base + private delta + diff mask), executed in five steps without materializing a private copy (Sec. 3.2, Fig. 3). (3) MQ PIM: a double-buffered MQ operand buffer with M resident query slots per bank lets each shared-K column / shared-V slice be read once per compatible group; per-query aligned operand, score accumulator, diff mask, softmax state (Sec. 3.3, Fig. 2(c), Fig. 4). Probabilities are streamed from the logic die rather than stored in-bank (Sec. 3.3). (4) Adaptive per-layer GPU-vs-PIM placement for selective prefill; decode always MQ PIM (Sec. 4.1). Substrate = AttAcc bank MAC datapath + operand-selection logic + per-query accumulation state (Sec. 3.3).
- **Simulation / implementation method**: AttAcc-based A100 + HBM3-PIM platform model, 5 PIM stacks per GPU (Sec. 5.1, Table 1). DRAM/PIM timing from Ramulator2 (command cycle 0.769 ns) collected offline; GPU side uses AttAcc's analytical compute-throughput / utilization / HBM-traffic / link-bandwidth models (Sec. 5.1). Runtime selector = offline-fitted nonnegative linear coefficients on command counts, calibrated on 180 PIM traces from 64 shapes (Sec. 4.1, Sec. 5.1). Bank, BG and logic-die components synthesized with Cadence Genus on ASAP7, pre-layout (Sec. 5.1). Level: trace-driven DRAM timing + analytical GPU + synthesis; no RTL of the full system, no P&R, no silicon.
- **Limitations**: no energy or power numbers anywhere; HBM timing constants (tRCD, tRP, tRAS, row size, bank/channel counts, bandwidth) are named but never given -- inherited from AttAcc [52]. GPU side is analytical (AttAcc model), not measured. Selective-prefill placement uses a fitted estimator whose accuracy is not reported. E2E gain diluted by projections/FFN/tensor-parallel comm outside PIM (Sec. 5.2, Sec. 5.3). Query-side alignment is acknowledged as prior art (LazyAttention [66]) (Sec. 6). Fixed decode horizons of 16/32 tokens (Sec. 5.1) shape the E2E numbers. Acknowledgements state ChatGPT/Claude generated parts of text, tables, graphs, code, data, citations.
- **Numbers worth reusing**: see the constants table below. Headline: E2E 1.24-5.22x vs GPU reuse (Abstract, Sec. 5.2, Fig. 5); mean-TBT 1.42-9.78x vs GPU reuse (Sec. 5.3, Fig. 6(b)); TTFT up to 3.25x vs Materialized PIM (Sec. 5.3, Fig. 6(a)); peak KV footprint 1.63-4.92x smaller than Materialized PIM (Sec. 5.4, Fig. 7); PIM-vs-GPU crossover moves from 54-55 to 512-545 queries with MQ (Sec. 5.5, Fig. 8); bank+BG extension area 1.36% of a 121 mm^2 DRAM die (Sec. 5.6).
- **Implications for SymFabric**: (a) The MQ operand buffer is exactly the "many queries share one streamed relation" pattern: stream a base column once, apply it to M resident query states; the ~10x crossover shift (Sec. 5.5) is the quantitative case for PIM only when Q is small-to-moderate and the shared structure is large -- otherwise the wide array wins. (b) Logical view = immutable shared base + per-query delta + diff mask, merged by SELECT before one global REDUCE, is a ready-made model for per-agent hypothetical facts / retractions over a shared KB with no copy. (c) The placement rule T_GPU = max(R + A, W) vs T_PIM = I + T_scan + S + O + W with a command-count linear estimator is directly reusable as SymFabric's systolic-vs-PIM operator placement; note the conservative choices (charge W in full, tie -> host). (d) Per-column compute budget in PIM is power-bound: 8 DRAM command cycles per column group regardless of PE frequency above ~1.3 GHz (Sec. 5.6, Fig. 9) -> the number of semiring ops per streamed column is capped; M-way sharing must fit that window. (e) Compatibility = same physical object, scan range, layer/head, layout (Sec. 3.3): a grouping key SymFabric can adopt for query batching.
- **Open source**: none stated.

## PIM cost model as stated

All symbols quoted from Sec. 4.1-4.2 (Eqs. 4-6); T_scan structure from Sec. 3.3 and Sec. 4.1.

GPU candidate (selective-prefill attention on the GPU) (Sec. 4.1, Eq. 4):
    T_GPU = max(R + A, W)
  R = retained-KV readback time over the GPU-memory link (readback precedes attention);
  A = GPU attention time (AttAcc GPU model: Q.K^T, softmax, P.V costed separately with
      compute-throughput, utilization, HBM-traffic and link-bandwidth terms, Sec. 5.1);
  W = private-KV update (write) time, overlappable with the R + A branch.

PIM candidate, selector prediction (Sec. 4.1, Eq. 5):
    T_hat_PIM = I + T_hat_scan + S + O + W
  I = input time for query operands + logical-view metadata (one query variant per distinct
      position shift Delta); S = softmax service time on the logic die; O = context-return time;
  W charged in full (conservative, no overlap assumed).

PIM candidate, replayed/actual (Sec. 4.2, Eq. 6):
    T_PIM,replay = I + T_scan + S + O + max(0, W - D)
  T_scan = aggregate command-stream time of the Q.K^T and P.V scans, excluding softmax;
  D = time from scan start to the first command that touches private KV (scan and private
      writes start concurrently after the input transfer completes).

Decision rule (Sec. 4.1): select PIM iff T_hat_PIM < T_GPU; tie -> GPU. Decided independently per
selective-prefill layer. Memory feasibility checked separately. Decode never invokes the selector
(always MQ PIM). A plain query-count or context-length threshold is rejected because it misses the
fixed command overhead of short KV objects and partially filled MQ groups (Sec. 4.1).

How T_hat_scan is built (Sec. 4.1): the estimator replays the executor's command grouping -- GQA
expansion, resident-query capacity (M = 8 slots), full and partial Q.K^T / P.V groups -- and counts
  rounded MAC work, operand and result movement, softmax commands, barriers, row visits,
then applies nonnegative coefficients fitted offline to command-level scan times (frozen before
evaluation; 180 traces from 64 shapes, Sec. 5.1). No trace is built and nothing is simulated online.
  T_hat_scan ~= sum_k c_k * count_k(logical view, model geometry, HW config),  c_k >= 0
(this summation form is my paraphrase of "nonnegative coefficients ... fit to command-level scan
times"; the paper does not print the equation).

Scan-time mechanics behind T_scan (Sec. 3.3, Fig. 4): per shared-K column the bank performs one MAC
per resident query while the column is available; the MQ group interval is bounded by
  max( time for all resident MACs at f_PE , DRAM command timing (ACT, tRCD to first column,
       tCCD_L between column commands in a bank group) , power constraint (nCCDAB = MACAB-to-MACAB
       interval on the all-bank path) ).
Evaluated point: 8 resident queries complete within the power-limited interval of 8 DRAM command
cycles at f_PE = 1.3 GHz with the DRAM command clock unchanged (Sec. 3.3, Sec. 5.6, Fig. 9).
Compatible queries beyond M slots are split into additional full or partial sweeps; partial groups
keep their full modeled command cost (Sec. 3.3, Sec. 4.2). P.V: each probability scalar is broadcast
across the bank's 16 MAC lanes; a 128-dim head over 4 banks gives 32 components per bank in two
16-component tiles, so each probability vector is streamed twice (Sec. 3.3).

Crossover as measured (Sec. 5.5, Fig. 8; LLaMA-7B, 1024-token cached context, one layer's complete
attention service, same link bandwidth and GPU curve): single-query PIM falls below GPU attention at
54-55 new queries; MQ PIM at 512-545 new queries. Beyond that, query-dependent compute outweighs the
saved KV movement and the GPU's matrix throughput wins (Sec. 5.5).

## Hardware constants (for reuse with source)

| quantity | value | unit | location | conditions |
|---|---|---|---|---|
| DRAM command cycle (tCK) | 0.769 | ns | Sec. 5.1 | Ramulator2 HBM3 model; AttAcc-based platform |
| Power-limited column-group interval | 8 | DRAM command cycles | Sec. 3.3, Sec. 5.6, Fig. 9 | 8 resident queries; all-bank path (nCCDAB) |
| KVChime PE / bank-logic frequency | 1.3 | GHz | Sec. 3.3, Sec. 5.1, Sec. 5.6 | ASAP7 pre-layout, nonnegative slack; = 1/tCK (derived) |
| AttAcc baseline bank-logic frequency | 0.666 | GHz | Sec. 5.1 | same synthesis flow |
| MAC lanes per bank | 16 | lanes | Sec. 3.3 | AttAcc bank MAC datapath retained |
| MQ operand buffer per bank | 512 | B | Sec. 3.3, Table 1 note | double-buffered |
| Resident query slots M | 8 | slots of 64 B | Sec. 3.3, Table 1 note | D = 128; larger heads -> fewer slots |
| Banks per head | 4 | banks | Sec. 3.3 | 128-dim head -> 32 components per bank, 2 tiles of 16 |
| Bank groups / banks per BG (figure labels only) | BG 0..31; Bank 0..3 | -- | Fig. 2(b) | not stated in text; inherited AttAcc HBM3 layout |
| tRCD, tCCD_L, nCCDAB | named, no values | -- | Sec. 3.3 | NOT given |
| tRP, tRAS, row size, banks/channel, channels/stack, HBM bandwidth | not stated | -- | -- | NOT given; see AttAcc [52] |
| Illustrative command interval (single-query / MQ) | 6 / 7 | tCK | Fig. 4(c),(d) | 3-query toy example; explicitly NOT the evaluated design point |
| PIM stacks per GPU | 5 | HBM stacks | Sec. 5.1, Table 1 | 1/1/4/3/4 GPUs -> 5/5/20/15/20 stacks |
| GPU baseline | A100 | -- | Sec. 5.1 | AttAcc GPU model; FP16 operands |
| GPU-memory link (NVLink 3, one-way) | 300 | GB/s | Sec. 5.1 | |
| GPU peak FLOPS / HBM bandwidth | not stated | -- | -- | inside AttAcc model |
| DRAM die area | 121 | mm^2 | Sec. 5.6 | |
| DRAM banks share of die | 87.80 | % | Sec. 5.6 | leaves 12.20% peripheral budget |
| AttAcc baseline PIM logic | 6.85 | % of die | Sec. 5.6 | |
| KVChime bank + BG extensions | 1.36 | % of die | Sec. 5.6 | total PIM logic 8.20%; 4.00% left |
| KVChime logic-die addition | 1.03 | mm^2 | Sec. 5.6 | total synthesized logic-die components 1.63 mm^2 |
| Estimator calibration set | 180 traces / 64 shapes | -- | Sec. 5.1 | coefficients frozen before replay |
| Head dim D; Q/KV heads; layers | 128; 32-80; 32-80 | -- | Table 1 | Llama 7B/GPT 13B/Llama 65B/OPT 66B/MT 76B |
| Retained / recomputed / private tokens | 7,968/48/10; 7,968/48/12; 6,741/1,283/58; 31,846/176/12; 15,688/64/80 | tokens per reader | Table 2 | Doc 8k; Readers 4; MuSiQue; Long 32k; LTM recall |
| Readers per workload | 1; 4; 1; 2; 1 | -- | Table 2 | same order |
| Decode output horizon | 32 (MuSiQue); 16 (others) | tokens | Sec. 5.1 | |
| MuSiQue selected fraction | 16% (1,283 of 8,024) | tokens | Sec. 5.1 | CacheBlend-style selection [71] |
| Motivation: TTFT, full recompute vs SW reuse | 3,298.6 -> 91.2 | ms | Sec. 1, Fig. 1(a) | Llama 7B, 4 readers, ~8K tokens, AttAcc single-GPU HBM3-PIM |
| Motivation: remote KV reads share of TTFT | 61.1 | % | Sec. 1, Fig. 1(a) | after SW reuse, not hidden by compute |
| E2E throughput vs GPU reuse | 1.24-5.22 | x | Abstract, Sec. 5.2, Sec. 7, Fig. 5 | all 25 model-workload pairs |
| E2E vs GPU reuse, Readers 4 | 2.15-2.39 / 1.43-1.50 | x | Sec. 5.2 | Llama-7B, GPT-13B / Llama-65B, OPT-66B, MT-76B |
| E2E vs Materialized PIM, Readers 4 | 1.18-1.65 | x | Sec. 5.2, Sec. 7 | |
| Mean-TBT vs GPU reuse | 1.42-9.78 | x | Sec. 1, Sec. 5.3, Fig. 6(b) | max = OPT-66B Long 32k (KV exceeds GPU capacity) |
| E2E, OPT-66B Long 32k | 5.22 | x | Sec. 5.3 | vs GPU reuse |
| TTFT vs Materialized PIM | 1.72-2.69 (Doc 8k); 1.47-1.85 (LTM); max 3.25 | x | Sec. 5.3, Fig. 6(a) | max = Readers 4 on GPT-13B |
| Layers/pairs where selector picks PIM | 18 of 25 | pairs | Sec. 5.3 | selective prefill |
| Worst TTFT slowdown vs GPU reuse | 0.06 | % | Sec. 5.3 | |
| Worst mean-TBT slowdown vs Materialized PIM | 4.24 | % | Sec. 5.3 | |
| Peak KV footprint, Materialized PIM / KVChime | 1.63-4.92 | x | Sec. 5.4, Fig. 7 | Readers 4: 4.85-4.92; MuSiQue: 1.63-1.68 |
| PIM->GPU crossover, single-query PIM | 54-55 | new queries | Sec. 5.5, Fig. 8 | LLaMA-7B, 1024-token cached context |
| PIM->GPU crossover, MQ PIM | 512-545 | new queries | Sec. 5.5, Fig. 8 | same setup; ~10x wider PIM region |
| Energy / power | not reported | -- | -- | only a qualitative "power constraint" on nCCDAB |

## Mechanisms reusable for symbolic workloads

- MQ scan: keep M query operands resident per bank, stream each shared column once, do one MAC per resident query per column; per-query score accumulator / diff mask / softmax state (Sec. 3.3, Fig. 4(b)). Symbolic analog: M concurrent queries scanning one shared relation/index.
- Compatibility key for grouping: same physical shared object, scan range, model, layer, KV head, data layout; differing positions and diff masks do not block grouping (Sec. 3.3).
- Overflow policy: more compatible queries than slots -> additional full or partial sweeps; partial sweeps pay full modeled command cost (Sec. 3.3, Sec. 4.2).
- Logical view = immutable shared base + private delta: base addresses, diff mask, logical positions, reference positions; INDEX resolves each logical position to exactly one entry (Sec. 3.2 step 1, Fig. 3(a)).
- Two operand versions per query (Q' for shared, Q for private), i.e. a per-partition transform folded into the query instead of rewriting the shared operand (Sec. 3.1 Eq. 3, Sec. 3.2 step 2).
- Score repair + one global normalization over the merged view (SELECT then REDUCE), so base and delta contribute exactly once (Sec. 3.2 steps 3, 5).
- Probability routing by mask (zero the non-selected path) instead of gathering (Sec. 3.2 step 4).
- Streaming the second-pass operand (probabilities) from the logic die, broadcast across 16 lanes, two 16-wide tiles per bank, rather than storing it in-bank (Sec. 3.3).
- Bank-level scheduling: MQ fills idle PE cycles inside the DRAM command interval; interval = max(compute, DRAM timing, power/nCCDAB); PE frequency chosen where compute interval meets the 8-cycle power floor (Sec. 3.3, Sec. 5.6, Fig. 9).
- Hierarchical REDUCE: bank PEs -> BG accumulators -> logic-die accumulation over TSVs; softmax/mask on the logic die (Sec. 3, Fig. 2(b)).
- Private (per-query) partitions are scanned unshared and concurrently with private writes; only the shared partition is amortized (Sec. 3.3, Sec. 4.2).
- Per-operator placement with a command-count linear cost model, conservative charging, tie -> host, no online simulation (Sec. 4.1).
- Decode-style many-agent sharing: one query per ready request grouped on the same shared pool; a scan waits for any newly generated KV it consumes (Sec. 4.2).
