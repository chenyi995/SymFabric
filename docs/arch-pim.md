# SymFabric memory side: bank-level PIM model and placement selector

Status: analytical model in `pim/`, used only where the compute-side model (`systolicarch/`) shows an
operator to be memory-bound. Constants and their sources are in `pim/config.py`.

## 1. What the memory side can do

Two execution shapes, both derived from KVChime's multi-query (MQ) bank scan and logical view:

- **MQ scan** (`pim.model.mq_scan`): the large operand is bank-interleaved and streamed row by row
  inside every bank; Q query operands sit in the bank PE's slots (8 in KVChime, Sec. 3.3) and every
  streamed tuple is compared against all of them. Row streaming is bounded per pseudo channel by
  min(16 banks / tRC, 4 / tFAW) (Ramulator2 HBM3 preset: tRC 45 ns, tFAW 15 ns), i.e. 8.7 TB/s of
  row bandwidth per stack against 819 GB/s at the interface. The PE side is n_kb x Q compares at
  16 lanes per bank and 0.666 GHz (AttAcc bank logic). Q above the slot count costs extra passes.
- **Fused fixpoint round** (`pim.model.fused_round`): combine -> mask -> update of one round runs
  inside the banks: the delta is routed by join key to the owning banks, matched candidates are
  shuffled over the buffer die to the banks that own the state partition, probe and update are
  bank-local, and only the delta (in) and the changed set (out) cross the interface. The shuffle
  bandwidth has no source yet (TBD[src], swept).

Operators with no memory-side form: select (priority queue) and dense semiring GEMMs.

## 2. What the model says before any workload is run

With the baseline bank PE (16 lanes, 0.666 GHz) an MQ scan over 4-byte tuples is **PE-bound, not
row-bound** (`tests/test_pim_model.py`): a 1 KB row holds 256 tuples, comparing them against 8
resident queries takes 128 PE cycles = 192 ns, four times the 45 ns row cycle. Symbolic scans are
compare-heavy per byte, so bank-PE compute density, not internal bandwidth, is the first design
question for a symbolic PIM. The optimistic point (1.3 GHz, 32 lanes) closes most of the gap.

## 3. Placement

`pim.selector.select_placement` groups a trace into fused rounds, prices each group on the chip
(sum of the systolic model's cycles) and memory-side, and takes the minimum. The comparison is
per group and per instance, so the crossover in Q or in working set appears in the results rather
than being assumed.

## 4. Sourced reference point for the bank PE (cited)

The group's KVChime RTL gives the only synthesized bank-level datapath available to this project:
the 16-lane FP16 `gemv_unit` is 6,205 um2 at a 1,502 ps period (AttAcc point, 0.666 GHz) and
7,343 um2 at 769 ps (1.3 GHz), ASAP7 TT 0.7 V, Cadence Genus 25.1, 1,024 instances per stack, with
slack >= 0 (KV-PIM/kvpim-rtl/docs/KVChime/experiment-2-area/KVChime-components.csv, rows
`gemv_flop_p1501` and `gemv_flop_p769`; synthesis only, no place-and-route). A symbolic bank PE
replaces the FP16 multiply-accumulate by compare / AND / OR / min / max / popcount lanes on 1–32-bit
operands, so this number is an upper bound on its arithmetic area; the query-slot registers and the
key comparators are the new cost and have not been synthesized (E4 in the plan).

## 5. Not modelled

Bank conflicts between PIM commands and normal traffic (NeuPIMs' concern), refresh, DRAM-process
area of the bank PE beyond KVChime's 1.36 % budget statement, energy, and the cost of laying out a
relation bank-interleaved in the first place (charged once per instance in a later revision).
