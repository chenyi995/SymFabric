"""Architecture configurations. Every constant has a `source:` comment (aimd rules section 18):
reference = paper + location; design = an architectural choice of this project, not a measurement;
TBD[src] = no source yet, exposed as a parameter and swept in the sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ArchConfig:
    name: str
    freq_hz: float
    rows: int                 # PE array rows (D in Tessera's notation)
    cols: int                 # PE array columns
    strip_cols: int           # fracturable strip width d; strip_cols == cols means monolithic
    modes: frozenset          # subset of {"dense", "merge", "pq", "bitmap", "stream"}
    fill_drain: str           # "classical" or "twist" (Tessera twist-torus)
    sram_bytes: int
    dram_bw_Bps: float
    random_access_bytes: int  # bytes moved per random (non-streaming) DRAM access
    op_launch_cycles: int     # fixed per-operator control / reconfiguration cost
    merge_lanes: int          # sorted-stream merge / intersect units
    stream_lanes: int         # elementwise / filter / reduce lanes
    pq_capacity: int          # entries of the systolic priority queue
    notes: str = ""

    @property
    def pes(self) -> int:
        return self.rows * self.cols

    @property
    def bytes_per_cycle(self) -> float:
        return self.dram_bw_Bps / self.freq_hz


# claude: decision start — edge configuration mirrors REASON's evaluated platform so that results
# are comparable with the group's internal baseline; the array size is Tessera's RTL scale.
EDGE28 = ArchConfig(
    name="edge28",
    freq_hz=500e6,            # source: reference — REASON, Table III footnote (p. 11): 500 MHz stated
                              #   for the DeepScaleTool-scaled rows; the 28 nm clock is not stated.
                              #   Assumption flagged; swept in sensitivity.
    rows=32, cols=32,         # source: reference — Tessera, Sec. V-A-1 (p. 8): 32x32 RTL arrays
    strip_cols=8,             # source: reference — Tessera, Sec. VI-B (p. 10): d = 8 EDP-optimal
    modes=frozenset({"dense", "merge", "pq", "bitmap", "stream"}),   # source: design
    fill_drain="twist",       # source: reference — Tessera, Sec. III-C (p. 5): fold M+2D-1, II=max(M,D)
    sram_bytes=1_250_000,     # source: reference — REASON, Fig. 10 / Table III (p. 11): 1.25 MB SRAM
    dram_bw_Bps=104e9,        # source: reference — REASON, Fig. 10 (p. 11): 104 GB/s LPDDR5 peak
    random_access_bytes=64,   # source: TBD[src] — cache-line granularity assumption; swept 32/64/128
    op_launch_cycles=0,       # source: TBD[src] — Tessera reports no reform latency (Sec. IV p. 5);
                              #   swept 0/32/256 in sensitivity
    merge_lanes=32,           # source: design — one merge unit per array column
    stream_lanes=32,          # source: design — one streaming lane per array column
    pq_capacity=1024,         # source: design — one queue cell per PE (Leiserson 1979 systolic PQ)
    notes="REASON-like edge platform: LPDDR5, 1.25 MB SRAM, 28 nm, 500 MHz",
)
# claude: decision end

# claude: decision start — datacenter configuration follows Tessera's 128x128 cycle model with one
# HBM3E stack; SRAM follows the TPU v1 unified buffer as the only sourced number for a 28 nm die.
DC_HBM3 = ArchConfig(
    name="dc_hbm3",
    freq_hz=1e9,              # source: reference — Tessera, Sec. V-B-1 (p. 9): 1 GHz cycle model
    rows=128, cols=128,       # source: reference — Tessera, Sec. V-B-2 (p. 9): 128x128 evaluated
    strip_cols=8,             # source: reference — Tessera, Abstract / Sec. V-B-2: 16 strips of 128x8
    modes=frozenset({"dense", "merge", "pq", "bitmap", "stream"}),   # source: design
    fill_drain="twist",       # source: reference — Tessera, Sec. III-C (p. 5)
    sram_bytes=24 * 1024 * 1024,   # source: reference — Jouppi et al., ISCA 2017 (TPU v1), Sec. 2:
                                   #   24 MiB Unified Buffer
    dram_bw_Bps=1.2e12,       # source: reference — Tessera, Sec. V-B-1 (p. 9): 1.2 TB/s HBM3E
    random_access_bytes=64,   # source: TBD[src] — see EDGE28
    op_launch_cycles=0,       # source: TBD[src] — see EDGE28
    merge_lanes=128,          # source: design
    stream_lanes=128,         # source: design
    pq_capacity=16384,        # source: design — one queue cell per PE
    notes="Tessera-like datacenter platform: one HBM3E stack, 24 MiB buffer, 1 GHz",
)
# claude: decision end

# The TPU-like baseline: same array and memory as DC_HBM3 but a fixed multiply-accumulate datapath.
# Sparse operators must be executed as dense semiring GEMMs over the full index domains, there is
# no merge, bitmap or priority-queue mode, and the array is monolithic.
TPU_LIKE = replace(DC_HBM3, name="tpu_like", modes=frozenset({"dense"}), strip_cols=128,
                   fill_drain="classical",   # source: reference — Tessera, Sec. II-B (p. 3): classical WS
                   notes="fixed-MAC 128x128 weight-stationary array, dense execution only")

# HBM4 variant of the datacenter platform (memory-side comparison point).
DC_HBM4 = replace(DC_HBM3, name="dc_hbm4",
                  dram_bw_Bps=2.75e12,   # source: reference — AMMA, Table 1 (p. 9): 2.75 TB/s per HBM4 cube
                  notes="DC_HBM3 with one HBM4 cube")

# claude: decision start — "all-PE" variants: every PE can be reconfigured into a merge / probe /
# stream lane (systolic relational operations in the sense of Kung & Lehman 1980), so the lane
# count equals the PE count instead of one lane per column. Same array, SRAM and DRAM as the base
# configuration; the difference isolates the value of making the PEs themselves reconfigurable.
EDGE28_ALLPE = replace(EDGE28, name="edge28_allpe", merge_lanes=EDGE28.pes, stream_lanes=EDGE28.pes,
                       notes=EDGE28.notes + "; all 1024 PEs usable as merge/probe/stream lanes")
DC_HBM3_ALLPE = replace(DC_HBM3, name="dc_hbm3_allpe", merge_lanes=DC_HBM3.pes, stream_lanes=DC_HBM3.pes,
                        notes=DC_HBM3.notes + "; all 16384 PEs usable as merge/probe/stream lanes")
# claude: decision end

CONFIGS = {c.name: c for c in (EDGE28, DC_HBM3, DC_HBM4, TPU_LIKE, EDGE28_ALLPE, DC_HBM3_ALLPE)}
