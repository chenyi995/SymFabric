"""PIM platform constants. Every value has a `source:` comment (aimd rules section 18)."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PimConfig:
    name: str
    # ---- DRAM organisation and timing (HBM3) ---------------------------------------------
    stacks: int
    channels_per_stack: int
    pcs_per_channel: int
    banks_per_pc: int
    row_bytes: int
    access_bytes: int
    tCK_s: float
    nRC: int                  # ACT-to-ACT, same bank
    nFAW: int                 # four-activate window, per pseudo channel
    nRCDRD: int
    ext_bw_Bps: float         # external interface bandwidth per stack
    # ---- bank-level processing element -------------------------------------------------------
    pe_lanes: int             # parallel compare / min / add lanes per bank
    pe_freq_hz: float
    query_slots: int          # query operands resident per bank for multi-query scans
    # ---- buffer (logic) die and inter-stack ---------------------------------------------------------
    fabric_bw_Bps: float      # cross-bank shuffle bandwidth on the buffer die, per stack
    global_reduce_s: float    # one all-bank OR / min / count reduction
    cmd_latency_s: float      # fixed latency of one memory-side command group (issue, operand
                              # broadcast, result read-back), paid once per placed group
    d2d_hop_s: float
    d2d_bw_Bps: float
    d2d_links_per_stack: int = 4   # mesh degree
    notes: str = ""

    @property
    def pcs(self) -> int:
        return self.stacks * self.channels_per_stack * self.pcs_per_channel

    @property
    def banks(self) -> int:
        return self.pcs * self.banks_per_pc

    @property
    def mesh_side(self) -> int:
        import math
        return max(1, int(round(math.sqrt(self.stacks))))

    @property
    def mesh_avg_hops(self) -> float:
        """Average Manhattan distance between two random nodes of a side x side mesh (derived)."""
        n = self.mesh_side
        return 2 * (n * n - 1) / (3 * n) if n > 1 else 0.0

    @property
    def mesh_diameter(self) -> int:
        return 2 * (self.mesh_side - 1)

    @property
    def tRC_s(self) -> float:
        return self.nRC * self.tCK_s

    @property
    def rows_per_s(self) -> float:
        """Sustained row activations per second over the whole memory, limited per pseudo channel
        by both the per-bank row cycle and the four-activate window."""
        per_pc = min(self.banks_per_pc / self.tRC_s, 4.0 / (self.nFAW * self.tCK_s))
        return per_pc * self.pcs

    @property
    def internal_row_bw_Bps(self) -> float:
        return self.rows_per_s * self.row_bytes


# claude: decision start — HBM3 numbers are taken from the Ramulator2 HBM3 presets (which cite
# JESD238 Table 4) because the group's PIM simulators (AttAcc, KVChime) are built on them; the bank
# PE follows KVChime/AttAcc; the buffer-die fabric bandwidth has no source and is a swept parameter.
HBM3_PIM = PimConfig(
    name="hbm3_pim",
    stacks=1,
    channels_per_stack=16,       # source: reference — JEDEC JESD238 HBM3: 16 channels per stack
    pcs_per_channel=2,           # source: reference — Ramulator2 python/ramulator/dram/hbm3.py, org preset HBM3_4Gb
    banks_per_pc=16,             # source: reference — same preset: 4 bank groups x 4 banks
    row_bytes=1024,              # source: reference — same preset: 256 columns x 4 B (dq 32); 4 Gb / 2 PC / 16 banks / 2^14 rows = 1 KB
    access_bytes=32,             # source: reference — same preset: dq 32 bit x BL8 (nBL 2 at 2 beats/CK)
    tCK_s=625e-12,               # source: reference — Ramulator2 HBM3 timing preset HBM3_6400Mbps: tCK_ps 625
    nRC=72,                      # source: reference — same preset: nRC 72 (45 ns)
    nFAW=24,                     # source: reference — same preset: nFAW 24
    nRCDRD=31,                   # source: reference — same preset: nRCDRD 31
    ext_bw_Bps=819.2e9,          # source: derived — 1024-bit HBM3 interface x 6.4 Gbps (JESD238; preset rate 6400)
    pe_lanes=16,                 # source: reference — KVChime Sec. 3.3: 16 MAC lanes per bank (AttAcc datapath)
    pe_freq_hz=0.666e9,          # source: reference — KVChime Sec. 5.1: AttAcc baseline bank-logic 0.666 GHz
                                 #   (KVChime's own 1.3 GHz is ASAP7 pre-layout; used as the optimistic sweep point)
    query_slots=8,               # source: reference — KVChime Sec. 3.3, Table 1 note: M = 8 resident query slots
    fabric_bw_Bps=819.2e9,       # source: TBD[src] — buffer-die cross-bank shuffle bandwidth assumed equal to the
                                 #   external interface; swept 0.5x / 1x / 4x
    global_reduce_s=1e-6,        # source: TBD[src] — one all-bank OR/min/count reduction on the buffer die; swept
    cmd_latency_s=200e-9,        # source: reference — AMMA Sec. 4.2 (p. 5): 200 ns HBM access latency, used as the
                                 #   round-trip lower bound of one PIM command group; swept
    d2d_hop_s=15e-9,             # source: reference — AMMA Table 1 (p. 9): UCIe 3.0, 15 ns per hop
    d2d_bw_Bps=1.5e12,           # source: reference — AMMA Table 1 (p. 9): 1500 GB/s per D2D link
    d2d_links_per_stack=4,       # source: reference — AMMA Sec. 4.1 / 4.2: four D2D ports per cube, 2-D mesh
    notes="one HBM3 stack with bank-level PEs; KVChime-style multi-query slots",
)
# claude: decision end

HBM3_PIM_FAST = replace(HBM3_PIM, name="hbm3_pim_fast",
                        pe_freq_hz=1.3e9,     # source: reference — KVChime Sec. 3.3 / 5.1 / 5.6: 1.3 GHz (ASAP7 pre-layout)
                        pe_lanes=32,          # source: design — sweep point (2x lanes)
                        query_slots=32,       # source: design — sweep point
                        notes="optimistic bank PE: 1.3 GHz, 32 lanes, 32 query slots")

# claude: decision start — edge platform: REASON's 128-bit LPDDR5 interface = 8 x16 channels at
# 6.4 Gbps (8 x 16 bit x 6.4 Gbps = 102.4 GB/s, matching REASON's 104 GB/s); LPDDR5 organisation and
# timing from the Ramulator2 LPDDR5 presets (JESD209-5C); the bank PE is the same design as on HBM.
LPDDR5_PIM = PimConfig(
    name="lpddr5_pim",
    stacks=1,
    channels_per_stack=8,        # source: derived — REASON Sec. VII-A (p. 12): 128-bit LPDDR5 interface / 16-bit channels
    pcs_per_channel=1,           # source: reference — LPDDR5 has no pseudo channels (Ramulator2 lpddr5.py org preset: rank 1)
    banks_per_pc=16,             # source: reference — Ramulator2 lpddr5.py preset LPDDR5_8Gb_x16: 4 bank groups x 4 banks
    row_bytes=2048,              # source: reference — same preset: 1024 columns x 2 B (dq 16)
    access_bytes=32,             # source: reference — same preset: dq 16 bit x BL16 (nBL_min 2 at CKR 4:1)
    tCK_s=1250e-12,              # source: reference — Ramulator2 lpddr5.py timing preset LPDDR5_6400: tCK_ps 1250
    nRC=49,                      # source: reference — same preset: nRC 49 (61 ns)
    nFAW=16,                     # source: reference — lpddr5.py _resolve_nFAW: JESD209-5C Table 381 tFAW = 20 ns -> 16 CK
    nRCDRD=15,                   # source: reference — same preset: nRCD 15
    ext_bw_Bps=102.4e9,          # source: derived — 128 bit x 6.4 Gbps
    pe_lanes=16,                 # source: reference — KVChime Sec. 3.3 (design transferred to LPDDR)
    pe_freq_hz=0.666e9,          # source: reference — KVChime Sec. 5.1 (AttAcc bank logic)
    query_slots=8,               # source: reference — KVChime Sec. 3.3
    fabric_bw_Bps=102.4e9,       # source: TBD[src] — assumed equal to the external interface; swept
    global_reduce_s=1e-6,        # source: TBD[src] — swept
    cmd_latency_s=200e-9,        # source: reference — AMMA Sec. 4.2 (p. 5), HBM figure reused for LPDDR5 (no LPDDR
                                 #   source yet); swept
    d2d_hop_s=15e-9,             # source: reference — AMMA Table 1 (not used on the edge platform)
    d2d_bw_Bps=1.5e12,           # source: reference — AMMA Table 1 (not used on the edge platform)
    notes="edge LPDDR5-PIM: 8 x16 channels, 128 banks, bank PEs as on HBM",
)
# claude: decision end

# Multi-stack variants: 4 and 16 HBM3 stacks in a 2-D mesh of UCIe links (AMMA's 4x4 mesh of 16
# cubes, Sec. 4.1). Bank parallelism scales with the stacks; the cross-bank shuffle of a fused
# round then partly crosses stacks (HBM-to-HBM traffic), see pim/model.py.
HBM3_PIM_4S = replace(HBM3_PIM, name="hbm3_pim_4s", stacks=4,
                      fabric_bw_Bps=4 * 819.2e9,   # source: TBD[src] — per-stack fabric assumption scaled by stacks
                      notes="4 HBM3 stacks, 2x2 mesh")
HBM3_PIM_16S = replace(HBM3_PIM, name="hbm3_pim_16s", stacks=16,
                       fabric_bw_Bps=16 * 819.2e9, # source: TBD[src] — see above
                       notes="16 HBM3 stacks, 4x4 mesh (AMMA Sec. 4.1)")

PIM_CONFIGS = {c.name: c for c in (HBM3_PIM, HBM3_PIM_FAST, LPDDR5_PIM, HBM3_PIM_4S, HBM3_PIM_16S)}
