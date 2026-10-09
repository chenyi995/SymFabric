"""Aggregate OpCosts into per-workload summaries and roofline rows."""
from __future__ import annotations

from collections import defaultdict

from .config import ArchConfig
from .mapper import OpCost


def summarize(costs: list[OpCost], cfg: ArchConfig) -> dict:
    tot_cyc = sum(c.cycles for c in costs)
    tot_comp = sum(c.cycles_compute for c in costs)
    tot_mem = sum(c.cycles_mem for c in costs)
    tot_bytes = sum(c.bytes_dram for c in costs)
    useful = sum(c.useful_ops for c in costs)
    muls = sum(c.mul_ops for c in costs)
    by_mode = defaultdict(float)
    by_bound = defaultdict(float)
    by_op = defaultdict(float)
    for c in costs:
        by_mode[c.mode] += c.cycles
        by_bound[c.bound] += c.cycles
        by_op[c.op] += c.cycles
    top = sorted(costs, key=lambda c: -c.cycles)[:3]
    return {
        "config": cfg.name,
        "ops": len(costs),
        "cycles": tot_cyc,
        "time_ms": tot_cyc / cfg.freq_hz * 1e3,
        "cycles_compute_sum": tot_comp,
        "cycles_mem_sum": tot_mem,
        "dram_bytes": tot_bytes,
        "avg_bw_util": (tot_bytes / (tot_cyc / cfg.freq_hz)) / cfg.dram_bw_Bps if tot_cyc else 0.0,
        "useful_ops": useful,
        "mul_ops": muls,
        # utilisation counts one multiplication (one hotspot event) per PE per cycle; the fused
        # add of a MAC or of a merge lane is not counted twice
        "pe_util": muls / (tot_cyc * cfg.pes) if tot_cyc else 0.0,
        "frac_memory_bound": by_bound.get("memory", 0.0) / tot_cyc if tot_cyc else 0.0,
        "compute_lower_bound_cycles": muls / cfg.pes,
        "memory_lower_bound_cycles": tot_bytes / cfg.bytes_per_cycle if cfg.bytes_per_cycle else 0.0,
        "by_mode": dict(by_mode),
        "by_op": dict(by_op),
        "top_ops": [(c.op, c.mode, c.tag, round(c.cycles), c.bound) for c in top],
    }


def roofline_rows(costs: list[OpCost], cfg: ArchConfig) -> list[dict]:
    rows = []
    for c in costs:
        if c.cycles <= 0:
            continue
        rows.append({
            "idx": c.idx, "op": c.op, "mode": c.mode, "tag": c.tag,
            "intensity_ops_per_byte": (c.useful_ops / c.bytes_dram) if c.bytes_dram else float("inf"),
            "ops_per_cycle": c.useful_ops / c.cycles,
            "bound": c.bound, "cycles": c.cycles, "bytes": c.bytes_dram,
        })
    return rows
