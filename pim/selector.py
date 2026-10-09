"""Placement: for each fused group choose chip (systolic model) or memory side (PIM model)."""
from __future__ import annotations

from dataclasses import dataclass, field

from algebra.trace import OpRecord
from systolicarch.config import ArchConfig
from systolicarch.mapper import OpCost

from .config import PimConfig
from .model import fuse_groups, pim_cost_group


@dataclass
class Placement:
    group: list
    chip_s: float
    pim_s: float | None
    choice: str                    # "chip" | "pim"
    shape: str
    pim_breakdown: dict = field(default_factory=dict)


def select_placement(records: list[OpRecord], chip_costs: list[OpCost], arch: ArchConfig,
                     pim: PimConfig, allow_pim: bool = True) -> tuple[list[Placement], dict]:
    groups = fuse_groups(records)
    placements = []
    t_chip_only = 0.0
    t_selected = 0.0
    for g in groups:
        chip_s = sum(chip_costs[k].cycles for k in g) / arch.freq_hz
        pc = pim_cost_group(pim, records, g) if allow_pim else None
        pim_s = pc.t_total_s if pc else None
        if pim_s is not None and pim_s < chip_s:
            choice, t = "pim", pim_s
        else:
            choice, t = "chip", chip_s
        placements.append(Placement(g, chip_s, pim_s, choice, pc.shape if pc else "none",
                                    {"rows": pc.t_rows_s, "pe": pc.t_pe_s, "shuffle": pc.t_shuffle_s,
                                     "ext": pc.t_ext_s, "global": pc.t_global_s, "cmd": pc.t_cmd_s} if pc else {}))
        t_chip_only += chip_s
        t_selected += t
    n_pim = sum(1 for p in placements if p.choice == "pim")
    summary = {"t_chip_only_s": t_chip_only, "t_selected_s": t_selected,
               "speedup_from_pim": (t_chip_only / t_selected) if t_selected else 1.0,
               "groups": len(groups), "groups_on_pim": n_pim,
               "time_share_on_pim": sum(p.pim_s for p in placements if p.choice == "pim") / t_selected
               if t_selected else 0.0}
    return placements, summary
