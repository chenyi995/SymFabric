"""Memory-side (bank-level PIM) execution model and the chip-vs-PIM placement selector.

Used only after the compute-side model shows an operator to be memory-bound. The model answers:
if this operator (or a fused group of operators of one fixpoint round) ran next to the DRAM banks,
how long would it take, counting row activations, bank-PE throughput, the cross-bank shuffle over
the buffer die, and the traffic that still has to cross the external interface.
"""
from .config import PimConfig, PIM_CONFIGS
from .model import pim_cost_group, fuse_groups, PimCost
from .selector import select_placement

__all__ = ["PimConfig", "PIM_CONFIGS", "pim_cost_group", "fuse_groups", "PimCost", "select_placement"]
