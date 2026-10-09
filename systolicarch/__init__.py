"""Analytical model of a reconfigurable semiring systolic array executing algebra traces.

Inputs: OpRecords from algebra.trace (sizes and counts only). Output: per-operator cycles, DRAM
bytes, execution mode, and whether the operator is compute- or memory-bound, under a named
ArchConfig whose every constant carries a source comment.

This is a first-order model (SCALE-Sim-style closed forms per mode), not a cycle-level simulator.
It is meant to answer "where does the time go and what is the bottleneck" per workload, and to
compare a reconfigurable array against a fixed-MAC (TPU-like) array on the same traces.
"""
from .config import ArchConfig, CONFIGS
from .mapper import map_trace, OpCost
from .report import summarize, roofline_rows

__all__ = ["ArchConfig", "CONFIGS", "map_trace", "OpCost", "summarize", "roofline_rows"]
