"""Closed-form cycle models for each execution mode.

All formulas return compute cycles for one operator instance. Memory time is handled by the mapper.
"""
from __future__ import annotations

import math

from .config import ArchConfig


def ceil_div(a: int, b: int) -> int:
    return -(-a // b) if b else 0


def dense_gemm_cycles(cfg: ArchConfig, M: int, K: int, N: int) -> tuple[int, dict]:
    """Weight-stationary semiring GEMM: stationary operand K x N tiled onto the array, M rows
    stream through. Returns (cycles, detail).

    Classical WS (Tessera Sec. II-B, p. 3): Q back-to-back folds of M rows on a D x D array take
    Q*M + 3D - 2 cycles; a fold with M < D still needs D cycles to load its weights (Sec. III-B: the
    first weight column is ready after D - 1 cycles).
    Twist-torus (Tessera Sec. III-C, p. 5): fold latency M + 2D - 1, steady-state II = max(M, D).
    Fracturing (Tessera Sec. IV): a weight tile narrower than the array uses a strip of width
    ceil(N / d) * d, and the remaining strips run other tiles of the same GEMM in parallel.
    """
    D = cfg.rows
    if cfg.strip_cols < cfg.cols:
        cols_eff = min(cfg.cols, max(cfg.strip_cols, ceil_div(N, cfg.strip_cols) * cfg.strip_cols))
    else:
        cols_eff = cfg.cols
    parallel_strips = max(1, cfg.cols // cols_eff)
    tiles = ceil_div(K, D) * ceil_div(N, cols_eff)
    tiles_seq = ceil_div(tiles, parallel_strips)
    if cfg.fill_drain == "twist":
        cycles = tiles_seq * max(M, D) + 2 * D - 1
    else:
        cycles = tiles_seq * max(M, D) + 3 * D - 2
    useful = M * K * N
    util = useful / (cycles * cfg.pes) if cycles else 0.0
    return cycles, {"M": M, "K": K, "N": N, "tiles": tiles, "cols_eff": cols_eff,
                    "parallel_strips": parallel_strips, "util": util}


def merge_join_cycles(cfg: ArchConfig, n_left: int, n_right: int, candidates: int,
                      add_ops: int) -> tuple[int, dict]:
    """Sorted-stream merge / intersection join on `merge_lanes` units. Each unit consumes one
    input element per cycle and emits at most one matched candidate per cycle; the semiring mul
    and the fused accumulate are pipelined behind the emission. Conservative serial form:
    (inputs + candidates) / lanes. Skewed fan-out is captured by `candidates`.
    """
    lanes = cfg.merge_lanes
    cycles = ceil_div(n_left + n_right + candidates, lanes)
    util = (candidates + add_ops) / (cycles * cfg.pes) if cycles else 0.0
    return cycles, {"lanes": lanes, "util": util}


def stream_cycles(cfg: ArchConfig, n: int) -> int:
    """Elementwise pass over n tuples on stream_lanes."""
    return ceil_div(n, cfg.stream_lanes)


def bitmap_probe_cycles(cfg: ArchConfig, n_probe: int) -> int:
    """Probe n keys against an on-chip bitmap / hash table, one probe per lane per cycle."""
    return ceil_div(n_probe, cfg.stream_lanes)


def pq_cycles(cfg: ArchConfig, inserts: int, pops: int) -> int:
    """Systolic priority queue (Leiserson 1979): one insert or one extract-min per cycle."""
    return inserts + pops


def sort_select_cycles(cfg: ArchConfig, n: int, k: int) -> int:
    """Selection without a priority queue: a bitonic-style pass, n log2 n compare-exchanges on
    stream_lanes comparators (upper bound used for the TPU-like baseline)."""
    if n <= 1:
        return 1
    return ceil_div(int(n * math.log2(n)), cfg.stream_lanes)
