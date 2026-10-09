"""Cost of executing operators memory-side.

Two execution shapes are modelled:

1. Multi-query scan (MQ): the large operand (knowledge base: clause DB, edge relation, codebook)
   is bank-interleaved and streamed row by row inside every bank; the small operand (Q queries)
   is resident in the bank PE's query slots and every streamed tuple is compared against all
   resident queries. Work is n_kb * Q compares regardless of how many match (no index), done at
   pe_lanes per cycle; the row stream is bounded by tRC / tFAW. Q > query_slots costs passes.

2. Partitioned fused round: for a fixpoint round combine -> mask -> update, the join-side
   operand is partitioned by join key across banks, the delta is routed to owning banks, matched
   candidates are shuffled over the buffer die to the banks owning the state partition, and the
   probe + update happen bank-locally. Only the delta (in) and the changed flags (out) cross the
   external interface.

Both return seconds. Operators with no memory-side form (select, dense GEMM) return None.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from algebra.trace import OpRecord

from .config import PimConfig


@dataclass
class PimCost:
    records: list                 # indices of the OpRecords in the group
    shape: str                    # "mq_scan" | "fused_round" | "bank_stream" | "none"
    t_rows_s: float = 0.0         # row-activation-bound time
    t_pe_s: float = 0.0          # bank-PE compute time
    t_shuffle_s: float = 0.0     # cross-bank traffic over the buffer die
    t_ext_s: float = 0.0         # traffic over the external interface (inputs in, results out)
    t_global_s: float = 0.0      # all-bank reductions / termination flags
    t_cmd_s: float = 0.0         # fixed command round-trip of the group
    detail: dict = field(default_factory=dict)

    @property
    def t_total_s(self) -> float:
        # rows and PE overlap inside a bank (double-buffered row reads); the rest is serial
        return (max(self.t_rows_s, self.t_pe_s) + self.t_shuffle_s + self.t_ext_s + self.t_global_s
                + self.t_cmd_s)


def _tuple_bytes(bits: int) -> int:
    return math.ceil(bits / 8)


def _rows(nbytes: float, cfg: PimConfig) -> float:
    return math.ceil(nbytes / cfg.row_bytes)


def mq_scan(cfg: PimConfig, n_kb: int, kb_tuple_bytes: int, n_q: int, q_tuple_bytes: int,
            n_hits: int, out_tuple_bytes: int, hits_leave_memory: bool = True) -> PimCost:
    """Scan n_kb tuples against n_q resident queries in every bank."""
    passes = max(1, math.ceil(n_q / cfg.query_slots))
    kb_bytes = n_kb * kb_tuple_bytes
    rows = _rows(kb_bytes, cfg)
    t_rows = passes * rows / cfg.rows_per_s
    q_per_pass = min(n_q, cfg.query_slots)
    compares = n_kb * q_per_pass * passes
    t_pe = compares / (cfg.pe_lanes * cfg.banks * cfg.pe_freq_hz)
    # queries are broadcast to every pseudo channel once per pass
    t_ext = passes * n_q * q_tuple_bytes * cfg.pcs / cfg.ext_bw_Bps
    if hits_leave_memory:
        t_ext += n_hits * out_tuple_bytes / cfg.ext_bw_Bps
    return PimCost([], "mq_scan", t_rows_s=t_rows, t_pe_s=t_pe, t_ext_s=t_ext,
                   t_cmd_s=cfg.cmd_latency_s * passes,
                   detail={"passes": passes, "rows": rows, "compares": compares})


def fused_round(cfg: PimConfig, n_delta: int, delta_tuple_bytes: int, n_kb: int,
                kb_tuple_bytes: int, candidates: int, cand_tuple_bytes: int, n_state: int,
                state_tuple_bytes: int, n_changed: int) -> PimCost:
    """combine(delta, kb) -> mask(., state) -> update(state, .) executed inside the banks."""
    # delta enters memory-side once (routed by join key to owning banks); every stack has its
    # own interface, so the aggregate link bandwidth scales with the stack count
    ext_bw = cfg.ext_bw_Bps * cfg.stacks
    t_ext = n_delta * delta_tuple_bytes / ext_bw
    # join: each distinct delta key opens the rows holding its kb partition; bound by the
    # number of delta tuples (one row each, worst case) and by the kb size (all rows at most)
    rows_join = min(n_delta, _rows(n_kb * kb_tuple_bytes, cfg))
    t_rows = rows_join / cfg.rows_per_s
    t_pe = candidates / (cfg.pe_lanes * cfg.banks * cfg.pe_freq_hz)
    # shuffle candidates to the state owners: with S stacks a fraction (S-1)/S of the bytes leaves
    # the stack over the D2D mesh (HBM-to-HBM traffic); the mesh carries stacks x links x bw,
    # diluted by the average hop count of a uniform shuffle
    shuffle_bytes = candidates * cand_tuple_bytes
    if cfg.stacks > 1:
        remote = shuffle_bytes * (cfg.stacks - 1) / cfg.stacks
        local = shuffle_bytes - remote
        d2d_total = cfg.stacks * cfg.d2d_links_per_stack * cfg.d2d_bw_Bps / max(1.0, cfg.mesh_avg_hops)
        t_shuffle = max(local / cfg.fabric_bw_Bps, remote / d2d_total) + cfg.mesh_diameter * cfg.d2d_hop_s
    else:
        t_shuffle = shuffle_bytes / cfg.fabric_bw_Bps
    # probe + update: candidates sorted per bank open at most one row per candidate and at most
    # every row of the state once
    rows_probe = min(candidates, _rows(n_state * state_tuple_bytes, cfg))
    t_rows += rows_probe / cfg.rows_per_s
    t_pe += candidates / (cfg.pe_lanes * cfg.banks * cfg.pe_freq_hz)
    # changed flags / new delta leave memory for the controller
    t_ext += n_changed * state_tuple_bytes / ext_bw
    # the termination flag / changed-set OR crosses the mesh diameter once per round
    t_global = cfg.global_reduce_s + (cfg.mesh_diameter * cfg.d2d_hop_s if cfg.stacks > 1 else 0.0)
    return PimCost([], "fused_round", t_rows_s=t_rows, t_pe_s=t_pe, t_shuffle_s=t_shuffle,
                   t_ext_s=t_ext, t_global_s=t_global, t_cmd_s=cfg.cmd_latency_s,
                   detail={"rows_join": rows_join, "rows_probe": rows_probe,
                           "shuffle_bytes": shuffle_bytes, "stacks": cfg.stacks})


def bank_stream(cfg: PimConfig, n_in: int, in_tuple_bytes: int, n_out: int,
                out_tuple_bytes: int) -> PimCost:
    """filter / scan / reduce over a bank-interleaved relation; results come back over the link."""
    rows = _rows(n_in * in_tuple_bytes, cfg)
    t_rows = rows / cfg.rows_per_s
    t_pe = n_in / (cfg.pe_lanes * cfg.banks * cfg.pe_freq_hz)
    t_ext = n_out * out_tuple_bytes / cfg.ext_bw_Bps
    return PimCost([], "bank_stream", t_rows_s=t_rows, t_pe_s=t_pe, t_ext_s=t_ext,
                   t_global_s=cfg.global_reduce_s, t_cmd_s=cfg.cmd_latency_s, detail={"rows": rows})


def fuse_groups(records: list[OpRecord]) -> list[list[int]]:
    """Group consecutive combine -> [rename]* -> mask -> update records of the same iteration into
    one fused round; everything else is a singleton group."""
    groups: list[list[int]] = []
    i = 0
    n = len(records)
    while i < n:
        r = records[i]
        if r.op == "combine" and i + 2 < n:
            j = i + 1
            while j < n and records[j].op == "rename":
                j += 1
            if (j + 1 < n and records[j].op == "mask" and records[j + 1].op == "update"
                    and records[j].iteration == r.iteration):
                groups.append(list(range(i, j + 2)))
                i = j + 2
                continue
        groups.append([i])
        i += 1
    return groups


def pim_cost_group(cfg: PimConfig, records: list[OpRecord], group: list[int]) -> PimCost | None:
    recs = [records[k] for k in group]
    ops = [r.op for r in recs]
    if "combine" in ops and "mask" in ops and "update" in ops:
        c = recs[0]
        m = next(r for r in recs if r.op == "mask")
        u = next(r for r in recs if r.op == "update")
        # delta = the smaller combine input, kb = the larger
        small, large = (0, 1) if c.n_in[0] <= c.n_in[1] else (1, 0)
        pc = fused_round(cfg, n_delta=c.n_in[small], delta_tuple_bytes=_tuple_bytes(c.in_bits[small]),
                         n_kb=c.n_in[large], kb_tuple_bytes=_tuple_bytes(c.in_bits[large]),
                         candidates=c.candidates, cand_tuple_bytes=_tuple_bytes(c.key_bits + c.val_bits),
                         n_state=m.n_in[1], state_tuple_bytes=_tuple_bytes(m.in_bits[1]),
                         n_changed=u.n_out)
        pc.records = list(group)
        return pc
    r = recs[0]
    if r.op == "combine":
        small, large = (0, 1) if r.n_in[0] <= r.n_in[1] else (1, 0)
        if r.hint.get("dense"):
            return None                                   # dense GEMMs stay on the array
        pc = mq_scan(cfg, n_kb=r.n_in[large], kb_tuple_bytes=_tuple_bytes(r.in_bits[large]),
                     n_q=max(1, r.n_in[small]), q_tuple_bytes=_tuple_bytes(r.in_bits[small]),
                     n_hits=r.n_out, out_tuple_bytes=_tuple_bytes(r.key_bits + r.val_bits))
        pc.records = list(group)
        return pc
    if r.op in ("filter", "scan", "reduce"):
        pc = bank_stream(cfg, r.n_in[0], _tuple_bytes(r.in_bits[0]), r.n_out,
                         _tuple_bytes(r.key_bits + r.val_bits))
        pc.records = list(group)
        return pc
    if r.op in ("mask", "update"):
        # probe stream against a bank-resident table: rows opened bounded by probes and table rows
        n_probe, n_table = r.n_in[0], r.n_in[1]
        rows = min(n_probe, _rows(n_table * _tuple_bytes(r.in_bits[1]), cfg))
        pc = PimCost(list(group), "bank_probe",
                     t_rows_s=rows / cfg.rows_per_s,
                     t_pe_s=n_probe / (cfg.pe_lanes * cfg.banks * cfg.pe_freq_hz),
                     t_ext_s=(n_probe * _tuple_bytes(r.in_bits[0]) + r.n_out * _tuple_bytes(r.key_bits + r.val_bits)) / cfg.ext_bw_Bps,
                     t_global_s=cfg.global_reduce_s, t_cmd_s=cfg.cmd_latency_s, detail={"rows": rows})
        return pc
    return None
