"""Map an algebra trace onto an ArchConfig: one OpCost per OpRecord.

Residency: relations are tracked by uid in an LRU set bounded by sram_bytes. A relation larger
than the SRAM is never resident (it is streamed). Renames alias the input's residency; `update`
replaces the old state by the new one in place.

For every operator the mapper chooses the cheapest legal execution (e.g. random probes vs a
partitioned streaming pass for a probe table that does not fit on chip) and records the choice in
`detail["choice"]` so that the decision is auditable.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, asdict

from algebra.trace import OpRecord

from .config import ArchConfig
from . import model as M


@dataclass
class OpCost:
    idx: int
    op: str
    semiring: str
    tag: str
    iteration: int
    mode: str
    cycles_compute: float
    bytes_dram: float
    cycles_mem: float
    cycles: float
    bound: str                      # "compute" | "memory" | "none"
    useful_ops: int                 # semiring mul + add
    mul_ops: int                    # semiring mul only (one per hotspot event; utilisation basis)
    n_in: tuple
    n_out: int
    detail: dict = field(default_factory=dict)

    def as_dict(self):
        d = asdict(self)
        d["n_in"] = list(self.n_in)
        d["detail"] = {k: v for k, v in self.detail.items() if not isinstance(v, (list, dict))}
        return d


class Residency:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.used = 0
        self.items: "OrderedDict[int, int]" = OrderedDict()   # uid -> bytes, LRU order
        self.alias: dict[int, int] = {}                        # uid -> canonical uid

    def canon(self, uid: int) -> int:
        while uid in self.alias:
            uid = self.alias[uid]
        return uid

    def is_resident(self, uid: int) -> bool:
        u = self.canon(uid)
        if u in self.items:
            self.items.move_to_end(u)
            return True
        return False

    def insert(self, uid: int, nbytes: int) -> bool:
        """Make uid resident if it fits (evicting LRU). Returns True if resident afterwards."""
        u = self.canon(uid)
        if nbytes > self.capacity:
            self.items.pop(u, None)
            return False
        if u in self.items:
            self.used -= self.items[u]
            del self.items[u]
        while self.used + nbytes > self.capacity and self.items:
            _, b = self.items.popitem(last=False)
            self.used -= b
        self.items[u] = nbytes
        self.used += nbytes
        return True

    def drop(self, uid: int):
        u = self.canon(uid)
        if u in self.items:
            self.used -= self.items.pop(u)

    def add_alias(self, new_uid: int, old_uid: int):
        self.alias[new_uid] = self.canon(old_uid)


def _prod(xs):
    p = 1
    for x in xs:
        if not x:
            return None
        p *= x
    return p


def map_trace(records: list[OpRecord], cfg: ArchConfig) -> list[OpCost]:
    res = Residency(cfg.sram_bytes)
    bpc = cfg.bytes_per_cycle
    costs: list[OpCost] = []
    last_delta_for: dict[int, int] = {}    # canonical uid of a relation -> size of the last delta that built it

    for idx, r in enumerate(records):
        tuple_bytes_in = [math.ceil(b / 8) for b in r.in_bits]
        in_bytes = [n * tb for n, tb in zip(r.n_in, tuple_bytes_in)]
        out_tuple_bytes = math.ceil((r.key_bits + r.val_bits) / 8)
        out_bytes = r.n_out * out_tuple_bytes
        useful = r.mul_ops + r.add_ops
        muls = r.mul_ops
        detail: dict = {}
        mode = r.op
        compute = 0.0
        extra_bytes = 0.0

        if r.op == "rename":
            res.add_alias(r.out_uid, r.in_uids[0])
            costs.append(OpCost(idx, r.op, r.semiring, r.tag, r.iteration, "alias", 0, 0, 0, 0,
                                "none", 0, 0, r.n_in, r.n_out, {"alias": True}))
            continue

        # ---- operand reads --------------------------------------------------------------
        reads = 0.0
        resident_in = []
        for uid, nb in zip(r.in_uids, in_bytes):
            if res.is_resident(uid):
                resident_in.append(True)
            else:
                reads += nb
                res.insert(uid, nb)
                resident_in.append(res.is_resident(uid))

        # ---- operator-specific compute --------------------------------------------------------
        if r.op == "combine":
            h = r.hint
            force_dense = "merge" not in cfg.modes
            K = _prod(h.get("shared_dom") or [])
            volR = _prod(h.get("dom_R") or [])
            volS = _prod(h.get("dom_S") or [])
            can_dense = K is not None and volR is not None and volS is not None and K > 0
            if (h.get("dense") or force_dense) and can_dense and "dense" in cfg.modes:
                Mrows, Ncols = max(1, volR // K), max(1, volS // K)
                if h.get("fused_reduce"):
                    compute, det = M.dense_gemm_cycles(cfg, Mrows, K, Ncols)
                    mode = "dense_gemm"
                else:
                    # un-fused join of dense operands = broadcast elementwise product M x K x N
                    compute = M.ceil_div(Mrows * K * Ncols, cfg.pes)
                    det = {"M": Mrows, "K": K, "N": Ncols}
                    mode = "dense_elementwise"
                detail.update(det)
                if force_dense:
                    # dense execution touches the full domains, not only the non-zeros
                    dense_in = [volR * math.ceil(r.val_bits / 8), volS * math.ceil(r.val_bits / 8)]
                    detail["dense_volume"] = h.get("dense_volume")
                    useful = Mrows * K * Ncols          # padded work actually performed
                    muls = useful
                    reads = sum(b for b, rs in zip(dense_in, resident_in) if not rs)
                    out_bytes = Mrows * Ncols * math.ceil(r.val_bits / 8) if h.get("fused_reduce") \
                        else Mrows * K * Ncols * math.ceil(r.val_bits / 8)
            elif "merge" in cfg.modes:
                compute, det = M.merge_join_cycles(cfg, r.n_in[0], r.n_in[1], r.candidates, r.add_ops)
                mode = "merge_join"
                detail.update(det)
            else:
                # no merge mode and no domain information: fall back to candidate streaming
                compute = M.ceil_div(r.candidates + r.add_ops, cfg.stream_lanes)
                mode = "stream_fallback"
        elif r.op == "reduce":
            compute = M.stream_cycles(cfg, r.n_in[0])
            mode = "segmented_reduce"
            if out_bytes > cfg.sram_bytes:
                # accumulators do not fit: one partitioning pass over the input
                extra_bytes += 2 * in_bytes[0]
                compute += M.stream_cycles(cfg, r.n_in[0])
                detail["choice"] = "partitioned"
        elif r.op in ("filter", "scan"):
            compute = M.stream_cycles(cfg, r.n_in[0])
            mode = "stream"
            cu = res.canon(r.in_uids[0])
            if cu in last_delta_for:                       # delta lineage passes through unary ops
                last_delta_for[r.out_uid] = min(last_delta_for[cu], r.n_out)
        elif r.op == "mask":
            n_probe, n_table = r.n_in[0], r.n_in[1]
            table_resident = resident_in[1]
            prev = records[idx - 1] if idx > 0 else None
            if ("pq" in cfg.modes and prev is not None and prev.op == "select" and r.hint.get("negate")
                    and r.in_uids[1] == prev.out_uid and res.canon(r.in_uids[0]) == res.canon(prev.in_uids[0])):
                # removing the just-selected entries from the queue is the pop itself
                compute = M.pq_cycles(cfg, 0, n_table)
                mode = "pq_pop_delete"
                cu = res.canon(r.in_uids[0])
                if cu in last_delta_for:
                    last_delta_for[r.out_uid] = last_delta_for[cu]
            elif table_resident and "bitmap" in cfg.modes:
                compute = M.bitmap_probe_cycles(cfg, n_probe)
                mode = "onchip_probe"
            else:
                # (a) random probes into DRAM  vs  (b) partitioned streaming pass over both
                a_bytes = n_probe * cfg.random_access_bytes
                a_comp = M.bitmap_probe_cycles(cfg, n_probe)
                b_bytes = in_bytes[1] + 2 * in_bytes[0]
                b_comp = M.stream_cycles(cfg, n_probe + n_table)
                ta = max(a_comp, a_bytes / bpc)
                tb = max(b_comp, b_bytes / bpc)
                if ta <= tb:
                    compute, extra_bytes, mode = a_comp, a_bytes, "random_probe"
                else:
                    compute, extra_bytes, mode = b_comp, b_bytes, "partitioned_probe"
                detail["choice"] = mode
            cu = res.canon(r.in_uids[0])
            if cu in last_delta_for:                       # delta lineage passes through mask
                last_delta_for[r.out_uid] = min(last_delta_for[cu], r.n_out)
        elif r.op == "update":
            n_state, n_delta = r.n_in[0], r.n_in[1]
            state_uid_new = r.hint.get("state_uid", r.out_uid)
            state_bytes_new = r.hint.get("state", n_state) * tuple_bytes_in[0]
            if resident_in[0] and "bitmap" in cfg.modes:
                compute = M.stream_cycles(cfg, n_delta)
                mode = "onchip_update"
            else:
                a_bytes = 2 * n_delta * cfg.random_access_bytes          # read-modify-write per key
                a_comp = M.stream_cycles(cfg, n_delta)
                b_bytes = in_bytes[0] + state_bytes_new + in_bytes[1]    # stream-merge old state -> new
                b_comp = M.stream_cycles(cfg, n_state + n_delta)
                if max(a_comp, a_bytes / bpc) <= max(b_comp, b_bytes / bpc):
                    compute, extra_bytes, mode = a_comp, a_bytes, "random_rmw"
                else:
                    compute, extra_bytes, mode = b_comp, b_bytes, "stream_merge"
                detail["choice"] = mode
            # the new state replaces the old one in place
            res.drop(r.in_uids[0])
            res.insert(state_uid_new, state_bytes_new)
            last_delta_for[res.canon(state_uid_new)] = r.n_out
            detail["state_bytes"] = state_bytes_new
        elif r.op == "select":
            k = r.hint.get("k", r.n_out)
            n = r.n_in[0]
            if "pq" in cfg.modes:
                cu = res.canon(r.in_uids[0])
                inserts = last_delta_for.get(cu, n)
                compute = M.pq_cycles(cfg, inserts, k)
                mode = "systolic_pq"
                if n > cfg.pq_capacity:
                    compute += M.stream_cycles(cfg, n)
                    detail["choice"] = "pq_spill"
                detail["inserts"] = inserts
            else:
                compute = M.sort_select_cycles(cfg, n, k)
                mode = "sort_select"
        else:
            compute = M.stream_cycles(cfg, sum(r.n_in))
            mode = "stream"

        compute += cfg.op_launch_cycles

        # ---- output placement -------------------------------------------------------------
        if r.op == "update":
            writes = 0.0                       # state handled above; delta is consumed next
            res.insert(r.out_uid, out_bytes)
        else:
            if res.insert(r.out_uid, out_bytes):
                writes = 0.0
            else:
                writes = out_bytes             # spilled to DRAM
                detail["spill"] = True
        bytes_total = reads + extra_bytes + writes
        cyc_mem = bytes_total / bpc if bpc else 0.0
        cycles = max(compute, cyc_mem)
        bound = "memory" if cyc_mem > compute else "compute"
        detail["reads"] = reads
        detail["writes"] = writes
        detail["extra"] = extra_bytes
        costs.append(OpCost(idx, r.op, r.semiring, r.tag, r.iteration, mode, compute, bytes_total,
                            cyc_mem, cycles, bound, useful, muls, r.n_in, r.n_out, detail))
    return costs
