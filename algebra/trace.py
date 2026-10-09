"""Execution trace: one record per operator instance.

The trace is the interface between the algebra (what was computed) and the architecture models
(how long it would take). Records carry sizes and counts only; they never carry data.

Counting conventions (used consistently by every workload's work-efficiency ratio):
  candidates : number of (left, right) tuple pairs whose keys matched and were combined, or
               the number of input tuples examined by a unary operator.
  mul_ops    : number of semiring "mul" evaluations.
  add_ops    : number of semiring "add" evaluations.
  n_out      : number of non-zero output tuples.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, asdict
from typing import Iterator


@dataclass
class OpRecord:
    op: str                       # combine | reduce | filter | mask | update | select | rename | project | scan
    semiring: str
    n_in: tuple                   # nnz of each input relation
    n_out: int
    candidates: int
    mul_ops: int
    add_ops: int
    key_bits: int                 # bits of one output key
    val_bits: int                 # bits of one value
    tag: str = ""                 # workload phase, set by the caller through Trace.tag()
    iteration: int = -1           # fixpoint / iterate round, -1 outside loops
    hint: dict = field(default_factory=dict)   # e.g. {"shared_attrs": 1, "dense": True, "Q": 64}
    in_uids: tuple = ()           # identities of input relations (operand-reuse tracking)
    out_uid: int = 0              # identity of the output relation
    in_bits: tuple = ()           # (key_bits + val_bits) of each input tuple

    def as_dict(self):
        d = asdict(self)
        d["n_in"] = list(self.n_in)
        d["in_uids"] = list(self.in_uids)
        d["in_bits"] = list(self.in_bits)
        return d


class Trace:
    def __init__(self):
        self.records: list[OpRecord] = []
        self._tag = ""
        self._iteration = -1

    def add(self, rec: OpRecord):
        rec.tag = self._tag
        rec.iteration = self._iteration
        self.records.append(rec)

    @contextlib.contextmanager
    def tag(self, name: str) -> Iterator[None]:
        prev = self._tag
        self._tag = name if not prev else f"{prev}/{name}"
        try:
            yield
        finally:
            self._tag = prev

    def set_iteration(self, k: int):
        self._iteration = k

    # ---- summaries -------------------------------------------------------------------
    def total(self, field_name: str) -> int:
        return sum(getattr(r, field_name) for r in self.records)

    @property
    def sr_ops(self) -> int:
        return self.total("mul_ops") + self.total("add_ops")

    def by_op(self) -> dict:
        out: dict[str, dict] = {}
        for r in self.records:
            d = out.setdefault(r.op, {"count": 0, "candidates": 0, "n_out": 0, "mul_ops": 0,
                                      "add_ops": 0})
            d["count"] += 1
            d["candidates"] += r.candidates
            d["n_out"] += r.n_out
            d["mul_ops"] += r.mul_ops
            d["add_ops"] += r.add_ops
        return out

    def summary(self) -> dict:
        return {
            "records": len(self.records),
            "candidates": self.total("candidates"),
            "n_out": self.total("n_out"),
            "mul_ops": self.total("mul_ops"),
            "add_ops": self.total("add_ops"),
            "sr_ops": self.sr_ops,
            "by_op": self.by_op(),
        }


_ACTIVE: list[Trace] = [Trace()]


def current_trace() -> Trace:
    return _ACTIVE[-1]


@contextlib.contextmanager
def use_trace(trace: Trace | None = None) -> Iterator[Trace]:
    """Run a block with a fresh (or given) trace as the active one."""
    t = trace if trace is not None else Trace()
    _ACTIVE.append(t)
    try:
        yield t
    finally:
        _ACTIVE.pop()
