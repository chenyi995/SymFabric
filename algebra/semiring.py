"""Semirings (value algebras).

A semiring (K, add, mul, zero, one) fixes *what* is computed: add (the symbol "+" below) merges
candidates that land on the same output key; mul (the symbol "*") combines the values of one
candidate. The relation layer stores only values different from `zero`.

The semirings listed here are the ones that appear in the workload decompositions under
softwareworkload/. Adding one is a data change, not a code change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Semiring:
    name: str
    zero: Any
    one: Any
    add: Callable[[Any, Any], Any]
    mul: Callable[[Any, Any], Any]
    idempotent: bool            # add(a, a) == a  (OR, min, max)   -> deltas are exact
    selective: bool             # add(a, b) in {a, b}  (min, max, OR) -> argmin/argmax exist
    numeric: bool               # values are floats; equality needs a tolerance
    # "better" order for SELECT: returns True if a should rank before b. None if no order.
    better: Callable[[Any, Any], bool] | None = None
    bits: int = 32              # storage width of one value; used only by the trace

    def sum(self, values):
        acc = self.zero
        for v in values:
            acc = self.add(acc, v)
        return acc

    def prod(self, values):
        acc = self.one
        for v in values:
            acc = self.mul(acc, v)
        return acc

    def is_zero(self, v) -> bool:
        if self.numeric and isinstance(self.zero, float) and math.isinf(self.zero):
            return v == self.zero
        return v == self.zero

    def equal(self, a, b, tol: float = 1e-9) -> bool:
        if not self.numeric:
            return a == b
        if a == b:
            return True
        if isinstance(a, float) and isinstance(b, float):
            if math.isinf(a) or math.isinf(b):
                return a == b
            return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
        return a == b


def _lse(a: float, b: float) -> float:
    """log(e^a + e^b) with -inf as the zero element."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log1p(math.exp(-abs(a - b)))


BOOLEAN = Semiring("boolean", False, True, lambda a, b: a or b, lambda a, b: a and b,
                   idempotent=True, selective=True, numeric=False, better=None, bits=1)
COUNTING = Semiring("counting", 0, 1, lambda a, b: a + b, lambda a, b: a * b,
                    idempotent=False, selective=False, numeric=False, better=None, bits=64)
PROB = Semiring("prob", 0.0, 1.0, lambda a, b: a + b, lambda a, b: a * b,
                idempotent=False, selective=False, numeric=True,
                better=lambda a, b: a > b, bits=32)
LOGPROB = Semiring("logprob", -math.inf, 0.0, _lse, lambda a, b: a + b,
                   idempotent=False, selective=False, numeric=True,
                   better=lambda a, b: a > b, bits=32)
VITERBI = Semiring("viterbi", -math.inf, 0.0, max, lambda a, b: a + b,       # log-domain (max,+)
                   idempotent=True, selective=True, numeric=True,
                   better=lambda a, b: a > b, bits=32)
MAXPLUS = VITERBI
MINPLUS = Semiring("minplus", math.inf, 0.0, min, lambda a, b: a + b,          # shortest path
                   idempotent=True, selective=True, numeric=True,
                   better=lambda a, b: a < b, bits=32)
REAL = Semiring("real", 0.0, 1.0, lambda a, b: a + b, lambda a, b: a * b,   # (+, x) over R
                idempotent=False, selective=False, numeric=True,
                better=lambda a, b: a > b, bits=32)
BOTTLENECK = Semiring("bottleneck", math.inf, -math.inf, min, max,               # (min, max): h^max
                      idempotent=True, selective=True, numeric=True,
                      better=lambda a, b: a < b, bits=32)
FUZZY = Semiring("fuzzy", 0.0, 1.0, max, min,                                    # (max, min)
                 idempotent=True, selective=True, numeric=True,
                 better=lambda a, b: a > b, bits=32)

SEMIRINGS = {s.name: s for s in (BOOLEAN, COUNTING, PROB, LOGPROB, VITERBI, MINPLUS, FUZZY,
                                 REAL, BOTTLENECK)}
