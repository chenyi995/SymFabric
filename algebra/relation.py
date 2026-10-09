"""Finite weighted relations R : D_1 x ... x D_n -> K with zero suppression.

A Relation is a schema (attribute names), a semiring, and a dict {key tuple: value} that holds
only values different from the semiring zero. Optional `domain` gives the size of each attribute's
index domain so that key widths and density can be reported in the trace; it does not change
semantics.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, Iterator

from .semiring import Semiring, BOOLEAN


_UID = itertools.count(1)


class Relation:
    __slots__ = ("schema", "sr", "data", "domain", "uid")

    def __init__(self, schema: Iterable[str], sr: Semiring, data: dict | None = None,
                 domain: dict[str, int] | None = None):
        self.uid = next(_UID)          # identity for operand-reuse tracking in the trace
        self.schema: tuple[str, ...] = tuple(schema)
        self.sr = sr
        self.domain: dict[str, int] = dict(domain) if domain else {}
        self.data: dict[tuple, Any] = {}
        if data:
            z = sr.is_zero
            self.data = {k: v for k, v in data.items() if not z(v)}

    # ---- constructors ----------------------------------------------------------------
    @classmethod
    def from_pairs(cls, schema, sr: Semiring, pairs: Iterable[tuple[tuple, Any]],
                   domain=None) -> "Relation":
        """Build from (key, value) pairs; duplicate keys are merged with the semiring add."""
        r = cls(schema, sr, None, domain)
        d = r.data
        add = sr.add
        for k, v in pairs:
            if k in d:
                d[k] = add(d[k], v)
            else:
                d[k] = v
        z = sr.is_zero
        for k in [k for k, v in d.items() if z(v)]:
            del d[k]
        return r

    @classmethod
    def from_set(cls, schema, keys: Iterable[tuple], domain=None) -> "Relation":
        r = cls(schema, BOOLEAN, None, domain)
        r.data = {tuple(k): True for k in keys}
        return r

    @classmethod
    def dense(cls, schema, sr: Semiring, values, domain: dict[str, int]) -> "Relation":
        """From a nested list / numpy array indexed in schema order."""
        r = cls(schema, sr, None, domain)
        shape = [domain[a] for a in r.schema]
        z = sr.is_zero

        def rec(prefix, arr, depth):
            if depth == len(shape):
                v = arr
                if hasattr(v, "item"):
                    v = v.item()
                if not z(v):
                    r.data[tuple(prefix)] = v
                return
            for i in range(shape[depth]):
                rec(prefix + [i], arr[i], depth + 1)

        rec([], values, 0)
        return r

    # ---- basic access ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[tuple]:
        return iter(self.data)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def get(self, key: tuple, default=None):
        if default is None:
            default = self.sr.zero
        return self.data.get(key, default)

    def __contains__(self, key) -> bool:
        return key in self.data

    @property
    def nnz(self) -> int:
        return len(self.data)

    def pos(self, attr: str) -> int:
        return self.schema.index(attr)

    def empty_like(self, schema=None) -> "Relation":
        return Relation(schema if schema is not None else self.schema, self.sr, None, self.domain)

    def copy(self) -> "Relation":
        r = Relation(self.schema, self.sr, None, self.domain)
        r.data = dict(self.data)
        return r

    # ---- sizes for the trace -------------------------------------------------------------
    def key_bits(self) -> int:
        bits = 0
        for a in self.schema:
            n = self.domain.get(a)
            bits += max(1, math.ceil(math.log2(n))) if n and n > 1 else (1 if n == 1 else 32)
        return bits

    def density(self) -> float | None:
        if not self.domain or any(a not in self.domain for a in self.schema):
            return None
        total = 1
        for a in self.schema:
            total *= self.domain[a]
        return len(self.data) / total if total else None

    # ---- comparison -------------------------------------------------------------------------
    def equals(self, other: "Relation", tol: float = 1e-9) -> bool:
        if set(self.schema) != set(other.schema):
            return False
        if self.schema != other.schema:
            perm = [other.schema.index(a) for a in self.schema]
            other_data = {tuple(k[p] for p in perm): v for k, v in other.data.items()}
        else:
            other_data = other.data
        if len(self.data) != len(other_data):
            return False
        eq = self.sr.equal
        for k, v in self.data.items():
            if k not in other_data or not eq(v, other_data[k], tol):
                return False
        return True

    def __repr__(self) -> str:
        return f"Relation({self.schema}, {self.sr.name}, nnz={len(self.data)})"
