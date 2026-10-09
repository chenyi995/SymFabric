"""The operators. Each records one OpRecord in the active trace.

  rename   (R, mapping)             INDEX   : attribute renaming, metadata only (no record)
  project  (R, keep)                REDUCE  : alias of reduce
  combine  (R, S, keep=None)        COMBINE : natural semiring join on shared attributes,
                                              optionally fused with a REDUCE onto `keep`
  reduce   (R, keep)                REDUCE  : semiring add over the dropped attributes
  filter   (R, pred)                FILTER  : keep tuples satisfying pred(key, value)
  mask     (R, S, negate=False)     FILTER  : semijoin (keep keys whose shared part is in S)
                                              or antijoin (negate=True)
  update   (X, D)                   UPDATE  : X'(k) = X(k) add D(k); returns (X', changed)
  select_k (R, k)                   SELECT  : the k best values by the semiring's order
  scan     (R, fn, fn_name)         map     : elementwise O(1) function, may re-key

Implementation is hash-based so that the recorded `candidates` reflect a work-efficient
execution (matching pairs only), never a nested-loop product.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any, Callable, Iterable

from .relation import Relation
from .semiring import Semiring
from .trace import OpRecord, current_trace


def _rec(op, sr: Semiring, ins, out: Relation, candidates, mul_ops, add_ops, hint=None):
    current_trace().add(OpRecord(op=op, semiring=sr.name, n_in=tuple(len(r) for r in ins),
                                 n_out=len(out), candidates=candidates, mul_ops=mul_ops,
                                 add_ops=add_ops, key_bits=out.key_bits(), val_bits=sr.bits,
                                 hint=hint or {}, in_uids=tuple(r.uid for r in ins),
                                 out_uid=out.uid,
                                 in_bits=tuple(r.key_bits() + r.sr.bits for r in ins)))


# ---------------------------------------------------------------------------------------------
def rename(R: Relation, mapping: dict[str, str]) -> Relation:
    out = Relation([mapping.get(a, a) for a in R.schema], R.sr, None,
                   {mapping.get(a, a): n for a, n in R.domain.items()})
    out.data = R.data          # shared: rename never copies
    _rec("rename", R.sr, (R,), out, 0, 0, 0, {"alias": True})
    return out


def _shared(R: Relation, S: Relation, on) -> list[str]:
    if on is not None:
        return list(on)
    return [a for a in R.schema if a in S.schema]


def combine(R: Relation, S: Relation, keep: Iterable[str] | None = None,
            on: Iterable[str] | None = None) -> Relation:
    """Natural semiring join. Output schema = R.schema + (S.schema minus shared), or `keep`.

    With `keep`, the join is fused with a reduce: values landing on the same kept key are merged
    with the semiring add as they are produced (this is how a hardware pipeline would do it).
    """
    sr = R.sr
    assert sr is S.sr, f"semiring mismatch {R.sr.name} vs {S.sr.name}"
    shared = _shared(R, S, on)
    r_sh = [R.pos(a) for a in shared]
    s_sh = [S.pos(a) for a in shared]
    s_rest = [i for i, a in enumerate(S.schema) if a not in shared]
    out_schema = list(R.schema) + [S.schema[i] for i in s_rest]
    domain = dict(S.domain)
    domain.update(R.domain)
    keep_list = list(keep) if keep is not None else None
    if keep_list is not None:
        keep_pos = [out_schema.index(a) for a in keep_list]
    # build hash on the smaller side
    build_small = len(R) <= len(S)
    B, P = (R, S) if build_small else (S, R)
    b_sh = r_sh if build_small else s_sh
    p_sh = s_sh if build_small else r_sh
    index: dict[tuple, list] = defaultdict(list)
    for k, v in B.items():
        index[tuple(k[i] for i in b_sh)].append((k, v))
    out: dict[tuple, Any] = {}
    add, mul = sr.add, sr.mul
    cand = 0
    adds = 0
    for pk, pv in P.items():
        bucket = index.get(tuple(pk[i] for i in p_sh))
        if not bucket:
            continue
        for bk, bv in bucket:
            cand += 1
            rk, rv, sk, sv = (bk, bv, pk, pv) if build_small else (pk, pv, bk, bv)
            val = mul(rv, sv)
            key = rk + tuple(sk[i] for i in s_rest)
            if keep_list is not None:
                key = tuple(key[i] for i in keep_pos)
            if key in out:
                out[key] = add(out[key], val)
                adds += 1
            else:
                out[key] = val
    res = Relation(keep_list if keep_list is not None else out_schema, sr, out, domain)
    # dense_volume = product of the domain sizes over all attributes of the join (M*K*N for a
    # matrix product): the number of multiplications a dense execution would perform.
    all_attrs = list(R.schema) + [S.schema[i] for i in s_rest]
    dense_volume = 1
    for a in all_attrs:
        n = domain.get(a)
        if not n:
            dense_volume = None
            break
        dense_volume *= n
    _rec("combine", sr, (R, S), res, cand, cand, adds,
         {"shared_attrs": len(shared), "build": len(B), "probe": len(P),
          "fused_reduce": keep_list is not None,
          "dense": (R.density() or 0) > 0.5 and (S.density() or 0) > 0.5,
          "dense_volume": dense_volume,
          "dom_R": [R.domain.get(a) for a in R.schema], "dom_S": [S.domain.get(a) for a in S.schema],
          "shared_dom": [domain.get(a) for a in shared]})
    return res


def reduce(R: Relation, keep: Iterable[str], use_mul: bool = False) -> Relation:
    """Group by `keep` and fold the dropped attributes with the semiring add (default) or, with
    use_mul=True, with the semiring mul (a product node of a probabilistic circuit, the sum of
    precondition costs in a (min,+) planning heuristic). Folding with mul is a fold, not a
    semiring aggregation: it is only meaningful when every group is complete.

    WARNING (measured in softwareworkload/prob_circuit, 2026-09-25): relations are zero-suppressed,
    so a group member whose value is the semiring zero is absent and the mul-fold silently skips
    it; the fold then returns the product of the non-zero members instead of zero. Use a chain of
    binary combine() calls (which see the absence as a missing match) or guarantee that no group
    member can be zero (e.g. count the members with the COUNTING semiring and mask incomplete
    groups) before using use_mul=True."""
    keep_list = list(keep)
    pos = [R.pos(a) for a in keep_list]
    sr = R.sr
    fold = sr.mul if use_mul else sr.add
    out: dict[tuple, Any] = {}
    folds = 0
    for k, v in R.items():
        key = tuple(k[i] for i in pos)
        if key in out:
            out[key] = fold(out[key], v)
            folds += 1
        else:
            out[key] = v
    res = Relation(keep_list, sr, out, {a: R.domain[a] for a in keep_list if a in R.domain})
    _rec("reduce", sr, (R,), res, len(R), folds if use_mul else 0,
         0 if use_mul else folds, {"groups": len(out), "use_mul": use_mul})
    return res


project = reduce


def filter(R: Relation, pred: Callable[[tuple, Any], bool], name: str = "pred") -> Relation:
    out = {k: v for k, v in R.items() if pred(k, v)}
    res = Relation(R.schema, R.sr, out, R.domain)
    _rec("filter", R.sr, (R,), res, len(R), 0, 0, {"pred": name})
    return res


def mask(R: Relation, S: Relation, negate: bool = False, on=None) -> Relation:
    """Semijoin (negate=False): keep R tuples whose shared attributes occur in S's support.
    Antijoin (negate=True): keep R tuples whose shared attributes do NOT occur in S.
    Only S's support matters, never its values (this is set difference, not subtraction)."""
    shared = _shared(R, S, on)
    r_sh = [R.pos(a) for a in shared]
    s_sh = [S.pos(a) for a in shared]
    if s_sh == list(range(len(S.schema))):
        support = S.data.keys()
    else:
        support = {tuple(k[i] for i in s_sh) for k in S.keys()}
    if r_sh == list(range(len(R.schema))):
        out = {k: v for k, v in R.items() if (k in support) != negate}
    else:
        out = {k: v for k, v in R.items() if (tuple(k[i] for i in r_sh) in support) != negate}
    res = Relation(R.schema, R.sr, out, R.domain)
    _rec("mask", R.sr, (R, S), res, len(R), 0, 0,
         {"negate": negate, "probe_table": len(S), "shared_attrs": len(shared)})
    return res


def update(X: Relation, D: Relation) -> tuple[Relation, Relation]:
    """X'(k) = X(k) add D(k). Returns (X', changed) where `changed` holds the keys whose value
    differs from X, with their new values. For idempotent semirings `changed` is the exact
    semi-naive delta; for non-idempotent ones every touched key with a non-zero contribution
    changes and the caller must handle contribution counting itself."""
    sr = X.sr
    assert sr is D.sr
    add, eq = sr.add, sr.equal
    newdata = dict(X.data)
    changed: dict[tuple, Any] = {}
    adds = 0
    for k, v in D.items():
        if k in newdata:
            nv = add(newdata[k], v)
            adds += 1
            if not eq(nv, newdata[k]):
                newdata[k] = nv
                changed[k] = nv
        else:
            newdata[k] = v
            changed[k] = v
    Xn = Relation(X.schema, sr, None, X.domain)
    Xn.data = newdata
    Dn = Relation(X.schema, sr, changed, X.domain)
    _rec("update", sr, (X, D), Dn, len(D), 0, adds, {"state": len(Xn), "state_uid": Xn.uid})
    return Xn, Dn


def select_k(R: Relation, k: int, key: Callable[[tuple, Any], Any] | None = None,
             better: Callable[[Any, Any], bool] | None = None) -> Relation:
    """Keep the k best tuples. Order: `better(a, b)` on values (default: the semiring's), or
    `key` mapped to a float where smaller is better. Ties are broken by key tuple for
    determinism."""
    sr = R.sr
    n = len(R)
    if k >= n:
        res = R.copy()
        _rec("select", sr, (R,), res, n, 0, 0, {"k": k})
        return res
    if key is not None:
        items = heapq.nsmallest(k, R.items(), key=lambda kv: (key(kv[0], kv[1]), kv[0]))
    else:
        bt = better or sr.better
        assert bt is not None, "semiring has no order; pass key= or better="
        # convert a strict 'better' into a sort: better(a,b) True means a first
        import functools

        def cmp(x, y):
            if bt(x[1], y[1]):
                return -1
            if bt(y[1], x[1]):
                return 1
            return (x[0] > y[0]) - (x[0] < y[0])

        items = heapq.nsmallest(k, R.items(), key=functools.cmp_to_key(cmp))
    res = Relation(R.schema, sr, dict(items), R.domain)
    _rec("select", sr, (R,), res, n, 0, 0, {"k": k})
    return res


def scan(R: Relation, fn: Callable[[tuple, Any], Iterable[tuple[tuple, Any]]],
         schema: Iterable[str] | None = None, fn_name: str = "fn", sr: Semiring | None = None,
         domain=None) -> Relation:
    """Elementwise map: fn(key, value) yields zero or more (key', value') pairs. Duplicate
    output keys are merged with the output semiring's add. fn must be O(1) per tuple; the
    workload documentation lists every fn used."""
    out_sr = sr or R.sr
    pairs = []
    for k, v in R.items():
        pairs.extend(fn(k, v))
    res = Relation.from_pairs(schema if schema is not None else R.schema, out_sr, pairs,
                              domain if domain is not None else R.domain)
    _rec("scan", out_sr, (R,), res, len(R), 0, max(0, len(pairs) - len(res)),
         {"fn": fn_name, "emitted": len(pairs)})
    return res
