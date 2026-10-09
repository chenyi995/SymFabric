"""Variable elimination (VE) for one marginal query in a discrete Bayesian network.

Instance: a random DAG over n variables with domain sizes 2 or 3, in-degree <= 3, every parent
within a window of `window` earlier variables (a band structure, so the induced width stays
bounded), random Dirichlet CPTs; a query variable q and hard evidence on a few other variables.
Result: P(X_q | evidence).

Elimination order: min-fill over the interaction graph of the non-evidence variables. This is
plain Python on both sides (CONTROL RESIDUE); its pair checks are counted in extra_info().

Algebra program, PROB and LOGPROB:
    F_i(x_pa(i), x_i) = CPT_i                          Relation.dense per variable
    F_i <- filter(F_i, x_e == e)   for every evidence variable e in F_i's schema
    for v in order:   involved = factors whose schema contains x_v
        P   = combine(F_1, F_2); P = combine(P, F_3); ...        pairwise natural joins
        F'  = combine(P, F_last, keep = all attributes except x_v)  fused sum over x_v
        (a single involved factor: F' = reduce(F_1, keep))
    M   = product of the remaining factors, reduce to (x_q,)
    Z   = reduce(M, keep=());   answer = scan(M, v -> v / Z)   (v - log Z in LOGPROB)
Evidence attributes stay in the schemas with a single value and are dropped in the last reduce.

Golden: the same VE over plain Python dicts (dense factor tables) with the same order; a second
golden, `brute_force()`, enumerates the full joint with numpy and is compared with the dict VE
at the tiny scale in the tests (n = 12; the small-scale joint is up to 3^25 entries).
Hotspot event on both sides: one multiplication per output entry of a pairwise factor product.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from algebra import PROB, LOGPROB, Relation, combine, reduce, filter as afilter, scan, current_trace

NAME = "bayes_net_ve"
CATEGORY = "probabilistic"
SEMIRINGS = ["prob", "logprob"]
STRUCTURE = "dag"
HOTSPOT_EVENT = "factor-product entries: one multiplication per output entry of a pairwise factor product"
# claude: decision start — parents are drawn from a window of earlier variables (a band structure
# bounds the moral graph's bandwidth); p2/p3 are the probabilities of a second and a third parent
# (in-degree <= 3). An instance whose min-fill elimination would create a factor above
# MAX_FACTOR_ENTRIES is regenerated (rejection sampling on the induced width, reported in
# extra_info as "regenerations"). "tiny" exists only for the brute-force cross-check.
SCALES = {"tiny": {"n": 12, "window": 4, "p2": 0.5, "p3": 0.25, "n_evidence": 2},
          "small": {"n": 25, "window": 10, "p2": 0.9, "p3": 0.8, "n_evidence": 3},
          "large": {"n": 60, "window": 12, "p2": 0.9, "p3": 0.8, "n_evidence": 5}}
MAX_FACTOR_ENTRIES = 2 ** 14
# claude: decision end


def _generate(p, rng):
    n, window, n_ev = p["n"], p["window"], p["n_evidence"]
    card = [int(c) for c in rng.integers(2, 4, size=n)]           # 2 or 3 values each
    parents = [[] for _ in range(n)]
    for i in range(1, n):
        cands = list(range(max(0, i - window), i))
        k = min(len(cands), 1 + int(rng.random() < p["p2"]) + int(rng.random() < p["p3"]))
        parents[i] = sorted(int(x) for x in rng.choice(cands, size=k, replace=False))
    cpts = []
    for i in range(n):
        shape = [card[pa] for pa in parents[i]] + [card[i]]
        rows = int(np.prod(shape[:-1]))
        cpts.append(rng.dirichlet(np.ones(card[i]), size=rows).reshape(shape))
    q = int(rng.integers(0, n))
    others = [v for v in range(n) if v != q]
    ev_vars = sorted(int(v) for v in rng.choice(others, size=n_ev, replace=False))
    evidence = {v: int(rng.integers(0, card[v])) for v in ev_vars}
    return {"n": n, "card": card, "parents": parents, "cpts": cpts, "query": q,
            "evidence": evidence, "window": window}


def _max_factor_entries(inst, order) -> int:
    """Largest factor (dense entries) that VE with `order` creates; same bookkeeping as VE itself."""
    ev = set(inst["evidence"])
    scopes = [frozenset(inst["parents"][i] + [i]) - ev for i in range(inst["n"])]
    biggest = 0
    for v in order:
        involved = [s for s in scopes if v in s]
        scopes = [s for s in scopes if v not in s]
        union = frozenset().union(*involved)
        size = 1
        for u in union:
            size *= inst["card"][u]
        biggest = max(biggest, size)
        scopes.append(union - {v})
    return biggest


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        inst = _generate(p, rng)
        order, _ = _elimination_setup(inst)
        if _max_factor_entries(inst, order) <= MAX_FACTOR_ENTRIES:
            inst["regenerations"] = attempt
            return inst
    raise RuntimeError("could not generate an instance within the factor-size bound")


# ---- control residue: min-fill elimination order (plain Python, shared by both sides) ---------
def min_fill_order(scopes, elim_vars):
    """Greedy min-fill over the interaction graph induced by `scopes` (iterables of variable ids).
    Returns (order, pair_checks); ties are broken by the smaller variable id."""
    adj: dict[int, set] = {}
    for sc in scopes:
        sc = set(sc)
        for a in sc:
            adj.setdefault(a, set()).update(sc - {a})
    remaining = set(elim_vars)
    order, checks = [], 0
    while remaining:
        best, best_fill = None, None
        for v in sorted(remaining):
            nb = sorted(adj[v])
            fill = 0
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    checks += 1
                    if nb[j] not in adj[nb[i]]:
                        fill += 1
            if best_fill is None or fill < best_fill:
                best, best_fill = v, fill
        nb = sorted(adj[best])
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                adj[nb[i]].add(nb[j])
                adj[nb[j]].add(nb[i])
        for u in nb:
            adj[u].discard(best)
        del adj[best]
        remaining.remove(best)
        order.append(best)
    return order, checks


def _elimination_setup(inst):
    ev = inst["evidence"]
    scopes = [set(inst["parents"][i] + [i]) - set(ev) for i in range(inst["n"])]
    elim = [v for v in range(inst["n"]) if v != inst["query"] and v not in ev]
    return min_fill_order(scopes, elim)


# ---- golden: dict-based VE ---------------------------------------------------------------------
def _lse(values):
    m = max(values)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


class _DictVE:
    def __init__(self, inst, log: bool):
        self.inst, self.log = inst, log
        self.card = inst["card"]
        self.ops = 0           # hotspot: factor-product entries
        self.adds = 0
        self.max_entries = 0

    def factors(self):
        out = []
        for i in range(self.inst["n"]):
            vars_ = tuple(self.inst["parents"][i]) + (i,)
            t = self.inst["cpts"][i]
            table = {}
            for idx in itertools.product(*[range(self.card[v]) for v in vars_]):
                val = float(t[idx])
                table[idx] = math.log(val) if self.log else val
            out.append((vars_, table))
        return out

    def restrict(self, f, e, val):
        vars_, table = f
        pos = vars_.index(e)
        nv = vars_[:pos] + vars_[pos + 1:]
        return nv, {k[:pos] + k[pos + 1:]: v for k, v in table.items() if k[pos] == val}

    def product(self, f, g):
        fv, ft = f
        gv, gt = g
        uv = fv + tuple(v for v in gv if v not in fv)
        fpos = [uv.index(v) for v in fv]
        gpos = [uv.index(v) for v in gv]
        table = {}
        for idx in itertools.product(*[range(self.card[v]) for v in uv]):
            a = ft[tuple(idx[p] for p in fpos)]
            b = gt[tuple(idx[p] for p in gpos)]
            table[idx] = a + b if self.log else a * b
            self.ops += 1
        self.max_entries = max(self.max_entries, len(table))
        return uv, table

    def sum_out(self, f, v):
        fv, ft = f
        pos = fv.index(v)
        nv = fv[:pos] + fv[pos + 1:]
        groups: dict[tuple, list] = {}
        for k, val in ft.items():
            groups.setdefault(k[:pos] + k[pos + 1:], []).append(val)
        table = {}
        for k, vals in groups.items():
            table[k] = _lse(vals) if self.log else sum(vals)
            self.adds += len(vals) - 1
        return nv, table

    def run(self, order):
        inst = self.inst
        facs = self.factors()
        for e, val in inst["evidence"].items():
            facs = [self.restrict(f, e, val) if e in f[0] else f for f in facs]
        for v in order:
            involved = [f for f in facs if v in f[0]]
            others = [f for f in facs if v not in f[0]]
            prod = involved[0]
            for f in involved[1:]:
                prod = self.product(prod, f)
            facs = others + [self.sum_out(prod, v)]
        prod = facs[0]
        for f in facs[1:]:
            prod = self.product(prod, f)
        q = inst["query"]
        assert prod[0] == (q,), prod[0]
        vals = [prod[1][(i,)] for i in range(self.card[q])]
        if self.log:
            z = _lse(vals)
            return np.array([v - z for v in vals])
        z = sum(vals)
        return np.array([v / z for v in vals])


def native(inst):
    order, checks = _elimination_setup(inst)
    ve_p = _DictVE(inst, log=False)
    marg = ve_p.run(order)
    ve_l = _DictVE(inst, log=True)
    logmarg = ve_l.run(order)
    native.last_info = {"minfill_pair_checks": checks, "order": order, "adds": ve_p.adds,
                        "max_factor_entries": ve_p.max_entries}
    return {"marginal": marg, "log_marginal": logmarg}, ve_p.ops + ve_l.ops


def brute_force(inst):
    """Second golden: full joint by numpy broadcasting (feasible only for small n)."""
    n, card = inst["n"], inst["card"]
    joint = np.ones(tuple(card), dtype=np.float64)
    for i in range(n):
        axes = inst["parents"][i] + [i]                  # ascending: parents precede children
        shape = [1] * n
        for ax, c in zip(axes, inst["cpts"][i].shape):
            shape[ax] = c
        joint = joint * inst["cpts"][i].reshape(shape)
    index = [slice(None)] * n
    for e, val in inst["evidence"].items():
        index[e] = val
    cond = joint[tuple(index)]                          # evidence axes removed
    kept = [v for v in range(n) if v not in inst["evidence"]]
    qpos = kept.index(inst["query"])
    marg = cond.sum(axis=tuple(a for a in range(len(kept)) if a != qpos))
    return marg / marg.sum()


# ---- algebra -------------------------------------------------------------------------------------
def _attr(v: int) -> str:
    return f"x{v}"


def _ve_algebra(inst, sr, order):
    n, card, q = inst["n"], inst["card"], inst["query"]
    conv = np.log if sr is LOGPROB else (lambda t: t)
    factors = []
    for i in range(n):
        schema = tuple(_attr(v) for v in inst["parents"][i] + [i])
        dom = {_attr(v): card[v] for v in inst["parents"][i] + [i]}
        factors.append(Relation.dense(schema, sr, conv(inst["cpts"][i]), dom))
    for e, val in inst["evidence"].items():
        a = _attr(e)
        for idx, F in enumerate(factors):
            if a in F.schema:
                pos = F.pos(a)
                factors[idx] = afilter(F, lambda k, v, p=pos, x=val: k[p] == x, name="evidence_eq")
    for v in order:
        a = _attr(v)
        involved = [F for F in factors if a in F.schema]
        others = [F for F in factors if a not in F.schema]
        keep = list(involved[0].schema)
        for F in involved[1:]:
            keep += [x for x in F.schema if x not in keep]
        keep.remove(a)
        if len(involved) == 1:
            new = reduce(involved[0], keep)
        else:
            P = involved[0]
            for F in involved[1:-1]:
                P = combine(P, F)
            new = combine(P, involved[-1], keep=keep)
        factors = others + [new]
    P = factors[0]
    for F in factors[1:]:
        P = combine(P, F)
    M = P if P.schema == (_attr(q),) else reduce(P, keep=(_attr(q),))
    Z = reduce(M, keep=()).get(())
    if sr is LOGPROB:
        norm = scan(M, lambda k, v, z=Z: [(k, v - z)], fn_name="sub_logZ")
    else:
        norm = scan(M, lambda k, v, z=Z: [(k, v / z)], fn_name="div_Z")
    return np.array([norm.get((i,)) for i in range(card[q])])


def algebra(inst):
    order, checks = _elimination_setup(inst)          # CONTROL RESIDUE (plain Python)
    tr = current_trace()
    out, stats = {}, {}
    for name, sr in (("prob", PROB), ("logprob", LOGPROB)):
        n0 = len(tr.records)
        with tr.tag(name):
            vals = _ve_algebra(inst, sr, order)
        recs = tr.records[n0:]
        # largest factor product = the candidates of the largest combine (same definition as the
        # golden's largest product table)
        stats[name] = {"mul_ops": sum(r.mul_ops for r in recs),
                       "add_ops": sum(r.add_ops for r in recs),
                       "candidates": sum(r.candidates for r in recs),
                       "max_factor_entries": max(r.candidates for r in recs if r.op == "combine")}
        out["marginal" if sr is PROB else "log_marginal"] = vals
    out["_stats"] = stats
    algebra.last_info = {"minfill_pair_checks": checks, "order": order}
    return out


def compare(a, b) -> bool:
    ok = bool(np.allclose(a["marginal"], b["marginal"], rtol=1e-9, atol=0.0))
    ok = ok and bool(np.allclose(a["log_marginal"], b["log_marginal"], rtol=1e-9, atol=1e-9))
    for r in (a, b):        # the two semirings agree with each other on each side
        ok = ok and bool(np.allclose(np.log(r["marginal"]), r["log_marginal"], rtol=0.0, atol=1e-8))
    return ok


def extra_info(inst, res):
    st = res.get("_stats", {})
    info = {"n": inst["n"], "query": inst["query"], "n_evidence": len(inst["evidence"]),
            "residue_minfill_pair_checks": algebra.last_info["minfill_pair_checks"],
            "max_factor_entries": max(s["max_factor_entries"] for s in st.values()),
            "max_factor_bound": MAX_FACTOR_ENTRIES, "regenerations": inst["regenerations"],
            "elimination_steps": len(algebra.last_info["order"])}
    for name, s in st.items():
        info[f"mul_ops_{name}"] = s["mul_ops"]
        info[f"add_ops_{name}"] = s["add_ops"]
        info[f"candidates_{name}"] = s["candidates"]
    return info
