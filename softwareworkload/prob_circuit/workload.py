"""Layered probabilistic-circuit (PC) evaluation under three semirings.

The circuit is a random layered DAG in the style of pyjuice / RAT-SPN: leaves, then L internal
layers that alternate product (odd) and sum (even), the last sum layer being a single root.
Sum node n:      S(b, n) = sum_c  W(n, c) * V(b, c)          fan-in f_s, weights normalised
Product node n:  P(b, n) = prod_c V(b, c)   over its f_p children (no weights)
The same DAG is evaluated for a batch of B inputs under
    prob     (+, x)     likelihood of B complete assignments; leaves are Bernoulli densities
    logprob  (LSE, +)   the same in the log domain (checked against log of the prob result)
    counting (+, x)     leaves are 0/1 literal indicators evaluated at B partial assignments
                        (evidence) and every weight is 1: the number of complete sub-circuits
                        consistent with the evidence, i.e. the model count when the circuit is
                        a deterministic decomposable one (the random generator does not
                        guarantee determinism; see docs/decompositions/prob_circuit.md, sec. 1).

Algebra program, per layer (b = batch element, c = child, n = node):
    sum layer      V' = combine(W(n, c), V(b, c), keep=(b, n))                 # join + fused add
    product layer  V' = combine(...combine(P_0, P_1)..., P_{f_p-1})            # "chain" mode
                       with P_k = combine(E_k(n, c), V(b, c), keep=(b, n))     # slot-k gather
                   or  V' = reduce(combine(E(n, c), V(b, c)), keep=(b, n), use_mul=True)  # "fold"
    followed by rename(n -> c).
The "chain" mode is the default: it is sound under zero suppression (a missing child removes
the product tuple), while the "fold" mode multiplies only the children that are present and is
therefore wrong whenever a child value is the semiring zero (the counting run has such
children). Both are measured in the decomposition document.

Golden: layer-wise numpy evaluation (float64 for prob, log-sum-exp for logprob, Python ints
for counting). Both sides count one "edge evaluation per batch element": a child value
multiplied by a weight and accumulated into a sum node, or a child value folded into a
product node. The golden counts every edge densely (as a tensorised PC library does);
the algebra skips children whose value is the semiring zero.
"""
from __future__ import annotations

import math

import numpy as np

from algebra import PROB, LOGPROB, COUNTING, Relation, combine, reduce, rename, current_trace

NAME = "prob_circuit"
CATEGORY = "probabilistic"
SEMIRINGS = ["prob", "logprob", "counting"]
STRUCTURE = "dag"
HOTSPOT_EVENT = ("edge evaluations per batch element: child value x weight accumulated into a sum "
                 "node, or child value folded into a product node")
# claude: decision start — product fan-in 2 and sum fan-in 8 follow the RAT-SPN / HCLT shape
# (binary partitions, dense sum blocks); L is the number of internal layers and must be even so
# that the last layer is the sum root.
SCALES = {"small": {"L": 6, "W": 64, "B": 16, "f_s": 8, "f_p": 2},
          "large": {"L": 8, "W": 256, "B": 32, "f_s": 8, "f_p": 2}}
# claude: decision end

FLOAT32_TINY = float(np.finfo(np.float32).tiny)   # smallest normal float32, for the underflow note


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    L, W, B, f_s, f_p = p["L"], p["W"], p["B"], p["f_s"], p["f_p"]
    assert L % 2 == 0 and W % 2 == 0 and f_p <= W and f_s <= W
    rng = np.random.default_rng(seed)
    n_vars = W // 2                      # leaf l is the literal [X_{l mod n_vars} = l // n_vars]
    layers = []
    prev_width = W
    for depth in range(1, L + 1):
        if depth % 2 == 1:               # product layer
            width = W
            children = np.stack([rng.choice(prev_width, size=f_p, replace=False)
                                 for _ in range(width)])
            layers.append({"type": "prod", "width": width, "children": children})
        elif depth < L:                  # sum layer
            width = W
            children = np.stack([rng.choice(prev_width, size=f_s, replace=False)
                                 for _ in range(width)])
            weights = rng.dirichlet(np.ones(f_s), size=width)
            layers.append({"type": "sum", "width": width, "children": children,
                           "weights": weights})
        else:                            # root: one sum node over the whole previous layer
            width = 1
            children = np.arange(prev_width)[None, :]
            weights = rng.dirichlet(np.ones(prev_width), size=1)
            layers.append({"type": "sum", "width": width, "children": children,
                           "weights": weights})
        prev_width = width
    # prob / logprob leaves: Bernoulli(theta_l) over variable v(l), evaluated at complete inputs
    theta = rng.uniform(0.05, 0.95, size=W)
    x = rng.integers(0, 2, size=(B, n_vars))
    var_of = np.arange(W) % n_vars
    pol_of = np.arange(W) // n_vars
    leaf_prob = np.where(x[:, var_of] == 1, theta[None, :], 1.0 - theta[None, :])
    # counting leaves: literal indicators at partial assignments with n_vars // 4 observed vars
    k_ev = max(1, n_vars // 4)
    leaf_count = np.ones((B, W), dtype=np.int64)
    evidence = []
    for b in range(B):
        obs_vars = rng.choice(n_vars, size=k_ev, replace=False)
        obs_vals = rng.integers(0, 2, size=k_ev)
        evidence.append((obs_vars, obs_vals))
        for v, val in zip(obs_vars, obs_vals):
            leaf_count[b, var_of == v] = (pol_of[var_of == v] == val).astype(np.int64)
    return {"L": L, "W": W, "B": B, "f_s": f_s, "f_p": f_p, "n_vars": n_vars, "layers": layers,
            "leaf_prob": leaf_prob, "leaf_count": leaf_count, "evidence": evidence}


def _edges(inst) -> int:
    return sum(int(layer["children"].size) for layer in inst["layers"])


def _lse_last(a: np.ndarray) -> np.ndarray:
    m = np.max(a, axis=-1, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=-1, keepdims=True)))[..., 0]


def _native_eval(layers, leaves, mode: str):
    """Layer-wise dense evaluation. mode: prob | logprob | counting."""
    V = leaves
    for layer in layers:
        G = V[:, layer["children"]]                    # (B, width, fan-in)
        if layer["type"] == "sum":
            if mode == "prob":
                V = np.sum(G * layer["weights"][None, :, :], axis=-1)
            elif mode == "logprob":
                V = _lse_last(G + np.log(layer["weights"])[None, :, :])
            else:                                      # counting: unit weights, Python ints
                V = np.sum(G, axis=-1)
        else:
            V = np.sum(G, axis=-1) if mode == "logprob" else np.prod(G, axis=-1)
    return V[:, 0]


def native(inst):
    layers = inst["layers"]
    B = inst["B"]
    edges = _edges(inst)
    res = {"prob": _native_eval(layers, inst["leaf_prob"].astype(np.float64), "prob"),
           "logprob": _native_eval(layers, np.log(inst["leaf_prob"]), "logprob"),
           "counting": [int(v) for v in
                        _native_eval(layers, inst["leaf_count"].astype(object), "counting")]}
    ops = 3 * B * edges                                # one edge evaluation per batch element
    return res, ops


def _eval_circuit(inst, sr, leaves, weight_fn, product_mode: str):
    B, W = inst["B"], inst["W"]
    V = Relation.dense(("b", "c"), sr, leaves, {"b": B, "c": W})
    prev_width = W
    for layer in inst["layers"]:
        width, ch = layer["width"], layer["children"]
        dom = {"n": width, "c": prev_width}
        if layer["type"] == "sum":
            wts = weight_fn(layer["weights"])
            Wt = Relation(("n", "c"), sr, {(n, int(ch[n, k])): wts[n, k]
                                           for n in range(width) for k in range(ch.shape[1])}, dom)
            S = combine(Wt, V, keep=("b", "n"))                  # sum_c W(n,c) (x) V(b,c)
        elif product_mode == "chain":
            # claude: decision start — product node as a chain of joins on (b, n): one unit-weight
            # gather per child slot, then f_p-1 joins that multiply; sound when children are missing.
            P = None
            for k in range(ch.shape[1]):
                Ek = Relation(("n", "c"), sr, {(n, int(ch[n, k])): sr.one for n in range(width)}, dom)
                Pk = combine(Ek, V, keep=("b", "n"))             # gather slot k: V(b, child_k(n))
                P = Pk if P is None else combine(P, Pk)          # (b, n) x (b, n): product
            S = P
            # claude: decision end
        else:                                                    # "fold": gather all, mul-fold
            E = Relation(("n", "c"), sr, {(n, int(c)): sr.one for n in range(width) for c in ch[n]},
                         dom)
            S = reduce(combine(E, V), keep=("b", "n"), use_mul=True)
        V = rename(S, {"n": "c"})
        prev_width = width
    return [V.get((b, 0)) for b in range(B)]


def _stats(records):
    return {"mul_ops": sum(r.mul_ops for r in records), "add_ops": sum(r.add_ops for r in records),
            "candidates": sum(r.candidates for r in records), "records": len(records)}


def algebra(inst, product_mode: str = "chain"):
    tr = current_trace()
    runs = [("prob", PROB, inst["leaf_prob"].astype(np.float64), lambda w: w),
            ("logprob", LOGPROB, np.log(inst["leaf_prob"]), np.log),
            ("counting", COUNTING, inst["leaf_count"], lambda w: np.ones(w.shape, dtype=np.int64))]
    out, stats = {}, {}
    for name, sr, leaves, weight_fn in runs:
        n0 = len(tr.records)
        with tr.tag(name):
            vals = _eval_circuit(inst, sr, leaves, weight_fn, product_mode)
        stats[name] = _stats(tr.records[n0:])
        out[name] = [int(v) for v in vals] if sr is COUNTING else np.array(vals, dtype=np.float64)
    out["_stats"] = stats
    return out


def compare(a, b) -> bool:
    ok = bool(np.allclose(a["prob"], b["prob"], rtol=1e-9, atol=0.0))
    ok = ok and bool(np.allclose(a["logprob"], b["logprob"], rtol=1e-9, atol=1e-9))
    ok = ok and list(a["counting"]) == list(b["counting"])
    # cross-check of the two real-valued semirings on each side: log(prob) == logprob
    for r in (a, b):
        ok = ok and bool(np.all(r["prob"] > 0.0))
        ok = ok and bool(np.allclose(np.log(r["prob"]), r["logprob"], rtol=0.0, atol=1e-8))
    return ok


def extra_info(inst, res):
    B, edges = inst["B"], _edges(inst)
    per_native = B * edges                            # native edge evaluations per semiring
    st = res.get("_stats", {})
    info = {"edges": edges, "B": B, "native_ops_per_semiring": per_native}
    for name, s in st.items():
        info[f"ratio_mul_{name}"] = s["mul_ops"] / per_native
        info[f"ratio_sr_{name}"] = (s["mul_ops"] + s["add_ops"]) / per_native
        info[f"ratio_cand_{name}"] = s["candidates"] / per_native
    # zero suppression seen by the counting run: leaves and root values that are zero
    info["counting_zero_leaves_frac"] = float(np.mean(inst["leaf_count"] == 0))
    info["counting_zero_roots"] = int(sum(1 for v in res["counting"] if v == 0))
    info["counting_root_max"] = int(max(res["counting"]))
    info["counting_root_max_bits"] = int(max(res["counting"])).bit_length()
    # underflow note: float64 minimum root likelihood, and what float32 (pyjuice default) would do
    p = np.asarray(res["prob"], dtype=np.float64)
    info["prob_root_min"] = float(np.min(p))
    info["prob_root_below_float32_tiny"] = int(np.sum(p < FLOAT32_TINY))
    p32 = _native_eval(inst["layers"], inst["leaf_prob"].astype(np.float32), "prob")
    info["prob_roots_underflow_in_float32"] = int(np.sum(p32 == 0.0))
    info["logprob_vs_log_prob_max_abs"] = float(np.max(np.abs(np.log(p) - np.asarray(res["logprob"]))))
    return info
