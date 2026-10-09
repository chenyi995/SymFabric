"""HMM forward (filtering) and Viterbi decoding as a fixed-count iteration.

Forward, in the log semiring (LSE, +) so long sequences do not underflow:
    a_t(j) = log b_j(o_t) + LSE_i ( a_{t-1}(i) + log A_ij )
Algebra step:  a_t = combine( combine(a_{t-1}, logA, keep=(j,)), logB[:, o_t] )

Viterbi with back-pointers uses a value type (score, bp) under the semiring
    add = max by score (ties: smaller bp), mul = (s1 + s2, bp of the right operand)
where the transition relation stores (log A_ij, i) so that the reduce over i keeps the argmax
predecessor. The path is read back with T filter operations of O(N) each (recorded).

Golden: numpy implementations with explicit logsumexp / argmax. Both sides count
"transition multiply-accumulates": T * N * N.
"""
from __future__ import annotations

import math
import random

import numpy as np

from algebra import LOGPROB, Semiring, Relation, combine, iterate, filter as afilter

NAME = "hmm_viterbi"
CATEGORY = "probabilistic"
SEMIRINGS = ["logprob", "viterbi_bp"]
STRUCTURE = "dense-iterate"
HOTSPOT_EVENT = "transition multiply-accumulates (i -> j) per time step"
SCALES = {"small": {"N": 16, "M": 8, "T": 40, "Q": 1}, "large": {"N": 48, "M": 16, "T": 120, "Q": 1},
          "batched": {"N": 48, "M": 16, "T": 30, "Q": 64}}   # Q sequences share A (multi-query)

# Viterbi-with-backpointer semiring: value = (log score, predecessor state)
NEG = -math.inf


def _vadd(a, b):
    if a[0] > b[0]:
        return a
    if b[0] > a[0]:
        return b
    return a if a[1] <= b[1] else b


def _vmul(a, b):
    # the back-pointer travels with whichever operand carries one (-1 = none, e.g. emissions)
    return (a[0] + b[0], b[1] if b[1] >= 0 else a[1])


VITERBI_BP = Semiring("viterbi_bp", (NEG, -1), (0.0, -1), _vadd, _vmul, idempotent=True,
                      selective=True, numeric=False, better=lambda a, b: a[0] > b[0], bits=64)


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = np.random.default_rng(seed)
    N, M, T = p["N"], p["M"], p["T"]
    A = rng.dirichlet(np.ones(N) * 0.5, size=N)          # row-stochastic, moderately peaked
    B = rng.dirichlet(np.ones(M) * 0.5, size=N)
    pi = rng.dirichlet(np.ones(N))
    Q = p.get("Q", 1)
    obs = rng.integers(0, M, size=(Q, T))
    return {"N": N, "M": M, "T": T, "Q": Q, "logA": np.log(A), "logB": np.log(B),
            "logpi": np.log(pi), "obs": obs}


def _lse(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))


def _native_one(inst, obs):
    N, T = inst["N"], inst["T"]
    logA, logB, logpi = inst["logA"], inst["logB"], inst["logpi"]
    ops = 0
    a = logpi + logB[:, obs[0]]
    for t in range(1, T):
        nxt = np.empty(N)
        for j in range(N):
            nxt[j] = _lse(a + logA[:, j]) + logB[j, obs[t]]
            ops += N
        a = nxt
    loglik = _lse(a)
    v = logpi + logB[:, obs[0]]
    bp = np.zeros((T, N), dtype=int)
    for t in range(1, T):
        nxt = np.empty(N)
        for j in range(N):
            cand = v + logA[:, j]
            k = int(np.argmax(cand))            # first max = smallest index on ties
            nxt[j] = cand[k] + logB[j, obs[t]]
            bp[t, j] = k
            ops += N
        v = nxt
    path = [int(np.argmax(v))]
    for t in range(T - 1, 0, -1):
        path.append(int(bp[t, path[-1]]))
    path.reverse()
    return {"loglik": float(loglik), "alpha_T": a.copy(), "path": path, "best": float(np.max(v))}, ops


def native(inst):
    res, ops = [], 0
    for q in range(inst["Q"]):
        r, o = _native_one(inst, inst["obs"][q])
        res.append(r)
        ops += o
    return res, ops


def algebra(inst):
    """Q sequences run together: the state relation is (q, i) and every step is one semiring
    GEMM (Q x N) x (N x N) followed by an elementwise join with the per-sequence emission column."""
    N, T, Q = inst["N"], inst["T"], inst["Q"]
    logA, logB, logpi, obs = inst["logA"], inst["logB"], inst["logpi"], inst["obs"]
    dom = {"q": Q, "i": N, "j": N}
    # --- forward in (LSE, +) -------------------------------------------------------------
    A_rel = Relation.dense(("i", "j"), LOGPROB, logA, dom)
    Bcols = [Relation(("q", "j"), LOGPROB, {(q, j): float(logB[j, obs[q, t]]) for q in range(Q)
                                             for j in range(N)}, dom) for t in range(T)]
    a0 = Relation(("q", "i"), LOGPROB, {(q, i): float(logpi[i] + logB[i, obs[q, 0]])
                                        for q in range(Q) for i in range(N)}, dom)

    def fwd_step(a, t):
        s = combine(a, A_rel, keep=("q", "j"))          # LSE_i a(q,i) + logA(i,j)
        s = combine(s, Bcols[t + 1])                     # + log b_j(o_{q,t})
        out = Relation(("q", "i"), LOGPROB, None, dom)
        out.data = s.data                                # rename j -> i without copying
        return out

    aT = iterate(T - 1, fwd_step, a0)
    # --- Viterbi with back-pointers --------------------------------------------------------
    Abp = Relation(("i", "j"), VITERBI_BP,
                   {(i, j): (float(logA[i, j]), i) for i in range(N) for j in range(N)}, dom)
    Bbp = [Relation(("q", "j"), VITERBI_BP, {(q, j): (float(logB[j, obs[q, t]]), -1) for q in range(Q)
                                              for j in range(N)}, dom) for t in range(T)]
    v0 = Relation(("q", "i"), VITERBI_BP, {(q, i): (float(logpi[i] + logB[i, obs[q, 0]]), -1)
                                           for q in range(Q) for i in range(N)}, dom)
    layers = [v0]

    def vit_step(v, t):
        s = combine(v, Abp, keep=("q", "j"))
        s = combine(s, Bbp[t + 1])
        out = Relation(("q", "i"), VITERBI_BP, None, dom)
        out.data = s.data
        layers.append(out)
        return out

    vT = iterate(T - 1, vit_step, v0)
    results = []
    for q in range(Q):
        alpha_T = np.array([aT.get((q, i)) for i in range(N)])
        loglik = LOGPROB.sum(alpha_T.tolist())
        best_i, best_val = max(((i, vT.get((q, i))) for i in range(N)),
                               key=lambda kv: (kv[1][0], -kv[0]))
        path = [best_i]
        for t in range(T - 1, 0, -1):
            cur = path[-1]
            sel = afilter(layers[t], lambda k, val, c=cur, qq=q: k == (qq, c), name="eq_state")
            path.append(sel.get((q, cur))[1])
        path.reverse()
        results.append({"loglik": float(loglik), "alpha_T": alpha_T, "path": path, "best": best_val[0]})
    return results


def _compare_one(a, b) -> bool:
    return (abs(a["loglik"] - b["loglik"]) <= 1e-7 * max(1.0, abs(a["loglik"]))
            and np.allclose(a["alpha_T"], b["alpha_T"], rtol=1e-7, atol=1e-9)
            and a["path"] == b["path"]
            and abs(a["best"] - b["best"]) <= 1e-7 * max(1.0, abs(a["best"])))


def compare(a, b) -> bool:
    return len(a) == len(b) and all(_compare_one(x, y) for x, y in zip(a, b))


def extra_info(inst, res):
    return {"Q": inst["Q"], "T": inst["T"], "N": inst["N"]}
