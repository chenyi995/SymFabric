"""Ctrl-G style constrained-decoding step: HMM forward over the (hmm state, DFA state) product
space for a token prefix, then the next-token scores that keep the constraint satisfiable.

Instance: an HMM (N hidden states, vocabulary V: transition A, emission B, initial pi), a
partial DFA over tokens (S states; from every non-trap state ~3 explicitly labelled transitions
to other states plus, from every state, a self-loop on every token of a fixed "default" label
set; every other token has no transition and kills the constraint; S/4 non-accepting trap states
have only the self-loops and are therefore dead), accepting states F, a prefix of T tokens sampled from the
HMM under the constraint (every prefix token has a transition into a live state).

    live(s)            = exists path in the DFA from s to an accepting state   (boolean fixpoint)
    fwd_1(h, s)        = pi(h) B(h, tok_0) [delta(s0, tok_0) = s]
    fwd_{t+1}(h', s')  = sum_{h, s} fwd_t(h, s) A(h, h') B(h', tok_t) [delta(s, tok_t) = s']
    score(tok)         = sum_{h, s, h'} fwd_T(h, s) A(h, h') B(h', tok) [delta(s, tok) in live]

Algebra program (PROB and LOGPROB for the real-valued part, BOOLEAN for liveness):
    liveness   Rev(s2, s) = reduce(Delta(s, tok, s2), keep=(s2, s))
               fixpoint: New = mask(rename(combine(D, Rev, keep=(s,)), s->s2), X, negate=True);
                         X, D = update(X, New)                       started from the accepting set
               LiveST(s, tok) = reduce(mask(Delta, Live), keep=(s, tok))
    forward    p = combine(fwd(h, s), A(h, h2), keep=(h2, s))         # dense N x N per live s
               p = combine(p, B_tok(h2))                              # emission column
               p = combine(p, Delta_tok(s, s2), keep=(h2, s2))        # DFA move, unit weights
               rename h2 -> h, s2 -> s;  iterate(T - 1)
    scores     pred = combine(fwd_T, A, keep=(h2, s))
               pl   = combine(pred, scan(LiveST -> unit weights))    # (h2, s, tok): live tokens only
               sc   = combine(pl, B(h2, tok), keep=(tok,))            # sum over h2 and s
Delta_tok / B_tok are the transition and emission relations indexed by token; building that
index once is a layout step (not an operator), as in hmm_viterbi.

Golden: explicit numpy loops. Because the DFA is deterministic, the golden tracks the single DFA
state of the prefix and runs the HMM forward on N states; the algebra reaches the same work by
zero suppression (fwd has one non-empty DFA column). Hotspot event on both sides: one
product-state transition MAC fwd(h, s) * A(h, h') per (t, h, h', live s), for the T-1 prefix
steps and the next-token prediction, plus one MAC per (h', live token) in the scores. The
liveness fixpoint (BOOLEAN) is not a MAC: its cost is reported separately in extra_info.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from algebra import (PROB, LOGPROB, BOOLEAN, Relation, combine, reduce, mask, update, rename, scan,
                     fixpoint, iterate, current_trace)

NAME = "ctrlg_dfa_hmm"
CATEGORY = "probabilistic"
SEMIRINGS = ["prob", "logprob", "boolean"]
STRUCTURE = "dense-iterate"
HOTSPOT_EVENT = ("product-state transition MACs fwd(h,s) x A(h,h') per prefix step and for the "
                 "next-token prediction, plus one MAC per (h', live token) in the token scores")
# claude: decision start — N, V, S follow the task; T is short enough that the PROB forward stays
# far from float64 underflow (margin reported in extra_info); the default label set is V/8 tokens;
# S/4 states are non-accepting traps (default self-loops only) so that liveness is not trivial.
SCALES = {"small": {"N": 16, "V": 64, "S": 12, "T": 32, "n_default": 8, "out_labels": 3,
                    "n_accept": 3, "n_trap": 3},
          "large": {"N": 48, "V": 256, "S": 32, "T": 48, "n_default": 32, "out_labels": 3,
                    "n_accept": 8, "n_trap": 8}}
# claude: decision end


def _live_states(delta: dict, accepting, S: int):
    """Plain BFS on the reversed DFA (instance generation and golden). Returns (live, relaxations)."""
    rev = defaultdict(set)
    for (s, _tok), s2 in delta.items():
        rev[s2].add(s)
    live = set(accepting)
    frontier = sorted(accepting)
    relax = 0
    while frontier:
        nxt = []
        for s2 in frontier:
            for s in sorted(rev[s2]):
                relax += 1
                if s not in live:
                    live.add(s)
                    nxt.append(s)
        frontier = nxt
    return live, relax


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    N, V, S, T = p["N"], p["V"], p["S"], p["T"]
    rng = np.random.default_rng(seed)
    A = rng.dirichlet(np.ones(N), size=N)
    B = rng.dirichlet(np.ones(V), size=N)
    pi = rng.dirichlet(np.ones(N))
    default = sorted(int(t) for t in rng.choice(V, size=p["n_default"], replace=False))
    non_default = [t for t in range(V) if t not in set(default)]
    s0 = 0
    while True:
        traps = set(int(t) for t in rng.choice(range(1, S), size=p["n_trap"], replace=False))
        delta = {}
        for s in range(S):
            if s not in traps:
                for tok in rng.choice(non_default, size=p["out_labels"], replace=False):
                    s2 = int(rng.integers(0, S - 1))
                    delta[(s, int(tok))] = s2 + (1 if s2 >= s else 0)  # never a self-loop
            for tok in default:
                delta[(s, tok)] = s
        non_trap = [t for t in range(S) if t not in traps]
        accepting = sorted(int(a) for a in rng.choice(non_trap, size=p["n_accept"], replace=False))
        live, _ = _live_states(delta, accepting, S)
        if s0 in live:
            break
    # prefix sampled from the HMM, restricted to tokens whose transition leads to a live state
    s = s0
    h = int(rng.choice(N, p=pi))
    tokens = []
    for _ in range(T):
        cands = [tok for tok in range(V) if delta.get((s, tok)) in live]
        w = B[h, cands]
        tok = int(rng.choice(cands, p=w / w.sum()))
        tokens.append(tok)
        s = delta[(s, tok)]
        h = int(rng.choice(N, p=A[h]))
    return {"N": N, "V": V, "S": S, "T": T, "A": A, "B": B, "pi": pi,
            "delta": sorted((s, tok, s2) for (s, tok), s2 in delta.items()),
            "accepting": accepting, "s0": s0, "tokens": tokens, "traps": sorted(traps)}


def _lse(v: np.ndarray) -> float:
    m = np.max(v)
    if m == -math.inf:
        return m
    return float(m + np.log(np.sum(np.exp(v - m))))


def native(inst):
    N, V, S, T = inst["N"], inst["V"], inst["S"], inst["T"]
    A, B, pi, tokens = inst["A"], inst["B"], inst["pi"], inst["tokens"]
    dmap = {(s, tok): s2 for s, tok, s2 in inst["delta"]}
    live, relax = _live_states(dmap, inst["accepting"], S)
    macs = 0
    # ---- PROB ----------------------------------------------------------------------------
    fwd = pi * B[:, tokens[0]]
    s = dmap[(inst["s0"], tokens[0])]
    for t in range(1, T):
        nxt = np.empty(N)
        for j in range(N):
            nxt[j] = np.dot(fwd, A[:, j]) * B[j, tokens[t]]
            macs += N
        fwd = nxt
        s = dmap[(s, tokens[t])]
    fwd_T = np.zeros((N, S))
    fwd_T[:, s] = fwd
    pred = np.empty(N)
    for j in range(N):
        pred[j] = np.dot(fwd, A[:, j])
        macs += N
    scores = np.zeros(V)
    live_tokens = [tok for tok in range(V) if dmap.get((s, tok)) in live]
    for tok in live_tokens:
        scores[tok] = np.dot(pred, B[:, tok])
        macs += N
    # ---- LOGPROB ---------------------------------------------------------------------------
    logA, logB = np.log(A), np.log(B)
    lf = np.log(pi) + logB[:, tokens[0]]
    for t in range(1, T):
        nxt = np.empty(N)
        for j in range(N):
            nxt[j] = _lse(lf + logA[:, j]) + logB[j, tokens[t]]
            macs += N
        lf = nxt
    log_fwd_T = np.full((N, S), -math.inf)
    log_fwd_T[:, s] = lf
    lpred = np.empty(N)
    for j in range(N):
        lpred[j] = _lse(lf + logA[:, j])
        macs += N
    log_scores = np.full(V, -math.inf)
    for tok in live_tokens:
        log_scores[tok] = _lse(lpred + logB[:, tok])
        macs += N
    native.last_info = {"liveness_relaxations": relax, "s_T": s, "n_live_tokens": len(live_tokens),
                        "n_live": len(live)}
    return {"fwd_T": fwd_T, "scores": scores, "log_fwd_T": log_fwd_T, "log_scores": log_scores,
            "live": sorted(live)}, macs


def _forward_and_scores(inst, sr, LiveST, score_mode: str):
    N, V, S, T = inst["N"], inst["V"], inst["S"], inst["T"]
    tokens = inst["tokens"]
    conv = np.log if sr is LOGPROB else (lambda x: x)
    A_rel = Relation.dense(("h", "h2"), sr, conv(inst["A"]), {"h": N, "h2": N})
    B_rel = Relation.dense(("h2", "tok"), sr, conv(inst["B"]), {"h2": N, "tok": V})
    # layout: emission columns and DFA transitions indexed by the prefix tokens (as hmm_viterbi)
    Bcols = {tok: Relation.dense(("h2",), sr, conv(inst["B"][:, tok]), {"h2": N})
             for tok in set(tokens)}
    by_tok = defaultdict(dict)
    for s, tok, s2 in inst["delta"]:
        by_tok[tok][(s, s2)] = sr.one
    Dtok = {tok: Relation(("s", "s2"), sr, by_tok[tok], {"s": S, "s2": S}) for tok in set(tokens)}
    init = Relation(("h2", "s"), sr, {(h, inst["s0"]): float(conv(inst["pi"])[h]) for h in range(N)},
                    {"h2": N, "s": S})
    f = combine(init, Bcols[tokens[0]])                          # emission of token 0
    f = combine(f, Dtok[tokens[0]], keep=("h2", "s2"))           # DFA move on token 0
    f = rename(f, {"h2": "h", "s2": "s"})

    def step(f, t):
        tok = tokens[t + 1]
        p = combine(f, A_rel, keep=("h2", "s"))                  # sum_h fwd(h,s) A(h,h2)
        p = combine(p, Bcols[tok])                               # x B(h2, tok)
        p = combine(p, Dtok[tok], keep=("h2", "s2"))             # DFA move (unit weights)
        return rename(p, {"h2": "h", "s2": "s"})

    fT = iterate(T - 1, step, f)
    pred = combine(fT, A_rel, keep=("h2", "s"))                  # next-token prediction
    if score_mode == "sparse_first":
        # claude: decision start — apply the sparse live-transition selector before the dense
        # emission matrix: MACs only for live tokens, as the golden does.
        LT = scan(LiveST, lambda k, v, one=sr.one: [(k, one)], fn_name=f"bool_to_{sr.name}", sr=sr)
        pl = combine(pred, LT)                                   # (h2, s, tok) for live (s, tok)
        sc = combine(pl, B_rel, keep=("tok",))                   # sum over h2, s
        # claude: decision end
    else:                                                        # "dense_first": measured only
        E = combine(pred, B_rel, keep=("s", "tok"))              # N x V MACs, all tokens
        sc = reduce(mask(E, LiveST), keep=("tok",))
    fwd_arr = np.full((N, S), sr.zero)
    for (h, s), v in fT.items():
        fwd_arr[h, s] = v
    scores = np.full(V, sr.zero)
    for (tok,), v in sc.items():
        scores[tok] = v
    return fwd_arr, scores


def algebra(inst, score_mode: str = "sparse_first"):
    tr = current_trace()
    S, V = inst["S"], inst["V"]
    stats = {}
    n0 = len(tr.records)
    with tr.tag("liveness"):
        Delta = Relation.from_set(("s", "tok", "s2"), inst["delta"], {"s": S, "tok": V, "s2": S})
        Rev = reduce(Delta, keep=("s2", "s"))                    # OR over tok: reversed DFA
        Acc = Relation.from_set(("s2",), [(a,) for a in inst["accepting"]], {"s2": S})

        def lstep(X, D, k):
            New = rename(combine(D, Rev, keep=("s",)), {"s": "s2"})   # predecessors of the delta
            New = mask(New, X, negate=True)
            return update(X, New)

        Live, info = fixpoint(lstep, Acc)
        LiveST = reduce(mask(Delta, Live), keep=("s", "tok"))    # [exists live successor]
    stats["liveness"] = _stats(tr.records[n0:])
    out = {"live": sorted(k[0] for k in Live.keys())}
    for name, sr in (("prob", PROB), ("logprob", LOGPROB)):
        n0 = len(tr.records)
        with tr.tag(name):
            fwd_arr, scores = _forward_and_scores(inst, sr, LiveST, score_mode)
        stats[name] = _stats(tr.records[n0:])
        if sr is PROB:
            out["fwd_T"], out["scores"] = fwd_arr, scores
        else:
            out["log_fwd_T"], out["log_scores"] = fwd_arr, scores
    out["_stats"] = stats
    algebra.last_info = {"liveness_rounds": info["rounds"], "liveness_delta_sizes": info["delta_sizes"]}
    return out


def _stats(records):
    return {"mul_ops": sum(r.mul_ops for r in records), "add_ops": sum(r.add_ops for r in records),
            "candidates": sum(r.candidates for r in records), "records": len(records)}


def _same_support_close(a: np.ndarray, b: np.ndarray, rtol: float) -> bool:
    za, zb = a == 0.0, b == 0.0
    if not np.array_equal(za, zb):
        return False
    return bool(np.allclose(a[~za], b[~zb], rtol=rtol, atol=0.0))


def _same_support_close_log(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    ia, ib = np.isinf(a), np.isinf(b)
    if not np.array_equal(ia, ib):
        return False
    return bool(np.allclose(a[~ia], b[~ib], rtol=tol, atol=tol))


def compare(a, b) -> bool:
    ok = a["live"] == b["live"]
    ok = ok and _same_support_close(a["fwd_T"], b["fwd_T"], 1e-9)
    ok = ok and _same_support_close(a["scores"], b["scores"], 1e-9)
    ok = ok and _same_support_close_log(a["log_fwd_T"], b["log_fwd_T"], 1e-9)
    ok = ok and _same_support_close_log(a["log_scores"], b["log_scores"], 1e-9)
    for r in (a, b):                                 # PROB and LOGPROB agree on each side
        for key in ("fwd_T", "scores"):
            p, lp = r[key], r["log_" + key]
            nz = p > 0.0
            ok = ok and bool(np.all(np.isinf(lp[~nz])))
            ok = ok and bool(np.allclose(np.log(p[nz]), lp[nz], rtol=0.0, atol=1e-8))
    return ok


def extra_info(inst, res):
    st = res.get("_stats", {})
    N, T = inst["N"], inst["T"]
    ni = native.last_info
    per_native = (T - 1) * N * N + N * N + N * ni["n_live_tokens"]      # MACs per semiring
    info = {"native_macs_per_semiring": per_native, "n_delta": len(inst["delta"]),
            "n_live_states": ni["n_live"], "n_live_tokens_from_s_T": ni["n_live_tokens"],
            "s_T": ni["s_T"], "liveness_rounds": algebra.last_info["liveness_rounds"],
            "liveness_native_relaxations": ni["liveness_relaxations"],
            "liveness_algebra_candidates": st["liveness"]["candidates"],
            "liveness_algebra_mul_ops": st["liveness"]["mul_ops"],
            "fwd_T_support": int(np.sum(res["fwd_T"] > 0.0)),
            "fwd_T_min_nonzero": float(np.min(res["fwd_T"][res["fwd_T"] > 0.0]))}
    for name in ("prob", "logprob"):
        s = st[name]
        info[f"ratio_mul_{name}"] = s["mul_ops"] / per_native
        info[f"ratio_sr_{name}"] = (s["mul_ops"] + s["add_ops"]) / per_native
        info[f"ratio_cand_{name}"] = s["candidates"] / per_native
    nz = res["scores"] > 0.0
    info["log_scores_vs_log_max_abs"] = float(np.max(np.abs(np.log(res["scores"][nz]) - res["log_scores"][nz])))
    return info
