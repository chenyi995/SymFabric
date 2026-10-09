"""Beam search decoding over a sparse Markov language model, scored in the Viterbi (max, +) semiring.

The model is a token set of size V with a transition relation T(s, t) = log P(t | s) that holds 32
successors per token (a top-k sparsified LM). Decoding starts from token 0 and runs L steps with
beam width K. A beam entry is a hypothesis: (beam id b, last token s) with its accumulated score.
Hypotheses are never merged (two beams ending in the same token are different prefixes).

Algebra program, one step (iterate L times):
    cand = combine(beam(b, s), T(s, t), keep=(t, b))   # score(b) + log P(t|s); one tuple per (t, b)
    surv = select_k(cand, K)                            # K best; ties by key (t asc, b asc)
    beam'= scan(surv, reindex_by_rank)                  # (t, b) -> (rank, t): dense new beam ids
The rank of a survivor (score desc, token asc, parent beam asc) and the per-step parent map used
for back-tracking are computed in plain Python: CONTROL RESIDUE, counted in extra_info().

Golden: plain Python beam search with heapq (nsmallest on the key (-score, token, parent beam)).
Both sides count candidate expansions: one per (beam entry, successor) pair per step. Tie-breaking
is identical on both sides: score descending, then token ascending, then parent beam ascending.
"""
from __future__ import annotations

import heapq
import math
import random

from algebra import VITERBI, Relation, combine, select_k, scan, iterate, current_trace

NAME = "beam_search"
CATEGORY = "search"
SEMIRINGS = ["viterbi"]
STRUCTURE = "select-loop"
HOTSPOT_EVENT = "candidate expansions: (beam entry, successor token) pairs scored per step"
SCALES = {"small": {"V": 200, "K": 8, "L": 20, "succ": 32},
          "large": {"V": 2000, "K": 32, "L": 20, "succ": 32}}


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    V, n_succ = p["V"], p["succ"]
    # claude: decision start — the sparse LM is generated directly: each token gets 32 distinct
    # successors with Gaussian logits, log-softmax normalised over those 32 (what a top-32
    # sparsified dense LM stores after renormalisation); no dense V x V matrix is materialised.
    succ = []
    for _ in range(V):
        toks = sorted(rng.sample(range(V), n_succ))
        logits = [rng.gauss(0.0, 1.0) for _ in toks]
        m = max(logits)
        lse = m + math.log(sum(math.exp(x - m) for x in logits))
        succ.append([(t, x - lse) for t, x in zip(toks, logits)])
    # claude: decision end
    return {"V": V, "K": p["K"], "L": p["L"], "start": 0, "succ": succ}


def native(inst):
    K, L, succ = inst["K"], inst["L"], inst["succ"]
    beams = [(0.0, [inst["start"]])]          # rank-ordered (score, token sequence)
    expansions = 0
    for _ in range(L):
        cands = []
        for b, (score, seq) in enumerate(beams):
            for tok, lp in succ[seq[-1]]:
                expansions += 1
                cands.append((score + lp, tok, b))
        top = heapq.nsmallest(K, cands, key=lambda c: (-c[0], c[1], c[2]))
        beams = [(sc, beams[b][1] + [tok]) for sc, tok, b in top]
    return [(sc, seq) for sc, seq in beams], expansions


def algebra(inst):
    V, K, L, succ = inst["V"], inst["K"], inst["L"], inst["succ"]
    T = Relation(("s", "t"), VITERBI,
                 {(s, t): lp for s, lst in enumerate(succ) for t, lp in lst}, {"s": V, "t": V})
    beam0 = Relation(("b", "s"), VITERBI, {(0, inst["start"]): 0.0}, {"b": K, "s": V})
    # CONTROL RESIDUE: per-step parent maps parents[t][new id] = (parent beam id, token) and the
    # rank computation that assigns dense beam ids; counted below.
    parents = []
    residue = {"rank_sort_items": 0, "backpointer_stores": 0, "backpointer_reads": 0}

    def step(beam, t):
        # claude: decision start — the kept schema is (t, b) so that select_k's key tie-break
        # (token ascending, then parent beam ascending) equals the golden's rule.
        cand = combine(beam, T, keep=("t", "b"))
        # claude: decision end
        surv = select_k(cand, K)
        # CONTROL RESIDUE: rank survivors (score desc, token asc, parent beam asc), K log K compares
        order = sorted(surv.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        rank = {k: i for i, (k, _) in enumerate(order)}
        parents.append([(k[1], k[0]) for k, _ in order])
        residue["rank_sort_items"] += len(order)
        residue["backpointer_stores"] += len(order)
        return scan(surv, lambda k, v: [((rank[k], k[0]), v)], schema=("b", "s"),
                    fn_name="reindex_by_rank", domain={"b": K, "s": V})

    final = iterate(L, step, beam0)
    out = []
    for (b, _s), score in sorted(final.items()):
        # CONTROL RESIDUE: back-tracking through the parent maps, L reads per final beam
        cur, seq = b, []
        for t in range(L - 1, -1, -1):
            pb, tok = parents[t][cur]
            seq.append(tok)
            cur = pb
            residue["backpointer_reads"] += 1
        seq.append(inst["start"])
        seq.reverse()
        out.append((score, seq))
    algebra.last_info = {"residue": residue,
                         "select": current_trace().by_op().get("select", {})}
    return out


def compare(a, b) -> bool:
    if len(a) != len(b):
        return False
    for (sa, qa), (sb, qb) in zip(a, b):
        if qa != qb or abs(sa - sb) > 1e-9 * max(1.0, abs(sa)):
            return False
    return True


def extra_info(inst, res):
    li = algebra.last_info
    sel = li["select"]
    return {
        "V": inst["V"], "K": inst["K"], "L": inst["L"],
        "select_calls": sel.get("count", 0),
        "select_candidates": sel.get("candidates", 0),
        "select_out": sel.get("n_out", 0),
        "best_score": res[0][0],
        "control_residue": li["residue"],
    }
