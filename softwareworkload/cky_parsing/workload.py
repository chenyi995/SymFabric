"""CKY inside algorithm for a PCFG in Chomsky normal form (Goodman's semiring parsing).

    span(i, i+1, A) = rule(A -> w_i)
    span(i, j, A)   = (+)_{k, B, C}  rule(A -> B C) (x) span(i, k, B) (x) span(k, j, C)

Run twice: PROB (inside probability; (+, x)) and VITERBI (best-parse log score; (max, +) on log
rule weights). Both passes use the same operator chain; only the semiring changes.

Algebra program (default: the prescribed "pair-first" join order), one round per span length l:
    L      = rename(spans, j->k, A->B)                        # (i, k, B)       metadata only
    Rr     = scan(spans, shift_end_l)                         # (k, i, C): i = j - l, kept only if j >= l
    P      = combine(L, Rr, keep=(i, B, C))                   # join on (i, k): exactly the adjacent pairs
                                                              # with total length l; fused (+) over k
    cell   = combine(P, rules(A, B, C), keep=(i, A))          # join on (B, C); fused (+) over (B, C)
    new    = scan(cell, add_end_l)                            # (i, A) -> (i, i + l, A)
    spans  = update(spans, new)                               # union on disjoint keys
The length constraint j - i = l is carried by the re-keying scan, so the natural join equates
i_left = j_right - l as well as k; no candidate is produced and then discarded.

Module switches (read by algebra(); the harness and the tests measure every setting):
    ORDER       = "pair_first" (prescribed) | "rule_first" (T(i,k,A,C) = (+)_B span(i,k,B) (x) rule(A,B,C)
                  is built once per completed cell and joined with the right child on (k, i, C))
    LENGTH_MODE = "rekey" (scan-based, default) | "filter" (join all adjacent spans on k, then
                  filter j - i == l; the filter variant materializes every adjacent pair every round)

Golden: the textbook CKY triple loop with rules indexed by their left child, plain Python dicts.
Both sides count "(i, k, j, rule) combinations": one per binary rule A -> B C applied at a split
point where both children have a non-zero value (one ternary product), per semiring pass.
"""
from __future__ import annotations

import math
import random

from algebra import PROB, VITERBI, Relation, combine, update, rename, scan, filter as afilter

NAME = "cky_parsing"
CATEGORY = "parsing"
SEMIRINGS = ["prob", "viterbi"]
STRUCTURE = "dag"
HOTSPOT_EVENT = ("(i, k, j, rule) combinations: one ternary product rule(A->BC) x span(i,k,B) x "
                 "span(k,j,C) with both children non-zero, counted in each of the two semiring passes")
SCALES = {"small": {"N": 12, "n": 20}, "large": {"N": 24, "n": 45}}

ORDER = "pair_first"          # "pair_first" | "rule_first"
LENGTH_MODE = "rekey"         # "rekey" | "filter"


# ---- instance -------------------------------------------------------------------------------
def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = random.Random(seed)
    N, n = p["N"], p["n"]
    # claude: decision start — vocabulary size V = N; each non-terminal gets 3..5 binary rules with
    # distinct right-hand sides and 3 lexical rules, weights uniform in [0.5, 1.5] normalized per
    # left-hand side so the grammar is a proper PCFG; the start symbol is non-terminal 0.
    V = N
    binary = []                                  # (A, B, C, p)
    lexical = []                                 # (A, w, p)
    for A in range(N):
        nb = rng.randint(3, 5)
        rhs = set()
        while len(rhs) < nb:
            rhs.add((rng.randrange(N), rng.randrange(N)))
        words = rng.sample(range(V), 3)
        w_bin = [rng.uniform(0.5, 1.5) for _ in rhs]
        w_lex = [rng.uniform(0.5, 1.5) for _ in words]
        tot = sum(w_bin) + sum(w_lex)
        for (B, C), w in zip(sorted(rhs), w_bin):
            binary.append((A, B, C, w / tot))
        for wd, w in zip(words, w_lex):
            lexical.append((A, wd, w / tot))
    # claude: decision end
    # claude: decision start — the sentence is generated from the grammar: a uniformly random
    # binary bracketing with n leaves is labelled top-down from the start symbol by sampling one of
    # the label's binary rules (internal node) or lexical rules (leaf) in proportion to its weight.
    bin_by_lhs = {}
    lex_by_lhs = {}
    for A, B, C, pr in binary:
        bin_by_lhs.setdefault(A, []).append((B, C, pr))
    for A, wd, pr in lexical:
        lex_by_lhs.setdefault(A, []).append((wd, pr))

    def shape(m):
        if m == 1:
            return None
        s = rng.randint(1, m - 1)
        return (s, shape(s), shape(m - s))

    def label(tree, A, out):
        if tree is None:
            opts = lex_by_lhs[A]
            r = rng.random() * sum(pr for _, pr in opts)
            for wd, pr in opts:
                r -= pr
                if r <= 0:
                    break
            out.append(wd)
            return
        opts = bin_by_lhs[A]
        r = rng.random() * sum(pr for _, _, pr in opts)
        for B, C, pr in opts:
            r -= pr
            if r <= 0:
                break
        label(tree[1], B, out)
        label(tree[2], C, out)

    sentence = []
    label(shape(n), 0, sentence)
    # claude: decision end
    return {"N": N, "V": V, "n": n, "binary": binary, "lexical": lexical, "sentence": sentence}


# ---- golden -----------------------------------------------------------------------------------
def _cky(inst, mul, add, lex_val, rule_val):
    """Generic CKY over a (add, mul) pair; returns the chart {(i, j, A): value} and the event count.
    Rules are indexed by their left child; a combination is counted when both children exist."""
    n = inst["n"]
    words = inst["sentence"]
    by_left = {}
    for A, B, C, pr in inst["binary"]:
        by_left.setdefault(B, []).append((A, C, rule_val(pr)))
    lex_by_word = {}
    for A, wd, pr in inst["lexical"]:
        lex_by_word.setdefault(wd, []).append((A, lex_val(pr)))
    cells = {}
    for i in range(n):
        cells[(i, i + 1)] = {A: v for A, v in lex_by_word[words[i]]}
    events = 0
    probes = 0
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            cell = {}
            for k in range(i + 1, j):
                left = cells[(i, k)]
                right = cells[(k, j)]
                for B, vb in left.items():
                    for A, C, vr in by_left.get(B, ()):
                        vc = right.get(C)
                        if vc is None:
                            probes += 1
                            continue
                        events += 1
                        v = mul(mul(vr, vb), vc)
                        if A in cell:
                            cell[A] = add(cell[A], v)
                        else:
                            cell[A] = v
            cells[(i, j)] = cell
    chart = {(i, j, A): v for (i, j), cell in cells.items() for A, v in cell.items()}
    return chart, events, probes


def native(inst):
    n = inst["n"]
    chart_p, ev_p, pr_p = _cky(inst, lambda a, b: a * b, lambda a, b: a + b,
                               lambda p: p, lambda p: p)
    chart_v, ev_v, pr_v = _cky(inst, lambda a, b: a + b, max,
                               lambda p: math.log(p), lambda p: math.log(p))
    native.last_probes = pr_p + pr_v
    return {"inside_root": chart_p.get((0, n, 0), 0.0), "best_root": chart_v.get((0, n, 0), -math.inf),
            "chart_prob": chart_p, "chart_vit": chart_v}, ev_p + ev_v


# ---- algebra ------------------------------------------------------------------------------------
def _shift_end(length):
    """scan fn: span (i_r, j_r, C) -> key (k = i_r, i = j_r - length, C); dropped when j_r < length.
    O(1) per tuple: one subtraction and one comparison."""
    def fn(k, v):
        if k[1] >= length:
            return ((k[0], k[1] - length, k[2]), v),
        return ()
    return fn


def _add_end(length):
    """scan fn: (i, A) -> (i, i + length, A). O(1) per tuple: one addition."""
    def fn(k, v):
        return ((k[0], k[0] + length, k[1]), v),
    return fn


def _len_eq(length):
    """filter predicate for the filter variant: keep (i, j, ...) with j - i == length. O(1)."""
    def pred(key, v):
        return key[1] - key[0] == length
    return pred


def _parse(inst, sr, rule_val):
    N, n, V = inst["N"], inst["n"], inst["V"]
    dom = {"i": n + 1, "j": n + 1, "k": n + 1, "A": N, "B": N, "C": N, "w": V}
    words = Relation(("i", "j", "w"), sr, {(i, i + 1, w): sr.one for i, w in enumerate(inst["sentence"])},
                     dom)
    lex = Relation(("w", "A"), sr, {(wd, A): rule_val(pr) for A, wd, pr in inst["lexical"]}, dom)
    rules = Relation(("A", "B", "C"), sr, {(A, B, C): rule_val(pr) for A, B, C, pr in inst["binary"]},
                     dom)
    spans = combine(words, lex, keep=("i", "j", "A"))                      # lexical spans, length 1
    if ORDER == "rule_first":
        # T(i, k, A, C) = (+)_B span(i, k, B) (x) rule(A, B, C), built once per completed cell
        T = combine(rename(spans, {"j": "k", "A": "B"}), rules, keep=("i", "k", "A", "C"))
    for length in range(2, n + 1):
        if LENGTH_MODE == "filter":
            # claude: decision start — filter variant kept for measurement only: every adjacent pair
            # of known spans is joined on k each round and the length constraint is applied afterwards.
            Rr = rename(spans, {"i": "k", "A": "C"})                           # (k, j, C), every length
            if ORDER == "rule_first":
                cand = combine(T, Rr, keep=("i", "j", "A"))                    # join on (k, C)
                new = afilter(cand, _len_eq(length), name="len_eq")
            else:
                L = rename(spans, {"j": "k", "A": "B"})                        # (i, k, B)
                P = combine(L, Rr, keep=("i", "j", "B", "C"))                  # join on k, every total length
                P = afilter(P, _len_eq(length), name="len_eq")
                new = combine(P, rules, keep=("i", "j", "A"))
            # claude: decision end
        else:
            # claude: decision start — the constraint j - i = l is expressed by re-keying the right
            # operand (scan, O(1) per tuple) so that the natural join on (i, k) admits exactly the
            # adjacent pairs of total length l; per-length relations would need `update` as the union
            # over split points, and update() drops PROB contributions below its 1e-9 tolerance.
            Rr = scan(spans, _shift_end(length), schema=("k", "i", "C"), fn_name="shift_end", domain=dom)
            if ORDER == "rule_first":
                cell = combine(T, Rr, keep=("i", "A"))                         # join on (i, k, C)
            else:
                L = rename(spans, {"j": "k", "A": "B"})                        # (i, k, B)
                P = combine(L, Rr, keep=("i", "B", "C"))                       # join on (i, k)
                cell = combine(P, rules, keep=("i", "A"))                      # join on (B, C)
            new = scan(cell, _add_end(length), schema=("i", "j", "A"), fn_name="add_end", domain=dom)
            # claude: decision end
        spans, _ = update(spans, new)                                          # disjoint keys
        if ORDER == "rule_first" and length < n:
            Tn = combine(rename(new, {"j": "k", "A": "B"}), rules, keep=("i", "k", "A", "C"))
            T, _ = update(T, Tn)                                               # disjoint keys
    return spans


def algebra(inst):
    n = inst["n"]
    sp = _parse(inst, PROB, lambda p: p)
    sv = _parse(inst, VITERBI, lambda p: math.log(p))
    algebra.last_info = {"order": ORDER, "length_mode": LENGTH_MODE,
                         "spans_prob": len(sp), "spans_vit": len(sv)}
    return {"inside_root": sp.get((0, n, 0)), "best_root": sv.get((0, n, 0)),
            "chart_prob": dict(sp.items()), "chart_vit": dict(sv.items())}


def _close(a, b, rtol=1e-9):
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


def _same_chart(a, b):
    if len(a) != len(b):
        return False
    for k, v in a.items():
        w = b.get(k)
        if w is None or not _close(v, w):
            return False
    return True


def compare(a, b) -> bool:
    return (_close(a["inside_root"], b["inside_root"]) and _close(a["best_root"], b["best_root"])
            and _same_chart(a["chart_prob"], b["chart_prob"])
            and _same_chart(a["chart_vit"], b["chart_vit"]))


def extra_info(inst, res):
    N, n = inst["N"], inst["n"]
    n_cells = n * (n + 1) // 2
    return {"order": ORDER, "length_mode": LENGTH_MODE, "binary_rules": len(inst["binary"]),
            "distinct_rhs": len({(B, C) for _, B, C, _ in inst["binary"]}),
            "chart_size": len(res["chart_prob"]), "cell_fill": len(res["chart_prob"]) / (n_cells * N),
            "native_probes_no_right_child": getattr(native, "last_probes", None),
            "inside_root": res["inside_root"], "best_root": res["best_root"]}
