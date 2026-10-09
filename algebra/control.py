"""Execution structures: fixed-count iteration and monotone fixpoint (semi-naive).

fixpoint(step, X0): repeat X, D = step(X, D, round) until D is empty. `step` receives the
current state X and the current delta D (the tuples that changed in the previous round) and
must return the new state and the new delta; use `update` to compute both. Each round's operator
records are tagged with the round number in the trace.
"""
from __future__ import annotations

from typing import Callable

from .relation import Relation
from .trace import current_trace


def fixpoint(step: Callable[[Relation, Relation, int], tuple[Relation, Relation]],
             X0: Relation, D0: Relation | None = None, max_rounds: int = 1_000_000):
    X, D = X0, (D0 if D0 is not None else X0)
    tr = current_trace()
    rounds = 0
    delta_sizes = []
    while len(D) > 0 and rounds < max_rounds:
        tr.set_iteration(rounds)
        X, D = step(X, D, rounds)
        delta_sizes.append(len(D))
        rounds += 1
    tr.set_iteration(-1)
    return X, {"rounds": rounds, "delta_sizes": delta_sizes}


def iterate(n: int, step: Callable[[Relation, int], Relation], X0: Relation) -> Relation:
    X = X0
    tr = current_trace()
    for t in range(n):
        tr.set_iteration(t)
        X = step(X, t)
    tr.set_iteration(-1)
    return X
