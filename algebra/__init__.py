"""SymFabric operator algebra: semirings, weighted relations, six operators, control.

The algebra is the "what": a small set of typed operators over finite weighted relations
R : D_1 x ... x D_n -> K, where K is a semiring. Every operator records the work it did
(candidates examined, outputs produced, semiring operations) into the active Trace so that
workload decompositions can be checked for work efficiency and later mapped onto hardware.

Nothing in this package knows about hardware. The architecture models in systolicarch/ and
pim/ consume the Trace produced here.
"""
from .semiring import (Semiring, BOOLEAN, COUNTING, PROB, LOGPROB, VITERBI, MAXPLUS,
                       MINPLUS, FUZZY, REAL, BOTTLENECK, SEMIRINGS)
from .relation import Relation
from .ops import combine, reduce, filter, mask, update, select_k, rename, project, scan
from .control import fixpoint, iterate
from .trace import Trace, OpRecord, current_trace, use_trace

__all__ = [
    "Semiring", "BOOLEAN", "COUNTING", "PROB", "LOGPROB", "VITERBI", "MAXPLUS", "MINPLUS",
    "FUZZY", "REAL", "BOTTLENECK", "SEMIRINGS", "Relation", "combine", "reduce", "filter", "mask", "update",
    "select_k", "rename", "project", "scan", "fixpoint", "iterate", "Trace", "OpRecord",
    "current_trace", "use_trace",
]
