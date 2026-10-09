# Planning (Fast Downward, h^add / h^max) — notes

**Hot loop.** Heuristic evaluation of every generated state: a relaxed-reachability fixpoint over the
propositions and unary operators of the delete-relaxed task. h^add: h(p) = min over achievers o of (cost(o) +
sum of h(q) over preconditions) — (min, +) with a sum over preconditions; h^max: the same with max — (min, max).
Driven by a priority queue and recomputed per state. Around it: best-first search (open-list top-k SELECT,
successor generation, packed-state registry) — the control residue. Q = 1.

**Software counters.** The search component prints `Expanded`, `Evaluated`, `Generated`, `Reopened` state
counts, `Search time`, `Total time`, `Bytes per state`, `Number of registered states` (run.sh copies them into
`sw_counters` / W); `--overall-time-limit` bounds the run.

**What to look for.** `perf_report.txt`, `dso=downward` rows: heuristic classes vs `EagerSearch`, `OpenList`,
`StateRegistry`, `SuccessorGenerator`. If the Python driver or translator dominates the samples (small instance)
use the larger problem or the software's own time split. h^max (A*) expands far more states than greedy h^add;
the two rows show whether the heuristic share depends on the search strategy.
