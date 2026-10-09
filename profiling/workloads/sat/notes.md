# SAT / BCP (Kissat) — notes

**Hot loop.** Boolean constraint propagation: for each newly assigned literal walk its watch list; for each
clause whose watched literal became false find a replacement watch, else derive a unit (implication) or a
conflict. Algebra: Boolean (clause = OR of literals, formula = AND of clauses); the "false literals per clause"
count is the (+, x) counting view REASON's reduction tree uses. Structure: dynamic sparse traversal (watch lists
mutate on every visit, learned clauses grow the DB). Q = 1 per solver instance.

**Software counters.** `kissat -s` statistics block: `propagations`, `conflicts`, `decisions`, `restarts`,
`learned`, `process-time`, `maximum-resident-set-size` (copied into `sw_counters` by run.sh); `-v` adds progress
lines; `--help` lists the limits (`--time`, `--conflicts`; names to verify).

**What to look for.** `perf_report.txt`: `*propagate*` / `*assign*` samples = acceleratable BCP share;
`analyze`, `decide`, `reduce`, `restart`, `vivify`, `eliminate`, `walk` = control residue R. `perf_stat.txt`:
LLC-load-misses per k-instructions across the two instance sizes vs W is the clause-DB locality signal for
H4/H7; `PERF_MEM=1` adds the memory-level breakdown as the reuse-distance proxy.

**Instances.** Random 3-SAT near the threshold is propagation-heavy but not industrial; SAT Competition instances
(structured, large clause DBs, Tier 1) are the target; SATLIB `uf250` files are sanity checks only.
