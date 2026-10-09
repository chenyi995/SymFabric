# Datalog (Souffle) — notes

**Hot loop.** Semi-naive evaluation of a recursive rule, `delta' = (delta JOIN R) minus known`: indexed nested-loop
joins over B-tree (or Brie) indices. Algebra: Boolean — AND over body atoms, OR / set union over derivations.
Structure: sparse fixpoint over dynamic relations that grow every iteration. `tc.dl` is the two-atom join;
`andersen.dl` has three-atom joins through the mutually dependent `pt`, the rule shape of Doop-style points-to.

**Software counters.** `-p profile.log`: per-rule / per-relation times, tuple and iteration counts (read with
`souffleprof profile.log`; verify flags with `souffleprof -h`). IDB sizes: `wc -l out/path.csv`, `out/pt.csv`
(used for W). `-j N` = worker threads = Q.

**What to look for.** `perf_report.txt`: B-tree search/insert symbols (INDEX / COMBINE) vs parsing, symbol
tables, allocation (control residue). `perf_stat.txt`: LLC-load-misses per instruction grows at the large scale
once the indices leave L2/L3 (metric 5); compare W across scales.

**Not scripted.** Real program-analysis inputs: Doop (https://bitbucket.org/yanniss/doop; Java, Gradle, large).
GPU Datalog: GDlog (https://github.com/harp-lab/GDlog) is cloned by setup_env.sh; whether it is maintained and builds
against current CUDA was not verified. No maintained cuDF-based Datalog engine is known to the author.
