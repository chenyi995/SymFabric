# Knowledge-graph multi-hop (NBFNet stand-in) — notes

**Hot loop.** Bellman-Ford-style propagation from a source entity, x_{k+1}(v) = (+)_{(u,v) in E} x_k(u) (x) w(u,v),
iterated L hops. NBFNet learns the (+)/(x) (sum or max, relation-conditioned messages) on top of exactly this
message passing; the stand-in fixes the semiring: (max, x) best path, Boolean reachability, (+, x) path count.
Structure: sparse gather along edges (x[src]) + scatter-reduce by destination; static graph shared by B queries
(Q = B), state N x B dense.

**Software counters.** `maxprod_hops.py` prints W (edge list + state), Q, node/edge counts, hot time and the
(mul, reduce) pair count. NBFNet (option A) logs epoch time and metrics through torchdrug.

**What to look for.** nsys: `kg_hops` range vs wall; gather (`index`) vs scatter-reduce kernel shares. ncu:
sectors per request of the gather kernel is the random-access signature (metric 5); DRAM utilisation of the
scatter-reduce; arithmetic intensity should grow with B — the plan's batching (Q) argument. Compare the small
graph (fits in L2) with the large or a SNAP file (`KG_EDGES_LARGE=$SYMFABRIC_EXT/data/soc-LiveJournal1.txt.gz`).
