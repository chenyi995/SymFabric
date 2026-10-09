# Graph BFS / SSSP / triangle count (cuGraph, SuiteSparse:GraphBLAS) — notes

**Hot loop.** Masked sparse matrix-vector / matrix-matrix products over a semiring. BFS: frontier' = (A^T (x)
frontier) masked by not-visited, Boolean (any, pair) in GraphBLAS terms, a frontier fixpoint. SSSP
(delta-stepping in LAGraph, Dijkstra-like in cuGraph): dist' = min(dist, dist (min,+) A), a (min, +) fixpoint.
Triangle count: sum of (L (x) U) masked by A, (+, x) over Boolean matrices, one masked SpGEMM, static. The graph
is shared; cuGraph runs one source per call (Q = 1); workloads/kg holds the multi-source variant.

**Software counters.** `cugraph_run.py` prints W (edge list + CSR offsets), Q, node/edge counts, per-algorithm
times; cuGraph result frames give reached vertices per source. The LAGraph demos print per-trial times and the
number of sources on stdout (`stdout.log`); `snap_to_mtx.py` writes node/edge counts to `<file>.mtx.stats`.

**What to look for.** GPU (nsys): per-algorithm NVTX ranges vs wall; frontier kernels vs renumbering/memops.
ncu on BFS: sectors per request (neighbour-list gathers), L2 hit rate on web-Google (CSR fits L2) vs
soc-LiveJournal1 (does not). CPU (perf): `GB_AxB*` / `GB_masker` / `GB_emult` (semiring kernels) vs `GB_wait`,
memory management, I/O (control); LLC-load-misses per instruction across scales is the working-set signal.
