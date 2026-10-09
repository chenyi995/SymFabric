#!/usr/bin/env python3
"""cuGraph BFS / SSSP / triangle count on a SNAP edge list, one NVTX range per algorithm.  API per the cuGraph
docs (Graph.from_cudf_edgelist, cugraph.bfs, cugraph.sssp, cugraph.triangle_count); not checked on this host.
Each BFS / SSSP call has one source (Q = 1); --sources runs several back to back."""
import argparse, time  # noqa: E401
import cudf, cugraph, cupy, nvtx  # noqa: E401  (nvtx ships with RAPIDS)
ap = argparse.ArgumentParser()
ap.add_argument("--edges", required=True, help="SNAP txt or txt.gz")
ap.add_argument("--sources", type=int, default=8)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--algos", default="bfs,sssp,tc")
a = ap.parse_args()
df = cudf.read_csv(a.edges, sep="\t", comment="#", names=["src", "dst"], dtype=["int32", "int32"])
df["w"] = cudf.Series(cupy.random.RandomState(0).random_sample(len(df), dtype=cupy.float32) + 0.1)
G = cugraph.Graph(directed=True); G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="w", renumber=True)
Gu = cugraph.Graph(directed=False); Gu.from_cudf_edgelist(df[["src", "dst"]], source="src", destination="dst", renumber=True)
srcs = df["src"].unique().to_arrow().to_pylist()[: a.sources]
algos = a.algos.split(","); t = {k: 0.0 for k in algos}
for _ in range(a.iters):
    for k in algos:
        cupy.cuda.Device().synchronize(); t0 = time.perf_counter()
        with nvtx.annotate(k):
            if k == "bfs":
                for s in srcs: cugraph.bfs(G, start=s)
            elif k == "sssp":
                for s in srcs: cugraph.sssp(G, source=s)
            elif k == "tc":
                cugraph.triangle_count(Gu)
        cupy.cuda.Device().synchronize(); t[k] += time.perf_counter() - t0
print(f"SYMFABRIC_METRIC W_bytes {len(df) * 12 + (G.number_of_vertices() + 1) * 8}")  # edge list (2 x int32 + f32) + CSR offsets
print(f"SYMFABRIC_METRIC Q 1")                                                       # one source per BFS / SSSP call
print(f"SYMFABRIC_METRIC nodes {G.number_of_vertices()}\nSYMFABRIC_METRIC edges {G.number_of_edges()}")
for k in algos: print(f"SYMFABRIC_METRIC {k}_time_s {t[k]:.4f}")
