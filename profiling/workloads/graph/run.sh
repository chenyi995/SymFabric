#!/usr/bin/env bash
# Graph BFS / SSSP / triangle count: cuGraph (GPU: nsys + ncu) and SuiteSparse:GraphBLAS through the LAGraph demos
# (CPU: perf) on SNAP graphs fetched by setup_env.sh.  Untested on this host.  usage: bash run.sh [small|large|all]
# env: GRAPH_SMALL / GRAPH_LARGE (SNAP txt.gz; default web-Google / soc-LiveJournal1); SKIP_CUGRAPH=1; SKIP_GRAPHBLAS=1
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PROF=$(cd "$HERE/../.." && pwd); S=$PROF/scripts
EXT=${SYMFABRIC_EXT:-/data2/$USER/symfabric_ext}; D=$EXT/data; LAG=$EXT/LAGraph/build/src/benchmark
export SW_VERSION_EXTRA="cugraph=$(python3 -c 'import cugraph; print(cugraph.__version__)' 2>/dev/null || echo none); graphblas=$(git -C "$EXT/GraphBLAS" describe --tags 2>/dev/null || echo unknown)"
semiring() { case $1 in bfs) echo "Boolean (OR, AND) / (any, pair): next = A^T frontier, masked by unvisited" ;;
                       sssp) echo "(min, +): dist = min(dist, dist (min,+) A)" ;;
                       tc) echo "(+, x) on Boolean adjacency: sum of (L * U) masked by A (LAGraph tc uses plus_pair)" ;; esac; }
run_cugraph() {  # $1 scale  $2 SNAP file
  local scale=$1 g=$2 inst; inst=$(basename "$g" .txt.gz)
  bash "$S/nsys_wrap.sh" graph "cugraph-$scale" -- python3 "$HERE/cugraph_run.py" --edges "$g"
  for algo in bfs sssp tc; do   # one row per NVTX range (S per algorithm)
    python3 "$S/parse_nsys.py" --trace-dir "$(cat "$PROF/traces/graph/LATEST")" --workload graph --tier 2 --scale "$scale" \
      --instance "$inst cugraph $algo" --nvtx-range "$algo" --semiring "$(semiring $algo)"
  done
  NCU_NVTX_INCLUDE="bfs/" bash "$S/ncu_wrap.sh" graph "cugraph-bfs-$scale" -- python3 "$HERE/cugraph_run.py" --edges "$g" --iters 1 --algos bfs
  python3 "$S/parse_ncu.py" --trace-dir "$(cat "$PROF/traces/graph/LATEST")" --workload graph --tier 2 --scale "$scale" \
    --instance "$inst cugraph bfs" --semiring "$(semiring bfs)"
}
run_graphblas() {  # $1 scale  $2 SNAP file
  local scale=$1 g=$2 base; base=$D/$(basename "$g" .txt.gz)
  [ -s "$base.mtx" ] || python3 "$HERE/snap_to_mtx.py" "$g" "$base.mtx" --symmetric
  [ -s "$base-w.mtx" ] || python3 "$HERE/snap_to_mtx.py" "$g" "$base-w.mtx" --weights
  # metric 1: CSR with 64-bit indices (GraphBLAS default), edges x 8 + (nodes + 1) x 8 (values excluded for pattern)
  local n m; read -r _ n _ m < "$base.mtx.stats"
  local demo mtx
  for demo in bfs_demo sssp_demo tc_demo; do   # demo names and argument conventions: verify with `ls $LAG` and the sources
    mtx=$base.mtx; [ $demo = sssp_demo ] && mtx=$base-w.mtx
    bash "$S/perf_wrap.sh" graph "graphblas-${demo%_demo}-$scale" -- "$LAG/$demo" "$mtx"
    python3 "$S/parse_perf.py" --trace-dir "$(cat "$PROF/traces/graph/LATEST")" --workload graph --tier 2 --scale "$scale" \
      --instance "$(basename "$g" .txt.gz) graphblas ${demo%_demo}" --Q 1 --W-bytes $((m * 8 + (n + 1) * 8)) \
      --W-method "CSR: edges x 8 + (nodes+1) x 8, 64-bit indices, values excluded" --semiring "$(semiring ${demo%_demo})" \
      --hot-regex 'GB_AxB|saxpy|GB_dot|GB_emult|GB_ewise|GB_masker|GB_select|GB_bitmap|GB_reduce|GB_apply|GB_add' \
      --control-regex 'GB_wait|GB_transplant|GB_alloc|GB_free|GB_convert|GB_builder|GB_sort|malloc|free|LAGraph_|LG_|MM_|read' \
      --notes "symbol regexes are first guesses; re-classify from perf_report.txt; demo prints its own per-trial times"
  done
}
SCALE=${1:-all}
for sc in small large; do
  [ "$SCALE" = all ] || [ "$SCALE" = "$sc" ] || continue
  [ $sc = small ] && g=${GRAPH_SMALL:-$D/web-Google.txt.gz} || g=${GRAPH_LARGE:-$D/soc-LiveJournal1.txt.gz}
  [ "${SKIP_CUGRAPH:-0}" = 1 ] || run_cugraph "$sc" "$g"
  [ "${SKIP_GRAPHBLAS:-0}" = 1 ] || run_graphblas "$sc" "$g"
done
