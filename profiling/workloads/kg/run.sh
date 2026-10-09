#!/usr/bin/env bash
# Knowledge-graph multi-hop: option B (default) = sparse (max, x) propagation stand-in (maxprod_hops.py);
# option A = NBFNet (KG_OPTION=nbfnet; needs torchdrug; command unverified, see $SYMFABRIC_EXT/NBFNet/README.md).
# Untested on this host.  usage: bash run.sh [small|large|all]   env: KG_EDGES_LARGE (SNAP file for the large scale)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PROF=$(cd "$HERE/../.." && pwd); S=$PROF/scripts
EXT=${SYMFABRIC_EXT:-/data2/$USER/symfabric_ext}
export SW_VERSION_EXTRA="maxprod_hops.py stand-in"
run_standin() {  # $1 scale  $2 semiring  $3.. maxprod_hops.py arguments
  local scale=$1 sr=$2; shift 2
  local inst="$sr-$*" ; inst=${inst//[ -]/_}
  local semiring; case $sr in maxprod) semiring="(max, x): best-path score over <= L hops" ;; boolean) semiring="Boolean (OR, AND): L-hop reachability" ;; sumprod) semiring="(+, x): weighted path count" ;; esac
  bash "$S/nsys_wrap.sh" kg "$scale-$sr" -- python3 "$HERE/maxprod_hops.py" --semiring "$sr" "$@"
  python3 "$S/parse_nsys.py" --trace-dir "$(cat "$PROF/traces/kg/LATEST")" --workload kg --tier 2 --scale "$scale" \
    --instance "$inst" --nvtx-range kg_hops --semiring "$semiring" --share "scatter=scatter|reduce" --share "gather=index|gather"
  NCU_NVTX_INCLUDE="kg_hops/" bash "$S/ncu_wrap.sh" kg "$scale-$sr" -- python3 "$HERE/maxprod_hops.py" --semiring "$sr" "$@" --iters 2
  python3 "$S/parse_ncu.py" --trace-dir "$(cat "$PROF/traces/kg/LATEST")" --workload kg --tier 2 --scale "$scale" \
    --instance "$inst" --semiring "$semiring"
}
run_nbfnet() {  # option A, unverified: NBFNet's own driver (config path per its README)
  (cd "$EXT/NBFNet" && bash "$S/nsys_wrap.sh" kg "nbfnet-$1" -- python3 script/run.py -c config/knowledge_graph/fb15k237.yaml --gpus "[0]")
}
SCALE=${1:-all}
if [ "${KG_OPTION:-standin}" = nbfnet ]; then run_nbfnet "$SCALE"; exit; fi
[ "$SCALE" = large ] || { run_standin small maxprod --nodes 100000 --edges 1000000 --batch 64; run_standin small boolean --nodes 100000 --edges 1000000 --batch 64; }
if [ "$SCALE" != small ]; then
  if [ -n "${KG_EDGES_LARGE:-}" ]; then run_standin large maxprod --edges-file "$KG_EDGES_LARGE" --batch 16
  else run_standin large maxprod --nodes 1000000 --edges 10000000 --batch 16; fi
fi
