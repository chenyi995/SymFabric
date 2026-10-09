#!/usr/bin/env bash
# VSA (torchhd) cleanup memory and resonator: nsys + ncu.  Untested on this host.  usage: bash run.sh [small|large|all]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PROF=$(cd "$HERE/../.." && pwd); S=$PROF/scripts
export SW_VERSION_EXTRA="torchhd=$(python3 -c 'import torchhd; print(torchhd.__version__)' 2>/dev/null || echo none)"
SEMIRING="(+, x) inner product then argmax (max selection); bind = elementwise x; bundle = + then sign"
run_one() {  # $1 scale  $2 items  $3 factors  $4 batch
  local scale=$1 inst="d10000-M$2-F$3-B$4" args=(--items "$2" --factors "$3" --batch "$4")
  bash "$S/nsys_wrap.sh" vsa "$scale" -- python3 "$HERE/vsa_cleanup.py" "${args[@]}"
  # two NVTX ranges: one parse row per range so cleanup and resonator get separate S values
  for r in vsa_cleanup vsa_resonator; do
    python3 "$S/parse_nsys.py" --trace-dir "$(cat "$PROF/traces/vsa/LATEST")" --workload vsa --tier 3 --scale "$scale" \
      --instance "$inst $r" --nvtx-range "$r" --semiring "$SEMIRING" --share "gemm=gemm|gemv|cutlass|xmma" --share "argmax=argmax|reduce"
  done
  NCU_NVTX_INCLUDE="vsa_cleanup/" bash "$S/ncu_wrap.sh" vsa "$scale-cleanup" -- python3 "$HERE/vsa_cleanup.py" "${args[@]}" --iters 2
  python3 "$S/parse_ncu.py" --trace-dir "$(cat "$PROF/traces/vsa/LATEST")" --workload vsa --tier 3 --scale "$scale" --instance "$inst cleanup" --semiring "$SEMIRING"
  NCU_NVTX_INCLUDE="vsa_resonator/" bash "$S/ncu_wrap.sh" vsa "$scale-resonator" -- python3 "$HERE/vsa_cleanup.py" "${args[@]}" --iters 2
  python3 "$S/parse_ncu.py" --trace-dir "$(cat "$PROF/traces/vsa/LATEST")" --workload vsa --tier 3 --scale "$scale" --instance "$inst resonator" --semiring "$SEMIRING"
}
SCALE=${1:-all}
[ "$SCALE" = large ] || run_one small 10000 100 256
[ "$SCALE" = small ] || run_one large 50000 1000 256     # 50000 x 10000 float32 memory = 2 GB on the device
