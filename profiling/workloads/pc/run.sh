#!/usr/bin/env bash
# pyjuice probabilistic-circuit marginal inference: nsys (metric 2 timeline) + ncu (metrics 5, 7, 8).  Untested on this host.
# usage: bash run.sh [small|large|all]     env: NCU_LAUNCH_SKIP / NCU_LAUNCH_COUNT pass through to ncu_wrap.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PROF=$(cd "$HERE/../.." && pwd); S=$PROF/scripts
export SW_VERSION_EXTRA="pyjuice=$(python3 -c 'import importlib.metadata as m; print(m.version("pyjuice"))' 2>/dev/null || echo unknown)"
SEMIRING="(+, x) probability semiring (pyjuice layers compute sums in log space)"
run_one() {  # $1 scale  $2 num_latents  $3 batch
  local scale=$1 inst="hclt-L$2-B$3" args=(--num-latents "$2" --batch "$3")
  bash "$S/nsys_wrap.sh" pc "$scale" -- python3 "$HERE/pc_marginal.py" "${args[@]}"
  python3 "$S/parse_nsys.py" --trace-dir "$(cat "$PROF/traces/pc/LATEST")" --workload pc --tier 1 --scale "$scale" \
    --instance "$inst" --nvtx-range pc_marginal --semiring "$SEMIRING" --share "elementwise=elementwise|vectorized"
  # NVTX filter syntax for push/pop ranges ("<range>/") believed correct; verify with `ncu --help` (--nvtx-include)
  NCU_NVTX_INCLUDE="pc_marginal/" bash "$S/ncu_wrap.sh" pc "$scale" -- python3 "$HERE/pc_marginal.py" "${args[@]}" --iters 2
  python3 "$S/parse_ncu.py" --trace-dir "$(cat "$PROF/traces/pc/LATEST")" --workload pc --tier 1 --scale "$scale" \
    --instance "$inst" --semiring "$SEMIRING"
}
SCALE=${1:-all}
[ "$SCALE" = large ] || run_one small 64 512
[ "$SCALE" = small ] || run_one large 512 512
