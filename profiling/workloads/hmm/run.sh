#!/usr/bin/env bash
# HMM / constrained decoding: option B (default) = plain PyTorch HMM forward pass stand-in (hmm_forward.py);
# option A = Ctrl-G's own HMM (HMM_OPTION=ctrlg; command below is unverified, follow the Ctrl-G README).
# Untested on this host.  usage: bash run.sh [small|large|all]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PROF=$(cd "$HERE/../.." && pwd); S=$PROF/scripts
EXT=${SYMFABRIC_EXT:-/data2/$USER/symfabric_ext}
export SW_VERSION_EXTRA="hmm_forward.py stand-in"
run_standin() {  # $1 scale  $2 states  $3 steps  $4 batch  $5 semiring
  local scale=$1 inst="K$2-T$3-B$4-$5" args=(--states "$2" --steps "$3" --batch "$4" --semiring "$5") semiring
  case $5 in sumprod) semiring="(+, x) forward algorithm, renormalised per step" ;; maxplus) semiring="(max, +) Viterbi in log space" ;; esac
  bash "$S/nsys_wrap.sh" hmm "$scale-$5" -- python3 "$HERE/hmm_forward.py" "${args[@]}"
  python3 "$S/parse_nsys.py" --trace-dir "$(cat "$PROF/traces/hmm/LATEST")" --workload hmm --tier 1 --scale "$scale" \
    --instance "$inst" --nvtx-range hmm_forward --semiring "$semiring" --share "gemm=gemm|gemv|cutlass|xmma" --share "gather=index|gather"
  NCU_NVTX_INCLUDE="hmm_forward/" bash "$S/ncu_wrap.sh" hmm "$scale-$5" -- python3 "$HERE/hmm_forward.py" "${args[@]}" --iters 2
  python3 "$S/parse_ncu.py" --trace-dir "$(cat "$PROF/traces/hmm/LATEST")" --workload hmm --tier 1 --scale "$scale" \
    --instance "$inst" --semiring "$semiring"
}
run_ctrlg() {  # option A, unverified: Ctrl-G ships pretrained HMMs on the Hugging Face hub (ctrlg/hmm_gpt2-large_*);
  # the model id, class name and generation API must be taken from $EXT/Ctrl-G/README.md before this is run
  bash "$S/nsys_wrap.sh" hmm "ctrlg-$1" -- python3 - <<'PY'
import ctrlg, torch  # API not verified on this host
hmm = ctrlg.HMM.from_pretrained("ctrlg/hmm_gpt2-large_common-gen_4096").to("cuda")
raise SystemExit("fill in the constrained-generation call from the Ctrl-G README (tutorial notebook) before profiling")
PY
}
SCALE=${1:-all}
if [ "${HMM_OPTION:-standin}" = ctrlg ]; then run_ctrlg "$SCALE"; exit; fi
[ "$SCALE" = large ] || { run_standin small 256 64 32 sumprod; run_standin small 256 64 32 maxplus; }
[ "$SCALE" = small ] || { run_standin large 4096 128 32 sumprod; run_standin large 4096 128 8 maxplus; }
