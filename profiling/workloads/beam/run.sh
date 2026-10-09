#!/usr/bin/env bash
# Beam search over an LLM (Hugging Face generate with num_beams): nsys separates beam-selection kernels
# (topk / sort / gather) from matmuls; ncu on the same run.  Untested on this host.  usage: bash run.sh [small|large|all]
# GPT-2 weights come from the Hugging Face hub on first use (export HF_HOME=$SYMFABRIC_EXT/hf to keep the cache external).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PROF=$(cd "$HERE/../.." && pwd); S=$PROF/scripts
export SW_VERSION_EXTRA="transformers=$(python3 -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo none)"
SEMIRING="(max, +) over log-probabilities: beam score = sum of token log-probs, top-k selection over batch x beams x vocab"
SEL='topk|radix|sort|gather|index_select|scatter|cumsum|argmax'      # beam-selection kernels (symbolic part)
MM='gemm|gemv|cutlass|xmma|matmul|wgmma|mma'                        # transformer matmuls (neural part)
run_one() {  # $1 scale  $2 model  $3 beams  $4 batch
  local scale=$1 inst="$2-beams$3-batch$4" args=(--model "$2" --num-beams "$3" --batch "$4")
  bash "$S/nsys_wrap.sh" beam "$scale" -- python3 "$HERE/beam_generate.py" "${args[@]}"
  python3 "$S/parse_nsys.py" --trace-dir "$(cat "$PROF/traces/beam/LATEST")" --workload beam --tier 2 --scale "$scale" \
    --instance "$inst" --symbolic-regex "$SEL" --semiring "$SEMIRING" --share "matmul=$MM" --share "softmax=softmax" \
    --notes "S here = beam-selection kernel time / wall; NVTX range generate_beam holds the whole decode loop"
  # ncu on the selection kernels only (the matmul rows are the neural baseline; profile them with NCU_KERNEL_REGEX="$MM")
  NCU_KERNEL_REGEX="$SEL" bash "$S/ncu_wrap.sh" beam "$scale-select" -- python3 "$HERE/beam_generate.py" "${args[@]}" --iters 1
  python3 "$S/parse_ncu.py" --trace-dir "$(cat "$PROF/traces/beam/LATEST")" --workload beam --tier 2 --scale "$scale" \
    --instance "$inst selection-kernels" --semiring "$SEMIRING"
}
SCALE=${1:-all}
[ "$SCALE" = large ] || run_one small gpt2 4 1
[ "$SCALE" = small ] || run_one large gpt2 16 8
