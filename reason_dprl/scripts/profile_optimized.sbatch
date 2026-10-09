#!/bin/bash
#SBATCH --partition=athena
#SBATCH --nodelist=node2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=dprl-opt-profile
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
# Separate cache directories avoid interference between independent experiments.
export SYMFABRIC_NODE_CACHE="${SYMFABRIC_CACHE_BASE:-/tmp/symfabric-opt-profile-$SLURM_JOB_ID}/symfabric-${USER}-torch29-cu130"
mkdir -p "$SYMFABRIC_NODE_CACHE"
export MPLCONFIGDIR="$SYMFABRIC_NODE_CACHE/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$SYMFABRIC_NODE_CACHE/inductor"
export TRITON_CACHE_DIR="$SYMFABRIC_NODE_CACHE/triton"
export CUDA_CACHE_PATH="$SYMFABRIC_NODE_CACHE/cuda"
export TORCHINDUCTOR_COMPILE_THREADS=4
export SYMFABRIC_REPO_ROOT="$PWD"
profiler="$1"
output="$2"
shift 2
mkdir -p "$(dirname "$output")"
mkdir "$output"
profile_local="$SYMFABRIC_NODE_CACHE/profile-$SLURM_JOB_ID"
mkdir -p "$profile_local"
if [[ "$profiler" == nsys ]]; then
    # Node tracing is required to see individual kernels inside CUDA Graphs.
    nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
        --output="$profile_local/native" reason_dprl/.venv/bin/python "$@"
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
        --format csv --output "$profile_local/stats" "$profile_local/native.nsys-rep"
elif [[ "$profiler" == ncu ]]; then
    filters=()
    if [[ -n "${NCU_KERNEL_REGEX:-}" ]]; then
        filters+=(--kernel-name "regex:${NCU_KERNEL_REGEX}")
    fi
    /usr/local/cuda/bin/ncu --target-processes all --nvtx --nvtx-include 'native_query/' \
        "${filters[@]}" --launch-count 12 --set basic --csv \
        --log-file "$profile_local/counters.csv" reason_dprl/.venv/bin/python "$@"
else
    exit 2
fi
cp -a "$profile_local/." "$output/"
