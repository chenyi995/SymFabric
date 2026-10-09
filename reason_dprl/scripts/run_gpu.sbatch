#!/bin/bash
#SBATCH --partition=athena
#SBATCH --nodelist=node2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=dprl-baselines
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
# Compute-node NFS metadata/write latency dominated cold imports and compilation.
# Keep generated compiler files on node-local storage; retain benchmark outputs.
export SYMFABRIC_NODE_CACHE="${SYMFABRIC_CACHE_BASE:-/dev/shm}/symfabric-${USER}-torch29-cu130"
mkdir -p "$SYMFABRIC_NODE_CACHE"
export MPLCONFIGDIR="$SYMFABRIC_NODE_CACHE/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$SYMFABRIC_NODE_CACHE/inductor"
export TRITON_CACHE_DIR="$SYMFABRIC_NODE_CACHE/triton"
export CUDA_CACHE_PATH="$SYMFABRIC_NODE_CACHE/cuda"
export TORCHINDUCTOR_COMPILE_THREADS=4
export SYMFABRIC_REPO_ROOT="$PWD"
if [[ -n "${SYMFABRIC_HARNESS_ARCHIVE:-}" ]]; then
    runner="$1"
    shift
    harness="$SYMFABRIC_NODE_CACHE/harness-$SLURM_JOB_ID"
    mkdir -p "$harness"
    tar -xzf "$SYMFABRIC_HARNESS_ARCHIVE" -C "$harness"
    exec reason_dprl/.venv/bin/python "$harness/$(basename "$runner")" "$@"
fi
exec reason_dprl/.venv/bin/python "$@"
