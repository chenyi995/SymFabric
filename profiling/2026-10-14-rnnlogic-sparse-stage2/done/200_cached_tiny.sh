#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash run_pinned.sh test_cached_sparse_predictor.py --device cuda --output cached_sparse_tiny_gpu.json
