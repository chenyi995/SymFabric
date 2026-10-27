#!/bin/bash
set -uo pipefail
TASK_ROOT=/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-16-rnnlogic-stage3
cd "$TASK_ROOT"
export BENCH_CPU_LIST=$(python3 cpu_support/map_allocation.py)
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$TASK_ROOT:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-sparse-stage2:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-sparse-stage2/deps:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-strong-baselines:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-candidate-tests/kg/deps"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs done
if [[ -n "$BENCH_CPU_LIST" ]]; then
  bash cpu_support/005_trace.sh > logs/005_trace.log 2>&1
  task_status=$?
else
  task_status=3
fi
echo "$task_status" > done/005_trace.status
mv cpu_support/005_trace.sh done/005_trace.sh
exit "$task_status"
