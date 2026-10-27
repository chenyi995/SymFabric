#!/bin/bash
set -uo pipefail
TASK_ROOT=/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-16-rnnlogic-stage3
cd "$TASK_ROOT"
mkdir -p queue running done logs triton-cache
export BENCH_CPU_LIST=$(python3 map_allocation.py)
[[ -n "$BENCH_CPU_LIST" ]] || exit 3
export PYTHONUNBUFFERED=1
export PYTHONPATH="/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-sparse-stage2:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-sparse-stage2/deps:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-strong-baselines:$TASK_ROOT:""$TASK_ROOT/deps:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-candidate-tests/kg/deps"
export TRITON_CACHE_DIR="$TASK_ROOT/triton-cache"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=spread
nvidia-smi -q > gpu_metadata.txt
lscpu > lscpu.txt
while [[ ! -f STOP ]]; do
  for task_file in "$TASK_ROOT"/queue/*.sh; do
    [[ -f "$task_file" ]] || continue
    task_name=$(basename "$task_file" .sh)
    mv "$task_file" "$TASK_ROOT/running/$task_name.sh"
    bash "$TASK_ROOT/running/$task_name.sh" > "$TASK_ROOT/logs/$task_name.log" 2>&1
    task_status=$?
    echo "$task_status" > "$TASK_ROOT/done/$task_name.status"
    mv "$TASK_ROOT/running/$task_name.sh" "$TASK_ROOT/done/$task_name.sh"
  done
  sleep 2
done
