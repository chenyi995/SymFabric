#!/bin/bash
set -euo pipefail
TASK_ROOT=/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-16-rnnlogic-stage3
cd "$TASK_ROOT"
trap 'echo $? > "$TASK_ROOT/spa_cpu/status"' EXIT
export BENCH_CPU_LIST=$(python3 spa_cpu/map_allocation.py)
export BENCH_PINNED_LINUX_CPUS="$BENCH_CPU_LIST"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$TASK_ROOT:$TASK_ROOT/../2026-10-14-rnnlogic-sparse-stage2:$TASK_ROOT/../2026-10-14-rnnlogic-strong-baselines:$TASK_ROOT/../2026-10-14-candidate-tests/kg/deps"
python3 snapshot_sources.py test_spa_native.py
taskset -c "$BENCH_CPU_LIST" /home/cw636/venvs/epic/bin/python safe_runtime_launch.py test_spa_native.py --output results/spa_tiny_cpu.json
python3 snapshot_sources.py bench_spa_native.py
taskset -c "$BENCH_CPU_LIST" /home/cw636/venvs/epic/bin/python safe_runtime_launch.py bench_spa_native.py --threads 16 --groups 3 --warmup 2 --repeats 5 --max-seconds 780 --output results/spa_native_real.json
