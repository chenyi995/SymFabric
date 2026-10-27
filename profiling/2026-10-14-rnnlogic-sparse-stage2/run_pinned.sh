#!/bin/bash
set -euo pipefail
: "${BENCH_CPU_LIST:?worker must provide verified allocation CPUs}"
export BENCH_PINNED_LINUX_CPUS="$BENCH_CPU_LIST"
export OMP_PLACES=$(python3 - <<'PY'
import os
print(','.join('{'+x+'}' for x in os.environ['BENCH_CPU_LIST'].split(',')))
PY
)
export OMP_PROC_BIND=spread OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1
python3 snapshot_sources.py "$1"
exec taskset -c "$BENCH_CPU_LIST" /home/cw636/venvs/epic/bin/python "$@"
