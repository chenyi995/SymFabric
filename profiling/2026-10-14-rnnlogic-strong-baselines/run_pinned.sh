#!/bin/bash
set -euo pipefail
if [[ -z "${BENCH_CPU_LIST:-}" ]]; then
  if [[ "${SLURM_JOB_ID:-}" != 222516 ]]; then
    echo 'Set BENCH_CPU_LIST to the Linux CPUs belonging to the current Slurm allocation; see AFFINITY_NOTES.txt.' >&2
    exit 2
  fi
  BENCH_CPU_LIST=8-23
fi
export BENCH_PINNED_LINUX_CPUS="$BENCH_CPU_LIST"
export OMP_PLACES=$(/usr/bin/python3 - <<'PY'
import os
cpus=[]
for term in os.environ['BENCH_PINNED_LINUX_CPUS'].split(','):
    ends=list(map(int,term.split('-')))
    cpus.extend(range(ends[0],ends[-1]+1))
print(','.join('{'+str(c)+'}' for c in cpus))
PY
)
export OMP_PROC_BIND=spread
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
exec taskset -c "$BENCH_PINNED_LINUX_CPUS" /home/cw636/venvs/epic/bin/python "$@"
