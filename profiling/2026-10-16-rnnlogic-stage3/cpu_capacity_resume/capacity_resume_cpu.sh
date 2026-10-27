#!/bin/bash
set -uo pipefail
TASK_ROOT=/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-16-rnnlogic-stage3
cd "$TASK_ROOT"
export BENCH_CPU_LIST=$(python3 cpu_capacity_resume/map_allocation.py)
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$TASK_ROOT:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-sparse-stage2:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-sparse-stage2/deps:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-strong-baselines:/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-candidate-tests/kg/deps"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs done
if [[ -n "$BENCH_CPU_LIST" ]]; then
  CENSUS_BACKEND=$(python3 - <<'CHECK'
import json
from pathlib import Path
root=Path('results')
try:
 real=json.loads((root/'spa_native_real.json').read_text())
 tiny=json.loads((root/'spa_tiny_cpu.json').read_text())
 passed=real.get('complete') is True and len(real.get('cases',[]))==30 and tiny.get('complete') is True and tiny.get('passed') is True
except (OSError,ValueError):
 passed=False
print('spa' if passed else 'stdmap')
CHECK
)
  bash run_pinned.sh safe_runtime_launch.py native_capacity_census.py --backend "$CENSUS_BACKEND" --threads 16 --datasets FB15k-237 --resume --max-dataset-seconds 7200 --max-total-seconds 7500 > logs/007_capacity_resume.log 2>&1
  task_status=$?
else
  task_status=3
fi
echo "$task_status" > done/007_capacity_resume.status
exit "$task_status"
