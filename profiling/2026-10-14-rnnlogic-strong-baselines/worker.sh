#!/bin/bash
set -uo pipefail
TASK_ROOT=/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-strong-baselines
cd "$TASK_ROOT"
mkdir -p queue running done logs triton-cache
export PYTHONUNBUFFERED=1
export PYTHONPATH=/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-candidate-tests/kg/deps
export TRITON_CACHE_DIR="$TASK_ROOT/triton-cache"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
/home/cw636/venvs/epic/bin/python -c 'import json,os,socket,torch;print(json.dumps({"host":socket.gethostname(),"job":os.getenv("SLURM_JOB_ID"),"affinity":sorted(os.sched_getaffinity(0)),"torch":torch.__version__,"gpu":torch.cuda.get_device_name()},indent=2))' > allocation_metadata.json
lscpu > lscpu.txt
nvidia-smi -q > gpu_metadata.txt
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
