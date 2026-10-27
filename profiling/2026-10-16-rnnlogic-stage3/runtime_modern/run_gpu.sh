#!/bin/bash
set -uo pipefail
TASK_ROOT=/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-16-rnnlogic-stage3
MODERN_ROOT="$TASK_ROOT/runtime_modern"
cd "$TASK_ROOT"
export BENCH_METADATA_DIR="$MODERN_ROOT/gpu_runs/$SLURM_JOB_ID"
mkdir -p "$BENCH_METADATA_DIR"
export BENCH_CPU_LIST=$(python3 "$TASK_ROOT/map_allocation.py")
[[ -n "$BENCH_CPU_LIST" ]] || exit 3
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONPATH
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=FALSE OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
unset OMP_PLACES GOMP_CPU_AFFINITY
export TORCHINDUCTOR_CACHE_DIR="$MODERN_ROOT/inductor-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$MODERN_ROOT/triton-$SLURM_JOB_ID"
export TORCHINDUCTOR_COMPILE_THREADS=4
nvidia-smi -q > "$BENCH_METADATA_DIR/gpu_metadata.txt"
lscpu > "$BENCH_METADATA_DIR/lscpu.txt"
: "${PNA_FIXTURE_DIR:?fixture directory required}"
RUNNER="$MODERN_ROOT/standalone_layer_bench.py"
sha256sum "$RUNNER" > "$BENCH_METADATA_DIR/runner.sha256"
taskset -c "$BENCH_CPU_LIST" /home/cw636/venvs/epic/bin/python "$RUNNER" --fixtures "$PNA_FIXTURE_DIR" --output "$MODERN_ROOT/legacy_eager.json" --reference-dir "$MODERN_ROOT/legacy_reference"
legacy_status=$?
echo "$legacy_status" > "$MODERN_ROOT/legacy_eager.status"
taskset -c "$BENCH_CPU_LIST" "$MODERN_ROOT/venv/bin/python" "$RUNNER" --fixtures "$PNA_FIXTURE_DIR" --output "$MODERN_ROOT/modern_compile.json" --reference-dir "$MODERN_ROOT/modern_reference" --legacy-reference-dir "$MODERN_ROOT/legacy_reference" --compile
modern_status=$?
echo "$modern_status" > "$MODERN_ROOT/modern_compile.status"
exit "$modern_status"
