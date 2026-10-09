# Sourced by nsys_wrap.sh, ncu_wrap.sh and perf_wrap.sh (not executable on its own).
# Creates profiling/traces/<workload>/<timestamp>-<label>/, records hardware and software versions
# (versions.txt), the exact command (cmd.txt) and the wall time of the profiled run (wall_s= in versions.txt).
# Wrapper usage:  <wrapper>.sh <workload> <label> -- <command> [args...]
# Optional env:   SW_VERSION_EXTRA="kissat=<git rev>"  (copied into versions.txt as sw_extra=)
wrap_init() {
  TOOL=$1; WORKLOAD=$2; LABEL=$3; shift 3; [ "${1:-}" = "--" ] && shift
  CMD=("$@"); [ ${#CMD[@]} -gt 0 ] || { echo "usage: $TOOL_wrap.sh <workload> <label> -- <command...>" >&2; exit 2; }
  PROF=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  DIR=$PROF/traces/$WORKLOAD/$(date +%Y%m%d-%H%M%S)-$LABEL; mkdir -p "$DIR"
  printf '%q ' "${CMD[@]}" > "$DIR/cmd.txt"
  {
    echo "tool=$TOOL"; echo "date=$(date -Iseconds)"; echo "host=$(hostname)"
    echo "gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | paste -sd'|' || echo unknown)"
    echo "cpu=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
    echo "nsys=$(nsys --version 2>/dev/null | tail -1)"; echo "ncu=$(ncu --version 2>/dev/null | tail -1)"
    echo "perf=$(perf --version 2>/dev/null)"; echo "python=$(python3 --version 2>&1)"
    echo "torch=$(python3 -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null || echo none)"
    echo "sw_extra=${SW_VERSION_EXTRA:-}"
  } > "$DIR/versions.txt"
  echo "$DIR" > "$PROF/traces/$WORKLOAD/LATEST"   # consumed by the workload run.sh to locate this run
  echo "[wrap:$TOOL] trace dir $DIR" >&2
}
wrap_time_start() { T0=$(date +%s%N); }
wrap_time_end() { echo "wall_s=$(awk "BEGIN{printf \"%.3f\", ($(date +%s%N) - $T0)/1e9}")" >> "$DIR/versions.txt"; }
