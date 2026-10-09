#!/usr/bin/env bash
# Run a command under Nsight Systems and export summary CSVs.  Untested on this host (no GPU).
# usage: nsys_wrap.sh <workload> <label> -- <command...>
# env:   NSYS_EXTRA  extra `nsys profile` flags, e.g. "--python-sampling=true" (Python backtraces) or
#                    "--gpu-metrics-devices=cuda-visible" (DRAM-bandwidth timeline; needs GPU performance-counter
#                    permission; flag name verified against nsys 2025.1.3 `nsys profile --help`)
# outputs: profile.nsys-rep, profile.sqlite, stats_<report>.csv, stdout.log, stderr.log, versions.txt, cmd.txt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_wrap_common.sh"
wrap_init nsys "$@"
wrap_time_start
# --stats=true prints the summary tables to stdout.log and creates profile.sqlite
nsys profile --trace=cuda,nvtx,osrt --stats=true --force-overwrite=true -o "$DIR/profile" ${NSYS_EXTRA:-} \
  "${CMD[@]}" > "$DIR/stdout.log" 2> "$DIR/stderr.log" || echo "[wrap] command exit status $?" >> "$DIR/stderr.log"
wrap_time_end
# One CSV per report, named <basename>_<report>.csv by nsys (checked in `nsys stats --help`): stats_cuda_gpu_kern_sum.csv ...
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_pushpop_sum,cuda_api_sum \
  --format csv --output "$DIR/stats" --force-overwrite true "$DIR/profile.nsys-rep" > "$DIR/stats_console.txt" 2>&1
echo "$DIR"
