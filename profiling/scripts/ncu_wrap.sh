#!/usr/bin/env bash
# Run a command under Nsight Compute and store the per-kernel metrics as CSV.  Untested on this host (no ncu here).
# usage: ncu_wrap.sh <workload> <label> -- <command...>
# env:   NCU_METRICS       override the metric list below
#        NCU_KERNEL_REGEX  profile only kernels whose name matches (ncu --kernel-name regex:<re>)
#        NCU_LAUNCH_SKIP   kernels skipped before profiling starts (default 20 = warm-up); NCU_LAUNCH_COUNT profiled
#                          launches (default 200; every profiled kernel is replayed several times, so keep this bounded)
#        NCU_NVTX_INCLUDE  restrict to an NVTX range, e.g. "hmm_forward/" (adds --nvtx --nvtx-include)
#        NCU_EXTRA         any other ncu flag
# outputs: ncu.csv (long format: one row per launch x metric), stdout.log, stderr.log, versions.txt, cmd.txt
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_wrap_common.sh"
wrap_init ncu "$@"
# First seven metrics are the project spec.  gpu__time_duration.sum weights the per-kernel averages in parse_ncu.py;
# the two l1tex sectors/requests metrics give metric 5 (sectors per request).  Metric names could not be checked on
# this host: verify every name with `ncu --query-metrics` and drop the ones that are rejected.
METRICS=${NCU_METRICS:-dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,smsp__inst_executed.sum,gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum}
wrap_time_start
# --print-units base keeps byte/second values unscaled (verify the flag with `ncu --help`; parse_ncu.py also
# understands the Kbyte/Mbyte/usecond prefixes ncu prints without it).  --csv output goes to --log-file.
ncu --target-processes all --csv --print-units base --log-file "$DIR/ncu.csv" --metrics "$METRICS" \
  --launch-skip "${NCU_LAUNCH_SKIP:-20}" --launch-count "${NCU_LAUNCH_COUNT:-200}" \
  ${NCU_KERNEL_REGEX:+--kernel-name "regex:$NCU_KERNEL_REGEX"} \
  ${NCU_NVTX_INCLUDE:+--nvtx --nvtx-include "$NCU_NVTX_INCLUDE"} ${NCU_EXTRA:-} \
  "${CMD[@]}" > "$DIR/stdout.log" 2> "$DIR/stderr.log" || echo "[wrap] command exit status $?" >> "$DIR/stderr.log"
wrap_time_end
echo "$DIR"
