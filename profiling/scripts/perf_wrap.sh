#!/usr/bin/env bash
# Run a command under Linux perf: counters (perf stat), sampled call graph (perf record/report) and, optionally,
# memory-access samples (perf mem).  Flags verified against perf 4.18 `--help` on this host; the runs themselves
# are untested here.  The command is executed once per tool (2-3 runs); wall time is taken from the perf record run.
# usage: perf_wrap.sh <workload> <label> -- <command...>
# env:   PERF_EVENTS     counter list (default below; events a CPU lacks show as <not supported> in perf_stat.txt and
#                        become TBD[src]; check `perf list`).  On kernel.perf_event_paranoid > 1 hosts append :u to
#                        each event or ask the administrator to lower it.
#        PERF_FREQ       sampling frequency (default 999 Hz);  PERF_CALLGRAPH=dwarf for binaries without frame pointers
#        PERF_MEM=1      add a `perf mem record` run (Intel PEBS / AMD IBS required; see `perf mem --help`)
# outputs: perf_stat.txt (perf stat -x ';'), perf.data, perf_report.txt, [perf_mem.data, perf_mem.txt], stdout*.log
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_wrap_common.sh"
wrap_init perf "$@"
EVENTS=${PERF_EVENTS:-cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses}
perf stat -x ';' -o "$DIR/perf_stat.txt" -e "$EVENTS" -- "${CMD[@]}" > "$DIR/stdout_stat.log" 2> "$DIR/stderr_stat.log" \
  || echo "[wrap] perf stat run exit status $?" >> "$DIR/stderr_stat.log"
CG=(-g); [ "${PERF_CALLGRAPH:-fp}" = dwarf ] && CG=(--call-graph dwarf)
wrap_time_start
perf record "${CG[@]}" -F "${PERF_FREQ:-999}" -o "$DIR/perf.data" -- "${CMD[@]}" > "$DIR/stdout.log" 2> "$DIR/stderr.log" \
  || echo "[wrap] perf record run exit status $?" >> "$DIR/stderr.log"
wrap_time_end
# flat self-time profile by shared object and symbol; parse_perf.py reads this file
perf report -i "$DIR/perf.data" --stdio --no-children --sort dso,sym --percent-limit 0.1 > "$DIR/perf_report.txt" 2>> "$DIR/stderr.log" || true
if [ "${PERF_MEM:-0}" = 1 ]; then
  perf mem record -o "$DIR/perf_mem.data" -- "${CMD[@]}" > "$DIR/stdout_mem.log" 2>&1 || true
  # memory-level breakdown of sampled loads (L1 / L2 / L3 / DRAM hits) = coarse reuse-distance classes;
  # sort key 'mem' taken from `perf report --sort` docs, verify with `perf mem report --help`
  perf mem report -i "$DIR/perf_mem.data" --stdio --sort mem > "$DIR/perf_mem.txt" 2>&1 || true
fi
echo "$DIR"
