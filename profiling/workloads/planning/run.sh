#!/usr/bin/env bash
# Fast Downward planning profiling with perf: h^add (greedy) and h^max (A*) on IPC domains.  Untested on this host.
# usage: bash run.sh [small|large|all]    needs $SYMFABRIC_EXT/downward (built) and downward-benchmarks
# Problem file names below follow the downward-benchmarks layout (gripper/prob01.pddl, logistics00/probLOGISTICS-4-0.pddl);
# confirm with ls before running.  env: FD_TIME (driver --overall-time-limit, default 30m)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PROF=$(cd "$HERE/../.." && pwd); S=$PROF/scripts
EXT=${SYMFABRIC_EXT:-/data2/$USER/symfabric_ext}; FD=$EXT/downward/fast-downward.py; B=$EXT/downward-benchmarks
export SW_VERSION_EXTRA="downward=$(git -C "$EXT/downward" rev-parse --short HEAD 2>/dev/null || echo unknown)"
run_one() {  # $1 scale  $2 domain dir  $3 problem file  $4 heuristic label  $5 search string
  local scale=$1 dom=$2 prob=$3 h=$4 search=$5 semiring
  case $h in hadd) semiring="(min, +) relaxed fixpoint: h(fact) = min over achievers of (cost + sum of precondition h)" ;;
             hmax) semiring="(min, max) relaxed fixpoint: h(fact) = min over achievers of (cost + max of precondition h)" ;; esac
  # perf record follows the driver's child processes, so the C++ search binary ('downward' dso) is sampled too
  bash "$S/perf_wrap.sh" planning "$dom-$h-$scale" -- python3 "$FD" --overall-time-limit "${FD_TIME:-30m}" \
    "$B/$dom/domain.pddl" "$B/$dom/$prob" --search "$search"
  local td; td=$(cat "$PROF/traces/planning/LATEST")
  # metric 1: state registry size = registered states x bytes per state (both printed by the search component);
  # successor generator and the heuristic's relaxed-task tables are not included (lower bound)
  local W; W=$(awk '/^Bytes per state:/{b=$4} /^Number of registered states:/{n=$5} END{if(b&&n) print b*n; else print "TBD[src]"}' "$td/stdout.log")
  python3 "$S/parse_perf.py" --trace-dir "$td" --workload planning --tier 2 --scale "$scale" --instance "$dom/$prob $h" \
    --Q 1 --W-bytes "$W" --W-method "registered states x bytes per state (state registry only, lower bound)" --semiring "$semiring" \
    --hot-regex 'Heuristic|Relaxation|relaxed|Proposition|UnaryOperator|compute_heuristic|hmax|hadd|Additive|priority_queue|AdaptiveQueue' \
    --control-regex 'Search|OpenList|StateRegistry|SuccessorGenerator|IntPacker|Operator|axiom|State::|search' \
    --counter-regex '^(Expanded|Evaluated|Generated) ([0-9]+) state|^(Search time|Total time): ([0-9.]+)s' \
    --notes "symbol regexes are first guesses; re-classify from perf_report.txt (dso=downward rows)"
}
SCALE=${1:-all}
if [ "$SCALE" != large ]; then
  run_one small gripper prob04.pddl hadd "eager_greedy([add()])"
  run_one small gripper prob04.pddl hmax "astar(hmax())"
fi
if [ "$SCALE" != small ]; then
  run_one large logistics00 probLOGISTICS-9-0.pddl hadd "eager_greedy([add()])"
  run_one large logistics00 probLOGISTICS-7-0.pddl hmax "astar(hmax())"
fi
