#!/usr/bin/env bash
# Kissat (CDCL SAT) BCP profiling with perf.  Untested on this host.  usage: bash run.sh [small|large|all]
# env: CNF_SMALL / CNF_LARGE  DIMACS files, e.g. SAT Competition instances (https://benchmark-database.de/, put them in
#      $SYMFABRIC_EXT/data/sat/); default = random 3-SAT from gen_cnf.py (n=350 / 450; raise n until a run takes 10-60 s)
#      TIME_LIMIT  seconds for kissat --time (default 600; verify the option name with `kissat --help`)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PROF=$(cd "$HERE/../.." && pwd); S=$PROF/scripts
EXT=${SYMFABRIC_EXT:-/data2/$USER/symfabric_ext}; KISSAT=${KISSAT:-$EXT/kissat/build/kissat}
export SW_VERSION_EXTRA="kissat=$("$KISSAT" --version 2>/dev/null || echo unknown)"
run_one() {  # $1 scale  $2 cnf
  local scale=$1 cnf=$2 W; W=$(python3 "$HERE/gen_cnf.py" --stats "$cnf" 2>/dev/null)
  # -s prints the full statistics block (propagations, conflicts, decisions, ...); -n suppresses the model
  bash "$S/perf_wrap.sh" sat "$scale" -- "$KISSAT" -s -n --time="${TIME_LIMIT:-600}" "$cnf"
  # metric 4: Q = 1, one assignment trail reads the clause database (cube-and-conquer would raise it)
  python3 "$S/parse_perf.py" --trace-dir "$(cat "$PROF/traces/sat/LATEST")" --workload sat --tier 1 --scale "$scale" \
    --instance "$(basename "$cnf")" --Q 1 --W-bytes "$W" \
    --W-method "4 B x literal occurrences (lower bound); upper bound = kissat maximum-resident-set-size in sw_counters" \
    --semiring "Boolean: clause = OR of literals, formula = AND of clauses; BCP = unit propagation on two-watched-literal lists" \
    --hot-regex 'propagat|assign|watch' \
    --control-regex 'analy|decide|restart|reduce|rephase|minimiz|bump|vivif|elimin|probe|walk|backtrack|learn|shrink|subsum' \
    --counter-regex '^c (propagations|conflicts|decisions|restarts|learned|process-time|maximum-resident-set-size):\s+([0-9.]+)' \
    --notes "symbol regexes are first guesses (kissat inlines proplit.h into *_propagate); re-classify from perf_report.txt"
}
gen() { local f=$EXT/data/sat/rand3sat-n$1.cnf; mkdir -p "$EXT/data/sat"; [ -s "$f" ] || python3 "$HERE/gen_cnf.py" --n "$1" --out "$f"; echo "$f"; }
SCALE=${1:-all}
[ "$SCALE" = large ] || run_one small "${CNF_SMALL:-$(gen 350)}"
[ "$SCALE" = small ] || run_one large "${CNF_LARGE:-$(gen 450)}"
