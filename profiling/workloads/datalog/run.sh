#!/usr/bin/env bash
# Datalog (Souffle) profiling with perf: transitive closure and Andersen points-to on generated EDBs.
# Untested on this host.  usage: bash run.sh [small|large|all]
# needs: $SYMFABRIC_EXT/souffle/build/src/souffle (setup_env.sh build), perf.   env: THREADS (Souffle -j, default 8)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PROF=$(cd "$HERE/../.." && pwd); S=$PROF/scripts
EXT=${SYMFABRIC_EXT:-/data2/$USER/symfabric_ext}; SOUFFLE=${SOUFFLE:-$EXT/souffle/build/src/souffle}
THREADS=${THREADS:-8}   # metric 4: Q = worker threads reading the same relation tables (Souffle -j)
export SW_VERSION_EXTRA="souffle=$(git -C "$EXT/souffle" rev-parse --short HEAD 2>/dev/null || echo unknown)"
run_one() {  # $1 scale  $2 program (tc | andersen)  $3.. gen_facts.py arguments
  local scale=$1 prog=$2; shift 2
  local work=$EXT/data/datalog/$prog-$scale; mkdir -p "$work/facts" "$work/out"
  python3 "$HERE/gen_facts.py" --out "$work/facts" "$@" > "$work/gen.log"
  # compile once (souffle -o): the join loops of the compiled program then carry their own symbols in perf
  [ -x "$work/$prog" ] || "$SOUFFLE" -o "$work/$prog" "$HERE/$prog.dl"
  # -p writes Souffle's own per-rule/per-relation profile (inspect with souffleprof; verify with `souffleprof -h`)
  bash "$S/perf_wrap.sh" datalog "$prog-$scale" -- "$work/$prog" -F "$work/facts" -D "$work/out" -j "$THREADS" -p "$work/profile.log"
  # metric 1: W = 8 B x (EDB + IDB tuples), two 32-bit numbers per tuple; B-tree node overhead excluded (lower bound)
  local tuples; tuples=$(cat "$work"/facts/*.facts "$work"/out/*.csv | wc -l)
  python3 "$S/parse_perf.py" --trace-dir "$(cat "$PROF/traces/datalog/LATEST")" --workload datalog --tier 2 \
    --scale "$scale" --instance "$prog" --Q "$THREADS" --W-bytes $((8 * tuples)) \
    --W-method "8 B x (EDB+IDB tuples), lower bound" \
    --semiring "Boolean (OR, AND): join = AND over matching tuples, union/dedup = OR" \
    --hot-regex 'btree|Brie|equalRange|lower_bound|upper_bound|insert|contains|Interval' \
    --control-regex 'SymbolTable|RecordTable|Profile|malloc|free|operator new|ReadStream|WriteStream|parse' \
    --notes "symbol regexes are first guesses: re-classify after reading perf_report.txt; rule times in $work/profile.log"
}
SCALE=${1:-all}
[ "$SCALE" = large ] || { run_one small tc --nodes 1000 --edges 3000; run_one small andersen --vars 2000 --stmts 6000; }
[ "$SCALE" = small ] || { run_one large tc --nodes 5000 --edges 15000; run_one large andersen --vars 50000 --stmts 200000; }
