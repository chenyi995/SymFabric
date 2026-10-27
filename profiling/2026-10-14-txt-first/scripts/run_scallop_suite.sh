#!/bin/bash
set -euo pipefail
cd /home/cw636/chenyi/SymFabric
python3 profiling/2026-10-14-txt-first/scripts/run_scallop_programs.py --binary /home/cw636/chenyi/SymFabric/profiling/2026-10-14-txt-first/bin/c1-e2e-original --label original
python3 profiling/2026-10-14-txt-first/scripts/run_scallop_programs.py --binary /home/cw636/chenyi/SymFabric/profiling/2026-10-14-txt-first/bin/c1-e2e-profile --label cached --cached-score
python3 profiling/2026-10-14-txt-first/scripts/run_scallop_programs.py --binary /home/cw636/chenyi/SymFabric/profiling/2026-10-14-txt-first/bin/c1-e2e-profile --label diagnostic --profile --reps 2
