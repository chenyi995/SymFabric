#!/bin/bash
set -euo pipefail
TASK_ROOT=/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-strong-baselines
cd "$TASK_ROOT"
bash run_pinned.sh affinity_probe.py --threads 16 --output runtime_probe_default.json
OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 bash run_pinned.sh affinity_probe.py --threads 16 --output runtime_probe_passive.json
