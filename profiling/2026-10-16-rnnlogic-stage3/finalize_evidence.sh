#!/bin/bash
set -euo pipefail
cd /zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-16-rnnlogic-stage3
python3 merge_fb_screens.py
python3 refresh_proposal_stage3.py
python3 collect_stage3.py
if python3 - <<'PY'
import json
from pathlib import Path
paths=[Path('results/stream_wn_uncached.json'),Path('results/stream_wn_cache8g.json')]
assert all(p.exists() and json.loads(p.read_text())['complete'] for p in paths)
PY
then /home/cw636/venvs/epic/bin/python plot_stream_results.py
fi
