#!/bin/bash
set -euo pipefail
/home/cw636/venvs/epic/bin/python gpu_backend.py --self-test --output gpu_correctness_runtime_edges.json
