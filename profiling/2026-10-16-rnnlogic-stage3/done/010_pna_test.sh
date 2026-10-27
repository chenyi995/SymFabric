set -euo pipefail
bash run_pinned.sh test_pna_predictor.py --device cpu --output results/pna_tiny_cpu.json
bash run_pinned.sh test_pna_predictor.py --device cuda --output results/pna_tiny_gpu.json
