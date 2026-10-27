set -euo pipefail
bash run_pinned.sh test_cached_sparse_faststats.py --device cuda --output cached_sparse_faststats_tiny_gpu.json
bash run_pinned.sh safe_runtime_launch.py 222_cache_capacity_faststats.py --threads 16 --batch 32 --budget-mib 4096 --output cache_capacity_faststats.json
