set -euo pipefail
bash run_pinned.sh safe_runtime_launch.py bench_cached_sparse.py --threads 16 --grad --training --output cached_sparse_real.json
