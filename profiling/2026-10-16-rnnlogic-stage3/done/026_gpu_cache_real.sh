set -euo pipefail
[[ $(cat done/025_gpu_cache_test.status) == 0 ]]
bash run_pinned.sh safe_runtime_launch.py gpu_cache_bench.py --output results/gpu_cache_real.json
