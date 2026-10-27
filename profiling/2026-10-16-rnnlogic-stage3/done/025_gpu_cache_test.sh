set -euo pipefail
[[ $(cat done/010_pna_test.status) == 0 ]]
[[ $(cat done/015_cache_test.status) == 0 ]]
bash run_pinned.sh test_gpu_cache_builder.py --output results/gpu_cache_tiny.json
