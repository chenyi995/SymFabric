set -euo pipefail
[[ $(cat done/020_pna_real.status) == 0 ]]
[[ $(cat done/026_gpu_cache_real.status) == 0 ]]
bash run_pinned.sh safe_runtime_launch.py pna_profile.py --output results/pna_profile.json
