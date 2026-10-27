set -euo pipefail
[[ $(cat done/020_pna_real.status) == 0 ]]
[[ $(cat done/026_gpu_cache_real.status) == 0 ]]
bash run_pinned.sh safe_runtime_launch.py allrelation_screen.py --dataset wn18rr --pna-tile 2048 --native-kind spa --output results/allrelation_wn18rr_tile2048.json
