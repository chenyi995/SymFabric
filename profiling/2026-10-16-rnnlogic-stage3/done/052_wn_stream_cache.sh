set -euo pipefail
[[ $(cat done/050_wn_stream_pilot.status) == 0 ]]
bash run_pinned.sh safe_runtime_launch.py stream_replay.py --dataset wn18rr --screen results/allrelation_wn18rr_tile2048.json --output results/stream_wn_cache8g.json --mode cache --epochs 2 --pna-tile 2048 --budget-gib 8 --check-every 503 --max-seconds 720
