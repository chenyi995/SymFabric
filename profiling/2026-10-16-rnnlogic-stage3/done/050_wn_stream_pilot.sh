set -euo pipefail
python3 -c "import json; assert json.load(open('results/allrelation_wn18rr_tile2048.json'))['complete']"
bash run_pinned.sh safe_runtime_launch.py stream_replay.py --dataset wn18rr --screen results/allrelation_wn18rr_tile2048.json --output results/stream_wn_pilot.json --mode cache --epochs 2 --pna-tile 2048 --budget-gib 8 --max-batches 16 --check-every 5 --max-seconds 180
