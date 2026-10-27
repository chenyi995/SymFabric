set -euo pipefail
[[ $(cat done/010_pna_test.status) == 0 ]]
[[ $(cat done/015_cache_test.status) == 0 ]]
[[ $(cat done/025_gpu_cache_test.status) == 0 ]]
python3 -c "import json; a=json.load(open('results/spa_native_real.json')); b=json.load(open('results/spa_tiny_cpu.json')); assert a['complete'] and len(a['cases'])==30 and b['complete'] and b['passed']"
while [[ ! -f query_stream_wn18rr.npz ]]; do sleep 2; done
bash run_pinned.sh safe_runtime_launch.py allrelation_screen.py --native-kind spa --dataset wn18rr --output results/screen_wn.json
