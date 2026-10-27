set -euo pipefail
[[ $(cat done/010_pna_test.status) == 0 ]]
bash run_pinned.sh safe_runtime_launch.py pna_bench.py --threads 16 --build-batch 32 --warmup 2 --repeats 5 --training --output results/pna_official_wn18rr.json
