set -euo pipefail
bash run_pinned.sh safe_runtime_launch.py 221_cache_capacity_digest.py --cases FB15k-237:272 --threads 16 --batch 32 --budget-mib 4096 --output cache_capacity_digest_selected_relations.json
