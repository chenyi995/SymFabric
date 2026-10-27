set -euo pipefail
bash run_pinned.sh safe_runtime_launch.py 220_cache_capacity.py --threads 16 --batch 32 --budget-mib 768 --output cache_capacity_selected_relations.json
