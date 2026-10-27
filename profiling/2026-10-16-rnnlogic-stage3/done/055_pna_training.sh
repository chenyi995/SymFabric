set -euo pipefail
[[ $(cat done/020_pna_real.status) == 0 ]]
timeout --signal=INT --kill-after=10s 120s bash run_pinned.sh safe_runtime_launch.py pna_training_control.py --output results/pna_training_control.json
