set -euo pipefail
python3 collect_stage3.py
python3 refresh_proposal_stage3.py
touch STOP
