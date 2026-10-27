set -euo pipefail
[[ $(cat done/020_pna_real.status) == 0 ]]
[[ $(cat done/046_pna_profile.status) == 0 ]]
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-pna-compile"
timeout --signal=INT --kill-after=10s 240s bash run_pinned.sh safe_runtime_launch.py pna_compile_bench.py --output results/pna_compile.json
