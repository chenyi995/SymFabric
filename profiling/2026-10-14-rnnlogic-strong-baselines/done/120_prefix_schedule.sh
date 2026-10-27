set -e
bash run_pinned.sh profile_phases.py --cases FB15k-237:272 wn18rr:12 --variants torch_all atomic_last csr_last --output phase_profile.json --trace-dir traces
bash run_pinned.sh run_bench.py --device cuda --cases FB15k-237:272 wn18rr:12 --threads 1 --batches 32 --splits train valid --rule-order prefix --variants upstream csr_last csr_lastlru32 csr_lastlru128 --repeats 5 --warmup 2 --gradient-check --output gpu_prefix_schedule.json
