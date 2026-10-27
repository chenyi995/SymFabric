bash run_pinned.sh safe_runtime_launch.py bench_memoized.py --device cuda --threads 16 --cases FB15k-237:272 wn18rr:12 --build-batch 32 --block-size 512 --grad --output memoized_hybrid16.json
