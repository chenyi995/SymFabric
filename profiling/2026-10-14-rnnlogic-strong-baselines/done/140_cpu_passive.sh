export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=0
bash run_pinned.sh run_bench.py --device cpu --cases FB15k-237:272 wn18rr:12 --threads 16 --batches 32 --splits train valid --variants upstream torch_all native_all native_last native_lru32 --repeats 3 --warmup 1 --output cpu_passive.json
