"""Launch a benchmark with two OpenMP runtimes inside one verified CPU pool.

Must run before importing torch/GraphBLAS/NumPy. Unlike binding to individual
OpenMP places, process-pool binding cannot truncate a later runtime's place set
when the first runtime narrows the initial thread to a single place.
"""
import os
from pathlib import Path
import runpy
import sys

if not sys.argv[1:]:
    raise SystemExit('usage: safe_runtime_launch.py benchmark.py [arguments]')
cpus = os.environ.get('BENCH_CPU_LIST')
if not cpus:
    raise SystemExit('BENCH_CPU_LIST must be supplied by verified Slurm mapping')
allowed = {int(cpu) for cpu in cpus.split(',')}
if not allowed:
    raise SystemExit('empty benchmark CPU pool')
# This launcher runs under taskset from run_pinned.sh. Do not expand that pool.
if not allowed.issubset(os.sched_getaffinity(0)):
    raise SystemExit('verified benchmark CPU pool is outside launcher affinity')
os.environ['OMP_PROC_BIND'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['GOMP_SPINCOUNT'] = '0'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
for key in ['OMP_PLACES', 'GOMP_CPU_AFFINITY']:
    os.environ.pop(key, None)
os.sched_setaffinity(0, allowed)
os.environ['BENCH_RUNTIME_BINDING'] = 'pool_only_passive_before_all_runtime_imports'
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name='__main__')
