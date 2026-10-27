"""Inspect both OpenMP libraries and execute a native team in each runtime.

Run in the Slurm worker, normally via safe_runtime_launch.py. A native helper
calls each loaded library's own GOMP_parallel symbol (not a linker's chosen
runtime) and records every team member's real CPU, affinity, and CPU time.
It establishes team placement/availability, not application scaling.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))

def libraries():
    return sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if '/' in line and any(x in line for x in ['libgomp','libomp','libiomp','libopenblas','libmkl'])})

def snap(label):
    entries=[]
    for p in Path('/proc/self/task').iterdir():
        try: entries.append({'tid':int(p.name),'affinity':sorted(os.sched_getaffinity(int(p.name)))})
        except ProcessLookupError: pass
    return {'label':label,'main_affinity':sorted(os.sched_getaffinity(0)),
            'threads':entries,'loaded_libraries':libraries()}

p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=16);p.add_argument('--output',required=True);args=p.parse_args()
report={'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'pid':os.getpid(),
        'purpose':'native OpenMP runtime team placement, not model timing',
        'threads_requested':args.threads,
        'env':{k:os.getenv(k) for k in ['BENCH_CPU_LIST','BENCH_RUNTIME_BINDING','OMP_PROC_BIND','OMP_PLACES','OMP_WAIT_POLICY','GOMP_SPINCOUNT','OMP_NUM_THREADS']},
        'snapshots':[snap('before_torch')]}
import torch
report['snapshots'].append(snap('after_torch_import'))
torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
a=torch.ones(512,512);assert int((a+a).sum())==2*512*512
report['snapshots'].append(snap('after_torch_work'))
import graphblas as gb
gb.ss.config['nthreads']=args.threads
import numpy as np
rng=np.random.default_rng(42)
r=rng.integers(0,1024,8192);c=rng.integers(0,1024,8192)
a=gb.Matrix.from_coo(r,c,np.ones(8192,np.int64),nrows=1024,ncols=1024,dup_op=gb.binary.plus)
b=a.mxm(a,gb.semiring.plus_times).new();b.wait()
report['snapshots'].append(snap('after_graphblas_work'))
report['graphblas']={'python':gb.__version__,'c_library':list(gb.ss.about['library_version']),'thread_limit':gb.ss.config['nthreads']}
source=ROOT/'runtime_team_probe.c'
digest=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
so=ROOT/f'runtime_team_probe_{digest}.so'
if not so.exists(): subprocess.run(['gcc','-O2','-std=c11','-fPIC','-shared',str(source),'-o',str(so)],check=True)
class Record(ctypes.Structure):
    _fields_=[('index',ctypes.c_int),('tid',ctypes.c_int),('team_size',ctypes.c_int),('cpu_start',ctypes.c_int),('cpu_end',ctypes.c_int),('affinity_count',ctypes.c_int),('affinity_cpus',ctypes.c_int*1024),('cpu_ns',ctypes.c_int64),('wall_start_ns',ctypes.c_int64),('wall_end_ns',ctypes.c_int64),('checksum',ctypes.c_uint64)]
helper=ctypes.CDLL(str(so));helper.probe_runtime.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint,ctypes.POINTER(Record),ctypes.c_int];helper.probe_runtime.restype=ctypes.c_int
report['runtime_teams']=[]
for path in libraries():
    if 'libgomp' not in path:continue
    lib=ctypes.CDLL(path,mode=os.RTLD_NOLOAD|os.RTLD_LOCAL)
    entry={'path':path,'symbols':{}}
    functions={}
    for name in ['GOMP_parallel','omp_get_thread_num','omp_get_num_threads','omp_get_num_procs','omp_get_max_threads','omp_get_dynamic','omp_get_num_places']:
        fn=getattr(lib,name);fn.restype=ctypes.c_int
        functions[name]=fn
        entry['symbols'][name]={'address':ctypes.cast(fn,ctypes.c_void_p).value}
        if name not in ['GOMP_parallel','omp_get_thread_num','omp_get_num_threads']:
            entry['symbols'][name]['value']=fn()
    records=(Record*args.threads)()
    start=time.perf_counter()
    n=helper.probe_runtime(*[ctypes.cast(functions[name],ctypes.c_void_p) for name in ['GOMP_parallel','omp_get_thread_num','omp_get_num_threads']],args.threads,records,args.threads)
    entry['wall_ms']=(time.perf_counter()-start)*1000
    entry['team_count']=n
    entry['records']=[{'index':r.index,'tid':r.tid,'team_size':r.team_size,'cpu_start':r.cpu_start,'cpu_end':r.cpu_end,
                       'affinity':list(r.affinity_cpus[:r.affinity_count]),'cpu_ms':r.cpu_ns/1e6, 'work_wall_start_ns':r.wall_start_ns,'work_wall_end_ns':r.wall_end_ns} for r in records[:n]]
    entry['unique_start_cpus']=sorted({r.cpu_start for r in records[:n]})
    entry['unique_end_cpus']=sorted({r.cpu_end for r in records[:n]})
    entry['sum_thread_cpu_ms']=sum(r.cpu_ns for r in records[:n])/1e6
    entry['work_wall_span_ms']=(max(r.wall_end_ns for r in records[:n])-min(r.wall_start_ns for r in records[:n]))/1e6
    entry['average_simultaneous_cpu_equivalent']=entry['sum_thread_cpu_ms']/entry['work_wall_span_ms']
    report['runtime_teams'].append(entry)
report['snapshots'].append(snap('after_native_runtime_teams'))
allowed={int(x) for x in os.environ['BENCH_CPU_LIST'].split(',')}
report['all_teams_requested_size']=all(x['team_count']==args.threads for x in report['runtime_teams'])
report['all_team_affinities_within_verified_pool']=all(set(r['affinity']).issubset(allowed) for x in report['runtime_teams'] for r in x['records'])
report['all_team_affinities_full_verified_pool']=all(set(r['affinity'])==allowed for x in report['runtime_teams'] for r in x['records'])
report['distinct_start_cpus_per_runtime']=[len(x['unique_start_cpus']) for x in report['runtime_teams']]
Path(args.output).write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ['host','job','threads_requested','all_teams_requested_size','all_team_affinities_within_verified_pool','all_team_affinities_full_verified_pool','distinct_start_cpus_per_runtime']},indent=2))
