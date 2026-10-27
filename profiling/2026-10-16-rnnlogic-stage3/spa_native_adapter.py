"""Local dense SPA/touched-list alternative to native64 std::map frontiers.

Same original adjacency, query-parallel independent-rule execution, int64 bound
proof, removed-triple semantics and canonical sparse output. Only frontier count
accumulation changes. Scratch is two N-element int64 buffers plus two touched
lists per OpenMP worker; zero-filled once per batch, touched slots cleared after
each rule layer. No cross-call count/scratch cache or prefix sharing is present.
"""
from pathlib import Path
import ctypes
import fcntl
import hashlib
import os
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
S2=ROOT.parent/'2026-10-14-rnnlogic-sparse-stage2'
UPSTREAM=ROOT.parent/'2026-10-14-candidate-tests/kg/upstream/RNNLogic/miner'
if str(S2) not in sys.path:sys.path.insert(0,str(S2))
from official_native_adapter import OfficialNativeBackend,PTR,ptr,cpu_i64


def load_spa():
    source=ROOT/'spa_native_adapter.cpp'
    originals=[UPSTREAM/'rnnlogic.cpp',UPSTREAM/'rnnlogic.h']
    flags=['-std=gnu++17','-O3','-fPIC','-shared','-fopenmp','-pthread']
    digest=hashlib.sha256(source.read_bytes()+b''.join(path.read_bytes() for path in originals)+repr(flags).encode()).hexdigest()[:16]
    library=ROOT/f'spa_native_{digest}.so'
    with (ROOT/'spa_native_build.lock').open('w') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if not library.exists():
            temporary=library.with_suffix(f'.{os.getpid()}.tmp')
            cmd=['g++',*flags,'-I',str(UPSTREAM),str(source),str(originals[0]),'-o',str(temporary)]
            result=subprocess.run(cmd,text=True,capture_output=True)
            if result.returncode:raise RuntimeError(result.stderr)
            temporary.replace(library)
    lib=ctypes.CDLL(str(library))
    lib.official_graph_create.argtypes=[ctypes.c_int,ctypes.c_int,PTR,PTR,PTR]
    lib.official_graph_create.restype=ctypes.c_void_p
    lib.official_graph_destroy.argtypes=[ctypes.c_void_p]
    lib.official_ground_batch.argtypes=[ctypes.c_void_p,PTR,ctypes.c_int64,PTR,PTR,ctypes.c_int64,PTR,PTR,PTR,ctypes.c_int,ctypes.c_int]
    lib.official_ground_batch.restype=ctypes.c_void_p
    for name in ['offsets','flat','values']:
        method=getattr(lib,'official_result_'+name);method.argtypes=[ctypes.c_void_p];method.restype=PTR
    lib.official_result_nnz.argtypes=[ctypes.c_void_p];lib.official_result_nnz.restype=ctypes.c_int64
    lib.official_result_destroy.argtypes=[ctypes.c_void_p]
    return lib,{'library':str(library),'compiler':'g++','flags':flags,
                'upstream_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}}


class SPANativeBackend(OfficialNativeBackend):
    """Reuse unchanged native64 prepare/batch validation, conversion and ABI."""
    def __init__(self,graph,threads=1,bits=64):
        start=time.perf_counter()
        self.entity_size=int(graph.entity_size);self.relation_size=len(graph.relation2adjacency)
        self.threads=int(threads);self.bits=int(bits)
        if self.threads<1 or self.bits!=64:raise ValueError('SPA control requires positive threads and bits=64')
        if self.entity_size>=2**31 or self.relation_size>=2**31:raise ValueError('upstream ID width exceeded')
        self.lib,info=load_spa()
        self.adjacency=[];self.max_indegree=[]
        sources=[];destinations=[];offsets=[0]
        for index,_ in graph.relation2adjacency:
            index=cpu_i64(index)
            if index.ndim!=2 or index.shape[0]!=2:raise ValueError('adjacency shape')
            dst,src=index
            if index.size and (index.min()<0 or index.max()>=self.entity_size):raise ValueError('entity range')
            encoded=src*self.entity_size+dst
            if len(np.unique(encoded))!=len(src):raise ValueError('duplicate triples unsupported by official triple-deletion baseline')
            sources.append(src);destinations.append(dst);self.adjacency.append((src,dst))
            offsets.append(offsets[-1]+len(src))
            self.max_indegree.append(int(np.bincount(dst,minlength=self.entity_size).max(initial=0)))
        self._graph_arrays=[np.asarray(offsets,np.int64),np.concatenate(sources),np.concatenate(destinations)]
        self.handle=self.lib.official_graph_create(self.entity_size,self.relation_size,*[ptr(array) for array in self._graph_arrays])
        if not self.handle:raise RuntimeError('SPA graph construction failed')
        self.metadata={'backend':'local_native64_dense_spa_touched','threads':self.threads,
            'graph_nodes':self.entity_size,'graph_edges':offsets[-1],'graph_relations':self.relation_size,
            'algorithm':'independent rules; dense two-buffer SPA count accumulation, touched-only clear, sorted terminal ids',
            'original_method_used':False,'local_changes':['std::map frontiers replaced by two dense int64 accumulators and touched lists','terminal touched ids sorted once','empty rule identity'],
            'query_parallelism':'OpenMP independent query batches, static scheduling',
            'graph_layout':'original vector[N][R] neighbor lists',
            'duplicate_policy':'reject, matching Python loader','inverse_policy':'no synthesized edges',
            'count_safety':'unchanged OfficialNativeBackend.prepare_rules conservative intermediate int64 bound',
            'scratch_count_buffer_bytes_per_thread':self.entity_size*16,
            'scratch_touched_capacity_bytes_per_thread':self.entity_size*8,
            'scratch_scope':'allocated/zero initialized inside each timed batch; touched cleared per layer; no persistent scratch',
            'setup_seconds':time.perf_counter()-start,**info}
