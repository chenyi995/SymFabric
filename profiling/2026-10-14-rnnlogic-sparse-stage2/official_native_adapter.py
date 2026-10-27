"""Official RNNLogic miner std::map count algorithm, adapted to PredictorPlus.

native32 calls the unchanged upstream method after a conservative overflow proof.
native64 changes count width and fixes empty-rule identity only. OpenMP batches
independent queries. No prefix sharing or cross-forward count cache is present.
"""
from pathlib import Path
import ctypes
import fcntl
import hashlib
import os
import subprocess
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
UPSTREAM=ROOT.parent/'2026-10-14-candidate-tests/kg/upstream/RNNLogic/miner'
PTR=ctypes.POINTER(ctypes.c_int64)
def ptr(a): return a.ctypes.data_as(PTR)
def cpu_i64(x):
    if isinstance(x,torch.Tensor):
        if x.device.type!='cpu' or x.dtype!=torch.int64: raise TypeError('CPU int64 query required')
        x=x.detach().numpy()
    x=np.asarray(x)
    if x.dtype!=np.int64: raise TypeError('int64 input required')
    return np.ascontiguousarray(x)

def load_native():
    source=ROOT/'official_native_adapter.cpp'
    originals=[UPSTREAM/'rnnlogic.cpp',UPSTREAM/'rnnlogic.h']
    flags=['-std=gnu++17','-O3','-fPIC','-shared','-fopenmp','-pthread']
    digest=hashlib.sha256(source.read_bytes()+b''.join(p.read_bytes() for p in originals)+repr(flags).encode()).hexdigest()[:16]
    path=ROOT/f'official_native_{digest}.so'
    with (ROOT/'official_native_build.lock').open('w') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if not path.exists():
            tmp=path.with_suffix(f'.{os.getpid()}.tmp')
            cmd=['g++',*flags,'-I',str(UPSTREAM),str(source),str(originals[0]),'-o',str(tmp)]
            result=subprocess.run(cmd,text=True,capture_output=True)
            if result.returncode: raise RuntimeError(result.stderr)
            tmp.replace(path)
    lib=ctypes.CDLL(str(path))
    lib.official_graph_create.argtypes=[ctypes.c_int,ctypes.c_int,PTR,PTR,PTR]
    lib.official_graph_create.restype=ctypes.c_void_p
    lib.official_graph_destroy.argtypes=[ctypes.c_void_p]
    lib.official_ground_batch.argtypes=[ctypes.c_void_p,PTR,ctypes.c_int64,PTR,PTR,ctypes.c_int64,PTR,PTR,PTR,ctypes.c_int,ctypes.c_int]
    lib.official_ground_batch.restype=ctypes.c_void_p
    for name in ['offsets','flat','values']:
        fn=getattr(lib,'official_result_'+name);fn.argtypes=[ctypes.c_void_p];fn.restype=PTR
    lib.official_result_nnz.argtypes=[ctypes.c_void_p];lib.official_result_nnz.restype=ctypes.c_int64
    lib.official_result_destroy.argtypes=[ctypes.c_void_p]
    return lib,dict(library=str(path),compiler='g++',flags=flags,upstream_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in originals})

class OfficialNativeBackend:
    def __init__(self,graph,threads=1,bits=64):
        start=time.perf_counter()
        self.entity_size=int(graph.entity_size)
        self.relation_size=len(graph.relation2adjacency)
        self.threads=int(threads);self.bits=int(bits)
        if self.threads<1 or self.bits not in (32,64): raise ValueError('invalid threads or width')
        if self.entity_size>=2**31 or self.relation_size>=2**31: raise ValueError('upstream ID width exceeded')
        self.lib,info=load_native()
        self.adjacency=[];self.max_indegree=[]
        srcs=[];dsts=[];offsets=[0]
        for index,_ in graph.relation2adjacency:
            index=cpu_i64(index)
            if index.ndim!=2 or index.shape[0]!=2: raise ValueError('adjacency shape')
            dst,src=index
            if index.size and (index.min()<0 or index.max()>=self.entity_size): raise ValueError('entity range')
            # Original miner removes matching triples, not one duplicate edge.
            # PredictorPlus loader rejects duplicate triples, so reject them here too.
            encoded=src*self.entity_size+dst
            if len(np.unique(encoded))!=len(src): raise ValueError('duplicate triples unsupported by official triple-deletion baseline')
            srcs.append(src);dsts.append(dst);self.adjacency.append((src,dst))
            offsets.append(offsets[-1]+len(src))
            self.max_indegree.append(int(np.bincount(dst,minlength=self.entity_size).max(initial=0)))
        self._graph_arrays=[np.asarray(offsets,np.int64),np.concatenate(srcs),np.concatenate(dsts)]
        self.handle=self.lib.official_graph_create(self.entity_size,self.relation_size,*[ptr(a) for a in self._graph_arrays])
        if not self.handle: raise RuntimeError('official graph construction failed')
        self.metadata=dict(backend=f'official_miner_std_map_native{self.bits}',threads=self.threads,
            graph_nodes=self.entity_size,graph_edges=offsets[-1],graph_relations=self.relation_size,
            algorithm='upstream KnowledgeGraph::rule_destination independent rules; no prefix sharing',
            original_method_used=self.bits==32,local_changes=[] if self.bits==32 else ['int64 count widening','std::vector replaces VLA','empty rule returns identity'],
            query_parallelism='OpenMP independent query batches',graph_layout='original vector[N][R] neighbor lists',
            duplicate_policy='reject, matching Python loader',inverse_policy='no synthesized edges, same relation IDs and adjacency as Python',
            setup_seconds=time.perf_counter()-start,**info)
    def close(self):
        if getattr(self,'handle',None): self.lib.official_graph_destroy(self.handle);self.handle=None
    def __del__(self): self.close()
    def prepare_rules(self,bodies):
        bodies=[tuple(int(x) for x in b) for b in bodies]
        bounds=[]
        for b in bodies:
            if self.bits==32 and not b: raise ValueError('unmodified official native32 has different empty-rule semantics')
            bound=1
            for r in b:
                if r<0 or r>=self.relation_size: raise ValueError('relation ID range')
                bound*=self.max_indegree[r]
                if bound>2**(self.bits-1)-1: raise OverflowError('cannot prove all-query intermediate count safety')
            bounds.append(bound)
        offsets=np.asarray([0]+list(np.cumsum([len(b) for b in bodies])),np.int64)
        body=np.asarray([r for b in bodies for r in b],np.int64)
        return bodies,offsets,body,bounds
    def batch(self,prepared,h,relation,removed=None):
        bodies,offsets,body,bounds=prepared
        h=cpu_i64(h);relation=int(relation)
        if h.ndim!=1 or (h.size and (h.min()<0 or h.max()>=self.entity_size)): raise ValueError('head shape/range')
        if relation<0 or relation>=self.relation_size: raise ValueError('query relation range')
        n=len(h)
        if removed is None:
            rh=rr=rt=np.full(n,-1,np.int64)
        else:
            removed=cpu_i64(removed)
            src,dst=self.adjacency[relation]
            if removed.shape!=(n,) or (removed.size and (removed.min()<0 or removed.max()>=len(src))): raise ValueError('removed edge shape/range')
            rh=np.ascontiguousarray(src[removed]);rt=np.ascontiguousarray(dst[removed]);rr=np.full(n,relation,np.int64)
        result=self.lib.official_ground_batch(self.handle,ptr(h),n,ptr(offsets),ptr(body),len(bodies),ptr(rh),ptr(rr),ptr(rt),self.bits,self.threads)
        if not result: raise RuntimeError('official native grounding failed')
        try:
            nnz=self.lib.official_result_nnz(result)
            boundaries=np.ctypeslib.as_array(self.lib.official_result_offsets(result),shape=(len(bodies)+1,)).copy()
            flat=np.ctypeslib.as_array(self.lib.official_result_flat(result),shape=(nnz,)).copy() if nnz else np.empty(0,np.int64)
            values=np.ctypeslib.as_array(self.lib.official_result_values(result),shape=(nnz,)).copy() if nnz else np.empty(0,np.int64)
        finally:self.lib.official_result_destroy(result)
        flat=torch.from_numpy(flat);values=torch.from_numpy(values)
        return [(flat[boundaries[k]:boundaries[k+1]],values[boundaries[k]:boundaries[k+1]]) for k in range(len(bodies))]

class OfficialNativeController:
    def __init__(self,backend,bodies):
        self.backend=backend;self.prepared=backend.prepare_rules(bodies)
        self.outputs=None;self.cursor=0;self.stats=None
    def begin(self,h,relation,removed):
        if self.outputs is not None: raise RuntimeError('previous forward not finished')
        self.outputs=self.backend.batch(self.prepared,h,relation,removed);self.cursor=0
        self.stats={'backend':self.backend.metadata['backend'],'prefix_sharing':False,
            'rules':len(self.outputs),'terminal_nnz':sum(x.numel() for x,_ in self.outputs),
            'all_terminal_bytes_logical':sum((x.numel()+v.numel())*8 for x,v in self.outputs),
            'all_rules_computed_in_timed_begin':True,'cross_forward_cache':False}
    def grounding(self,h,relation,body,removed):
        if self.outputs is None or self.cursor>=len(self.outputs): raise RuntimeError('controller order error')
        if tuple(body)!=self.prepared[0][self.cursor]: raise ValueError('rule order differs from plan')
        out=self.outputs[self.cursor];self.cursor+=1
        return out
    def finish(self, require_complete=True):
        if require_complete and self.outputs is not None and self.cursor!=len(self.outputs): raise RuntimeError('incomplete rule consumption')
        self.outputs=None
        return self.stats
    def abort(self):self.outputs=None
