"""Exact sparse-frontier control using existing PyTorch CUDA operations.

This is a local software adapter, not a native cuSPARSE int64 SpGEMM claim.
Variable-size repeat/unique/filter operations can synchronize; timed calls must
include them. No device counts are cached across forwards.
"""
from dataclasses import dataclass
import time
import torch

@dataclass
class SparseState:
    keys: torch.Tensor  # entity * batch + query; sorted and unique
    values: torch.Tensor  # positive int64, upstream overflow limits apply
    batch: int

class SparseGPUBackend:
    def __init__(self,graph,device='cuda'):
        self.graph=graph;self.n=graph.entity_size;self.device=torch.device(device)
        if self.device.index is None and self.device.type=='cuda':self.device=torch.device('cuda',torch.cuda.current_device())
        self.rows=[]
        start=time.perf_counter()
        for adj,_ in graph.relation2adjacency:
            dst,src=adj.cpu().long();order=torch.argsort(src,stable=True)
            degree=torch.bincount(src,minlength=self.n)
            ptr=torch.cat((torch.zeros(1,dtype=torch.long),degree.cumsum(0)))
            self.rows.append((ptr.to(self.device),dst[order].to(self.device),order.to(self.device)))
        if self.device.type=='cuda':torch.cuda.synchronize()
        self.metadata={'implementation':'PyTorch sparse frontier: source CSR expansion, per-query original-edge exclusion, sort/unique and int64 index_add',
            'torch':torch.__version__,'setup_seconds':time.perf_counter()-start,'resident_bytes':sum(t.numel()*t.element_size() for row in self.rows for t in row),'arithmetic':'int64','dynamic_shape_syncs_included':True}
    @torch.no_grad()
    def initial(self,h):
        b=len(h);keys=h*b+torch.arange(b,device=h.device)
        keys=keys.sort().values
        return SparseState(keys,torch.ones_like(keys),b)
    @torch.no_grad()
    def propagate(self,x,relation,removed=None):
        if x.keys.device!=self.device or x.values.dtype!=torch.int64:raise TypeError('device/int64 mismatch')
        if not x.keys.numel():return x
        ptr,dst,original_id=self.rows[relation]
        entity=x.keys//x.batch;query=x.keys%x.batch
        start=ptr[entity];degree=ptr[entity+1]-start
        parent=torch.repeat_interleave(torch.arange(x.keys.numel(),device=self.device),degree)
        # Both repeated parent construction and unique allocate dynamic sizes.
        if not parent.numel():return SparseState(x.keys[:0],x.values[:0],x.batch)
        offsets=degree.cumsum(0)-degree
        edge=start[parent]+torch.arange(parent.numel(),device=self.device)-offsets[parent]
        q=query[parent];values=x.values[parent]
        if removed is not None:
            keep=original_id[edge]!=removed[q]
            edge=edge[keep];q=q[keep];values=values[keep]
        keys=dst[edge]*x.batch+q
        unique,inverse=torch.unique(keys,sorted=True,return_inverse=True)
        out=torch.zeros(unique.numel(),dtype=torch.long,device=self.device)
        out.index_add_(0,inverse,values)
        return SparseState(unique,out,x.batch)
    def to_sparse(self,state):
        return (state.keys%state.batch)*self.n+state.keys//state.batch,state.values
    def to_dense(self,state):
        idx,val=self.to_sparse(state)
        out=torch.zeros(state.batch*self.n,dtype=torch.long,device=self.device)
        out[idx]=val
        return out.view(state.batch,self.n)

class SparseGPUPrefixGrounder:
    def __init__(self,backend,rule_bodies,return_format='dense',collect_stats=False):
        if return_format not in ['dense','sparse']:raise ValueError('return_format')
        self.backend=backend;self.bodies=[tuple(b) for b in rule_bodies];self.return_format=return_format;self.collect_stats=collect_stats
        self.last={};self.release=[[] for _ in self.bodies]
        for i,b in enumerate(self.bodies):
            for j in range(len(b)+1):self.last[b[:j]]=i
        for pre,i in self.last.items():self.release[i].append(pre)
        self.active=False
    def begin(self,h,head,removed):
        if self.active:raise RuntimeError('active forward')
        self.h=h;self.head=int(head);self.removed=removed;self.i=0;self.active=True
        self.cache={():self.backend.initial(h)}
        self.statistics={'propagations':0,'max_cache_bytes':0,'max_state_nnz':0,'terminal_nnz':0,'representation':'int64 sparse keys/values; excludes dynamic expansion temporaries'}
    def grounding(self,h,head,body,removed):
        if not self.active or h is not self.h or removed is not self.removed or int(head)!=self.head:raise ValueError('forward context changed')
        body=tuple(body)
        if self.i>=len(self.bodies) or body!=self.bodies[self.i]:raise ValueError('rule order changed')
        j=len(body)
        while body[:j] not in self.cache:j-=1
        state=self.cache[body[:j]]
        for k in range(j,len(body)):
            state=self.backend.propagate(state,body[k],removed if body[k]==head else None)
            prefix=body[:k+1]
            if self.last[prefix]>self.i:self.cache[prefix]=state
            if self.collect_stats:
                self.statistics['propagations']+=1
                self.statistics['max_state_nnz']=max(self.statistics['max_state_nnz'],state.keys.numel())
                self.statistics['max_cache_bytes']=max(self.statistics['max_cache_bytes'],sum((v.keys.numel()+v.values.numel())*8 for v in self.cache.values()))
        if self.collect_stats:self.statistics['terminal_nnz']+=state.keys.numel()
        for prefix in self.release[self.i]:self.cache.pop(prefix,None)
        self.i+=1
        return self.backend.to_dense(state) if self.return_format=='dense' else self.backend.to_sparse(state)
    def finish(self):
        complete=self.i==len(self.bodies)
        self.cache.clear();self.active=False
        return dict(self.statistics,complete=complete,rule_calls=self.i)
