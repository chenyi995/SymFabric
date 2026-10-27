"""Small correctness checks only; real-graph timing belongs on a Slurm worker."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sparse_cpu_backend import SparseCPUBackend, SparsePrefixGrounder


def dense_step(graph, x, relation, removed):
    dst, src = graph.relation2adjacency[relation][0]
    message = x[src].clone()
    if removed is not None:
        message[removed, torch.arange(x.shape[1])] = 0
    y = torch.zeros_like(x)
    y.index_add_(0, dst, message)
    return y


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output'); args=p.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20261014)
    count = 0
    metadata = []
    for library in ['scipy', 'graphblas']:
        for seed in range(8):
            n, batch = 13, seed + 1
            a = torch.randint(n, (2, 39), generator=generator)
            # Deliberate parallel edges, self loop, and isolated vertices.
            a[:, :4] = torch.tensor([[2, 2, 2, 3], [1, 1, 1, 3]])
            b = torch.randint(n, (2, 17), generator=generator)
            graph = SimpleNamespace(entity_size=n, relation2adjacency=[(a,None),(b,None),(torch.empty((2,0),dtype=torch.int64),None)])
            backend=SparseCPUBackend(graph,library=library,threads=1)
            h=torch.randint(n,(batch,),generator=generator)
            removed=torch.randint(a.shape[1],(batch,),generator=generator)
            h[0]=1;removed[0]=0
            x=backend.initial(h)
            ref=torch.nn.functional.one_hot(h,n).T
            for step, relation in enumerate([0,0,1,0,1,0,2]):
                rem=removed if relation==0 else None
                x=backend.propagate(x,relation,rem)
                ref=dense_step(graph,ref,relation,rem)
                torch.testing.assert_close(backend.to_dense(x),ref.T,rtol=0,atol=0)
                idx,val=backend.to_sparse(x)
                assert val.dtype==torch.int64 and idx.dtype==torch.int64
                assert len(idx)==len(torch.unique(idx)) and torch.all(val != 0)
                dense=torch.zeros(batch*n,dtype=torch.int64);dense[idx]=val
                torch.testing.assert_close(dense.reshape(batch,n),ref.T,rtol=0,atol=0)
                count+=2
            bodies=[(),(0,),(0,0),(0,1),(0,1,0),(0,1,0),(1,0),(2,)]
            for fmt in ['dense','sparse']:
                for policy, budget in [('all',None),('last_use',None),('lru',128),('last_use_lru',128)]:
                    ctrl=SparsePrefixGrounder(graph,bodies,backend,return_format=fmt,policy=policy,budget_bytes=budget)
                    ctrl.begin(h,0,removed)
                    for body in bodies:
                        ref=torch.nn.functional.one_hot(h,n).T
                        for relation in body:
                            ref=dense_step(graph,ref,relation,removed if relation==0 else None)
                        got=ctrl.grounding(h,0,body,removed)
                        if fmt=='sparse':
                            idx,val=got;got=torch.zeros(batch*n,dtype=torch.int64);got[idx]=val;got=got.reshape(batch,n)
                        torch.testing.assert_close(got,ref.T,rtol=0,atol=0)
                        count+=1
                    assert ctrl.finish()['complete']
            if seed==0: metadata.append(backend.metadata)
        # No-query case and empty state.
        h=torch.empty(0,dtype=torch.int64)
        x=backend.propagate(backend.initial(h),0,None)
        assert backend.to_dense(x).shape==(0,n)
        count+=1
    result={'passed':True,'exact_checks':count,'libraries':metadata,
            'covers':['duplicate graph edges','one original edge deletion per query','repeated head relation','zero/empty states','empty rule','duplicate rules','same-batch prefix reuse','bounded-cache recomputation','dense and sparse terminal adapters']}
    print(json.dumps(result,indent=2))
    if args.output: Path(args.output).write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
