"""Allocated-GPU cache builder correctness and storage accounting checks."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parent
S2=ROOT.parent/'2026-10-14-rnnlogic-sparse-stage2'
S1=ROOT.parent/'2026-10-14-rnnlogic-strong-baselines'
sys.path[:0]=[str(S2),str(S1)]
import torch
from test_streaming_predictor import TinyGraph,evaluate,compare
from predictors import PredictorPlus
from gpu_backend import GPUBackend
from cached_sparse_faststats_predictor import CachedSparseFastStatsPredictor
from pna_predictor import CachedPNAPredictor
from gpu_cache_builder import install_gpu_preloader,preload_gpu_new,gpu_csr_terminals,GPUCSRCountBuilder,GPU_DIGEST_SENTINEL,terminals_to_gpu
from cache_lru import AllocationBlockLRU


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('GPU test requires allocated worker')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',torch.cuda.current_device());start=time.perf_counter()
    graph=TinyGraph();backend=GPUBackend(graph,device=device,strategy='csr')
    h=torch.tensor([0,0,1,2,256]);r=torch.zeros_like(h);removed=torch.tensor([0,8,1,2,3])
    cases=[]
    for aggregator in ['sum','pna']:
        torch.manual_seed(56)
        model=PredictorPlus(graph,type='emb',hidden_dim=8,aggregator=aggregator,entity_feature='bias')
        model.set_rules([[0],[0,0],[1,1],[0,0],[0,0,1],[0,1],[0,3]])
        model.to(device)
        cls=CachedPNAPredictor if aggregator=='pna' else CachedSparseFastStatsPredictor
        cache=install_gpu_preloader(cls(model,namespace='gpu-tiny',rule_version='v1'))
        cpu_cache=cls(model,namespace='cpu-tiny',rule_version='v1')
        plan=cache.rules_for_relation(0)
        builder=GPUCSRCountBuilder(graph,backend,plan)
        terminals,source_stats=builder.batch(h.to(device),0,removed.to(device))
        repeated,_=builder.batch(h.to(device),0,removed.to(device))
        for (i,v),(ii,vv) in zip(terminals,repeated):
            assert torch.equal(i,ii) and torch.equal(v,vv)
        for (_,(_,body)),(index,counts) in zip(plan,terminals):
            reference=graph.grounding(h.to(device),0,body,removed.to(device))
            dense=torch.zeros(reference.numel(),dtype=torch.int64,device=device);dense[index]=counts
            torch.testing.assert_close(dense.reshape_as(reference),reference,rtol=0,atol=0)
        cpu_terminals=[(index.cpu(),count.cpu()) for index,count in terminals]
        transferred=terminals_to_gpu(cpu_terminals,device)
        for (i,v),(ii,vv) in zip(terminals,transferred):
            assert torch.equal(i,ii) and torch.equal(v,vv)
        assert len({i.untyped_storage().data_ptr() for i,_ in transferred})==1
        assert len({v.untyped_storage().data_ptr() for _,v in transferred})==1
        cpu_cache.preload_from_terminals(h,r,removed,cpu_terminals)
        manager=AllocationBlockLRU(cache,2**20)
        context=manager.lookup(h,r,removed)
        admitted,report=manager.admit(context,h,r,removed,terminals)
        assert admitted
        invariant=manager.verify_invariants()
        assert cache.cache_statistics()['persistent_unique_storage_bytes']==cpu_cache.cache_statistics()['persistent_unique_storage_bytes']
        for key,entry in cache.cache.items():
            cpu_key=(cpu_cache.namespace,*key[1:])
            other=cpu_cache.cache[cpu_key]
            for field in ['candidates','crow','columns','values']:
                torch.testing.assert_close(getattr(entry,field),getattr(other,field),rtol=0,atol=0)
            assert entry.integer_digest==GPU_DIGEST_SENTINEL
        trusted=install_gpu_preloader(cls(model,namespace='gpu-trusted',rule_version='v1'),trusted_unique_positive_terminals=True)
        bad_metadata=[(i,v.float()) for i,v in terminals]
        try:
            trusted.preload_from_terminals(h,r,removed,bad_metadata)
        except ValueError:
            pass
        else:
            raise AssertionError('trusted mode accepted float counts')
        trusted_report=trusted.preload_from_terminals(h,r,removed,terminals)
        assert trusted_report['trusted_unique_positive_terminals'] is True
        assert report['preload']['trusted_unique_positive_terminals'] is False
        for key,entry in cache.cache.items():
            trusted_key=(trusted.namespace,*key[1:])
            other=trusted.cache[trusted_key]
            for field in ['candidates','crow','columns','values']:
                torch.testing.assert_close(getattr(entry,field),getattr(other,field),rtol=0,atol=0)
        # Reordering/repetition of already admitted queries exercises normal
        # warm cache reuse; the narrower preload contract does not restrict it.
        selection=torch.tensor([4,2,0,1,0,3]);hh=h[selection].to(device);rr=r[selection].to(device);ee=removed[selection].to(device)
        error=compare(evaluate(model,cache,hh,rr,ee),evaluate(model,model,hh,rr,ee))
        cases.append({'aggregator':aggregator,'exact_rule_terminal_checks':len(terminals),'forward_error':error[0],
                      'gradient_error':error[1],'admission':report,'invariants':invariant,'source_stats':source_stats,
                      'bulk_h2d_exact_two_shared_storages':True,'trusted_mode_bit_exact_parity':True,'trusted_mode_rejects_invalid_dtype':True})
        # New-only/key and exact sparse-coordinate constraints fail visibly.
        errors=[]
        def reject(label,fn):
            try:fn()
            except (ValueError,TypeError):errors.append(label)
            else:raise AssertionError(label+' accepted')
        reject('existing_keys',lambda:preload_gpu_new(cache,h,r,removed,terminals))
        cache.reset(namespace='gates',rule_version='v2');manager.clear()
        reject('duplicate_query_keys',lambda:preload_gpu_new(cache,h[:2],r[:2],removed[:1].expand(2),terminals))
        duplicate=[(i.clone(),v.clone()) for i,v in terminals]
        duplicate[0]=(torch.cat([duplicate[0][0],duplicate[0][0][:1]]),torch.cat([duplicate[0][1],duplicate[0][1][:1]]))
        reject('duplicate_terminal_coordinate',lambda:preload_gpu_new(cache,h,r,removed,duplicate))
        nonpositive=[(i.clone(),v.clone()) for i,v in terminals];nonpositive[0][1][0]=0
        reject('nonpositive_count',lambda:preload_gpu_new(cache,h,r,removed,nonpositive))
        assert not cache.cache
        cases[-1]['rejected_invalid_inputs']=errors
        # Exact int64 values above float32 resolution convert at the same
        # boundary as CPU preload, while no stronger digest claim is made.
        high=[(torch.empty(0,dtype=torch.int64,device=device),torch.empty(0,dtype=torch.int64,device=device)) for _ in plan]
        high[0]=(torch.tensor([1],device=device),torch.tensor([2**24+1],device=device))
        cache.preload_from_terminals(h[:1],r[:1],removed[:1],high)
        entry=next(iter(cache.cache.values()));assert entry.values.tolist()==[float(torch.tensor(2**24+1).float())]
        for relation in [0,2]:
            cache.reset(namespace='empty'+str(relation),rule_version='v3');manager.clear()
            empty_plan=cache.rules_for_relation(relation)
            empty=[(torch.empty(0,dtype=torch.int64,device=device),torch.empty(0,dtype=torch.int64,device=device)) for _ in empty_plan]
            empty_r=torch.full_like(r,relation)
            context=manager.lookup(h,empty_r,removed)
            ok,empty_report=manager.admit(context,h,empty_r,removed,empty)
            assert ok and empty_report['actual_new_storage_bytes']==h.numel()*8
            manager.verify_invariants()
    result={'passed':True,'complete':True,'device':str(device),'gpu':torch.cuda.get_device_name(),'cases':cases,
            'seconds':time.perf_counter()-start,'scope':'GPU exact int64 source+new-key builder; cache output/gradients, packed storage, manager accounting and validation; no existing-key digest support'}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({'passed':True,'cases':len(cases),'seconds':result['seconds']}))


if __name__=='__main__':main()
