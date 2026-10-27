"""Explicit preloaded CSR cache: mixed/repeated queries, gradients and reset gates."""
import argparse
import json
from pathlib import Path
import time
import torch
from test_streaming_predictor import TinyGraph, SparseInterface, evaluate, compare
from predictors import PredictorPlus
from cached_sparse_predictor import CachedSparsePredictor


def terminals_for(adapter, graph, h, r, removed):
    relation=int(r[0]);ctrl=SparseInterface(graph,[body for _,(_,body) in adapter.rules_for_relation(relation)])
    ctrl.begin(h,relation,removed)
    result=[ctrl.grounding(h,head,body,removed) for _,(head,body) in adapter.rules_for_relation(relation)]
    ctrl.finish();return result


def expect_error(error, callback):
    try:callback()
    except error:return
    raise AssertionError(f'expected {error.__name__}')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output');args=parser.parse_args()
    device=torch.device(args.device)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    graph=TinyGraph()
    patterns=[[],[0],[1],[3],[0,0],[0,1],[1,0],[2,0],[0,1,0],[2,2,0],[3,0],[1,2]]
    rules=[[0]+patterns[(i*7)%len(patterns)] for i in range(137)]+[[1,1]]
    groups=[(torch.tensor([0,1,2,256]),torch.tensor([0,1,2,3])),(torch.tensor([3,4,0]),torch.tensor([4,5,0]))]
    queries=[([4,1,0,4,256,2],[5,1,0,5,3,2]),([256,3,0],[3,4,0])]
    rows=[];preloads=[];start=time.perf_counter()
    for kind in ('emb','lstm'):
        for feature in ('bias','none'):
            for order in ('original','prefix'):
                torch.manual_seed(1943)
                model=PredictorPlus(graph,type=kind,num_layers=2,hidden_dim=8,entity_feature=feature,aggregator='sum')
                model.set_rules(rules);model.to(device)
                adapter=CachedSparsePredictor(model,SparseInterface,namespace='tiny-graph-v1',rule_version='rules-v1',rule_order=order,collect_stats=True)
                for split in ('train','valid'):
                    for group,(h,removed) in enumerate(groups):
                        r=torch.zeros_like(h);rem=removed if split=='train' else None
                        if group==0:report=adapter.preload(h,r,rem)
                        else:
                            terminals=terminals_for(adapter,graph,h,r,rem)
                            report=adapter.preload_from_terminals(h,r,rem,terminals,namespace='tiny-graph-v1',rule_version='rules-v1')
                        preloads.append(report)
                        assert report['new_keys']==(4 if group==0 else 2)
                assert len(adapter.cache)==12
                for weight_version in (0,1):
                    if weight_version:
                        # Weights may change without invalidating discrete counts.
                        with torch.no_grad():model.score_model.layers[-1].bias.add_(.125)
                    for split in ('train','valid'):
                        for group,(heads,removed) in enumerate(queries):
                            h=torch.tensor(heads,device=device);r=torch.zeros_like(h)
                            rem=torch.tensor(removed,device=device) if split=='train' else None
                            expected=evaluate(model,model,h,r,rem)
                            errors=compare(evaluate(model,adapter,h,r,rem),expected)
                            assert not adapter.last_stats['dense_full_entity_feature_pool_created']
                            rows.append({'kind':kind,'feature':feature,'rule_order':order,'split':split,'query_group':group,
                                'weight_version':weight_version,'max_logit_error':errors[0],'max_gradient_error':errors[1],
                                'stats':adapter.last_stats})
    # Empty outcomes: zero-count rules and no rules for the requested relation.
    empty_checks=0
    for feature in ('bias','none'):
        model=PredictorPlus(graph,type='emb',hidden_dim=8,entity_feature=feature,aggregator='sum')
        model.set_rules([[0,3],[1,0]]);model.to(device)
        adapter=CachedSparsePredictor(model,SparseInterface,namespace='empty',rule_version='v1',collect_stats=True)
        for relation in (0,2):
            h_cpu=torch.tensor([0,1,256]);r_cpu=torch.full_like(h_cpu,relation)
            adapter.preload(h_cpu,r_cpu,None)
            h=h_cpu.to(device);r=r_cpu.to(device)
            with torch.no_grad():out=adapter(h,r,None);expected=model(h,r,None)
            assert torch.equal(out[0],expected[0]) and torch.equal(out[1],expected[1])
            assert adapter.last_stats['csr_rows']==0
            if feature=='bias':compare(evaluate(model,adapter,h,r,None),evaluate(model,model,h,r,None))
            empty_checks+=1
    # Key semantics and strict invalidation/error gates on a compact rule list.
    model=PredictorPlus(graph,type='emb',hidden_dim=8,entity_feature='bias',aggregator='sum')
    model.set_rules([[0,0],[0,0],[0,1],[1,1]]);model.to(device)
    adapter=CachedSparsePredictor(model,SparseInterface,namespace='keys',rule_version='v1',collect_stats=True)
    h=torch.tensor([0,0]);r=torch.zeros_like(h);removed=torch.tensor([0,0])
    terminals=terminals_for(adapter,graph,h,r,removed)
    report=adapter.preload_from_terminals(h,r,removed,terminals)
    assert report['new_keys']==1 and report['unique_input_keys']==1
    altered=[(i.clone(),v.clone()) for i,v in terminals]
    hit=torch.nonzero(altered[0][0]>=graph.entity_size,as_tuple=True)[0][0]
    altered[0][1][hit]+=1
    expect_error(ValueError,lambda:adapter.preload_from_terminals(h,r,removed,altered))
    duplicated=[(i.clone(),v.clone()) for i,v in terminals]
    duplicated[0]=(torch.cat([duplicated[0][0],duplicated[0][0][:1]]),torch.cat([duplicated[0][1],duplicated[0][1][:1]]))
    expect_error(ValueError,lambda:adapter.preload_from_terminals(h,r,removed,duplicated))
    zero=[(i.clone(),v.clone()) for i,v in terminals];zero[0][1][0]=0
    expect_error(ValueError,lambda:adapter.preload_from_terminals(h,r,removed,zero))
    expect_error(ValueError,lambda:adapter.preload_from_terminals(h,r,removed,terminals,namespace='other'))
    expect_error(ValueError,lambda:adapter.preload_from_terminals(h,r,removed,terminals,rule_version='v2'))
    expect_error(KeyError,lambda:adapter(torch.tensor([1],device=device),torch.tensor([0],device=device),torch.tensor([0],device=device)))
    # Same h/r but removed=None is a distinct key and must not hit a training key.
    expect_error(KeyError,lambda:adapter(h.to(device),r.to(device),None))
    adapter.reset(namespace='keys',rule_version='v2');assert not adapter.cache
    expect_error(KeyError,lambda:adapter(h.to(device),r.to(device),removed.to(device)))
    adapter.preload(h,r,removed)
    compare(evaluate(model,adapter,h.to(device),r.to(device),removed.to(device)),evaluate(model,model,h.to(device),r.to(device),removed.to(device)))
    # Actual autograd saved CSR operand, not a dense count matrix.
    saved=[]
    def pack(tensor):
        row={'shape':list(tensor.shape),'layout':str(tensor.layout),'dtype':str(tensor.dtype)}
        if tensor.layout==torch.sparse_csr:row['nnz']=tensor._nnz()
        saved.append(row);return tensor
    model.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        out,mask=adapter(h.to(device),r.to(device),removed.to(device));out.square().mean().backward()
    assert any(row['layout']=='torch.sparse_csr' for row in saved)
    result={'passed':True,'device':str(device),'torch':torch.__version__,'cases':rows,'preloads':preloads,
        'empty_cases':empty_checks,'key_and_validation_gates_passed':True,'autograd_saved_csr_verified':True,
        'saved_tensor_metadata':saved,'seconds':time.perf_counter()-start,
        'scope':'tiny consumer/cache correctness; source exact counts derive from original dense graph oracle, not a native runtime claim'}
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'device':str(device),'forward_gradient_cases':len(rows),'empty_cases':empty_checks,
        'max_logit_error':max(x['max_logit_error'] for x in rows),'max_gradient_error':max(x['max_gradient_error'] for x in rows),
        'seconds':result['seconds']}))


if __name__=='__main__':main()
