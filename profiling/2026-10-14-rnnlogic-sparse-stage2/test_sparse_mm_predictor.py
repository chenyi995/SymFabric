"""Sparse float-MM consumption: upstream outputs, gradients and empty cases."""
import argparse
import json
from pathlib import Path
import time
import torch
from test_streaming_predictor import TinyGraph, SparseInterface, evaluate, compare
from predictors import PredictorPlus
from sparse_mm_predictor import SparseMMPredictor
from bench_stage2 import CPUInputController


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output');parser.add_argument('--compact-device',choices=['source','model','both'],default='both');args=parser.parse_args()
    device=torch.device(args.device)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    graph=TinyGraph()
    patterns=[[],[0],[1],[3],[0,0],[0,1],[1,0],[2,0],[0,1,0],[2,2,0],[3,0],[1,2]]
    rules=[[0]+patterns[(i*7)%len(patterns)] for i in range(137)]+[[1,1]]
    h=torch.tensor([0,1,2,256],device=device);r=torch.zeros_like(h);removed=torch.tensor([0,1,2,3],device=device)
    sources=['device'] if device.type=='cpu' else ['device','cpu']
    placements=['source','model'] if args.compact_device=='both' else [args.compact_device]
    def make_factory(source):
        if source=='cpu':return lambda g,bodies:CPUInputController(SparseInterface(g,bodies))
        return SparseInterface
    rows=[];start=time.perf_counter()
    for kind in ('emb','lstm'):
        for feature in ('bias','none'):
            torch.manual_seed(1943)
            model=PredictorPlus(graph,type=kind,num_layers=2,hidden_dim=8,entity_feature=feature,aggregator='sum')
            model.set_rules(rules);model.to(device)
            for split,rem in [('train',removed),('valid',None)]:
                expected=evaluate(model,model,h,r,rem)
                for order in ('original','prefix'):
                    for block in (8,32,128):
                        for source in sources:
                            for placement in placements:
                                adapter=SparseMMPredictor(model,block_size=block,rule_order=order,
                                    grounder_factory=make_factory(source),collect_stats=True,compact_device=placement)
                                errors=compare(evaluate(model,adapter,h,r,rem),expected)
                                assert adapter.last_stats['dense_compact_materialized'] is False
                                rows.append({'kind':kind,'feature':feature,'split':split,'order':order,'block':block,'terminal_source':source,
                                    'compact_device':placement,'max_logit_error':errors[0],'max_gradient_error':errors[1],'stats':adapter.last_stats})
    empty_checks=0
    for feature in ('bias','none'):
        model=PredictorPlus(graph,type='emb',hidden_dim=8,entity_feature=feature,aggregator='sum')
        model.set_rules([[0,3],[0,3,0]]);model.to(device)
        for relation in (0,2):
            rr=torch.full_like(h,relation)
            for source in sources:
                for placement in placements:
                    adapter=SparseMMPredictor(model,grounder_factory=make_factory(source),compact_device=placement)
                    with torch.no_grad():expected=model(h,rr,None);out=adapter(h,rr,None)
                    assert torch.equal(out[0],expected[0]) and torch.equal(out[1],expected[1])
                    if feature=='bias':compare(evaluate(model,adapter,h,rr,None),evaluate(model,model,h,rr,None))
                    empty_checks+=1
    # Inspect the real autograd saved tensors: sparse MM must save sparse counts,
    # not silently materialize a full rule-by-candidate dense operand.
    model=PredictorPlus(graph,type='emb',hidden_dim=8,entity_feature='bias',aggregator='sum')
    model.set_rules(rules);model.to(device)
    adapter=SparseMMPredictor(model,block_size=512,grounder_factory=make_factory(sources[-1]),collect_stats=True,compact_device=placements[-1])
    saved=[]
    def pack(tensor):
        entry={'layout':str(tensor.layout),'shape':list(tensor.shape),'dtype':str(tensor.dtype)}
        if tensor.layout==torch.sparse_coo:
            entry['nnz']=tensor._nnz();entry['storage_bytes']=tensor._indices().numel()*tensor._indices().element_size()+tensor._values().numel()*tensor._values().element_size()
        else:entry['storage_bytes']=tensor.numel()*tensor.element_size()
        saved.append(entry);return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        out,mask=adapter(h,r,removed);out.square().mean().backward()
    sparse_saved=[x for x in saved if x['layout']=='torch.sparse_coo']
    assert sparse_saved,'sparse MM did not retain a sparse count operand; inspect backend behavior'
    compact_shapes=[row['shape'] for row in adapter.last_stats['coo_blocks']]
    for entry in saved:
        if entry['layout']=='torch.strided':
            assert entry['shape'] not in compact_shapes and list(reversed(entry['shape'])) not in compact_shapes,entry
    result={'passed':True,'device':str(device),'torch':torch.__version__,'cases':rows,'empty_cases_passed':empty_checks,
        'saved_tensor_check':{'passed':True,'saved':saved,'compact_stats':adapter.last_stats},
        'seconds':time.perf_counter()-start,
        'scope':'tiny consumer correctness; sparse interface derives counts from unchanged dense graph oracle; not a sparse grounding performance result'}
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'device':str(device),'cases':len(rows),'empty_cases':empty_checks,
        'max_logit_error':max(x['max_logit_error'] for x in rows),'max_gradient_error':max(x['max_gradient_error'] for x in rows),
        'saved_sparse_operands':len(sparse_saved),'seconds':result['seconds']}))

if __name__=='__main__':main()
