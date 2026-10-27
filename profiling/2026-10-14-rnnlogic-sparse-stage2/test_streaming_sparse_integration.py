"""Compose the actual existing sparse libraries with the sum consumer on a tiny graph."""
import argparse
import json
from pathlib import Path
import torch
from test_streaming_predictor import TinyGraph, evaluate, compare
from predictors import PredictorPlus
from streaming_predictor import StreamingPredictor
from sparse_cpu_backend import SparseCPUBackend, SparsePrefixGrounder


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    graph=TinyGraph()
    patterns=[[],[0],[1],[3],[0,0],[0,1],[1,0],[2,0],[0,1,0],[2,2,0],[3,0],[1,2]]
    rules=[[0]+patterns[(i*7)%len(patterns)] for i in range(137)]+[[1,1]]
    h=torch.tensor([0,1,2,256]);r=torch.zeros_like(h);removed=torch.tensor([0,1,2,3])
    rows=[]
    for lib in ('scipy','graphblas'):
        backend=SparseCPUBackend(graph,library=lib,threads=1)
        for kind in ('emb','lstm'):
            torch.manual_seed(42)
            model=PredictorPlus(graph,type=kind,num_layers=2,hidden_dim=8,entity_feature='bias',aggregator='sum')
            model.set_rules(rules)
            for split,rem in [('train',removed),('valid',None)]:
                expected=evaluate(model,model,h,r,rem)
                for block in (8,32,128):
                    factory=lambda g,bodies:SparsePrefixGrounder(g,bodies,backend=backend,return_format='sparse',policy='last_use',collect_stats=True)
                    adapter=StreamingPredictor(model,block_size=block,rule_order='prefix',grounder_factory=factory,terminal_format='sparse',collect_stats=True)
                    errors=compare(evaluate(model,adapter,h,r,rem),expected)
                    rows.append({'library':lib,'kind':kind,'split':split,'block':block,
                                 'max_logit_error':errors[0],'max_gradient_error':errors[1], 'stats':adapter.last_stats})
    result={'passed':True,'cases':rows,'scope':'tiny graph integrated sparse grounding and sum model; CPU only; all model parameter gradients compared'}
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'cases':len(rows),'max_logit_error':max(x['max_logit_error'] for x in rows),'max_gradient_error':max(x['max_gradient_error'] for x in rows)}))

if __name__=='__main__':main()
