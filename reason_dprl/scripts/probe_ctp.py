#!/usr/bin/env python3
"""Official CTP execution probe, not a trained accuracy reproduction.

Use a small fixed subset of its bundled Kinship facts and seeded random frozen
embeddings. Record operator costs and whether data values change operation
shapes/counts. This tests an execution mechanism, not DPLA* or ACTP performance.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import sys

from bench_common import ROOT, provenance, save, measure, graph_measure, profile, sha
sys.path.insert(0,str(ROOT/'softwareworkload/upstream/ctp'))
import torch
from ctp.clutrr.models import BatchHoppy, BatchNeuralKB
from ctp.reformulators import LinearReformulator
from ctp.kernels import GaussianKernel


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repeat',type=int,default=20)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    upstream=ROOT/'softwareworkload/upstream/ctp'
    data=upstream/'data/kinship/train.tsv'
    triples=[x.split() for x in data.read_text().splitlines()[:32]]
    entities=sorted({e for s,p,o in triples for e in (s,o)})
    predicates=sorted({p for s,p,o in triples})
    ei={x:i for i,x in enumerate(entities)};pi={x:i for i,x in enumerate(predicates)}
    b,e=16,32
    et=torch.randn(len(entities),e,device='cuda')*.1
    pt=torch.randn(len(predicates),e,device='cuda')*.1
    facts=[pt[torch.tensor([pi[p] for s,p,o in triples],device='cuda')],
           et[torch.tensor([ei[s] for s,p,o in triples],device='cuda')],
           et[torch.tensor([ei[o] for s,p,o in triples],device='cuda')]]
    facts=[x[None].repeat(b,1,1) for x in facts]
    q=tuple(x[:,0] for x in facts)
    other=tuple(x[:,7] for x in facts)
    nf=torch.full((b,),len(triples),device='cuda',dtype=torch.long)
    ne=torch.full((b,),len(entities),device='cuda',dtype=torch.long)
    eb=et[None].repeat(b,1,1)
    metadata=provenance()
    metadata.update(ctp_commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
                    input_sha256=sha(data),prefix_facts=len(triples),entities=len(entities),
                    batch=b,embedding_size=e,seed=42,tf32=False,
                    scope='official BatchHoppy; Kinship fact prefix; random frozen weights; mechanism probe only')
    save(args.output/'provenance.json',metadata)
    rows=[]
    with torch.no_grad():
        for depth,nreform in ((0,1),(1,1),(2,1),(2,2)):
            model=BatchHoppy(BatchNeuralKB(GaussianKernel()),
                             [(LinearReformulator(2,e),False) for _ in range(nreform)],k=2,depth=depth).cuda().eval()
            def run(query=q):
                return model.score(*query,facts,nf,eb,ne)
            output,timing=measure(run,repeat=args.repeat)
            expected=output.clone()
            captured,graph_timing=graph_measure(run,repeat=args.repeat)
            torch.testing.assert_close(captured,expected,atol=0,rtol=0)
            assert torch.isfinite(output).all()
            sequences=[]
            for query in (q,other):
                sequence=[]
                handles=[]
                for i,module in enumerate(model._hops_lst):
                    handles.append(module.register_forward_hook(
                        lambda m,a,o,idx=i:sequence.append(('reformulator',idx,tuple(a[0].shape)))))
                original_score=model.model.score
                original_forward=model.model.forward
                def score(*a,**kw):
                    sequence.append(('fact_score',tuple(a[0].shape)))
                    return original_score(*a,**kw)
                def forward(*a,**kw):
                    sequence.append(('fact_forward',tuple(a[0].shape)))
                    return original_forward(*a,**kw)
                model.model.score=score;model.model.forward=forward
                run(query)
                model.model.score=original_score;model.model.forward=original_forward
                for h in handles:
                    h.remove()
                sequences.append(sequence)
            name=f'depth{depth}-reform{nreform}'
            prof=profile(run,args.output,name)
            row={'depth':depth,'reformulators':nreform,'latency':timing,'profile':prof,
                 'fixed_query_graph_latency':graph_timing,'graph_output_equal':True,
                 'query_values_differ':not all(torch.equal(x,y) for x,y in zip(q,other)),
                 'two_query_operation_shapes_equal':sequences[0]==sequences[1],
                 'operation_counts':dict(Counter(s[0] for s in sequences[0])),
                 'operation_sequence':sequences[0]}
            rows.append(row);save(args.output/'benchmark.json',rows)
            print('CTP_RESULT',depth,nreform,timing['median_ms'],row['operation_counts'],flush=True)
    print('CTP_PROBE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
