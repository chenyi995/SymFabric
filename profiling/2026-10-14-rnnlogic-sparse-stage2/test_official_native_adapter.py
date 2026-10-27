"""Semantics and real graph checks; timings here are not performance claims."""
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'2026-10-14-candidate-tests/kg'
sys.path[:0]=[str(OLD/'deps'),str(OLD/'upstream/RNNLogic/src')]
from data import KnowledgeGraph
from predictors import PredictorPlus
from official_native_adapter import OfficialNativeBackend,OfficialNativeController
from streaming_predictor import StreamingPredictor
from test_streaming_predictor import TinyGraph,evaluate,compare

def dense(pair,batch,n):
    a=torch.zeros(batch*n,dtype=torch.int64);a[pair[0]]=pair[1];return a.view(batch,n)

def tiny():
    g=TinyGraph()
    try: OfficialNativeBackend(g)
    except ValueError as e: assert 'duplicate' in str(e)
    else: raise AssertionError('duplicate triple accepted')
    # Retain first copy exactly as an admissible Python-loaded graph would.
    for i,(index,val) in enumerate(g.relation2adjacency):
        keys=index[1]*g.entity_size+index[0];seen=set();keep=[]
        for j,k in enumerate(keys.tolist()):
            if k not in seen:keep.append(j);seen.add(k)
        g.relation2adjacency[i]=[index[:,keep],val[keep]]
    h=torch.tensor([0,1,2,256]);r=torch.zeros_like(h);removed=torch.tensor([0,1,2,3])
    bodies=[(0,),(1,),(3,),(0,0),(0,1),(1,0),(2,0),(0,1,0),(2,2,0),(3,0),(1,2)]
    rows=[]
    for bits in [32,64]:
        backend=OfficialNativeBackend(g,threads=1,bits=bits)
        tested=bodies+([()] if bits==64 else [])
        for split,rem in [('train',removed),('valid',None)]:
            results=backend.batch(backend.prepare_rules(tested),h,0,rem)
            for body,pair in zip(tested,results):
                assert torch.equal(dense(pair,len(h),g.entity_size),g.grounding(h,0,list(body),rem))
            for kind in ['emb','lstm']:
                torch.manual_seed(42)
                m=PredictorPlus(g,type=kind,num_layers=2,hidden_dim=8,entity_feature='bias',aggregator='sum')
                m.set_rules([[0]+list(body) for body in tested])
                adapter=StreamingPredictor(m,block_size=8,rule_order='prefix',terminal_format='sparse',
                    grounder_factory=lambda graph,b:OfficialNativeController(backend,b),collect_stats=True)
                err=compare(evaluate(m,adapter,h,r,rem),evaluate(m,m,h,r,rem))
                rows.append(dict(bits=bits,split=split,kind=kind,logit_error=err[0],gradient_error=err[1],stats=adapter.last_stats))
        if bits==32:
            try:backend.prepare_rules([()])
            except ValueError:pass
            else:raise AssertionError('raw32 empty semantic difference not guarded')
        backend.close()
    # 2-node complete directed graph: 32 hops produces 2^31 paths per output.
    g.entity_size=2;g.relation_size=1
    g.relation2adjacency=[[torch.tensor([[0,1,0,1],[0,0,1,1]]),torch.ones(4)]]
    b64=OfficialNativeBackend(g,bits=64);p=b64.prepare_rules([(0,)*32]);out=b64.batch(p,torch.tensor([0]),0)[0]
    assert torch.equal(out[1],torch.full((2,),2**31,dtype=torch.int64))
    b32=OfficialNativeBackend(g,bits=32)
    for b,L in [(b32,32),(b64,64)]:
        try:b.prepare_rules([(0,)*L])
        except OverflowError:pass
        else:raise AssertionError('unsafe bound accepted')
    b32.close();b64.close()
    return dict(passed=True,cases=rows,duplicate_rejected=True,empty_rule_difference_guarded=True,
        count_above_int32_verified=True,int32_and_int64_bound_guards_passed=True)

def real(threads):
    rows=[]
    for ds,rel in [('FB15k-237',272),('wn18rr',12)]:
        g=KnowledgeGraph(str(OLD/'upstream/RNNLogic/data'/ds))
        bodies=[]
        for line in (Path(g.data_path)/'rnnlogic_rules.txt').read_text().splitlines():
            v=list(map(int,line.split()))
            if v[0]==rel:bodies.append(tuple(v[1:]))
        backends={bits:OfficialNativeBackend(g,threads=threads,bits=bits) for bits in [32,64]}
        for split in ['train','valid']:
            facts=[x for x in getattr(g,split+'_facts') if x[1]==rel][:4]
            h=torch.tensor([x[0] for x in facts]);rem=None
            if split=='train':rem=torch.tensor([g.relation2ht2index[rel][g.encode_ht(x[0],x[2])] for x in facts])
            outputs={}
            for bits,backend in backends.items():
                selected=[(k,b) for k,b in enumerate(bodies) if bits==64 or b]
                results=backend.batch(backend.prepare_rules([b for _,b in selected]),h,rel,rem)
                outputs[bits]={k:pair for (k,b),pair in zip(selected,results)}
            count=0;maxcount=0;nnz=0
            for k,body in enumerate(bodies):
                oracle=g.grounding(h,rel,list(body),rem)
                for bits in backends:
                    if k not in outputs[bits]:continue
                    assert torch.equal(dense(outputs[bits][k],len(h),g.entity_size),oracle),(ds,split,bits,k)
                    count+=1
                maxcount=max(maxcount,int(oracle.max()));nnz+=int(oracle.count_nonzero())
            rows.append(dict(dataset=ds,relation=rel,split=split,queries=len(h),rules=len(bodies),terminal_tensor_checks=count,
                all_exact=True,native32_skipped_empty_rules=sum(not b for b in bodies),terminal_nnz=nnz,max_terminal_count=maxcount,threads=threads,metadata={bits:b.metadata for bits,b in backends.items()}))
            print({k:v for k,v in rows[-1].items() if k!='metadata'},flush=True)
        for b in backends.values():b.close()
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--real',action='store_true');ap.add_argument('--threads',type=int,default=1);ap.add_argument('--output',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out={'complete':False,'scope':'correctness only; official sparse count algorithm, no shared prefixes','tiny':tiny()}
    if a.real:out['real']=real(a.threads)
    out['complete']=True;Path(a.output).write_text(json.dumps(out,indent=2));print({'complete':True,'real':a.real,'tiny_cases':len(out['tiny']['cases'])})
