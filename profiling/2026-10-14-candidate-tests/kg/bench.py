import os, sys, time, json, statistics, argparse, socket, subprocess, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT/'upstream/RNNLogic/src')]
import torch
from data import KnowledgeGraph
from predictors import PredictorPlus
p=argparse.ArgumentParser(); p.add_argument('--device',default='cpu'); p.add_argument('--datasets',nargs='+',default=['FB15k-237','wn18rr']); a=p.parse_args()
torch.set_num_threads(1); torch.manual_seed(42)
dev=torch.device(a.device)
def sync():
 if dev.type=='cuda': torch.cuda.synchronize()
def timed(fn,n=3):
 fn(); sync(); ts=[]
 for _ in range(n):
  sync(); t=time.perf_counter(); fn(); sync(); ts.append((time.perf_counter()-t)*1000)
 return {'median_ms':statistics.median(ts),'samples_ms':ts}
results={'device':str(dev),'gpu':torch.cuda.get_device_name() if dev.type=='cuda' else None,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'torch':torch.__version__,'commit':subprocess.check_output(['git','-C',str(ROOT/'upstream/RNNLogic'),'rev-parse','HEAD'],text=True).strip(),'scope':'real published KG and learned rules; full PredictorPlus forward/backward; model weights only briefly fitted, not paper accuracy reproduction','cases':[]}
for dataset in a.datasets:
 g=KnowledgeGraph(str(ROOT/'upstream/RNNLogic/data'/dataset))
 m=PredictorPlus(g,type='lstm',num_layers=3,hidden_dim=16).to(dev)
 m.set_rules(str(ROOT/'upstream/RNNLogic/data'/dataset/'rnnlogic_rules.txt')); m=m.to(dev)
 orig=g.grounding; agg_orig=m.rule_to_entity.forward; adj=[[x.clone() for x in aa] for aa in g.relation2adjacency]
 facts=collections.defaultdict(list)
 for h,r,t in g.train_facts: facts[r].append((h,t))
 rels=sorted([r for r in facts if m.relation2rules[r]],key=lambda r:len(m.relation2rules[r]))
 selected=list(dict.fromkeys([rels[0],rels[len(rels)//2],rels[-1]]))
 for r in selected:
  g.grounding=orig; m.rule_to_entity.forward=agg_orig
  ht=facts[r][:32]; h=torch.tensor([x[0] for x in ht],device=dev); rr=torch.full_like(h,r); target=torch.tensor([x[1] for x in ht],device=dev)
  edges=torch.tensor([g.relation2ht2index[r][g.encode_ht(x,y)] for x,y in ht],device=dev)
  rules=m.relation2rules[r]; calls=sum(len(z[1][1]) for z in rules); prefixes={tuple(z[1][1][:i]) for z in rules for i in range(1,len(z[1][1])+1)}
  case={'dataset':dataset,'entities':g.entity_size,'train_edges':len(g.train_facts),'relation':r,'batch':len(ht),'rules':len(rules),'original_propagations':calls,'unique_prefixes':len(prefixes),'variants':{}}
  opt=torch.optim.Adam(m.parameters(),lr=.005)
  g.relation2adjacency=[[x.to(dev) for x in aa] for aa in adj]
  losses=[]
  for _ in range(5):
   opt.zero_grad(); score,_=m(h,rr,edges); loss=torch.nn.functional.cross_entropy(score,target); loss.backward(); opt.step(); losses.append(float(loss.detach()))
  case['brief_fit_losses']=losses
  for mode in ['original','resident','prefix','prefix_gemm']:
   g.relation2adjacency=adj if mode=='original' else [[x.to(dev) for x in aa] for aa in adj]
   cache={}
   def cached(hh,head,body,removed):
    key=()
    if key not in cache: cache[key]=torch.nn.functional.one_hot(hh,g.entity_size).T.unsqueeze(-1)
    for relation in body:
     nextkey=key+(relation,)
     if nextkey not in cache: cache[nextkey]=g.propagate(cache[key],relation,removed if relation==head else None)
     key=nextkey
    return cache[key].squeeze(-1).T
   g.grounding=cached if mode.startswith('prefix') else orig
   def agg_gemm(count,emb,batch_ids):
    return torch.relu(m.rule_to_entity.layer_norm(m.rule_to_entity.add_model(count.T @ emb)))
   m.rule_to_entity.forward=agg_gemm if mode=='prefix_gemm' else agg_orig
   def forward():
    cache.clear()
    with torch.no_grad(): return m(h,rr,edges)
   hook_calls=[]
   hook=m.score_model.register_forward_hook(lambda module,inputs,out:hook_calls.append(list(inputs[0].shape)))
   out,mask=forward(); hook.remove()
   if mode=='original': reference=out.detach().clone(); refmask=mask.clone()
   err=float((out-reference).abs().max()); assert torch.equal(mask,refmask) and torch.allclose(out,reference,rtol=2e-4,atol=2e-5),err
   def grounds():
    cache.clear()
    for _,(_,body) in rules: g.grounding(h,r,body,edges)
   def step():
    cache.clear(); opt.zero_grad(); score,_=m(h,rr,edges); loss=torch.nn.functional.cross_entropy(score,target); loss.backward()
   case['variants'][mode]={'forward':timed(forward),'grounding_only':timed(grounds),'forward_backward':timed(step),'max_logit_error':err,'score_model_shapes':hook_calls}
   if dev.type=='cuda':
    torch.cuda.reset_peak_memory_stats(); forward(); sync(); case['variants'][mode]['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
   print(json.dumps({'case':dataset,'r':r,'mode':mode,'result':case['variants'][mode]}),flush=True)
  results['cases'].append(case)
  (ROOT/f'results_v3_{a.device}.json').write_text(json.dumps(results,indent=2))
