"""Matched real-hardware software controls; stage1 source/checkpoints unchanged."""
import argparse,gc,hashlib,json,os,random,socket,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent;S1=ROOT.parent/'2026-10-14-rnnlogic-strong-baselines';OLD=ROOT.parent/'2026-10-14-candidate-tests/kg'
sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(S1),str(OLD/'deps'),str(OLD/'upstream/RNNLogic/src')]
import torch
from data import KnowledgeGraph
from predictors import PredictorPlus
from run_bench import stats,sync
from prefix_cache import PrefixGrounder
from streaming_predictor import StreamingPredictor

class UnusedController:
 def begin(self,*args):raise ValueError('CaseRunner adapter is prepared only for the selected relation')

class CPUInputController:
 """Include CPU query copies inside each forward, never hidden setup."""
 def __init__(self,controller):self.controller=controller
 def begin(self,h,head,removed):
  self.h_original=h;self.removed_original=removed
  self.h=h.detach().cpu();self.removed=None if removed is None else removed.detach().cpu()
  self.controller.begin(self.h,head,self.removed)
 def grounding(self,h,head,body,removed):
  if h is not self.h_original or removed is not self.removed_original:raise ValueError('query changed')
  return self.controller.grounding(self.h,head,body,self.removed)
 def finish(self,require_complete=True):
  try:out=self.controller.finish(require_complete=require_complete)
  except TypeError:
   if not require_complete and hasattr(self.controller,'abort'):self.controller.abort();out={}
   else:out=self.controller.finish()
  self.h=self.removed=self.h_original=self.removed_original=None
  return out

class CapturedController:
 def __init__(self,c):self.c=c
 def begin(self,h,head,removed):self.c.prepare(h,head,removed)
 def grounding(self,h,head,body,removed):return self.c.grounding(h,head,body,removed)
 def finish(self,require_complete=True):self.c.finish(require_complete);return {'captured':True}

def sample(g,relation,batch,split,group,device):
 facts=[(h,t) for h,r,t in getattr(g,split+'_facts') if r==relation]
 random.Random(20261014+relation).shuffle(facts)
 facts=facts[group*batch:(group+1)*batch]
 if not facts:return None
 h=torch.tensor([x[0] for x in facts],device=device);r=torch.full_like(h,relation);t=torch.tensor([x[1] for x in facts],device=device)
 removed=torch.tensor([g.relation2ht2index[relation][g.encode_ht(a,b)] for a,b in facts],device=device) if split=='train' else None
 target=torch.zeros(len(h),g.entity_size,device=device)
 for i,(hh,_) in enumerate(facts):target[i,g.hr2o.get(g.encode_hr(hh,relation),[])]=1
 target=target*.2+torch.nn.functional.one_hot(t,g.entity_size)*.8
 return h,r,t,removed,target

def loss_from(out,target):
 logits,mask=out
 return -((torch.softmax(logits,dim=1)+1e-8).log()[mask]*target[mask]).sum()/target[mask].sum().clamp(min=1)

class CaseRunner:
 def __init__(self,dataset,relation,device,threads,variants):
  self.device=device;self.relation=relation
  self.g=KnowledgeGraph(str(OLD/'upstream/RNNLogic/data'/dataset));g=self.g
  self.original_g=g.grounding;self.original_p=g.propagate;self.adj=g.relation2adjacency
  self.m=PredictorPlus(g,type='lstm',num_layers=3,hidden_dim=16,aggregator='sum',entity_feature='bias')
  self.m.set_rules(str(OLD/'upstream/RNNLogic/data'/dataset/'rnnlogic_rules.txt'))
  self.checkpoint=S1/'checkpoints'/f'{dataset}_{relation}_lstm_sum_officialloss_v2_fit5.pt'
  ck=torch.load(self.checkpoint,map_location='cpu',weights_only=True);self.m.load_state_dict(ck['state']);self.m.to(device).eval()
  self.m.rule_features=self.m.rule_features.to(device);self.origagg=self.m.rule_to_entity.forward
  self.entries=list(self.m.relation2rules[relation]);self.bodies=[tuple(b) for _,(_,b) in self.entries]
  self.sorted_bodies=sorted(self.bodies)
  self.backends={};self.adapters={};self.captures={};self.metadata={}
  if any(v.startswith(('csr','stream','graph','gstream')) for v in variants):
   if device.type=='cuda':
    from gpu_backend import GPUBackend
    self.backends['dense']=GPUBackend(g,device=device,strategy='csr')
   else:
    from cpu_backend import CPUBackend
    self.backends['dense']=CPUBackend(g,threads=threads)
  if any(v.startswith('sgpu') for v in variants):
   from sparse_gpu_backend import SparseGPUBackend
   self.backends['sgpu']=SparseGPUBackend(g,device=device)
  for lib in ['scipy','graphblas']:
   if any(v.startswith(lib) for v in variants):
    from sparse_cpu_backend import SparseCPUBackend
    self.backends[lib]=SparseCPUBackend(g,library=lib,threads=threads)
  if any(v.startswith('gbopt_') for v in variants):
   from sparse_cpu_backend_optimized import SparseCPUBackendOptimized
   self.backends['gbopt']=SparseCPUBackendOptimized(g,threads=threads)
  for bits in [32,64]:
   if any(v.split('_')[0] in [f'native{bits}',f'native{bits}coo',f'native{bits}gpu'] for v in variants):
    from official_native_adapter import OfficialNativeBackend
    self.backends[f'native{bits}']=OfficialNativeBackend(g,threads=threads,bits=bits)
  self.ctrl=PrefixGrounder(g,self.bodies,policy='last_use',collect_stats=False)
  self.metadata['backends']={k:getattr(v,'metadata',{}) for k,v in self.backends.items()}
 def gemm(self,count,emb,batch):return torch.relu(self.m.rule_to_entity.layer_norm(self.m.rule_to_entity.add_model(count.T@emb)))
 def prepare(self,variant,h,removed):
  if variant in ['upstream','csr']:return
  if variant=='graph':
   from bench_cuda_graph import CapturedGrounding
   self.captures[variant]=CapturedGrounding(self.g.entity_size,self.bodies,self.relation,self.backends['dense'],h,removed)
   return
  kind,block=variant.split('_');block=int(block)
  if kind=='stream':factory=None;fmt='dense'
  elif kind=='gstream':
   from bench_cuda_graph import CapturedGrounding
   capture=CapturedGrounding(self.g.entity_size,self.sorted_bodies,self.relation,self.backends['dense'],h,removed)
   self.captures[variant]=capture
   def factory(g,bodies):
    return CapturedController(capture) if tuple(map(tuple,bodies))==tuple(self.sorted_bodies) else PrefixGrounder(g,bodies,policy='last_use',collect_stats=False)
   fmt='dense'
  elif kind=='sgpu':
   from sparse_gpu_backend import SparseGPUPrefixGrounder
   factory=lambda g,bodies:SparseGPUPrefixGrounder(self.backends[kind],bodies,return_format='sparse',collect_stats=False)
   fmt='sparse'
  elif kind in ['native32','native64','native64coo','native64gpu']:
   from official_native_adapter import OfficialNativeController
   factory=lambda g,bodies:CPUInputController(OfficialNativeController(self.backends['native32' if kind=='native32' else 'native64'],bodies)) if tuple(map(tuple,bodies))==tuple(self.sorted_bodies) else UnusedController()
   fmt='sparse'
  elif kind in ['scipy','graphblas','gbopt']:
   from sparse_cpu_backend import SparsePrefixGrounder
   factory=lambda g,bodies:CPUInputController(SparsePrefixGrounder(g,bodies,self.backends[kind],return_format='sparse',policy='last_use',collect_stats=False))
   fmt='sparse'
  else:raise ValueError(variant)
  if kind in ['native64coo','native64gpu']:
   from sparse_mm_predictor import SparseMMPredictor
   self.adapters[variant]=SparseMMPredictor(self.m,block_size=block,rule_order='prefix',grounder_factory=factory,compact_device='model' if kind.endswith('gpu') else 'source')
  else:self.adapters[variant]=StreamingPredictor(self.m,block_size=block,rule_order='prefix',grounder_factory=factory,terminal_format=fmt)
 def forward(self,variant,h,r,removed):
  g=self.g;m=self.m;g.relation2adjacency=self.adj;m.rule_to_entity.forward=self.origagg
  if variant=='upstream':g.grounding=self.original_g;g.propagate=self.original_p;return m(h,r,removed)
  if variant=='csr':
   g.propagate=self.backends['dense'].propagate;g.grounding=self.ctrl.grounding;m.rule_to_entity.forward=self.gemm
   self.ctrl.begin(h,self.relation,removed)
   try:return m(h,r,removed)
   finally:self.ctrl.finish()
  if variant=='graph':
   c=self.captures[variant];c.prepare(h,self.relation,removed);g.grounding=c.grounding;m.rule_to_entity.forward=self.gemm
   try:return m(h,r,removed)
   finally:c.finish()
  if variant.startswith('stream'):g.propagate=self.backends['dense'].propagate
  return self.adapters[variant](h,r,removed)
 def close(self):
  for c in self.captures.values():c.close()
  self.captures.clear();self.adapters.clear()

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--cases',nargs='+',default=['FB15k-237:272','wn18rr:12']);p.add_argument('--batches',type=int,nargs='+',default=[32]);p.add_argument('--splits',nargs='+',default=['train']);p.add_argument('--groups',type=int,default=1);p.add_argument('--threads',type=int,default=1);p.add_argument('--variants',nargs='+',default=['upstream','csr','stream_32','stream_128','sgpu_128']);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=5);p.add_argument('--grad',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(42)
 device=torch.device(a.device,torch.cuda.current_device()) if a.device=='cuda' else torch.device('cpu')
 result={'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'pinned_cpus':os.getenv('BENCH_CPU_LIST'),'torch':torch.__version__,'gpu':torch.cuda.get_device_name() if a.device=='cuda' else None,'config':vars(a),'scope':'resident complete PredictorPlus sum forward; CPU sparse hybrid includes query D2H and terminal H2D; no EM/full epoch/cold setup','timing':'synchronized wall clock, randomized interleaved variants; setup and correctness outside timing','cases':[]}
 out=Path(a.output)
 def save():
  tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(out)
 save()
 for spec in a.cases:
  ds,relation=spec.split(':');relation=int(relation)
  for batch in a.batches:
   for split in a.splits:
    for group in range(a.groups):
     gc.collect()
     if device.type=='cuda':torch.cuda.empty_cache()
     start=time.perf_counter();runner=CaseRunner(ds,relation,device,a.threads,a.variants);g=runner.g;m=runner.m
     inp=sample(g,relation,batch,split,group,device)
     if inp is None:runner.close();continue
     h,r,t,removed,target=inp
     row={'dataset':ds,'relation':relation,'batch':len(h),'split':split,'group':group,'threads':a.threads,'checkpoint':str(runner.checkpoint),'checkpoint_sha256':hashlib.sha256(runner.checkpoint.read_bytes()).hexdigest(),'setup_seconds':time.perf_counter()-start,'backends':runner.metadata,'variants':{},'failures':{}}
     result['cases'].append(row);save()
     with torch.no_grad():ref,refmask=runner.forward('upstream',h,r,removed)
     ref=ref.detach();refmask=refmask.detach();active=[]
     for v in a.variants:
      try:
       if removed is None and v.startswith(('graph','gstream')):raise ValueError('capture training deletion path only')
       print('validate',spec,split,group,v,flush=True)
       start=time.perf_counter();runner.prepare(v,h,removed);sync(device);prep=time.perf_counter()-start
       with torch.no_grad():pred,mask=runner.forward(v,h,r,removed)
       assert torch.equal(mask,refmask) and torch.allclose(pred,ref,atol=2e-5,rtol=2e-4),(v,float((pred-ref).abs().max()))
       row['variants'][v]={'logit_max_abs_error':float((pred-ref).abs().max()),'mask_equal':True,'prepare_seconds':prep}
       if v in runner.captures:row['variants'][v]['capture']=runner.captures[v].metadata
       active.append(v);del pred,mask
      except Exception as e:
       row['failures'][v]={'stage':'prepare/forward correctness','error':str(e),'traceback':traceback.format_exc()};print('FAILED',spec,v,str(e),flush=True)
      save()
     for _ in range(a.warmup):
      for v in active:
       with torch.no_grad():runner.forward(v,h,r,removed)
     times={v:[] for v in active}
     for k in range(a.repeats):
      order=list(active);random.Random(31415+k).shuffle(order)
      for v in order:
       sync(device);start=time.perf_counter()
       with torch.no_grad():pred=runner.forward(v,h,r,removed)
       sync(device);times[v].append((time.perf_counter()-start)*1000);del pred
     for v in active:
      rr=row['variants'][v];rr['forward']=stats(times[v])
      gc.collect();sync(device)
      if device.type=='cuda':base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
      with torch.no_grad():pred=runner.forward(v,h,r,removed)
      sync(device)
      if device.type=='cuda':rr['memory']={'base_allocated_bytes':base,'incremental_peak_bytes':torch.cuda.max_memory_allocated()-base}
      del pred
     save();print(json.dumps({'case':spec,'batch':len(h),'split':split,'group':group,'ms':{v:row['variants'][v]['forward']['median_ms'] for v in active}}),flush=True)
     if a.grad and split=='train':
      m.train();refgrads=None
      for v in ['upstream']+[x for x in active if x!='upstream']:
       m.zero_grad(set_to_none=True);loss=loss_from(runner.forward(v,h,r,removed),target);loss.backward();gg={n:p.grad.detach().clone() for n,p in m.named_parameters() if p.grad is not None}
       if refgrads is None:refgrads=gg;refloss=float(loss.detach())
       else:
        errors={n:float((x-refgrads[n]).abs().max()) for n,x in gg.items()}
        ok=set(gg)==set(refgrads) and all(torch.allclose(x,refgrads[n],rtol=3e-3,atol=3e-5) for n,x in gg.items())
        row['variants'][v]['gradient']={'allclose':ok,'loss_abs_error':abs(float(loss.detach())-refloss),'max_gradient_abs_error':max(errors.values(),default=0)}
        if not ok:row['failures'][v]={'stage':'gradient','errors':errors}
       del loss
      m.zero_grad(set_to_none=True);m.eval();del refgrads,gg
     save();runner.close();del runner,m,g,inp,ref,refmask,h,r,t,removed,target
     gc.collect()
 result['complete']=True;save()
if __name__=='__main__':main()
