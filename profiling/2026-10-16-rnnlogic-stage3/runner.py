"""One graph/model for all relations; local software controls, upstream untouched."""
from pathlib import Path
import sys,time,hashlib
ROOT=Path(__file__).resolve().parent
S2=ROOT.parent/'2026-10-14-rnnlogic-sparse-stage2'
S1=ROOT.parent/'2026-10-14-rnnlogic-strong-baselines'
OLD=ROOT.parent/'2026-10-14-candidate-tests/kg'
sys.path[:0]=[str(ROOT),str(S2),str(S2/'deps'),str(S1),str(OLD/'deps'),str(OLD/'upstream/RNNLogic/src')]
import torch
from data import KnowledgeGraph
from predictors import PredictorPlus
from gpu_backend import GPUBackend
from prefix_cache import PrefixGrounder
from official_native_adapter import OfficialNativeBackend,OfficialNativeController
from bench_stage2 import CPUInputController,loss_from
from sparse_mm_predictor import SparseMMPredictor
from cached_sparse_faststats_predictor import CachedSparseFastStatsPredictor

class LazyNative:
 def __init__(self,backend,bodies):self.backend=backend;self.bodies=bodies;self.impl=None
 def begin(self,*args):
  if self.impl is None:self.impl=CPUInputController(OfficialNativeController(self.backend,self.bodies))
  self.impl.begin(*args)
 def grounding(self,*args):return self.impl.grounding(*args)
 def finish(self,**kwargs):return self.impl.finish(**kwargs)

class Runner:
 def __init__(self,dataset,threads=16,pna_tile=512,native_kind="stdmap"):
  start=time.perf_counter();self.dataset=dataset;self.device=torch.device('cuda',torch.cuda.current_device())
  self.g=KnowledgeGraph(str(OLD/'upstream/RNNLogic/data'/dataset));g=self.g
  pna=dataset=='wn18rr';self.aggregator='pna' if pna else 'sum'
  self.m=PredictorPlus(g,type='emb' if pna else 'lstm',num_layers=3,hidden_dim=16,aggregator=self.aggregator,entity_feature='bias')
  self.m.set_rules(str(OLD/'upstream/RNNLogic/data'/dataset/'rnnlogic_rules.txt'))
  self.checkpoint=S1/'checkpoints'/('wn18rr_12_official_officialloss_v2_fit5.pt' if pna else 'FB15k-237_272_lstm_sum_officialloss_v2_fit5.pt')
  self.m.load_state_dict(torch.load(self.checkpoint,map_location='cpu',weights_only=True)['state']);self.m.to(self.device).eval();self.m.rule_features=self.m.rule_features.to(self.device)
  self.original_g=g.grounding;self.original_p=g.propagate;self.adj=g.relation2adjacency;self.origagg=self.m.rule_to_entity.forward
  self.gpu=GPUBackend(g,device=self.device,strategy='csr')
  if native_kind=='spa':
   from spa_native_adapter import SPANativeBackend
   self.native=SPANativeBackend(g,threads=threads,bits=64)
  elif native_kind=='stdmap':self.native=OfficialNativeBackend(g,threads=threads,bits=64)
  else:raise ValueError(native_kind)
  self.native_kind=native_kind
  self.controllers={};self.prepared={};self.gpu_builders={};self.rule_version=hashlib.sha256((OLD/'upstream/RNNLogic/data'/dataset/'rnnlogic_rules.txt').read_bytes()).hexdigest()
  factory=lambda graph,bodies:LazyNative(self.native,bodies)
  if pna:
   from pna_predictor import PNAPredictor,CachedPNAPredictor
   self.sparse=PNAPredictor(self.m,factory,candidate_block_size=pna_tile)
   self.cached=CachedPNAPredictor(self.m,namespace=dataset+':train-static',rule_version=self.rule_version,candidate_block_size=pna_tile)
  else:
   self.sparse=SparseMMPredictor(self.m,block_size=128,rule_order='original',grounder_factory=factory,compact_device='model')
   self.cached=CachedSparseFastStatsPredictor(self.m,namespace=dataset+':train-static',rule_version=self.rule_version,rule_order='original')
  self.pna_tile=pna_tile;torch.cuda.synchronize()
  self.metadata={'setup_seconds':time.perf_counter()-start,'dataset':dataset,'model_type':self.m.type,'aggregator':self.aggregator,'checkpoint':str(self.checkpoint),'checkpoint_sha256':hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),'checkpoint_scope':'5 short fitting steps on one relation; no accuracy or convergence claim','native_kind':native_kind,'native':self.native.metadata,'gpu':self.gpu.jit_metadata}
 def gemm(self,count,emb,batch):return torch.relu(self.m.rule_to_entity.layer_norm(self.m.rule_to_entity.add_model(count.T@emb)))
 def pna(self,count,emb,batch):
  if self.pna_tile==0:return self.origagg(count,emb,batch)
  from pna_predictor import pna_aggregate
  return pna_aggregate(self.m.rule_to_entity,count,emb,batch,self.pna_tile)
 def plan(self,r):return [tuple(body) for _,(_,body) in self.m.relation2rules[r]]
 def native_terminals(self,h,r,removed):
  rr=int(r[0]);
  if rr not in self.prepared:self.prepared[rr]=self.native.prepare_rules(self.plan(rr))
  return self.native.batch(self.prepared[rr],h,rr,removed)
 def gpu_terminals(self,h,r,removed):
  from gpu_cache_builder import GPUCSRCountBuilder
  rr=int(r[0])
  if rr not in self.prepared:self.prepared[rr]=self.native.prepare_rules(self.plan(rr))
  if rr not in self.gpu_builders:self.gpu_builders[rr]=GPUCSRCountBuilder(self.g,self.gpu,self.cached.rules_for_relation(rr))
  return self.gpu_builders[rr].batch(h.to(self.device),rr,None if removed is None else removed.to(self.device))[0]
 def forward(self,variant,h,r,removed):
  g=self.g;m=self.m;g.relation2adjacency=self.adj;m.rule_to_entity.forward=self.origagg
  if variant=='upstream':g.grounding=self.original_g;g.propagate=self.original_p;return m(h,r,removed)
  if variant in ('csr','csr_tiled'):
   rr=int(r[0]);g.propagate=self.gpu.propagate
   if rr not in self.controllers:self.controllers[rr]=PrefixGrounder(g,self.plan(rr),policy='last_use',collect_stats=False)
   c=self.controllers[rr];g.grounding=c.grounding
   if self.aggregator=='sum':m.rule_to_entity.forward=self.gemm
   elif variant=='csr_tiled':m.rule_to_entity.forward=self.pna
   c.begin(h,rr,removed)
   try:return m(h,r,removed)
   finally:c.finish()
  if variant=='native':return self.sparse(h,r,removed)
  if variant=='cached':return self.cached(h,r,removed)
  raise ValueError(variant)
 def inputs(self,array):
  cpu=torch.as_tensor(array.copy(),dtype=torch.int64)
  h,r,t,removed=[cpu[:,i].contiguous() for i in range(4)]
  return (h,r,t,removed),(h.to(self.device),r.to(self.device),t.to(self.device),removed.to(self.device))
 def target(self,h,r,t):
  target=torch.zeros(len(h),self.g.entity_size)
  for i,hh in enumerate(h.tolist()):target[i,self.g.hr2o.get(self.g.encode_hr(hh,int(r[i])),[])]=1
  return (target*.2+torch.nn.functional.one_hot(t,self.g.entity_size)*.8).to(self.device)
 def close(self):
  self.cached.reset(namespace='closed',rule_version=self.rule_version);self.native.close()
