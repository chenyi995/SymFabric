import ast,sys,time,json,os,subprocess,socket
from pathlib import Path
b=Path(__file__).resolve().parent;repo=b/'upstream/UPIX';out=b/'upix_raw.json';source=repo/'evaluation/pedestrian/run_comp.py'
budget={'init_n_samples':32,'init_estimate_weight_n_samples':1024,'estimate_weight_n_samples':4096}
class Budget(ast.NodeTransformer):
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id=='DCCConfig':
   for kw in node.keywords:
    if kw.arg in budget:kw.value=ast.Constant(budget[kw.arg])
  return node
module=ast.fix_missing_locations(Budget().visit(ast.parse(source.read_text())))
meta={'scope':'official UPIX pedestrian stochastic-support model and existing DCCConfig/DHMC; reduced sample budgets; no neural component','host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'budget':budget,'mcmc_chains':1,'samples_per_chain':64,'status':'started'}
out.write_text(json.dumps(meta,indent=2));os.chdir(repo);sys.path[:0]=[str(repo),str(source.parent)]
sys.argv=[str(source),'sequential','vmap_local','-n_chains','1','-n_samples_per_chain','64','--cpu','-host_device_count','1','-seed','0','--no_save','--no_progress']
ns={'__name__':'__main__','__file__':str(source)};a=time.perf_counter()
def brief(x,depth=0):
 if x is None or isinstance(x,(str,int,float,bool)):return x
 if hasattr(x,'shape') and hasattr(x,'dtype'):
  import numpy as np
  arr=np.asarray(x)
  return arr.tolist() if arr.size<=4096 else {'shape':list(arr.shape),'dtype':str(arr.dtype)}
 if depth>4:return repr(x)[:200]
 if isinstance(x,dict):return {str(k):brief(v,depth+1) for k,v in list(x.items())[:64]}
 if isinstance(x,(list,tuple)):return [brief(v,depth+1) for v in x[:64]]
 if hasattr(x,'__dict__'):return {'class':type(x).__name__,'fields':brief(vars(x),depth+1)}
 return repr(x)[:500]
try:
 exec(compile(module,str(source),'exec'),ns)
 meta.update(status='completed',whole_script_s=time.perf_counter()-a,official_timings=brief(ns.get('timings')),result=brief(ns.get('result')))
except BaseException as e:
 meta.update(status='error',error=repr(e),whole_script_s=time.perf_counter()-a);out.write_text(json.dumps(meta,indent=2));raise
out.write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k!='result'},indent=2))
