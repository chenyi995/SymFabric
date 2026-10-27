from pathlib import Path
setup=Path(__file__).with_name('bench.py').read_text().split('queries=')[0]
exec(compile(setup,__file__,'exec'))
from problog.logic import list2term
from deepproblog.semiring.graph_semiring import GraphSemiring
import numpy as np
from problog.evaluator import SemiringProbability
class InferenceScalarSemiring(GraphSemiring, SemiringProbability):
 def in_domain(self,a):
  return -self.eps <= float(a) <= 1.0+self.eps
 def result_in_domain(self,a):
  return self.in_domain(a)
 def __init__(self,model,substitution,values):
  assert not torch.is_grad_enabled()
  vals={k:(v.detach().cpu().tolist() if torch.is_tensor(v) else v) for k,v in values.items()}
  super().__init__(model,substitution,vals)
allrows=[]
for mode in ['native_probability_inference']:
 model.set_engine(ExactEngine(model),cache=True,semiring=GraphSemiring if mode=='original' else InferenceScalarSemiring)
 seen={}
 with torch.no_grad():
  for trial in range(6):
   for n in [1,2]:
    vs=[Term('p'+str(i)) for i in range(2*n)];ids=list(range(trial*4,trial*4+2*n))
    term=Term('addition',*vs,Var('S')) if n==1 else Term('multi_addition',list2term(vs[:n]),list2term(vs[n:]),Var('S'))
    q=Query(term,{v:Term('tensor',Term('images',Constant(i))) for v,i in zip(vs,ids)})
    t=time.perf_counter();res=model.solve([q])[0];lat=(time.perf_counter()-t)*1000
    probs=np.zeros(2*(10**n-1)+1)
    for term,value in res.result.items():probs[int(term.args[-1])]=float(value)
    p=cnn(images[ids]).numpy();aa=p[0];bb=p[n]
    for j in range(1,n):aa=np.outer(aa,p[j]).reshape(-1);bb=np.outer(bb,p[n+j]).reshape(-1)
    ref=np.convolve(aa,bb);err=float(np.max(np.abs(probs-ref)));assert err<1e-5,err
    ac=model.solver.cache.get(q)
    if n in seen:assert id(ac)==seen[n]
    seen[n]=id(ac)
    row={'variant':mode,'n':n,'trial':trial,'cold':trial==0,'solve_ms':lat,'max_error':err,'cold_ground_ms':res.ground_time*1000 if trial==0 else 0,'cold_compile_ms':res.compile_time*1000 if trial==0 else 0,'sdd_evaluate_ms':res.eval_time*1000,'probability_sum':float(probs.sum())};allrows.append(row);print(json.dumps(row),flush=True)
    (b/'native_raw.json').write_text(json.dumps({'scope':'official multi_addition program + trained MNIST CNN; original semiring and our inference-only scalar specialization via official semiring API; no algorithm replacement','host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'rows':allrows},indent=2))
