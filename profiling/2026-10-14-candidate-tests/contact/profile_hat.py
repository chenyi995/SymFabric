import ast,sys,os,time,json,socket
from pathlib import Path
b=Path(__file__).resolve().parent;repo=b/'upstream/DiffCloth';out=b/'raw.json';src=repo/'src/python_code/hatController.py'
sys.path[:0]=[str(b/'build'),str(b.parent/'planning/deps'),str(src.parent)]
import numpy as np,torch
mod=ast.parse(src.read_text());body=[]
for node in mod.body:
 if isinstance(node,ast.Delete):continue
 if isinstance(node,ast.If) and isinstance(node.test,ast.UnaryOp) and isinstance(node.test.operand,ast.Name) and node.test.operand.id=='eval_mode':continue
 body.append(node)
mod.body=body
class Threads(ast.NodeTransformer):
 def visit_Call(self,n):
  self.generic_visit(n)
  if isinstance(n.func,ast.Attribute) and n.func.attr=='enableOpenMP':
   for k in n.keywords:
    if k.arg=='n_threads':k.value=ast.Constant(1)
  return n
mod=ast.fix_missing_locations(Threads().visit(mod));os.chdir(src.parent);torch.set_num_threads(1)
sys.argv=[str(src),'--eval','--load_expname','20210809-trainedBest']
ns={'__name__':'profile_hat','__file__':str(src)};exec(compile(mod,str(src),'exec'),ns)
ckpt=ns['exp_path']/'testBestEpoch.pth';ns['loadCheckpoint'](ckpt.parent,ckpt.name)
controller=ns['controller'];opt=ns['optimizer'];pairs=ns['X0A0pairs_eval'][:1];rows=[]
meta={'scope':'official pretrained DiffCloth hat neural controller full rollout training step; batch1 CPU1; stage diagnostic','checkpoint':str(ckpt),'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'rows':rows}
for i in range(3):
 a=time.perf_counter_ns();loss,details,sims=ns['simulateAndGetLoss'](pairs,render=False);bb=time.perf_counter_ns();opt.zero_grad();loss.backward();c=time.perf_counter_ns()
 grads=[p.grad for p in controller.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
 norm=torch.nn.utils.clip_grad_norm_(controller.parameters(),1.);opt.step();d=time.perf_counter_ns()
 rows.append(dict(iteration=i,loss=float(loss),success=bool(details[0]['succeed']),grad_norm_before_clip=float(norm),parameters_with_grad=len(grads),forward_loss_ms=(bb-a)/1e6,backward_ms=(c-bb)/1e6,gradcheck_clip_optimizer_ms=(d-c)/1e6,total_ms=(d-a)/1e6,frames=len(sims[0])))
 out.write_text(json.dumps(meta,indent=2));print(rows[-1],flush=True)
clean=[]
for i in range(3):
 ns['common'].setRandomSeed(42+i);a=time.perf_counter();loss=ns['trainStep'](ns['sim'],opt,trainSeqSampleNum=1,render=False);clean.append({'loss':loss,'e2e_ms':(time.perf_counter()-a)*1000})
meta['clean_random_train_steps']=clean;out.write_text(json.dumps(meta,indent=2))
