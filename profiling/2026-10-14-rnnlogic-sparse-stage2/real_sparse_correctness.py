import json,sys,torch
from pathlib import Path
p=Path(__file__).resolve().parent
sys.path[:0]=[str(p),str(p/'deps'),str(p.parent/'2026-10-14-rnnlogic-strong-baselines'),str(p.parent/'2026-10-14-candidate-tests/kg/deps'),str(p.parent/'2026-10-14-candidate-tests/kg/upstream/RNNLogic/src')]
from bench_stage2 import CaseRunner,sample
from sparse_cpu_backend import SparseCPUBackend,SparsePrefixGrounder
from sparse_gpu_backend import SparseGPUBackend,SparseGPUPrefixGrounder

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 out={'complete':False,'scope':'real graph exact int64 terminal and per-step sparse-state comparison; not a timing benchmark','cases':[]}
 for ds,rel in [('FB15k-237',272),('wn18rr',12)]:
  runner=CaseRunner(ds,rel,torch.device('cuda',torch.cuda.current_device()),1,['upstream']);g=runner.g
  libs={lib:SparseCPUBackend(g,library=lib,threads=1) for lib in ['scipy','graphblas']};gpu=SparseGPUBackend(g)
  bodies=sorted(runner.bodies)
  for split in ['train','valid']:
   h,r,t,removed,target=sample(g,rel,7,split,0,runner.device);hc=h.cpu();rc=None if removed is None else removed.cpu()
   ctr={lib:SparsePrefixGrounder(g,bodies,back,return_format='dense',policy='last_use',collect_stats=True) for lib,back in libs.items()}
   ctr['torch_sparse_gpu']=SparseGPUPrefixGrounder(gpu,bodies,return_format='dense',collect_stats=True)
   for name,c in ctr.items():c.begin(h if name=='torch_sparse_gpu' else hc,rel,removed if name=='torch_sparse_gpu' else rc)
   checked=0;nnz=0;maxcount=0
   for body in bodies:
    expected=runner.original_g(h,rel,list(body),removed).cpu()
    for name,c in ctr.items():
     y=c.grounding(h if name=='torch_sparse_gpu' else hc,rel,body,removed if name=='torch_sparse_gpu' else rc).cpu()
     assert torch.equal(y,expected),(ds,rel,split,body,name)
     checked+=1
    nnz+=int(expected.count_nonzero());maxcount=max(maxcount,int(expected.max()))
   stats={name:c.finish() for name,c in ctr.items()}
   # Independent prefix-by-prefix states for 24 distributed rules, no sharing.
   sample_bodies=bodies[::max(1,len(bodies)//24)][:24];steps=0
   for body in sample_bodies:
    states={lib:back.initial(hc) for lib,back in libs.items()};gx=gpu.initial(h)
    dense=torch.nn.functional.one_hot(h,g.entity_size).T.unsqueeze(-1).long()
    for rr in body:
     dense=runner.original_p(dense,rr,removed if rr==rel else None)
     for lib,back in libs.items():
      states[lib]=back.propagate(states[lib],rr,rc if rr==rel else None)
      assert torch.equal(back.to_dense(states[lib]),dense[:,:,0].T.cpu())
      steps+=1
     gx=gpu.propagate(gx,rr,removed if rr==rel else None)
     assert torch.equal(gpu.to_dense(gx),dense[:,:,0].T);steps+=1
   row={'dataset':ds,'relation':rel,'split':split,'batch':7,'rules':len(bodies),'terminal_tensor_checks':checked,'sampled_intermediate_tensor_checks':steps,'all_exact':True,'terminal_nnz':nnz,'terminal_elements':len(bodies)*7*g.entity_size,'max_terminal_count':maxcount,'cache_statistics':stats}
   out['cases'].append(row);(p/'real_sparse_correctness.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in row.items() if k!='cache_statistics'},flush=True)
  runner.close()
 out['complete']=True;(p/'real_sparse_correctness.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
