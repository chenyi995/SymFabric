import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from sparse_gpu_backend import SparseGPUBackend,SparseGPUPrefixGrounder
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p.parent/'2026-10-14-rnnlogic-strong-baselines'))
from gpu_backend import _reference

def main():
 torch.set_num_threads(1);torch.manual_seed(42)
 n=17;adj=[torch.randint(n,(2,43)),torch.empty(2,0,dtype=torch.long),torch.randint(n,(2,64))]
 adj[0][:,1]=adj[0][:,0]
 g=SimpleNamespace(entity_size=n,relation2adjacency=[(a,torch.ones(a.shape[1])) for a in adj])
 backend=SparseGPUBackend(g)
 passed=0
 for b in [1,7,32]:
  h=torch.randint(n,(b,),device='cuda');removed=torch.randint(43,(b,),device='cuda')
  initial=backend.initial(h)
  expected=torch.nn.functional.one_hot(h,n).T.unsqueeze(-1).long()
  assert torch.equal(backend.to_dense(initial),expected[:,:,0].T)
  bodies=[(),(0,),(0,0),(0,2),(2,0,0),(1,),(1,0),(0,2)]
  for train in [False,True]:
   rm=removed if train else None
   for body in bodies:
    x=initial;y=expected
    for r in body:
     x=backend.propagate(x,r,rm if r==0 else None)
     y=_reference(y,adj[r],rm if r==0 else None)
     assert torch.equal(backend.to_dense(x),y[:,:,0].T)
     passed+=1
   for fmt in ['dense','sparse']:
    ctr=SparseGPUPrefixGrounder(backend,bodies,fmt,True);ctr.begin(h,0,rm)
    for body in bodies:
     out=ctr.grounding(h,0,body,rm);y=expected
     for r in body:y=_reference(y,adj[r],rm if r==0 else None)
     if fmt=='sparse':
      ii,v=out;out=torch.zeros(b*n,device='cuda',dtype=torch.long);out[ii]=v;out=out.view(b,n)
     assert torch.equal(out,y[:,:,0].T);passed+=1
    assert ctr.finish()['complete']
 support={}
 a=torch.tensor([[1,0],[2,3]],device='cuda',dtype=torch.int64).to_sparse_csr()
 for fmt in ['csr','coo']:
  try:
   aa=a if fmt=='csr' else a.to_sparse_coo();z=torch.sparse.mm(aa,torch.eye(2,device='cuda',dtype=torch.long));support[fmt]={'supported':True,'result':z.cpu().tolist()}
  except Exception as e:support[fmt]={'supported':False,'error':str(e)}
 out={'complete':True,'passed_tensor_comparisons':passed,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'int64_sparse_mm_probe':support,'metadata':backend.metadata}
 (p/'sparse_gpu_correctness.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
