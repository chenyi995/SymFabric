import sys,os,ast,time,json,statistics,socket,subprocess
from pathlib import Path
b=Path(__file__).resolve().parent;prev=b.parent.parent/'2026-10-14-txt-first';up=b/'upstream/deepproblog'
sys.path[:0]=[str(b/'deps'),str(up/'src')]
import torch,torchvision
from problog.logic import Term,Var,Constant
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.query import Query
from deepproblog.engines import ExactEngine
torch.set_num_threads(1)
source=prev/'upstream/dolphin/experiments/mnist/sum_n.py';tree=ast.parse(source.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MNISTNet'];ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional};exec(compile(tree,str(source),'exec'),ns)
cnn=ns['MNISTNet']();cnn.load_state_dict(torch.load(prev/'results/dolphin_digit_checkpoint.pt',map_location='cpu',weights_only=True));cnn.eval()
ds=torchvision.datasets.MNIST(prev/'results/dolphin_mnist_data',train=False,download=False,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((.1307,),(.3081,))]))
images=torch.stack([ds[i][0] for i in range(64)])
net=Network(cnn,'mnist_net',batching=True);model=Model(str(up/'src/deepproblog/examples/MNIST/models/addition.pl'),[net]);model.set_engine(ExactEngine(model),cache=True);model.add_tensor_source('images',type('Source',(),{'__getitem__':lambda self,key:images[int(key[0])]})())
model.eval()
queries=[Query(Term('addition',Term('a'),Term('b'),Var('S')),{Term('a'):Term('tensor',Term('images',Constant(2*i))),Term('b'):Term('tensor',Term('images',Constant(2*i+1)))}) for i in range(32)]
rows=[]
with torch.inference_mode():
 for i,q in enumerate(queries):
  t=time.perf_counter();res=model.solve([q])[0];elapsed=(time.perf_counter()-t)*1000
  probs=torch.zeros(19)
  for term,p in res.result.items():probs[int(term.args[-1])]=float(p)
  p=cnn(images[2*i:2*i+2]);ref=torch.zeros(19)
  for j in range(10):ref[j:j+10]+=p[0,j]*p[1]
  err=float((probs-ref).abs().max());assert err<2e-6,(err,probs,ref)
  rows.append({'query':i,'e2e_ms':elapsed,'max_error':err,'argmax_correct':int(probs.argmax())==ds[2*i][1]+ds[2*i+1][1],'ground_time_ms':res.ground_time*1000,'compile_time_ms':res.compile_time*1000,'eval_time_ms':res.eval_time*1000})
 clean=[]
 for _ in range(5):
  t=time.perf_counter()
  for q in queries:model.solve([q])
  clean.append((time.perf_counter()-t)*1000/len(queries))
result={'scope':'official DeepProbLog addition; trained CNN and heldout MNIST; fixed query template and changing tensor substitutions; no synthetic graph churn','host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'commit':subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip(),'rows':rows,'warm_ms_per_query':clean,'note':'Result compile_time/ground_time are cached metadata; not repeated work. Count cache misses separately.'}
(b/'raw.json').write_text(json.dumps(result,indent=2));print('cold_ms',rows[0]['e2e_ms'],'warm_ms',statistics.median(clean),'max_error',max(r['max_error'] for r in rows),flush=True)
