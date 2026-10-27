from pathlib import Path
exec(compile(Path(__file__).with_name('bench_native.py').read_text().split('allrows=[]')[0],__file__,'exec'))
import cProfile,pstats,math
models={}
for name,sem in [('original',GraphSemiring),('native',InferenceScalarSemiring)]:
 m=Model(str(up/'src/deepproblog/examples/MNIST/models/addition.pl'),[Network(cnn,'mnist_net',batching=True)]);m.set_engine(ExactEngine(m),cache=True,semiring=sem);m.add_tensor_source('images',type('Source',(),{'__getitem__':lambda self,key:images[int(key[0])]})());m.eval();models[name]=m
rows=[];caches={}
with torch.no_grad():
 for trial in range(4):
  for n in [1,2]:
   vs=[Term('p'+str(i)) for i in range(2*n)];ids=list(range(trial*4,trial*4+2*n));term=Term('addition',*vs,Var('S')) if n==1 else Term('multi_addition',list2term(vs[:n]),list2term(vs[n:]),Var('S'));q=Query(term,{v:Term('tensor',Term('images',Constant(i))) for v,i in zip(vs,ids)});allp={};lat={}
   for name in (['original','native'] if trial%2 else ['native','original']):
    m=models[name];t=time.perf_counter();res=m.solve([q])[0];lat[name]=(time.perf_counter()-t)*1000;p=np.zeros(2*(10**n-1)+1)
    for key,val in res.result.items():p[int(key.args[-1])]=float(val)
    allp[name]=p;ac=m.solver.cache.get(q);cachekey=(name,n)
    if cachekey in caches:assert caches[cachekey]==id(ac)
    caches[cachekey]=id(ac)
   err=np.abs(allp['original']-allp['native']);assert err.max()<1e-5
   label=sum(ds[ids[j]][1]*10**(n-j-1) for j in range(n))+sum(ds[ids[n+j]][1]*10**(n-j-1) for j in range(n))
   rows.append({'trial':trial,'n':n,'cold':trial==0,'ms':lat,'max_abs':float(err.max()),'l1':float(err.sum()),'sums':{k:float(v.sum()) for k,v in allp.items()},'argmax':{k:int(v.argmax()) for k,v in allp.items()},'true_sum':label,'true_sum_probability':{k:float(v[label]) for k,v in allp.items()}})
 # Shared NN evidence must be counted once; outputs should be only even sums.
 v=Term('same');qshared=Query(Term('addition',v,v,Var('S')),{v:Term('tensor',Term('images',Constant(2)))})
 shared={}
 for name,m in models.items():
  r=m.solve([qshared])[0];p=np.zeros(19)
  for key,val in r.result.items():p[int(key.args[-1])]=float(val)
  ref=np.zeros(19);ref[::2]=cnn(images[2:3]).numpy()[0];assert np.max(np.abs(p-ref))<1e-5;shared[name]={'max_abs':float(np.max(np.abs(p-ref))),'odd_mass':float(p[1::2].sum())}
 pr=cProfile.Profile();pr.enable();models['native'].solve([q]);pr.disable()
with (b/'native_warm_profile.txt').open('w') as f:pstats.Stats(pr,stream=f).sort_stats('cumulative').print_stats(40)
stats=pr.getstats();names=[str(x.code) for x in stats];assert any('evaluate_standard' in x for x in names);assert not any('evaluate_custom' in x for x in names)
blocked=False
try:InferenceScalarSemiring(models['native'],{}, {})
except AssertionError:blocked=True
assert blocked
(b/'native_validation.json').write_text(json.dumps({'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'rows':rows,'shared_evidence':shared,'gradient_enabled_rejected':blocked,'native_standard_path_confirmed':True,'scope':'only tested MNIST addition inference; general evidence and trainable logic parameters not validated'},indent=2));print('validated',len(rows),'paired trials, shared evidence, backend dispatch, gradient guard',flush=True)
