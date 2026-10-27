import sys,time,json,statistics,os,socket
from pathlib import Path
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'deps'))
import torch,pyro
import pyro.distributions as dist
from pyro import poutine
torch.set_num_threads(1);pyro.set_rng_seed(23)
net=torch.nn.Sequential(torch.nn.Linear(2,12),torch.nn.Tanh(),torch.nn.Linear(12,4))
x=torch.randn(256,2);W=torch.tensor([[.8,-.3],[.1,.4],[-.4,.5],[.2,-.6]])
target=x@W.T;opt=torch.optim.Adam(net.parameters(),lr=.02)
for _ in range(200):
 opt.zero_grad();loss=(net(x)-target).square().mean();loss.backward();opt.step()
net.eval();test=torch.randn(128,2)
with torch.no_grad():mse=float((net(test)-test@W.T).square().mean())
def model(prior,mu,y):
 k=pyro.sample('n',dist.Categorical(probs=prior))+1;total=mu*0
 for j in range(int(k)):total=total+pyro.sample('x_'+str(j),dist.Normal(mu+.05*j,.7))
 pyro.sample('obs',dist.Normal(total,.4),obs=y)
 return total
def draw(prior,mu,y):return poutine.trace(model).get_trace(prior,mu,y)
with torch.no_grad():prior=net(torch.tensor([[.2,-.15]])).softmax(-1)[0]
mu=torch.tensor(.15);y=torch.tensor(.7);t=draw(prior,mu,y);t.compute_log_prob()
r=poutine.trace(poutine.replay(model,trace=t)).get_trace(prior,mu,y)
assert torch.equal(t.nodes['_RETURN']['value'],r.nodes['_RETURN']['value'])
for k in [1,4]:
 q=poutine.trace(poutine.condition(model,data={'n':torch.tensor(k-1)})).get_trace(prior,mu,y)
 assert sum(name.startswith('x_') for name in q.nodes)==k
ks=torch.arange(1,5,dtype=torch.float32)
exact=(prior.log()+dist.Normal(ks*mu+.05*ks*(ks-1)/2,torch.sqrt(ks*.49+.16)).log_prob(y)).softmax(0)
manual=sum(site['fn'].log_prob(site['value']).sum() for site in t.nodes.values() if site['type']=='sample')
assert torch.allclose(t.log_prob_sum(),manual)
rounds=[]
with torch.no_grad():
 for _ in range(20):draw(prior,mu,y)
 for trial in range(3):
  start=time.perf_counter_ns();prior=net(torch.tensor([[.2,-.15]])).softmax(-1)[0];after_nn=time.perf_counter_ns()
  labels=[];weights=[];timings=[];churn=[];last=None
  for i in range(2048):
   a=time.perf_counter_ns();tr=draw(prior,mu,y);b=time.perf_counter_ns();tr.compute_log_prob();c=time.perf_counter_ns()
   sites={name for name in tr.nodes if name.startswith('x_')}
   if last is not None:churn.append(len(sites.symmetric_difference(last)))
   last=sites;labels.append(int(tr.nodes['n']['value']));weights.append(tr.nodes['obs']['log_prob']);timings.append([(b-a)/1e6,(c-b)/1e6])
  reduce_start=time.perf_counter_ns();w=torch.stack(weights).softmax(0);estimate=torch.zeros(4).scatter_add_(0,torch.tensor(labels),w)
  ess=float(1/w.square().sum());finish=time.perf_counter_ns();err=float((estimate-exact).abs().max());assert err<.12
  rounds.append(dict(trial=trial,samples=2048,e2e_ms=(finish-start)/1e6,nn_ms=(after_nn-start)/1e6,reduce_ms=(finish-reduce_start)/1e6,ESS=ess,ESS_per_s=ess/((finish-start)/1e9),posterior=estimate.tolist(),exact=exact.tolist(),max_abs_error=err,support_churn_fraction=sum(v>0 for v in churn)/len(churn),mean_added_deleted_sites=statistics.mean(churn),trace_and_logprob_ms=timings))
 analytic=[]
 for _ in range(100):
  a=time.perf_counter_ns();p=net(torch.tensor([[.2,-.15]])).softmax(-1)[0];ans=(p.log()+dist.Normal(ks*mu+.05*ks*(ks-1)/2,torch.sqrt(ks*.49+.16)).log_prob(y)).softmax(0);analytic.append((time.perf_counter_ns()-a)/1e6)
assert torch.allclose(ans,exact,atol=1e-6)
out=dict(scope='synthetic trained-neural-prior variable-support Pyro program; importance sampling, not incremental MCMC or production E2E',host=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'),torch=torch.__version__,pyro=pyro.__version__,threads=1,heldout_prior_logit_MSE=mse,semantic_checks=['replay equality','conditioned support count','joint score equality'],analytic_e2e_ms=analytic,rounds=rounds)
(root/'raw.json').write_text(json.dumps(out,indent=2));print('mse',mse,'analytic_ms',statistics.median(analytic),'sampling_ms',[r['e2e_ms'] for r in rounds],flush=True)
