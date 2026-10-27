import sys,os,time,json,socket,cProfile,pstats,statistics,argparse
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT/'upstream/AdvancedIndexing'),str(ROOT/'upstream/Jacinle'),str(ROOT/'upstream/PDSketch')]
import torch,jacinle
import hacl.pdsketch as pds
import hacl.envs.gridworld.minigrid.minigrid_v20220407 as lib
from hacl.p.kfac.minigrid.models.model_abskin import ModelAbskin
from hacl.p.kfac.minigrid.models.model_full import ModelFull
from hacl.p.kfac.minigrid.data_generator import worker_offline
from hacl.pdsketch.interface.v2.strips.heuristics import StripsHFFHeuristic
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=500);p.add_argument('--eval-n',type=int,default=10);p.add_argument('--task',default='gotosingle');p.add_argument('--structure',default='abskin');a=p.parse_args()
torch.set_num_threads(1);jacinle.reset_global_seed(42)
args=SimpleNamespace(structure_mode=a.structure,task=a.task)
lib.set_domain_mode('envaction',a.structure);domain=lib.get_domain();model=(ModelFull if a.structure=='full' else ModelAbskin)(domain,goal_loss_weight=1.,action_loss_weight=1.)
env=lib.make(a.task);filt=lambda action:action.name in ['move','forward','lturn','rturn']
opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=5e-4)
ckpt=ROOT/f'model_{a.task}_{a.structure}_{a.steps}.pt';losses=[]
if ckpt.exists(): model.load_state_dict(torch.load(ckpt,weights_only=True))
else:
 model.train();t=time.perf_counter()
 for i in range(a.steps):
  n=0;loss=0
  while n<32:
   states,actions,dones,goal,succ,extra=worker_offline(args,domain,env,filt)
   ll,mon,out=model(feed_dict={'states':states,'actions':actions,'dones':dones,'goal_expr':goal},forward_augmented=True)
   loss+=ll*len(dones);n+=len(dones)
  loss=loss/n
  if loss.requires_grad: opt.zero_grad();loss.backward();opt.step()
  if i%50==0: losses.append([i,float(loss.detach())]);print('train',i,float(loss.detach()),time.perf_counter()-t,flush=True)
 torch.save(model.state_dict(),ckpt)
model.eval()
report={'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'task':a.task,'structure':a.structure,'training_steps':a.steps,'losses':losses,'scope':'learned PDSketch predicates and transitions + hFF search + environment execution; official model trained locally, not published checkpoint','cases':[]}
def plan(obs):
 return pds.heuristic_search_strips(domain,obs['state'],obs['mission'],max_depth=10,max_expansions=500,action_filter=filt,use_tuple_desc=False,use_quantized_state=False,forward_augmented=True,track_most_promising_trajectory=True,strips_heuristic='hff',strips_forward_relevance_analysis=False,strips_backward_relevance_analysis=True,strips_use_sas=False,use_strips_op=False,verbose=False,return_extra_info=True)
original=StripsHFFHeuristic.compute
for objects in ([1] if a.task=='gotosingle' else [4,8,16]):
 env.set_options(nr_objects=objects)
 for k in range(a.eval_n):
  seed=1000+k;jacinle.reset_global_seed(seed);obs=env.reset()
  with torch.no_grad():
   t=time.perf_counter();actions,info=plan(obs);elapsed=time.perf_counter()-t
  success=False
  if actions:
   for action in actions:
    _,_,done,_=env.step(action)
    if done: success=True;break
  measured={'objects':objects,'seed':seed,'e2e_search_ms':elapsed*1000,'success':success,'plan_length':None if actions is None else len(actions),'info':info}
  htimes=[]
  def wrapped(self,state):
   t=time.perf_counter();value=original(self,state);htimes.append(time.perf_counter()-t);return value
  StripsHFFHeuristic.compute=wrapped;jacinle.reset_global_seed(seed);obs=env.reset();pr=cProfile.Profile()
  with torch.no_grad():
   t=time.perf_counter();pr.enable();actions2,info2=plan(obs);pr.disable();diagnostic=time.perf_counter()-t
  StripsHFFHeuristic.compute=original
  measured.update({'diagnostic_ms':diagnostic*1000,'hff_ms':sum(htimes)*1000,'hff_calls':len(htimes),'hff_share_diagnostic':sum(htimes)/diagnostic,'same_plan':str(actions)==str(actions2)})
  if k==0:
   stats=pstats.Stats(pr);stats.sort_stats('cumulative');stats.dump_stats(str(ROOT/'results'/f'{a.task}_{a.structure}_{a.steps}_{objects}.pstats'))
   measured['top_cumulative']=[{'file':key[0],'line':key[1],'function':key[2],'calls':v[1],'self_s':v[2],'cum_s':v[3]} for key,v in sorted(stats.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:35]]
  report['cases'].append(measured);print(json.dumps(measured,default=str),flush=True)
  (ROOT/'results'/f'{a.task}_{a.structure}_{a.steps}.json').write_text(json.dumps(report,indent=2,default=str))
