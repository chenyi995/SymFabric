import sys,time,json,cProfile,pstats,contextlib,os,platform,statistics,subprocess
from pathlib import Path
b=Path(__file__).resolve().parent; u=b/'upstream/alphageometry2';sys.path.insert(0,str(u))
from test import problems_without_aux,problems_with_aux
from ddar import DDAR
from parse import AGProblem
cases={**problems_without_aux,**problems_with_aux}
names=['2000_p1','2005_p5','2022_p4','2005_p1','2008_p6','2013_p3']
def solve(s):
 p=AGProblem.parse(s); d=DDAR(p.points)
 for pred in p.preds:d.force_pred(pred)
 d.deduction_closure()
 return bool(d.check_pred(p.goal))
result={'scope':'official DDAR symbolic core; supplied auxiliary points; no neural generation','host':platform.node(),'job':os.getenv('SLURM_JOB_ID'),'affinity':sorted(os.sched_getaffinity(0)),'commit':subprocess.check_output(['git','-C',str(u),'rev-parse','HEAD'],text=True).strip(),'cases':[]}
for name in names:
 row={'name':name,'with_supplied_aux':name in problems_with_aux,'wall_s':[],'ok':[]}
 for rep in range(3):
  with (b/(name+'.zero_hoist.baseline'+str(rep)+'.log')).open('w') as f,contextlib.redirect_stdout(f):
   t=time.perf_counter();ok=solve(cases[name]);dt=time.perf_counter()-t
  row['wall_s'].append(dt);row['ok'].append(ok)
 row['median_s']=statistics.median(row['wall_s']);pr=cProfile.Profile()
 with (b/(name+'.zero_hoist.diagnostic.log')).open('w') as f,contextlib.redirect_stdout(f):
  pr.enable();ok=solve(cases[name]);pr.disable()
 pr.dump_stats(str(b/(name+'.zero_hoist.pstats')));st=pstats.Stats(pr);rows=[]
 for (file,line,fn),(cc,nc,tt,ct,callers) in st.stats.items():
  rows.append({'file':file,'line':line,'function':fn,'calls':nc,'self_s':tt,'cumulative_s':ct})
 row['diagnostic']={'ok':ok,'total_s':st.total_tt,'top_self':sorted(rows,key=lambda x:-x['self_s'])[:40],'top_cumulative':sorted(rows,key=lambda x:-x['cumulative_s'])[:40]}
 result['cases'].append(row);(b/'results_zero_hoist.json').write_text(json.dumps(result,indent=2));print(name,row['median_s'],all(row['ok']),flush=True)
assert all(all(x['ok']) and x['diagnostic']['ok'] for x in result['cases'])
