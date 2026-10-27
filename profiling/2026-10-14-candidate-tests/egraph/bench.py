import os,json,time,re,subprocess,statistics,socket
from pathlib import Path
b=Path(__file__).resolve().parent;out=b/'results';out.mkdir(exist_ok=True)
rows=[]
for suite,name in [('math','diff_power_harder'),('math','math_ematching_bench'),('lambda','lambda_compose_many'),('lambda','lambda_ematching_bench')]:
 binary=next(p for p in (b/'build/release/deps').glob(suite+'-*') if p.suffix!='.d')
 for rep in range(6):
  start=time.perf_counter()
  try:
   p=subprocess.run([str(binary),name,'--exact','--nocapture','--test-threads=1'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60);txt=p.stdout;rc=p.returncode
  except subprocess.TimeoutExpired as e:txt=str(e.stdout);rc=-999
  r={'test':name,'rep':rep,'warmup':rep==0,'wall_s':time.perf_counter()-start,'exitcode':rc,'passed':rc==0 and '1 passed;' in txt}
  for label,key in [('Total time','total_s'),('Search','search_s'),('Apply','apply_s'),('Rebuild','rebuild_s')]:
   m=re.search(r'^\s*'+label+r':\s*(?:\([^)]*\)\s*)?([0-9.eE+-]+)',txt,re.M)
   if m:r[key]=float(m.group(1))
  for label,key in [('Iterations','iterations'),('Rebuilds','rebuilds')]:
   m=re.search(r'^\s*'+label+r':\s*(\d+)',txt,re.M)
   if m:r[key]=int(m.group(1))
  m=re.search(r'Egraph size:\s*(\d+)\s+nodes,\s*(\d+)\s+classes,\s*(\d+)\s+memo',txt)
  if m:r.update(zip(('nodes','classes','memo'),map(int,m.groups())))
  (out/f'{name}_{rep}.log').write_text(txt);rows.append(r);print(json.dumps(r),flush=True)
  (b/'raw.json').write_text(json.dumps({'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'scope':'official symbolic tests, unchanged Runner timers; no neural model','rows':rows},indent=2))
  if not r['passed']:break
