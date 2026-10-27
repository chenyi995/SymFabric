#!/usr/bin/env python3
import os,time,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
t=time.monotonic()
while time.monotonic()-t<1.0: sum(range(10000))
rows=[]
for name,k in [('hwf_3_controlled.scl',4),('hwf_5_controlled.scl',8),('path_dag20_controlled.scl',8)]:
 for pair in range(3):
  for label in ['original','cached','cached','original']:
   env=os.environ.copy();env['SCALLOP_CACHE_SCORES']='1' if label=='cached' else '0';env['SCALLOP_PROFILE']='0'
   binary=R/'bin'/('c1-e2e-original' if label=='original' else 'c1-e2e-profile')
   out=subprocess.run([str(binary),str(R/'fixtures'/name),'result',str(k),'7'],env=env,capture_output=True,text=True,check=True,timeout=15)
   for line in out.stdout.splitlines():
    row=json.loads(line);row.update(label=label,pair=pair,affinity=sorted(os.sched_getaffinity(0)));rows.append(row)
 (R/'results/scallop_paired.json').write_text(json.dumps(rows,indent=2))
 print(name,k,'ABBA complete',flush=True)
