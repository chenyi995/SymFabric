#!/usr/bin/env python3
"""Paired same-output controls: per-call SDD eval memo and GC visited dedup."""
import os,time,json,subprocess,random
from pathlib import Path
R=Path(__file__).resolve().parents[1]
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
t=time.monotonic()
while time.monotonic()-t<1: sum(range(10000))
binary=R/'bin/c1-e2e-wmc'
rows=[];details=[];rng=random.Random(20261014)
for name,k in [('hwf_3_controlled.scl',4),('hwf_5_controlled.scl',8),('path_dag20_controlled.scl',8)]:
 variants=[('original',False,False),('memo_eval',True,False),('memo_gc',False,True),('memo_both',True,True)]
 for trial in range(3):
  rng.shuffle(variants)
  for label,ev,gc in variants:
   env=os.environ.copy();env.update(SCALLOP_MEMO_EVAL=str(int(ev)),SCALLOP_MEMO_GC=str(int(gc)),SCALLOP_CACHE_SCORES='0',SCALLOP_PROFILE='0',SCALLOP_WMC_PROFILE='0')
   cmd=[str(binary),str(R/'fixtures'/name),'result',str(k),'5']
   proc=subprocess.run(cmd,env=env,capture_output=True,text=True,check=True,timeout=20)
   for line in proc.stdout.splitlines():
    row=json.loads(line);row.update(label=label,trial=trial);rows.append(row)
 for label,ev,gc in [('original',False,False),('memo_both',True,True)]:
  env=os.environ.copy();env.update(SCALLOP_MEMO_EVAL=str(int(ev)),SCALLOP_MEMO_GC=str(int(gc)),SCALLOP_CACHE_SCORES='0',SCALLOP_PROFILE='0',SCALLOP_WMC_PROFILE='1')
  proc=subprocess.run([str(binary),str(R/'fixtures'/name),'result',str(k),'1'],env=env,capture_output=True,text=True,check=True,timeout=20)
  details.append({'file':name,'k':k,'label':label,'stderr':proc.stderr,'stdout':proc.stdout})
 same=[x for x in rows if x['file'].endswith(name) and x['k']==k]
 assert len({x['output_hash'] for x in same})==1,'optimized output mismatch'
 (R/'results/scallop_wmc_controls.json').write_text(json.dumps(rows,indent=2))
 (R/'results/scallop_wmc_diagnostic.json').write_text(json.dumps(details,indent=2))
 print(name,k,'all output hashes match; WMC controls done',flush=True)
