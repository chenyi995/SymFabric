#!/usr/bin/env python3
"""Run original/optimized symbolic-program probes. These exclude the neural model."""
import argparse, json, os, subprocess, platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--binary',default='/tmp/scallop-txt-target/release/c1-e2e-profile'); p.add_argument('--label',default='original'); p.add_argument('--reps',type=int,default=7); p.add_argument('--profile',action='store_true'); p.add_argument('--cached-score',action='store_true'); a=p.parse_args()
fixtures=[ROOT/'upstream/scallop/experiments/hwf/examples/hwf_length_3_top_3_prob.scl',ROOT/'fixtures/hwf_3_controlled.scl',ROOT/'fixtures/hwf_5_controlled.scl',ROOT/'fixtures/path_dag20_controlled.scl']
env=os.environ.copy(); env['SCALLOP_PROFILE']='1' if a.profile else '0'; env['SCALLOP_CACHE_SCORES']='1' if a.cached_score else '0'
result_path=ROOT/'results'/f'scallop_{a.label}.json'
detail_path=ROOT/'results'/f'scallop_{a.label}_detail.json'
rows=json.loads(result_path.read_text()) if result_path.exists() else []
details=json.loads(detail_path.read_text()) if detail_path.exists() else []
done={(r['file'],r['k']) for r in rows}

for path in fixtures:
    for k in [1,4,8,16]:
        if (str(path),k) in done: continue
        if path.name=='hwf_5_controlled.scl' and k==16: continue  # original >90s timeout; documented separately
        cmd=[a.binary,str(path),'result',str(k),str(a.reps)]
        try:
            proc=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=35)
        except subprocess.TimeoutExpired as err:
            details.append({'command':cmd,'status':'timeout','budget_seconds':35,'partial_stdout':(err.stdout or b'').decode() if isinstance(err.stdout, bytes) else err.stdout,'stderr':(err.stderr or b'').decode() if isinstance(err.stderr, bytes) else err.stderr})
            detail_path.write_text(json.dumps(details,indent=2))
            print(path.name,k,'timeout_35s',flush=True)
            continue
        if proc.returncode: raise RuntimeError(proc.stderr)
        new=[json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')]
        for row in new: row['label']=a.label; row['host']=platform.node()
        rows.extend(new)
        details.append({'command':cmd,'stderr':proc.stderr})
        (ROOT/'results'/f'scallop_{a.label}.json').write_text(json.dumps(rows,indent=2))
        (ROOT/'results'/f'scallop_{a.label}_detail.json').write_text(json.dumps(details,indent=2))
        print(path.name,k,len(new),flush=True)
