import os,time,json,subprocess,socket,re
from pathlib import Path
b=Path(__file__).resolve().parent;rows=[]
for variant in ['baseline','diagnostic']:
 cmd=[str(b/('vampire_'+variant)),'-t','30','-i','0','--input_syntax','tptp','-sa','otter','--random_seed','5',str(b/'tptp/Problems/CSR/CSR188+1.p'),'-p','on']
 log=b/('symbolic_'+variant+'.log');t=time.perf_counter()
 with log.open('w') as f:p=subprocess.run(cmd,env=dict(os.environ,TPTP=str(b/'tptp'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,timeout=50)
 wall=time.perf_counter()-t;s=log.read_text();rows.append({'variant':variant,'command':cmd,'returncode':p.returncode,'wall_s':wall,'status':re.findall(r'SZS status ([^\n]+)',s),'stat_lines':[l for l in s.splitlines() if any(x in l.lower() for x in ['time elapsed','subsum','generated clauses','retained clauses','termination','nesy_forward'])]})
 (b/'symbolic_raw.json').write_text(json.dumps({'scope':'30-second symbolic-only control on official problem; no neural model; not substitute for failed neural pipeline','host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'rows':rows},indent=2));print(rows[-1],flush=True)
