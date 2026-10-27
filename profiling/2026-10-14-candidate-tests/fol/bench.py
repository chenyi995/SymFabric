import os,sys,time,json,subprocess,socket,re
from pathlib import Path
b=Path(__file__).resolve().parent
variant=sys.argv[1] if len(sys.argv)>1 else 'baseline'
model=b/'upstream/materials/tptpOverfit100/models/loop5/script-model.pt'
cmd=[str(b/('vampire_'+variant)),'-t','300','-i','100000','--input_syntax','tptp','-sa','otter','-ncem',str(model),'-npcc','on','-npcct','0.333','-si','on','-rtra','on','--random_seed','5',str(b/'tptp/Problems/CSR/CSR188+1.p'),'-p','on']
env=dict(os.environ,TPTP=str(b/'tptp'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
rows=[]
for i in range(3 if variant=='baseline' else 1):
 log=b/(variant+'_'+str(i)+'.log');t=time.perf_counter()
 with log.open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=330)
 wall=time.perf_counter()-t;s=log.read_text();statuses=re.findall(r'SZS status ([^\n]+)',s)
 row={'trial':i,'wall_s':wall,'returncode':p.returncode,'szs_status':statuses,'log':str(log),'stat_lines':[l for l in s.splitlines() if any(x in l.lower() for x in ['time elapsed','subsum','generated clauses','retained clauses','gnn','bulk'])]};rows.append(row)
 (b/(variant+'_raw.json')).write_text(json.dumps({'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'command':cmd,'scope':'official neural-guided Vampire; published pretrained model and CSR188+1 problem; full process time includes model load','rows':rows},indent=2));print(json.dumps(row),flush=True)
 if p.returncode or not statuses:break
