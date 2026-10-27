import hashlib,json,os,sys,time
from pathlib import Path
p=Path(__file__).resolve().parent;files=[]
for root in [p,p.parent/'2026-10-14-rnnlogic-strong-baselines']:
 for q in root.glob('*.py'):
  data=q.read_bytes();files.append({'path':str(q),'sha256':hashlib.sha256(data).hexdigest(),'source':data.decode()})
(p/'source_snapshots').mkdir(exist_ok=True)
(p/'source_snapshots'/f'{time.time_ns()}_{Path(sys.argv[1]).stem}.json').write_text(json.dumps({'job':os.getenv('SLURM_JOB_ID'),'script':sys.argv[1],'files':files},indent=2))
