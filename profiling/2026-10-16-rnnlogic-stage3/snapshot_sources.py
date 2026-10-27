import sys, pathlib, hashlib, json, time, os
root=pathlib.Path(__file__).resolve().parent
sources={}
for directory in (root,root.parent/'2026-10-14-rnnlogic-sparse-stage2',root.parent/'2026-10-14-rnnlogic-strong-baselines'):
 for pattern in ('*.py','*.cpp','*.sh'):
  for p in directory.glob(pattern): sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
out=root/'results'/('sources_'+str(time.time_ns())+'.json')
out.write_text(json.dumps({'entrypoint':sys.argv[1:],'job':os.getenv('SLURM_JOB_ID'),'sha256':sources},indent=2))
