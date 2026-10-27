"""Combine disjoint matched relation screens while retaining hardware provenance."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
sources=[];rows={}
for name in ['screen_fb.json','screen_fb_remaining.json']:
 p=ROOT/'results'/name
 if not p.exists():continue
 x=json.loads(p.read_text());sources.append({'file':name,'complete':x['complete'],'host':x['host'],'job':x['job'],'gpu':x['gpu'],'config':x['config'],'model':x['model']})
 for c in x['cases']:
  if not any('forward' in d for d in c['variants'].values()):continue
  key=(c['relation'],c['group']);assert key not in rows,'overlapping completed cases'
  c['measurement_source']={k:sources[-1][k] for k in ['file','host','job','gpu']};rows[key]=c
if not sources:raise SystemExit('no completed source')
assert len({s['model']['checkpoint_sha256'] for s in sources})==1
assert all(s['model']['native_kind']=='spa' and s['config']['dataset']=='FB15k-237' and s['config']['repeats']==3 and s['config']['warmup']==2 for s in sources)
result={'complete':set(rows)=={(r,0) for r in range(474)},'scope':'One matched actual loader batch per FB relation; separate allocation provenance retained per row. Do not aggregate absolute times across different hardware as one measured dataset traversal.','sources':sources,'all_same_host_gpu':len({(s['host'],s['gpu']) for s in sources})==1,'integer_terminals_all_equal':all(c.get('integer_terminals_equal',False) and c.get('gpu_native_integer_terminals_equal',False) for c in rows.values()),'cases':[rows[k] for k in sorted(rows)]}
p=ROOT/'results/screen_fb_combined.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(p)
print(json.dumps({'complete':result['complete'],'relations':len(rows),'integer_terminals_all_equal':result['integer_terminals_all_equal']}))
