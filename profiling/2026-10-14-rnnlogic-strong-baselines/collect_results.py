"""Export measured rows without conflating different benchmark scopes."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FILES=['gpu_main.json','gpu_batches.json','cpu_main.json','cpu_passive.json','gpu_official_pna.json','gpu_bounded_lifetime.json','gpu_small_relation.json','gpu_prefix_schedule.json','cpu_prefix_schedule.json']
rows=[];completion={}
for filename in FILES:
 p=ROOT/filename
 if not p.exists():continue
 d=json.loads(p.read_text());completion[filename]=d.get('complete',False)
 for c in d['cases']:
  for v,r in c['variants'].items():
   cache=r.get('cache') or {};gpu=r.get('gpu_memory') or {}
   baseline=c['variants'].get('upstream',{}).get('forward',{}).get('median_ms')
   row={'file':filename,'device':d['device'],'dataset':c['dataset'],'relation':c['relation'],'split':c['split'],'batch':c['batch'],'threads':c['threads'],'model':d['config']['model'],'variant':v,'median_ms':r['forward']['median_ms'],'p95_ms':r['forward']['p95_ms'],'speedup_vs_upstream':baseline/r['forward']['median_ms'] if baseline else None,'cache_peak_MiB':cache.get('max_cache_bytes',0)/2**20 if cache else None,'propagations':cache.get('propagations'),'recomputes':cache.get('recomputes'),'incremental_gpu_peak_MiB':gpu.get('incremental_peak_bytes',0)/2**20 if gpu else None,'max_logit_error':r['max_logit_error']}
   rows.append(row)
with (ROOT/'results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
summary={'completed':completion,'rows':len(rows),'gpu_train_b32':[r for r in rows if r['file']=='gpu_main.json' and r['split']=='train'],'cpu_best':[],'cpu_default_passive_comparison':[]}
for dataset,relation in [('FB15k-237',272),('wn18rr',12)]:
 for split in ['train','valid']:
  for variant in ['upstream','torch_all','native_all','native_last','native_lru32']:
   candidates=[r for r in rows if r['device']=='cpu' and r['dataset']==dataset and r['relation']==relation and r['split']==split and r['variant']==variant]
   if candidates:summary['cpu_best'].append(min(candidates,key=lambda r:r['median_ms']))
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'completed':completion,'rows':len(rows)},indent=2))
