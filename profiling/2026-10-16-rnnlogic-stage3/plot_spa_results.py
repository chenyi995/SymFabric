from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
result=json.loads((root/'results/spa_native_real.json').read_text())
assert result['complete']
groups={}
for row in result['cases']:groups.setdefault((row['dataset'],row['relation']),[]).append(row)
fig,(left,right)=plt.subplots(1,2,figsize=(11,6),gridspec_kw={'width_ratios':[1.45,1]},sharey=True)
labels=[]
for y,((dataset,relation),rows) in enumerate(groups.items()):
 labels.append(('FB' if dataset=='FB15k-237' else 'WN')+f' r{relation}')
 for name,color,offset in [('stdmap','#69788b',-.13),('spa','#147d92',.13)]:
  values=[row['timing'][name]['median_ms'] for row in rows]
  left.plot([min(values),max(values)],[y+offset]*2,color=color,linewidth=2,alpha=.6)
  left.scatter(values,[y+offset]*len(values),color=color,s=24,label={'stdmap':'Original std::map','spa':'Local array accumulator'}[name] if y==0 else None,zorder=3)
 ratios=[row['spa_speedup'] for row in rows]
 right.plot([min(ratios),max(ratios)],[y,y],color='#147d92',linewidth=2)
 right.scatter(ratios,[y]*len(ratios),color='#147d92',s=24,zorder=3)
left.set_yticks(range(len(labels)),labels);left.invert_yaxis();left.set_xscale('log')
left.set_xlabel('Count construction per B32 batch (ms, log scale)');left.legend(loc='lower right',frameon=False,fontsize=9)
right.axvline(1,color='#9aa4af',linestyle='--');right.set_xlim(0,13);right.set_xlabel('Matched count-stage speedup (x)')
for axis in (left,right):
 axis.grid(axis='x',alpha=.18);axis.set_axisbelow(True);axis.spines[['top','right']].set_visible(False)
fig.suptitle('A standard CPU accumulator removes substantial RNNLogic counting cost',fontsize=13,x=.52,y=.98)
fig.text(.5,.925,'16 physical cores on Xeon Gold 6338 | 3 actual batches per relation | median of 5 interleaved repeats',ha='center',fontsize=9,color='#505967')
fig.text(.5,.025,'Each dot is a different query batch. This is count construction, not model E2E; selected relations are not dataset-weighted.\n10,308 rule terminal tensors matched exactly. GPU comparisons are reported separately.',ha='center',fontsize=8.5,color='#505967')
fig.subplots_adjust(left=.11,right=.97,top=.87,bottom=.16,wspace=.12)
for extension in ('png','pdf'):fig.savefig(root/f'spa_count_results.{extension}',dpi=180)
print(root/'spa_count_results.png')
