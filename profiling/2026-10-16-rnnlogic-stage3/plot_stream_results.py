"""Standalone scientific figure from completed actual query replay results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
u=json.loads((ROOT/'results/stream_wn_uncached.json').read_text())
c=json.loads((ROOT/'results/stream_wn_cache8g.json').read_text())
assert u['complete'] and c['complete'] and len(u['epochs'])==len(c['epochs'])==2
assert u['job']==c['job'] and u['gpu']==c['gpu']
for a,b in zip(u['epochs'],c['epochs']):
 assert a['epoch']==b['epoch'] and a['counter']['queries']==b['counter']['queries']==173670
fig,ax=plt.subplots(1,2,figsize=(11,5.6),gridspec_kw={'width_ratios':[1.35,1]})
fig.patch.set_facecolor('#f7f8fb')
x=np.arange(2);width=.34
ut=np.array([e['batch_forward_sum_seconds'] for e in u['epochs']]);ct=np.array([e['batch_forward_sum_seconds'] for e in c['epochs']])
a=ax[0].bar(x-width/2,ut,width,color='#345b87',label='Strong uncached dispatch')
b=ax[0].bar(x+width/2,ct,width,color='#df8b37',label='8 GiB query-count cache')
for bars in (a,b):ax[0].bar_label(bars,fmt='%.1f s',padding=5,fontsize=10)
ax[0].set_xticks(x,['Pass 1: cold cache','Pass 2: reused counts']);ax[0].set_ylabel('Sum of measured batch times (seconds)');ax[0].set_ylim(0,max(ut.max(),ct.max())*1.22)
ax[0].legend(loc='upper right',frameon=False,fontsize=9);ax[0].set_title('Same actual shuffled query streams',loc='left',fontweight='bold')
for i,(v,w) in enumerate(zip(ut,ct)):ax[0].text(i,max(v,w)+22,('~ same time' if abs(v/w-1)<.02 else f'{v/w:.2f}x uncached/cache'),ha='center',fontsize=10)
phases=[]
for e in c['epochs']:
 p=e['phase_seconds'];ground=p.get('miss_grounding_and_terminal_placement_seconds',0);admit=p.get('admission_seconds',0)
 phases.append([ground,admit,e['batch_forward_sum_seconds']-ground-admit])
values=np.array(phases);bottom=np.zeros(2)
for j,(label,color) in enumerate([('Miss counts + placement','#9b4d4d'),('Cache packing + admission','#e8bd79'),('Forward + input + lookup','#447f78')]):
 ax[1].bar(x,values[:,j],.52,bottom=bottom,label=label,color=color);bottom+=values[:,j]
ax[1].set_xticks(x,['Pass 1','Pass 2']);ax[1].set_ylabel('Cache mode measured seconds');ax[1].set_title('Cold construction is charged',loc='left',fontweight='bold');ax[1].legend(frameon=False,fontsize=9,loc='upper right');ax[1].set_ylim(0,ct.max()*1.25)
for i,e in enumerate(c['epochs']):
 q=e['counter'];hits=q.get('hits',0)/max(1,q.get('hits',0)+q.get('misses',0));ax[1].text(i,ct[i]+ct.max()*.03,f'{hits:.0%} hits',ha='center')
for a in ax:
 a.spines[['top','right']].set_visible(False);a.set_axisbelow(True);a.grid(axis='y',alpha=.18)
fig.suptitle('RNNLogic WN18RR: complete frozen-forward replay',x=.075,ha='left',fontsize=16,fontweight='bold')
fig.text(.075,.89,'RTX A5000 + 16 EPYC 7352 cores | official emb/PNA, tile 2048 | 173,670 queries per pass',fontsize=10,color='#444')
fig.text(.075,.055,'One measured traversal per mode and pass; no whole-run confidence interval. Includes query assembly/H2D,\nmiss counting, cache admission/eviction, forward, synchronization. Excludes backward/Adam and explicit oracles.\nCount payload: 5.206 GiB. Results use a short-fit checkpoint; no convergence or accuracy claim.',fontsize=8.5,color='#444',linespacing=1.5)
fig.subplots_adjust(left=.075,right=.975,bottom=.24,top=.79,wspace=.32)
for suffix in ('png','pdf'):fig.savefig(ROOT/f'wn_full_stream_results.{suffix}',dpi=180,facecolor=fig.get_facecolor())
print('created wn_full_stream_results.png/.pdf')
