"""Standalone scientific figure from recorded measurements (no fitted data)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
load=lambda name:json.loads((ROOT/name).read_text())
main=load('gpu_main.json');sched=load('gpu_prefix_schedule.json');cg=load('cuda_graph.json')
assert main.get('complete') and sched.get('complete') and cg.get('complete')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(16,4.8))
colors=['#5A7184','#D99527','#23856D']
cases=[c for c in main['cases'] if c['split']=='train']
x=np.arange(len(cases));width=.24
for offset,(key,label,color) in enumerate(zip(['upstream','torch_all','csr_last'],['Upstream','Resident + prefix + GEMM','Fused CSR + last-use'],colors)):
 values=[c['variants'][key]['forward']['median_ms'] for c in cases]
 bars=axes[0].bar(x+(offset-1)*width,values,width,label=label,color=color)
 axes[0].bar_label(bars,fmt='%.0f',fontsize=8,padding=2)
axes[0].set_xticks(x,[('FB' if c['dataset']=='FB15k-237' else 'WN')+str(c['relation']) for c in cases])
axes[0].set_ylabel('Synchronized forward latency (ms)')
axes[0].set_title('(a) Measured eager GPU forwards')
axes[0].legend(fontsize=8)
labels=['FB272','WN12'];pairs=[('FB15k-237',272),('wn18rr',12)];x=np.arange(2)
cachevals=[[],[],[]]
for ds,r in pairs:
 c=next(c for c in cases if c['dataset']==ds and c['relation']==r)
 s=next(c for c in sched['cases'] if c['dataset']==ds and c['relation']==r and c['split']=='train')
 for target,row in zip(cachevals,[c['variants']['csr_all'],c['variants']['csr_last'],s['variants']['csr_last']]):
  target.append(row['cache']['max_cache_bytes']/2**20)
for i,(values,label,color) in enumerate(zip(cachevals,['Retain all prefixes','Last-use, original order','Last-use, prefix order'],colors)):
 bars=axes[1].bar(x+(i-1)*width,values,width,label=label,color=color)
 axes[1].bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
axes[1].set_xticks(x,labels);axes[1].set_yscale('log');axes[1].set_ylim(1,20000)
axes[1].set_ylabel('Logical prefix-cache peak (MiB, log scale)')
axes[1].set_title('(b) Software state management')
axes[1].legend(fontsize=8)
groups=[(c,g) for c in cg['cases'] for g in c.get('query_groups',[])]
if groups:
 x=np.arange(len(groups));width=.34
 for i,(key,label,color) in enumerate(zip(['csr_last','captured_csr_last'],['Eager CSR + last-use','Grounding CUDA Graph'],[colors[0],colors[2]])):
  vals=[g['variants'][key]['forward']['median_ms'] for c,g in groups]
  bars=axes[2].bar(x+(i-.5)*width,vals,width,label=label,color=color)
  axes[2].bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
 axes[2].set_xticks(x,[('FB' if c['dataset']=='FB15k-237' else 'WN')+str(c['relation'])+'\nquery group '+str(g['group_id']) for c,g in groups],fontsize=8)
 axes[2].set_ylabel('Synchronized forward latency (ms)')
 axes[2].legend(fontsize=8)
else:
 axes[2].text(.5,.5,'CUDA Graph capture failed;\nsee recorded exceptions.',ha='center',va='center',transform=axes[2].transAxes)
axes[2].set_title('(c) Existing CUDA Graph software control')
fig.suptitle('RNNLogic on real RTX A5000: B32, fixed LSTM/sum model',fontsize=14)
fig.text(.5,.015,'Medians of 5 timed repeats. Forward scope excludes setup and full EM. Panel (c) uses two different real query groups; graph persistent memory is reported separately.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.06,1,.94])
fig.savefig(ROOT/'results_overview.pdf',bbox_inches='tight')
fig.savefig(ROOT/'results_overview.png',dpi=180,bbox_inches='tight')
