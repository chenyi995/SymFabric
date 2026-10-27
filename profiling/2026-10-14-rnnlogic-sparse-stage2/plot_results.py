import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
d=json.loads((root/'cached_sparse_real.json').read_text());assert d['complete']
rows=[c for c in d['cases'] if c['split']=='train']
variants=['upstream','graph','native64gpu_512','cached_gpu_csr']
labels=['Upstream','GPU + CUDA Graph','Native CPU + sparse GPU','Preloaded GPU CSR']
colors=['#82909e','#4179bb','#408e79','#db8b35']
fig,axes=plt.subplots(1,2,figsize=(13,5.1))
for ax,kind,title in zip(axes,['forward','training'],['Complete resident predictor forward','Sampled training batch (transfers + backward + Adam)']):
    for i,c in enumerate(rows):
        g=next(g for g in c['mixed_groups'] if g['batch']==32)
        for j,v in enumerate(variants):
            s=g['forward'][v] if kind=='forward' else g['training_batch']['variants'][v]['timing']
            x=i+(j-1.5)*.185
            ax.bar(x,s['median_ms'],width=.16,color=colors[j],label=labels[j] if i==0 else None)
            ax.scatter([x]*len(s['samples_ms']),s['samples_ms'],color='black',alpha=.35,s=8,zorder=3)
            ax.text(x,s['median_ms']*1.18,f"{s['median_ms']:.2f}",ha='center',fontsize=8)
    ax.set_xticks([0,1],['FB15k-237 / relation 272','WN18RR / relation 12'])
    ax.set_yscale('log');ax.set_ylim(1,700);ax.set_ylabel('Wall time (ms; log scale)')
    ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
handles,labs=axes[0].get_legend_handles_labels()
fig.legend(handles,labs,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.93))
fig.suptitle('Real RTX A5000 + 16 allocated EPYC cores | matched B32 mixed queries | LSTM / sum',fontsize=12,y=.99)
fig.text(.5,.04,'Bars: medians of 5 interleaved runs; dots: individual runs. Cached results exclude explicit cold build.\n64 training keys: FB 84.04 MiB / 2080.61 ms build; WN 0.890 MiB / 63.44 ms build. Not full epoch / EM.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.12,1,.84]);fig.savefig(root/'real_hardware_comparison.png',dpi=180);fig.savefig(root/'real_hardware_comparison.pdf')
