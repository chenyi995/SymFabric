#!/usr/bin/env python3
"""Rebuild measured summaries and a scope-labelled figure from saved raw results."""
from pathlib import Path
import collections, json, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((R / name).read_text())
groups = collections.defaultdict(list)
for r in read('results/scallop_wmc_controls.json'):
    groups[(Path(r['file']).name, r['k'], r['label'])].append(r)
wmc=[]
for (file,k,label),rows in sorted(groups.items()):
    hashes = {r['output_hash'] for r in rows}
    assert len(hashes)==1
    entry={'file':file,'k':k,'variant':label,'samples':len(rows),'output_hash':next(iter(hashes))}
    for field in ['load_ms','compile_ms','run_ms','recover_ms','symbolic_online_ms']:
        vals=[r[field] for r in rows]
        entry[field]={'median':statistics.median(vals),'min':min(vals),'max':max(vals)}
    wmc.append(entry)
for file in {r['file'] for r in wmc}:
    assert len({r['output_hash'] for r in wmc if r['file']==file})==1
stages=[]
for d in read('results/scallop_wmc_diagnostic.json'):
    events=[json.loads(line.split(' ',1)[1]) for line in d['stderr'].splitlines() if line.startswith('WMC_PROFILE_JSON ')]
    # The final event follows the two warmups; do not mix warmup diagnostics into measured results.
    p=events[-1]['profile']
    stages.append({'file':d['file'],'k':d['k'],'variant':d['label'],
                   'stage_ms':dict(zip(p['stage_order'],[v/1e6 for v in p['stage_ns']['wmc']])),
                   'sdd_nodes_total':p['sdd_nodes_total'],'sdd_nodes_max':p['sdd_nodes_max'],
                   'repeated_exact_formula_calls':p['repeated_exact_formula_calls'],
                   'scope':'Separate instrumented diagnostic, one measured repetition; build includes GC and builder cleanup.'})
exact=read('results/dolphin_exact_comparison.json')
assert exact['complete']
gpu8=next(r for r in exact['timing'] if r['device']=='cuda' and r['sum_n']==8)
controls={r['variant']:r for r in wmc if r['file']=='hwf_5_controlled.scl'}
metrics={k:v['median_ms'] for k,v in gpu8['host_to_prediction'].items()}
summary={
 'date':'2026-10-14',
 'decision':'Prioritize validation of input-dependent proof-DNF to SDD online construction; no hardware GO established.',
 'boundaries':{
  'dolphin':'Real heldout MNIST, trained official CNN, host-to-device through probability output; steady inference. Not paper training.',
  'scallop':'Controlled symbolic inputs, run+recover only; no CNN, no load/compile in online time.',
  'native':'Synthetic bounded operator inputs; not E2E.',
  'diagnostics':'Separate perturbed runs; do not interpret as exact natural E2E shares.'},
 'scallop_controls':wmc,'scallop_wmc_diagnostic':stages,
 'dolphin_exact_comparison':exact,
 'software_speedups':{
  'sum8_gpu_exact_over_dolphin_k1':metrics['dolphin_k1']/metrics['exact_convolution'],
  'sum8_gpu_exact_over_dolphin_k16':metrics['dolphin_k16']/metrics['exact_convolution'],
  'controlled_hwf5_symbolic_memo_both_over_original':controls['original']['symbolic_online_ms']['median']/controls['memo_both']['symbolic_online_ms']['median']},
 'no_claims':['No ASIC or FPGA implementation or speedup measured.','No global absence-of-prior-art claim.','Online SDD construction has not passed real neural E2E or mature-library baseline gates.']}
(R/'results/summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(16,5.2),gridspec_kw={'width_ratios':[1,1.12,1.2]})
colors=['#168777','#71859d','#527196','#324d72']
keys=['exact_convolution','dolphin_k1','dolphin_k4','dolphin_k16']
bars=axs[0].bar(range(4),[metrics[k] for k in keys],color=colors)
axs[0].set_xticks(range(4),['Exact\nconv.','DOLPHIN\nk=1','DOLPHIN\nk=4','DOLPHIN\nk=16'])
axs[0].bar_label(bars,fmt='%.2f',padding=4)
axs[0].set_ylim(0,29);axs[0].set_ylabel('Host-to-prediction latency (ms)')
axs[0].set_title('Real MNIST sum-8 | GPU\nBatch 16, same trained CNN')
variants=['original','memo_eval','memo_gc','memo_both']
vals=[controls[k]['symbolic_online_ms']['median'] for k in variants]
bars=axs[1].bar(range(4),vals,color=['#71859d','#527196','#324d72','#168777'])
axs[1].set_xticks(range(4),['Original','Eval\ncache','GC\nvisited','Both'])
axs[1].bar_label(bars,fmt='%.2f',padding=4)
axs[1].set_ylim(0,86);axs[1].set_ylabel('Symbolic run + recover (ms)')
axs[1].set_title('Controlled HWF-5, k=8 | CPU\n15 samples per software variant; no CNN')
ws={r['variant']:r for r in stages if r['file']=='hwf_5_controlled.scl'}
for j,(key,label) in enumerate([('build_sdd','SDD build\n(includes GC)'),('eval_sdd','SDD evaluation')]):
    for i,(variant,color) in enumerate([('original','#71859d'),('memo_both','#168777')]):
        val=ws[variant]['stage_ms'][key]
        axs[2].barh(j+(i-.5)*.28,val,height=.26,color=color,label={'original':'Original','memo_both':'Both controls'}[variant] if j==0 else None)
        axs[2].text(val+.7,j+(i-.5)*.28,f'{val:.2f}',va='center',fontsize=10)
axs[2].set_yticks([0,1],['SDD build\n(includes GC)','SDD evaluation']);axs[2].invert_yaxis()
axs[2].set_xlim(0,69);axs[2].set_xlabel('Separately instrumented stage time (ms)')
axs[2].set_title('Controlled HWF-5 recovery | CPU\nDiagnostic only; not E2E shares')
axs[2].legend(loc='lower right',frameon=False)
fig.suptitle('TXT-first re-profile: software controls change the hardware shortlist',fontsize=15,y=.99)
fig.text(.5,.015,'Measured on Slurm: RTX A5000 (node4) / AMD EPYC 7352 (node3). No dedicated-hardware speedup is claimed.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.05,1,.93],w_pad=2.2)
fig.savefig(R/'profile_summary.png',dpi=170)
print('Saved results/summary.json and profile_summary.png')
