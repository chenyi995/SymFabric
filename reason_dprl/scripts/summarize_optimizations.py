#!/usr/bin/env python3
"""Aggregate completed GPU experiments without treating batches as replicates.

Each table entry is the arithmetic mean of the per-batch latency medians.
The per-batch range and all samples remain available. A single variant is
selected per dataset, rather than choosing a different winner for each query.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('runs',nargs='+',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    summary={}
    flat=[]
    for path in args.runs:
        data=json.loads((path/'benchmark.json').read_text())
        meta=json.loads((path/'provenance.json').read_text())
        dataset=data['dataset']
        if dataset in summary:
            raise ValueError('Provide one primary run per dataset')
        rows=data['batches']
        names=set(rows[0]['variants'])
        if len(rows)<3 or any(set(row['variants'])!=names for row in rows):
            raise ValueError(f'Incomplete multi-batch experiment: {path}')
        output={'source':str(path),'slurm_job_id':meta['slurm_job_id'],
                'checkpoint_sha256':meta['checkpoint_sha256'],'batch':data['batch'],
                'batches':len(rows),'precision':meta['precision'],
                'query_sha256':[row['query_sha256'] for row in rows],
                'aggregation':'mean of per-batch medians; not a pooled latency percentile',
                'variants':{}}
        for name in sorted(names):
            values=[row['variants'][name] for row in rows]
            if name.startswith('fast_'):
                assert all(v['correctness']['passed'] and v['repeatability']['passed'] for v in values)
            medians=[v['median_ms'] for v in values]
            entry={'mean_batch_median_ms':statistics.mean(medians),
                   'batch_medians_ms':medians,'batch_p95_ms':[v['p95_ms'] for v in values],
                   'samples_per_batch':[len(v['samples_ms']) for v in values]}
            if 'stage_spans_ms' in values[0]:
                stages=defaultdict(list)
                for v in values:
                    for k,x in v['stage_spans_ms'].items():
                        stages[k].append(x)
                entry['mean_instrumented_stage_spans_ms']={k:statistics.mean(v) for k,v in stages.items()}
                entry['max_valid_logit_error']=max(v['correctness']['max_valid_logit_error'] for v in values)
                entry['all_active_actions_and_persistent_states_equal']=True
                entry['output_sha256']=[v['output_sha256'] for v in values]
            output['variants'][name]=entry
            for row,v in zip(rows,values):
                flat.append({'dataset':dataset,'batch_index':row['index'],'variant':name,
                             'median_ms':v['median_ms'],'p95_ms':v['p95_ms'],
                             'mean_ms':v['mean_ms'],'repeat':len(v['samples_ms'])})
        control=output['variants']['native_graph']['mean_batch_median_ms']
        for entry in output['variants'].values():
            entry['speedup_vs_native_graph']=control/entry['mean_batch_median_ms']
        best=min(names,key=lambda n:output['variants'][n]['mean_batch_median_ms'])
        output['best_variant']=best
        summary[dataset]=output
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output/'per_batch.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    lines=['| Dataset | Variant | Mean batch median (ms) | Batch medians (ms) | Speedup vs graph |',
           '|---|---|---:|---|---:|']
    for dataset,data in summary.items():
        for name,v in data['variants'].items():
            per=', '.join(f'{x:.3f}' for x in v['batch_medians_ms'])
            lines.append(f'| {dataset} | {name} | {v["mean_batch_median_ms"]:.3f} | {per} | {v["speedup_vs_native_graph"]:.2f}x |')
    (args.output/'table.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Tables complete; matplotlib unavailable for optional plots')
        return
    order=['native_compiled','native_graph','fast_policy_graphs','fast_active_graphs','fast_fused_graphs']
    labels=['Compiled native','Full CUDA Graph','Compact neural + fast pack','Also compact queries','Also fuse query packing']
    fig,axes=plt.subplots(1,len(summary),figsize=(12,4.4),squeeze=False)
    for ax,(dataset,data) in zip(axes[0],summary.items()):
        means=[data['variants'][n]['mean_batch_median_ms'] for n in order]
        ax.barh(labels,means,color=['#8093a6','#557995','#39a29a','#288376','#156958'])
        for i,v in enumerate(means):
            ax.text(v+max(means)*.02,i,f'{v:.2f}',va='center',fontsize=9)
        ax.set_xlim(0,max(means)*1.17);ax.invert_yaxis()
        ax.set_title(dataset);ax.set_xlabel('Completed batch latency (ms)')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Measured GPU inference, RTX A5000, batch 64, strict FP32\nMean of 3 disjoint-batch medians; 20 warm repetitions each',fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output/'latency.png',dpi=180)
    fig.savefig(args.output/'latency.pdf')
    plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':
    main()
