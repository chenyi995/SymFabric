#!/usr/bin/env python3
"""Validate complete dynamic-state experiments and aggregate paired results."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def summarize(path, batches, repeat):
    data=json.loads((path/'benchmark.json').read_text())
    meta=json.loads((path/'provenance.json').read_text())
    edges=json.loads((path/'edge_cases.json').read_text())
    assert len(edges)==4 and all(e['passed'] for e in edges)
    assert len(data['batches'])==batches
    seen=set();totals=Counter();goals=Counter();variants={}
    for row in data['batches']:
        assert row['query_sha256'] not in seen
        seen.add(row['query_sha256'])
        totals.update(row['state_audit']['totals'])
        goals.update(row['state_audit']['active_goal_length_histogram'])
        assert set(row['variants'])=={'previous_best','compact','indexed'}
        expected=row['variants']['previous_best']['output_sha256']
        for name,variant in row['variants'].items():
            assert len(variant['samples_ms'])==repeat
            assert variant['correctness']['passed']
            assert variant['correctness']['persistent_states_equal']
            assert variant['correctness']['all_active_actions_equal']
            assert variant['repeatability']['passed']
            assert variant['repeatability']['hashes']==[expected]*5
            assert variant['output_sha256']==expected
            variants.setdefault(name,[]).append(variant)
    result={'dataset':data['dataset'],'slurm_job_id':meta['slurm_job_id'],
            'checkpoint_sha256':meta['checkpoint_sha256'],'batches':batches,
            'repeat':repeat,'validation_passed':True,'variants':{},
            'trace_totals':dict(totals),'active_goal_length_histogram':dict(goals),
            'identical_parent_atom_fraction':totals['identical_parent_atoms']/max(1,totals['retained_atoms']),
            'argument_match_fraction':totals['argument_matching_facts_before_exclusion']/max(1,totals['predicate_prefix_facts']),
            'bound_argument_fraction':totals['bound_argument_slots']/max(1,2*totals['active_state_visits']),
            'state_capacity':data['batches'][0]['state_audit'],
            'indexes':{k:v for k,v in meta.items() if k in ('compact_adapter','indexed_adapter')}}
    old=statistics.mean(v['median_ms'] for v in variants['previous_best'])
    for name,rows in variants.items():
        mean=statistics.mean(v['median_ms'] for v in rows)
        result['variants'][name]={
            'mean_of_batch_medians_ms':mean,'batch_medians_ms':[v['median_ms'] for v in rows],
            'speedup_vs_previous_best':old/mean,
            'diagnostic_mean_stage_spans_ms':{s:statistics.mean(v['stage_spans_ms'].get(s,0) for v in rows)
                                             for s in rows[0]['stage_spans_ms']},
            'max_valid_logit_error':max(v['correctness']['max_valid_logit_error'] for v in rows)}
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('runs',type=Path,nargs='+')
    ap.add_argument('--batches',type=int,default=3)
    ap.add_argument('--repeat',type=int,default=20)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=[summarize(p,args.batches,args.repeat) for p in args.runs]
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in result:
        print(r['dataset'],{k:round(v['mean_of_batch_medians_ms'],3) for k,v in r['variants'].items()},
              'atom_reuse',round(r['identical_parent_atom_fraction'],4),
              'fact_match',round(r['argument_match_fraction'],4))


if __name__=='__main__':
    main()
