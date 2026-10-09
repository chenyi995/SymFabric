#!/usr/bin/env python3
"""Falsifiable DPrL state-construction experiment on the frozen GPU workload.

Compare the previous best GPU implementation with exact fact-descriptor
compaction and an argument index. Timings include all online index lookups,
matching, overflow-proof checking, state construction and policy inference.
"""
import argparse
from collections import Counter, defaultdict
import copy
from pathlib import Path
import time

from bench_common import provenance, save, sha, digest, measure
import torch
from kge_experiments.builder import KGEConfig, create_env, create_policy
from kge_experiments.ppo import PPO
import kge_experiments.unification as u
from optimize_kg import reference_trace
from optimize_kg_graphs import GraphCompactRunner
from optimize_kg_fast import FusedPackRunner
import stable_candidates
import fast_candidates
import indexed_candidates


def edge_cases():
    """Proof after retained K, repeated variables, exclusion and terminal rows."""
    device = torch.device('cuda')
    facts = torch.tensor([[1,1,2],[1,1,3],[1,1,4],[1,2,2],
                          [1,0,2],[2,4,5],[3,2,7],[3,3,7]], device=device)
    ranges = torch.zeros((21,2),dtype=torch.long,device=device)
    ranges[1]=torch.tensor([0,5],device=device)
    ranges[2]=torch.tensor([5,6],device=device)
    ranges[3]=torch.tensor([6,8],device=device)
    engine = u.UnificationEngineVectorized(
        facts, torch.tensor([[[3,11,12]]], device=device),
        torch.tensor([1], device=device), torch.tensor([[4,11,12]], device=device),
        padding_idx=0, constant_no=10, runtime_var_end_index=50,
        true_pred_idx=18, false_pred_idx=19, max_arity=2,
        predicate_range_map=ranges, device=device, predicate_no=20,
        max_derived_per_state=2, max_fact_pairs=5, max_rule_pairs=1,
        padding_atoms=4, compile_standardize=False)
    rows = [[[1,1,11],[2,11,5]],  # Third matching fact proves; K=2.
            [[1,11,11],[4,11,7]], [[1,1,11],[3,11,7]],
            [[1,1,4]], [[1,1,8]], [[1,11,12],[4,11,12]],
            [[4,1,2]], [[18,0,0]], [[19,0,0]], [],
            [[1,0,11],[4,11,7]]]
    states = torch.zeros((len(rows),4,3), dtype=torch.long, device=device)
    for i,row in enumerate(rows):
        if row:
            states[i,:len(row)] = torch.tensor(row,device=device)
    variables = torch.full((len(rows),),21,device=device,dtype=torch.long)
    outcomes = []
    for exclude in (None, states[:,:1].clone()):
        expected = engine.get_derived_states_compiled(states, variables, exclude)
        # The original source must detect the third fact's proof before K.
        assert expected[0][0,0,0,0].item() == 18
        for mode in ('compact','indexed'):
            adapted = copy.copy(engine)
            indexed_candidates.install(adapted, mode)
            fn = torch.compile(adapted.get_derived_states_compiled, fullgraph=True)
            actual = fn(states,variables,exclude)
            for a,e in zip(actual,expected):
                torch.testing.assert_close(a,e,atol=0,rtol=0)
            outcomes.append({'mode':mode,'excluded':exclude is not None,
                             'rows':len(rows),'passed':True})
    return outcomes


def state_audit(env, initial, refs):
    """Untimed trace statistics; logical tensor size is not measured DRAM traffic."""
    engine = env.engine
    facts = engine.facts_idx.cpu().tolist()
    starts, lengths = engine.fact_seg_starts.cpu().tolist(), engine.fact_seg_lens.cpu().tolist()
    domains = {p:facts[s:s+min(l,engine.max_fact_pairs)] for p,(s,l) in enumerate(zip(starts,lengths))}
    total = Counter(); goals = Counter(); lengths_hist = Counter(); matched_hist = Counter()
    for state in [initial] + [r['state'] for r in refs[:-1]]:
        current = state['current_states'].cpu().tolist()
        candidates = state['derived_states'].cpu().tolist()
        counts = state['derived_counts'].cpu().tolist()
        done = state['done'].cpu().tolist()
        for parent, children, count, terminal in zip(current,candidates,counts,done):
            if terminal:
                continue
            total['active_state_visits'] += 1
            atoms = [tuple(a) for a in parent if a[0] != 0]
            lengths_hist[len(atoms)] += 1
            goals[tuple(atoms)] += 1
            p,a,c = parent[0]
            va,vc = a > engine.constant_no,c > engine.constant_no
            total['bound_argument_slots'] += int(not va)+int(not vc)
            domain = domains.get(p,[])
            total['predicate_prefix_facts'] += len(domain)
            matching = sum((va or a==x) and (vc or c==y) and
                           not (va and vc and a==c and x!=y and x!=0 and y!=0)
                           for _,x,y in domain)
            total['argument_matching_facts_before_exclusion'] += matching
            matched_hist[matching] += 1
            for child in children[:count]:
                child_atoms = [tuple(x) for x in child if x[0]!=0]
                total['retained_candidates'] += 1
                total['retained_atoms'] += len(child_atoms)
                total['identical_parent_atoms'] += sum((Counter(atoms)&Counter(child_atoms)).values())
                total['identical_parent_slots'] += sum(x==y for x,y in zip(parent,child))
                total['candidate_slots'] += len(child)
    b,a = initial['current_states'].shape[:2]
    return {'totals':dict(total), 'active_goal_length_histogram':dict(lengths_hist),
            'matching_facts_histogram':dict(matched_hist), 'unique_goal_states':len(goals),
            'identical_parent_atom_fraction':total['identical_parent_atoms']/max(1,total['retained_atoms']),
            'logical_dense_fact_state_bytes_per_derive':b*engine.max_fact_pairs*(a-1)*3*8,
            'logical_compact_fact_state_bytes_per_derive':b*engine.K_max*(a-1)*3*8,
            'scope':'active pre-selection states; match counts precede exclusion; bytes describe one logical tensor, not DRAM traffic',
            'capacities':{'B':b,'A':a,'F':engine.max_fact_pairs,'K':engine.K_max,'M':engine.M_max}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['family','wn18rr'], required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--batches',type=int,default=3)
    ap.add_argument('--repeat',type=int,default=20)
    ap.add_argument('--capture-only',choices=['previous_best','compact','indexed'])
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch._dynamo.config.recompile_limit=256
    torch._dynamo.config.accumulated_recompile_limit=1024
    stable=stable_candidates.install()
    meta=provenance()
    meta.update(checkpoint_sha256=sha(args.checkpoint),stable_adapter=stable,
                precision={'dtype':'float32','tf32':False},
                timing_boundary='resident queries to completed outputs; online lookup/matching/overflow proof scans included',
                preprocessing='index build is reported separately; fixed KB, no cached proof trajectory')
    save(args.output/'provenance.json',meta)
    save(args.output/'edge_cases.json',edge_cases())
    print('EDGE_CASES_PASSED',flush=True)
    cfg=KGEConfig(dataset=args.dataset,n_envs=args.batch,fixed_batch_size=args.batch,
                  device='cuda',seed=42,kge_inference=False,use_callbacks=False,load_model=False)
    env=create_env(cfg);policy=create_policy(cfg)
    policy.load_state_dict(torch.load(args.checkpoint,map_location='cuda',weights_only=False)['policy'])
    policy.eval();env.eval()
    PPO(policy,env,cfg,eval_only=True)
    results={'dataset':args.dataset,'batch':args.batch,'batches':[]}
    inputs=[];references=[]
    with torch.no_grad():
        for n in range(args.batches):
            q=cfg._components['test_queries'][n*args.batch:(n+1)*args.batch].to('cuda').clone()
            assert len(q)==args.batch
            inputs.append(q)
            _, initial=env.reset(q)
            initial=initial.clone()
            refs=reference_trace(env,policy,q,cfg.max_steps)
            references.append(refs)
            results['batches'].append({'index':n,'query_sha256':digest([q]),'variants':{},
                                      'state_audit':state_audit(env,initial,refs)})
            save(args.output/'benchmark.json',results)
            print('REFERENCE',n,flush=True)
        meta['packing_adapter']=fast_candidates.install()
        for name in ('previous_best','compact','indexed'):
            if name=='previous_best':
                experiment_env=env
            else:
                experiment_env=copy.copy(env)
                experiment_env.engine=copy.copy(env.engine)
                adapter=indexed_candidates.install(experiment_env.engine,name)
                source=adapter.pop('generated_source')
                (args.output/(name+'_generated.py')).write_text(source)
                meta[name+'_adapter']=adapter
            if name=='previous_best' and args.dataset=='wn18rr':
                runner=FusedPackRunner(experiment_env,policy,args.batch,cfg.max_steps)
            else:
                runner=GraphCompactRunner(experiment_env,policy,args.batch,cfg.max_steps)
                runner.active=False
            save(args.output/'provenance.json',meta)
            for n,q in enumerate(inputs):
                started=time.perf_counter()
                actual=runner(q,references=references[n])
                expected=tuple(references[n][-1]['state'][k] for k in ('success','depths'))
                assert all(torch.equal(a,b) for a,b in zip(actual,expected))
                print('CHECKED',n,name,time.perf_counter()-started,flush=True)
                actual,timing=measure(lambda:runner(q),repeat=args.repeat)
                hashes=[digest(runner(q)) for _ in range(5)]
                assert all(h==digest(expected) for h in hashes)
                _,spans,work=runner(q,record=True)
                timing.update(correctness=runner.audit,repeatability={'passed':True,'hashes':hashes},
                              stage_spans_ms=spans,work=work,output_sha256=digest(actual))
                results['batches'][n]['variants'][name]=timing
                save(args.output/'benchmark.json',results)
                print('RESULT',n,name,timing['median_ms'],flush=True)
                if args.capture_only==name:
                    torch.cuda.cudart().cudaProfilerStart()
                    with torch.cuda.nvtx.range('dynamic_validation'):
                        runner(q,record=True)
                        torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    return
    print('VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
