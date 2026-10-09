#!/usr/bin/env python3
"""Matched end-to-end ablation of contention-free GPU candidate packing."""
import argparse
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
import time

from bench_common import provenance, save, sha, digest, measure, graph_measure
import torch
import torch.nn.functional as F
from optimize_kg import power2, reference_trace
from optimize_kg_graphs import GraphCall, GraphCompactRunner
from kge_experiments.builder import KGEConfig, create_env, create_policy
from kge_experiments.ppo import PPO
import stable_candidates
import fast_candidates


class FusedPackRunner(GraphCompactRunner):
    """Compile the state gathers instead of launching one eager op per field."""
    def __init__(self, env, policy, batch, depth):
        super().__init__(env, policy, batch, depth)
        self.pack = GraphCall(torch.compile(self._pack, fullgraph=True, dynamic=False))

    def _pack(self, state, owners, selected, live):
        packed = state[selected].contiguous()
        packed['done'] = (packed['done'].bool() | ~live).to(torch.uint8)
        ids = torch.where(live, owners[selected], self.batch)
        return packed, ids

    def __call__(self, queries, references=None, record=False):
        spans = defaultdict(list)
        def run(name, fn):
            if not record:
                return fn()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            with torch.cuda.nvtx.range(name):
                result = fn()
            end.record()
            spans[name].append((start,end))
            return result
        _,state=run('reset',lambda:self.reset(queries))
        owners=torch.arange(self.batch,device='cuda')
        successes=torch.zeros(self.batch+1,dtype=torch.uint8,device='cuda')
        depths=torch.zeros(self.batch+1,dtype=torch.long,device='cuda')
        work=[]
        max_error=0.
        for t in range(self.depth):
            def indices():
                idx=(~state['done'].bool()).nonzero().flatten()
                n=idx.numel()
                cap=power2(n)
                selected=F.pad(idx,(0,cap-n),value=0)
                live=torch.arange(cap,device='cuda')<n
                return selected,live,n
            selected,live,n=run('active_index',indices)
            if not n:
                break
            state,owners=run('active_pack',lambda:self.pack(state,owners,selected,live))
            logits,valid_count=run('compact_policy',lambda:self.policy(state))
            actions=run('selection',lambda:logits.argmax(-1))
            if references is not None:
                ref=references[t]
                real=owners<self.batch
                active_mask=~state['done'][real].bool()
                ids=owners[real][active_mask]
                torch.testing.assert_close(actions[real][active_mask],ref['actions'][ids],atol=0,rtol=0)
                valid=ref['mask'][ids].bool()
                actual=logits[real][active_mask][valid]
                expected=ref['logits'][ids][valid]
                if actual.numel():
                    max_error=max(max_error,float((actual-expected).abs().max()))
                    torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-5)
            _,step=self.small_env(state.batch_size[0])
            _,state=run('environment',lambda:step(state,actions))
            def finish():
                successes.scatter_(0,owners,state['success'])
                depths.scatter_(0,owners,state['depths'])
            run('output_update',finish)
            if references is not None:
                real=owners<self.batch
                for key in ('current_states','derived_states','derived_counts','next_var_indices',
                            'history_hashes','history_count','done','success','depths'):
                    torch.testing.assert_close(state[key][real],ref['state'][key][owners[real]],atol=0,rtol=0)
            if record or references is not None:
                work.append({'step':t,'query_capacity':state.batch_size[0],
                             'active_queries':n,'valid_candidates':valid_count,
                             'neural_candidate_capacity':power2(valid_count)})
        if references is not None:
            self.audit={'passed':True,'max_valid_logit_error':max_error,
                        'all_active_actions_equal':True,'persistent_states_equal':True,'work':work}
        result=(successes[:-1],depths[:-1])
        if record:
            torch.cuda.synchronize()
            return result,{k:sum(a.elapsed_time(b) for a,b in events) for k,events in spans.items()},work
        return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['family','wn18rr'],required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--batches',type=int,default=3)
    ap.add_argument('--repeat',type=int,default=20)
    ap.add_argument('--capture-only',choices=['fast_policy_graphs','fast_active_graphs','fast_fused_graphs'])
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch._dynamo.config.recompile_limit=128
    torch._dynamo.config.accumulated_recompile_limit=512
    stable=stable_candidates.install()
    cfg=KGEConfig(dataset=args.dataset,n_envs=args.batch,fixed_batch_size=args.batch,
                  device='cuda',seed=42,kge_inference=False,use_callbacks=False,load_model=False)
    env=create_env(cfg);policy=create_policy(cfg)
    policy.load_state_dict(torch.load(args.checkpoint,map_location='cuda',weights_only=False)['policy'])
    policy.eval();env.eval()
    agent=PPO(policy,env,cfg,eval_only=True)
    meta=provenance()
    meta.update(checkpoint_sha256=sha(args.checkpoint),native_adapter=stable,
                config={f.name:getattr(cfg,f.name) for f in fields(cfg) if not f.name.startswith('_')},
                precision={'dtype':'float32','tf32':False},
                boundary='resident GPU queries through completed outputs; online indices and input copies included')
    save(args.output/'provenance.json',meta)
    results={'dataset':args.dataset,'batch':args.batch,'batches':[]}
    inputs=[];refs=[];expected=[]
    with torch.no_grad():
        for n in range(args.batches):
            q=cfg._components['test_queries'][n*args.batch:(n+1)*args.batch].to('cuda').clone()
            assert len(q)==args.batch
            inputs.append(q)
            output,native=measure(lambda:agent.evaluate_policy(q),repeat=args.repeat)
            expected.append([x.clone() for x in output])
            actual,graph=graph_measure(lambda:agent.evaluate_policy(q),repeat=args.repeat)
            assert all(torch.equal(a,b) for a,b in zip(actual,expected[-1]))
            hashes=[digest(agent.evaluate_policy(q)) for _ in range(5)]
            assert all(h==digest(expected[-1]) for h in hashes)
            row={'index':n,'query_sha256':digest([q]),'native_repeatability':{'passed':True,'hashes':hashes},
                 'variants':{'native_compiled':native,'native_graph':graph}}
            results['batches'].append(row)
            refs.append(reference_trace(env,policy,q,cfg.max_steps))
            assert all(torch.equal(refs[-1][-1]['state'][k],v) for k,v in zip(('success','depths'),expected[-1]))
            save(args.output/'benchmark.json',results)
            print('BASELINE',n,native['median_ms'],graph['median_ms'],flush=True)
        meta['optimized_adapter']=fast_candidates.install()
        save(args.output/'provenance.json',meta)
        fixed=GraphCompactRunner(env,policy,args.batch,cfg.max_steps)
        fixed.active=False
        runners={'fast_policy_graphs':fixed,
                 'fast_active_graphs':GraphCompactRunner(env,policy,args.batch,cfg.max_steps),
                 'fast_fused_graphs':FusedPackRunner(env,policy,args.batch,cfg.max_steps)}
        for n,q in enumerate(inputs):
            row=results['batches'][n]
            for name,runner in runners.items():
                begin=time.perf_counter()
                actual=runner(q,references=refs[n])
                assert all(torch.equal(a,b) for a,b in zip(actual,expected[n]))
                print('CHECKED',n,name,time.perf_counter()-begin,flush=True)
                actual,timing=measure(lambda:runner(q),repeat=args.repeat)
                assert all(torch.equal(a,b) for a,b in zip(actual,expected[n]))
                hashes=[digest(runner(q)) for _ in range(5)]
                assert all(h==digest(expected[n]) for h in hashes)
                _,spans,work=runner(q,record=True)
                timing.update(correctness=runner.audit,repeatability={'passed':True,'hashes':hashes},
                              stage_spans_ms=spans,work=work,output_sha256=digest(actual),
                              speedup_vs_native_graph=row['variants']['native_graph']['median_ms']/timing['median_ms'])
                row['variants'][name]=timing
                save(args.output/'benchmark.json',results)
                print('OPTIMIZED',n,name,timing['median_ms'],timing['speedup_vs_native_graph'],flush=True)
                if args.capture_only==name:
                    torch.cuda.cudart().cudaProfilerStart()
                    with torch.cuda.nvtx.range('native_query'):
                        runner(q,record=True)
                        torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    print('CAPTURE_COMPLETE',flush=True)
                    return
    print('OPTIMIZATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
