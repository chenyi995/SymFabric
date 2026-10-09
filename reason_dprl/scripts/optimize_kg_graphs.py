#!/usr/bin/env python3
"""CUDA Graphs for runtime-selected compact DPrL shapes.

Graph inputs are copied from the live proof state each invocation. Counts,
indices, and query selection remain online and their costs are timed.
"""
import argparse
from pathlib import Path
import time

from bench_common import provenance, save, sha, digest, measure, graph_measure
import torch
from tensordict import TensorDict
from optimize_kg import CompactPolicy, CompactRunner, reference_trace
from kge_experiments.builder import KGEConfig, create_env, create_policy
from kge_experiments.ppo import PPO
from stable_candidates import install


class GraphCall:
    def __init__(self, fn):
        self.fn = fn
        self.cache = {}

    def __call__(self, *args):
        def flatten(values):
            flat, specs = [], []
            for value in values:
                if isinstance(value, TensorDict):
                    keys = tuple(sorted(value.keys()))
                    flat.extend(value[k] for k in keys)
                    specs.append(('dict', keys, tuple(value.batch_size)))
                else:
                    flat.append(value)
                    specs.append(('tensor',))
            return flat, specs
        flat, specs = flatten(args)
        key = (tuple(specs), tuple((tuple(v.shape), tuple(v.stride()), v.dtype) for v in flat))
        if key not in self.cache:
            cloned = [a.clone() for a in args]
            static, _ = flatten(cloned)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.fn(*cloned)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.fn(*cloned)
            torch.cuda.synchronize()
            self.cache[key] = (graph, static, output, cloned)
        graph, static, output, _ = self.cache[key]
        torch._foreach_copy_(static, flat)
        graph.replay()
        return output


class GraphCompactPolicy(CompactPolicy):
    def __init__(self, policy):
        super().__init__(policy)
        self.compute = GraphCall(self.compute)


class GraphCompactRunner(CompactRunner):
    def __init__(self, env, policy, batch, depth):
        super().__init__(env, policy, batch, depth, active=True)
        self.policy = GraphCompactPolicy(policy)
        self.reset = GraphCall(self.reset)
        self.steps = {k: GraphCall(v) for k,v in self.steps.items()}

    def small_env(self, size):
        env, step = super().small_env(size)
        if not isinstance(step, GraphCall):
            self.steps[size] = GraphCall(step)
        return env, self.steps[size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['family','wn18rr'], required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--batches', type=int, default=3)
    ap.add_argument('--repeat', type=int, default=20)
    ap.add_argument('--capture-only', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch._dynamo.config.recompile_limit=64
    torch._dynamo.config.accumulated_recompile_limit=256
    # The expanded WN18RR query set exposes the same upstream scatter race as
    # Family. Both the native control and every optimized variant use this fix.
    adapter=install()
    cfg=KGEConfig(dataset=args.dataset,n_envs=args.batch,fixed_batch_size=args.batch,
                  device='cuda',seed=42,kge_inference=False,use_callbacks=False,load_model=False)
    env=create_env(cfg)
    policy=create_policy(cfg)
    policy.load_state_dict(torch.load(args.checkpoint,map_location='cuda',weights_only=False)['policy'])
    policy.eval();env.eval()
    agent=PPO(policy,env,cfg,eval_only=True)
    meta=provenance()
    meta.update(checkpoint_sha256=sha(args.checkpoint),upstream_adapter=adapter,
                precision={'dtype':'float32','tf32':False},
                boundary='resident GPU queries to completed results; online shape decisions and input copies included',
                candidate_semantics='same stable first-K order and unchanged search budget')
    save(args.output/'provenance.json',meta)
    results={'dataset':args.dataset,'batch':args.batch,'batches':[]}
    runners={
        'compact_policy':CompactRunner(env,policy,args.batch,cfg.max_steps,active=False),
        'compact_active':CompactRunner(env,policy,args.batch,cfg.max_steps,active=True),
        'compact_active_graphs':GraphCompactRunner(env,policy,args.batch,cfg.max_steps),
    }
    with torch.no_grad():
        for n in range(args.batches):
            queries=cfg._components['test_queries'][n*args.batch:(n+1)*args.batch].to('cuda').clone()
            if len(queries)!=args.batch:
                raise ValueError('Not enough disjoint queries')
            row={'index':n,'query_sha256':digest([queries]),'variants':{}}
            results['batches'].append(row)
            expected,timing=measure(lambda:agent.evaluate_policy(queries),repeat=args.repeat)
            expected=[x.clone() for x in expected]
            row['variants']['native_compiled']=timing
            hashes=[digest(agent.evaluate_policy(queries)) for _ in range(5)]
            assert all(h==digest(expected) for h in hashes)
            row['native_repeatability']={'passed':True,'hashes':hashes}
            actual,graph=graph_measure(lambda:agent.evaluate_policy(queries),repeat=args.repeat)
            assert all(torch.equal(a,b) for a,b in zip(actual,expected))
            row['variants']['native_graph']=graph
            save(args.output/'benchmark.json',results)
            print('BASELINE',n,timing['median_ms'],graph['median_ms'],flush=True)
            refs=reference_trace(env,policy,queries,cfg.max_steps)
            assert all(torch.equal(refs[-1]['state'][k],v) for k,v in zip(('success','depths'),expected))
            for name,runner in runners.items():
                start=time.perf_counter()
                actual=runner(queries,references=refs)
                assert all(torch.equal(a,b) for a,b in zip(actual,expected))
                print('CHECKED',n,name,time.perf_counter()-start,flush=True)
                actual,timing=measure(lambda:runner(queries),repeat=args.repeat)
                assert all(torch.equal(a,b) for a,b in zip(actual,expected))
                repeated=[digest(runner(queries)) for _ in range(5)]
                assert all(h==digest(expected) for h in repeated)
                _,spans,work=runner(queries,record=True)
                timing.update(correctness=runner.audit,repeatability={'passed':True,'hashes':repeated},
                              stage_spans_ms=spans,work=work,output_sha256=digest(actual),
                              speedup_vs_native_graph=graph['median_ms']/timing['median_ms'])
                row['variants'][name]=timing
                save(args.output/'benchmark.json',results)
                print('OPTIMIZED',n,name,timing['median_ms'],timing['speedup_vs_native_graph'],flush=True)
            if args.capture_only:
                runner=runners['compact_active_graphs']
                torch.cuda.cudart().cudaProfilerStart()
                with torch.cuda.nvtx.range('native_query'):
                    runner(queries,record=True)
                    torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                break
            del refs
    print('OPTIMIZATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
