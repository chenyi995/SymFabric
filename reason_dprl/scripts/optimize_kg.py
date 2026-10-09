#!/usr/bin/env python3
"""Online candidate/query compaction for a matched DPrL GPU baseline.

All compaction indices are obtained from the current GPU state on every run.
torch.nonzero's device-to-host shape synchronization is included in latency.
No cached proof trajectory, reduced search budget, or reduced precision is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import fields
from pathlib import Path
import time

from bench_common import provenance, save, sha, digest, measure, graph_measure
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from kge_experiments.builder import KGEConfig, create_env, create_policy
from kge_experiments.ppo import PPO


def power2(n):
    return 1 << (max(1, n) - 1).bit_length()


class CompactPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.compute = torch.compile(self._compute, fullgraph=True, dynamic=True)

    def _compute(self, current, candidates, indices):
        b, s, a, _ = candidates.shape
        net = self.policy.mlp_extractor
        # Invalid padding indices target a discarded output slot, never a real
        # candidate. Only the padded rows perform unnecessary neural work.
        selected = candidates.reshape(b*s, a, 3)[indices.clamp(max=b*s-1)]
        states = torch.cat((current, selected), dim=0).unsqueeze(1)
        embedded = self.policy.features_extractor.embedder.get_embeddings_batch(states)
        vectors = net.policy_head_final(net.policy_head_fused(net._get_shared_features(embedded))).squeeze(1)
        if net.use_l2_norm:
            vectors = F.normalize(vectors, dim=-1)
        query = vectors[:b]
        candidate = vectors[b:]
        owners = (indices // s).clamp(max=b-1)
        logits = (query[owners] * candidate).sum(-1)
        if net.sqrt_scale:
            logits = logits / (net.embed_dim ** .5)
        if net.temperature is not None:
            logits = logits / net.temperature
        dense = logits.new_full((b*s+1,), float('-inf')).scatter(0, indices, logits)
        return dense[:-1].view(b, s)

    def __call__(self, state):
        b, s = state['derived_states'].shape[:2]
        valid = (torch.arange(s, device='cuda')[None, :] < state['derived_counts'][:, None])
        valid = valid & ~state['done'].bool()[:, None]
        indices = valid.flatten().nonzero().flatten()
        count = indices.numel()  # Shape is materialized by nonzero; synchronization is timed.
        indices = F.pad(indices, (0, power2(count)-count), value=b*s)
        return self.compute(state['current_states'], state['derived_states'], indices), count


class CompactRunner:
    def __init__(self, env, policy, batch, depth, active=False):
        if env.use_exact_memory or env.skip_unary_actions:
            raise ValueError('This adapter requires the recorded hash-history/non-unary configuration')
        self.env, self.batch, self.depth, self.active = env, batch, depth, active
        self.policy = CompactPolicy(policy)
        self.reset = torch.compile(env.reset, fullgraph=True, dynamic=False)
        self.environments = {batch: env}
        self.steps = {batch: torch.compile(env.step, fullgraph=True, dynamic=False)}
        self.audit = None

    def small_env(self, size):
        if size not in self.environments:
            e = copy.copy(self.env)
            e._batch_size = size
            e._arange_B = torch.arange(size, device='cuda')
            for name in ('_ones_B', '_reset_ones_B'):
                setattr(e, name, torch.ones(size, dtype=torch.long, device='cuda'))
            e._reset_zeros_B = torch.zeros(size, dtype=torch.long, device='cuda')
            e._reset_mask_true = torch.ones(size, dtype=torch.bool, device='cuda')
            e._allocate_buffers()
            self.environments[size] = e
            self.steps[size] = torch.compile(e.step, fullgraph=True, dynamic=False)
        return self.environments[size], self.steps[size]

    def __call__(self, queries, references=None, record=False):
        spans = defaultdict(list)
        def run(name, fn):
            if not record:
                return fn()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            with torch.cuda.nvtx.range(name):
                out = fn()
            end.record()
            spans[name].append((start, end))
            return out
        _, state = run('reset', lambda: self.reset(queries))
        owners = torch.arange(self.batch, device='cuda')
        successes = torch.zeros(self.batch+1, dtype=torch.uint8, device='cuda')
        depths = torch.zeros(self.batch+1, dtype=torch.long, device='cuda')
        work = []
        max_error = 0.
        for t in range(self.depth):
            if self.active:
                def pack():
                    idx = (~state['done'].bool()).nonzero().flatten()
                    n = idx.numel()
                    if not n:
                        return None
                    cap = power2(n)
                    selected = F.pad(idx, (0, cap-n), value=0)
                    packed = state[selected].contiguous()
                    live = torch.arange(cap, device='cuda') < n
                    packed['done'] = (packed['done'].bool() | ~live).to(torch.uint8)
                    ids = torch.where(live, owners[selected], self.batch)
                    return packed, ids, n
                packed = run('active_pack', pack)
                if packed is None:
                    break
                state, owners, active_count = packed
            else:
                active_count = None
            logits, valid_count = run('compact_policy', lambda: self.policy(state))
            actions = run('selection', lambda: logits.argmax(-1))
            if references is not None:
                ref = references[t]
                real = owners < self.batch
                ids = owners[real]
                active_mask = ~state['done'][real].bool()
                compared = ids[active_mask]
                torch.testing.assert_close(actions[real][active_mask], ref['actions'][compared], atol=0, rtol=0)
                valid = ref['mask'][compared].bool()
                actual = logits[real][active_mask][valid]
                expected = ref['logits'][compared][valid]
                if actual.numel():
                    error = float((actual-expected).abs().max())
                    max_error = max(max_error, error)
                    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
            _, step = self.small_env(state.batch_size[0])
            _, state = run('environment', lambda: step(state, actions))
            if self.active:
                def finish():
                    successes.scatter_(0, owners, state['success'])
                    depths.scatter_(0, owners, state['depths'])
                run('output_update', finish)
            if references is not None:
                ref_state = references[t]['state']
                real = owners < self.batch
                # Compare all persistent proof state for every retained real row.
                for key in ('current_states', 'derived_states', 'derived_counts', 'next_var_indices',
                            'history_hashes', 'history_count', 'done', 'success', 'depths'):
                    torch.testing.assert_close(state[key][real], ref_state[key][owners[real]], atol=0, rtol=0)
            if record or references is not None:
                work.append({'step': t, 'query_capacity': state.batch_size[0],
                             'active_queries': active_count, 'valid_candidates': valid_count,
                             'neural_candidate_capacity': power2(valid_count)})
        if references is not None:
            self.audit = {'passed': True, 'max_valid_logit_error': max_error,
                          'all_active_actions_equal': True, 'persistent_states_equal': True,
                          'work': work}
        result = (successes[:-1], depths[:-1]) if self.active else (state['success'], state['depths'])
        if record:
            torch.cuda.synchronize()
            return result, {k: sum(a.elapsed_time(b) for a,b in events) for k,events in spans.items()}, work
        return result


def reference_trace(env, policy, queries, depth):
    get_logits = torch.compile(policy.get_logits, fullgraph=True)
    step = torch.compile(env.step, fullgraph=True)
    obs, state = env.reset(queries)
    trace = []
    for _ in range(depth):
        logits = get_logits(obs)
        actions = logits.argmax(-1)
        mask = obs['action_mask'].clone()
        obs, state = step(state, actions)
        trace.append({'logits': logits.clone(), 'actions': actions.clone(), 'mask': mask,
                      'state': state.clone()})
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['family','wn18rr'], required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--batches', type=int, default=3)
    ap.add_argument('--repeat', type=int, default=20)
    ap.add_argument('--stable-candidates', action='store_true')
    ap.add_argument('--capture-only', choices=['compact_policy','compact_active'])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = 256
    adapter = None
    if args.stable_candidates:
        from stable_candidates import install
        adapter = install()
    cfg = KGEConfig(dataset=args.dataset,n_envs=args.batch,fixed_batch_size=args.batch,
                    device='cuda',seed=42,kge_inference=False,use_callbacks=False,load_model=False)
    env = create_env(cfg)
    policy = create_policy(cfg)
    checkpoint = torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval(); env.eval()
    agent = PPO(policy,env,cfg,eval_only=True)
    meta = provenance()
    meta.update(checkpoint_sha256=sha(args.checkpoint),upstream_adapter=adapter,
                config={f.name:getattr(cfg,f.name) for f in fields(cfg) if not f.name.startswith('_')},
                precision={'dtype':'float32','tf32':False},
                contract='same ordered candidates, depth budget and frozen policy; online exact work compaction')
    save(args.output/'provenance.json',meta)
    results = {'dataset':args.dataset,'batch':args.batch,'batches':[],
               'timing_boundary':'resident queries through completed outputs, including online compaction and host synchronization'}
    compact = CompactRunner(env,policy,args.batch,cfg.max_steps,active=False)
    active = CompactRunner(env,policy,args.batch,cfg.max_steps,active=True)
    with torch.no_grad():
        for n in range(args.batches):
            queries=cfg._components['test_queries'][n*args.batch:(n+1)*args.batch].to('cuda').clone()
            if len(queries)!=args.batch:
                raise ValueError('Insufficient disjoint test queries')
            row={'index':n,'query_sha256':digest([queries]),'variants':{}}
            expected,native=measure(lambda:agent.evaluate_policy(queries),repeat=args.repeat)
            expected=[x.clone() for x in expected]
            row['variants']['native_compiled']=native
            actual,graph=graph_measure(lambda:agent.evaluate_policy(queries),repeat=args.repeat)
            assert all(torch.equal(a,b) for a,b in zip(actual,expected))
            row['variants']['native_graph']=graph
            print('BASELINE',n,native['median_ms'],graph['median_ms'],flush=True)
            refs=reference_trace(env,policy,queries,cfg.max_steps)
            assert all(torch.equal(refs[-1]['state'][k],v) for k,v in zip(('success','depths'),expected))
            for name,runner in [('compact_policy',compact),('compact_active',active)]:
                begin=time.perf_counter()
                actual=runner(queries,references=refs)
                assert all(torch.equal(a,b) for a,b in zip(actual,expected))
                print('CHECKED',n,name,'setup_s',time.perf_counter()-begin,flush=True)
                actual,timing=measure(lambda:runner(queries),repeat=args.repeat)
                assert all(torch.equal(a,b) for a,b in zip(actual,expected))
                _,spans,work=runner(queries,record=True)
                timing.update(correctness=runner.audit,stage_spans_ms=spans,work=work,
                              output_sha256=digest(actual),speedup_vs_native_graph=graph['median_ms']/timing['median_ms'])
                row['variants'][name]=timing
                print('OPTIMIZED',n,name,timing['median_ms'],timing['speedup_vs_native_graph'],flush=True)
                if args.capture_only == name:
                    torch.cuda.cudart().cudaProfilerStart()
                    with torch.cuda.nvtx.range('native_query'):
                        runner(queries,record=True)
                        torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
            results['batches'].append(row)
            save(args.output/'benchmark.json',results)
            del refs
            if args.capture_only:
                break
    print('OPTIMIZATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
