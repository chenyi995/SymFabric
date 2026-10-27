"""Explicit CPU native64 bulk cache construction and different-group E2E replay.

Run only in a pinned Slurm worker. Cold native construction and cache filling are
reported separately; warm forwards use different batch order/grouping and size.
Exact upstream count checks and neural/gradient checks are outside timed loops.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time
import traceback

import torch
from bench_stage2 import CaseRunner, CPUInputController, UnusedController, sample, loss_from
from run_bench import stats, sync
from streaming_predictor import StreamingPredictor
from memoized_grounder import MemoizedGrounder
from sparse_cpu_backend_optimized import SparseCPUBackendOptimized

ROOT=Path(__file__).resolve().parent

def assert_outputs(a,b):
    torch.testing.assert_close(a[0],b[0],rtol=2e-5,atol=2e-5)
    assert torch.equal(a[1],b[1])
    finite=torch.isfinite(a[0]) & torch.isfinite(b[0])
    return float((a[0][finite]-b[0][finite]).abs().max()) if finite.any() else 0.0

def mixed_groups(first,second):
    all_arrays=[torch.cat([x,y],dim=0) if x is not None else None for x,y in zip(first,second)]
    n1=len(first[0]);n2=len(second[0])
    # Both new batches deliberately interleave the independently built groups.
    order=[]
    for i in range(max(n1,n2)):
        if i<n2:order.append(n1+n2-1-i)
        if i<n1:order.append(i)
    outputs=[]
    for batch in [8,32]:
        index=torch.tensor(order[:min(batch,len(order))],device=first[0].device)
        outputs.append((f'mixed_B{len(index)}',[None if x is None else x[index] for x in all_arrays]))
    return outputs


def exact_counts(runner,memo,query):
    h,r,t,removed,target=query
    hc=h.cpu();rc=None if removed is None else removed.cpu()
    memo.collect_stats=True;memo.begin(hc,runner.relation,rc)
    checked=0
    runner.g.propagate=runner.original_p;runner.g.relation2adjacency=runner.adj
    try:
        for body in memo.rule_bodies:
            ids,values=memo.grounding(hc,runner.relation,body,rc)
            dense=torch.zeros(len(h)*runner.g.entity_size,dtype=torch.int64)
            dense[ids]=values
            expected=runner.original_g(h,runner.relation,body,removed).cpu()
            torch.testing.assert_close(dense.reshape_as(expected),expected,rtol=0,atol=0)
            checked+=1
    finally:
        diagnostic=memo.finish(require_complete=checked==len(memo.rule_bodies))
    assert diagnostic['misses']==0
    return {'terminal_exact_checks':checked,'cache':diagnostic}


def train_check(runner,adapter,memo,query):
    """Untimed matched training check; preserve caller model/RNG/mode state."""
    h,r,t,removed,target=query;m=runner.m
    initial=copy.deepcopy(m.state_dict())
    initial_grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in m.named_parameters()}
    initial_modes=[(module,module.training) for module in adapter.modules()]
    initial_collect_stats=memo.collect_stats
    def rng_snapshot():
        return (torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,random.getstate())
    def rng_restore(state):
        torch.set_rng_state(state[0])
        if state[1] is not None:torch.cuda.set_rng_state_all(state[1])
        random.setstate(state[2])
    initial_rng=rng_snapshot()
    try:
        # cuDNN LSTM backward requires a training-mode forward. Forward timing
        # remains in eval mode outside this independent correctness routine.
        adapter.train();m.train()
        options={'lr':0.001}
        m.zero_grad(set_to_none=True)
        refopt=torch.optim.Adam(m.parameters(),**options)
        rng_restore(initial_rng)
        ref=runner.forward('upstream',h,r,removed)
        refloss=loss_from(ref,target);refloss.backward()
        grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in m.named_parameters()}
        refopt.step()
        expected_params={n:p.detach().clone() for n,p in m.named_parameters()}
        expected_opt=copy.deepcopy(refopt.state_dict())
        changed_parameters=sum(not torch.equal(expected_params[n],initial[n]) for n in expected_params)
        assert changed_parameters>0, 'training check did not change NN weights'
        m.load_state_dict(initial);m.zero_grad(set_to_none=True)
        opt=torch.optim.Adam(m.parameters(),**options)
        memo.collect_stats=True
        rng_restore(initial_rng)
        got=adapter(h,r,removed);gotloss=loss_from(got,target);gotloss.backward()
        torch.testing.assert_close(gotloss,refloss,rtol=2e-5,atol=2e-5)
        checks=0;max_error=0.0
        for n,p in m.named_parameters():
            if grads[n] is None:
                assert p.grad is None
            else:
                assert p.grad is not None
                torch.testing.assert_close(p.grad,grads[n],rtol=2e-4,atol=2e-5)
                max_error=max(max_error,float((p.grad-grads[n]).abs().max()));checks+=1
        assert memo.stats['misses']==0
        opt.step()
        for n,p in m.named_parameters():
            torch.testing.assert_close(p,expected_params[n],rtol=2e-4,atol=2e-5)
        gotopt=opt.state_dict()
        state_checks=0
        for key,state in expected_opt['state'].items():
            for field,value in state.items():
                torch.testing.assert_close(gotopt['state'][key][field],value,rtol=2e-4,atol=2e-5)
                state_checks+=1
        # Updated NN parameters consume the same cache. Give both forwards the
        # same stochastic state; restore buffers between them as well.
        updated_state=copy.deepcopy(m.state_dict());updated_rng=rng_snapshot()
        with torch.no_grad():
            rng_restore(updated_rng)
            after=runner.forward('upstream',h,r,removed)
            m.load_state_dict(updated_state);rng_restore(updated_rng)
            cached_after=adapter(h,r,removed)
            updated_error=assert_outputs(cached_after,after)
        assert memo.stats['misses']==0
        return {'passed':True,'gradient_parameters_checked':checks,'max_gradient_abs_error':max_error,
                'loss_abs_error':float((gotloss-refloss).abs().detach()),'adam_state_tensors_checked':state_checks,
                'adam_parameter_update_checked':True,'nn_parameters_changed_by_adam':changed_parameters,
                'updated_weights_cached_logits_max_abs_error':updated_error,
                'cache_misses_after_weight_update':memo.stats['misses'],
                'matched_initial_model_optimizer_rng':True,'training_mode_for_cudnn_backward':True,
                'caller_model_mode_weights_gradients_and_rng_restored':True}
    finally:
        m.load_state_dict(initial)
        for n,p in m.named_parameters():
            p.grad=None if initial_grads[n] is None else initial_grads[n].clone()
        for module,mode in initial_modes:module.training=mode
        memo.collect_stats=initial_collect_stats
        rng_restore(initial_rng)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',nargs='+',default=['FB15k-237:272','wn18rr:12'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--threads',type=int,default=16)
    p.add_argument('--build-batch',type=int,default=32)
    p.add_argument('--block-size',type=int,default=512)
    p.add_argument('--capacity-mib',type=float)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--grad',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42)
    device=torch.device('cuda',torch.cuda.current_device()) if a.device=='cuda' else torch.device('cpu')
    result={'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),
            'config':vars(a),'binding':os.getenv('BENCH_RUNTIME_BINDING'),'torch':torch.__version__,
            'scope':'fixed graph/rule exact query cache: explicit native64 CPU bulk cold build and fill; mixed/reordered B8/B32 warm complete PredictorPlus forward',
            'not_measured':['complete epoch/EM','full dataset cache sizing','cache retention across changing graph/rule versions'],
            'cases':[]}
    output=Path(a.output)
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(output)
    save()
    for case in a.cases:
        dataset,relation=case.split(':');relation=int(relation)
        runner=CaseRunner(dataset,relation,device,a.threads,['csr','native64_512'])
        try:
            build_groups=[sample(runner.g,relation,a.build_batch,'train',i,device) for i in [0,1]]
            if any(q is None for q in build_groups):raise ValueError('need two nonempty independent query groups')
            native=runner.backends['native64'];prepared=native.prepare_rules(runner.sorted_bodies)
            setup_started=time.perf_counter()
            miss_backend=SparseCPUBackendOptimized(runner.g,threads=a.threads)
            namespace=f'{dataset}:fixed-upstream-c890b025:rules-'+hashlib.sha256(repr(runner.sorted_bodies).encode()).hexdigest()[:16]
            memo=MemoizedGrounder(runner.g,runner.sorted_bodies,miss_backend,namespace=namespace,
                  capacity_bytes=None if a.capacity_mib is None else int(a.capacity_mib*1024*1024),collect_stats=False)
            def factory(g,bodies):
                return CPUInputController(memo) if tuple(map(tuple,bodies))==tuple(runner.sorted_bodies) else UnusedController()
            adapter=StreamingPredictor(runner.m,block_size=a.block_size,rule_order='prefix',grounder_factory=factory,terminal_format='sparse')
            runner.prepare('native64_512',build_groups[0][0],build_groups[0][3])
            row={'case':case,'namespace':namespace,'memo_setup_seconds':time.perf_counter()-setup_started,
                 'native_backend':native.metadata,'miss_backend':miss_backend.metadata,'cache_metadata':memo.metadata,
                 'cold_build':[],'mixed_groups':[],'checkpoint':str(runner.checkpoint)}
            result['cases'].append(row);save()
            for index,query in enumerate(build_groups):
                h,r,t,removed,target=query
                sync(device);start=time.perf_counter()
                hc=h.cpu();rc=None if removed is None else removed.cpu()
                transferred=time.perf_counter()
                terminals=native.batch(prepared,hc,relation,rc)
                built=time.perf_counter()
                prime=memo.prime_batch(hc,relation,rc,terminals)
                finished=time.perf_counter()
                row['cold_build'].append({'source_group':index,'query_count':len(h),'query_input_D2H_ms':(transferred-start)*1000,
                  'native64_grounding_ms':(built-transferred)*1000,'cache_prime_wall_ms':(finished-built)*1000,
                  'native_plus_prime_wall_ms':(finished-start)*1000,'prime':prime,
                  'native_output_payload_bytes':sum((x.numel()+v.numel())*8 for x,v in terminals)})
                del terminals
                save()
            mixes=mixed_groups(*build_groups)
            for label,query in mixes:
                h,r,t,removed,target=query
                checks=exact_counts(runner,memo,query)
                with torch.no_grad():
                    reference=runner.forward('upstream',h,r,removed)
                    memo.collect_stats=True
                    cached=adapter(h,r,removed)
                    error=assert_outputs(cached,reference)
                    cache_diag=memo.stats
                    assert cache_diag['misses']==0
                    memo.collect_stats=False
                    functions={'upstream':lambda:runner.forward('upstream',h,r,removed),
                               'gpu_or_cpu_dense_csr':lambda:runner.forward('csr',h,r,removed),
                               'native64_512':lambda:runner.forward('native64_512',h,r,removed),
                               'memoized':lambda:adapter(h,r,removed)}
                    for _ in range(a.warmup):
                        for fn in functions.values():fn();sync(device)
                    samples={k:[] for k in functions}
                    for repeat in range(a.repeats):
                        order=list(functions);random.Random(1000+repeat).shuffle(order)
                        for name in order:
                            sync(device);start=time.perf_counter();functions[name]();sync(device)
                            samples[name].append((time.perf_counter()-start)*1000)
                group={'label':label,'batch':len(h),'exact_counts':checks,'forward_max_abs_error':error,
                       'diagnostic_cache':cache_diag,'forward':{k:stats(v) for k,v in samples.items()},
                       'cached_forward_includes':'query D2H, key lookup and reconstruction, sparse block packing, H2D compact counts, NN forward, final device sync'}
                row['mixed_groups'].append(group)
                save()
                print(json.dumps({'case':case,'group':label,'ms':{k:v['median_ms'] for k,v in group['forward'].items()},'cache_entries':memo.stats['current_cache_entries']}),flush=True)
            if a.grad:row['gradient_and_adam']=train_check(runner,adapter,memo,mixes[0][1])
            row['final_cache']=memo.stats
            row['cold_total_ms']=sum(x['native_plus_prime_wall_ms'] for x in row['cold_build'])
            save()
        finally:
            runner.close()
            for backend in runner.backends.values():
                if hasattr(backend,'close'):backend.close()
            del runner
            gc.collect()
    result['complete']=True;save()

if __name__=='__main__':main()
