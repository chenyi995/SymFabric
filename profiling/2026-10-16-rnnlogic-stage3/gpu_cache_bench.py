"""Matched native-CPU vs CSR-GPU cold construction with exact real counts.

One fixed real relation per dataset is a focused cache-builder control, not an
all-relation cold-cache estimate. GPU JIT and static native bounds checks occur
before timing. CPU queries are the common start, GPU-resident count cache the
common end. Counts have no persistent cache before each measured construction.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback
from runner import Runner,torch
from pna_predictor import CachedPNAPredictor
from cached_sparse_faststats_predictor import CachedSparseFastStatsPredictor
from gpu_cache_builder import GPUCSRCountBuilder,install_gpu_preloader,terminals_to_gpu
from spa_native_adapter import SPANativeBackend
from bench_stage2 import sample
from bench_cached_sparse import gradient_check
from bench_memoized import assert_outputs
from run_bench import stats


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['wn18rr:12','FB15k-237:272'])
    p.add_argument('--batch',type=int,default=32);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',required=True);a=p.parse_args()
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('requires allocated stage-3 worker')
    torch.set_num_threads(16);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(42)
    result={'complete':False,'job':os.getenv('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),'config':vars(a),
            'scope':'focused actual cold cache construction; common CPU queries to GPU cache; exact source/count/output/all-gradient checks; not an epoch estimate',
            'gpu_validation_mode':'trusted source only after static int64 bounds proof plus full exact terminal comparison on this same query batch',
            'integer_digest_limit':'GPU direct preloader requires unique absent keys and records no exact-int64 digest; existing/repeated keys rejected',
            'cases':[]}
    path=Path(a.output)
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
    gate_root=Path(__file__).resolve().parent/'results'
    spa_gate=gate_root/'spa_native_real.json';spa_tiny=gate_root/'spa_tiny_cpu.json'
    spa_verified=spa_gate.exists() and spa_tiny.exists() and json.loads(spa_gate.read_text()).get('complete',False) and json.loads(spa_tiny.read_text()).get('passed',False)
    result['spa_control_gate']={'enabled':spa_verified,'real_result':str(spa_gate),'tiny_result':str(spa_tiny),'reason':'complete real exact-parity plus tiny checks required'}
    save();runner=None;spa_backend=None
    try:
        for case in a.cases:
            dataset,relation=case.split(':');relation=int(relation)
            runner=Runner(dataset,threads=16,pna_tile=512);device=runner.device
            query=sample(runner.g,relation,a.batch,'train',0,torch.device('cpu'))
            h,r,t,removed,target=query
            gpu_query=tuple(None if x is None else x.to(device) for x in query)
            cls=CachedPNAPredictor if runner.aggregator=='pna' else CachedSparseFastStatsPredictor
            cpu_cache=cls(runner.m,namespace=dataset+':cpu',rule_version=runner.rule_version)
            gpu_cache=install_gpu_preloader(cls(runner.m,namespace=dataset+':gpu',rule_version=runner.rule_version),
                                            trusted_unique_positive_terminals=True)
            plan=gpu_cache.rules_for_relation(relation)
            # Conservative static64 safety proof before trusting GPU arithmetic.
            plan_start=time.perf_counter()
            prepared=runner.native.prepare_rules([body for _,(_,body) in plan])
            native_plan_seconds=time.perf_counter()-plan_start
            plan_start=time.perf_counter();gpu_builder=GPUCSRCountBuilder(runner.g,runner.gpu,plan)
            gpu_plan_seconds=time.perf_counter()-plan_start
            spa_backend=SPANativeBackend(runner.g,threads=16,bits=64) if spa_verified else None
            spa_prepared=spa_backend.prepare_rules([body for _,(_,body) in plan]) if spa_backend else None
            native=runner.native.batch(prepared,h,relation,removed)
            gpu_terms,_=gpu_builder.batch(gpu_query[0],relation,gpu_query[3])
            exact_checks=0
            runner.g.propagate=runner.original_p;runner.g.relation2adjacency=runner.adj
            for (_,(_,body)),(ni,nv),(gi,gv) in zip(plan,native,gpu_terms):
                torch.testing.assert_close(gi.cpu(),ni,rtol=0,atol=0)
                torch.testing.assert_close(gv.cpu(),nv,rtol=0,atol=0)
                dense=runner.original_g(gpu_query[0],relation,body,gpu_query[3]).cpu().reshape(-1)
                index=torch.nonzero(dense).flatten()
                assert torch.equal(index,ni) and torch.equal(dense[index],nv)
                exact_checks+=1
            hybrid_terms=terminals_to_gpu(native,device)
            for (i,v),(ii,vv) in zip(gpu_terms,hybrid_terms):
                assert torch.equal(i,ii) and torch.equal(v,vv)
            if spa_backend:
                spa_terms=spa_backend.batch(spa_prepared,h,relation,removed)
                assert len(spa_terms)==len(native) and all(torch.equal(i,j) and torch.equal(v,w) for (i,v),(j,w) in zip(native,spa_terms))
                del spa_terms
            cpu_cache.preload_from_terminals(h,r,removed,native)
            gpu_cache.preload_from_terminals(h,r,removed,gpu_terms)
            assert cpu_cache.cache_statistics()['persistent_unique_storage_bytes']==gpu_cache.cache_statistics()['persistent_unique_storage_bytes']
            gpu_cache.reset(namespace=dataset+':hybrid-check',rule_version=runner.rule_version)
            gpu_cache.preload_from_terminals(h,r,removed,hybrid_terms)
            for cpu_entry,gpu_entry in zip(cpu_cache.cache.values(),gpu_cache.cache.values()):
                for field in ['candidates','crow','columns','values']:
                    assert torch.equal(getattr(cpu_entry,field),getattr(gpu_entry,field))
            with torch.no_grad():
                ref=runner.forward('upstream',gpu_query[0],gpu_query[1],gpu_query[3])
                output_error=assert_outputs(gpu_cache(gpu_query[0],gpu_query[1],gpu_query[3]),ref)
            # Bound gradient-check resources while still checking every parameter.
            gradient_query=tuple(None if x is None else x[:8] for x in gpu_query)
            gradients=gradient_check(runner,gpu_cache,gradient_query)
            row={'case':case,'batch':len(h),'model':runner.metadata,'rules':len(plan),
                 'exact_native_gpu_upstream_terminal_checks':exact_checks,'max_logit_error':output_error,
                 'gradients':gradients,'persistent_cache':gpu_cache.cache_statistics(),
                 'static_int64_intermediate_bounds':prepared[3],
                 'static_plan_setup_seconds':{'native_cpu':native_plan_seconds,'csr_gpu':gpu_plan_seconds},
                 'spa_backend':spa_backend.metadata if spa_backend else None,'native_bulk_h2d_canonical_arrays_exact':True,
                 'spa_native_terminals_exact':bool(spa_backend),'timing':{}}
            result['cases'].append(row);save()
            del native,gpu_terms,hybrid_terms,ref
            def build(kind):
                cache=cpu_cache if kind in ['native_cpu','spa_cpu'] else gpu_cache
                cache.reset(namespace=dataset+':'+kind,rule_version=runner.rule_version)
                torch.cuda.synchronize();start=time.perf_counter()
                if kind.startswith('native'):
                    terminals=runner.native.batch(prepared,h,relation,removed)
                elif kind.startswith('spa'):
                    terminals=spa_backend.batch(spa_prepared,h,relation,removed)
                else:
                    hh=h.to(device);ee=removed.to(device)
                    terminals,_=gpu_builder.batch(hh,relation,ee)
                torch.cuda.synchronize();counted=time.perf_counter()
                if kind.endswith('_gpu_pack'):
                    terminals=terminals_to_gpu(terminals,device)
                report=cache.preload_from_terminals(h,r,removed,terminals)
                torch.cuda.synchronize();end=time.perf_counter()
                return {'count_ms':(counted-start)*1000,'preload_ms':(end-counted)*1000,'total_ms':(end-start)*1000,
                        'preload':report}
            kinds=['native_cpu','native_gpu_pack','csr_gpu']+(['spa_cpu','spa_gpu_pack'] if spa_backend else [])
            for _ in range(a.warmup):
                for kind in kinds:build(kind)
            samples={kind:[] for kind in kinds}
            for repeat in range(a.repeats):
                order=list(samples);random.Random(120+repeat).shuffle(order)
                for kind in order:samples[kind].append(build(kind))
            for kind,runs in samples.items():
                row['timing'][kind]={field:stats([run[field] for run in runs]) for field in ['count_ms','preload_ms','total_ms']}
                row['timing'][kind]['last_preload']=runs[-1]['preload']
            print(json.dumps({'case':case,'exact_rule_checks':exact_checks,'cold_ms':{kind:item['total_ms']['median_ms'] for kind,item in row['timing'].items()}}),flush=True)
            save();runner.close();runner=None
            if spa_backend:spa_backend.close();spa_backend=None
            del cpu_cache,gpu_cache;gc.collect();torch.cuda.empty_cache()
        result['complete']=True;save()
    except Exception as error:
        result['error']=repr(error);result['traceback']=traceback.format_exc();save();raise
    finally:
        if runner is not None:runner.close()
        if spa_backend is not None:spa_backend.close()

if __name__=='__main__':main()
