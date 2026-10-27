"""Allocated-device phase profile; profiler overhead is excluded from wall timings."""
import argparse, gc, json, os, time, traceback
from pathlib import Path
from runner import Runner, torch, ROOT
from query_stream import iter_trace_batches
from gpu_cache_builder import install_gpu_preloader, terminals_to_gpu
from run_bench import stats
import pna_predictor


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True)
    ap.add_argument('--relations',type=int,nargs='+',default=[12,8,7]); args=ap.parse_args()
    if not os.getenv('SLURM_JOB_ID'): raise RuntimeError('allocated worker required')
    torch.set_num_threads(16); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42)
    result={'complete':False,'job':os.getenv('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),
      'scope':'WN18RR official emb/PNA checkpoint, first real epoch0 B32 groups; SPA cold phases and cache warm aggregation',
      'profile_note':'one diagnostic CPU+CUDA profiler iteration after warmups; timing samples are separate synchronized unprofiled calls; nested phase CPU/device times are not additive',
      'cases':[]}
    path=Path(args.output)
    def save():
        tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2)); tmp.replace(path)
    runner=Runner('wn18rr',pna_tile=2048,native_kind='spa')
    install_gpu_preloader(runner.cached,trusted_unique_positive_terminals=True)
    original=pna_predictor.pna_aggregate
    shapes=[]
    def profiled_aggregate(layer,count,emb,batch,block=128):
        shapes.append({'rules':count.size(0),'candidates':count.size(1),'hidden':emb.size(1),
                       'single_CRH_float_bytes':count.numel()*emb.size(1)*count.element_size()})
        with torch.profiler.record_function('phase_pna_aggregate'):
            return original(layer,count,emb,batch,block)
    pna_predictor.pna_aggregate=profiled_aggregate
    def profile_call(fn,label):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(); torch.cuda.synchronize()
        trace=path.parent/(path.stem+'_'+label+'.trace.json'); prof.export_chrome_trace(str(trace))
        events=[]
        for e in prof.key_averages():
            events.append({'name':e.key,'device_type':str(e.device_type),'calls':e.count,'self_cpu_us':e.self_cpu_time_total,
              'cpu_total_us':e.cpu_time_total,'device_total_us':e.device_time_total,'self_device_us':e.self_device_time_total})
        return {'trace':str(trace),'phases':[e for e in events if e['name'].startswith('phase_')],
                'top_self_cpu':sorted(events,key=lambda e:e['self_cpu_us'],reverse=True)[:20],
                'top_self_device':sorted(events,key=lambda e:e['self_device_us'],reverse=True)[:20]}
    def timed(fn):
        for _ in range(2): fn()
        times=[]
        for _ in range(5):
            torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        return stats(times)
    groups={}
    for b in iter_trace_batches(ROOT/'query_stream_wn18rr.npz',epoch=0):
        rr=int(b['queries'][0,1])
        if rr in args.relations and rr not in groups and len(b['queries'])==32: groups[rr]=b
    try:
        for rr in args.relations:
            b=groups[rr]; cpu,gpu=runner.inputs(b['queries']); h,r,t,removed=gpu
            runner.cached.reset(namespace='pna-profile',rule_version=runner.rule_version)
            terms=runner.native_terminals(cpu[0],cpu[1],cpu[3])
            runner.cached.preload_from_terminals(cpu[0],cpu[1],cpu[3],terminals_to_gpu(terms,runner.device))
            del terms
            row={'relation':rr,'batch':len(h),'actual_batch_index':b['batch_index'],'query_sha256':b['query_sha256'],
                 'cache':runner.cached.cache_statistics(),'warm':{}}
            result['cases'].append(row)
            with torch.no_grad():
                runner.cached.candidate_block_size=512
                ref=runner.forward('cached',h,r,removed); row['shape']=shapes[-1]
                for tile in [0,512,2048]:
                    if tile==0 and row['shape']['single_CRH_float_bytes']>8*1024**3:
                        row['warm']['0']={'skipped':'single CRH temporary >8GiB diagnostic memory limit'}; continue
                    runner.cached.candidate_block_size=tile
                    def warm():
                        with torch.profiler.record_function('phase_cached_total'):
                            return runner.forward('cached',h,r,removed)
                    try:
                        pred=warm(); error=float((pred[0]-ref[0]).abs().max())
                        assert torch.equal(pred[1],ref[1]) and torch.allclose(pred[0],ref[0],atol=2e-5,rtol=2e-4)
                        del pred
                        row['warm'][str(tile)]={'max_abs_error_vs512':error,'wall':timed(warm),
                            'profile':profile_call(warm,f'r{rr}_tile{tile}')}
                    except torch.cuda.OutOfMemoryError as error:
                        row['warm'][str(tile)]={'oom':str(error),'note':'single CRH bound is not total allocation bound'}
                        if 'pred' in locals(): del pred
                        gc.collect(); torch.cuda.empty_cache()
                    save()
                runner.cached.candidate_block_size=2048
                def cold():
                    with torch.profiler.record_function('phase_cold_total'):
                        runner.cached.reset(namespace='pna-profile',rule_version=runner.rule_version)
                        with torch.profiler.record_function('phase_spa_count_cpu'):
                            terms=runner.native_terminals(cpu[0],cpu[1],cpu[3])
                        with torch.profiler.record_function('phase_bulk_pack_H2D'):
                            placed=terminals_to_gpu(terms,runner.device)
                        with torch.profiler.record_function('phase_gpu_canonical_preload'):
                            runner.cached.preload_from_terminals(cpu[0],cpu[1],cpu[3],placed)
                        with torch.profiler.record_function('phase_cached_total'):
                            return runner.forward('cached',h,r,removed)
                row['cold_spa_gpu_pack_tile2048']={'wall':timed(cold),'profile':profile_call(cold,f'r{rr}_cold')}
            print(json.dumps({'relation':rr,'shape':row['shape'],'warm_ms':{k:v.get('wall',{}).get('median_ms') for k,v in row['warm'].items()},'cold_ms':row['cold_spa_gpu_pack_tile2048']['wall']['median_ms']}),flush=True)
            save(); del ref,cpu,gpu,h,r,t,removed; gc.collect()
        result['complete']=True; save()
    except Exception:
        result['error']=traceback.format_exc(); save(); raise
    finally:
        pna_predictor.pna_aggregate=original; runner.close()
if __name__=='__main__': main()
