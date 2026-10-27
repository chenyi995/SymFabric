"""Actual shuffled upstream query stream, uncached dispatch or bounded LRU.

This is frozen-weight forward traversal, NOT a complete training epoch. Timers
include per-batch CPU tensor assembly, query H2D, lookup, miss grounding, cache
packing/admission/eviction, forward and synchronization. Sampler generation,
static graph/model setup and explicit oracle checks remain separately reported.
"""
import argparse,json,time,gc,socket,os,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from runner import Runner,torch,ROOT
from query_stream import iter_trace_batches
from cache_lru import AllocationBlockLRU
from allrelation_screen import save
from run_bench import stats

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--screen',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['uncached','cache'],required=True);p.add_argument('--epochs',type=int,default=2);p.add_argument('--budget-gib',type=float,default=8);p.add_argument('--pna-tile',type=int,default=512);p.add_argument('--check-every',type=int,default=503);p.add_argument('--max-seconds',type=float,default=5400);p.add_argument('--max-batches',type=int,default=0);p.add_argument('--builder',choices=['native','native_gpu','gpu','dispatch'],default='dispatch');a=p.parse_args()
 torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(42)
 calibration=json.loads(Path(a.screen).read_text());assert calibration['complete'],'screen must be complete before independent stream replay'
 assert calibration['config']['dataset']==a.dataset and calibration['model']['dataset']==a.dataset,'calibration dataset mismatch'
 assert calibration['config']['pna_tile']==a.pna_tile,'calibration PNA tile mismatch'
 assert calibration['config'].get('relations') is None,'stream requires complete relation coverage calibration'
 choice={};calibration_groups=defaultdict(list);calibration_scores={}
 for row in calibration['cases']:calibration_groups[row['relation']].append(row)
 for relation,rows in calibration_groups.items():
  common=set(('csr','csr_tiled','native'))
  for row in rows:
   common.intersection_update(v for v,d in row['variants'].items() if 'forward' in d and v not in row['failures'])
  scores={v:sum(row['variants'][v]['forward']['median_ms'] for row in rows)/len(rows) for v in sorted(common)}
  calibration_scores[relation]={'group_count':len(rows),'actual_batch_indices':[row['actual_batch_index'] for row in rows],'mean_of_group_medians_ms':scores,'coverage_policy':'candidate must succeed in every calibration group'}
  if scores:choice[relation]=min(scores,key=scores.get)
 build_choice={};builder_calibration={}
 for relation,rows in calibration_groups.items():
  scores={}
  for builder,count_field,preload_field,failure_field in [('native','native_build_seconds','cache_preload_seconds','cache_build'),('gpu','gpu_build_seconds','gpu_cache_preload_seconds','gpu_cache_build'),('native_gpu','native_build_seconds','native_gpu_preload_seconds','native_gpu_cache_build')]:
   if all(count_field in row and preload_field in row and failure_field not in row['failures'] for row in rows):
    scores[builder]=sum(row[count_field]+row[preload_field] for row in rows)/len(rows)
  if scores:build_choice[relation]=min(scores,key=scores.get)
  builder_calibration[relation]={'group_count':len(rows),'actual_batch_indices':[row['actual_batch_index'] for row in rows],'mean_matched_group_build_plus_preload_seconds':scores,'coverage_policy':'candidate must have successful construction and preload in every same calibration group','timing_scope':'one setup-inclusive cold construction per group; may include first-use rule-plan setup and GPU JIT; no strongest-builder claim'}
 runner=Runner(a.dataset,pna_tile=a.pna_tile,native_kind=calibration['model'].get('native_kind','stdmap'))
 assert runner.metadata['checkpoint_sha256']==calibration['model']['checkpoint_sha256'],'calibration checkpoint mismatch'
 from gpu_cache_builder import install_gpu_preloader,terminals_to_gpu
 install_gpu_preloader(runner.cached,trusted_unique_positive_terminals=True)
 manager=AllocationBlockLRU(runner.cached,int(a.budget_gib*(1<<30)),trusted_unique_positive_terminals=True) if a.mode=='cache' else None
 result={'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),'config':vars(a),'model':runner.metadata,'scope':__doc__,'selection':'per-relation fixed uncached choice from prior separate allrelation screen; calibration cost excluded and recorded in separate screen, no per-batch oracle selection','uncached_choices':choice,'count_builder_choices':build_choice,'builder_calibration_aggregation':builder_calibration,'count_builder_selection_scope':'per-relation average across matched calibration groups of one-shot setup-inclusive build plus preload timings; no claim of strongest or warm-runtime-optimal builder','calibration_aggregation':calibration_scores,'epochs':[],'oracle_checks':[],'errors':[]}
 save(a.output,result);records=Path(a.output).with_suffix('.jsonl').open('w');start_all=time.perf_counter();unsupported={};stopped=False;any_truncated=False
 try:
  for epoch in range(a.epochs):
   epoch_truncated=False
   times=[];relation_times=defaultdict(list);counters=defaultdict(int);phases=defaultdict(float);before=manager.statistics() if manager else None
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start_allocated=torch.cuda.memory_allocated();start_reserved=torch.cuda.memory_reserved();epoch_start=time.perf_counter();oracle_seconds=0;production_peak_allocated=start_allocated;production_peak_reserved=start_reserved;oracle_peak_allocated=0;oracle_peak_reserved=0
   for batch in iter_trace_batches(ROOT/f'query_stream_{a.dataset}.npz',epoch=epoch):
    if time.perf_counter()-start_all>a.max_seconds:stopped=True;break
    if a.max_batches and counters['batches']>=a.max_batches:epoch_truncated=True;any_truncated=True;break
    torch.cuda.synchronize();st=time.perf_counter();cpu,gpu=runner.inputs(batch['queries']);h,r,t,removed=gpu;rr=int(cpu[1][0]);variant=choice.get(rr,'csr_tiled' if runner.aggregator=='pna' else 'csr');admission=None
    with torch.no_grad():
     if manager is None:out=runner.forward(variant,h,r,removed)
     else:
      ts=time.perf_counter();context=manager.lookup(cpu[0],cpu[1],cpu[3]);phases['lookup_seconds']+=time.perf_counter()-ts
      counters['hits']+=context['hits'];counters['misses']+=context['misses'];fit=True
      if context['miss_indices']:
       idx=context['miss_indices'];mh,mr,md=cpu[0][idx],cpu[1][idx],cpu[3][idx]
       if rr in unsupported:fit=False;counters['guard_fallback_batches']+=1
       else:
        ts=time.perf_counter()
        try:
         builder=build_choice.get(rr,'native') if a.builder=='dispatch' else a.builder
         terminals=runner.gpu_terminals(mh,mr,md) if builder=='gpu' else runner.native_terminals(mh,mr,md)
         if builder=='native_gpu':terminals=terminals_to_gpu(terminals,runner.device)
        except OverflowError as e:unsupported[rr]=str(e);fit=False;counters['guard_fallback_batches']+=1
        torch.cuda.synchronize();phases['miss_grounding_and_terminal_placement_seconds']+=time.perf_counter()-ts
        if fit:
         ts=time.perf_counter();fit,admission=manager.admit(context,mh,mr,md,terminals);torch.cuda.synchronize();phases['admission_seconds']+=time.perf_counter()-ts;del terminals
         counters['new_payload_bytes_built']+=admission['prospective_bytes']
         if not fit:counters['capacity_fallback_batches']+=1
      if fit:out=runner.forward('cached',h,r,removed);used='cached'
      else:out=runner.forward(variant,h,r,removed);used=variant
     torch.cuda.synchronize();elapsed=time.perf_counter()-st
    times.append(elapsed*1000);relation_times[rr].append(elapsed*1000);counters['queries']+=len(h);counters['batches']+=1
    record={'epoch':epoch,'batch_index':batch['batch_index'],'relation':rr,'batch':len(h),'ms':elapsed*1000,'variant':variant if manager is None else used}
    if manager:record.update(hits=context['hits'],misses=context['misses'],cache_bytes=manager.statistics()['persistent_unique_storage_bytes'],admission=admission)
    records.write(json.dumps(record)+'\n')
    if (batch['epoch_batch_index']%a.check_every==0):
     # Peak bookkeeping is outside the measured production-batch interval.
     production_peak_allocated=max(production_peak_allocated,torch.cuda.max_memory_allocated());production_peak_reserved=max(production_peak_reserved,torch.cuda.max_memory_reserved())
     torch.cuda.reset_peak_memory_stats()
     torch.cuda.synchronize();ts=time.perf_counter()
     oracle_reference='unchanged upstream PredictorPlus';oracle_fallback=None
     try:
      with torch.no_grad():ref=runner.forward('upstream',h,r,removed)
     except torch.cuda.OutOfMemoryError as error:
      oracle_fallback=str(error)
     if oracle_fallback is not None:
      gc.collect();torch.cuda.empty_cache()
      oracle_reference='GPU CSR/tiled consumer after upstream oracle OOM'
      with torch.no_grad():ref=runner.forward('csr_tiled' if runner.aggregator=='pna' else 'csr',h,r,removed)
     error=float((ref[0]-out[0]).abs().max());ok=torch.equal(ref[1],out[1]) and torch.allclose(ref[0],out[0],atol=2e-5,rtol=2e-4)
     result['oracle_checks'].append({'epoch':epoch,'batch_index':batch['batch_index'],'relation':rr,'max_abs_error':error,'pass':bool(ok),'reference':oracle_reference,'upstream_oom':oracle_fallback})
     assert ok,('stream score mismatch',epoch,rr,error)
     del ref;torch.cuda.synchronize();oracle_seconds+=time.perf_counter()-ts
     oracle_peak_allocated=max(oracle_peak_allocated,torch.cuda.max_memory_allocated());oracle_peak_reserved=max(oracle_peak_reserved,torch.cuda.max_memory_reserved())
     torch.cuda.reset_peak_memory_stats()
    del out,cpu,gpu,h,r,t,removed
    if counters['batches']%100==0:
     records.flush();print(json.dumps({'mode':a.mode,'epoch':epoch,'batches':counters['batches'],'measured_seconds':sum(times)/1000,'hits':counters.get('hits',0),'misses':counters.get('misses',0),'cache_gib':manager.statistics()['persistent_unique_storage_bytes']/(1<<30) if manager else 0}),flush=True)
   torch.cuda.synchronize();wall=time.perf_counter()-epoch_start
   production_peak_allocated=max(production_peak_allocated,torch.cuda.max_memory_allocated());production_peak_reserved=max(production_peak_reserved,torch.cuda.max_memory_reserved())
   row={'epoch':epoch,'trace_complete':not stopped and not epoch_truncated,'batch_forward_sum_seconds':sum(times)/1000,'wall_seconds_including_trace_access_jsonl_oracles':wall,'oracle_seconds':oracle_seconds,'counter':dict(counters),'phase_seconds':dict(phases),'latency':stats(times) if times else {},'per_relation':{r:{'batches':len(v),'total_seconds':sum(v)/1000,**stats(v)} for r,v in relation_times.items()},'start_gpu_allocated_bytes':start_allocated,'start_gpu_reserved_bytes':start_reserved,'peak_gpu_allocated_bytes':max(production_peak_allocated,oracle_peak_allocated),'peak_gpu_reserved_bytes':max(production_peak_reserved,oracle_peak_reserved),'production_peak_gpu_allocated_bytes':production_peak_allocated,'production_peak_gpu_reserved_bytes':production_peak_reserved,'oracle_peak_gpu_allocated_bytes':oracle_peak_allocated,'oracle_peak_gpu_reserved_bytes':oracle_peak_reserved,'memory_scope_note':'Production allocated peaks exclude oracle temporaries; oracle peaks include still-live production outputs/cache. Allocator reserved bytes may retain free buffers from earlier oracle calls; total peaks are max(production,oracle). Peak bookkeeping is outside production timings.','end_gpu_allocated_bytes':torch.cuda.memory_allocated(),'end_gpu_reserved_bytes':torch.cuda.memory_reserved(),'cache_before':before,'cache_after':manager.statistics() if manager else None}
   result['epochs'].append(row);save(a.output,result);print(json.dumps({'epoch_complete':epoch,'seconds':sum(times)/1000,'counter':dict(counters)}),flush=True)
   if stopped:break
  if manager:result['cache_invariants']=manager.verify_invariants()
  result['unsupported_native_relations']=unsupported;result['complete']=not stopped and not any_truncated;result['time_limit_reached']=stopped;result['per_epoch_prefix_limit_reached']=any_truncated
 except Exception as e:
  result['errors'].append({'error':str(e),'traceback':traceback.format_exc()});raise
 finally:
  result['total_wall_seconds']=time.perf_counter()-start_all;records.close();save(a.output,result)
  if manager:manager.clear()
  runner.close()
if __name__=='__main__':main()
