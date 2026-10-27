"""Matched all-relation forward screen on actual upstream-loader queries.

One first full batch per relation (largest if no full batch). Weighted times are
stratified estimates, never reported as measured complete training epochs.
"""
import argparse,json,time,random,gc,traceback,socket,os
from pathlib import Path
import numpy as np
from runner import Runner,torch,ROOT
from query_stream import iter_trace_batches
from run_bench import stats

def save(path,data):
 p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=3);p.add_argument('--relations',nargs='*',type=int);p.add_argument('--groups',type=int,default=1);p.add_argument('--pna-tile',type=int,default=512);p.add_argument('--native-kind',choices=['stdmap','spa'],default='stdmap');a=p.parse_args()
 torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(42)
 runner=Runner(a.dataset,pna_tile=a.pna_tile,native_kind=a.native_kind)
 from gpu_cache_builder import install_gpu_preloader,terminals_to_gpu
 install_gpu_preloader(runner.cached,trusted_unique_positive_terminals=True)
 result={'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),'pinned_cpus':os.getenv('BENCH_CPU_LIST'),'config':vars(a),'model':runner.metadata,'scope':'resident frozen PredictorPlus forward on actual one-rank training-loader batches; excludes sampler, targets, transfer of queries, backward/optimizer, graph static setup and cache construction from warm forward timings; cold build separately measured','selection':'first full batches per relation in actual epoch0 stream; largest available if no full batch; no synthetic query repetition within sample','timing':'2 warmups / 3 randomized interleaved synchronized wall-clock samples by default; not full training epoch','cases':[]}
 save(a.output,result)
 groups={};counts={};sizes={}
 for b in iter_trace_batches(ROOT/f'query_stream_{a.dataset}.npz',epoch=0):
  rr=int(b['queries'][0,1]);counts[rr]=counts.get(rr,0)+1;sizes.setdefault(rr,[]).append(len(b['queries']))
  pool=groups.setdefault(rr,[])
  if len(b['queries'])==32:
   pool[:]=[x for x in pool if len(x['queries'])==32]
   if len(pool)<a.groups:pool.append(b)
  elif not pool or (len(pool[0]['queries'])<32 and len(b['queries'])>len(pool[0]['queries'])):pool[:]=[b]
 for rr in sorted(groups):
  if a.relations is not None and rr not in a.relations:continue
  for group,b in enumerate(groups[rr]):
   cpu,gpu=runner.inputs(b['queries']);h,r,t,removed=gpu
   row={'relation':rr,'group':group,'batch':len(h),'actual_batch_index':b['batch_index'],'query_sha256':b['query_sha256'],'epoch_batches_for_relation':counts[rr],'epoch_queries_for_relation':sum(sizes[rr]),'epoch_tail_size':min(sizes[rr]),'rules':len(runner.plan(rr)),'variants':{},'failures':{}}
   result['cases'].append(row);save(a.output,result)
   runner.cached.reset(namespace=a.dataset+':screen',rule_version=runner.rule_version)
   terminals=None
   try:
    st=time.perf_counter();terminals=runner.native_terminals(cpu[0],cpu[1],cpu[3]);row['native_build_seconds']=time.perf_counter()-st
    torch.cuda.synchronize();st=time.perf_counter();runner.cached.preload_from_terminals(cpu[0],cpu[1],cpu[3],terminals);torch.cuda.synchronize();row['cache_preload_seconds']=time.perf_counter()-st;row['cache']=runner.cached.cache_statistics()
   except Exception as e:row['failures']['cache_build']={'error':str(e),'traceback':traceback.format_exc()}
   if terminals is not None:
    try:
     runner.cached.reset(namespace=a.dataset+':screen',rule_version=runner.rule_version)
     torch.cuda.synchronize();st=time.perf_counter();placed_terms=terminals_to_gpu(terminals,runner.device)
     runner.cached.preload_from_terminals(cpu[0],cpu[1],cpu[3],placed_terms);torch.cuda.synchronize();row['native_gpu_preload_seconds']=time.perf_counter()-st
     row['native_gpu_cache']=runner.cached.cache_statistics();assert row['native_gpu_cache']['persistent_unique_storage_bytes']==row['cache']['persistent_unique_storage_bytes']
     del placed_terms
    except Exception as e:
     row['failures']['native_gpu_cache_build']={'error':str(e),'traceback':traceback.format_exc()}
     if 'placed_terms' in locals():del placed_terms
    try:
     runner.cached.reset(namespace=a.dataset+':screen',rule_version=runner.rule_version)
     torch.cuda.synchronize();st=time.perf_counter();gpu_terms=runner.gpu_terminals(cpu[0],cpu[1],cpu[3]);torch.cuda.synchronize();row['gpu_build_seconds']=time.perf_counter()-st
     assert len(gpu_terms)==len(terminals) and all(torch.equal(i.cpu(),j) and torch.equal(v.cpu(),w) for (i,v),(j,w) in zip(gpu_terms,terminals)),('GPU exact integer mismatch',rr)
     row['gpu_native_integer_terminals_equal']=True
     runner.cached.reset(namespace=a.dataset+':screen',rule_version=runner.rule_version)
     torch.cuda.synchronize();st=time.perf_counter();runner.cached.preload_from_terminals(cpu[0],cpu[1],cpu[3],gpu_terms);torch.cuda.synchronize();row['gpu_cache_preload_seconds']=time.perf_counter()-st
     row['gpu_cache']=runner.cached.cache_statistics();assert row['gpu_cache']['persistent_unique_storage_bytes']==row['cache']['persistent_unique_storage_bytes']
     del gpu_terms
    except Exception as e:
     row['failures']['gpu_cache_build']={'error':str(e),'traceback':traceback.format_exc()}
     if 'gpu_terms' in locals():del gpu_terms
     runner.cached.reset(namespace=a.dataset+':screen',rule_version=runner.rule_version)
     gc.collect();torch.cuda.empty_cache()
     runner.cached.preload_from_terminals(cpu[0],cpu[1],cpu[3],terminals)
   # Independently compare every native int64 terminal to upstream scatter.
   cursor=[0];original=runner.original_g
   def checked_grounding(hh,head,body,edge):
    dense=original(hh,head,body,edge)
    if terminals is not None:
     idx,val=terminals[cursor[0]];flat=dense.detach().reshape(-1).cpu();nonzero=torch.nonzero(flat).flatten()
     assert torch.equal(nonzero,idx) and torch.equal(flat[idx],val),('exact integer mismatch',rr,cursor[0])
    cursor[0]+=1
    return dense
   runner.original_g=checked_grounding
   upstream_fits=True
   try:
    with torch.no_grad():ref,refmask=runner.forward('upstream',h,r,removed)
    row['reference']='unchanged upstream PredictorPlus'
   except torch.cuda.OutOfMemoryError as e:
    row['failures']['upstream']={'error':str(e),'stage':'reference OOM'};upstream_fits=False
    runner.original_g=original;gc.collect();torch.cuda.empty_cache()
    with torch.no_grad():ref,refmask=runner.forward('csr_tiled' if runner.aggregator=='pna' else 'csr',h,r,removed)
    row['reference']='CSR with tiled consumer after upstream OOM; exact native integer terminals independently checked for completed rules'
   finally:runner.original_g=original
   row['integer_terminals_equal']=terminals is not None and cursor[0]==len(terminals)
   row['integer_rules_checked']=cursor[0]
   del terminals
   variants=(['upstream'] if upstream_fits else [])+['csr']+(['csr_tiled'] if runner.aggregator=='pna' else [])+['native']+(['cached'] if 'cache' in row else [])
   active=[]
   for v in variants:
    try:
     torch.cuda.synchronize();st=time.perf_counter()
     with torch.no_grad():pred,mask=runner.forward(v,h,r,removed)
     torch.cuda.synchronize();prep=time.perf_counter()-st
     error=float((pred-ref).abs().max());assert torch.equal(mask,refmask) and torch.allclose(pred,ref,atol=2e-5,rtol=2e-4),(v,error)
     row['variants'][v]={'max_abs_error':error,'mask_equal':True,'first_call_seconds':prep};active.append(v);del pred,mask
    except Exception as e:
     row['failures'][v]={'error':str(e),'traceback':traceback.format_exc()};gc.collect();torch.cuda.empty_cache()
   for _ in range(a.warmup):
    for v in active:
     with torch.no_grad():runner.forward(v,h,r,removed)
   times={v:[] for v in active}
   for repeat in range(a.repeats):
    order=list(active);random.Random(628+repeat).shuffle(order)
    for v in order:
     torch.cuda.synchronize();st=time.perf_counter()
     with torch.no_grad():out=runner.forward(v,h,r,removed)
     torch.cuda.synchronize();times[v].append((time.perf_counter()-st)*1000);del out
   for v in active:row['variants'][v]['forward']=stats(times[v])
   save(a.output,result);print(json.dumps({'dataset':a.dataset,'relation':rr,'group':group,'b':len(h),'rules':row['rules'],'timings_ms':{v:row['variants'][v]['forward']['median_ms'] for v in active},'failures':list(row['failures'])}),flush=True)
   del ref,refmask,h,r,t,removed,cpu,gpu;runner.cached.reset(namespace='empty',rule_version=runner.rule_version);gc.collect()
 result['complete']=True;save(a.output,result);runner.close()
if __name__=='__main__':main()
