"""Actual selected-relation training-key cache census with a strict byte budget.

Enumerates real unique (h, relation, removed-edge) keys. It does not extrapolate
from 64 queries or claim that all relations fit. Every GPU cache upload has a
CPU-computed payload bound before allocation. Native count building and cache
filling are measured separately; capacity-only accounting overhead is separate.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import traceback

from bench_stage2 import CaseRunner, sync
from cached_sparse_predictor import CachedSparsePredictor
import torch


def actual_train_keys(graph, relation):
    seen=set();keys=[];facts=0
    for head, rel, tail in graph.train_facts:
        if rel!=relation:continue
        facts+=1
        removed=graph.relation2ht2index[relation][graph.encode_ht(head,tail)]
        key=(int(head),int(removed))
        if key not in seen:
            seen.add(key);keys.append((int(head),int(tail),int(removed)))
    return facts,keys


def subset_terminals(terminals, keep, entities):
    bound=keep*entities
    out=[]
    for ids,values in terminals:
        mask=ids<bound
        out.append((ids[mask],values[mask]))
    return out


def spotcheck(runner, adapter, keys, device):
    if not keys:return {'skipped':'no loaded keys'}
    positions=sorted({0,len(keys)//2,len(keys)-1})
    selected=[keys[index] for index in positions]
    h=torch.tensor([key[0] for key in selected],device=device,dtype=torch.int64)
    r=torch.full_like(h,runner.relation)
    removed=torch.tensor([key[2] for key in selected],device=device,dtype=torch.int64)
    with torch.no_grad():
        reference,refmask=runner.forward('upstream',h,r,removed)
        output,mask=adapter(h,r,removed)
    assert torch.equal(mask,refmask)
    torch.testing.assert_close(output,reference,atol=2e-5,rtol=2e-4)
    return {'passed':True,'loaded_key_positions':positions,
            'queries':[{'h':key[0],'t':key[1],'removed':key[2]} for key in selected],
            'complete_forward_max_abs_error':float((output-reference).abs().max()),'mask_equal':True,
            'scope':'full original PredictorPlus logits/mask for first/middle/last loaded training key; no repeated all-key upstream integer audit'}


def exact_digest_audit(adapter):
    groups={}
    total_payload=0
    for entry in adapter.cache.values():
        payload=sum(t.numel()*t.element_size() for t in (entry.candidates,entry.crow,entry.columns,entry.values))
        total_payload+=payload
        if entry.integer_digest not in groups:
            groups[entry.integer_digest]={'payload_bytes':payload,'keys':0,'heads':set()}
        group=groups[entry.integer_digest]
        assert group['payload_bytes']==payload
        group['keys']+=1;group['heads'].add(entry.key[2])
    unique_payload=sum(group['payload_bytes'] for group in groups.values())
    repeated=sorted(({'integer_digest':digest,'keys':group['keys'],'distinct_heads':len(group['heads']),
                     'one_matrix_payload_bytes':group['payload_bytes']} for digest,group in groups.items() if group['keys']>1),
                    key=lambda row:row['keys'],reverse=True)
    return {'cached_keys':len(adapter.cache),'distinct_heads':len({entry.key[2] for entry in adapter.cache.values()}),
            'distinct_exact_integer_terminal_matrices':len(groups),
            'current_per_key_payload_bytes':total_payload,'potential_exact_matrix_dedup_payload_bytes':unique_payload,
            'potential_duplicate_payload_bytes':total_payload-unique_payload,
            'keys_sharing_another_exact_matrix':len(adapter.cache)-len(groups),'largest_repeated_groups':repeated[:10],
            'actual_dedup_implemented':False,
            'note':'potential software content dedup only; actual GPU storage remains cache.persistent_unique_storage_bytes; fingerprint compares canonical int64 counts before float conversion; no 24GB capacity extrapolation'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['FB15k-237:272','wn18rr:12'])
    parser.add_argument('--threads',type=int,default=16)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--budget-mib',type=int,default=768)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.batch<=0 or args.budget_mib<=0:parser.error('batch and budget must be positive')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',torch.cuda.current_device())
    limit=args.budget_mib*2**20
    result={'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),
        'gpu':torch.cuda.get_device_name(device),'torch':torch.__version__,
        'config':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        'scope':'actual complete chosen-relation training-key census, subject to explicit GPU count-cache storage budget; no full-dataset/all-relation extrapolation',
        'ordering':'first occurrence in upstream graph.train_facts, deduplicated by (h,removed-edge) within the chosen relation',
        'budget_definition':'unique backing bytes of GPU cached candidates, CSR rowptr/col/value tensors; graph/model/allocator reserve and warm-forward workspace are separate',
        'upload_bound':'12*nnz + 16*active_candidates + 8*query_count bytes; active_candidates<=min(nnz,batch*entities), tightened exactly on CPU near limit; only an admissible query prefix is uploaded',
        'cold_timing':'native count and adapter cache-fill separately measured; extra capacity-only accounting is separately reported and excluded from cold_build_plus_fill_seconds',
        'notes':['native integer correctness already tested independently; selected first/middle/last cached forwards rechecked here',
                 'cold cost is this implemented native64+CPU CSR conversion+H2D path, not a claimed minimum',
                 'a budget-limited result reports observed keys and storage, never a linear estimate of the remainder'],
        'cases':[]}
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(args.output)
    save()
    try:
        for spec in args.cases:
            ds,rr=spec.split(':');relation=int(rr)
            gc.collect();torch.cuda.empty_cache()
            setup_start=time.perf_counter()
            runner=CaseRunner(ds,relation,device,args.threads,['native64_512'])
            native=runner.backends['native64'];graph=runner.g
            namespace=f'{ds}:c890b025:static-graph'
            version=hashlib.sha256(repr(runner.entries).encode()).hexdigest()
            adapter=CachedSparsePredictor(runner.m,namespace=namespace,rule_version=version,rule_order='prefix')
            plan=adapter.rules_for_relation(relation);bodies=[body for _,(_,body) in plan]
            prepared=native.prepare_rules(bodies)
            fact_count,keys=actual_train_keys(graph,relation)
            key_digest=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()
            sync(device)
            row={'case':spec,'relation':relation,'entity_count':graph.entity_size,'rule_count':len(plan),
                 'train_fact_count':fact_count,'unique_train_key_count':len(keys),'duplicate_train_keys_removed':fact_count-len(keys),
                 'unique_key_list_sha256':key_digest,'rule_plan_fingerprint':adapter.rule_fingerprint,
                 'native_metadata':native.metadata,'setup_seconds':time.perf_counter()-setup_start,
                 'initial_gpu_allocated_bytes':torch.cuda.memory_allocated(device),
                 'loaded_keys':0,'coverage_fraction':0.,'fully_cached_selected_relation':False,
                 'native_count_seconds':0.,'cache_fill_seconds':0.,'capacity_accounting_seconds':0.,
                 'cold_build_plus_fill_seconds':0.,'chunks':[]}
            result['cases'].append(row);save()
            loaded=[];stop_reason='all_actual_training_keys_loaded'
            for offset in range(0,len(keys),args.batch):
                selected=keys[offset:offset+args.batch]
                h=torch.tensor([key[0] for key in selected],dtype=torch.int64)
                r=torch.full_like(h,relation)
                removed=torch.tensor([key[2] for key in selected],dtype=torch.int64)
                sync(device);build_start=time.perf_counter()
                terminals=native.batch(prepared,h,relation,removed)
                native_seconds=time.perf_counter()-build_start
                row['native_count_seconds']+=native_seconds
                accounting_start=time.perf_counter()
                current=adapter.cache_statistics()['persistent_unique_storage_bytes']
                nnz=sum(values.numel() for _,values in terminals)
                upper_candidates=min(nnz,len(selected)*graph.entity_size)
                upper_bytes=12*nnz+16*upper_candidates+8*len(selected)
                keep=len(selected);exact=False;actual_candidates=None;admissible_bytes=upper_bytes
                per_key_payload=None
                if current+upper_bytes>limit:
                    exact=True
                    flat=torch.cat([ids for ids,_ in terminals]) if terminals else torch.empty(0,dtype=torch.int64)
                    unique=torch.unique(flat)
                    nonzeros_per_query=torch.bincount(flat//graph.entity_size,minlength=len(selected))
                    candidates_per_query=torch.bincount(unique//graph.entity_size,minlength=len(selected))
                    per_key_payload=(12*nonzeros_per_query+16*candidates_per_query+8).tolist()
                    actual_candidates=unique.numel()
                    running=0;keep=0
                    for payload in per_key_payload:
                        if current+running+payload>limit:break
                        running+=payload;keep+=1
                    admissible_bytes=running
                    del flat,unique,nonzeros_per_query,candidates_per_query
                capacity_seconds=time.perf_counter()-accounting_start
                row['capacity_accounting_seconds']+=capacity_seconds
                chunk={'source_key_offset':offset,'native_queries_computed':len(selected),'queries_admitted':keep,
                       'native_count_seconds':native_seconds,'terminal_nnz_computed':nnz,
                       'current_cache_bytes_before':current,'initial_payload_upper_bound_bytes':upper_bytes,
                       'capacity_bound_refined_exactly':exact,'admitted_payload_bound_bytes':admissible_bytes,
                       'exact_chunk_candidate_count_if_refined':actual_candidates,
                       'capacity_accounting_seconds':capacity_seconds}
                if not keep:
                    chunk['cache_fill_seconds']=0.;chunk['stop_reason']='next_actual_query_does_not_fit_budget'
                    row['chunks'].append(chunk);stop_reason=chunk['stop_reason'];del terminals;save();break
                if keep<len(selected):
                    filter_start=time.perf_counter()
                    terminals=subset_terminals(terminals,keep,graph.entity_size)
                    extra=time.perf_counter()-filter_start
                    chunk['capacity_accounting_seconds']+=extra;row['capacity_accounting_seconds']+=extra
                    h=h[:keep];r=r[:keep];removed=removed[:keep]
                assert current+admissible_bytes<=limit
                sync(device);fill_start=time.perf_counter()
                fill=adapter.preload_from_terminals(h,r,removed,terminals)
                sync(device);fill_seconds=time.perf_counter()-fill_start
                cache=adapter.cache_statistics();after=cache['persistent_unique_storage_bytes']
                assert after<=limit,(after,limit)
                assert after-current<=admissible_bytes,(after-current,admissible_bytes)
                assert fill['new_keys']==keep,(fill['new_keys'],keep)
                loaded.extend(selected[:keep]);del terminals
                chunk.update(cache_fill_seconds=fill_seconds,cache_fill=fill,actual_storage_growth_bytes=after-current,
                             cache_bytes_after=after,preallocation_bound_verified=True)
                row['cache_fill_seconds']+=fill_seconds
                row['cold_build_plus_fill_seconds']=row['native_count_seconds']+row['cache_fill_seconds']
                row['chunks'].append(chunk);row['loaded_keys']=len(loaded)
                row['coverage_fraction']=len(loaded)/len(keys) if keys else 1.
                row['cache']=cache;row['gpu_allocated_bytes_after_fill']=torch.cuda.memory_allocated(device)
                save()
                print(json.dumps({'case':spec,'loaded':len(loaded),'total_unique':len(keys),'cache_MiB':after/2**20,
                    'cold_native_plus_fill_s':row['cold_build_plus_fill_seconds'],'capacity_cpu_s':row['capacity_accounting_seconds']}),flush=True)
                if keep<len(selected):
                    stop_reason='remaining_actual_query_prefix_exceeds_budget';break
            row['cold_build_plus_fill_seconds']=row['native_count_seconds']+row['cache_fill_seconds']
            row['fully_cached_selected_relation']=len(loaded)==len(keys)
            row['stop_reason']=stop_reason
            row['loaded_keys']=len(loaded);row['coverage_fraction']=len(loaded)/len(keys) if keys else 1.
            row['cache']=adapter.cache_statistics()
            row['unloaded_actual_keys']=len(keys)-len(loaded)
            row['keys_computed_but_not_cached']=sum(chunk['native_queries_computed']-chunk['queries_admitted'] for chunk in row['chunks'])
            row['exact_matrix_digest_audit']=exact_digest_audit(adapter)
            row['correctness_spotcheck']=spotcheck(runner,adapter,loaded,device)
            row['complete']=True;save()
            print(json.dumps({'case':spec,'complete_selected_relation':row['fully_cached_selected_relation'],
                'loaded':len(loaded),'total':len(keys),'cache_MiB':row['cache']['persistent_unique_storage_bytes']/2**20,
                'cold_native_plus_fill_s':row['cold_build_plus_fill_seconds'],'stop_reason':stop_reason}),flush=True)
            runner.close();native.close()
            del adapter,native,graph,runner,loaded
            gc.collect();torch.cuda.empty_cache()
        result['complete']=True;save()
    except Exception:
        result['failure']=traceback.format_exc();save();raise


if __name__=='__main__':main()
