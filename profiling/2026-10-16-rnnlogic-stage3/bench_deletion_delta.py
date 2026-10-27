"""Allocated CPU-only, exact FB272 first-edge-delta materialization control."""
import argparse, hashlib, json, os, random, resource, socket, time, traceback
from pathlib import Path
import numpy as np
import torch
from query_stream import ROOT,UPSTREAM,iter_trace_batches,upstream_data,sha256_file
from spa_native_adapter import SPANativeBackend
from deletion_delta_control import FirstEdgeDelta
from test_spa_native import Graph


def equal_outputs(a,b):
    assert len(a)==len(b)
    for i,((ai,av),(bi,bv)) in enumerate(zip(a,b)):
        assert torch.equal(ai,bi) and torch.equal(av,bv),('int64 terminal mismatch',i)
        assert ai.numel()<2 or bool((ai[1:]>ai[:-1]).all())
        assert bool((av>0).all())
    return len(a)


def tiny():
    graph=Graph(); engine=SPANativeBackend(graph,threads=4)
    checks=[]
    try:
        for relation,bodies,removed in [
            (0,[(),(1,),(0,),(0,1),(0,1,1),(0,1),(0,)+(1,)*33],np.array([0,1,2,3,4,5,6,7],np.int64)),
            (1,[(),(0,),(1,),(1,0),(1,0,0),(1,0,2)],np.array([0,1,2,3],np.int64))]:
            source,dest=engine.adjacency[relation]; h=source[removed]; t=dest[removed]
            delta=FirstEdgeDelta(engine,relation,bodies,h)
            expected=engine.batch(engine.prepare_rules(bodies),h,relation,removed)
            actual=delta.batch(h,t,removed); checked=equal_outputs(expected,actual)
            # Reversed query slots and repeated queries stress base/tail lookup.
            reorder=np.array(list(range(len(h)-1,-1,-1))+[0,0],np.int64)
            equal_outputs(engine.batch(engine.prepare_rules(bodies),h[reorder],relation,removed[reorder]),
                          delta.batch(h[reorder],t[reorder],removed[reorder]))
            checks.append({'relation':relation,'batch':len(h),'rules':checked,
                'self_loops':int(np.count_nonzero(h==t)),'max_base_count':max((int(x.max()) for x in delta.owners[1::3] if len(x)),default=0)})
            wrong=t.copy(); wrong[0]=(wrong[0]+1)%engine.entity_size
            try:delta.batch(h,wrong,removed)
            except ValueError:pass
            else:raise AssertionError('bad query edge accepted')
        for body in [(1,0),(0,1,0),(0,0)]:
            try:FirstEdgeDelta(engine,0,[body],np.array([0],np.int64))
            except ValueError:pass
            else:raise AssertionError('unsupported rule accepted')
        try:FirstEdgeDelta(engine,0,[(0,)+(1,)*64],np.array([0],np.int64))
        except OverflowError:pass
        else:raise AssertionError('unsafe helper accepted')
        assert checks[0]['max_base_count']>2**24
        return {'passed':True,'checks':checks,'empty_suffix_identity':True,'duplicate_bodies_retained':True,
                'unsupported_middle_repeated_query_relation_rejected':True,'wrong_removed_edge_rejected':True,
                'int64_bound_rejection':True,'subtraction_before_float':True}
    finally:engine.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=16)
    p.add_argument('--passes',type=int,default=3);p.add_argument('--max-seconds',type=float,default=1080);a=p.parse_args()
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('allocated CPU-only worker required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.perf_counter();path=Path(a.output);engine=None
    result={'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'config':vars(a),
      'affinity':sorted(os.sched_getaffinity(0)),'pinned_linux_cpus':os.getenv('BENCH_CPU_LIST'),
      'scope':'FB272 only: six unmasked head bases, 67 suffix words, exact subtraction and full CPU int64 terminal materialization versus independent SPA64 masked counts. No GPU transfer, cache packing, neural forward, optimizer or all-relation speedup.',
      'timing':'synchronous CPU wall clock, all actual epoch0 relation272 batches in their original order; paired variants randomized within each batch; output equality outside timing. Static graph/library excluded. Base plan/count/layout measured and charged once in cold one-pass totals.',
      'storage_note':'base payload is exact owning NumPy index/int64-count/boundary arrays; excludes Python metadata, graph, native scratch and transient terminal outputs. Peak RSS includes simultaneous reference/output validation.',
      'passes':[]}
    def save():
        result['elapsed_seconds']=time.perf_counter()-start
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
    save()
    try:
        result['tiny']=tiny();save()
        dataset='FB15k-237';relation=272;directory=UPSTREAM/'data'/dataset
        trace=ROOT/f'query_stream_{dataset}.npz'
        batches=[b for b in iter_trace_batches(trace,epoch=0) if int(b['queries'][0,1])==relation]
        query=np.concatenate([b['queries'] for b in batches]);heads=np.unique(query[:,0])
        bodies=[]
        for line in (directory/'rnnlogic_rules.txt').read_text().splitlines():
            row=list(map(int,line.split()))
            if row and row[0]==relation:bodies.append(tuple(row[1:]))
        assert len(query)==1210 and len(heads)==6 and len(bodies)==276
        graph=upstream_data().KnowledgeGraph(str(directory));engine=SPANativeBackend(graph,threads=a.threads)
        setup=time.perf_counter();delta=FirstEdgeDelta(engine,relation,bodies,heads)
        base_seconds=time.perf_counter()-setup;plan=engine.prepare_rules(bodies)
        assert len(delta.affected)==67 and len(delta.suffix_plan[0])==67
        result.update(backend=engine.metadata,input_hashes={name:sha256_file(directory/name) for name in ['entities.dict','relations.dict','train.txt','rnnlogic_rules.txt']},
          trace_sha256=sha256_file(trace),relation=relation,queries=len(query),distinct_heads=heads.tolist(),distinct_tails=len(np.unique(query[:,2])),
          batch_sizes=[len(b['queries']) for b in batches],rules=len(bodies),invariant_rules=len(bodies)-len(delta.affected),affected_rules=len(delta.affected),
          distinct_suffixes=len(delta.suffix_plan[0]),base={'total_seconds':base_seconds,'plan_seconds':delta.plan_seconds,'count_seconds':delta.base_count_seconds,
          'layout_seconds':delta.base_layout_seconds,'owning_array_bytes':delta.base_storage_bytes,'nonzeros':delta.base_nnz,
          'base_bound_max':max(delta.base_plan[3]),'suffix_bound_max':max(delta.suffix_plan[3])})
        save()
        # Warm only the first real batch; its work is explicitly outside samples.
        warm=batches[0]['queries'];warm_h,warm_t,warm_r=[warm[:,i].copy() for i in [0,2,3]]
        for _ in range(2):
            engine.batch(plan,warm_h,relation,warm_r);delta.batch(warm_h,warm_t,warm_r)
        for pass_id in range(a.passes):
            rows=[];sums={'spa_masked':0.0,'base_delta':0.0};phase_sums={};checks=0
            for number,batch in enumerate(batches):
                if time.perf_counter()-start>a.max_seconds:raise TimeoutError('CPU control soft deadline')
                array=batch['queries'];h,t,removed=[array[:,i].copy() for i in [0,2,3]]
                order=['spa_masked','base_delta'];random.Random(1837+pass_id*101+number).shuffle(order)
                outputs={};timing={};phases=None
                for name in order:
                    begin=time.perf_counter()
                    outputs[name]=engine.batch(plan,h,relation,removed) if name=='spa_masked' else delta.batch(h,t,removed)
                    timing[name]=time.perf_counter()-begin;sums[name]+=timing[name]
                    if name=='base_delta':phases=dict(delta.last)
                checks+=equal_outputs(outputs['spa_masked'],outputs['base_delta'])
                nnz=sum(len(i) for i,v in outputs['base_delta'])
                for key,value in phases.items():
                    if key.endswith('_seconds'):phase_sums[key]=phase_sums.get(key,0)+value
                rows.append({'actual_batch_index':batch['batch_index'],'query_sha256':batch['query_sha256'],'batch':len(h),
                  'timing_seconds':timing,'phase':phases,'terminal_nonzeros':nnz,'terminal_payload_bytes':nnz*16})
                del outputs
            row={'pass':pass_id,'queries':len(query),'batches':len(rows),'terminal_checks':checks,
              'sum_measured_batch_seconds':sums,'base_delta_plus_once_base_seconds':sums['base_delta']+base_seconds,
              'once_base_charge_note':'same measured six-head construction is charged once to each independent one-pass comparison; repeated pass samples reused base state',
              'speedup_excluding_base':sums['spa_masked']/sums['base_delta'],
              'speedup_including_once_base':sums['spa_masked']/(sums['base_delta']+base_seconds),'phase_seconds':phase_sums,'rows':rows}
            result['passes'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)
        result['complete']=True;result['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['loaded_openmp_libraries']=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libgomp' in line})
        save()
    except Exception as error:
        result['error']=repr(error);result['traceback']=traceback.format_exc();save();raise
    finally:
        if engine:engine.close()

if __name__=='__main__':main()
