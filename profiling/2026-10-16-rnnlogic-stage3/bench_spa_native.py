"""Matched native64 std::map vs dense SPA on actual epoch-zero query batches."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import socket
import statistics
import sys
import time
import traceback
import torch
from query_stream import ROOT,UPSTREAM,iter_trace_batches,upstream_data,sha256_file
from spa_native_adapter import SPANativeBackend
from official_native_adapter import OfficialNativeBackend


def summarize(samples):
    values=sorted(samples);at=(len(values)-1)*.95;lo=int(at)
    return {'median_ms':statistics.median(samples),'p95_ms':values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(at-lo),
            'min_ms':min(samples),'max_ms':max(samples),'samples_ms':samples}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=16)
    p.add_argument('--groups',type=int,default=3);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--max-seconds',type=float,default=780);a=p.parse_args()
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('requires allocated CPU-only worker')
    if a.groups<3 or a.warmup<2 or a.repeats<5:raise ValueError('require three groups, two warmups, five repeats')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.perf_counter();output=Path(a.output)
    result={'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'config':vars(a),
            'pinned_linux_cpus':os.getenv('BENCH_CPU_LIST'),'initial_affinity':sorted(os.sched_getaffinity(0)),
            'runtime_binding':os.getenv('BENCH_RUNTIME_BINDING'),
            'openmp_env':{key:os.getenv(key) for key in ['OMP_PROC_BIND','OMP_WAIT_POLICY','GOMP_SPINCOUNT','OMP_DYNAMIC']},
            'scope':'CPU standard software control, independent exact int64 rules; no GPU, cache admission, neural forward, training or hardware estimate',
            'selection':'first three full B32 batches per selected relation in actual upstream epoch-zero trace',
            'timing':'synchronous CPU wall-clock, same prebuilt graph/rule plans and CPU tensors, two warmups and five randomized interleaved repeats; includes scratch allocation, exact grounding, terminal sort/assembly and native-to-NumPy/Torch result copies',
            'excluded':['static graph construction','library compilation','rule bound proof','query trace/tensor assembly','untimed exact terminal comparisons'],
            'cases':[],'datasets':[]}
    def save():
        result['elapsed_seconds']=time.perf_counter()-start
        temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(output)
    save();engines={}
    try:
        targets={'FB15k-237':[239,272,295,312,385,423],'wn18rr':[12,7,0,4]}
        for dataset,relations in targets.items():
            trace=ROOT/f'query_stream_{dataset}.npz';groups=defaultdict(list)
            for batch in iter_trace_batches(trace,epoch=0):
                relation=int(batch['queries'][0,1])
                if relation in relations and len(batch['queries'])==32 and len(groups[relation])<a.groups:
                    groups[relation].append(batch)
                if all(len(groups[relation])==a.groups for relation in relations):break
            if any(len(groups[relation])!=a.groups for relation in relations):raise ValueError('not enough full actual batches for '+dataset)
            setup=time.perf_counter();graph=upstream_data().KnowledgeGraph(str(UPSTREAM/'data'/dataset))
            engines={'stdmap':OfficialNativeBackend(graph,threads=a.threads,bits=64),
                     'spa':SPANativeBackend(graph,threads=a.threads,bits=64)}
            plans=defaultdict(list)
            for line in (UPSTREAM/'data'/dataset/'rnnlogic_rules.txt').read_text().splitlines():
                row=list(map(int,line.split()))
                if row:plans[row[0]].append(tuple(row[1:]))
            result['datasets'].append({'dataset':dataset,'setup_seconds':time.perf_counter()-setup,
                 'trace_sha256':sha256_file(trace),'backends':{name:engine.metadata for name,engine in engines.items()},
                 'controls_note':'WN0 and WN4 are additional controls; FB six requested expensive relations and WN12/WN7 main controls'})
            save()
            for relation in relations:
                prepared={name:engine.prepare_rules(plans[relation]) for name,engine in engines.items()}
                for number,batch in enumerate(groups[relation]):
                    if time.perf_counter()-start>a.max_seconds:raise TimeoutError('soft CPU benchmark budget reached; results remain incomplete')
                    array=batch['queries'];h=torch.from_numpy(array[:,0].copy());removed=torch.from_numpy(array[:,3].copy())
                    outputs={name:engine.batch(prepared[name],h,relation,removed) for name,engine in engines.items()}
                    checked=nonzeros=largest=0
                    digest=hashlib.sha256()
                    for (index,value),(other_index,other_value) in zip(outputs['stdmap'],outputs['spa']):
                        assert torch.equal(index,other_index) and torch.equal(value,other_value),(dataset,relation,number,checked)
                        assert index.numel()<2 or bool((index[1:]>index[:-1]).all())
                        assert bool((value>0).all())
                        digest.update(index.numpy().tobytes());digest.update(value.numpy().tobytes())
                        checked+=1;nonzeros+=value.numel()
                        if value.numel():largest=max(largest,int(value.max()))
                    assert checked==len(plans[relation])
                    row={'dataset':dataset,'relation':relation,'group':number,'batch':len(h),
                         'trace_batch_index':batch['batch_index'],'query_sha256':batch['query_sha256'],
                         'exact_terminal_rules_checked':checked,'exact_outputs_sha256':digest.hexdigest(),
                         'terminal_nonzeros':nonzeros,'max_int64_count':largest,'terminal_index_value_bytes':nonzeros*16,
                         'static_intermediate_bound_max':max(prepared['stdmap'][3],default=0)}
                    del outputs
                    def run(name):return engines[name].batch(prepared[name],h,relation,removed)
                    for _ in range(a.warmup):
                        for name in engines:run(name)
                    samples={name:[] for name in engines}
                    for repeat in range(a.repeats):
                        order=list(engines);random.Random(1031+relation*3+number*7+repeat).shuffle(order)
                        for name in order:
                            begin=time.perf_counter();out=run(name);elapsed=time.perf_counter()-begin
                            samples[name].append(elapsed*1000);del out
                    row['timing']={name:summarize(values) for name,values in samples.items()}
                    row['spa_speedup']=row['timing']['stdmap']['median_ms']/row['timing']['spa']['median_ms']
                    result['cases'].append(row);save()
                    print(json.dumps({'dataset':dataset,'relation':relation,'group':number,'nonzeros':nonzeros,
                                      'ms':{name:timing['median_ms'] for name,timing in row['timing'].items()},'spa_speedup':row['spa_speedup']}),flush=True)
            for engine in engines.values():engine.close()
            engines={};del graph;gc.collect()
        result['aggregate']={'sum_group_median_ms':{name:sum(row['timing'][name]['median_ms'] for row in result['cases']) for name in ['stdmap','spa']},
                             'scope':'unweighted sum of the selected group medians; no dataset-wide extrapolation'}
        result['aggregate']['spa_speedup']=result['aggregate']['sum_group_median_ms']['stdmap']/result['aggregate']['sum_group_median_ms']['spa']
        result['final_affinity']=sorted(os.sched_getaffinity(0));result['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['loaded_openmp_libraries']=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libgomp' in line})
        result['complete']=True;save()
    except Exception as error:
        result['error']=repr(error);result['traceback']=traceback.format_exc();save();raise
    finally:
        for engine in engines.values():engine.close()

if __name__=='__main__':main()
