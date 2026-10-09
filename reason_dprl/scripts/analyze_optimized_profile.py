#!/usr/bin/env python3
"""Attribute graph kernels to the NVTX scope of their actual launch API.

GPU execution can outlive a CPU NVTX scope. Attribution by launch correlation
preserves asynchronous work, unlike checking kernel timestamps against that
scope. Only the enclosing complete capture window limits GPU activity.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from profile_summary import nsys


def analyze(path):
    result=nsys(path)
    c=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    names=dict(c.execute('select id,value from StringIds'))
    lo,hi=c.execute("select start,end from NVTX_EVENTS where text='native_query'").fetchone()
    allowed={'reset','compact_policy','environment','active_index','active_pack','selection','output_update'}
    ranges=[(a,b,n) for a,b,n in c.execute('select start,end,text from NVTX_EVENTS where start>=? and end<=?',(lo,hi)) if n in allowed]
    runtime=list(c.execute('select start,end,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?',(lo,hi)))
    attribution={}
    api_total=Counter();api_count=Counter()
    stage_apis=defaultdict(Counter)
    for a,b,ident,nameid in runtime:
        label=next((n for s,e,n in ranges if s<=a and b<=e),'outside_phase')
        attribution[ident]=label
        name=names[nameid]
        api_total[name]+=b-a;api_count[name]+=1
        stage_apis[label][name]+=b-a
    kernel_total=defaultdict(Counter);kernel_counts=defaultdict(Counter)
    for a,b,ident,nameid in c.execute('select start,end,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(lo,hi)):
        label=attribution.get(ident,'unattributed')
        kernel_total[label][names[nameid]]+=b-a
        kernel_counts[label][names[nameid]]+=1
    stages={}
    for label,total in kernel_total.items():
        stages[label]={'kernel_sum_ms':sum(total.values())/1e6,
                       'kernel_count':sum(kernel_counts[label].values()),
                       'top_kernels':[{'name':n,'ms':v/1e6,'calls':kernel_counts[label][n]} for n,v in total.most_common(12)]}
    assert abs(sum(s['kernel_sum_ms'] for s in stages.values())-result['kernel_sum_ms'])<1e-6
    copies=defaultdict(lambda:{'count':0,'bytes':0,'time_ns':0})
    for a,b,kind,size in c.execute('select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(lo,hi)):
        copies[kind]['count']+=1;copies[kind]['bytes']+=size;copies[kind]['time_ns']+=b-a
    result.update(stages_by_launch_scope=stages,
                  cuda_apis=[{'name':n,'count':api_count[n],'cpu_total_ms':v/1e6} for n,v in api_total.most_common()],
                  memcpy_by_cupti_kind=dict(copies),
                  attribution='Kernel correlationId -> CUDA launch API -> enclosing CPU NVTX phase; graph-node tracing enabled')
    c.close()
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    results={p.parent.name:analyze(p) for p in sorted(args.root.glob('nsys_*_fast/native.sqlite'))}
    with args.output.open('x') as stream:
        json.dump(results,stream,indent=2);stream.write('\n')
    for name,data in results.items():
        print(name,'window/kernel_ms',data['capture_window_ms'],data['kernel_sum_ms'])
        print('stages',{n:round(v['kernel_sum_ms'],3) for n,v in data['stages_by_launch_scope'].items()})
        print('copies',data['memcpy_by_cupti_kind'])
        print('kernels_below_10us',data['kernels_below_10us'],'/',data['kernel_count'])


if __name__=='__main__':
    main()
