#!/usr/bin/env python3
"""Launch-correlated Nsight attribution for the dynamic_validation capture."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3


def analyze(path):
    c=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    names=dict(c.execute('select id,value from StringIds'))
    windows=c.execute("select start,end from NVTX_EVENTS where text='dynamic_validation'").fetchall()
    assert len(windows)==1
    lo,hi=windows[0]
    scopes=c.execute('select start,end,text from NVTX_EVENTS where start>=? and end<=?',(lo,hi)).fetchall()
    allowed={'reset','compact_policy','environment','active_index','active_pack','selection','output_update'}
    scopes=[(s,e,n) for s,e,n in scopes if n in allowed]
    attribution={};apis=Counter();api_counts=Counter()
    for s,e,ident,name in c.execute('select start,end,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?',(lo,hi)):
        attribution[ident]=next((n for a,b,n in scopes if a<=s and e<=b),'outside_phase')
        apis[names[name]]+=e-s;api_counts[names[name]]+=1
    kernels=Counter();counts=Counter();stages=defaultdict(Counter);small=0;total=0
    for s,e,ident,name in c.execute('select start,end,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(lo,hi)):
        key=names[name];phase=attribution.get(ident,'unattributed')
        kernels[key]+=e-s;counts[key]+=1;stages[phase][key]+=e-s
        total+=1;small+=e-s<10000
    copies=defaultdict(lambda:Counter())
    for s,e,kind,size in c.execute('select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(lo,hi)):
        copies[kind].update(count=1,bytes=size,time_ns=e-s)
    c.close()
    return {'capture_window_ms':(hi-lo)/1e6,'kernel_sum_ms':sum(kernels.values())/1e6,
            'kernel_count':total,'kernels_below_10us':small,
            'stages':{n:{'kernel_sum_ms':sum(k.values())/1e6,
                         'top_kernels':[{'name':x,'ms':v/1e6} for x,v in k.most_common(10)]}
                      for n,k in stages.items()},
            'top_kernels':[{'name':n,'ms':v/1e6,'calls':counts[n]} for n,v in kernels.most_common(20)],
            'cuda_apis':[{'name':n,'cpu_ms':v/1e6,'calls':api_counts[n]} for n,v in apis.most_common(15)],
            'copies_by_cupti_kind':{str(k):dict(v) for k,v in copies.items()},
            'scope':'one instrumented invocation; kernel sums are not unprofiled wall latency; phases follow launch correlation IDs'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('databases',type=Path,nargs='+')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result={str(p):analyze(p) for p in args.databases}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for name,r in result.items():
        print(name,r['capture_window_ms'],r['kernel_sum_ms'],{k:v['kernel_sum_ms'] for k,v in r['stages'].items()})


if __name__=='__main__':
    main()
