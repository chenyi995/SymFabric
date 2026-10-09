#!/usr/bin/env python3
"""Read actual CUPTI kernels and selected NCU counters; retain profiling caveats."""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import sqlite3


def union(intervals):
    total=0
    a=b=None
    for start,end in sorted(intervals):
        if a is None:
            a,b=start,end
        elif start>b:
            total+=b-a;a,b=start,end
        else:
            b=max(b,end)
    return total+(b-a if a is not None else 0)


def nsys(path):
    conn=sqlite3.connect(f"file:{path}?mode=ro",uri=True)
    names=dict(conn.execute("select id,value from StringIds"))
    ranges=conn.execute("select start,end from NVTX_EVENTS where text='native_query'").fetchall()
    if len(ranges)!=1:
        raise ValueError(f"Expected one native_query capture, got {ranges}")
    lo,hi=ranges[0]
    kernels=conn.execute("select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?",(lo,hi)).fetchall()
    totals=Counter();counts=Counter()
    for a,b,n in kernels:
        totals[names[n]]+=b-a;counts[names[n]]+=1
    duration=sum(totals.values())
    copies=[]
    tables={t[0] for t in conn.execute("select name from sqlite_master where type='table'")}
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY","CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            copies.extend(conn.execute(f"select start,end from {table} where start>=? and end<=?",(lo,hi)).fetchall())
    result={"evidence":"Nsight Systems CUPTI GPU activities",
        "capture_window_ms":(hi-lo)/1e6,"kernel_count":len(kernels),
        "kernel_sum_ms":duration/1e6,"kernel_busy_union_ms":union([(a,b) for a,b,_ in kernels])/1e6,
        "captured_busy_fraction":union([(a,b) for a,b,_ in kernels]+copies)/(hi-lo),
        "kernels_below_10us":sum(b-a<10000 for a,b,_ in kernels),
        "copy_or_memset_count":len(copies),"copy_or_memset_sum_ms":sum(b-a for a,b in copies)/1e6,
        "top_kernels":[{"name":n,"calls":counts[n],"total_ms":t/1e6,"kernel_time_fraction":t/duration}
                       for n,t in totals.most_common(20)],
        "limitation":"Capture window includes profiler overhead; busy fraction is NOT unprofiled GPU utilization."}
    conn.close()
    return result


def ncu(path):
    text=path.read_text()
    start=text.find('"ID","Process ID"')
    if start<0:
        return {"error":"No metrics CSV header"}
    rows=list(csv.DictReader(io.StringIO(text[start:])))
    metrics=defaultdict(dict);names={}
    requested={"Duration","Compute (SM) Throughput","Memory Throughput","DRAM Throughput",
        "L1/TEX Cache Throughput","L2 Cache Throughput","Achieved Occupancy","Waves Per SM",
        "Registers Per Thread","Block Size","Grid Size"}
    for r in rows:
        if r.get("Metric Name") not in requested or not r.get("ID"):
            continue
        key=r["ID"];names[key]=r["Kernel Name"]
        try:
            val=float(r["Metric Value"].replace(",",""))
        except (ValueError,TypeError):
            continue
        metrics[key][r["Metric Name"]]={"value":val,"unit":r["Metric Unit"]}
    grouped=defaultdict(list)
    for k,v in metrics.items():grouped[names[k]].append(v)
    return {"evidence":"Nsight Compute selected kernel launches; hardware counters",
        "launch_count":len(metrics),"kernels":[{"name":n,"sampled_launches":len(v),
           "mean_metrics":{m:{"value":sum(x[m]["value"] for x in v if m in x)/sum(m in x for x in v),
                "unit":next(x[m]["unit"] for x in v if m in x)}
                for m in requested if any(m in x for x in v)}} for n,v in grouped.items()],
        "limitation":"Kernel replay changes cache/clock conditions; selected counters are not whole-workload averages."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    results={}
    for p in sorted(args.root.glob("nsys_*/native.sqlite")):
        results[p.parent.name]=nsys(p)
    for p in sorted(args.root.glob("ncu_*/counters.csv")):
        results[p.parent.name]=ncu(p)
    with args.output.open("x") as out:json.dump(results,out,indent=2);out.write("\n")
    for name,d in results.items():
        print(name,{k:v for k,v in d.items() if k in ["capture_window_ms","kernel_count","kernel_sum_ms","launch_count"]})
        for k in d.get("top_kernels",[])[:5]:print(k)
        if name.endswith("hot"):
            for k in d.get("kernels",[]):print(k)


if __name__=="__main__":main()
