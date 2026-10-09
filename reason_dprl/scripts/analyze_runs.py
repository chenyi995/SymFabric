#!/usr/bin/env python3
"""Generate compact tables and timeline evidence from retained run records."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def union_us(intervals):
    if not intervals:
        return 0.
    intervals=sorted(intervals)
    start,end=intervals[0]
    total=0.
    for a,b in intervals[1:]:
        if a>end:
            total+=end-start
            start,end=a,b
        else:
            end=max(end,b)
    return total+end-start


def timeline(path):
    data=json.loads(path.read_text())
    kernels=[e for e in data["traceEvents"] if e.get("cat")=="kernel" and e.get("ph")=="X"]
    copies=[e for e in data["traceEvents"] if e.get("cat") in ("gpu_memcpy","gpu_memset") and e.get("ph")=="X"]
    if not kernels:
        return {"gpu_kernels_found":False}
    labels=[e for e in data["traceEvents"] if e.get("name")==path.name.replace(".trace.json","") and e.get("ph")=="X"]
    lo=min(e["ts"] for e in kernels+labels)
    hi=max(e["ts"]+e["dur"] for e in kernels+copies+labels)
    totals=Counter()
    counts=Counter()
    for e in kernels:
        totals[e["name"]]+=e["dur"]
        counts[e["name"]]+=1
    busy=union_us([(e["ts"],e["ts"]+e["dur"]) for e in kernels+copies])
    return {"gpu_kernels_found":True,"kernel_count":len(kernels),"kernel_sum_ms":sum(e["dur"] for e in kernels)/1000,
            "window_ms":(hi-lo)/1000,"gpu_busy_union_ms":busy/1000,"gpu_busy_fraction":busy/(hi-lo),
            "kernels_below_10us":sum(e["dur"]<10 for e in kernels),
            "memcpy_count":len(copies),"memcpy_sum_ms":sum(e["dur"] for e in copies)/1000,
            "top_kernels":[{"name":k,"total_ms":v/1000,"calls":counts[k]} for k,v in totals.most_common(15)]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("run_root",type=Path)
    ap.add_argument("cases",nargs="+")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--mapping",type=Path)
    ap.add_argument("--profiles",type=Path)
    ap.add_argument("--graph",action="append",default=[],help="dataset=graph-run-directory")
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    details={}
    mappings=json.loads(args.mapping.read_text()) if args.mapping else {}
    graphs=dict(x.split("=",1) for x in args.graph)
    for name in args.cases:
        directory=args.run_root/name
        result=json.loads((directory/"benchmark.json").read_text())
        meta=json.loads((directory/"provenance.json").read_text())
        kg="dataset" in result
        gpu=result["native_compiled"] if kg else result["gpu_compiled"]
        hybrid=result.get("hybrid_reason",{}) if kg else result
        model=hybrid.get("simulations",{}).get("nominal",{})
        reason_ms=model.get("complete_query_batch_ms") if kg else model.get("hybrid_graph_ms",model.get("hybrid_warm_ms"))
        controls={"native_compiled":gpu}
        if not kg:
            controls.update({k:result[k] for k in ("gpu_cuda_graph","gpu_compact_control","gpu_compact_cuda_graph") if k in result})
        elif result["dataset"] in graphs:
            gp=args.run_root/graphs[result["dataset"]]
            gm=json.loads((gp/"provenance.json").read_text())
            gr=json.loads((gp/"benchmark.json").read_text())
            if gm["checkpoint_sha256"]!=meta["checkpoint_sha256"] or gm["query_sha256"]!=meta["query_sha256"]:
                raise RuntimeError(f"Graph control has different inputs or checkpoint: {gp}")
            if gr["matches_native"]:
                controls["cuda_graph"]=gr["cuda_graph"]
        best_kind=min(controls,key=lambda k:controls[k]["median_ms"])
        best=controls[best_kind]
        tiled_ms=mappings.get(name,{}).get("best_hybrid_ms")
        row={"case":name,"batch":result["batch"],"gpu_median_ms":gpu["median_ms"],
             "gpu_p95_ms":gpu["p95_ms"],"gpu_queries_per_s":1000*result["batch"]/gpu["median_ms"],
             "reason_model_ms":reason_ms,
             "nominal_model_speedup_vs_best_gpu":best["median_ms"]/reason_ms if reason_ms else None,
             "best_gpu_control":best_kind,"best_gpu_ms":best["median_ms"],"best_gpu_p95_ms":best["p95_ms"],
             "tiled_reason_model_ms":tiled_ms,"speedup_vs_best_gpu":best["median_ms"]/tiled_ms if tiled_ms else None,
             "gpu_peak_allocated_mib":gpu["peak_allocated_bytes"]/2**20,
             "slurm_job_id":meta["slurm_job_id"],"checkpoint_sha256":meta["checkpoint_sha256"]}
        rows.append(row)
        detail={"correctness":hybrid.get("correctness"),"repeatability":result.get("repeatability"),
                "provenance":meta,"timelines":{},"gpu_controls":controls,"query_tiling":mappings.get(name)}
        for p in directory.glob("*.trace.json"):
            detail["timelines"][p.name]=timeline(p)
        if kg:
            structure=result["structure"]
            total_capacity=sum(s["allocated_capacity"] for s in structure)
            active_capacity=sum(s["capacity"] for s in structure)
            detail.update(stage_spans_ms=result["stage_device_spans_ms"],
                staged_compiled_ms=result["staged_compiled"]["median_ms"],
                valid_to_allocated_fraction=sum(s["valid_candidates"] for s in structure)/total_capacity,
                valid_to_active_capacity_fraction=sum(s["valid_candidates"] for s in structure)/active_capacity,
                active_lane_fraction=active_capacity/total_capacity,
                output=result["output"],stop_when_finished=result.get("stop_when_finished"),
                ranking=result.get("ranking"),ranking_timing=result.get("ranking_timing"),
                structure=structure,reason_mapping=hybrid.get("mapping"),staged_matches_native=result["staged_matches_native"])
        else:
            detail.update(eager_ms=result["gpu_eager"]["median_ms"],
                compact_gpu_ms=result["gpu_compact_control"]["median_ms"],
                retained_gpu_before_ms=result["retained_gpu_before"]["median_ms"],
                retained_gpu_after_ms=result["retained_gpu_after"]["median_ms"],
                model_execution_us=model["execution_us"],model_handoff_us=model["handoff_us"],
                model_counts=model["counts"],model_issue_delays=model["issue_delay_cycles"])
        detail["sensitivity"]={k:{f:v for f,v in m.items() if f in (
            "hybrid_warm_ms","complete_query_batch_ms","execution_us","handoff_us",
            "speedup_vs_gpu_compiled","speedup_vs_native","configuration_us","mapping_s")}
            for k,m in hybrid.get("simulations",{}).items()}
        details[name]=detail
    with (args.output/"summary.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (args.output/"analysis.json").write_text(json.dumps(details,indent=2)+"\n")
    if args.profiles:
        (args.output/"profiles.json").write_text(args.profiles.read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(9,4.5))
    x=np.arange(len(rows))
    ax.bar(x-.25,[r["best_gpu_ms"] for r in rows],.25,label="GPU measured: best validated control",color="#3264a8")
    ax.bar(x,[r["tiled_reason_model_ms"] or float("nan") for r in rows],.25,label="Hybrid model: query tiling",color="#cf873b",hatch="//")
    ax.bar(x+.25,[r["reason_model_ms"] or float("nan") for r in rows],.25,label="Hybrid model: full-batch mapping",color="#b9a17e",hatch="..")
    ax.set_xticks(x,[r["case"] for r in rows],rotation=15,ha="right")
    ax.set_ylabel("Complete batch latency (ms, log scale)")
    ax.set_yscale("log");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(args.output/"latency.png",dpi=180);fig.savefig(args.output/"latency.pdf")
    kg_details=[(n,d) for n,d in details.items() if "structure" in d]
    if kg_details:
        fig,axes=plt.subplots(1,len(kg_details),figsize=(10,3.6),squeeze=False)
        for ax,(name,d) in zip(axes[0],kg_details):
            s=d["structure"]
            ax.plot([x["step"] for x in s],[100*x["capacity"]/x["allocated_capacity"] for x in s],label="Active query lanes")
            ax.plot([x["step"] for x in s],[100*x["valid_candidates"]/x["allocated_capacity"] for x in s],label="Valid active candidates")
            ax.set(xlabel="Proof step",ylabel="Allocated dense capacity (%)",title=name,ylim=(0,105))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(args.output/"candidate_work.png",dpi=180);fig.savefig(args.output/"candidate_work.pdf")
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
