#!/usr/bin/env python3
"""Quantify sensitivity to query tiling without rerunning or changing GPU work."""
import argparse
from pathlib import Path
import json
from reason_model import Config, MnistProgram, simulate_microbatches, simulate_tiled_dot

ap=argparse.ArgumentParser()
ap.add_argument("root",type=Path)
ap.add_argument("cases",nargs="+")
ap.add_argument("--output",type=Path,required=True)
args=ap.parse_args()
out={}
for case in args.cases:
    r=json.loads((args.root/case/"benchmark.json").read_text())
    kg="dataset" in r
    if kg and not r["hybrid_reason"]["correctness"]["passed"]:
        raise RuntimeError("Cannot model an invalid functional replay")
    if kg:
        cfg=json.loads((args.root/case/"provenance.json").read_text())["config"]
        retained=r["hybrid_reason"]["retained_gpu_replay"]["median_ms"]
    else:
        program=MnistProgram(r["seq_len"])
        retained=sum(r[f"retained_gpu_{x}_graph"]["median_ms"] for x in ("before","after"))
    variants={}
    for tile in sorted(set(min(n,r["batch"]) for n in (1,4,8,16,64))):
        if kg:
            m=simulate_tiled_dot(cfg["state_embedding_size"],cfg["padding_states"],r["batch"],8,Config(),query_tile=tile)
            latency=retained+r["depth_budget"]*m["warm_accelerator_us"]/1000
        else:
            m=simulate_microbatches(program,r["batch"],tile)
            latency=retained+m["warm_accelerator_us"]/1000
        variants[str(tile)]={"hybrid_ms":latency,"model":m}
        print(case,tile,latency,flush=True)
    best=min(variants,key=lambda k:variants[k]["hybrid_ms"])
    out[case]={"retained_gpu_ms":retained,"variants":variants,"best_tile":int(best),
        "best_hybrid_ms":variants[best]["hybrid_ms"],
        "caveat":"Uncalibrated mapping sensitivity, same work and hardware; compiler-selected query tiling"}
with args.output.open("x") as f:json.dump(out,f,indent=2);f.write("\n")
