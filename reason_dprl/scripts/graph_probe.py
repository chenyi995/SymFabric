#!/usr/bin/env python3
"""Controlled CUDA Graph replay ablation for upstream's fixed-depth proof API."""
import argparse
from pathlib import Path
import time

from bench_common import provenance,save,sha,digest,measure
import torch
from kge_experiments.builder import KGEConfig,create_env,create_policy
from kge_experiments.ppo import PPO


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--stable-candidates",action="store_true")
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    adapter=None
    if args.stable_candidates:
        from stable_candidates import install
        adapter=install()
    cfg=KGEConfig(dataset=args.dataset,n_envs=args.batch,fixed_batch_size=args.batch,
                  device="cuda",seed=42,kge_inference=False,use_callbacks=False,load_model=False)
    env=create_env(cfg)
    policy=create_policy(cfg)
    policy.load_state_dict(torch.load(args.checkpoint,map_location="cuda",weights_only=False)["policy"])
    policy.eval();env.eval()
    agent=PPO(policy,env,cfg,eval_only=True)
    queries=cfg._components["test_queries"][:args.batch].to("cuda").clone()
    meta=provenance()
    meta.update(checkpoint_sha256=sha(args.checkpoint),query_sha256=digest([queries]),upstream_adapter=adapter)
    save(args.output/"provenance.json",meta)
    with torch.no_grad():
        fn=lambda:agent.evaluate_policy(queries)
        expected,native=measure(fn)
        expected=[x.clone() for x in expected]
        result={"native_compiled":native,"batch":args.batch,"dataset":args.dataset}
        save(args.output/"benchmark.json",result)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        start=time.perf_counter()
        with torch.cuda.graph(graph):
            outputs=fn()
        torch.cuda.synchronize()
        result["capture_s"]=time.perf_counter()-start
        def replay():
            graph.replay()
            return outputs
        actual,result["cuda_graph"]=measure(replay,repeat=20)
        result["matches_native"]=all(torch.equal(a,b) for a,b in zip(actual,expected))
        result["speedup"]=native["median_ms"]/result["cuda_graph"]["median_ms"]
        result["output_sha256"]=digest(actual)
        result["scope"]="same resident queries, weights, work and depth budget; compiled kernels replayed as one CUDA Graph"
        save(args.output/"benchmark.json",result)
        print("GRAPH_ABLATION",result["cuda_graph"]["median_ms"],result["speedup"],result["matches_native"],flush=True)


if __name__=="__main__":
    main()
