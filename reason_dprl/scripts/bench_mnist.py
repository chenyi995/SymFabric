#!/usr/bin/env python3
"""Measure shipped PI scoring; validate and simulate its REASON sum-product DAG."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

from bench_common import ROOT, save, sha, digest, provenance, measure, profile, graph_measure
import numpy as np
import torch
from mnist_addition.builder import get_default_config, create_env, create_policy
from mnist_addition.algorithm import MNISTAddPIAlgo
from reason_model import MnistProgram, sensitivity, provenance as reason_provenance


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=["train","bench"])
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--iterations",type=int,default=1000)
    ap.add_argument("--repeat",type=int,default=20)
    ap.add_argument("--no-profile",action="store_true")
    ap.add_argument("--capture-only",action="store_true")
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    meta=provenance()
    meta["inference_precision"]={"dtype":"float32","matmul_tf32":False,"cudnn_tf32":False}
    cfg=get_default_config(algorithm="PI",device="cpu",seq_len=4,batch_size=args.batch,
        iterations=args.iterations,seed=42,data_path=str(ROOT/"reason_dprl/data"))
    setup=time.perf_counter()
    print("Loading MNIST on CPU",flush=True)
    env=create_env(cfg)
    cfg.device="cuda"
    env.device=torch.device("cuda")
    env.trans_mask=env.trans_mask.cuda()
    env.carry_mask=env.carry_mask.cuda()
    policy=create_policy(cfg).cuda()
    algo=MNISTAddPIAlgo(policy,env,cfg,device="cuda")
    meta["config"]=asdict(cfg)
    meta["input_sha256"]={str(p.relative_to(ROOT/"reason_dprl/data")):sha(p)
                          for p in sorted((ROOT/"reason_dprl/data").rglob("*.gz"))}
    save(args.output/"provenance.json",meta)
    def to_gpu(batch):
        return {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
    if args.action=="train":
        optimizer=torch.optim.Adam(policy.parameters(),lr=cfg.lr)
        history=[]
        start=time.perf_counter()
        for step in range(args.iterations):
            batch=to_gpu(env.reset(mode="train",counter=step))
            policy.train()
            loss=algo.compute_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step+1==args.iterations:
                value=loss.item()
                if not np.isfinite(value):
                    raise RuntimeError("Non-finite upstream training loss")
                history.append({"step":step+1,"loss":value,"elapsed_s":time.perf_counter()-start})
                print("TRAIN",history[-1],flush=True)
        torch.cuda.synchronize()
        torch.save({"policy":policy.state_dict(),"config":asdict(cfg),"iterations":args.iterations},args.output/"checkpoint.pt")
        # Upstream's prediction accuracy uses neural argmax, not the PI score.
        metrics=algo.evaluate([to_gpu(b) for b in env.reset(mode="test")])
        save(args.output/"training.json",{"history":history,"metrics":metrics,
            "setup_s":start-setup,"training_and_eval_s":time.perf_counter()-start,
            "checkpoint_sha256":sha(args.output/"checkpoint.pt"),
            "status":"locally trained profiling checkpoint; not a paper accuracy reproduction"})
        print("TRAINING_COMPLETE",flush=True)
        return
    if not args.checkpoint:
        ap.error("bench requires --checkpoint")
    checkpoint=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    # PI requires the same reversed digit order used during training.
    # Upstream builds its test queries in forward order for argmax evaluation.
    batch=to_gpu(env.reset(mode="test",counter=0))
    B,L=args.batch,cfg.seq_len
    batch["images"]=batch["images"].view(B,L,2,1,28,28).flip(1).reshape(B,2*L,1,28,28)
    batch["targets"]=batch["targets"].flip(1)
    meta.update(checkpoint_sha256=sha(args.checkpoint),checkpoint_iterations=checkpoint["iterations"],
                query_sha256=digest([batch["images"],batch["targets"]]),setup_s=time.perf_counter()-setup)
    save(args.output/"provenance.json",meta)

    def symbolic(log_probs):
        carry=algo.initialize(batch)
        total=torch.zeros(B,device="cuda")
        for t in range(L):
            carry,marginal=algo.step(carry,log_probs,batch,t)
            total=total+marginal
        return total+algo.final_check(carry,batch)
    def native():
        return symbolic(torch.log_softmax(policy(batch),-1))
    def before():
        return torch.softmax(policy(batch),-1).view(B,L,2,10)
    def after(values):
        digit_probs=values[:,:L*10].view(B,L,10)
        selected=digit_probs.gather(2,batch["targets"][:,:L,None]).squeeze(-1)
        final=values[:,L*10:].gather(1,batch["targets"][:,-1:]).squeeze(-1)
        return torch.log(selected).sum(-1)+torch.log(final)

    totals=torch.tensor([a+b+c for a in range(10) for b in range(10) for c in range(2)],device="cuda")
    digit_idx=torch.stack([torch.where(totals%10==i)[0] for i in range(10)])
    carry_idx=torch.stack([torch.where(totals//10==i)[0] for i in range(2)])
    def compact():
        p=before()
        carry=torch.zeros(B,2,device="cuda")
        carry[:,0]=1
        digits=[]
        for t in range(L):
            joint=(p[:,t,0,:,None,None]*p[:,t,1,None,:,None]*carry[:,None,None,:]).flatten(1)
            digits.append(joint[:,digit_idx].sum(-1))
            carry=joint[:,carry_idx].sum(-1)
        return after(torch.cat(digits+[carry],dim=-1))

    result={"batch":B,"seq_len":L,"contract":"shipped upstream PI marginal score, not exact joint sum probability",
            "reason":reason_provenance(),"timing_boundary":"ready GPU images and sum targets through completed per-query scores"}
    with torch.no_grad():
        eager_scores,result["gpu_eager"]=measure(native,repeat=args.repeat)
        compiled=torch.compile(native,fullgraph=True,mode="default")
        scores,result["gpu_compiled"]=measure(compiled,repeat=args.repeat)
        scores=scores.clone()
        result["compiled_eager_max_log_score_error"]=float((scores-eager_scores).abs().max())
        save(args.output/"benchmark.json",result)
        torch.testing.assert_close(scores,eager_scores,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(-eager_scores.mean(),algo.compute_loss(batch),atol=1e-6,rtol=1e-6)
        print("GPU_COMPILED",result["gpu_compiled"]["median_ms"],flush=True)
        if args.capture_only:
            save(args.output/"benchmark.json",result)
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range("native_query"):
                compiled()
                torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            return
        if not args.no_profile:
            result["profile"]=profile(compiled,args.output,"mnist_native")
        graph_scores,result["gpu_cuda_graph"]=graph_measure(compiled,repeat=args.repeat)
        torch.testing.assert_close(graph_scores,scores,atol=2e-5,rtol=2e-5)
        compact_fn=torch.compile(compact,fullgraph=True,mode="default")
        compact_scores,result["gpu_compact_control"]=measure(compact_fn,repeat=args.repeat)
        torch.testing.assert_close(compact_scores,scores,atol=2e-5,rtol=2e-5)
        graph_compact,result["gpu_compact_cuda_graph"]=graph_measure(compact_fn,repeat=args.repeat)
        torch.testing.assert_close(graph_compact,scores,atol=2e-5,rtol=2e-5)
        before_fn=torch.compile(before,fullgraph=True,mode="default")
        probabilities,result["retained_gpu_before"]=measure(before_fn,repeat=args.repeat)
        probabilities=probabilities.clone()
        graph_prob,result["retained_gpu_before_graph"]=graph_measure(before_fn,repeat=args.repeat)
        torch.testing.assert_close(graph_prob,probabilities,atol=2e-5,rtol=2e-5)
        program=MnistProgram(L)
        cpu_probabilities=probabilities.cpu().numpy()
        functional_start=time.perf_counter()
        values_cpu=program.execute(cpu_probabilities)
        result["functional_simulator_wall_s"]=time.perf_counter()-functional_start
        values=torch.from_numpy(values_cpu).cuda()
        after_fn=torch.compile(after,fullgraph=True,mode="default")
        reason_scores,result["retained_gpu_after"]=measure(lambda:after_fn(values),repeat=args.repeat)
        error=float((reason_scores-scores).abs().max())
        torch.testing.assert_close(reason_scores,scores,atol=2e-5,rtol=2e-5)
        graph_reason,result["retained_gpu_after_graph"]=graph_measure(lambda:after_fn(values),repeat=args.repeat)
        torch.testing.assert_close(graph_reason,reason_scores,atol=2e-5,rtol=2e-5)
        result["correctness"]={"passed":True,"max_abs_log_score_error":error,
             "gpu_output_sha256":digest([scores]),"reason_output_sha256":digest([reason_scores]),
             "tolerance":{"atol":2e-5,"rtol":2e-5},"query_count":B,
             "gpu_score_mean":float(scores.mean()),"reason_score_mean":float(reason_scores.mean())}
        models=sensitivity(program,B)
        retained=result["retained_gpu_before"]["median_ms"]+result["retained_gpu_after"]["median_ms"]
        retained_graph=result["retained_gpu_before_graph"]["median_ms"]+result["retained_gpu_after_graph"]["median_ms"]
        best_gpu=min(result[k]["median_ms"] for k in ("gpu_compiled","gpu_cuda_graph","gpu_compact_control","gpu_compact_cuda_graph"))
        for model in models.values():
            model.pop("instruction_trace",None)
            model["hybrid_warm_ms"]=retained+model["warm_accelerator_us"]/1000
            model["speedup_vs_gpu_compiled"]=result["gpu_compiled"]["median_ms"]/model["hybrid_warm_ms"]
            model["speedup_vs_gpu_eager"]=result["gpu_eager"]["median_ms"]/model["hybrid_warm_ms"]
            model["hybrid_graph_ms"]=retained_graph+model["warm_accelerator_us"]/1000
            model["speedup_vs_best_gpu"]=best_gpu/model["hybrid_graph_ms"]
            model["cold_with_host_mapping_ms"]=model["hybrid_warm_ms"]+1000*(program.lowering_s+model["mapping_s"])+model["configuration_us"]/1000
        result["simulations"]=models
        save(args.output/"program.json",{"ops":program.ops,"inputs":program.inputs,"outputs":program.outputs})
        np.savez(args.output/"functional_inputs_outputs.npz",probabilities=cpu_probabilities,
                 reason_values=values_cpu,gpu_scores=scores.cpu().numpy(),targets=batch["targets"].cpu().numpy())
        save(args.output/"benchmark.json",result)
        print("BENCHMARK_COMPLETE",models["nominal"]["speedup_vs_gpu_compiled"],flush=True)


if __name__=="__main__":
    main()
