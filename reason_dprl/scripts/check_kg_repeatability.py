#!/usr/bin/env python3
"""Audit native proof repeatability and locate the first divergent state."""
import argparse
from pathlib import Path
from bench_common import save, digest, provenance, sha
import torch
from kge_experiments.builder import KGEConfig, create_env, create_policy
from kge_experiments.ppo import PPO


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="family")
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--deterministic",action="store_true")
    ap.add_argument("--stable-candidates",action="store_true")
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.stable_candidates:
        from stable_candidates import install
        install()
    cfg=KGEConfig(dataset=args.dataset,n_envs=64,fixed_batch_size=64,device="cuda",seed=42,
                  kge_inference=False,use_callbacks=False,load_model=False)
    env=create_env(cfg); policy=create_policy(cfg)
    policy.load_state_dict(torch.load(args.checkpoint,map_location="cuda",weights_only=False)["policy"])
    policy.eval();env.eval()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    agent=PPO(policy,env,cfg,eval_only=True)
    q=cfg._components["test_queries"][:64].to("cuda").clone()
    save(args.output/"provenance.json",{**provenance(),"checkpoint_sha256":sha(args.checkpoint),
        "query_device":str(q.device),"deterministic":args.deterministic,"stable_candidates":args.stable_candidates})
    with torch.no_grad():
        native=[]
        for i in range(8):
            out=[x.clone() for x in agent.evaluate_policy(q)]
            torch.cuda.synchronize()
            native.append({"hash":digest(out),"success":out[0].tolist(),"depths":out[1].tolist()})
        print("NATIVE_HASHES",[x["hash"] for x in native],flush=True)
        runs=[]
        for i in range(3):
            obs,st=env.reset(q); obs,st=obs.clone(),st.clone()
            snapshots=[{k:v.clone() for k,v in st.items()}]
            for t in range(cfg.max_steps):
                obs,st=agent._compiled_eval_step(obs,st); obs,st=obs.clone(),st.clone()
                snapshots.append({k:v.clone() for k,v in st.items()})
            runs.append(snapshots)
        differences=[]
        for i in range(1,len(runs)):
            for t,(a,b) in enumerate(zip(runs[0],runs[i])):
                keys={k:int((a[k]!=b[k]).sum()) for k in a if not torch.equal(a[k],b[k])}
                if keys: differences.append({"run":i,"step":t,"differing_elements":keys})
        result={"native":native,"trajectory_differences":differences}
        save(args.output/"audit.json",result)
        print("DIFFERENCES",differences[:8],flush=True)


if __name__=="__main__":
    main()
