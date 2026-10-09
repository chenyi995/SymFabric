#!/usr/bin/env python3
"""Pinned DPrL KG training, native compiled inference and diagnostic replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
import time

from bench_common import DPRL, save, sha, digest, provenance, measure, profile
import torch
from kge_experiments.builder import KGEConfig, create_env, create_policy
from kge_experiments.ppo import PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["train", "bench"])
    ap.add_argument("--dataset", choices=["family", "wn18rr"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=32768)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--ranking-queries", type=int, default=16)
    ap.add_argument("--negatives", type=int, default=100)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--skip-reason", action="store_true")
    ap.add_argument("--capture-only", action="store_true")
    ap.add_argument("--stable-candidates", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    meta = provenance()
    if args.stable_candidates:
        from stable_candidates import install
        meta["upstream_adapter"]=install()
    meta["inference_precision"]={"dtype":"float32","matmul_tf32":False,"cudnn_tf32":False}
    cfg = KGEConfig(dataset=args.dataset, n_envs=args.batch, fixed_batch_size=args.batch,
                    n_steps=64, batch_size=256, total_timesteps=args.steps, seed=42,
                    device="cuda", use_callbacks=False, use_logger=False,
                    load_model=False, save_model=False, kge_inference=False)
    meta["config"] = {f.name: getattr(cfg, f.name) for f in fields(cfg) if not f.name.startswith("_")}
    meta["input_sha256"] = {p.name: sha(p) for p in sorted((DPRL / "kge_experiments/data" / args.dataset).glob("*.txt"))}
    save(args.output / "provenance.json", meta)
    setup = time.perf_counter()
    env = create_env(cfg)
    policy = create_policy(cfg)
    print("Environment and policy ready", flush=True)
    if args.action == "train":
        agent = PPO(policy, env, cfg)
        train_start = time.perf_counter()
        result = agent.learn(args.steps)
        torch.cuda.synchronize()
        torch.save({"policy": policy.state_dict(), "config": meta["config"],
                    "timesteps": agent.num_timesteps, "seed": 42}, args.output / "checkpoint.pt")
        save(args.output / "training.json", {"result": result, "setup_s": train_start-setup,
              "training_s": time.perf_counter()-train_start,
              "checkpoint_sha256": sha(args.output / "checkpoint.pt"),
              "status": "locally trained profiling checkpoint; convergence not established"})
        print("TRAINING_COMPLETE", flush=True)
        return
    if not args.checkpoint:
        ap.error("bench requires --checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    env.eval()
    agent = PPO(policy, env, cfg, eval_only=True)
    queries = cfg._components["test_queries"][:args.batch].to("cuda").clone()
    if len(queries) != args.batch:
        raise RuntimeError("Not enough queries for requested batch")
    meta.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                checkpoint_timesteps=checkpoint["timesteps"], query_sha256=digest([queries]),
                setup_s=time.perf_counter()-setup, query_device=str(queries.device))
    save(args.output / "provenance.json", meta)
    result = {"dataset": args.dataset, "batch": args.batch, "depth_budget": cfg.max_steps,
              "timing_boundary": "ready GPU queries through reset, 20 policy/environment steps, output completion"}
    with torch.no_grad():
        native = lambda: agent.evaluate_policy(queries)
        output, result["native_compiled"] = measure(native, repeat=args.repeat)
        success, depths = [x.clone() for x in output]
        reference_digest=digest([success,depths])
        replay_digests=[digest(native()) for _ in range(5)]
        result["repeatability"]={"passed":all(h==reference_digest for h in replay_digests),
            "reference_sha256":reference_digest,"replay_sha256":replay_digests,"repeats":5}
        result["output"] = {"sha256": digest([success, depths]), "success_fraction": success.float().mean().item(),
                            "depths": depths, "success": success}
        save(args.output / "benchmark.json", result)
        print("NATIVE", result["native_compiled"]["median_ms"], flush=True)
        if args.capture_only:
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range("native_query"):
                native()
                torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            return
        if not args.no_profile:
            result["profile"] = profile(native, args.output, "kg_native")

        # Attribution ablation: stop only after every query has finished. This
        # reduces inactive work while preserving final states and search limits.
        def stop_when_finished():
            torch.compiler.cudagraph_mark_step_begin()
            obs,state=env.reset(queries)
            obs,state=obs.clone(),state.clone()
            executed=0
            for _ in range(cfg.max_steps):
                torch.compiler.cudagraph_mark_step_begin()
                obs,state=agent._compiled_eval_step(obs,state)
                obs,state=obs.clone(),state.clone()
                executed+=1
                if bool(state["done"].all()):
                    break
            return state["success"],state["depths"],executed
        early,result["stop_when_finished"] = measure(stop_when_finished,repeat=args.repeat)
        result["stop_when_finished"].update(executed_steps=early[2],
            same_output=bool(torch.equal(early[0],success) and torch.equal(early[1],depths)))

        # Separate compiled stage replay: attribution experiment, never substituted
        # for native fused timing. Inputs/actions retain the same upstream semantics.
        policy_fn = torch.compile(policy.get_logits, fullgraph=True, mode="default")
        env_fn = torch.compile(env.step, fullgraph=True, mode="default")
        def staged(record=False):
            pairs = defaultdict(list)
            events = []
            def run(name, fn):
                if not record:
                    return fn()
                a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                with torch.profiler.record_function(name), torch.cuda.nvtx.range(name):
                    a.record()
                    value = fn()
                    b.record()
                pairs[name].append((a,b))
                return value
            obs, state = run("reset", lambda: env.reset(queries))
            obs, state = obs.clone(), state.clone()
            for step in range(cfg.max_steps):
                if record:
                    # Snapshots are synchronized only after replay; no per-step D2H.
                    events.append({"step": step, "active": (~state["done"].bool()).clone(),
                                   "valid": state["derived_counts"].clone()})
                logits = run("policy", lambda: policy_fn(obs))
                actions = run("selection", lambda: logits.masked_fill(obs["action_mask"] == 0, -3.4e38).argmax(-1))
                obs, state = run("environment", lambda: env_fn(state, actions))
                obs, state = run("state_copy", lambda: (obs.clone(), state.clone()))
            return state, pairs, events
        staged_out, result["staged_compiled"] = measure(staged, repeat=args.repeat)
        st = staged_out[0]
        result["staged_matches_native"] = bool(torch.equal(st["success"], success) and torch.equal(st["depths"], depths))
        st, pairs, events = staged(True)
        torch.cuda.synchronize()
        result["stage_device_spans_ms"] = {name: sum(a.elapsed_time(b) for a,b in times) for name,times in pairs.items()}
        structural = [{"step": e["step"], "active": int(e["active"].sum()),
                       "valid_candidates": int(e["valid"][e["active"]].sum()),
                       "capacity": int(e["active"].sum()) * cfg.padding_states,
                       "allocated_capacity": args.batch * cfg.padding_states} for e in events]
        result["structure"] = structural
        result["candidate_tensor_bytes"] = args.batch * cfg.padding_states * cfg.padding_atoms * 3 * 8
        if not args.no_profile:
            result["stage_profile"] = profile(lambda: staged(True), args.output, "kg_staged")
        save(args.output / "benchmark.json", result)
        print("STAGED", result["staged_compiled"]["median_ms"], "MATCH", result["staged_matches_native"], flush=True)

        if not args.skip_reason:
            from kg_reason import run_hybrid
            result["hybrid_reason"] = run_hybrid(policy,env,queries,cfg.max_steps,success,depths,args.repeat)
            for model in result["hybrid_reason"].get("simulations",{}).values():
                model["speedup_vs_native"] = result["native_compiled"]["median_ms"]/model["complete_query_batch_ms"]
            save(args.output / "benchmark.json", result)
            print("REASON_REPLAY_COMPLETE",flush=True)

        # Paper-style sampled ranking (no KGE fusion without its checkpoint).
        sampler = cfg._components["sampler"]
        ranking_queries = cfg._components["test_queries"][:args.ranking_queries]
        def ranking():
            torch.manual_seed(100)
            torch.cuda.manual_seed_all(100)
            return agent.evaluate(ranking_queries, sampler, n_corruptions=args.negatives,
                                  corruption_modes=("head", "tail"), verbose=False)
        metrics, result["ranking_timing"] = measure(ranking, warmup=2, repeat=3)
        result["ranking"] = {"query_count": len(ranking_queries), "negatives_per_side": args.negatives,
                             "exhaustive": False, "metrics": metrics}
        # Original REASON has no demonstrated first-order unification or hash-state
        # datapath. These operations remain on the same GPU in the reconstruction.
        result["reason_symbolic_mapping"] = {"offloaded_operations": [], "fallback": "all KG symbolic work",
            "explanation": "Policy dot-product offload is accounted separately; FOL unification is unsupported"}
        save(args.output / "benchmark.json", result)
        print("BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
