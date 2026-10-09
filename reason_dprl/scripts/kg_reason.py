"""Functional hybrid KG replay and priced, supported sum-product offload."""
from dataclasses import replace
import time

import numpy as np
import torch
import torch.nn.functional as F

from bench_common import digest, measure
from reason_model import Config, DotProgram, simulate_tiled_dot, provenance


def run_hybrid(policy,env,queries,depth,reference_success,reference_depths,repeat=10):
    net=policy.mlp_extractor
    B,S,E=queries.shape[0],env.padding_states,net.embed_dim
    tile=8
    programs={n:DotProgram(E,n) for n in {tile,S%tile} if n}
    def prefix(obs):
        obs_emb,act_emb,mask=policy.features_extractor(obs)
        a=net.policy_head_final(net.policy_head_fused(net._get_shared_features(obs_emb)))
        b=net.policy_head_final(net.policy_head_fused(net._get_shared_features(act_emb)))
        if net.use_l2_norm:
            a,b=F.normalize(a,dim=-1),F.normalize(b,dim=-1)
        return a.squeeze(1),b
    def suffix(dots,mask):
        if net.sqrt_scale:
            dots=dots/(E**.5)
        if net.temperature is not None:
            dots=dots/net.temperature
        return dots.masked_fill(~mask.bool(),float("-inf"))
    before=torch.compile(prefix,fullgraph=True,mode="default")
    after=torch.compile(suffix,fullgraph=True,mode="default")
    step_fn=torch.compile(env.step,fullgraph=True,mode="default")
    obs,state=env.reset(queries)
    before(obs)
    torch.cuda.synchronize()
    captured=[]
    action_equal=True
    max_error=0.
    start=time.perf_counter()
    # This actually executes REASON functional values in the closed loop.
    # CPU interpretation and PCIe round trips are not called chip latency.
    for t in range(depth):
        a,b=before(obs)
        aa,bb=a.cpu().numpy(),b.cpu().numpy()
        values=[]
        for offset in range(0,S,tile):
            n=min(tile,S-offset)
            values.append(programs[n].execute(aa,bb[:,offset:offset+n]))
        raw=torch.from_numpy(np.concatenate(values,axis=1)).cuda()
        logits=after(raw,obs["action_mask"])
        reference_logits=policy.get_logits(obs)
        valid=obs["action_mask"].bool()
        max_error=max(max_error,float((logits[valid]-reference_logits[valid]).abs().max()))
        torch.testing.assert_close(logits[valid],reference_logits[valid],atol=2e-5,rtol=2e-5)
        actions=logits.argmax(-1)
        action_equal=action_equal and torch.equal(actions,reference_logits.argmax(-1))
        captured.append(raw.clone())
        obs,state=step_fn(state,actions)
        obs,state=obs.clone(),state.clone()
    torch.cuda.synchronize()
    functional_s=time.perf_counter()-start
    matches=torch.equal(state["success"],reference_success) and torch.equal(state["depths"],reference_depths)
    if not action_equal or not matches:
        return {"correctness":{"passed":False,"actions_equal":action_equal,"terminal_equal":matches,
                               "max_logit_error":max_error},"performance_claim_allowed":False}

    def retained():
        obs,state=env.reset(queries)
        obs,state=obs.clone(),state.clone()
        for raw in captured:
            # Execute retained GPU neural work; use validated fixed-trajectory
            # accelerator results to measure the residual GPU/host makespan.
            before(obs)
            actions=after(raw,obs["action_mask"]).argmax(-1)
            obs,state=step_fn(state,actions)
            obs,state=obs.clone(),state.clone()
        return state["success"],state["depths"]
    _,timing=measure(retained,repeat=repeat)
    models={}
    for name,c in {"nominal":Config(),"handoff10us":replace(Config(),handoff_us=10),
                   "clock250":replace(Config(),clock_mhz=250)}.items():
        m=simulate_tiled_dot(E,S,B,tile,c)
        m["complete_query_batch_ms"]=timing["median_ms"]+depth*m["warm_accelerator_us"]/1000
        m["offload_calls"]=depth
        models[name]=m
    return {"correctness":{"passed":True,"actions_equal":True,"terminal_equal":True,
                "max_logit_error":max_error,"output_sha256":digest([state["success"],state["depths"]])},
            "mapping":{"offloaded":"normalized policy dot products (REASON Sec. V-B SpMSpM)",
                "retained":"embeddings, MLP, normalization, selection, unification, history, state update",
                "tile_candidates":tile,"candidate_work_unchanged":True},
            "functional_cpu_emulation_s":functional_s,"retained_gpu_replay":timing,
            "simulations":models,"provenance":provenance()}
