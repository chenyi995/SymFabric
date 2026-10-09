#!/usr/bin/env python3
"""Paired CPU schedule ablation: upstream matrix scan vs serial vector recurrence."""
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"softwareworkload/upstream/pytorch-struct"))
import numpy as np
import torch
from torch_struct import LinearChain
from torch_struct.semirings import LogSemiring, MaxSemiring
from baseline_torch_struct import golden


def serial(inp, mode):
    alpha = torch.zeros((inp.shape[0],inp.shape[-1]),dtype=inp.dtype)
    fold = torch.logsumexp if mode == "log" else torch.amax
    for t in range(inp.shape[1]):
        alpha = fold(inp[:,t]+alpha[:,None,:],dim=-1)
    return fold(alpha,dim=-1)


def main():
    out = Path(os.environ["SYMFABRIC_RUN_DIR"])
    rows = []
    torch.set_num_interop_threads(1)
    # claude: decision start — compare equal inference semantics before attributing scan overhead to hardware.
    for seed,states,batch,mode,threads in itertools.product((0,1),(16,48),(1,32),("log","max"),(1,8)):
        torch.set_num_threads(threads)
        arr = np.random.default_rng(seed).normal(size=(batch,31,states,states)).astype(np.float32)
        inp = torch.from_numpy(arr)
        oracle = golden(arr,mode)
        model = LinearChain(LogSemiring if mode == "log" else MaxSemiring)
        methods = {"matrix_scan":lambda:model.sum(inp), "vector_recurrence":lambda:serial(inp,mode)}
        timings = {key:[] for key in methods}
        with torch.no_grad():
            for fn in methods.values():
                np.testing.assert_allclose(fn().numpy(),oracle,atol=1e-4,rtol=1e-5)
            for trial in range(3):
                order = list(methods)
                random.Random(seed+trial).shuffle(order)
                for key in order:
                    start = time.perf_counter()
                    answer = methods[key]()
                    timings[key].append(time.perf_counter()-start)
                    np.testing.assert_allclose(answer.numpy(),oracle,atol=1e-4,rtol=1e-5)
        rows.append(dict(seed=seed,states=states,batch=batch,semiring=mode,threads=threads,
                         length=32,wall_s=timings,median_s={k:statistics.median(v) for k,v in timings.items()},
                         pass_check=True,provenance="measured paired CPU inference; synthetic potentials"))
        (out/"dp_schedule.json").write_text(json.dumps(rows,indent=2)+"\n")
    for mode in ("log","max"):
        for method in ("matrix_scan","vector_recurrence"):
            inp = torch.from_numpy(np.random.default_rng(0).normal(size=(32,31,48,48)).astype(np.float32))
            model = LinearChain(LogSemiring if mode == "log" else MaxSemiring)
            fn = (lambda:model.sum(inp)) if method == "matrix_scan" else (lambda:serial(inp,mode))
            with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                                          profile_memory=True,record_shapes=True) as prof:
                fn()
            events = [dict(key=e.key,count=e.count,self_cpu_time_us=e.self_cpu_time_total,
                           self_cpu_memory_bytes=e.self_cpu_memory_usage) for e in prof.key_averages()]
            (out/f"profile-{mode}-{method}.json").write_text(json.dumps(events,indent=2)+"\n")
            (out/f"profile-{mode}-{method}.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total",row_limit=20))
    print(json.dumps({"cases":len(rows),"validation":"PASS"}))
    # claude: decision end


if __name__ == "__main__":
    main()
