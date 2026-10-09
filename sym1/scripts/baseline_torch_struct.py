#!/usr/bin/env python3
"""CPU Torch-Struct inference, independent NumPy golden and operator profiler."""
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"softwareworkload/upstream/pytorch-struct"))
import numpy as np
import torch
from torch_struct import LinearChain
from torch_struct.semirings import LogSemiring, MaxSemiring


def golden(potentials, mode):
    a = np.zeros((potentials.shape[0], potentials.shape[-1]), dtype=np.float64)
    fold = np.logaddexp.reduce if mode == "log" else np.max
    for t in range(potentials.shape[1]):
        a = fold(potentials[:, t].astype(np.float64)+a[:, None, :], axis=-1)
    return fold(a, axis=-1)


def main():
    out = Path(os.environ["SYMFABRIC_RUN_DIR"])
    torch.set_num_interop_threads(1)
    rows = []
    # claude: decision start — synthetic potential tensors isolate DP cost and batch scaling.
    # These shapes and repetition counts are experimental controls, not benchmark-dataset claims.
    for seed, states, batch, mode, threads in itertools.product((0, 1), (16, 48), (1, 8, 32),
                                                              ("log", "max"), (1, 8)):
        torch.set_num_threads(threads)
        array = np.random.default_rng(seed).normal(size=(batch, 31, states, states)).astype(np.float32)
        inp = torch.from_numpy(array)
        model = LinearChain(LogSemiring if mode == "log" else MaxSemiring)
        reference = golden(array, mode)
        with torch.no_grad():
            actual = model.sum(inp).detach().numpy()
            np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-5)
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                actual = model.sum(inp)
                samples.append(time.perf_counter()-start)
        row = {"seed": seed, "states": states, "batch": batch, "length": 32, "semiring": mode,
               "threads": threads, "wall_s": samples, "median_s": statistics.median(samples),
               "max_abs_error": float(np.max(np.abs(actual.detach().numpy()-reference))), "pass": True,
               "input_bytes": array.nbytes, "torch_version": torch.__version__,
               "provenance": "measured CPU inference; synthetic potentials; no backward"}
        rows.append(row)
        (out/"torch_struct.json").write_text(json.dumps(rows, indent=2)+"\n")
        print(json.dumps(row), flush=True)
    # claude: decision end
    torch.set_num_threads(8)
    inp = torch.from_numpy(np.random.default_rng(0).normal(size=(32,31,48,48)).astype(np.float32))
    for mode, semiring in (("log", LogSemiring), ("max", MaxSemiring)):
        with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                                      record_shapes=True, profile_memory=True) as prof:
            LinearChain(semiring).sum(inp)
        (out/f"profile-{mode}.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        events = [{"key": e.key, "count": e.count, "self_cpu_time_us": e.self_cpu_time_total,
                   "cpu_time_us": e.cpu_time_total, "self_cpu_memory_bytes": e.self_cpu_memory_usage}
                  for e in prof.key_averages()]
        (out/f"profile-{mode}.json").write_text(json.dumps(events, indent=2)+"\n")
        prof.export_chrome_trace(str(out/f"profile-{mode}.trace.json"))


if __name__ == "__main__":
    main()
