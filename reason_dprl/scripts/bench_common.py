"""Shared provenance and timing helpers for real Slurm GPU experiments."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(os.environ.get("SYMFABRIC_REPO_ROOT",Path(__file__).resolve().parents[2])).resolve()
DPRL = ROOT / "softwareworkload/upstream/DPrL-AAAI"
PIN = "cfc66233c7aca3677cfb4ae74a71fdf9559ed89c"
sys.path.insert(0, str(DPRL))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, default=str) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def provenance():
    import torch
    revision = subprocess.check_output(["git", "-C", str(DPRL), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise RuntimeError(f"DPrL source mismatch: {revision}")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a real allocated CUDA GPU")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run GPU experiments through Slurm")
    return {
        "dprl_commit": revision, "python": sys.version, "torch": torch.__version__,
        "cuda": torch.version.cuda, "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": str(torch.cuda.get_device_properties(0)),
        "argv": sys.argv, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harness_sha256": {p.name:sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }


def stats(samples):
    ordered = sorted(samples)
    def percentile(p):
        index = (len(ordered) - 1) * p
        lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples), "min_ms": min(samples),
            "p95_ms": percentile(.95), "max_ms": max(samples)}


def measure(fn, warmup=3, repeat=10):
    import torch
    start = time.perf_counter()
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - start
    samples, device_samples = [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeat):
        torch.cuda.synchronize()
        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start = time.perf_counter()
        begin.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        device_samples.append(begin.elapsed_time(end))
    return result, {**stats(samples), "device_span": stats(device_samples),
                    "warmup_s": warmup_s, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def profile(fn, output, label):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=True) as prof:
        with torch.profiler.record_function(label):
            result = fn()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    prof.export_chrome_trace(str(Path(output) / f"{label}.trace.json"))
    rows = [{"name": e.key, "calls": e.count, "self_cpu_us": e.self_cpu_time_total,
             "cpu_us": e.cpu_time_total, "self_cuda_us": e.self_device_time_total,
             "cuda_us": e.device_time_total, "cuda_memory_bytes": e.self_device_memory_usage}
            for e in prof.key_averages()]
    save(Path(output) / f"{label}.operators.json", sorted(rows, key=lambda x: -x["self_cuda_us"]))
    return {"profile_wall_s": elapsed, "trace": f"{label}.trace.json"}


def graph_measure(fn, repeat=20):
    """Capture a warmed, fixed-input callable and measure completed replays."""
    import torch
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
    capture_s=time.perf_counter()-start
    def replay():
        graph.replay()
        return outputs
    actual,timing=measure(replay,repeat=repeat)
    timing.update(capture_s=capture_s,scope="fixed resident inputs, same compiled arithmetic")
    return actual,timing
