"""Small affinity diagnostic: run inside the existing allocated Slurm worker.

No timing or speedup claim is made. Capture affinity before importing torch,
after runtime initialization, and for every live native thread after warmups.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def status_fields(path):
    fields = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key in {"Name", "Pid", "Tgid", "Threads", "Cpus_allowed_list", "Mems_allowed_list"}:
                fields[key] = value.strip()
    except (FileNotFoundError, ProcessLookupError):
        fields["exited"] = True
    return fields


def snapshot(label):
    threads = []
    for entry in sorted(Path("/proc/self/task").iterdir(), key=lambda p: int(p.name)):
        row = {"tid": int(entry.name), **status_fields(entry / "status")}
        try:
            row["sched_affinity"] = sorted(os.sched_getaffinity(int(entry.name)))
        except (ProcessLookupError, FileNotFoundError):
            row["exited"] = True
        threads.append(row)
    union = sorted({cpu for row in threads for cpu in row.get("sched_affinity", [])})
    return {"label": label, "main_affinity": sorted(os.sched_getaffinity(0)),
            "process_status": status_fields("/proc/self/status"),
            "thread_count": len(threads), "all_thread_affinity_union": union,
            "threads": threads,
            "loaded_runtime_libraries": sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                if any(name in line for name in ("libgomp", "libomp", "libiomp", "libmkl", "libopenblas")) and "/" in line})}


def cgroup_info():
    result = {"proc_self_cgroup": Path("/proc/self/cgroup").read_text()}
    files = {}
    for line in result["proc_self_cgroup"].splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0":
            base = Path("/sys/fs/cgroup") / path.lstrip("/")
            names = ["cpuset.cpus", "cpuset.cpus.effective", "cpuset.mems.effective"]
        elif "cpuset" in controllers.split(","):
            base = Path("/sys/fs/cgroup/cpuset") / path.lstrip("/")
            names = ["cpuset.cpus", "cpuset.effective_cpus", "cpuset.mems"]
        else:
            continue
        for name in names:
            item = base / name
            if item.is_file():
                files[str(item)] = item.read_text().strip()
    result["cpuset_files"] = files
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=16)
parser.add_argument("--output")
args = parser.parse_args()
keys = ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC",
        "OMP_THREAD_LIMIT", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "GOMP_CPU_AFFINITY", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "SLURM_JOB_ID", "SLURM_STEP_ID",
        "SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPU_BIND",
        "SLURM_CPU_BIND_LIST", "SLURM_CPU_BIND_TYPE", "SLURM_CPUS_ON_NODE"]
report = {"host": socket.gethostname(), "pid": os.getpid(),
          "purpose": "affinity diagnostic, not a performance benchmark",
          "requested_threads": args.threads,
          "environment": {key: os.environ.get(key) for key in keys},
          "cgroups": cgroup_info(), "snapshots": [snapshot("before_torch_import")]}
try:
    report["linux_cpu_topology"] = subprocess.check_output(
        ["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
except (FileNotFoundError, subprocess.CalledProcessError) as error:
    report["linux_cpu_topology_error"] = str(error)

import torch
report["torch"] = torch.__version__
report["snapshots"].append(snapshot("after_torch_import"))
torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
# A few MB of elementwise work initializes the PyTorch intra-op team.
x = torch.ones(1024, 1024)
assert float((x + 1).sum()) == 2 * 1024 * 1024
report["torch_requested_threads"] = torch.get_num_threads()
report["torch_interop_threads"] = torch.get_num_interop_threads()
report["snapshots"].append(snapshot("after_torch_parallel_warmup"))

from types import SimpleNamespace
from cpu_backend import CPUBackend
rng = torch.Generator().manual_seed(876)
nodes = 521
adj = torch.randint(nodes, (2, 3113), generator=rng)
graph = SimpleNamespace(entity_size=nodes, relation2adjacency=[(adj, None)])
backend = CPUBackend(graph, threads=args.threads)
counts = torch.randint(0, 5, (nodes, 4, 1), generator=rng)
actual = backend.propagate(counts, 0)
reference = torch.zeros_like(counts).index_add_(0, adj[0], counts[adj[1]])
assert torch.equal(actual, reference)
report["native_correctness"] = True
report["native_metadata"] = backend.metadata
report["snapshots"].append(snapshot("after_native_openmp_warmup"))
import ctypes
report["openmp_runtime_symbols"] = []
for library in report["snapshots"][-1]["loaded_runtime_libraries"]:
    if not any(name in library for name in ("libgomp", "libomp", "libiomp")):
        continue
    entry = {"library": library}
    try:
        handle = ctypes.CDLL(library, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        for function in ["omp_get_max_threads", "omp_get_num_procs", "omp_get_dynamic"]:
            fn = getattr(handle, function)
            fn.restype = ctypes.c_int
            entry[function] = {"value": fn(), "symbol_address": ctypes.cast(fn, ctypes.c_void_p).value}
    except (OSError, AttributeError) as error:
        entry["error"] = str(error)
    report["openmp_runtime_symbols"].append(entry)
report["interpretation"] = [
    "Main-thread affinity after runtime initialization may be one OpenMP place.",
    "Use all-thread affinity union plus pre-import affinity and cgroup limits.",
    "Slurm CPU_IDs must be mapped to Linux CPU topology before numeric comparison.",
    "Live thread count includes idle runtime threads and is not simultaneously active cores."]
serialized = json.dumps(report, indent=2) + "\n"
if args.output:
    Path(args.output).write_text(serialized)
print(serialized, end="")
