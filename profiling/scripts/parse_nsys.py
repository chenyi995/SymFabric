#!/usr/bin/env python3
"""Parse the CSVs exported by nsys_wrap.sh and append one row to profiling/results/<workload>.csv.

Fills S_symbolic_fraction (metric 2), gpu_kernel_time_s, wall_time_s and kernel-time shares (notes).
S = time of the GPU kernels matching --symbolic-regex (or of the NVTX range --nvtx-range) divided by the
wrapper-measured wall time of the profiled run; that wall time includes nsys overhead and is named in S_method.
"""
import csv, re, sys  # noqa: E401
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TBD, append_row, argparser, base_row, f4, read_kv  # noqa: E402

ap = argparser(__doc__)
ap.add_argument("--symbolic-regex", default=".*", help="kernel names counted as symbolic/probabilistic (default all)")
ap.add_argument("--nvtx-range", default=None, help="NVTX push/pop range whose total time is used as symbolic time")
ap.add_argument("--share", action="append", default=[], help="name=regex: extra kernel-time share written to notes")
args = ap.parse_args()
td = Path(args.trace_dir)


def load(report):
    p = td / f"stats_{report}.csv"
    return list(csv.DictReader(p.open())) if p.exists() else []


def col(r, prefix):  # nsys headers carry units, e.g. "Total Time (ns)"
    return next(v for k, v in r.items() if k and k.startswith(prefix))


row = base_row(args, "parse_nsys")
notes = [args.notes] if args.notes else []
kern = load("cuda_gpu_kern_sum")
sym = tot = 0.0
if kern:
    times = [(float(col(r, "Total Time")) / 1e9, r["Name"]) for r in kern]
    tot = sum(t for t, _ in times)
    row["gpu_kernel_time_s"] = f4(tot)
    sym = sum(t for t, n in times if re.search(args.symbolic_regex, n))
    notes.append(f"sym_kernels/gpu_kernels={f4(sym / tot)}")
    for s in args.share:
        name, rx = s.split("=", 1)
        notes.append(f"{name}/gpu_kernels={f4(sum(t for t, n in times if re.search(rx, n)) / tot)}")
    notes.append("top kernels: " + ", ".join(f"{n[:40]}={f4(t / tot)}" for t, n in sorted(times, reverse=True)[:5]))
mem = load("cuda_gpu_mem_time_sum")
if mem:
    notes.append("memops_s=" + f4(sum(float(col(r, "Total Time")) for r in mem) / 1e9))
wall = read_kv(td / "versions.txt").get("wall_s")
nv = [r for r in load("nvtx_pushpop_sum") if args.nvtx_range and r.get("Range") == args.nvtx_range]
if wall and nv:
    t = float(col(nv[0], "Total Time")) / 1e9
    row["S_symbolic_fraction"] = f4(t / float(wall))
    row["S_method"] = f"NVTX range '{args.nvtx_range}' total {f4(t)}s / wrapper wall {wall}s (wall includes nsys overhead)"
elif wall and kern:
    row["S_symbolic_fraction"] = f4(sym / float(wall))
    row["S_method"] = f"GPU kernels matching /{args.symbolic_regex}/ = {f4(sym)}s / wrapper wall {wall}s (includes nsys overhead)"
else:
    row["S_method"] = "stats_cuda_gpu_kern_sum.csv or wall_s missing"
row["notes"] = "; ".join(notes) or TBD
append_row(args, row)
