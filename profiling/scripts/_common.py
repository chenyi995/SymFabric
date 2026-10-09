"""Shared schema and helpers for parse_nsys.py, parse_ncu.py and parse_perf.py.

Every parser appends one row to profiling/results/<workload>.csv (schema: profiling/results/README.md).
Cells that no measurement in the trace directory supports are written as TBD[src] (aimd rules section 17);
nothing is estimated here.  Rows written by different parsers for one run share trace_dir, the join key.
"""
import argparse, csv, re  # noqa: E401
from pathlib import Path

TBD = "TBD[src]"
PROF = Path(__file__).resolve().parents[1]
COLUMNS = [
    "workload", "tier", "scale", "instance", "hardware", "sw_version", "date", "trace_dir", "metric_source",
    "W_bytes", "W_method", "reuse_distance", "S_symbolic_fraction", "S_method", "R_control_residue", "R_method",
    "Q_sharing_degree", "access_pattern", "l1_hit_pct", "l2_hit_pct", "sectors_per_request", "semiring",
    "arith_intensity", "inst_executed", "dram_bytes", "dram_bw_util_pct", "sm_throughput_pct",
    "gpu_kernel_time_s", "wall_time_s", "work_efficiency", "sw_counters", "notes",
]


def argparser(desc):
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace-dir", required=True, help="profiling/traces/<workload>/<timestamp>-<label>/")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--tier", default=TBD)
    ap.add_argument("--scale", default=TBD)
    ap.add_argument("--instance", default=TBD)
    ap.add_argument("--semiring", default=TBD, help="metric 6, from code inspection (see the workload notes.md)")
    ap.add_argument("--Q", default=None, help="metric 4 when known from the software's own logs or configuration")
    ap.add_argument("--W-bytes", default=None, help="metric 1 when computed outside the run (e.g. gen_cnf.py --stats)")
    ap.add_argument("--W-method", default=None, help="how W_bytes was obtained (formula or counter)")
    ap.add_argument("--counter-regex", default=None,
                    help="regex tried on every stdout.log line; non-empty groups are taken pairwise as (name, value)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--results-dir", default=str(PROF / "results"))
    return ap


def read_kv(path):
    """key=value lines (versions.txt written by _wrap_common.sh)."""
    p = Path(path)
    lines = p.read_text(errors="replace").splitlines() if p.exists() else []
    return dict(ln.split("=", 1) for ln in lines if "=" in ln)


def stdout_metrics(trace_dir, counter_regex=None):
    """'SYMFABRIC_METRIC <name> <value>' lines printed by the driver scripts, plus --counter-regex matches."""
    m, p = {}, Path(trace_dir) / "stdout.log"
    rx = re.compile(counter_regex) if counter_regex else None
    for line in (p.read_text(errors="replace").splitlines() if p.exists() else []):
        t = line.split()
        if len(t) >= 3 and t[0] == "SYMFABRIC_METRIC":
            m[t[1]] = t[2]
        elif rx and (g := rx.search(line)):
            vals = [x for x in g.groups() if x is not None]
            m.update((k.strip(), v.strip()) for k, v in zip(vals[0::2], vals[1::2]))
    return m


def base_row(args, source):
    """Row with the identification columns filled from versions.txt / stdout.log and everything else TBD."""
    v = read_kv(Path(args.trace_dir) / "versions.txt")
    sw = stdout_metrics(args.trace_dir, args.counter_regex)
    row = {c: TBD for c in COLUMNS}
    row.update(workload=args.workload, tier=args.tier, scale=args.scale, instance=args.instance,
               trace_dir=str(args.trace_dir), metric_source=source, semiring=args.semiring,
               work_efficiency="see softwareworkload harness", notes=args.notes)
    row["hardware"] = " | ".join(x for x in (v.get("gpu", ""), v.get("cpu", "")) if x and x != "unknown") or TBD
    row["sw_version"] = "; ".join(f"{k}={v[k]}" for k in ("nsys", "ncu", "perf", "python", "torch", "sw_extra")
                                  if v.get(k)) or TBD
    row["date"] = v["date"][:10] if v.get("date") else TBD
    row["wall_time_s"] = v.get("wall_s", TBD)
    row["Q_sharing_degree"] = args.Q if args.Q is not None else sw.get("Q", TBD)
    row["W_bytes"] = args.W_bytes if args.W_bytes is not None else sw.get("W_bytes", TBD)
    row["W_method"] = args.W_method or ("SYMFABRIC_METRIC W_bytes printed by the driver script (tensor bytes)"
                                        if "W_bytes" in sw else TBD)
    row["sw_counters"] = ";".join(f"{k}={x}" for k, x in sw.items() if k not in ("Q", "W_bytes")) or TBD
    return row


def append_row(args, row):
    out = Path(args.results_dir) / f"{args.workload}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new: w.writeheader()
        w.writerow({c: row.get(c, TBD) for c in COLUMNS})
    print(f"[{row['metric_source']}] appended one row to {out}")


def f4(x):
    return f"{x:.4g}"
