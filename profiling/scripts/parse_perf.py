#!/usr/bin/env python3
"""Parse perf_stat.txt, perf_report.txt (and perf_mem.txt) written by perf_wrap.sh and append one row.

Fills S (metric 2, CPU-only programs), R (metric 3), CPU cache/branch counters (metrics 1 and 5) and the
software's own counters (--counter-regex).  The symbol classification is the manual annotation metric 3 needs;
each workload's run.sh passes it in:
  --hot-regex      symbols of the semiring hot loop (acceleratable)
  --control-regex  branch-heavy control code (hard to accelerate); the hot regex takes precedence
  S = (hot + control) share of samples, R = control share; the rest is I/O, parsing, allocation, unknown.
perf report percentages are shares of sampled cycles in the process tree, not shares of wall time.
"""
import re, sys  # noqa: E401
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TBD, append_row, argparser, base_row, f4  # noqa: E402

ap = argparser(__doc__)
ap.add_argument("--hot-regex", required=True)
ap.add_argument("--control-regex", required=True)
args = ap.parse_args()
td = Path(args.trace_dir)
row = base_row(args, "parse_perf")
notes = [args.notes] if args.notes else []
ev = {}  # perf stat -x ';' fields: value;unit;event;run time;pct running;...  ("<not supported>" values are skipped)
if (td / "perf_stat.txt").exists():
    for line in (td / "perf_stat.txt").read_text(errors="replace").splitlines():
        f = line.split(";")
        if len(f) >= 3 and not line.startswith("#"):
            try:
                ev[f[2].strip()] = float(f[0])
            except ValueError:
                pass


def per(a, b, scale=1.0):
    return f4(scale * ev[a] / ev[b]) if ev.get(a) is not None and ev.get(b) else None


stat = {"IPC": per("instructions", "cycles"), "branch_miss_rate": per("branch-misses", "branches"),
        "LLC_load_miss_per_kinst": per("LLC-load-misses", "instructions", 1e3),
        "L1d_load_miss_per_kinst": per("L1-dcache-load-misses", "instructions", 1e3),
        "cache_miss_per_kinst": per("cache-misses", "instructions", 1e3)}
stat = {k: v for k, v in stat.items() if v is not None}
if stat:
    row["access_pattern"] = "; ".join(f"{k}={v}" for k, v in stat.items()) + " (CPU counters; label needs manual check)"
mem = td / "perf_mem.txt"
if mem.exists():  # "  12.34%  <samples>  <memory level>" rows of `perf mem report --sort mem`
    lv = re.findall(r"^\s*([\d.]+)%\s+\d+\s+(.+?)\s*$", mem.read_text(errors="replace"), re.M)
    row["reuse_distance"] = "; ".join(f"{name}={pct}%" for pct, name in lv[:8]) if lv else f"see {mem.name} (unparsed)"
rx = re.compile(r"^\s*([\d.]+)%\s+(\S+)\s+\[(.)\]\s+(.+?)\s*$")  # perf report --sort dso,sym: "12.34%  <dso>  [.] <symbol>"
syms = []
if (td / "perf_report.txt").exists():
    for line in (td / "perf_report.txt").read_text(errors="replace").splitlines():
        if (m := rx.match(line)): syms.append((float(m.group(1)), m.group(2), m.group(4)))
if syms:
    hot = sum(p for p, _, s in syms if re.search(args.hot_regex, s))
    ctl = sum(p for p, _, s in syms if re.search(args.control_regex, s) and not re.search(args.hot_regex, s))
    row["S_symbolic_fraction"] = f4((hot + ctl) / 100)
    row["S_method"] = (f"perf samples in hot /{args.hot_regex}/ ({f4(hot)}%) + control /{args.control_regex}/ "
                       f"({f4(ctl)}%); symbols above 0.1% cover {f4(sum(p for p, _, _ in syms))}%")
    row["R_control_residue"] = f4(ctl / 100)
    row["R_method"] = "perf samples in control symbols; classification in run.sh, re-check against perf_report.txt"
    notes.append("top symbols: " + ", ".join(f"{s[:40]}={f4(p)}%" for p, _, s in syms[:8]))
else:
    row["S_method"] = row["R_method"] = "perf_report.txt missing or no symbol lines matched"
row["notes"] = "; ".join(notes) or TBD
append_row(args, row)
