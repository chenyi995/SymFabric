#!/usr/bin/env python3
"""Parse ncu.csv written by ncu_wrap.sh (`ncu --csv --metrics ...`) and append one row.

Fills metrics 5, 7 and 8: L1/L2 sector hit rates, sectors per request, arithmetic intensity (warp-level
instructions per DRAM byte, not FLOP per byte), DRAM bandwidth utilisation and SM throughput.  Sums and averages
cover the profiled launches only (NCU_LAUNCH_SKIP / NCU_LAUNCH_COUNT in ncu_wrap.sh); averages are weighted by
gpu__time_duration.sum when that metric is present (stated in notes).
"""
import csv, sys  # noqa: E401
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TBD, append_row, argparser, base_row, f4  # noqa: E402

UNIT = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,   # prefixes ncu prints without --print-units base
        "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1}
M_DRAM_R, M_DRAM_W = "dram__bytes_read.sum", "dram__bytes_write.sum"
M_INST, M_DUR = "smsp__inst_executed.sum", "gpu__time_duration.sum"
M_BW, M_SM = "dram__throughput.avg.pct_of_peak_sustained_elapsed", "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_L1, M_L2 = "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct"
M_SEC, M_REQ = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"

args = argparser(__doc__).parse_args()
p = Path(args.trace_dir) / "ncu.csv"
lines = p.read_text(errors="replace").splitlines() if p.exists() else []
start = next((i for i, l in enumerate(lines) if "Metric Name" in l and "Metric Value" in l), None)  # skip ==PROF== lines
row = base_row(args, "parse_ncu")
notes = [args.notes] if args.notes else []
if start is None:
    row["notes"] = "; ".join(notes + ["ncu.csv missing or without a CSV header (see stderr.log)"])
    append_row(args, row); sys.exit(0)
launches = defaultdict(dict)  # (launch id, kernel name) -> metric -> value in base units
for r in csv.DictReader(lines[start:]):
    try:
        v = float(r["Metric Value"].replace(",", ""))
    except (ValueError, KeyError):
        continue
    launches[(r.get("ID", "?"), r.get("Kernel Name", "?"))][r["Metric Name"]] = v * UNIT.get(r.get("Metric Unit", ""), 1)
L = list(launches.values())


def tot(m):
    xs = [x[m] for x in L if m in x]
    return sum(xs) if xs else None


def wmean(m):
    xs = [(x[m], x.get(M_DUR, 1.0)) for x in L if m in x]
    return sum(v * w for v, w in xs) / sum(w for _, w in xs) if xs else None


dram = None if tot(M_DRAM_R) is None else tot(M_DRAM_R) + (tot(M_DRAM_W) or 0.0)
inst = tot(M_INST)
row["dram_bytes"] = f4(dram) if dram is not None else TBD
row["inst_executed"] = f4(inst) if inst is not None else TBD
row["arith_intensity"] = f4(inst / dram) if inst and dram else TBD
for key, m in (("dram_bw_util_pct", M_BW), ("sm_throughput_pct", M_SM), ("l1_hit_pct", M_L1), ("l2_hit_pct", M_L2)):
    v = wmean(m); row[key] = f4(v) if v is not None else TBD
sec, req = tot(M_SEC), tot(M_REQ)
if sec and req:
    row["sectors_per_request"] = f4(sec / req)
    # claude: decision start - label rule: a fully coalesced 128-B warp request is 4 sectors; above 8 is called random
    row["access_pattern"] = ("sequential-ish" if sec / req <= 8 else "random") + " (rule sectors/request<=8; check manually)"
    # claude: decision end
weighted = any(M_DUR in x for x in L)
notes.append(f"ncu launches={len(L)}; averages {'weighted by gpu__time_duration' if weighted else 'unweighted'}; "
             "arith_intensity = warp instructions per DRAM byte")
if weighted:
    (_, name), _ = max(launches.items(), key=lambda kv: kv[1].get(M_DUR, 0.0))
    notes.append(f"longest launch: {name[:40]}")
row["notes"] = "; ".join(notes)
append_row(args, row)
