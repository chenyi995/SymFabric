"""Sensitivity of the architecture and PIM models to the parameters that have no source yet
(marked TBD[src] in systolicarch/config.py and pim/config.py), plus the assumed clock.

    python3 eval/run_sensitivity.py --scale large --only datalog_tc

Writes eval/results/<ts>-sens-<scale>/sensitivity.csv with one row per (workload, config, parameter, value).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from softwareworkload import _harness as H                       # noqa: E402
from algebra.trace import OpRecord                                 # noqa: E402
from systolicarch import CONFIGS, map_trace, summarize             # noqa: E402
from pim import PIM_CONFIGS, select_placement                      # noqa: E402

SWEEPS_ARCH = {
    "random_access_bytes": [32, 64, 128],
    "op_launch_cycles": [0, 32, 256],
    "freq_hz": {"edge28": [500e6, 700e6, 1e9], "dc_hbm3": [700e6, 1e9]},
    "sram_bytes": {"edge28": [1_250_000, 4 * 1_250_000, 16 * 1_250_000],
                   "dc_hbm3": [24 << 20, 96 << 20]},
}
SWEEPS_PIM = {
    "fabric_bw_Bps": [0.5 * 819.2e9, 819.2e9, 4 * 819.2e9],
    "global_reduce_s": [1e-7, 1e-6, 1e-5],
    "cmd_latency_s": [100e-9, 200e-9, 1e-6],
    "pe_freq_hz": [0.666e9, 1.3e9],
    "pe_lanes": [16, 32, 64],
    "query_slots": [8, 32, 128],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="large")
    ap.add_argument("--only", default=None)
    ap.add_argument("--configs", default="edge28,dc_hbm3")
    ap.add_argument("--pim", default="hbm3_pim")
    ap.add_argument("--from-traces", default=None, help="results directory whose traces/ to reuse")
    args = ap.parse_args()
    src = (Path(args.from_traces).resolve() / "traces") if args.from_traces else None
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "eval" / "results" / f"{ts}-sens-{args.scale}"
    out.mkdir(parents=True)
    rows = []
    for mod in H.discover():
        if args.only and mod.NAME != args.only:
            continue
        if src is not None:
            blob = json.loads((src / f"{mod.NAME}.json").read_text())
            records = [OpRecord(**{k: (tuple(v) if k in ("n_in", "in_uids", "in_bits") else v)
                                   for k, v in rec.items()}) for rec in blob["records"]]
        else:
            _, tr = H.run(mod, args.scale, 0, keep_trace=True)
            records = records
        for cname in args.configs.split(","):
            base = CONFIGS[cname]
            pim = PIM_CONFIGS[args.pim]
            for param, values in SWEEPS_ARCH.items():
                vals = values.get(cname, []) if isinstance(values, dict) else values
                for v in vals:
                    cfg = replace(base, **{param: v})
                    costs = map_trace(records, cfg)
                    s = summarize(costs, cfg)
                    _, ps = select_placement(records, costs, cfg, pim)
                    rows.append({"workload": mod.NAME, "config": cname, "side": "chip", "param": param,
                                 "value": v, "time_ms": s["time_ms"], "frac_memory_bound": s["frac_memory_bound"],
                                 "pe_util": s["pe_util"], "pim_speedup": ps["speedup_from_pim"]})
            costs = map_trace(records, base)
            for param, values in SWEEPS_PIM.items():
                for v in values:
                    pc = replace(pim, **{param: v})
                    _, ps = select_placement(records, costs, base, pc)
                    rows.append({"workload": mod.NAME, "config": cname, "side": "pim", "param": param,
                                 "value": v, "time_ms": ps["t_selected_s"] * 1e3, "frac_memory_bound": "",
                                 "pe_util": "", "pim_speedup": ps["speedup_from_pim"]})
            print(mod.NAME, cname, "done")
    with open(out / "sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("results:", out)


if __name__ == "__main__":
    main()
