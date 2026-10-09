"""Run every workload through the algebra, map the trace onto each architecture configuration,
and write a timestamped results directory (aimd rules section 21: new directory per run).

    python3 eval/run_arch.py --scale small [--configs edge28,dc_hbm3,tpu_like] [--only name]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from softwareworkload import _harness as H          # noqa: E402
from algebra.trace import OpRecord                  # noqa: E402
from systolicarch import CONFIGS, map_trace, summarize, roofline_rows   # noqa: E402
from pim import PIM_CONFIGS, select_placement                            # noqa: E402


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "uncommitted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--only", default=None)
    ap.add_argument("--name", default="arch")
    ap.add_argument("--pim", default=None, help="comma-separated PIM configs to evaluate the selector with")
    ap.add_argument("--from-traces", default=None,
                    help="results directory whose traces/ to re-map instead of re-running the workloads")
    args = ap.parse_args()
    pims = [PIM_CONFIGS[c] for c in args.pim.split(",")] if args.pim else []
    cfgs = [CONFIGS[c] for c in args.configs.split(",")]
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "eval" / "results" / f"{ts}-{args.name}-{args.scale}"
    (out / "per_op").mkdir(parents=True)
    (out / "traces").mkdir()
    src_traces = (Path(args.from_traces).resolve() / "traces") if args.from_traces else None
    meta = {"date": ts, "scale": args.scale, "seed": args.seed, "git": git_rev(),
            "python": platform.python_version(), "host": platform.node(),
            "configs": {c.name: {k: (list(v) if isinstance(v, frozenset) else v)
                                 for k, v in c.__dict__.items()} for c in cfgs}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    rows = []
    wl_rows = []
    for mod in H.discover():
        if args.only and mod.NAME != args.only:
            continue
        if src_traces is not None:
            blob = json.loads((src_traces / f"{mod.NAME}.json").read_text())
            r_dict, records = blob["result"], [OpRecord(**{k: (tuple(v) if k in ("n_in", "in_uids", "in_bits") else v)
                                                           for k, v in rec.items()}) for rec in blob["records"]]

            class _T:                                   # minimal stand-in for a Trace
                pass
            tr = _T()
            tr.records = records
            r = H.Result(**{k: v for k, v in r_dict.items()})
        else:
            r, tr = H.run(mod, args.scale, args.seed, keep_trace=True)
            (out / "traces" / f"{mod.NAME}.json").write_text(json.dumps(
                {"result": r.as_dict(), "records": [rec.as_dict() for rec in tr.records]}, default=str))
        wl_rows.append(r.as_dict())
        for cfg in cfgs:
            costs = map_trace(tr.records, cfg)
            s = summarize(costs, cfg)
            row = {"workload": mod.NAME, "category": mod.CATEGORY, "scale": args.scale,
                   "ok": r.ok, "native_ops": r.native_ops, "ratio_mul": round(r.ratio_mul, 4),
                   "ratio_cand": round(r.ratio_cand, 4), **{k: v for k, v in s.items()
                                                            if k not in ("by_mode", "by_op", "top_ops")},
                   "by_mode": json.dumps({k: round(v) for k, v in s["by_mode"].items()}),
                   "top_ops": json.dumps(s["top_ops"])}
            for pim in pims:
                placements, ps = select_placement(tr.records, costs, cfg, pim)
                row[f"pim_{pim.name}_speedup"] = round(ps["speedup_from_pim"], 3)
                row[f"pim_{pim.name}_groups_on_pim"] = ps["groups_on_pim"]
                row[f"pim_{pim.name}_time_share"] = round(ps["time_share_on_pim"], 3)
                with open(out / "per_op" / f"{mod.NAME}_{cfg.name}_{pim.name}_placement.csv", "w",
                          newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["group", "shape", "chip_s", "pim_s", "choice", "rows", "pe", "shuffle", "ext", "global"])
                    for pl in placements:
                        b = pl.pim_breakdown
                        w.writerow([pl.group, pl.shape, pl.chip_s, pl.pim_s, pl.choice,
                                    b.get("rows"), b.get("pe"), b.get("shuffle"), b.get("ext"), b.get("global")])
            rows.append(row)
            with open(out / "per_op" / f"{mod.NAME}_{cfg.name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(costs[0].as_dict().keys()) if costs else ["idx"])
                w.writeheader()
                for c in costs:
                    w.writerow(c.as_dict())
            with open(out / "per_op" / f"{mod.NAME}_{cfg.name}_roofline.json", "w") as f:
                json.dump(roofline_rows(costs, cfg), f)
            pim_txt = " ".join(f"{p.name}:x{row[f'pim_{p.name}_speedup']}" for p in pims)
            print(f"{mod.NAME:22s} {cfg.name:9s} cyc={s['cycles']:>12.3g} t={s['time_ms']:>9.3f}ms "
                  f"B={s['dram_bytes']:>9.3g} bw={s['avg_bw_util']:.2f} pe={s['pe_util']:.3f} "
                  f"mem%={s['frac_memory_bound']:.2f} ok={r.ok} rmul={r.ratio_mul:.2f} {pim_txt}")
    with open(out / "summary.csv", "w", newline="") as f:
        keys = []
        for rw in rows:
            for k in rw:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out / "workloads.json", "w") as f:
        json.dump(wl_rows, f, indent=1, default=str)
    print("results:", out)


if __name__ == "__main__":
    main()
