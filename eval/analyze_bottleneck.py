"""Classify every (workload, config) row of a results directory: where the time goes, whether the
compute side is memory-bound, and whether the placement selector moved work to the memory side.

    python3 eval/analyze_bottleneck.py eval/results/<dir>

Prints a table and writes <dir>/bottleneck.md. All numbers come from the directory's summary.csv
and per_op/*.csv (measured model outputs).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    args = ap.parse_args()
    rd = Path(args.results_dir).resolve()
    rows = list(csv.DictReader(open(rd / "summary.csv")))
    pim_cols = [k for k in rows[0].keys() if k.startswith("pim_") and k.endswith("_speedup")]
    out = ["# Bottleneck classification (generated)\n",
           f"Source: `{rd.name}`. Class: **compute** if < 20 % of cycles are memory-bound, **mixed** if 20–60 %, "
           "**memory** if > 60 %. 'top op' = the operator/mode holding the largest share of cycles.\n",
           "| workload | config | time (ms) | class | mem% | bw | pe | top op (share) | " +
           " | ".join(c.replace("pim_", "").replace("_speedup", "") + " x" for c in pim_cols) + " |",
           "|---|---|---|---|---|---|---|---|" + "---|" * len(pim_cols)]
    for r in rows:
        mem = float(r["frac_memory_bound"])
        cls = "compute" if mem < 0.2 else ("mixed" if mem <= 0.6 else "memory")
        per_op = rd / "per_op" / f"{r['workload']}_{r['config']}.csv"
        share = defaultdict(float)
        tot = 0.0
        if per_op.exists():
            for c in csv.DictReader(open(per_op)):
                cyc = float(c["cycles"])
                share[(c["op"], c["mode"])] += cyc
                tot += cyc
        top = max(share.items(), key=lambda kv: kv[1]) if share else (("-", "-"), 0)
        top_txt = f"{top[0][0]}/{top[0][1]} ({100 * top[1] / tot:.0f}%)" if tot else "-"
        out.append(f"| {r['workload']} | {r['config']} | {float(r['time_ms']):.3f} | {cls} | {mem:.2f} | "
                   f"{float(r['avg_bw_util']):.2f} | {float(r['pe_util']):.3f} | {top_txt} | " +
                   " | ".join(str(r.get(c, "")) for c in pim_cols) + " |")
    # ---- mode families: where does the modelled time go across all workloads? ------------------
    families = {"dense array": ("dense_gemm", "dense_elementwise"),
                "merge / join lanes": ("merge_join", "stream_fallback"),
                "probe / update": ("onchip_probe", "random_probe", "partitioned_probe", "onchip_update",
                                   "random_rmw", "stream_merge"),
                "priority queue": ("systolic_pq", "pq_pop_delete", "sort_select"),
                "stream (filter/scan/reduce)": ("stream", "segmented_reduce")}
    out.append("\n## Time by mode family (sum over all workloads, share of modelled cycles)\n")
    out.append("| config | " + " | ".join(families) + " |")
    out.append("|---|" + "---|" * len(families))
    cfgs = []
    for r in rows:
        if r["config"] not in cfgs:
            cfgs.append(r["config"])
    for c in cfgs:
        tot = defaultdict(float)
        for r in rows:
            if r["config"] != c:
                continue
            for m, v in json.loads(r["by_mode"]).items():
                tot[m] += v
        all_cyc = sum(tot.values()) or 1
        cells = []
        for fam, modes in families.items():
            cells.append(f"{100 * sum(tot.get(m, 0) for m in modes) / all_cyc:.1f}%")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    txt = "\n".join(out) + "\n"
    (rd / "bottleneck.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
