"""Plot a results directory written by eval/run_arch.py (bars of modelled time per configuration,
memory-bound share, PE utilisation, PIM speedup). Output: <results_dir>/plots/*.png

    python3 eval/plot_results.py eval/results/<dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    args = ap.parse_args()
    rd = Path(args.results_dir).resolve()
    rows = list(csv.DictReader(open(rd / "summary.csv")))
    out = rd / "plots"
    out.mkdir(exist_ok=True)
    wls = sorted({r["workload"] for r in rows})
    cfgs = []
    for r in rows:
        if r["config"] not in cfgs:
            cfgs.append(r["config"])
    x = np.arange(len(wls))
    w = 0.8 / len(cfgs)

    def grid(key, ylabel, fname, log=False):
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(wls)), 4))
        for i, c in enumerate(cfgs):
            vals = []
            for wl in wls:
                r = next((r for r in rows if r["workload"] == wl and r["config"] == c), None)
                vals.append(float(r[key]) if r and r.get(key) not in (None, "") else np.nan)
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(wls, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title(f"{rd.name}: {ylabel}")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)

    grid("time_ms", "modelled time (ms)", "time_ms.png", log=True)
    grid("frac_memory_bound", "share of cycles memory-bound", "memory_bound.png")
    grid("pe_util", "PE utilisation", "pe_util.png", log=True)
    grid("avg_bw_util", "DRAM bandwidth utilisation", "bw_util.png")
    pim_cols = [k for k in rows[0].keys() if k.startswith("pim_") and k.endswith("_speedup")]
    for col in pim_cols:
        grid(col, f"selector speedup ({col.replace('pim_', '').replace('_speedup', '')})",
             f"{col}.png", log=True)
    # mode breakdown per (workload, config): stacked bars
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(wls) * len(cfgs) / 2), 4.5))
    labels, bottoms = [], []
    modes_all = sorted({m for r in rows for m in json.loads(r["by_mode"])})
    data = {m: [] for m in modes_all}
    for wl in wls:
        for c in cfgs:
            r = next((r for r in rows if r["workload"] == wl and r["config"] == c), None)
            if not r:
                continue
            bm = json.loads(r["by_mode"])
            tot = sum(bm.values()) or 1
            labels.append(f"{wl}\n{c}")
            for m in modes_all:
                data[m].append(bm.get(m, 0) / tot)
    xx = np.arange(len(labels))
    base = np.zeros(len(labels))
    for m in modes_all:
        vals = np.array(data[m])
        if vals.sum() == 0:
            continue
        ax.bar(xx, vals, 0.8, bottom=base, label=m)
        base += vals
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("share of cycles by execution mode")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "mode_breakdown.png", dpi=130)
    plt.close(fig)
    print("plots:", out)


if __name__ == "__main__":
    main()
