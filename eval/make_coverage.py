"""Generate docs/coverage.md from a results directory written by eval/run_arch.py.

    python3 eval/make_coverage.py eval/results/<dir> [--out docs/coverage.md]

The table is the project's generality evidence: for every workload, which operators, semirings
and execution structure it used, the measured work-efficiency ratios, and where the time goes on
each architecture configuration. Every number in the file comes from the named results directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ["combine", "reduce", "mask", "update", "select", "filter", "scan"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=str(ROOT / "docs" / "coverage.md"))
    args = ap.parse_args()
    rd = Path(args.results_dir).resolve()
    meta = json.loads((rd / "meta.json").read_text())
    wls = json.loads((rd / "workloads.json").read_text())
    rows = list(csv.DictReader(open(rd / "summary.csv")))
    by_wl_cfg = {(r["workload"], r["config"]): r for r in rows}
    configs = []
    for r in rows:
        if r["config"] not in configs:
            configs.append(r["config"])

    lines = []
    lines.append("# Operator coverage and work efficiency (generated)\n")
    lines.append(f"Source: `{rd.relative_to(ROOT)}` (scale {meta['scale']}, seed {meta['seed']}, "
                 f"{meta['date']}, git {meta['git']}). Regenerate with `eval/make_coverage.py`; do not edit by hand.\n")
    lines.append("All ratios are measured by the harness: ratio_mul = semiring multiplications of the "
                 "algebra program / hotspot events of the native algorithm; ratio_cand = every tuple "
                 "touched by any operator / hotspot events. `ok` = algebra result equals the "
                 "independent golden.\n")
    # ---- table 1: algebra coverage ---------------------------------------------------------
    lines.append("## 1. Algebra coverage\n")
    lines.append("| workload | category | structure | semirings | ops used (count) | ok | native events | ratio_mul | ratio_cand | residue |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for w in wls:
        used = ", ".join(f"{op}({w['by_op'][op]['count']})" for op in OPS if op in w["by_op"])
        residue = ", ".join(f"{k}={v}" for k, v in (w.get("extra") or {}).items()
                            if "residue" in k.lower())
        lines.append(f"| {w['name']} | {w['category']} | {w['structure']} | {', '.join(w['semirings'])} | {used} | "
                     f"{'yes' if w['ok'] else 'NO'} | {w['native_ops']:,} | {w['ratio_mul']:.2f} | {w['ratio_cand']:.2f} | {residue or '-'} |")
    # ---- table 2: operator usage matrix ----------------------------------------------------------
    lines.append("\n## 2. Operator usage matrix (semiring operations per operator, share of the workload's total)\n")
    lines.append("| workload | " + " | ".join(OPS) + " |")
    lines.append("|---|" + "---|" * len(OPS))
    for w in wls:
        tot = sum(d["mul_ops"] + d["add_ops"] + (d["candidates"] if op in ("mask", "select", "filter", "scan") else 0)
                  for op, d in w["by_op"].items() if op in OPS) or 1
        cells = []
        for op in OPS:
            d = w["by_op"].get(op)
            if not d:
                cells.append("-")
            else:
                v = d["mul_ops"] + d["add_ops"] + (d["candidates"] if op in ("mask", "select", "filter", "scan") else 0)
                cells.append(f"{100 * v / tot:.0f}%")
        lines.append(f"| {w['name']} | " + " | ".join(cells) + " |")
    # ---- table 3: architecture ---------------------------------------------------------------------
    lines.append("\n## 3. Where the time goes on each configuration\n")
    lines.append("time = modelled cycles / clock; mem% = share of cycles in memory-bound operators; "
                 "pe = useful semiring ops / (cycles x PEs); bw = average DRAM bandwidth utilisation; "
                 "PIM x = speedup of the placement selector over chip-only (1.0 = nothing moved).\n")
    pim_cols = [k for k in rows[0].keys() if k.startswith("pim_") and k.endswith("_speedup")]
    hdr = "| workload | config | time (ms) | DRAM bytes | bw | pe | mem% | dominant modes |" + \
          "".join(f" {c.replace('pim_', '').replace('_speedup', '')} x |" for c in pim_cols)
    lines.append(hdr)
    lines.append("|---|---|---|---|---|---|---|---|" + "---|" * len(pim_cols))
    for w in wls:
        for c in configs:
            r = by_wl_cfg.get((w["name"], c))
            if not r:
                continue
            modes = json.loads(r["by_mode"])
            tot = sum(modes.values()) or 1
            dom = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(modes.items(), key=lambda kv: -kv[1])[:3] if v > 0)
            pim = "".join(f" {r.get(col, '')} |" for col in pim_cols)
            lines.append(f"| {w['name']} | {c} | {float(r['time_ms']):.3f} | {float(r['dram_bytes']):.3g} | "
                         f"{float(r['avg_bw_util']):.2f} | {float(r['pe_util']):.3f} | {float(r['frac_memory_bound']):.2f} | {dom} |{pim}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
