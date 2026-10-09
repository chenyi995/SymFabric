#!/usr/bin/env python3
"""Audit saved GPU profiles and REASON model costs without rerunning inference.

GPU kernel durations, staged CUDA-event spans, and model cycle accounting are
separate evidence sources. In particular, staged spans are not an additive
decomposition of the differently fused native or CUDA Graph control.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sqlite3


KG_RUNS = {
    "family_b64_final": ("family_graph_final", "nsys_family64_final_local"),
    "wn18rr_b64_final": ("wn18rr_graph", "nsys_wn64"),
}


def read_json(path):
    return json.loads(path.read_text())


def kernel_evidence(path, steps):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("""
            SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.demangledName
            GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC
        """).fetchall()
    dots = [r for r in rows if "gemv2T_kernel_val" in r[0]
            and "cublasGemvTensorStridedBatched" in r[0]]
    if sum(r[1] for r in dots) != steps:
        raise ValueError(f"Expected one batched policy dot kernel per step: {path}")
    total = sum(r[2] for r in rows)
    dot_ms = sum(r[2] for r in dots)
    return {
        "evidence": "measured Nsight CUPTI kernel durations, profiled native path",
        "kernel_sum_ms": total,
        "dot_ms": dot_ms,
        "dot_ms_per_step": dot_ms / steps,
        "dot_fraction_of_kernel_time": dot_ms / total,
        "dot_kernels": [{"name": n, "calls": c, "ms": t} for n, c, t in dots],
        "top_kernels": [{"name": n, "calls": c, "ms": t,
                         "fraction_of_kernel_time": t / total}
                        for n, c, t in rows[:12]],
    }


def model_evidence(model, steps):
    parts = model.get("parts", [{"repetitions": 1, "model": model}])
    delays, counts = Counter(), Counter()
    cycles = 0
    config = parts[0]["model"]["config"]
    for part in parts:
        repetitions, m = part["repetitions"], part["model"]
        assert m["config"] == config
        cycles += repetitions * m["cycles"]
        delays.update({k: repetitions * v for k, v in m["issue_delay_cycles"].items()})
        counts.update({k: repetitions * v for k, v in m["counts"].items()})
    assert math.isclose(cycles / config["clock_mhz"], model["execution_us"])
    # For these schedules nonzero issue-delay counters advance the global clock.
    # Bank-write delays need not do so in arbitrary schedules; reject that case.
    assert delays.get("bank_write", 0) == 0
    residual = cycles - sum(delays.values())
    assert residual >= 0
    delays["unattributed_pipeline_tail"] = residual
    scale = steps / config["clock_mhz"] / 1000
    return {
        "evidence": "uncalibrated model; logical issue-delay accounting",
        "clock_mhz": config["clock_mhz"],
        "cycles_per_invocation": cycles,
        "core_ms": steps * model["execution_us"] / 1000,
        "interface_ms": steps * model["handoff_us"] / 1000,
        "interface_bytes_per_invocation": model["interface_bytes"],
        "counts_per_invocation": dict(counts),
        "delays_ms": {k: v * scale for k, v in delays.items()},
        "delay_fractions": {k: v / cycles for k, v in delays.items()},
        "node_issue_utilization": counts["operations"] / (cycles * config["nodes"]),
        "caveat": "Residual is not pure arithmetic time; arithmetic overlaps delays. "
                  "Delay fractions are not independently removable hardware costs.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping = read_json(args.root / "mapping-final.json")
    details, rows, delays = {}, [], []
    for case, entry in mapping.items():
        b = read_json(args.root / case / "benchmark.json")
        steps = b.get("depth_budget", 1)
        m = entry["variants"][str(entry["best_tile"])]["model"]
        detail = {"batch": b["batch"], "invocations": steps,
                  "query_tile": entry["best_tile"], "model": model_evidence(m, steps)}
        if case in KG_RUNS:
            graph, profile = KG_RUNS[case]
            detail["gpu_kernel_evidence"] = kernel_evidence(args.root / profile / "native.sqlite", steps)
            detail["staged_gpu_ms"] = b["stage_device_spans_ms"]
            detail["staged_gpu_caveat"] = "Separately compiled instrumented replay; not native additive spans"
            detail["staged_gpu_ms_per_step"] = {
                k: v / steps for k, v in b["stage_device_spans_ms"].items() if k != "reset"}
            structure = b["structure"]
            capacity = sum(s["allocated_capacity"] for s in structure)
            detail["candidate_work"] = {
                "allocated_slots": capacity,
                "valid_active_slots": sum(s["valid_candidates"] for s in structure),
                "active_lane_fraction": sum(s["capacity"] for s in structure) / capacity,
            }
            best = read_json(args.root / graph / "benchmark.json")["cuda_graph"]["median_ms"]
            native = b["native_compiled"]["median_ms"]
            assert b["hybrid_reason"]["correctness"]["passed"]
            assert b["staged_matches_native"]
        else:
            best = b["gpu_compact_cuda_graph"]["median_ms"]
            native = b["gpu_compiled"]["median_ms"]
            detail["retained_gpu_segments_ms"] = {
                s: b[f"retained_gpu_{s}_graph"]["median_ms"] for s in ("before", "after")}
            assert b["correctness"]["passed"]
        core, interface = detail["model"]["core_ms"], detail["model"]["interface_ms"]
        total = entry["retained_gpu_ms"] + core + interface
        assert math.isclose(total, entry["best_hybrid_ms"], rel_tol=1e-12)
        assert math.isclose(sum(detail["model"]["delays_ms"].values()), core, rel_tol=1e-12)
        row = {"case": case, "native_gpu_ms": native, "best_gpu_ms": best,
               "retained_gpu_ms": entry["retained_gpu_ms"], "model_core_ms": core,
               "model_interface_ms": interface, "hybrid_estimate_ms": total}
        detail["totals"] = row
        details[case] = detail
        rows.append(row)
        delays.extend({"case": case, "category": k, "model_ms": v,
                       "fraction_of_model_core": detail["model"]["delay_fractions"][k]}
                      for k, v in detail["model"]["delays_ms"].items())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "breakdown.json").write_text(json.dumps(details, indent=2) + "\n")
    for filename, data in (("totals.csv", rows), ("model_delays.csv", delays)):
        with (args.output / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
