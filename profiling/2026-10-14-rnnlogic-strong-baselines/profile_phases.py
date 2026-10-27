"""Separate real-GPU phase diagnostics for the frozen RNNLogic baselines.

This script never trains or edits upstream. Checkpoints must already exist.
Unprofiled timings, instrumented CPU/CUDA traces, and cache diagnostics are
separate runs. Inclusive phase/device times overlap: never sum their rows.
"""
import argparse
import collections
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "2026-10-14-candidate-tests/kg"
sys.path[:0] = [str(OLD / "deps"), str(OLD / "upstream/RNNLogic/src")]
import torch
from data import KnowledgeGraph
from predictors import PredictorPlus
from prefix_cache import PrefixGrounder
from gpu_backend import GPUBackend
from run_bench import inputs


@contextlib.contextmanager
def phase_markers(graph, model, counters):
    """Temporarily add profiler ranges without changing model implementation."""
    originals = []

    def install(owner, attribute, label):
        original = getattr(owner, attribute)
        originals.append((owner, attribute, original))

        def wrapped(*args, **kwargs):
            counters[label] += 1
            with torch.profiler.record_function("phase/" + label):
                return original(*args, **kwargs)

        setattr(owner, attribute, wrapped)

    install(graph, "grounding", "grounding")
    install(graph, "propagate", "propagate")
    install(model, "encode_rules", "nn_encode_rules")
    install(model.rule_to_entity, "forward", "nn_aggregation")
    install(model.score_model, "forward", "nn_scoring")
    install(model.relation_emb, "forward", "nn_relation_embedding")
    try:
        yield
    finally:
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)


def metric(event, name, fallback=None):
    value = getattr(event, name, None)
    if value is None and fallback:
        value = getattr(event, fallback, 0.0)
    return float(value or 0.0)


def event_row(event):
    return {
        "name": event.key,
        "device_type": str(event.device_type),
        "count": int(event.count),
        "self_cpu_us": metric(event, "self_cpu_time_total"),
        "inclusive_cpu_us": metric(event, "cpu_time_total"),
        "self_device_us": metric(event, "self_device_time_total", "self_cuda_time_total"),
        "inclusive_device_us": metric(event, "device_time_total", "cuda_time_total"),
    }


def union_duration(intervals):
    """Actual interval union, so overlapping GPU streams are not double counted."""
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def summarize_trace(trace_path, top):
    """Count actual Chrome GPU activity, never user-annotation envelopes."""
    trace_path = Path(trace_path)
    trace = json.loads(trace_path.read_text())
    events = trace["traceEvents"]
    activities = [event for event in events if event.get("ph") == "X"
                  and event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}]
    kernels = [event for event in activities if event["cat"] == "kernel"]
    transfers = [event for event in activities if event["cat"] != "kernel"]
    grouped = collections.defaultdict(lambda: {"count": 0, "device_duration_sum_us": 0.0})
    for event in kernels:
        row = grouped[event["name"]]
        row["count"] += 1
        row["device_duration_sum_us"] += float(event.get("dur", 0.0))
    kernel_rows = [{"name": name, **row} for name, row in grouped.items()]
    kernel_rows.sort(key=lambda row: row["device_duration_sum_us"], reverse=True)
    api = collections.defaultdict(lambda: {"count": 0, "host_duration_sum_us": 0.0})
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in {"cuda_runtime", "cuda_driver"}:
            row = api[(event["cat"], event["name"])]
            row["count"] += 1
            row["host_duration_sum_us"] += float(event.get("dur", 0.0))
    api_rows = [{"category": category, "name": name, **row}
                for (category, name), row in api.items()]
    api_rows.sort(key=lambda row: row["host_duration_sum_us"], reverse=True)
    transfer_groups = collections.defaultdict(lambda: {"count": 0, "device_duration_sum_us": 0.0})
    for event in transfers:
        row = transfer_groups[(event["cat"], event["name"])]
        row["count"] += 1
        row["device_duration_sum_us"] += float(event.get("dur", 0.0))
    def intervals(items):
        return [(float(event["ts"]), float(event["ts"]) + float(event.get("dur", 0.0))) for event in items]
    return {
        "trace_kernel_metrics_source": "Chrome trace exact category kernel; gpu_memcpy/gpu_memset separately counted; gpu_user_annotation excluded",
        "trace_metrics_file": str(trace_path),
        "trace_metrics_mtime_ns": trace_path.stat().st_mtime_ns,
        "cuda_event_count": len(activities),
        "kernel_count": len(kernels),
        "transfer_or_memset_count": len(transfers),
        "excluded_gpu_annotation_event_count": sum(event.get("cat") == "gpu_user_annotation" for event in events),
        "kernel_duration_sum_us": sum(float(event.get("dur", 0.0)) for event in kernels),
        "kernel_interval_union_us": union_duration(intervals(kernels)),
        "device_interval_union_us": union_duration(intervals(activities)),
        "kernels_by_device_time": kernel_rows[:top],
        "cuda_runtime_calls": api_rows,
        "transfer_or_memset_by_name": [{"category": category, "name": name, **row}
                                      for (category, name), row in transfer_groups.items()],
    }


def summarize_profile(profiler, top, trace_path):
    events = list(profiler.events())
    rows = [event_row(event) for event in profiler.key_averages()]
    operators = [row for row in rows if "CPU" in row["device_type"] and not row["name"].startswith("phase/")]
    # CPU self times are thread-time sums, not critical-path wall time. GPU
    # user-annotation spans are kept apart from real GPU kernel activities.
    cpu_by_phase = collections.defaultdict(float)
    for event in events:
        if "CPU" not in str(event.device_type):
            continue
        parent = event
        label = "outside_phase"
        while parent is not None:
            if parent.name.startswith("phase/"):
                label = parent.name
                break
            parent = getattr(parent, "cpu_parent", None)
        cpu_by_phase[label] += metric(event, "self_cpu_time_total")
    return {
        "notes": [
            "All values cover every profiled step; divide by profile_steps for per-forward diagnostics.",
            "Inclusive CPU/device phase rows overlap (propagate is nested in grounding); do not add them.",
            "ATen self-device rows and raw CUDA kernel rows represent overlapping views of the same execution; do not add them.",
            "CPU phase attribution sums exclusive CPU event times across threads; annotation self time includes Python, profiler overhead and otherwise unrecorded work.",
            "GPU event interval unions are traced device busy intervals, not hardware bandwidth/occupancy counters; kernel sums can exceed union time due to overlap.",
            "Actual kernels require Chrome cat=kernel; GPU annotation envelopes are excluded from kernel counts, duration sums, and device busy unions.",
            "Triton driver launches may not be attributed to CPU phase inclusive_device_us by PyTorch; zero there does not mean zero GPU propagation work. Use actual named kernels in the Chrome trace.",
            "gpu_phase_annotations_inclusive contains GPU range envelopes, not operator execution durations or busy intervals.",
        ],
        "phases_inclusive": [row for row in rows if row["name"].startswith("phase/") and "CPU" in row["device_type"]],
        "gpu_phase_annotations_inclusive": [row for row in rows if row["name"].startswith("phase/") and "CUDA" in row["device_type"]],
        "cpu_self_us_by_nearest_phase": dict(sorted(cpu_by_phase.items())),
        "cpu_ops_by_self_time": sorted(operators, key=lambda row: row["self_cpu_us"], reverse=True)[:top],
        "device_ops_by_self_time": sorted(operators, key=lambda row: row["self_device_us"], reverse=True)[:top],
        **summarize_trace(trace_path, top),
    }


def save(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["FB15k-237:272", "wn18rr:12"])
    parser.add_argument("--variants", nargs="+", default=["torch_all", "atomic_last", "csr_last"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.repeats < 1 or args.profile_steps < 1 or args.warmup < 0:
        parser.error("batch/repeats/profile-steps must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA GPU required; this script does not simulate CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    random.seed(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = args.trace_dir or args.output.parent / (args.output.stem + "_traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    config = {"type": "lstm", "hidden_dim": 16, "num_layers": 3,
              "aggregator": "sum", "entity_feature": "bias"}
    variant_info = {}
    for variant in args.variants:
        kind, policy_name = variant.split("_", 1)
        if kind not in {"torch", "atomic", "csr"}:
            parser.error("variant backend must be torch, atomic, or csr")
        if policy_name in {"all", "last"}:
            policy, budget = ("last_use" if policy_name == "last" else "all"), None
        elif policy_name.startswith("lru"):
            policy, budget = "lru", int(policy_name[3:]) * 1024 ** 2
        else:
            parser.error("variant policy must be all, last, or lru<MIB>")
        variant_info[variant] = kind, policy, budget
    result = {
        "host": socket.gethostname(), "job": os.getenv("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "affinity": sorted(os.sched_getaffinity(0)),
        "upstream_commit": subprocess.check_output(["git", "-C", str(OLD / "upstream/RNNLogic"), "rev-parse", "HEAD"], text=True).strip(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scope": "frozen five-step official-loss-v2 brief-fit PredictorPlus, sampled real queries; not paper accuracy/full-test-set/EM E2E",
        "timing": "unprofiled synchronized wall-clock runs separate from diagnostic profiler and cache statistics; graph/JIT warm setup excluded",
        "profiler_caution": "record_function and Kineto add overhead; profile wall time is diagnostic, not a performance result; interval unions measure observed execution, not memory bandwidth or occupancy",
        "cases": [],
    }
    for case_name in args.cases:
        dataset, relation_text = case_name.split(":")
        relation = int(relation_text)
        checkpoint = args.checkpoints / f"{dataset}_{relation}_lstm_sum_officialloss_v2_fit5.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"required frozen checkpoint missing; no training fallback: {checkpoint}")
        graph = KnowledgeGraph(str(OLD / "upstream/RNNLogic/data" / dataset))
        original_propagate, original_grounding = graph.propagate, graph.grounding
        backends = {kind: GPUBackend(graph, device=device, strategy=kind)
                    for kind in sorted({info[0] for info in variant_info.values()} - {"torch"})}
        graph.relation2adjacency = [[tensor.to(device) for tensor in pair] for pair in graph.relation2adjacency]
        model = PredictorPlus(graph, **config).to(device)
        model.set_rules(str(OLD / "upstream/RNNLogic/data" / dataset / "rnnlogic_rules.txt"))
        model.to(device)
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint_data["state"])
        model.eval()
        model.zero_grad(set_to_none=True)
        del checkpoint_data
        h, relations, target, removed = inputs(graph, relation, args.batch, args.split, device)
        bodies = [tuple(body) for _, (_, body) in model.relation2rules[relation]]
        original_aggregation = model.rule_to_entity.forward

        def gemm_aggregation(count, embedding, batch_ids):
            return torch.relu(model.rule_to_entity.layer_norm(model.rule_to_entity.add_model(count.T @ embedding)))

        controllers = {variant: PrefixGrounder(graph, bodies, policy=info[1], budget_bytes=info[2], collect_stats=False)
                       for variant, info in variant_info.items()}

        def select(variant):
            kind = variant_info[variant][0]
            graph.propagate = original_propagate if kind == "torch" else backends[kind].propagate
            graph.grounding = controllers[variant].grounding
            model.rule_to_entity.forward = gemm_aggregation
            return controllers[variant]

        def forward(controller):
            controller.begin(h, relation, removed)
            try:
                with torch.no_grad():
                    output = model(h, relations, removed)
            except Exception:
                controller.finish(require_complete=False)
                raise
            diagnostic = controller.finish()
            return output, diagnostic

        with torch.no_grad():
            reference, reference_mask = model(h, relations, removed)
        torch.cuda.synchronize(device)
        row = {"dataset": dataset, "relation": relation, "split": args.split,
               "batch": len(h), "model_config": config, "rules": len(bodies),
               "body_steps": sum(map(len, bodies)),
               "unique_prefixes": len({body[:i] for body in bodies for i in range(1, len(body) + 1)}),
               "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "variants": {}}
        for variant in args.variants:
            controller = select(variant)
            for _ in range(max(1, args.warmup)):
                (output, mask), _ = forward(controller)
            torch.cuda.synchronize(device)
            error = float((output - reference).abs().max())
            torch.testing.assert_close(output, reference, rtol=2e-4, atol=2e-5)
            assert torch.equal(mask, reference_mask), variant
            row["variants"][variant] = {"max_logit_abs_error": error, "mask_equal": True}
            del output, mask
        timings = {variant: [] for variant in args.variants}
        for iteration in range(args.repeats):
            order = list(args.variants)
            random.Random(997 + iteration).shuffle(order)
            for variant in order:
                controller = select(variant)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                output, _ = forward(controller)
                torch.cuda.synchronize(device)
                timings[variant].append((time.perf_counter() - started) * 1000)
                del output
        for variant in args.variants:
            print("profile", case_name, variant, flush=True)
            controller = select(variant)
            samples = timings[variant]
            variant_row = row["variants"][variant]
            variant_row["unprofiled_forward"] = {"median_ms": statistics.median(samples), "samples_ms": samples}
            controller.collect_stats = True
            output, cache_stats = forward(controller)
            torch.cuda.synchronize(device)
            variant_row["cache_diagnostic"] = cache_stats
            controller.collect_stats = False
            del output
            counters = collections.Counter()
            torch.cuda.synchronize(device)
            with phase_markers(graph, model, counters):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=False, profile_memory=False, with_stack=False) as profiler:
                    profile_started = time.perf_counter()
                    for _ in range(args.profile_steps):
                        with torch.profiler.record_function("phase/forward_with_controller"):
                            output, _ = forward(controller)
                            with torch.profiler.record_function("phase/completion_wait"):
                                torch.cuda.synchronize(device)
                        del output
                        profiler.step()
                    profile_wall_ms = (time.perf_counter() - profile_started) * 1000
            trace_path = trace_dir / f"{dataset}_{relation}_{args.split}_b{len(h)}_{variant}.json"
            profiler.export_chrome_trace(str(trace_path))
            variant_row["profile"] = summarize_profile(profiler, args.top, trace_path)
            variant_row["profile"].update({"profile_steps": args.profile_steps,
                                            "profiled_loop_wall_ms": profile_wall_ms,
                                            "profiled_wall_per_forward_ms": profile_wall_ms / args.profile_steps,
                                            "instrumented_calls": dict(counters),
                                            "chrome_trace": str(trace_path)})
            if variant_row["profile"]["kernel_count"] == 0:
                variant_row["profile"]["warning"] = "No CUDA kernels captured; inspect CUPTI/Kineto access before interpreting GPU metrics."
            save(args.output, {**result, "cases": result["cases"] + [row]})
            del profiler
        result["cases"].append(row)
        save(args.output, result)
        graph.grounding, graph.propagate = original_grounding, original_propagate
        model.rule_to_entity.forward = original_aggregation
        del controllers, model, graph, backends, reference, reference_mask, h, relations, target, removed
        gc.collect()
        torch.cuda.empty_cache()
    result["complete"] = True
    save(args.output, result)


if __name__ == "__main__":
    main()
