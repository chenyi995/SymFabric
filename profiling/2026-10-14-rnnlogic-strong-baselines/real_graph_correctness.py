"""Exact real-graph integer propagation checks, outside performance timing.

Uses the unchanged upstream KnowledgeGraph/PredictorPlus rule parser. Every
intermediate from independent native CPU and CUDA paths is compared to upstream
CPU int64 propagation, then all four prefix-cache policies are checked against
saved upstream terminal counts. This does not train or evaluate model accuracy.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "2026-10-14-candidate-tests/kg"
UPSTREAM = OLD / "upstream/RNNLogic"
sys.path[:0] = [str(OLD / "deps"), str(UPSTREAM / "src")]

import torch
from data import KnowledgeGraph
from predictors import PredictorPlus
from cpu_backend import CPUBackend
from gpu_backend import GPUBackend
from prefix_cache import PrefixGrounder


def exact_equal(actual, expected_cpu, context):
    if actual.dtype != torch.int64 or expected_cpu.dtype != torch.int64:
        raise AssertionError(f"{context}: counts must remain int64")
    actual_cpu = actual.detach().cpu()
    if actual_cpu.shape != expected_cpu.shape:
        raise AssertionError(f"{context}: shape {actual_cpu.shape} != {expected_cpu.shape}")
    if not torch.equal(actual_cpu, expected_cpu):
        index = tuple(int(v) for v in (actual_cpu != expected_cpu).nonzero()[0])
        raise AssertionError(f"{context}: first mismatch at {index}: "
                             f"actual={int(actual_cpu[index])}, expected={int(expected_cpu[index])}")


def get_queries(graph, relation, batch, split):
    facts = [(h, t) for h, r, t in getattr(graph, split + "_facts") if r == relation]
    random.Random(20261014 + relation).shuffle(facts)
    facts = facts[:batch]
    if len(facts) != batch:
        raise ValueError(f"{split} relation {relation}: requested {batch}, only {len(facts)} available")
    h = torch.tensor([a for a, _ in facts], dtype=torch.int64)
    t = torch.tensor([b for _, b in facts], dtype=torch.int64)
    removed = None
    if split == "train":
        removed = torch.tensor([graph.relation2ht2index[relation][graph.encode_ht(a, b)]
                                for a, b in facts], dtype=torch.int64)
    return h, t, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["FB15k-237:272", "wn18rr:12"])
    parser.add_argument("--batch", type=int, default=7)
    parser.add_argument("--splits", nargs="+", choices=["train", "valid"], default=["train", "valid"])
    parser.add_argument("--native-threads", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--output", default=str(ROOT / "real_graph_correctness.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires an allocated real CUDA GPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261014)
    device = torch.device("cuda", torch.cuda.current_device())
    output = Path(args.output)
    result = {
        "complete": False, "host": socket.gethostname(), "job": os.getenv("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "upstream_commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "scope": "exact int64 intermediate and terminal counts on sampled real train/valid queries; no timings, no model quality claim",
        "reference": "unchanged upstream KnowledgeGraph.propagate on CPU int64",
        "mask_semantics": "original relation edge id per query lane, passed only when body relation equals query relation; no deletion for valid",
        "overflow_check": "each step verifies nonnegative counts and max(previous state)*max(destination indegree) <= int64 max; sufficient conservative bound, no wraparound accepted",
        "config": vars(args), "cases": []}

    def save():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(output)

    save()
    try:
        for case_spec in args.cases:
            dataset, relation_text = case_spec.split(":")
            relation = int(relation_text)
            graph = KnowledgeGraph(str(UPSTREAM / "data" / dataset))
            original_propagate = graph.propagate
            # Invoke the upstream rule parser, but do not execute or fit an NN.
            model = PredictorPlus(graph, type="lstm", num_layers=3, hidden_dim=16)
            rules_path = UPSTREAM / "data" / dataset / "rnnlogic_rules.txt"
            model.set_rules(str(rules_path))
            parsed = model.relation2rules[relation]
            rule_ids = [int(index) for index, _ in parsed]
            bodies = [tuple(int(r) for r in rule[1]) for _, rule in parsed]
            del model, parsed
            if not bodies:
                raise ValueError(f"No upstream rules for {case_spec}")
            native = {f"native_cpu_t{n}": CPUBackend(graph, threads=n) for n in args.native_threads}
            atomic = GPUBackend(graph, device=device, strategy="atomic", validate_values=True)
            csr = GPUBackend(graph, device=device, strategy="csr", validate_values=True)
            propagators = {"upstream_cuda": (original_propagate, device),
                           "atomic_cuda": (atomic.propagate, device),
                           "csr_cuda": (csr.propagate, device)}
            propagators.update({name: (backend.propagate, torch.device("cpu")) for name, backend in native.items()})
            cache_propagators = {"upstream_cpu": (original_propagate, torch.device("cpu")), **propagators}
            maximum_indegree = [int(degree.max()) if degree.numel() else 0
                                for degree in graph.relation2outdegree]
            global_bound=1
            for body in bodies:
                bound=1
                for rr in body:
                    bound*=maximum_indegree[rr]
                    global_bound=max(global_bound,bound)
            for split in args.splits:
                h_cpu, t_cpu, removed_cpu = get_queries(graph, relation, args.batch, split)
                h_cuda = h_cpu.to(device)
                removed_cuda = None if removed_cpu is None else removed_cpu.to(device)
                case = {
                    "dataset": dataset, "relation": relation, "split": split,
                    "batch": args.batch, "entities": graph.entity_size, "rules": len(bodies),
                    "body_steps": sum(map(len, bodies)),
                    "queries": [{"h": int(h_cpu[i]), "r": relation, "t": int(t_cpu[i]),
                                 "removed_edge": None if removed_cpu is None else int(removed_cpu[i])}
                                for i in range(args.batch)],
                    "rules_sha256": hashlib.sha256(rules_path.read_bytes()).hexdigest(),
                    "all_intermediates_exact": False, "all_cache_outputs_exact": False,
                    "max_count": 0, "max_conservative_step_bound": 0,
                    "all_query_nonnegative_count_upper_bound": global_bound,
                    "all_query_int64_overflow_excluded": global_bound <= torch.iinfo(torch.int64).max,
                    "independent_backend_step_checks": {name: 0 for name in propagators},
                    "rules_checked": [], "cache_checks": []}
                result["cases"].append(case)
                save()
                terminals = []
                print(f"begin exact steps {dataset} r{relation} {split} B{args.batch}", flush=True)
                for position, (rule_id, body) in enumerate(zip(rule_ids, bodies)):
                    expected = torch.nn.functional.one_hot(h_cpu, graph.entity_size).T.unsqueeze(-1)
                    states = {name: expected.to(dev).clone() for name, (_, dev) in propagators.items()}
                    record = {"rule_position": position, "upstream_rule_id": rule_id,
                              "body": list(body), "steps": [], "max_count": 1}
                    previous_max = 1
                    for step, body_relation in enumerate(body):
                        bound = previous_max * maximum_indegree[body_relation]
                        if bound > torch.iinfo(torch.int64).max:
                            raise OverflowError(f"{dataset}/{relation}/{split}/rule{position}/step{step}: "
                                                f"cannot prove no int64 overflow from conservative bound {bound}")
                        mask_cpu = removed_cpu if body_relation == relation else None
                        mask_cuda = removed_cuda if body_relation == relation else None
                        expected = original_propagate(expected, body_relation, mask_cpu)
                        current_max = int(expected.max())
                        current_min = int(expected.min())
                        if current_min < 0:
                            raise AssertionError("nonnegative path count became negative")
                        for name, (propagate, dev) in propagators.items():
                            mask = mask_cuda if dev.type == "cuda" else mask_cpu
                            states[name] = propagate(states[name], body_relation, mask)
                            exact_equal(states[name], expected,
                                        f"{dataset}/{relation}/{split}/rule{position}/step{step}/{name}")
                            case["independent_backend_step_checks"][name] += 1
                        record["steps"].append({"step": step, "relation": body_relation,
                                                "mask_applied": mask_cpu is not None,
                                                "max_count": current_max,
                                                "conservative_upper_bound": bound})
                        previous_max = current_max
                        record["max_count"] = max(record["max_count"], current_max)
                        case["max_conservative_step_bound"] = max(case["max_conservative_step_bound"], bound)
                    # Also covers empty-rule identity results.
                    for name, value in states.items():
                        exact_equal(value, expected, f"{dataset}/{relation}/{split}/rule{position}/terminal/{name}")
                    terminals.append(expected.squeeze(-1).T)
                    record["terminal_max_count"] = int(expected.max())
                    record["terminal_nonzero"] = int(torch.count_nonzero(expected))
                    record["all_backends_exact"] = True
                    case["max_count"] = max(case["max_count"], record["max_count"])
                    case["rules_checked"].append(record)
                    del states, expected
                    if (position + 1) % 100 == 0:
                        print(f"exact steps {dataset} r{relation} {split}: {position+1}/{len(bodies)} rules", flush=True)
                case["all_intermediates_exact"] = True
                save()
                for name, (propagate, dev) in cache_propagators.items():
                    h = h_cuda if dev.type == "cuda" else h_cpu
                    removed = removed_cuda if dev.type == "cuda" else removed_cpu
                    proxy = SimpleNamespace(entity_size=graph.entity_size, propagate=propagate)
                    for policy in ("all", "last_use", "lru", "last_use_lru"):
                        controller = PrefixGrounder(proxy, bodies, policy=policy,
                                                    budget_bytes=32 * 1024**2 if policy in {"lru", "last_use_lru"} else None,
                                                    collect_stats=True)
                        controller.begin(h, relation, removed)
                        completed = False
                        try:
                            for position, body in enumerate(bodies):
                                actual = controller.grounding(h, relation, body, removed)
                                exact_equal(actual, terminals[position],
                                            f"{dataset}/{relation}/{split}/{name}/{policy}/rule{position}")
                                del actual
                            completed = True
                        finally:
                            cache_stats = controller.finish(require_complete=completed)
                        case["cache_checks"].append({"backend": name,
                            "policy": {"lru": "lru32", "last_use_lru": "lastlru32"}.get(policy, policy),
                            "rules_checked": len(bodies), "exact": True, "cache_stats": cache_stats})
                        print(f"cache exact {dataset} r{relation} {split}: {name}/{policy}", flush=True)
                        save()
                        del controller
                case["all_cache_outputs_exact"] = True
                save()
                print(json.dumps({"case": case_spec, "split": split, "rules": len(bodies),
                                  "step_checks_per_backend": case["body_steps"], "max_count": case["max_count"],
                                  "cache_variants": len(case["cache_checks"]), "exact": True}), flush=True)
                del terminals
                gc.collect()
                torch.cuda.empty_cache()
            del graph, native, atomic, csr, propagators, cache_propagators
            gc.collect()
            torch.cuda.empty_cache()
        result["complete"] = True
        save()
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        save()
        raise


if __name__ == "__main__":
    main()
