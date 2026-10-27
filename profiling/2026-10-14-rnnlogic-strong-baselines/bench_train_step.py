"""Real GPU predictor training-batch benchmark with identical starting states.

Includes upstream trainer's CPU one-hot construction, host-to-device transfers,
smoothing, predictor forward, masked loss, backward, Adam update, zero_grad,
scalar logging reads, cache cleanup and final CUDA synchronization. CPU dataset
construction, checkpoint loading, graph preparation, JIT warmup and restoring the
common comparison state are outside the timing. This is not RNNLogic EM E2E.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "2026-10-14-candidate-tests/kg"
UPSTREAM = OLD / "upstream/RNNLogic"
sys.path[:0] = [str(OLD / "deps"), str(UPSTREAM / "src")]

import torch
from data import KnowledgeGraph, TrainDataset
from predictors import PredictorPlus
from gpu_backend import GPUBackend
from prefix_cache import PrefixGrounder


VARIANTS = ("upstream", "torch_all", "atomic_last", "csr_last")


def cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def summary(samples):
    ordered = sorted(samples)
    at = (len(ordered) - 1) * .95
    low = int(at)
    p95 = ordered[low] + (ordered[min(low + 1, len(ordered)-1)] - ordered[low]) * (at-low)
    return {"median_ms": statistics.median(samples), "p95_ms": p95,
            "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples}


def compare_update(model, optimizer, start_model, expected_model, expected_optimizer,
                   observation, expected_observation, parameter_atol, parameter_rtol):
    """Runs after the timed interval; reports actual parameter/Adam differences."""
    if observation["updated"] != expected_observation["updated"]:
        raise AssertionError("training update branch changed")
    if observation["mask_count"] != expected_observation["mask_count"]:
        raise AssertionError("training mask cardinality changed")
    loss_error = abs(observation["loss"] - expected_observation["loss"])
    if not math.isclose(observation["loss"], expected_observation["loss"], rel_tol=2e-5, abs_tol=2e-5):
        raise AssertionError(f"training loss mismatch: {observation} vs {expected_observation}")
    named = dict(model.named_parameters())
    parameter_errors = {}
    update_errors = {}
    for name, parameter in named.items():
        actual = parameter.detach().cpu()
        expected = expected_model[name]
        delta = actual - start_model[name]
        expected_delta = expected - start_model[name]
        error = float((actual - expected).abs().max())
        delta_error = float((delta - expected_delta).abs().max())
        parameter_errors[name] = error
        update_errors[name] = delta_error
        if not torch.allclose(actual, expected, rtol=parameter_rtol, atol=parameter_atol):
            raise AssertionError(f"post-Adam parameter mismatch: {name}, max_abs={error}")
        if not torch.allclose(delta, expected_delta, rtol=parameter_rtol, atol=parameter_atol):
            raise AssertionError(f"Adam update mismatch: {name}, max_abs={delta_error}")
    actual_optimizer = cpu_clone(optimizer.state_dict())
    moment_errors = {}
    if actual_optimizer["state"].keys() != expected_optimizer["state"].keys():
        raise AssertionError("Adam parameter-state keys changed")
    for parameter_id, state in actual_optimizer["state"].items():
        for key, value in state.items():
            expected = expected_optimizer["state"][parameter_id][key]
            if torch.is_tensor(value):
                error = float((value - expected).abs().max())
                moment_errors[key] = max(moment_errors.get(key, 0.), error)
                if key == "step" and not torch.equal(value, expected):
                    raise AssertionError("Adam step counter differs")
                if key != "step" and not torch.allclose(value, expected, rtol=3e-3, atol=3e-6):
                    raise AssertionError(f"Adam state mismatch: parameter {parameter_id} {key}, max_abs={error}")
            elif value != expected:
                raise AssertionError("Adam non-tensor state differs")
    return {"loss_abs_error": loss_error,
            "max_parameter_abs_error": max(parameter_errors.values(), default=0.),
            "max_update_abs_error": max(update_errors.values(), default=0.),
            "parameter_errors": parameter_errors,
            "adam_state_max_abs_errors": moment_errors,
            "parameter_and_update_allclose": True,
            "adam_state_allclose": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["FB15k-237:272", "wn18rr:12"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoothing", type=float, default=.2)
    parser.add_argument("--lr", type=float, default=.005)
    parser.add_argument("--parameter-atol", type=float, default=3e-6)
    parser.add_argument("--parameter-rtol", type=float, default=3e-3)
    parser.add_argument("--output", default=str(ROOT / "train_step.json"))
    args = parser.parse_args()
    if args.warmup < 2 or args.repeats < 5:
        parser.error("require at least 2 warmup runs per variant and 5 timed repeats")
    if not torch.cuda.is_available():
        raise RuntimeError("requires an allocated real CUDA GPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    device = torch.device("cuda", torch.cuda.current_device())
    output = Path(args.output)
    result = {
        "complete": False, "host": socket.gethostname(), "job": os.getenv("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "affinity": sorted(os.sched_getaffinity(0)),
        "upstream_commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "config": vars(args), "cases": [],
        "scope": "complete sampled PredictorPlus training batch on a real GPU; fixed learned rules and brief-fit lstm_sum checkpoint; not complete EM, complete epoch, or paper accuracy reproduction",
        "included": ["CPU one_hot(all_t)", "CPU-to-GPU h/r/multi-hot target/removed/one-hot target_t",
                     "GPU target smoothing", "predictor forward", "upstream mask.sum().item branch",
                     "softmax+1e-8 then log and normalized masked loss", "backward", "Adam.step",
                     "optimizer.zero_grad", "loss.item and mask.sum().item logging reads",
                     "prefix controller begin/finish", "final CUDA synchronization"],
        "excluded": ["disk/checkpoint loading", "TrainDataset construction and multi-hot CPU target assembly",
                     "DataLoader and distributed sampler", "graph/backend preparation", "cold JIT compilation",
                     "common-state restore and correctness comparisons", "epoch scheduler", "rule generation/EM"],
        "trainer_transfer_note": "Upstream trainer keeps all_t on CPU and transfers its one-hot target_t; this benchmark follows that path rather than adding an unused all_t GPU copy.",
        "common_state_note": "Load the unchanged fit5 checkpoint, initialize Adam through two untimed upstream training updates, snapshot model+optimizer; restore this identical snapshot outside EVERY variant warmup and timed repeat. Timed repeats compare the same next update rather than differing optimization trajectories.",
        "memory_note": "one model reused; prepared backend graph arrays coexist; CPU batch is pageable, no pinned-copy optimization",
        "tolerances": {"loss_atol": 2e-5, "loss_rtol": 2e-5,
                       "parameter_and_update_atol": args.parameter_atol,
                       "parameter_and_update_rtol": args.parameter_rtol,
                       "adam_moment_atol": 3e-6, "adam_moment_rtol": 3e-3}}

    def save():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(output)

    save()
    try:
        for case_spec in args.cases:
            dataset, relation_text = case_spec.split(":")
            relation = int(relation_text)
            checkpoint = ROOT / "checkpoints" / f"{dataset}_{relation}_lstm_sum_officialloss_v2_fit5.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Required existing checkpoint absent; benchmark will not create or alter it: {checkpoint}")
            checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            graph = KnowledgeGraph(str(UPSTREAM / "data" / dataset))
            original_propagate, original_grounding = graph.propagate, graph.grounding
            adj_cpu = graph.relation2adjacency
            resident_adj = [[value.to(device) for value in entry] for entry in adj_cpu]
            atomic = GPUBackend(graph, device=device, strategy="atomic")
            csr = GPUBackend(graph, device=device, strategy="csr")
            config = {"type": "lstm", "hidden_dim": 16, "num_layers": 3,
                      "aggregator": "sum", "entity_feature": "bias"}
            model = PredictorPlus(graph, **config)
            model.set_rules(str(UPSTREAM / "data" / dataset / "rnnlogic_rules.txt"))
            model.load_state_dict(ckpt["state"])
            model.to(device).train()
            model.rule_features = model.rule_features.to(device)
            agg_original = model.rule_to_entity.forward
            def agg_gemm(count, emb, batch_ids):
                return torch.relu(model.rule_to_entity.layer_norm(model.rule_to_entity.add_model(count.T @ emb)))
            bodies = [tuple(body) for _, (_, body) in model.relation2rules[relation]]
            controllers = {
                "torch_all": PrefixGrounder(graph, bodies, policy="all", collect_stats=False),
                "atomic_last": PrefixGrounder(graph, bodies, policy="last_use", collect_stats=False),
                "csr_last": PrefixGrounder(graph, bodies, policy="last_use", collect_stats=False)}
            # This is the upstream Dataset implementation, including its exact
            # multi-hot positives and original per-query removed-edge indices.
            random.seed(20261014 + relation)
            train_set = TrainDataset(graph, batch_size=args.batch)
            eligible = [i for i, facts in enumerate(train_set.batches)
                        if len(facts) == args.batch and facts[0][1] == relation]
            if not eligible:
                raise ValueError(f"No complete relation {relation} training batch of size {args.batch}")
            batch_index = eligible[0]
            cpu_batch = train_set[batch_index]
            all_h_cpu, all_r_cpu, all_t_cpu, target_cpu, removed_cpu = cpu_batch
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            optimizer.zero_grad()

            def select(variant):
                if variant == "upstream":
                    graph.propagate, graph.grounding = original_propagate, original_grounding
                    graph.relation2adjacency = adj_cpu
                    model.rule_to_entity.forward = agg_original
                    return None
                graph.relation2adjacency = resident_adj
                graph.propagate = (original_propagate if variant == "torch_all" else
                                   atomic.propagate if variant == "atomic_last" else csr.propagate)
                model.rule_to_entity.forward = agg_gemm
                graph.grounding = controllers[variant].grounding
                return controllers[variant]

            def step(controller):
                # Same within-batch sequence as trainer.py. One-hot construction
                # happens on CPU and is included; multi-hot assembly is outside.
                target_t_cpu = torch.nn.functional.one_hot(all_t_cpu, graph.entity_size)
                all_h = all_h_cpu.cuda(device=device)
                all_r = all_r_cpu.cuda(device=device)
                target = target_cpu.cuda(device=device)
                removed = removed_cpu.cuda(device=device)
                target_t = target_t_cpu.cuda(device=device)
                target = target * args.smoothing + target_t * (1 - args.smoothing)
                if controller is not None:
                    controller.begin(all_h, relation, removed)
                try:
                    logits, mask = model(all_h, all_r, removed)
                    if mask.sum().item() != 0:
                        logits = (torch.softmax(logits, dim=1) + 1e-8).log()
                        loss = -(logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        value = loss.item()
                        count = mask.sum().item()
                        observation = {"loss": value, "mask_count": count, "updated": True}
                    else:
                        raise AssertionError("unexpected no-update batch; cannot benchmark training update")
                finally:
                    if controller is not None:
                        controller.finish()
                return observation

            print(f"bootstrap identical training state {case_spec}", flush=True)
            controller = select("upstream")
            bootstrap_losses = [step(controller)["loss"] for _ in range(2)]
            torch.cuda.synchronize(device)
            common_model = cpu_clone(model.state_dict())
            common_optimizer = cpu_clone(optimizer.state_dict())
            common_rng_cpu = torch.get_rng_state().clone()
            common_rng_cuda = torch.cuda.get_rng_state(device).clone()

            def restore():
                model.load_state_dict(common_model)
                optimizer.load_state_dict(copy.deepcopy(common_optimizer))
                optimizer.zero_grad()
                torch.set_rng_state(common_rng_cpu)
                torch.cuda.set_rng_state(common_rng_cuda, device)
                model.train()
                torch.cuda.synchronize(device)

            restore()
            expected_observation = step(select("upstream"))
            torch.cuda.synchronize(device)
            expected_model = cpu_clone(model.state_dict())
            expected_optimizer = cpu_clone(optimizer.state_dict())
            case = {"dataset": dataset, "relation": relation, "batch": args.batch,
                    "rules": len(bodies), "model_config": config,
                    "optimizer_defaults": dict(optimizer.defaults),
                    "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_digest,
                    "checkpoint_brief_fit_losses": ckpt["losses"],
                    "untimed_upstream_bootstrap_losses": bootstrap_losses,
                    "reference_next_step": expected_observation,
                    "cpu_dataset_batch_index": batch_index,
                    "queries": [{"h": int(all_h_cpu[i]), "r": relation, "t": int(all_t_cpu[i]),
                                 "removed_edge": int(removed_cpu[i])} for i in range(args.batch)],
                    "target_positive_counts": target_cpu.sum(1).tolist(),
                    "variants": {variant: {"warmup_checks": [], "measured_steps": []} for variant in VARIANTS}}
            result["cases"].append(case)
            save()
            for variant in VARIANTS:
                for iteration in range(args.warmup):
                    restore()
                    observation = step(select(variant))
                    torch.cuda.synchronize(device)
                    check = compare_update(model, optimizer, common_model, expected_model, expected_optimizer,
                                           observation, expected_observation, args.parameter_atol, args.parameter_rtol)
                    case["variants"][variant]["warmup_checks"].append({"iteration": iteration,
                                                                          "observation": observation, **check})
                print(f"training warmup and update checks passed {case_spec} {variant}", flush=True)
            save()
            for iteration in range(args.repeats):
                order = list(VARIANTS)
                random.Random(20261014 + iteration).shuffle(order)
                for variant in order:
                    restore()
                    controller = select(variant)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    observation = step(controller)
                    torch.cuda.synchronize(device)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    check = compare_update(model, optimizer, common_model, expected_model, expected_optimizer,
                                           observation, expected_observation, args.parameter_atol, args.parameter_rtol)
                    case["variants"][variant]["measured_steps"].append({"iteration": iteration,
                        "elapsed_ms": elapsed_ms, "observation": observation, **check})
                save()
            for variant in VARIANTS:
                row = case["variants"][variant]
                row["training_batch"] = summary([run["elapsed_ms"] for run in row["measured_steps"]])
                row["queries_per_second"] = args.batch / (row["training_batch"]["median_ms"] / 1000)
                row["all_steps_verified"] = True
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_digest:
                raise AssertionError("checkpoint changed during benchmark")
            case["checkpoint_unchanged"] = True
            save()
            print(json.dumps({"case": case_spec, "training_batch_ms": {
                variant: case["variants"][variant]["training_batch"]["median_ms"] for variant in VARIANTS},
                "all_loss_and_update_checks_passed": True}), flush=True)
            graph.propagate, graph.grounding = original_propagate, original_grounding
            graph.relation2adjacency = adj_cpu
            model.rule_to_entity.forward = agg_original
            del controllers, model, optimizer, atomic, csr, graph, train_set, resident_adj
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
