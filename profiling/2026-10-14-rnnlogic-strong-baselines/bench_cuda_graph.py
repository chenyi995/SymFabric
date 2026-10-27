"""CUDA Graph replay of fixed-order int64 grounding, not full-model capture.

The neural predictor remains eager. Each timed captured forward includes copying
new GPU h/removed values into static buffers, graph replay, and ordered output
adapter calls. Graph construction/JIT and persistent memory are reported apart.
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
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "2026-10-14-candidate-tests/kg"
UPSTREAM = OLD / "upstream/RNNLogic"
sys.path[:0] = [str(OLD / "deps"), str(UPSTREAM / "src")]
import torch
from data import KnowledgeGraph, TrainDataset
from predictors import PredictorPlus
from gpu_backend import GPUBackend
from prefix_cache import PrefixGrounder


def memory(device):
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}


def statistics_ms(values):
    ordered = sorted(values)
    at = .95 * (len(ordered)-1)
    lo = int(at)
    return {"median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values),
            "p95_ms": ordered[lo]+(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])*(at-lo),
            "samples_ms": values}


class CapturedGrounding:
    """Capture only already-implemented CSR propagation and last-use sharing."""
    def __init__(self, entity_size, bodies, relation, backend, h, removed):
        self.device = h.device
        self.bodies = tuple(tuple(body) for body in bodies)
        self.relation = int(relation)
        self.outputs = []
        self.graph = None
        self._active = False
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        before = memory(self.device)
        self.h_static = h.clone()
        self.removed_static = removed.clone()
        proxy = SimpleNamespace(entity_size=entity_size, propagate=backend.propagate)
        controller = PrefixGrounder(proxy, self.bodies, policy="last_use", collect_stats=False)
        def execute():
            controller.begin(self.h_static, self.relation, self.removed_static)
            outputs = [controller.grounding(self.h_static, self.relation, body, self.removed_static)
                       for body in self.bodies]
            controller.finish()
            return outputs
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        warm_started = time.perf_counter()
        with torch.cuda.stream(self.stream), torch.no_grad():
            for _ in range(2):
                temporary = execute()
                del temporary
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        torch.cuda.synchronize(self.device)
        warm_seconds = time.perf_counter() - warm_started
        gc.collect()
        torch.cuda.empty_cache()
        before_capture = memory(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        capture_started = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream), torch.no_grad():
            self.outputs = execute()
        torch.cuda.synchronize(self.device)
        capture_seconds = time.perf_counter() - capture_started
        after_capture = memory(self.device)
        storages = {}
        for value in self.outputs:
            storage = value.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        self.metadata = {
            "capture_success": True, "captured_scope": "int64 grounding only: CSR kernels + last-use sharing, fixed rule order",
            "not_captured": "rule encoder, candidate nonzero/selection, neural aggregation, scoring and loss/backward",
            "warmup_seconds": warm_seconds, "capture_seconds": capture_seconds,
            "total_prepare_seconds": time.perf_counter()-started,
            "memory_before_prepare": before, "memory_after_warmup_before_capture": before_capture,
            "memory_after_capture": after_capture,
            "persistent_allocated_delta_bytes": after_capture["allocated_bytes"]-before["allocated_bytes"],
            "capture_reserved_delta_bytes": after_capture["reserved_bytes"]-before_capture["reserved_bytes"],
            "unique_terminal_storage_bytes": sum(storages.values()),
            "terminal_count": len(self.outputs), "unique_terminal_storages": len(storages),
            "memory_note": "all int64 terminal tensors stay live for replay; reserved delta includes graph-private pool buffers; report allocated/reserved separately, not as exact scratch SRAM demand"}

    def prepare(self, h, relation, removed):
        if self._active:
            raise RuntimeError("previous captured forward was not finished")
        if int(relation) != self.relation or h.shape != self.h_static.shape:
            raise ValueError("captured relation and batch size are fixed")
        if removed is None or removed.shape != self.removed_static.shape:
            raise ValueError("this benchmark captures the training-removal path")
        if h.device != self.device or removed.device != self.device or h.dtype != torch.int64 or removed.dtype != torch.int64:
            raise TypeError("captured inputs must be int64 on the original GPU")
        self._h, self._removed = h, removed
        self._index = 0
        self._active = True
        self.h_static.copy_(h, non_blocking=True)
        self.removed_static.copy_(removed, non_blocking=True)
        self.graph.replay()

    def grounding(self, h, head, body, removed=None):
        if not self._active or h is not self._h or removed is not self._removed or int(head) != self.relation:
            raise ValueError("captured adapter inputs do not match prepared query")
        if self._index >= len(self.bodies) or tuple(body) != self.bodies[self._index]:
            raise ValueError("captured adapter rule order mismatch")
        value = self.outputs[self._index]
        self._index += 1
        return value

    def finish(self, require_complete=True):
        complete = self._index == len(self.bodies)
        self._active = False
        self._h = self._removed = None
        if require_complete and not complete:
            raise RuntimeError("captured adapter did not consume every rule")

    def close(self):
        torch.cuda.synchronize(self.device)
        self.outputs.clear()
        if self.graph is not None:
            self.graph.reset()
        self.graph = None
        self._h = self._removed = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["FB15k-237:272", "wn18rr:12"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gradient-check", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "cuda_graph.json"))
    args = parser.parse_args()
    if args.warmup < 2 or args.repeats < 5:
        parser.error("at least warmup2/repeats5 required")
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA GPU allocation required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    device = torch.device("cuda", torch.cuda.current_device())
    output = Path(args.output)
    result = {"complete": False, "host": socket.gethostname(), "job": os.getenv("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "config": vars(args),
        "upstream_commit": subprocess.check_output(["git","-C",str(UPSTREAM),"rev-parse","HEAD"], text=True).strip(),
        "scope": "resident PredictorPlus forward on two real train query batches per relation; CUDA Graph captures only grounding, not full model or EM",
        "timed": "synchronized wall clock; captured forward includes GPU h/removed copies into fixed buffers, replay and ordered adapter, plus unchanged eager model; csr_last and captured share CSR backend and GEMM sum aggregation",
        "excluded": "CPU input construction/H2D upload, graph setup/JIT/capture and checkpoint loading; these are not train-step E2E timings",
        "cache_memory_note": "captured object remains resident during all variants in its case, so baselines share process-resident allocation; per-forward incremental peak is reported separately",
        "cases": []}
    def save():
        temporary = output.with_suffix(output.suffix+".tmp")
        temporary.write_text(json.dumps(result, indent=2)); temporary.replace(output)
    save()
    try:
        for case_spec in args.cases:
            dataset, relation_text = case_spec.split(":"); relation = int(relation_text)
            checkpoint = ROOT/"checkpoints"/f"{dataset}_{relation}_lstm_sum_officialloss_v2_fit5.pt"
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            graph = KnowledgeGraph(str(UPSTREAM/"data"/dataset))
            original_propagate, original_grounding = graph.propagate, graph.grounding
            adj_cpu = graph.relation2adjacency
            resident_adj = [[value.to(device) for value in entry] for entry in adj_cpu]
            backend = GPUBackend(graph, device=device, strategy="csr")
            config = {"type":"lstm", "num_layers":3, "hidden_dim":16, "aggregator":"sum", "entity_feature":"bias"}
            model = PredictorPlus(graph, **config)
            model.set_rules(str(UPSTREAM/"data"/dataset/"rnnlogic_rules.txt"))
            model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["state"])
            model.to(device).eval(); model.rule_features = model.rule_features.to(device)
            agg_original = model.rule_to_entity.forward
            def agg_gemm(count, emb, batch_ids):
                return torch.relu(model.rule_to_entity.layer_norm(model.rule_to_entity.add_model(count.T@emb)))
            bodies = [tuple(body) for _,(_,body) in model.relation2rules[relation]]
            random.seed(20261014+relation)
            train = TrainDataset(graph, batch_size=args.batch)
            eligible = [i for i,facts in enumerate(train.batches) if len(facts)==args.batch and facts[0][1]==relation]
            if len(eligible)<2: raise ValueError("need two complete real training batches")
            batches = []
            for index in eligible[:2]:
                h,r,t,target,removed = train[index]
                batches.append({"h":h.to(device),"r":r.to(device),"t":t.to(device),
                    "target":target.to(device),"removed":removed.to(device),
                    "queries":[list(map(int,triple)) for triple in train.batches[index]],"dataset_index":index})
            if batches[0]["queries"] == batches[1]["queries"]: raise AssertionError("query groups must differ")
            case = {"dataset":dataset,"relation":relation,"batch":args.batch,"rules":len(bodies),
                    "checkpoint":str(checkpoint),"checkpoint_sha256":checkpoint_sha,"model_config":config,
                    "query_groups":[],"capture":{"capture_success":False}}
            result["cases"].append(case); save()
            captured = None
            try:
                print(f"capture grounding {case_spec} B{args.batch}",flush=True)
                captured = CapturedGrounding(graph.entity_size,bodies,relation,backend,batches[0]["h"],batches[0]["removed"])
                case["capture"] = captured.metadata
            except Exception as error:
                case["capture"] = {"capture_success":False,"error_type":type(error).__name__,
                    "error":str(error),"traceback":traceback.format_exc()}
                save(); print(json.dumps({"case":case_spec,"capture_failed":case["capture"]}),flush=True)
                gc.collect(); torch.cuda.empty_cache()
                continue
            controller = PrefixGrounder(graph,bodies,policy="last_use",collect_stats=False)
            def forward(variant,batch,grad=False):
                h,r,removed = batch["h"],batch["r"],batch["removed"]
                if variant=="upstream":
                    graph.propagate,graph.grounding = original_propagate,original_grounding
                    graph.relation2adjacency = adj_cpu; model.rule_to_entity.forward = agg_original
                    active=None
                else:
                    graph.propagate=backend.propagate;graph.relation2adjacency=resident_adj
                    model.rule_to_entity.forward=agg_gemm
                    if variant=="csr_last":
                        graph.grounding=controller.grounding;controller.begin(h,relation,removed);active=controller
                    else:
                        graph.grounding=captured.grounding;captured.prepare(h,relation,removed);active=captured
                try:
                    with torch.set_grad_enabled(grad): return model(h,r,removed)
                finally:
                    if active is not None:active.finish()
            for group_id,batch in enumerate(batches):
                print(f"exact count checks {case_spec} query_group{group_id}",flush=True)
                captured.prepare(batch["h"],relation,batch["removed"])
                captured.finish(require_complete=False)
                graph.propagate,graph.grounding=original_propagate,original_grounding
                graph.relation2adjacency=adj_cpu
                maximum=0;terminal_nonzero=0
                for index,body in enumerate(bodies):
                    expected=original_grounding(batch["h"],relation,body,batch["removed"])
                    actual=captured.outputs[index]
                    if actual.dtype!=torch.int64 or not torch.equal(actual,expected):
                        raise AssertionError(f"captured vs upstream integer mismatch {case_spec} group{group_id} rule{index}")
                    maximum=max(maximum,int(actual.max()))
                    terminal_nonzero+=int(torch.count_nonzero(actual))
                    if int(actual.min())<0:raise AssertionError("negative path count")
                    del actual,expected
                graph.propagate=backend.propagate;controller.begin(batch["h"],relation,batch["removed"])
                for index,body in enumerate(bodies):
                    expected=controller.grounding(batch["h"],relation,body,batch["removed"])
                    if not torch.equal(expected,captured.outputs[index]):
                        raise AssertionError(f"captured vs csr_last integer mismatch {case_spec} group{group_id} rule{index}")
                    del expected
                controller.finish()
                reference,reference_mask=forward("upstream",batch)
                group={"group_id":group_id,"dataset_batch_index":batch["dataset_index"],"queries":batch["queries"],
                    "every_rule_int64_equal_upstream_and_csr":True,"max_terminal_count":maximum,
                    "terminal_nonzero":terminal_nonzero,"terminal_elements":len(bodies)*args.batch*graph.entity_size,
                    "terminal_density":terminal_nonzero/(len(bodies)*args.batch*graph.entity_size),"variants":{}}
                case["query_groups"].append(group)
                for variant in ("upstream","csr_last","captured_csr_last"):
                    actual,mask=forward(variant,batch)
                    error=float((actual-reference).abs().max())
                    if not torch.equal(mask,reference_mask) or not torch.allclose(actual,reference,rtol=2e-4,atol=2e-5):
                        raise AssertionError((case_spec,group_id,variant,"logit/mask mismatch",error))
                    group["variants"][variant]={"max_logit_abs_error":error,"mask_equal":True}
                    del actual,mask
                for _ in range(args.warmup):
                    for variant in group["variants"]:forward(variant,batch)
                timings={variant:[] for variant in group["variants"]}
                for repeat in range(args.repeats):
                    order=list(timings);random.Random(117+repeat).shuffle(order)
                    for variant in order:
                        torch.cuda.synchronize(device);started=time.perf_counter()
                        value=forward(variant,batch);torch.cuda.synchronize(device)
                        timings[variant].append((time.perf_counter()-started)*1000);del value
                for variant,values in timings.items():
                    row=group["variants"][variant];row["forward"]=statistics_ms(values)
                    gc.collect();torch.cuda.synchronize(device)
                    base=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
                    value=forward(variant,batch);torch.cuda.synchronize(device)
                    row["memory"]={"base_allocated_bytes":base,"peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
                        "incremental_peak_bytes":torch.cuda.max_memory_allocated(device)-base}
                    del value
                if args.gradient_check:
                    model.train();reference_grads=None;gradient_checks={}
                    for variant in group["variants"]:
                        model.zero_grad(set_to_none=True)
                        logits,mask=forward(variant,batch,grad=True)
                        target=batch["target"]*.2+torch.nn.functional.one_hot(batch["t"],graph.entity_size)*.8
                        logp=(torch.softmax(logits,dim=1)+1e-8).log()
                        loss=-(logp[mask]*target[mask]).sum()/torch.clamp(target[mask].sum(),min=1)
                        loss.backward()
                        grads={name:parameter.grad.detach().cpu().clone() for name,parameter in model.named_parameters() if parameter.grad is not None}
                        if reference_grads is None:reference_grads=grads;reference_loss=float(loss)
                        else:
                            if grads.keys()!=reference_grads.keys():raise AssertionError("gradient key mismatch")
                            errors={name:float((value-reference_grads[name]).abs().max()) for name,value in grads.items()}
                            if not all(torch.allclose(value,reference_grads[name],rtol=3e-3,atol=3e-5) for name,value in grads.items()):
                                raise AssertionError((variant,"gradient mismatch",errors))
                            gradient_checks[variant]={"loss_abs_error":abs(float(loss)-reference_loss),
                                "max_gradient_abs_error":max(errors.values(),default=0.),"allclose":True}
                        del loss,logits,mask,logp,target,grads
                    group["gradient_checks"]=gradient_checks;model.zero_grad(set_to_none=True);model.eval();del reference_grads
                save();print(json.dumps({"case":case_spec,"group":group_id,"forward_ms":{
                    key:row["forward"]["median_ms"] for key,row in group["variants"].items()},"all_counts_exact":True}),flush=True)
                del reference,reference_mask
            graph.propagate,graph.grounding=original_propagate,original_grounding;graph.relation2adjacency=adj_cpu
            model.rule_to_entity.forward=agg_original
            captured.close();del captured,controller,model,backend,graph,train,batches,resident_adj
            gc.collect();torch.cuda.empty_cache()
        result["complete"]=True
        result["all_captures_succeeded"]=all(case["capture"]["capture_success"] for case in result["cases"])
        save()
    except Exception as error:
        result["error"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        save();raise


if __name__=="__main__":main()
