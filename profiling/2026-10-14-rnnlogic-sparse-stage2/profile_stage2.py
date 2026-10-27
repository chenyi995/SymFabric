"""Diagnostic phase attribution, separate from uninstrumented performance runs.

Uses only external wrappers and a diagnostic TorchDispatchMode; it never changes
streaming_predictor.py or upstream. Actual device work is counted from exact
Chrome trace categories kernel/gpu_memcpy/gpu_memset. GPU annotation envelopes
are never counted as kernels. Instrumented host wall time is not a speed result.
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
import time

from bench_stage2 import CaseRunner, sample, sync, stats, ROOT, S1
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from profile_phases import summarize_profile


@contextlib.contextmanager
def instrument(runner, variant):
    originals=[]
    phase_stack=[]
    counts=collections.Counter()

    @contextlib.contextmanager
    def mark(label):
        counts[label]+=1
        phase_stack.append(label)
        try:
            with torch.profiler.record_function('phase/'+label):
                yield
        finally:
            phase_stack.pop()

    def install(owner, name, label):
        original=getattr(owner,name)
        originals.append((owner,name,original))
        def wrapped(*args,**kwargs):
            with mark(label):
                return original(*args,**kwargs)
        setattr(owner,name,wrapped)

    class BlockMatmulMarkers(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            # Only consumer top-level ops, not encoder/scorer/internal backend
            # matmuls. Context labels are supplied by method wrappers above.
            name=str(func)
            label=None
            if phase_stack and phase_stack[-1]=='forward':
                if name=='aten.mm.default':label='block_gemm'
                elif name=='aten.index_add_.default':label='feature_scatter'
                elif name=='aten.scatter_.src':label='score_scatter'
            if label:
                with mark(label):return func(*args,**(kwargs or {}))
            return func(*args,**(kwargs or {}))

    model=runner.m
    install(model,'encode_rules','encode')
    install(model.relation_emb,'forward','relation_embedding')
    install(model.rule_to_entity.add_model,'forward','sum_post_mlp')
    install(model.rule_to_entity.layer_norm,'forward','sum_post_norm')
    install(model.score_model,'forward','score_mlp')
    for backend in runner.backends.values():
        if hasattr(backend,'propagate'):install(backend,'propagate','propagate')
    if variant in runner.adapters:
        adapter=runner.adapters[variant]
        install(adapter,'_dense_block','compact')
        install(adapter,'_sparse_block','compact')
        controller=adapter.controllers[runner.relation]
        install(controller,'begin','query_prepare')
        install(controller,'grounding','grounding')
        install(controller,'finish','grounding_release')
        # CapturedController.prepare replays all grounding at begin. Its phase
        # is query_prepare, not each later tensor-returning grounding call.
    elif variant=='csr':
        install(runner.ctrl,'begin','query_prepare')
        install(runner.ctrl,'grounding','grounding')
        install(runner.ctrl,'finish','grounding_release')
    elif variant=='graph':
        capture=runner.captures[variant]
        install(capture,'prepare','query_prepare')
        install(capture,'grounding','grounding')
    elif variant=='upstream':
        install(runner,'original_g','grounding')
        install(runner,'original_p','propagate')
    try:
        with BlockMatmulMarkers():
            yield mark,counts
    finally:
        for owner,name,original in reversed(originals):setattr(owner,name,original)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['FB15k-237:272','wn18rr:12'])
    parser.add_argument('--variants',nargs='+',default=['stream_32','graphblas_100000'])
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--split',choices=['train','valid'],default='train')
    parser.add_argument('--group',type=int,default=0)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--profile-steps',type=int,default=1)
    parser.add_argument('--top',type=int,default=30)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('diagnostic requires actual CUDA hardware')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',torch.cuda.current_device())
    trace_dir=args.output.parent/(args.output.stem+'_traces');trace_dir.mkdir(exist_ok=True,parents=True)
    result={'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),
        'gpu':torch.cuda.get_device_name(device),'torch':torch.__version__,
        'config':{key:str(val) if isinstance(val,Path) else val for key,val in vars(args).items()},
        'scope':'diagnostic real GPU PredictorPlus sum forward; count compression/transfer included; no full EM, epoch, training-step speed or hardware simulation',
        'timing':'unprofiled synchronized measurements use original adapter without wrappers or TorchDispatchMode; phase traces are separate and instrumented wall time is not performance',
        'cautions':['phase inclusive rows overlap; do not add nested totals',
                    'CPU times include profiler/Python overhead and thread-time; not critical-path partitions',
                    'Trace kernel category is actual activity; GPU annotation envelopes excluded',
                    'No DRAM bandwidth, occupancy, or power counters are measured',
                    'Captured grounding replays in query_prepare; later grounding calls return retained outputs',
                    'Trace attribution must not be subtracted from a different unprofiled measurement'],
        'cases':[]}
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(args.output)
    save()
    for spec in args.cases:
        ds,rr=spec.split(':');relation=int(rr)
        gc.collect();torch.cuda.empty_cache()
        runner=CaseRunner(ds,relation,device,args.threads,args.variants)
        inp=sample(runner.g,relation,args.batch,args.split,args.group,device)
        if inp is None:runner.close();continue
        h,r,t,removed,target=inp
        row={'dataset':ds,'relation':relation,'batch':len(h),'group':args.group,'split':args.split,
             'checkpoint':str(runner.checkpoint),'checkpoint_sha256':hashlib.sha256(runner.checkpoint.read_bytes()).hexdigest(),
             'variants':{}}
        result['cases'].append(row);save()
        with torch.no_grad():reference,refmask=runner.forward('upstream',h,r,removed)
        for variant in args.variants:
            runner.prepare(variant,h,removed)
            for _ in range(max(1,args.warmup)):
                with torch.no_grad():output,mask=runner.forward(variant,h,r,removed)
            sync(device)
            assert torch.equal(mask,refmask)
            torch.testing.assert_close(output,reference,rtol=2e-4,atol=2e-5)
            row['variants'][variant]={'max_logit_abs_error':float((output-reference).abs().max()),'mask_equal':True}
            del output,mask
        times={v:[] for v in args.variants}
        for repeat in range(args.repeats):
            order=list(args.variants);random.Random(507+repeat).shuffle(order)
            for variant in order:
                sync(device);start=time.perf_counter()
                with torch.no_grad():output=runner.forward(variant,h,r,removed)
                sync(device);times[variant].append((time.perf_counter()-start)*1000);del output
        for variant in args.variants:
            entry=row['variants'][variant];entry['unprofiled_forward']=stats(times[variant])
            trace=trace_dir/f'{ds}_{relation}_{variant}_B{len(h)}_{args.split}_g{args.group}.json'
            with instrument(runner,variant) as (mark,counters):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=False,profile_memory=False,with_stack=False) as profiler:
                    sync(device);start=time.perf_counter()
                    for _ in range(args.profile_steps):
                        with mark('forward'),torch.no_grad():output,mask=runner.forward(variant,h,r,removed)
                        profiler.step()
                    sync(device);profile_wall_ms=(time.perf_counter()-start)*1000
            torch.testing.assert_close(output,reference,rtol=2e-4,atol=2e-5);assert torch.equal(mask,refmask)
            del output,mask
            profiler.export_chrome_trace(str(trace))
            entry['instrumented_wall_ms']=profile_wall_ms
            entry['phase_call_counts']=dict(counters)
            entry['trace_path']=str(trace)
            entry['profile']=summarize_profile(profiler,args.top,trace)
            save()
            print(json.dumps({'case':spec,'variant':variant,'unprofiled_ms':entry['unprofiled_forward']['median_ms'],
                 'kernel_count':entry['profile']['kernel_count'],'kernel_union_ms':entry['profile']['kernel_interval_union_us']/1000,
                 'marker_calls':entry['phase_call_counts']}),flush=True)
        runner.close();del runner,inp,h,r,t,removed,target,reference,refmask
        gc.collect()
    result['complete']=True;save()


if __name__=='__main__':main()
