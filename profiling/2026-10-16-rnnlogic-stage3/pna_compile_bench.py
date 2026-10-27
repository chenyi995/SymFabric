"""Bounded standard torch.compile control on unchanged original full PNA.

Graph breaks from original b_n.max().item() are permitted and disclosed. No
custom kernels, source edits, approximate reductions or production dispatch.
"""
import argparse, copy, gc, json, os, random, time, traceback
from pathlib import Path
from runner import Runner, torch, ROOT
from query_stream import iter_trace_batches
from test_pna_predictor import TinyGraph, SparseInterface, evaluate, compare, adam_checks
from predictors import PredictorPlus
from layers import FuncToNode
from pna_predictor import CachedPNAPredictor
from gpu_cache_builder import install_gpu_preloader, terminals_to_gpu
from run_bench import stats
import pna_predictor


def counters():
    from torch._dynamo.utils import counters as c
    return {k:dict(v) for k,v in c.items()}


def tiny_checks(compiled, original_aggregate, device):
    rows=[]; torch.manual_seed(337)
    for identity in [True,False]:
        model=PredictorPlus(TinyGraph(),type='emb',hidden_dim=8,entity_feature='bias',aggregator='pna').to(device)
        model.set_rules(([[0]] if identity else [])+[[0,0],[0,0],[0,1],[0,0,1],[0,3]])
        model.to(device)
        with torch.no_grad():
            off=int(identity); model.rule_emb[off+1].copy_(model.rule_emb[off])
        adapter=CachedPNAPredictor(model,SparseInterface,namespace='compile-tiny',rule_version='v1',candidate_block_size=0)
        heads=torch.tensor([0,1,2,3,4,256]); edges=torch.tensor([0,1,2,3,4,5])
        adapter.preload(heads,torch.zeros_like(heads),edges)
        for selected in [[5,0,4,1,4],[5,2,5],[5]]:
            h=heads[selected].to(device);r=torch.zeros_like(h);edge=edges[selected].to(device)
            ref=evaluate(model,model,h,r,edge);got=evaluate(model,adapter,h,r,edge)
            e=compare(got,ref);rows.append({'empty_body':identity,'queries':selected,'max_score_error':e[0],'max_all_gradient_error':e[1]})
    ties=[]
    for feature in [1,2]:
        layer=FuncToNode(4).to(device);layer.layer_norm=torch.nn.Identity()
        with torch.no_grad():
            linear=layer.add_model.layers[0];linear.weight.zero_();linear.bias.zero_()
            for d in range(4):linear.weight[d,(feature*4+d)*3]=1
        x=torch.ones(4,4,device=device,requires_grad=True)
        count=torch.tensor([[9,0,0],[2,4,0],[0,8,6],[3,1,5]],device=device).float()
        groups=torch.tensor([0,0,2],device=device)
        out=compiled(layer,count,x,groups);out.sum().backward()
        expected=torch.tensor([[1]*4,[1]*4,[1]*4,[0]*4],device=device).float()
        torch.testing.assert_close(out,torch.ones_like(out),rtol=0,atol=0)
        torch.testing.assert_close(x.grad,expected,rtol=0,atol=0);ties.append(feature)
    adam=adam_checks(device)
    return {'complete':True,'scores_all_gradients':rows,'first_index_tie_gradients_exact':ties,'three_adam_steps':adam}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('allocated worker required')
    torch.set_num_threads(16);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42)
    result={'complete':False,'job':os.getenv('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
      'scope':'standard torch.compile on unchanged original FuncToNode.forward only; full PNA, no candidate tiling in compiled variant',
      'compiler':{'dynamic':True,'fullgraph':False,'backend':'inductor','mode':'default','graph_breaks':'original max().item() retained; recorded in counters','production_runner_changed':False},
      'timing_note':'first compiled call includes lazy compile/recompile separately; 2 warmups and 5 randomized interleaved synchronized wall samples; same resident sparse cache and neural parameters',
      'cases':[]}
    path=Path(a.output)
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
    save();original=pna_predictor.pna_aggregate
    # Tiny query-group/normalizer variants must not exhaust the default cache
    # before the three real shapes reach compilation. The process is time capped.
    torch._dynamo.config.cache_size_limit=32
    result['compiler']['dynamo_cache_size_limit']=32
    compiled=torch.compile(FuncToNode.forward,backend='inductor',dynamic=True,fullgraph=False)
    tiny_mode=[True]
    def dispatch(layer,count,emb,batch,block=128):
        return compiled(layer,count,emb,batch) if tiny_mode[0] or block==-99 else original(layer,count,emb,batch,block)
    pna_predictor.pna_aggregate=dispatch;runner=None
    try:
        start=time.perf_counter();result['tiny']=tiny_checks(compiled,original,torch.device('cuda'))
        torch.cuda.synchronize();result['tiny_compile_and_validation_seconds']=time.perf_counter()-start
        result['counters_after_tiny']=counters();save();print('COMPILE TINY PASSED',flush=True)
        tiny_mode[0]=False
        runner=Runner('wn18rr',pna_tile=2048,native_kind='spa')
        install_gpu_preloader(runner.cached,trusted_unique_positive_terminals=True)
        groups={}
        for b in iter_trace_batches(ROOT/'query_stream_wn18rr.npz',epoch=0):
            rr=int(b['queries'][0,1])
            if rr in [12,8,7] and rr not in groups and len(b['queries'])==32:groups[rr]=b
        for rr in [12,8,7]:
            row={'relation':rr}
            try:
                b=groups[rr];cpu,gpu=runner.inputs(b['queries']);h,r,t,removed=gpu
                runner.cached.reset(namespace='compile-real',rule_version=runner.rule_version)
                terms=runner.native_terminals(cpu[0],cpu[1],cpu[3])
                runner.cached.preload_from_terminals(cpu[0],cpu[1],cpu[3],terminals_to_gpu(terms,runner.device));del terms
                row={'relation':rr,'query_sha256':b['query_sha256'],'cache':runner.cached.cache_statistics(),'variants':{}}
                result['cases'].append(row)
                def run(block):
                    runner.cached.candidate_block_size=block;return runner.forward('cached',h,r,removed)
                with torch.no_grad():
                    ref=run(0);before=counters();torch.cuda.synchronize();start=time.perf_counter();pred=run(-99);torch.cuda.synchronize()
                    row['compiled_first_call_seconds']=time.perf_counter()-start;row['counters_before']=before;row['counters_after_first_call']=counters()
                    error=float((pred[0]-ref[0]).abs().max());assert torch.equal(pred[1],ref[1]);torch.testing.assert_close(pred[0],ref[0],rtol=2e-4,atol=2e-5)
                    row['compiled_max_score_error']=error;del pred;save()
                    variants={'original_full':0,'eager_tile2048':2048,'compiled_full':-99}
                    for _ in range(2):
                        for block in variants.values():run(block)
                    timing={k:[] for k in variants}
                    for repeat in range(5):
                        order=list(variants);random.Random(311+repeat).shuffle(order)
                        for name in order:
                            torch.cuda.synchronize();start=time.perf_counter();run(variants[name]);torch.cuda.synchronize();timing[name].append((time.perf_counter()-start)*1000)
                    row['variants']={k:stats(v) for k,v in timing.items()}
                    row['incremental_cuda_memory']={}
                    for name,block in variants.items():
                        gc.collect();torch.cuda.synchronize();base=torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats();run(block);torch.cuda.synchronize()
                        row['incremental_cuda_memory'][name]={'base_allocated_bytes':base,
                            'incremental_peak_bytes':torch.cuda.max_memory_allocated()-base}
                row['counters_after_warm']=counters();save();print(json.dumps({'relation':rr,'first_compile_seconds':row['compiled_first_call_seconds'],'ms':{k:v['median_ms'] for k,v in row['variants'].items()}}),flush=True)
                del ref,cpu,gpu,h,r,t,removed;gc.collect()
            except torch.cuda.OutOfMemoryError as error:
                row['oom']=str(error); row['traceback']=traceback.format_exc()
                if not any(value is row for value in result['cases']): result['cases'].append(row)
                error.__traceback__=None
                ref=pred=terms=cpu=gpu=h=r=t=removed=None
                runner.cached.reset(namespace='compile-oom',rule_version=runner.rule_version)
                gc.collect();torch.cuda.empty_cache();save()
        result['complete']=True;result['final_counters']=counters();save()
    except Exception:
        result['error']=traceback.format_exc();result['final_counters']=counters();save();raise
    finally:
        pna_predictor.pna_aggregate=original
        if runner is not None:runner.close()
if __name__=='__main__':main()
