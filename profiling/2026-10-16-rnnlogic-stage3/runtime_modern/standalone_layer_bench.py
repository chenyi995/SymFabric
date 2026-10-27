"""Allocated-only official FuncToNode eager/compiler control, no graph extension."""
import argparse, copy, gc, hashlib, importlib.util, json, os, platform, random
import socket, statistics, subprocess, sys, time, traceback
from pathlib import Path
import torch
from runtime_report import write_report

STAGE = Path(__file__).resolve().parent.parent
SOURCE = STAGE.parent / '2026-10-14-candidate-tests/kg/upstream/RNNLogic/src/layers.py'
spec = importlib.util.spec_from_file_location('standalone_official_layers', SOURCE)
layers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layers)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def difference(a, b, *, rtol=2e-4, atol=2e-5):
    assert a.shape == b.shape
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {'finite': finite, 'max_abs': float((a-b).abs().max()) if a.numel() else 0.,
            'close': finite and bool(torch.allclose(a, b, rtol=rtol, atol=atol)),
            'rtol': rtol, 'atol': atol}


def construct(fixture):
    layer = layers.FuncToNode(int(fixture['hidden_dim']))
    if fixture.get('layer_norm_kind', 'layer_norm') == 'identity':
        layer.layer_norm = torch.nn.Identity()
    layer.load_state_dict(fixture['layer_state'], strict=True)
    return layer.cuda()


def evaluate(forward, layer, counts, embeddings, batch_ids, *, tie=False):
    layer.zero_grad(set_to_none=True)
    x = embeddings.detach().clone().requires_grad_(True)
    out = forward(layer, counts, x, batch_ids)
    if tie:
        loss = out.sum()
    else:
        weights = torch.linspace(.5, 1.5, out.shape[-1], device=out.device)
        loss = (out * weights).sum() / max(1, out.shape[0])
    loss.backward()
    torch.cuda.synchronize()
    packed = {'output': out.detach().cpu(), 'embedding_gradient': x.grad.detach().cpu(),
              'parameter_gradients': {k: None if p.grad is None else p.grad.detach().cpu()
                                      for k,p in layer.named_parameters()}}
    del out, x, loss
    layer.zero_grad(set_to_none=True)
    return packed


def compare_results(got, ref):
    out = {'output': difference(got['output'], ref['output']),
           'embedding_gradient': difference(got['embedding_gradient'], ref['embedding_gradient'], rtol=3e-4),
           'parameter_gradients': {}}
    assert got['parameter_gradients'].keys() == ref['parameter_gradients'].keys()
    for name, g in got['parameter_gradients'].items():
        r = ref['parameter_gradients'][name]
        out['parameter_gradients'][name] = {'close': g is None and r is None} if g is None or r is None else difference(g,r,rtol=3e-4)
    out['passed'] = out['output']['close'] and out['embedding_gradient']['close'] and all(x['close'] for x in out['parameter_gradients'].values())
    return out



def adam_three_steps(compiled, fixture, counts, embeddings, batch_ids):
    """Pure official-layer/rule-embedding training gate; optimizer stays eager."""
    eager_layer=construct(fixture); compiled_layer=construct(fixture)
    eager_x=torch.nn.Parameter(embeddings.detach().clone())
    compiled_x=torch.nn.Parameter(embeddings.detach().clone())
    pairs=[(layers.FuncToNode.forward,eager_layer,eager_x),(compiled,compiled_layer,compiled_x)]
    optimizers=[torch.optim.Adam([*layer.parameters(),x],lr=.001) for _,layer,x in pairs]
    rows=[]
    for step in range(3):
        packs=[]
        for (forward,layer,x),opt in zip(pairs,optimizers):
            opt.zero_grad(set_to_none=True)
            out=forward(layer,counts,x,batch_ids)
            weights=torch.linspace(.5,1.5,out.shape[-1],device=out.device)
            loss=(out*weights).sum()/max(1,out.shape[0]);loss.backward()
            packs.append({'output':out.detach().cpu(),'embedding_gradient':x.grad.detach().cpu(),
                          'parameter_gradients':{k:None if p.grad is None else p.grad.detach().cpu() for k,p in layer.named_parameters()}})
            opt.step()
        torch.cuda.synchronize()
        row={'step':step,'outputs_and_gradients':compare_results(packs[1],packs[0]),
             'embedding_after_step':difference(compiled_x.detach().cpu(),eager_x.detach().cpu(),rtol=3e-4),
             'parameters_after_step':{}}
        for (k,a),(other,b) in zip(eager_layer.named_parameters(),compiled_layer.named_parameters()):
            assert k==other
            row['parameters_after_step'][k]=difference(b.detach().cpu(),a.detach().cpu(),rtol=3e-4)
        row['passed']=row['outputs_and_gradients']['passed'] and row['embedding_after_step']['close'] and all(x['close'] for x in row['parameters_after_step'].values())
        rows.append(row)
    return {'scope':'three same-runtime Adam steps on official layer parameters and rule embeddings; fixed actual tiny mixed-query counts; optimizer eager',
            'passed':all(row['passed'] for row in rows),'steps':rows}


def benchmark_variants(variants, layer, counts, embeddings, batch_ids, repeats):
    samples = {name: [] for name in variants}
    with torch.no_grad():
        for _ in range(2):
            for fn in variants.values():
                fn(layer,counts,embeddings,batch_ids)
        torch.cuda.synchronize()
        for repeat in range(repeats):
            names=list(variants); random.Random(821+repeat).shuffle(names)
            for name in names:
                torch.cuda.synchronize(); start=time.perf_counter()
                variants[name](layer,counts,embeddings,batch_ids)
                torch.cuda.synchronize(); samples[name].append((time.perf_counter()-start)*1000)
    stats={name:{'samples_ms':v,'median_ms':statistics.median(v),'min_ms':min(v),'max_ms':max(v)} for name,v in samples.items()}
    for name,fn in variants.items():
        clean(); torch.cuda.synchronize(); base=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(): fn(layer,counts,embeddings,batch_ids)
        torch.cuda.synchronize()
        stats[name]['incremental_peak_bytes']=torch.cuda.max_memory_allocated()-base
    return stats


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fixtures',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--reference-dir',required=True)
    ap.add_argument('--legacy-reference-dir')
    ap.add_argument('--compile',action='store_true')
    ap.add_argument('--repeats',type=int,default=5)
    args=ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm allocation required')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    dest=Path(args.output); refdir=Path(args.reference_dir); refdir.mkdir(parents=True,exist_ok=True)
    result={'complete':False,'all_compiler_checks_passed':False if args.compile else None,
            'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'python':platform.python_version(),
            'job':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'source':str(SOURCE),
            'actual_cpu_affinity':sorted(os.sched_getaffinity(0)),'verified_cpu_list':os.environ.get('BENCH_CPU_LIST'),
            'source_sha256':digest(SOURCE),'fixture_dir':str(Path(args.fixtures).resolve()),
            'scope':'Standalone unchanged official FuncToNode + official MLP; no torch_scatter, graph loading, cache, score head or full-model E2E.',
            'compile_options':{'enabled':args.compile,'backend':'inductor','dynamic':True,'fullgraph':False,'mode':'default','original_item_graph_break_preserved':True},
            'gradient_note':'Output rtol2e-4/atol2e-5; gradient rtol3e-4/atol2e-5; exact isolated extrema-tie checks. All learned layer parameters and rule embeddings checked against same-runtime eager; counts are fixed inputs. Cross-version comparisons are diagnostic because 2.14 changes clamp-boundary subgradients.',
            'timing_note':'Forward-only resident-input inference. Compilation/gradient validation separate; 2 warmups then randomized interleaving of 5 synchronized samples. No E2E speedup claim.',
            'cases':[]}
    def save():
        tmp=dest.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(dest)
        write_report()
    save()
    try:
        result['gpu']=torch.cuda.get_device_name();result['gpu_capability']=list(torch.cuda.get_device_capability())
        result['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
        result['driver_version']=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip().splitlines()[0]
        props=torch.cuda.get_device_properties(0)
        result['gpu_total_memory_bytes']=props.total_memory
        result['gpu_multiprocessor_count']=props.multi_processor_count
        # A real CUDA execution must pass before relying on minor compatibility.
        x=torch.arange(16,device='cuda',dtype=torch.float32)
        assert float((x*x).sum()) == 1240.;torch.cuda.synchronize()
        result['cuda_eager_smoke_passed']=True
        if args.compile:
            torch._dynamo.config.cache_size_limit=64
            compiled=torch.compile(layers.FuncToNode.forward,backend='inductor',dynamic=True,fullgraph=False)
        manifest_path=Path(args.fixtures)/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        if not manifest.get('complete'):raise RuntimeError('fixture manifest incomplete')
        if manifest['source']['official_layers_sha256'] != result['source_sha256']:
            raise RuntimeError('official layers changed since fixture generation')
        result['fixture_manifest_sha256']=digest(manifest_path)
        # Frozen producer order preserves mixed groups before the known singleton.
        fixtures=[Path(args.fixtures)/Path(x['path']).name for x in manifest['fixtures']]
        expected_hashes={Path(x['path']).name:x['file_sha256'] for x in manifest['fixtures']}
        if len(fixtures) != 8:raise RuntimeError('expected all eight tiny/real fixtures')
        for path in fixtures:
            row={'fixture':path.name,'sha256':digest(path),'passed':False};result['cases'].append(row);save()
            if row['sha256'] != expected_hashes[path.name]:raise RuntimeError('fixture hash mismatch: '+path.name)
            fixture=counts=embeddings=batch_ids=layer=None
            try:
                fixture=torch.load(path,map_location='cpu',weights_only=True)
                counts=fixture['counts'].cuda();embeddings=fixture['embeddings'].cuda();batch_ids=fixture['batch_ids'].cuda()
                layer=construct(fixture)
                row.update({'name':fixture.get('name',path.stem),'counts_shape':list(counts.shape),'embedding_shape':list(embeddings.shape),'metadata':fixture.get('metadata',{})})
                tie=fixture.get('layer_norm_kind')=='identity'
                start=time.perf_counter();eager=evaluate(layers.FuncToNode.forward,layer,counts,embeddings,batch_ids,tie=tie)
                row['eager_validation_seconds']=time.perf_counter()-start
                eager['fixture_sha256']=row['sha256']
                torch.save(eager,refdir/path.name)
                if 'expected_output' in fixture:
                    row['expected_output']=difference(eager['output'],fixture['expected_output'],rtol=0,atol=0)
                    if not row['expected_output']['close']:raise AssertionError('eager expected tie output mismatch')
                if 'expected_embeddings_gradient' in fixture:
                    row['expected_embeddings_gradient']=difference(eager['embedding_gradient'],fixture['expected_embeddings_gradient'],rtol=0,atol=0)
                    if not row['expected_embeddings_gradient']['close']:raise AssertionError('eager expected tie gradient mismatch')
                if args.legacy_reference_dir:
                    legacy=Path(args.legacy_reference_dir)/path.name
                    if legacy.exists():
                        old=torch.load(legacy,weights_only=True)
                        if old['fixture_sha256'] != row['sha256']:raise RuntimeError('legacy fixture hash mismatch')
                        row['cross_version_eager_diagnostic']=compare_results(eager,old)
                variants={'eager':layers.FuncToNode.forward}
                if args.compile:
                    start=time.perf_counter();got=evaluate(compiled,layer,counts,embeddings,batch_ids,tie=tie)
                    row['first_compiled_forward_backward_seconds']=time.perf_counter()-start
                    row['compiler_correctness']=compare_results(got,eager)
                    if not row['compiler_correctness']['passed']:raise AssertionError('same-runtime compiler correctness mismatch')
                    if tie:
                        row['compiler_tie_output_exact']=difference(got['output'],eager['output'],rtol=0,atol=0)
                        row['compiler_tie_embedding_gradient_exact']=difference(got['embedding_gradient'],eager['embedding_gradient'],rtol=0,atol=0)
                        if not row['compiler_tie_output_exact']['close'] or not row['compiler_tie_embedding_gradient_exact']['close']:
                            raise AssertionError('compiled first-index tie behavior mismatch')
                    variants['compiled']=compiled
                    if path.stem=='tiny01_mixed5':
                        start=time.perf_counter()
                        row['three_adam_steps']=adam_three_steps(compiled,fixture,counts,embeddings,batch_ids)
                        row['three_adam_steps_seconds']=time.perf_counter()-start
                        if not row['three_adam_steps']['passed']:raise AssertionError('same-runtime three-Adam-step gate failed')
                    del got
                # A separate no_grad graph can compile; do not include it in warm timing.
                with torch.no_grad():
                    for name,fn in variants.items():
                        torch.cuda.synchronize();start=time.perf_counter();warm=fn(layer,counts,embeddings,batch_ids);torch.cuda.synchronize()
                        row.setdefault('first_inference_seconds',{})[name]=time.perf_counter()-start
                        check=difference(warm.detach().cpu(),eager['output'])
                        row.setdefault('inference_correctness',{})[name]=check
                        if not check['close']:raise AssertionError(name+' inference mismatch')
                del warm,eager
                row['forward_timing']=benchmark_variants(variants,layer,counts,embeddings,batch_ids,args.repeats)
                row['passed']=True
            except Exception as error:
                row['error']=traceback.format_exc();row['oom']=isinstance(error,torch.cuda.OutOfMemoryError)
                print('CASE FAILED',path.name,str(error),flush=True)
            finally:
                fixture=counts=embeddings=batch_ids=layer=eager=got=warm=None
                clean();save()
            print('CASE DONE',path.name,'passed='+str(row['passed']),flush=True)
        if args.compile:
            from torch._dynamo.utils import counters
            result['compiler_counters']={k:dict(v) for k,v in counters.items()}
            result['all_compiler_checks_passed']=all(row['passed'] for row in result['cases'])
        result['complete']=True;result['all_cases_passed']=all(row['passed'] for row in result['cases']);save()
    except Exception:
        result['error']=traceback.format_exc();save();raise
    finally:
        write_report()

if __name__=='__main__':main()
