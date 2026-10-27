"""Real hardware: explicit count CSR cache versus uncached paths.
Cold build/upload are separate; warm batches mix two independently preloaded groups.
"""
import argparse,copy,gc,hashlib,json,os,random,socket,time,traceback
from pathlib import Path
import torch
from bench_stage2 import CaseRunner,sample,loss_from
from run_bench import sync,stats
from bench_memoized import mixed_groups,assert_outputs
from bench_train_step import cpu_clone,compare_update
from cached_sparse_predictor import CachedSparsePredictor


def verify_terminals(runner,query,bodies,terminals):
    h,r,t,removed,target=query
    runner.g.propagate=runner.original_p;runner.g.relation2adjacency=runner.adj
    checked=0;nnz=0;max_count=0
    for body,(ids,values) in zip(bodies,terminals):
        expected=runner.original_g(h,runner.relation,body,removed).cpu()
        actual=torch.zeros(expected.numel(),dtype=torch.int64);actual[ids]=values
        torch.testing.assert_close(actual.reshape_as(expected),expected,rtol=0,atol=0)
        checked+=1;nnz+=values.numel()
        if values.numel():max_count=max(max_count,int(values.max()))
    return {'exact_terminal_tensor_checks':checked,'nnz':nnz,'max_int64_count':max_count}


def gradient_check(runner,adapter,query):
    m=runner.m;h,r,t,removed,target=query;m.train();m.zero_grad(set_to_none=True)
    reference=runner.forward('upstream',h,r,removed);loss=loss_from(reference,target);loss.backward()
    expected={n:None if p.grad is None else p.grad.detach().clone() for n,p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    got=adapter(h,r,removed);gotloss=loss_from(got,target);gotloss.backward()
    torch.testing.assert_close(gotloss,loss,rtol=2e-5,atol=2e-5)
    errors={}
    for n,p in m.named_parameters():
        if expected[n] is None:assert p.grad is None
        else:
            torch.testing.assert_close(p.grad,expected[n],rtol=2e-4,atol=2e-5)
            errors[n]=float((p.grad-expected[n]).abs().max())
    m.zero_grad(set_to_none=True);m.eval()
    return {'passed':True,'gradient_parameters':len(errors),'max_gradient_abs_error':max(errors.values(),default=0),
            'logit_max_abs_error':assert_outputs(got,reference),'loss_abs_error':float(abs(gotloss-loss))}


def train_benchmark(runner,adapter,query,variants,warmup,repeats):
    m=runner.m;g=runner.g;device=runner.device
    h,r,t,removed,_=query;hc=h.cpu();rc=r.cpu();tc=t.cpu();ec=removed.cpu()
    target_cpu=torch.zeros(len(h),g.entity_size)
    for i,head in enumerate(hc.tolist()):target_cpu[i,g.hr2o.get(g.encode_hr(head,runner.relation),[])]=1
    opt=torch.optim.Adam(m.parameters(),lr=.005);m.train();opt.zero_grad()
    def forward(v,hh,rr,ee):return adapter(hh,rr,ee) if v=='cached_gpu_csr' else runner.forward(v,hh,rr,ee)
    def step(v):
        onehot=torch.nn.functional.one_hot(tc,g.entity_size)
        hh=hc.to(device);rr=rc.to(device);ee=ec.to(device)
        target=target_cpu.to(device)*.2+onehot.to(device)*.8
        logits,mask=forward(v,hh,rr,ee);assert mask.sum().item()>0
        loss=loss_from((logits,mask),target);loss.backward();opt.step();opt.zero_grad()
        return {'loss':loss.item(),'mask_count':mask.sum().item(),'updated':True}
    bootstrap=[step('upstream')['loss'] for _ in range(2)];sync(device)
    state=cpu_clone(m.state_dict());optstate=cpu_clone(opt.state_dict())
    rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state(device).clone()
    def restore():
        m.load_state_dict(state);opt.load_state_dict(copy.deepcopy(optstate));opt.zero_grad();m.train()
        torch.set_rng_state(rng);torch.cuda.set_rng_state(crng,device);sync(device)
    restore();expected_observation=step('upstream');sync(device)
    expected_state=cpu_clone(m.state_dict());expected_opt=cpu_clone(opt.state_dict())
    def check(observation):
        return compare_update(m,opt,state,expected_state,expected_opt,observation,expected_observation,3e-6,3e-3)
    result={v:{'samples_ms':[],'update_checks':[]} for v in variants}
    for v in variants:
        for _ in range(warmup):restore();ob=step(v);sync(device);check(ob)
    for repeat in range(repeats):
        order=list(variants);random.Random(987+repeat).shuffle(order)
        for v in order:
            restore();sync(device);start=time.perf_counter();ob=step(v);sync(device)
            result[v]['samples_ms'].append((time.perf_counter()-start)*1000);result[v]['update_checks'].append(check(ob))
    for v,item in result.items():item['timing']=stats(item.pop('samples_ms'));item['all_updates_verified']=True
    restore();m.eval()
    with torch.no_grad():error=assert_outputs(adapter(h,r,removed),runner.forward('upstream',h,r,removed))
    return {'bootstrap_upstream_losses':bootstrap,'variants':result,'updated_weights_forward_max_abs_error':error,
            'included':['CPU onehot','H2D queries/removed/targets','smoothing','forward','loss','backward','Adam','zero_grad','scalar logging and final sync'],
            'excluded':['CPU dataset construction and multihot assembly','cold cache build','DataLoader','epoch/EM','state restoration and comparison']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['FB15k-237:272','wn18rr:12'])
    p.add_argument('--splits',nargs='+',default=['train','valid']);p.add_argument('--threads',type=int,default=16)
    p.add_argument('--build-batch',type=int,default=32);p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--grad',action='store_true');p.add_argument('--training',action='store_true')
    p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda',torch.cuda.current_device());out=Path(a.output)
    result={'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),
            'config':vars(a),'torch':torch.__version__,'scope':'controlled fixed-rule lstm/sum, explicit per-query cache on real GPU; new mixed/reordered batches',
            'timing':'synchronized wall-clock, randomized interleaved variants, setup excluded and separately reported',
            'not_measured':['all relations/full dataset cache','online miss policy/eviction','automatic graph or rule mutation detection','complete epoch/EM/quality'],'cases':[]}
    def save():
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(out)
    save()
    try:
        for spec in a.cases:
            ds,relation=spec.split(':');relation=int(relation)
            for split in a.splits:
                variants=['upstream','csr','native64gpu_512']+(['graph'] if split=='train' else [])
                runner=CaseRunner(ds,relation,device,a.threads,variants)
                try:
                    groups=[sample(runner.g,relation,a.build_batch,split,i,device) for i in [0,1]]
                    if any(q is None for q in groups):raise ValueError(f'not enough queries for two groups: {spec}/{split}')
                    namespace=f'{ds}:c890b025:static-graph';version=hashlib.sha256(repr(runner.entries).encode()).hexdigest()
                    adapter=CachedSparsePredictor(runner.m,namespace=namespace,rule_version=version,rule_order='prefix')
                    bodies=[b for _,(_,b) in adapter.rules_for_relation(relation)]
                    native=runner.backends['native64'];prepared=native.prepare_rules(bodies)
                    row={'case':spec,'split':split,'checkpoint':str(runner.checkpoint),'checkpoint_sha256':hashlib.sha256(runner.checkpoint.read_bytes()).hexdigest(),
                         'native_metadata':native.metadata,'rule_plan_fingerprint':adapter.rule_fingerprint,'cold_build':[],'mixed_groups':[]}
                    result['cases'].append(row);save()
                    for index,q in enumerate(groups):
                        h,r,t,removed,_=q;sync(device);start=time.perf_counter()
                        hc=h.cpu();rc=r.cpu();ec=None if removed is None else removed.cpu();transferred=time.perf_counter()
                        terminals=native.batch(prepared,hc,relation,ec);built=time.perf_counter()
                        fill=adapter.preload_from_terminals(hc,rc,ec,terminals);sync(device);end=time.perf_counter()
                        checks=verify_terminals(runner,q,bodies,terminals)
                        row['cold_build'].append({'source_group':index,'query_count':len(h),'query_D2H_ms':(transferred-start)*1000,
                            'native_count_ms':(built-transferred)*1000,'cache_fill_ms':(end-built)*1000,'cold_total_ms':(end-start)*1000,
                            'cache_fill':fill,'correctness':checks,'queries':[{'h':int(hh),'removed':None if ec is None else int(ec[i])} for i,hh in enumerate(hc)]})
                        del terminals;save()
                    row['cache']=adapter.cache_statistics();row['cold_total_ms']=sum(x['cold_total_ms'] for x in row['cold_build'])
                    for label,q in mixed_groups(*groups):
                        h,r,t,removed,target=q
                        for capture in runner.captures.values():capture.close()
                        runner.captures.clear()
                        for v in variants:runner.prepare(v,h,removed)
                        with torch.no_grad():
                            ref=runner.forward('upstream',h,r,removed)
                            funcs={v:(lambda v=v:runner.forward(v,h,r,removed)) for v in variants};funcs['cached_gpu_csr']=lambda:adapter(h,r,removed)
                            errors={v:assert_outputs(fn(),ref) for v,fn in funcs.items()}
                            for _ in range(a.warmup):
                                for fn in funcs.values():fn();sync(device)
                            samples={v:[] for v in funcs}
                            for repeat in range(a.repeats):
                                order=list(funcs);random.Random(109+repeat).shuffle(order)
                                for v in order:
                                    sync(device);start=time.perf_counter();funcs[v]();sync(device);samples[v].append((time.perf_counter()-start)*1000)
                            gc.collect();sync(device);base=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
                            adapter(h,r,removed);sync(device)
                            memory={'base_allocated_bytes':base,'incremental_peak_bytes':torch.cuda.max_memory_allocated(device)-base}
                            adapter.collect_stats=True;adapter(h,r,removed);sync(device);diag=adapter.last_stats;adapter.collect_stats=False
                        group={'label':label,'batch':len(h),'forward':{v:stats(s) for v,s in samples.items()},'logit_max_abs_error':errors,
                               'cache_diagnostic':diag,'cached_incremental_memory':memory,
                               'queries':[{'h':int(hh),'removed':None if removed is None else int(removed[i])} for i,hh in enumerate(h.cpu())]}
                        if a.grad and split=='train':group['gradient']=gradient_check(runner,adapter,q)
                        if a.training and split=='train' and len(h)==32:group['training_batch']=train_benchmark(runner,adapter,q,list(funcs),a.warmup,a.repeats)
                        row['mixed_groups'].append(group);save()
                        print(json.dumps({'case':spec,'split':split,'group':label,'forward_ms':{v:x['median_ms'] for v,x in group['forward'].items()},
                              'cached_storage_MiB':row['cache']['persistent_unique_storage_bytes']/2**20,
                              'training_ms':{v:x['timing']['median_ms'] for v,x in group.get('training_batch',{}).get('variants',{}).items()}}),flush=True)
                finally:
                    runner.close()
                    for backend in runner.backends.values():
                        if hasattr(backend,'close'):backend.close()
                    del runner;gc.collect();torch.cuda.empty_cache()
        result['complete']=True;save()
    except Exception:
        result['failure']=traceback.format_exc();save();raise

if __name__=='__main__':main()
