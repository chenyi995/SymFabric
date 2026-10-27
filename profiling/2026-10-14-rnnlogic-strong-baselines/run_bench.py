"""Frozen-model real-hardware ablations; upstream source is never edited."""
import argparse, collections, gc, hashlib, json, os, random, socket, statistics, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'2026-10-14-candidate-tests/kg'
sys.path[:0]=[str(OLD/'deps'), str(OLD/'upstream/RNNLogic/src')]
import torch
from data import KnowledgeGraph
from predictors import PredictorPlus
from prefix_cache import PrefixGrounder


def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def quantile(a,q):
    a=sorted(a); x=(len(a)-1)*q; i=int(x)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)


def stats(a):
    return {'median_ms':statistics.median(a),'p95_ms':quantile(a,.95),'min_ms':min(a),'max_ms':max(a),'samples_ms':a}


def official_loss(model,h,r,t,removed,graph,smoothing=.2):
    # Same normalized multi-label smoothing objective as upstream trainer.py.
    target=torch.zeros(len(h),graph.entity_size,device=h.device)
    for i,(hh,rr) in enumerate(zip(h.tolist(),r.tolist())):
        tails=graph.hr2o[graph.encode_hr(hh,rr)]
        target[i,tails]=1
    target=target*smoothing+torch.nn.functional.one_hot(t,graph.entity_size)*(1-smoothing)
    logits,mask=model(h,r,removed)
    logp=(torch.softmax(logits,dim=1)+1e-8).log()
    return -(logp[mask]*target[mask]).sum()/target[mask].sum().clamp(min=1)


def inputs(graph,relation,batch,split,device):
    facts=[(h,t) for h,r,t in getattr(graph,split+'_facts') if r==relation]
    random.Random(20261014+relation).shuffle(facts)
    facts=facts[:batch]
    if not facts: return None
    h=torch.tensor([x[0] for x in facts],device=device,dtype=torch.long)
    r=torch.full_like(h,relation); t=torch.tensor([x[1] for x in facts],device=device)
    removed=torch.tensor([graph.relation2ht2index[relation][graph.encode_ht(x,y)] for x,y in facts],device=device) if split=='train' else None
    return h,r,t,removed


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--cases',nargs='+',default=['FB15k-237:272','wn18rr:12'])
    p.add_argument('--threads',type=int,nargs='+',default=[1])
    p.add_argument('--batches',type=int,nargs='+',default=[32])
    p.add_argument('--splits',nargs='+',default=['train','valid'])
    p.add_argument('--variants',nargs='+',required=True)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--model',choices=['lstm_sum','official'],default='lstm_sum')
    p.add_argument('--fit-steps',type=int,default=5)
    p.add_argument('--rule-order',choices=['original','prefix'],default='original')
    p.add_argument('--gradient-check',action='store_true')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    device=torch.device(args.device)
    torch.set_num_threads(args.threads[0]); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42); random.seed(42)
    result={'device':str(device),'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'affinity':sorted(os.sched_getaffinity(0)),
      'pinned_linux_cpus':os.getenv('BENCH_PINNED_LINUX_CPUS'),'openmp_env':{k:os.getenv(k) for k in ['OMP_WAIT_POLICY','GOMP_SPINCOUNT','OMP_PLACES','OMP_PROC_BIND']},'torch':torch.__version__,'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
      'upstream_commit':subprocess.check_output(['git','-C',str(OLD/'upstream/RNNLogic'),'rev-parse','HEAD'],text=True).strip(),
      'config':vars(args),'scope':'resident PredictorPlus forward on sampled real train/valid queries; frozen brief-fit model, NOT paper accuracy or complete EM E2E',
      'memory_note':'all prepared backend graph allocations coexist; absolute peak is common-residency process peak, incremental peak isolates each forward; controller bytes are logical counts only',
      'timing':'synchronized wall clock; randomized interleaved variants; cache diagnostics and profiler excluded from timing; setup/JIT separate',
      'cases':[],'setup':[]}
    output=Path(args.output)
    def save():
        tmp=output.with_suffix(output.suffix+'.tmp'); tmp.write_text(json.dumps(result,indent=2)); tmp.replace(output)
    checkpoints=ROOT/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    grouped=collections.defaultdict(list)
    for case in args.cases:
        ds,rr=case.split(':'); grouped[ds].append(int(rr))
    for ds,relations in grouped.items():
        t0=time.perf_counter(); g=KnowledgeGraph(str(OLD/'upstream/RNNLogic/data'/ds))
        adj_cpu=g.relation2adjacency
        original_propagate=g.propagate; original_grounding=g.grounding
        resident_adj=[[z.to(device) for z in entry] for entry in adj_cpu]
        backends={}
        if device.type=='cuda':
            from gpu_backend import GPUBackend
            if any(v.startswith('atomic_') for v in args.variants): backends['atomic']=GPUBackend(g,device=device,strategy='atomic')
            if any(v.startswith('csr_') for v in args.variants): backends['csr']=GPUBackend(g,device=device,strategy='csr')
        elif any(v.startswith('native_') for v in args.variants):
            from cpu_backend import CPUBackend
            backends['native']=CPUBackend(g,threads=args.threads[0])
        sync(device)
        result['setup'].append({'dataset':ds,'graph_and_backend_seconds':time.perf_counter()-t0,
            'backends':{k:getattr(v,'metadata',{'setup_seconds':getattr(v,'setup_seconds',None),'resident_bytes':getattr(v,'resident_bytes',None),'jit_metadata':getattr(v,'jit_metadata',None)}) for k,v in backends.items()}})
        for relation in relations:
            torch.manual_seed(42+relation)
            config={'type':'lstm','hidden_dim':16,'num_layers':3,'aggregator':'sum','entity_feature':'bias'} if args.model=='lstm_sum' or ds=='FB15k-237' else {'type':'emb','hidden_dim':16,'num_layers':3,'aggregator':'pna','entity_feature':'bias'}
            m=PredictorPlus(g,**config).to(device)
            m.set_rules(str(OLD/'upstream/RNNLogic/data'/ds/'rnnlogic_rules.txt'));m.to(device)
            agg_original=m.rule_to_entity.forward
            def agg_gemm(count,emb,batch_ids):
                return torch.relu(m.rule_to_entity.layer_norm(m.rule_to_entity.add_model(count.T@emb)))
            g.grounding=original_grounding;g.propagate=original_propagate;g.relation2adjacency=resident_adj
            checkpoint=checkpoints/f'{ds}_{relation}_{args.model}_officialloss_v2_fit{args.fit_steps}.pt'
            if checkpoint.exists():
                ckpt=torch.load(checkpoint,map_location=device,weights_only=True);m.load_state_dict(ckpt['state']);fit_losses=ckpt['losses']
            else:
                h,r,t,removed=inputs(g,relation,32,'train',device)
                m.train(); opt=torch.optim.Adam(m.parameters(),lr=.005);fit_losses=[]
                for _ in range(args.fit_steps):
                    opt.zero_grad(set_to_none=True);loss=official_loss(m,h,r,t,removed,g);loss.backward();opt.step();fit_losses.append(float(loss.detach()))
                m.zero_grad(set_to_none=True)
                torch.save({'state':{k:v.detach().cpu() for k,v in m.state_dict().items()},'losses':fit_losses},checkpoint)
                if args.fit_steps: del opt,loss
            m.eval();m.zero_grad(set_to_none=True)
            if args.rule_order=='prefix' and config['aggregator']!='sum':
                raise ValueError('prefix-order gradient equivalence is only validated for sum aggregation')
            plan_start=time.perf_counter()
            original_relation_rules=list(m.relation2rules[relation])
            scheduled_relation_rules=sorted(original_relation_rules,key=lambda item:tuple(item[1][1])) if args.rule_order=='prefix' else original_relation_rules
            bodies=[tuple(body) for _,(_,body) in scheduled_relation_rules]
            schedule_setup_seconds=time.perf_counter()-plan_start
            for threads in args.threads:
                torch.set_num_threads(threads)
                if 'native' in backends:backends['native'].set_threads(threads)
                for split in args.splits:
                    for batch in args.batches:
                        inp=inputs(g,relation,batch,split,device)
                        if inp is None:continue
                        h,r,t,removed=inp
                        case={'dataset':ds,'relation':relation,'split':split,'requested_batch':batch,'batch':len(h),'threads':threads,
                          'rules':len(bodies),'body_steps':sum(map(len,bodies)), 'unique_prefixes':len({b[:i] for b in bodies for i in range(1,len(b)+1)}),
                          'model_config':config,'variant_rule_order':args.rule_order,'schedule_setup_seconds':schedule_setup_seconds,'checkpoint':str(checkpoint),'brief_fit_losses':fit_losses,'variants':{}}
                        controllers={};cache_stats={}
                        def select(variant):
                            if variant=='upstream':
                                m.relation2rules[relation]=original_relation_rules
                                g.grounding=original_grounding;g.propagate=original_propagate;g.relation2adjacency=adj_cpu
                                m.rule_to_entity.forward=agg_original
                                return None
                            m.relation2rules[relation]=scheduled_relation_rules
                            parts=variant.split('_');kind=parts[0];policy=parts[1]
                            g.relation2adjacency=resident_adj;g.propagate=original_propagate if kind=='torch' else backends[kind].propagate
                            m.rule_to_entity.forward=agg_gemm if config['aggregator']=='sum' else agg_original
                            if variant not in controllers:
                                budget=int(policy[7:])*1024**2 if policy.startswith('lastlru') else (int(policy[3:])*1024**2 if policy.startswith('lru') else None)
                                pol='last_use_lru' if policy.startswith('lastlru') else ('lru' if budget is not None else ('last_use' if policy=='last' else 'all'))
                                controllers[variant]=PrefixGrounder(g,bodies,policy=pol,budget_bytes=budget,collect_stats=False)
                            c=controllers[variant];g.grounding=c.grounding;c.begin(h,relation,removed);return c
                        def forward(variant,grad=False):
                            c=select(variant)
                            try:
                                with torch.set_grad_enabled(grad): out=m(h,r,removed)
                            finally:
                                if c is not None:cache_stats[variant]=c.finish()
                            return out
                        # Baseline remains same frozen model, input and exact graph semantics.
                        ref,refmask=forward('upstream');ref=ref.detach();refmask=refmask.detach()
                        for variant in args.variants:
                            print('validate',ds,relation,split,batch,variant,flush=True)
                            sync(device);t0=time.perf_counter();out,mask=forward(variant);sync(device)
                            first_call_ms=(time.perf_counter()-t0)*1000
                            err=float((out-ref).abs().max())
                            assert torch.equal(mask,refmask) and torch.allclose(out,ref,atol=2e-5,rtol=2e-4),(ds,relation,split,batch,variant,err)
                            case['variants'][variant]={'max_logit_error':err,'mask_equal':True,'first_call_ms':first_call_ms}
                            del out,mask
                        for _ in range(args.warmup):
                            for variant in args.variants:forward(variant)
                        timings={v:[] for v in args.variants}
                        for iteration in range(args.repeats):
                            order=list(args.variants);random.Random(997+iteration).shuffle(order)
                            for variant in order:
                                sync(device);t0=time.perf_counter();out=forward(variant);sync(device)
                                timings[variant].append((time.perf_counter()-t0)*1000);del out
                        for variant in args.variants:
                            row=case['variants'][variant];row['forward']=stats(timings[variant]);row['queries_per_second']=len(h)/(row['forward']['median_ms']/1000)
                            if variant in controllers:controllers[variant].collect_stats=True
                            if device.type=='cuda':
                                gc.collect();sync(device);base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                            out=forward(variant);sync(device)
                            if device.type=='cuda':
                                row['gpu_memory']={'base_allocated_bytes':base,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'incremental_peak_bytes':torch.cuda.max_memory_allocated()-base}
                            del out
                            row['cache']=cache_stats.get(variant)
                            if variant in controllers:controllers[variant].collect_stats=False
                        if args.gradient_check and split=='train' and batch==args.batches[0] and threads==args.threads[0]:
                            m.train();refgrads=None;grad_results={}
                            for variant in ['upstream']+[v for v in args.variants if v!='upstream']:
                                m.zero_grad(set_to_none=True);c=select(variant)
                                try:
                                    loss=official_loss(m,h,r,t,removed,g);loss.backward()
                                finally:
                                    if c is not None:c.finish()
                                grads={k:p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None}
                                if refgrads is None: refgrads=grads;refloss=float(loss.detach())
                                else:
                                    assert set(grads)==set(refgrads)
                                    errors={k:float((v-refgrads[k]).abs().max()) for k,v in grads.items()}
                                    ok=all(torch.allclose(v,refgrads[k],rtol=3e-3,atol=3e-5) for k,v in grads.items())
                                    assert ok,(variant,'gradient mismatch',errors)
                                    grad_results[variant]={'loss_abs_error':abs(float(loss.detach())-refloss),'max_gradient_abs_error':max(errors.values(),default=0),'allclose':ok}
                                del loss
                            case['gradient_checks']=grad_results;m.zero_grad(set_to_none=True);m.eval();del refgrads,grads
                        result['cases'].append(case);save()
                        print(json.dumps({'case':{k:case[k] for k in ['dataset','relation','split','batch','threads']},'ms':{v:case['variants'][v]['forward']['median_ms'] for v in args.variants}}),flush=True)
                        del controllers,ref,refmask
            g.grounding=original_grounding;g.propagate=original_propagate;g.relation2adjacency=adj_cpu
            del m
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
        del g,backends,resident_adj,adj_cpu
        gc.collect()
    result['complete']=True;save()

if __name__=='__main__':main()
