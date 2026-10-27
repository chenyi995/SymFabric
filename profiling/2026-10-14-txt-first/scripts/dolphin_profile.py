#!/usr/bin/env python3
"""Profile official DOLPHIN. Does not modify upstream code.
E2E: real MNIST, short supervised pretraining, original MNISTSumNNet forward.
Separate diagnostic instrumentation and synthetic kernel measurements.
"""
import argparse, ast, contextlib, copy, hashlib, json, os, platform, random
import statistics, subprocess, sys, time, traceback
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream" / "dolphin"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(UP))
import torch
import torchvision
from dolphin import Distribution
from dolphin.provenances.dtkp_am import DTKP_AM

def official_models():
    path = UP / "experiments/mnist/sum_n.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    # Only unused external telemetry is unavailable; no model/algorithm edits.
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Import) and any(a.name == "wandb" for a in n.names))]
    ns = {"__name__": "dolphin_profile_official_mnist", "__file__": str(path)}
    exec(compile(tree, str(path), "exec"), ns)
    return ns

def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

def clear_sources():
    Distribution._Distribution__input_instances.clear()

def stats(xs):
    ys = sorted(xs)
    return {"median_ms": statistics.median(ys), "min_ms": ys[0],
            "p95_ms": ys[min(len(ys)-1, int(.95*len(ys)))], "samples_ms": xs}

def measure(fn, device, reps, warm=2, setup=None):
    for _ in range(warm):
        if setup: setup()
        fn()
    sync(device)
    vals=[]
    for _ in range(reps):
        if setup: setup()
        sync(device); t=time.perf_counter()
        fn()
        sync(device); vals.append((time.perf_counter()-t)*1000)
    return stats(vals)

def fixtures(device):
    p=DTKP_AM(); p.k=4
    Distribution.provenance=p
    def tag(rows):
        x=torch.full((1,1,4,2),p._ZERO,device=device)
        x[0,0,:len(rows)] = torch.tensor(rows,device=device)
        return x
    def prob(x): return p.probs_from_tags(x.clone()).item()
    a=tag([[.7,p._NULL]])
    b=tag([[-.3,p._NULL]])
    rows=[]
    for name,x,expected in [
        ("shared_evidence",p.mul_batch(a,a),.7),
        ("conflicting_literal",p.mul_batch(a,b),0),
        ("duplicate_proof_add",p.add_batch(a,a),.7),
        ("absorption",tag([[.7,p._NULL],[.7,.6]]),.7),
        ("fully_specified_duplicate",tag([[.7,.6],[.7,.6]]),.42),
    ]:
        observed=prob(x)
        rows.append(dict(name=name,expected=expected,observed=observed,
                         passed=abs(observed-expected)<1e-6))
    for probabilities in ([.4,.6],[.5,.5]):
        clear_sources()
        d=Distribution(torch.tensor([probabilities],device=device),[0,1])
        observed=prob(p.mul_batch(d.tags[:,0:1],d.tags[:,1:2]))
        rows.append(dict(name="exclusive_same_source_"+str(probabilities),
                         expected=0.,observed=observed,passed=abs(observed)<1e-6))
    clear_sources()
    d=Distribution(torch.tensor([[.5,.5]],device=device),[0,1])
    actual=(d+d).get_probabilities().detach().cpu().tolist()[0]
    expected=[.5,0.,.5]
    rows.append(dict(name="same_distribution_reused_sum",expected=expected,
                     observed=actual,passed=all(abs(a-b)<1e-6 for a,b in zip(actual,expected))))
    clear_sources()
    d=Distribution(torch.tensor([[.5,.5]],device=device),[0,1])
    e=Distribution(torch.tensor([[.5,.5]],device=device),[0,1])
    actual=(d+e).get_probabilities().detach().cpu().tolist()[0]
    expected=[.25,.5,.25]
    rows.append(dict(name="independent_equal_probability_sources",expected=expected,
                     observed=actual,passed=all(abs(a-b)<1e-6 for a,b in zip(actual,expected))))
    clear_sources()
    return rows

@contextlib.contextmanager
def diagnostic_ranges(model, prov, device, synchronized=False):
    """Nested wall times; exclusive subtractions only for cartesian/mul."""
    totals=defaultdict(float); counts=defaultdict(int); originals=[]
    def attach(obj, attr, label):
        original=getattr(obj,attr); originals.append((obj,attr,original))
        def wrapped(*a,**kw):
            if synchronized: sync(device)
            t=time.perf_counter()
            with torch.profiler.record_function("dolphin/"+label):
                result=original(*a,**kw)
            if synchronized: sync(device)
            totals[label]+=(time.perf_counter()-t)*1000
            counts[label]+=1
            return result
        setattr(obj,attr,wrapped)
    attach(model.mnist_net,"forward","cnn")
    for method,label in [
        ("tags_from_probs","input_tags"),("combine_tag_sources","source_alignment"),
        ("cartesian_prod","cartesian_total"),("mul_batch","mul_batch"),
        ("reduce_symbols","reduce_symbols"),("probs_from_tags","recover")]:
        attach(prov,method,label)
    try: yield totals,counts
    finally:
        for obj,attr,old in reversed(originals): setattr(obj,attr,old)

def run_e2e(ns,args,result,save):
    data_root=OUT/"dolphin_mnist_data"
    transform=ns["mnist_img_transform"]
    trainset=torchvision.datasets.MNIST(data_root,train=True,download=True,transform=transform)
    testset=torchvision.datasets.MNIST(data_root,train=False,download=True,transform=transform)
    device="cuda"
    trainnet=ns["MNISTNet"]().to(device)
    opt=torch.optim.Adam(trainnet.parameters(),lr=.001)
    loader=torch.utils.data.DataLoader(trainset,batch_size=128,shuffle=True,num_workers=0)
    t0=time.perf_counter(); losses=[]; step=0
    trainnet.train()
    while step<args.train_steps:
        for x,y in loader:
            x=x.to(device);y=y.to(device)
            opt.zero_grad(set_to_none=True)
            out=trainnet(x)
            loss=torch.nn.functional.nll_loss(out.clamp_min(1e-8).log(),y)
            loss.backward();opt.step()
            losses.append(float(loss.detach()))
            step+=1
            if step>=args.train_steps: break
    sync(device)
    trainnet.eval(); correct=0; total=0
    with torch.no_grad():
        for x,y in torch.utils.data.DataLoader(torch.utils.data.Subset(testset,range(2000)),batch_size=128):
            pred=trainnet(x.to(device)).argmax(-1).cpu()
            correct+=int((pred==y).sum());total+=len(y)
    state={k:v.detach().cpu().clone() for k,v in trainnet.state_dict().items()}
    checkpoint=OUT/"dolphin_digit_checkpoint.pt"
    torch.save(state,checkpoint)
    result["training"]={"supervised_digit_steps":step,"batch_size":128,"lr":.001,
        "seconds_including_validation":time.perf_counter()-t0,
        "loss_last_20":sum(losses[-20:])/len(losses[-20:]),
        "heldout_digit_accuracy":correct/total,"heldout_digit_count":total,
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "scope":"Short supervised pretraining of official CNN; not paper E2E training or paper-final accuracy."}
    del trainnet,opt; torch.cuda.empty_cache(); save()
    # Use fixed held-out images, real labels; none used in training.
    samples=[testset[i] for i in range(2048)]
    all_x=torch.stack([x for x,y in samples]); all_y=torch.tensor([y for x,y in samples])
    result["e2e"]=[]
    for dev in ("cpu","cuda"):
        model=ns["MNISTSumNNet"]().to(dev).eval()
        model.mnist_net.load_state_dict(state)
        for n in (2,4,8):
            b=args.batch_size
            host=tuple(all_x[j*b:(j+1)*b].contiguous() for j in range(n))
            target=all_y[:n*b].view(n,b).sum(0)
            resident=tuple(x.to(dev) for x in host)
            for k in (1,2,4,8,16):
                p=DTKP_AM();p.k=k;Distribution.provenance=p
                def resident_fn(): return model(resident)
                def host_fn(): return model(tuple(x.to(dev) for x in host))
                with torch.no_grad():
                    rec={"device":dev,"sum_n":n,"batch_size":b,"k":k,
                         "scope":"steady model inference; CNN, Distribution, symbolic operations, probability recovery; no data download/load or model initialization"}
                    rec["resident"]=measure(resident_fn,dev,args.reps,setup=clear_sources)
                    rec["host_to_prediction"]=measure(host_fn,dev,args.reps,setup=clear_sources)
                    clear_sources(); y=resident_fn().detach().cpu()
                    rec["batch_sum_accuracy"]=float((y.argmax(-1)==target).float().mean())
                    rec["probability_mass_mean"]=float(y.sum(-1).mean())
                    rec["outputs_finite"]=bool(torch.isfinite(y).all())
                    # Separate 256-sample accuracy, not timed.
                    correct=0;count=0
                    for off in range(0,256,b):
                        inputs=tuple(all_x[j*256+off:j*256+off+b].to(dev) for j in range(n))
                        truth=torch.stack([all_y[j*256+off:j*256+off+b] for j in range(n)]).sum(0)
                        clear_sources(); probs=model(inputs).detach().cpu()
                        correct+=int((probs.argmax(-1)==truth).sum());count+=len(truth)
                    rec["heldout_sum_accuracy"]=correct/count;rec["heldout_sum_count"]=count
                    # Separate synchronized diagnostic; not used as unprofiled percentages.
                    clear_sources()
                    with diagnostic_ranges(model,p,dev,synchronized=True) as (times,counts):
                        resident_fn()
                    rec["diagnostic_synchronized_stage_ms"]=dict(times)
                    rec["diagnostic_counts"]=dict(counts)
                    rec["diagnostic_note"]="Separate run; each region synchronizes CUDA. Nested cartesian_total contains mul_batch. Stage timings do not equal natural unsynchronized critical path."
                    if dev=="cuda":
                        clear_sources();sync(dev)
                        base=torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        resident_fn();sync(dev)
                        rec["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
                        rec["peak_increment_bytes"]=rec["peak_allocated_bytes"]-base
                    result["e2e"].append(rec);save()
                    print("E2E",dev,n,k,round(rec["resident"]["median_ms"],3),
                          rec["heldout_sum_accuracy"],flush=True)
                    if dev=="cuda" and n==8 and k==16:
                        clear_sources()
                        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
                        with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as prof:
                            with diagnostic_ranges(model,p,dev,synchronized=False):
                                resident_fn()
                            sync(dev)
                        prof.export_chrome_trace(str(OUT/"dolphin_e2e_cuda_trace.json"))
                        (OUT/"dolphin_e2e_cuda_profiler.txt").write_text(
                            prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=40))
        del model;clear_sources()
        if dev=="cuda":torch.cuda.empty_cache()

def run_micro(args,result,save):
    result["micro"]=[]
    for dev in ("cpu","cuda"):
        for facts,length in ((16,2),(64,4),(256,8)):
            for k in (1,2,4,8,16):
                p=DTKP_AM();p.k=k
                gen=torch.Generator().manual_seed(4321)
                shape=(8,16,k,facts)
                probabilities=.1+.8*torch.rand((facts,),generator=gen)
                a=torch.full(shape,p._NULL);b=a.clone()
                for x in (a,b):
                    mask=torch.rand(shape,generator=gen).topk(length,dim=-1).indices
                    vals=probabilities.expand(shape).gather(-1,mask)
                    x.scatter_(-1,mask,vals)
                a=a.to(dev);b=b.to(dev)
                fn=lambda:p.mul_batch(a,b)
                with torch.no_grad():
                    rec={"device":dev,"k":k,"F_columns":facts,"L_literals":length,
                         "shape":list(shape),"synthetic":True,"E2E":False,
                         "mul_batch":measure(fn,dev,args.reps)}
                    aa=a.view(-1,k,facts);bb=b.view(-1,k,facts)
                    repeat=lambda:(aa.repeat_interleave(k,dim=1).view(-1,facts),
                                   bb.repeat(1,k,1).view(-1,facts))
                    ar,br=repeat()
                    union=lambda:p._DTKP_AM__union_proofs(ar,br).view(128,-1,facts)
                    c=union()
                    score=lambda:torch.prod(p._DTKP_AM__norm(c),dim=-1)
                    probs=score()
                    select=lambda:c[(torch.arange(c.shape[0]).unsqueeze(-1),torch.topk(probs,k).indices)]
                    rec["stage_replay"]={"repeat":measure(repeat,dev,args.reps),
                        "union":measure(union,dev,args.reps),"score":measure(score,dev,args.reps),
                        "topk_gather":measure(select,dev,args.reps)}
                    rec["stage_replay_matches_official"]=bool(torch.equal(select().view(shape),fn()))
                    rec["stage_note"]="Separate precomputed-stage replay, not additive E2E or optimized replacement."
                    if dev=="cuda":
                        sync(dev);base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                        fn();sync(dev)
                        rec["peak_increment_bytes"]=torch.cuda.max_memory_allocated()-base
                    result["micro"].append(rec);save()
                    print("MICRO",dev,facts,k,round(rec["mul_batch"]["median_ms"],3),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--train-steps",type=int,default=500)
    ap.add_argument("--batch-size",type=int,default=16);ap.add_argument("--reps",type=int,default=5)
    ap.add_argument("--micro-only",action="store_true");args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1234);random.seed(1234)
    torch.backends.cudnn.benchmark=False
    result={"scope":"Official DOLPHIN software baseline; diagnostic, not hardware speedup",
        "upstream_commit":subprocess.check_output(["git","-C",str(UP),"rev-parse","HEAD"],text=True).strip(),
        "upstream_dirty":subprocess.check_output(["git","-C",str(UP),"status","--porcelain"],text=True),
        "torch":torch.__version__,"torchvision":torchvision.__version__,
        "host":platform.node(),"cpu":platform.processor(),"threads":torch.get_num_threads(),
        "gpu":torch.cuda.get_device_name(),"cuda":torch.version.cuda,
        "slurm_job":os.getenv("SLURM_JOB_ID"),"args":vars(args),
        "notes":["Upstream untouched; unused import wandb omitted when loading official experiment module.",
                 "Source instance registry cleared between requests outside timing to avoid retained input tensors; no clearing within request.",
                 "Default absorption pre/post topk false, final probability absorption true.",
                 "No full paper training; supervised CNN pretraining with real held-out MNIST E2E inference."]}
    out=OUT/"dolphin_profile.json"
    def save():out.write_text(json.dumps(result,indent=2))
    try:
        result["fixtures"]={d:fixtures(d) for d in ("cpu","cuda")};save()
        if not args.micro_only:run_e2e(official_models(),args,result,save)
        run_micro(args,result,save)
        result["complete"]=True;save()
    except Exception:
        result["error"]=traceback.format_exc();save()
        raise

def trace_kernel_summary():
    events=json.loads((OUT/"dolphin_e2e_cuda_trace.json").read_text())["traceEvents"]
    cpu={e.get("args",{}).get("External id"):e for e in events if e.get("cat")=="cpu_op"}
    ranges=[e for e in events if e.get("cat")=="user_annotation"]
    totals=defaultdict(float); counts=defaultdict(int); memcpy=defaultdict(float)
    for e in events:
        if e.get("cat") not in ("kernel","gpu_memcpy"):continue
        c=cpu.get(e.get("args",{}).get("External id"))
        label="unmapped"
        if c:
            parents=[r for r in ranges if r.get("tid")==c.get("tid") and
                     r["ts"]<=c["ts"] and r["ts"]+r["dur"]>=c["ts"]+c["dur"]-0.1]
            if parents:label=min(parents,key=lambda r:r["dur"])["name"]
        if e["cat"]=="kernel":
            totals[label]+=e["dur"]/1000;counts[label]+=1
        else:memcpy[label]+=e["dur"]/1000
    result={"scope":"GPU kernel durations attributed once via External id to innermost CPU user range. Excludes GPU user annotation durations; not an E2E critical-path percentage.",
            "kernel_ms":dict(totals),"kernel_count":dict(counts),
            "kernel_total_ms":sum(totals.values()),"memcpy_ms":dict(memcpy)}
    (OUT/"dolphin_kernel_attribution.json").write_text(json.dumps(result,indent=2))
    return result

def quality_main():
    """Stronger software baseline for independent MNIST sums, not general NeSy."""
    torch.set_num_threads(4);torch.manual_seed(1234);random.seed(1234)
    ns=official_models();state=torch.load(OUT/"dolphin_digit_checkpoint.pt",map_location="cpu",weights_only=True)
    ds=torchvision.datasets.MNIST(OUT/"dolphin_mnist_data",train=False,download=False,
                                 transform=ns["mnist_img_transform"])
    samples=[ds[i] for i in range(2048)]
    allx=torch.stack([x for x,y in samples]);ally=torch.tensor([y for x,y in samples])
    output={"scope":"Exact probability convolution is equivalent for these independent-digit sum tasks only; does not replace shared-evidence/general NeSy proof semantics.",
            "upstream_commit":subprocess.check_output(["git","-C",str(UP),"rev-parse","HEAD"],text=True).strip(),
            "gpu":torch.cuda.get_device_name(),"host":platform.node(),"threads":4,
            "slurm_job":os.getenv("SLURM_JOB_ID"),
            "training_checkpoint_sha256":hashlib.sha256((OUT/"dolphin_digit_checkpoint.pt").read_bytes()).hexdigest(),
            "timing_protocol":"10 warmups per method; 7 randomized interleaved repetitions; inputs initially on host; same trained official CNN; synchronize whole call, no profiler; registry clear outside timed call.",
            "timing":[],"quality":[]}
    def save():(OUT/"dolphin_exact_comparison.json").write_text(json.dumps(output,indent=2))
    try:
        output["trace_kernel_attribution"]=trace_kernel_summary();save()
        for dev in ("cpu","cuda"):
            model=ns["MNISTSumNNet"]().to(dev).eval();model.mnist_net.load_state_dict(state)
            indices={length:(torch.arange(length,device=dev)[:,None]+
                     torch.arange(10,device=dev)[None,:]).reshape(1,-1)
                     for length in range(1,74)}
            def convolve(probs):
                acc=probs[0]
                for p in probs[1:]:
                    b,length=acc.shape
                    vals=(acc.unsqueeze(-1)*p.unsqueeze(1)).flatten(1)
                    nxt=torch.zeros((b,length+9),device=dev)
                    nxt.scatter_add_(1,indices[length].expand(b,-1),vals)
                    acc=nxt
                return acc
            def exact(inputs):
                return convolve([model.mnist_net(x) for x in inputs])
            with torch.no_grad():
                for n in (2,4,8):
                    b=16;host=tuple(allx[j*b:(j+1)*b] for j in range(n))
                    methods={"exact_convolution":lambda:exact(tuple(x.to(dev) for x in host))}
                    provs={}
                    for k in (1,4,16):
                        provs[k]=DTKP_AM();provs[k].k=k
                        def run(k=k):
                            Distribution.provenance=provs[k]
                            return model(tuple(x.to(dev) for x in host))
                        methods["dolphin_k"+str(k)]=run
                    for name,fn in methods.items():
                        for _ in range(10):
                            clear_sources();fn()
                        sync(dev)
                    vals={name:[] for name in methods}
                    for rep in range(7):
                        order=list(methods);random.shuffle(order)
                        for name in order:
                            clear_sources();sync(dev);t=time.perf_counter()
                            methods[name]();sync(dev)
                            vals[name].append((time.perf_counter()-t)*1000)
                    rec={"device":dev,"sum_n":n,"batch_size":b,
                         "host_to_prediction":{name:stats(v) for name,v in vals.items()}}
                    output["timing"].append(rec);save()
                    print("EXACT_TIMING",dev,n,{name:round(statistics.median(v),3) for name,v in vals.items()},flush=True)
                    # 256 held-out examples on GPU, first batch CPU/GPU difference recorded separately.
                    if dev!="cuda":continue
                    metric=defaultdict(lambda:defaultdict(float)); count=0
                    for off in range(0,256,b):
                        inputs=tuple(allx[j*256+off:j*256+off+b].to(dev) for j in range(n))
                        truth=torch.stack([ally[j*256+off:j*256+off+b] for j in range(n)]).sum(0).to(dev)
                        ref=exact(inputs)
                        for name in ("exact_convolution","dolphin_k1","dolphin_k4","dolphin_k16"):
                            if name=="exact_convolution":pred=ref
                            else:
                                k=int(name[9:]);Distribution.provenance=provs[k]
                                clear_sources();pred=model(inputs)
                            m=metric[name]
                            m["correct"]+=float((pred.argmax(-1)==truth).sum())
                            m["l1_to_exact_sum"]+=float((pred-ref).abs().sum(-1).sum())
                            m["probability_mass_sum"]+=float(pred.sum(-1).sum())
                            m["nll_sum"]+=float(-pred.gather(1,truth[:,None]).clamp_min(1e-12).log().sum())
                            m["argmax_matches_exact"]+=float((pred.argmax(-1)==ref.argmax(-1)).sum())
                        count+=len(truth)
                    q={"sum_n":n,"device":dev,"heldout_examples":count,
                       "metrics":{name:{"accuracy":m["correct"]/count,
                           "mean_L1_to_exact":m["l1_to_exact_sum"]/count,
                           "mean_probability_mass":m["probability_mass_sum"]/count,
                           "mean_NLL_unrenormalized":m["nll_sum"]/count,
                           "argmax_agreement_exact":m["argmax_matches_exact"]/count}
                           for name,m in metric.items()}}
                    output["quality"].append(q);save()
        output["complete"]=True;save()
    except Exception:
        output["error"]=traceback.format_exc();save();raise

if __name__=="__main__":
    if '--fixtures-only' in sys.argv:
        torch.set_num_threads(4)
        result={d:fixtures(d) for d in ('cpu','cuda')}
        (OUT/'dolphin_semantics.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    elif '--quality-only' in sys.argv: quality_main()
    else: main()

