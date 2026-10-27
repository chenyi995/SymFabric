"""Focused full/tile2048 real PNA training control using unchanged stage2 harness."""
import argparse, gc, hashlib, json, os, time, traceback
from pathlib import Path
from pna_bench import OfficialPNARunner, torch, sample, mixed_groups
from pna_predictor import CachedPNAPredictor
from bench_cached_sparse import verify_terminals, gradient_check, train_benchmark


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('allocated worker required')
    torch.set_num_threads(16);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42);device=torch.device('cuda')
    result={'complete':False,'job':os.getenv('SLURM_JOB_ID'),'gpu':torch.cuda.get_device_name(),
      'scope':'official WN18RR relation12 mixed B32 short training-batch control; original-full and tile2048 PNA on identical sparse cached counts; no full training epoch or compiled training claim',
      'timing':'unchanged stage2 training harness;2warmup5randomized interleaved restored-Adam samples; CPUtarget/H2D/smoothing/forward/loss/backward/Adam/logging included; restoration and checks excluded'}
    path=Path(a.output)
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
    save();runner=None
    try:
        runner=OfficialPNARunner(device,16)
        groups=[sample(runner.g,12,32,'train',i,device) for i in [0,1]]
        version=hashlib.sha256(repr(runner.entries).encode()).hexdigest()
        adapter=CachedPNAPredictor(runner.m,namespace='training-control',rule_version=version,candidate_block_size=0)
        prepared=runner.native.prepare_rules(runner.bodies);checks=[]
        for query in groups:
            h,r,t,removed,target=query
            terms=runner.native.batch(prepared,h.cpu(),12,removed.cpu())
            adapter.preload_from_terminals(h.cpu(),r.cpu(),removed.cpu(),terms)
            checks.append(verify_terminals(runner,query,runner.bodies,terms));del terms
        result['exact_count_checks']=checks
        label,query=next((label,q) for label,q in mixed_groups(*groups) if len(q[0])==32)
        result['mixed_group']=label;result['checkpoint']=str(runner.checkpoint)
        result['checkpoint_sha256']=hashlib.sha256(runner.checkpoint.read_bytes()).hexdigest()
        result['gradients']={}
        for block in [0,2048]:
            adapter.candidate_block_size=block
            result['gradients'][str(block)]=gradient_check(runner,adapter,query)
        save()
        original_forward=runner.forward
        def forward(variant,h,r,removed):
            if variant in ['cache_t0','cache_t2048']:
                adapter.candidate_block_size=int(variant[7:]);return adapter(h,r,removed)
            return original_forward(variant,h,r,removed)
        runner.forward=forward
        initial={n:t.detach().cpu().clone() for n,t in runner.m.state_dict().items()}
        try:
            result['training_batch']=train_benchmark(runner,adapter,query,
                ['upstream','csr','cache_t0','cache_t2048'],2,5)
        finally:
            runner.m.load_state_dict(initial);runner.m.zero_grad(set_to_none=True);runner.m.eval()
        result['complete']=True;save()
        print(json.dumps({'complete':True,'training_ms':{k:v['timing']['median_ms'] for k,v in result['training_batch']['variants'].items()}}),flush=True)
    except Exception:
        result['error']=traceback.format_exc();save();raise
    finally:
        if runner is not None:runner.close()
if __name__=='__main__':main()
