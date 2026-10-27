"""Matched real-GPU PredictorPlus training batches including transfers and Adam.

All variants start from exactly the same model, Adam and RNG snapshots on every
warmup, timed run, and separately measured memory run. The semantic reference is
an upstream next update from that state. This is not a complete epoch or EM.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time
import traceback

from bench_stage2 import CaseRunner, ROOT, S1, OLD, sync, stats
from bench_train_step import cpu_clone, compare_update
from data import TrainDataset
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',default=['FB15k-237:272','wn18rr:12'])
    parser.add_argument('--variants',nargs='+',default=['upstream','csr','graph','stream_128','gstream_128'])
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--group',type=int,default=0)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--smoothing',type=float,default=.2)
    parser.add_argument('--lr',type=float,default=.005)
    parser.add_argument('--parameter-atol',type=float,default=3e-6)
    parser.add_argument('--parameter-rtol',type=float,default=3e-3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.warmup<2 or args.repeats<5:parser.error('require at least 2 warmup and 5 timed runs')
    if not torch.cuda.is_available():raise RuntimeError('requires an allocated real CUDA GPU')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42)
    device=torch.device('cuda',torch.cuda.current_device())
    result={'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),
        'gpu':torch.cuda.get_device_name(device),'torch':torch.__version__,
        'affinity':sorted(os.sched_getaffinity(0)),
        'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'scope':'actual sampled PredictorPlus lstm_sum training batch, fixed learned rules and brief-fit checkpoint; not complete epoch/EM, convergence, or official WN emb/PNA',
        'included':['CPU one_hot(all_t)','h/r/multihot target/removed/onehot target_t H2D',
                    'target smoothing','complete PredictorPlus forward','mask sum scalar branch',
                    'softmax+1e-8/log/normalized masked loss','backward','Adam.step','optimizer.zero_grad',
                    'loss and mask scalar logging reads','final CUDA synchronization',
                    'any sparse hybrid D2H query and H2D compact terminal transfers inside forward'],
        'excluded':['TrainDataset construction and multihot CPU target assembly','DataLoader and sampler',
                    'checkpoint and graph loading','backend preparation and capture/JIT setup',
                    'identical-state restore and correctness comparison','epoch scheduler','rule generation/EM'],
        'state_note':'two untimed upstream training updates populate Adam; identical model/Adam/RNG snapshot restored outside every warmup, timed run and memory measurement; every next update compared to upstream',
        'timing_note':'synchronized wall-clock complete training step; randomized interleaved variant order; no memory/profiler instrumentation during timed repeats',
        'memory_note':'separate whole training-step peak; baseline includes all prepared backend graphs and simultaneously resident CUDA captures; incremental peak does not include that baseline; capture metadata reported separately',
        'tolerances':{'loss_atol':2e-5,'loss_rtol':2e-5,'parameter_update_atol':args.parameter_atol,
                      'parameter_update_rtol':args.parameter_rtol,'adam_moment_atol':3e-6,'adam_moment_rtol':3e-3},
        'cases':[]}
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(args.output)
    save()
    try:
        for spec in args.cases:
            ds,rr=spec.split(':');relation=int(rr)
            gc.collect();torch.cuda.empty_cache()
            setup_start=time.perf_counter()
            runner=CaseRunner(ds,relation,device,args.threads,args.variants)
            model=runner.m;graph=runner.g;model.train()
            checkpoint_hash=hashlib.sha256(runner.checkpoint.read_bytes()).hexdigest()
            random.seed(20261014+relation)
            train_set=TrainDataset(graph,batch_size=args.batch)
            eligible=[i for i,facts in enumerate(train_set.batches) if len(facts)==args.batch and facts[0][1]==relation]
            if len(eligible)<=args.group:raise ValueError(f'no full training batch group {args.group} for {spec}')
            batch_index=eligible[args.group]
            h_cpu,r_cpu,t_cpu,target_cpu,removed_cpu=train_set[batch_index]
            optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
            optimizer.zero_grad()
            prep_h=h_cpu.to(device);prep_removed=removed_cpu.to(device)
            prep={}
            for variant in args.variants:
                sync(device);started=time.perf_counter();runner.prepare(variant,prep_h,prep_removed);sync(device)
                prep[variant]={'seconds':time.perf_counter()-started}
                if variant in runner.captures:prep[variant]['capture']=runner.captures[variant].metadata
            del prep_h,prep_removed
            setup_seconds=time.perf_counter()-setup_start

            def step(variant):
                # This follows trainer.py: keep all_t CPU, transfer its onehot.
                target_t_cpu=torch.nn.functional.one_hot(t_cpu,graph.entity_size)
                h=h_cpu.to(device);r=r_cpu.to(device)
                target=target_cpu.to(device);removed=removed_cpu.to(device)
                target_t=target_t_cpu.to(device)
                target=target*args.smoothing+target_t*(1-args.smoothing)
                logits,mask=runner.forward(variant,h,r,removed)
                if mask.sum().item()==0:raise AssertionError('unexpected empty training update')
                logp=(torch.softmax(logits,dim=1)+1e-8).log()
                loss=-(logp[mask]*target[mask]).sum()/target[mask].sum().clamp(min=1)
                loss.backward();optimizer.step();optimizer.zero_grad()
                return {'loss':loss.item(),'mask_count':mask.sum().item(),'updated':True}

            print('bootstrap common next-update state',spec,flush=True)
            bootstrap=[step('upstream')['loss'] for _ in range(2)]
            sync(device)
            common_model=cpu_clone(model.state_dict());common_optimizer=cpu_clone(optimizer.state_dict())
            rng_cpu=torch.get_rng_state().clone();rng_cuda=torch.cuda.get_rng_state(device).clone()
            def restore():
                model.load_state_dict(common_model)
                optimizer.load_state_dict(copy.deepcopy(common_optimizer));optimizer.zero_grad()
                torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda,device)
                model.train();sync(device)
            restore();expected_observation=step('upstream');sync(device)
            expected_model=cpu_clone(model.state_dict());expected_optimizer=cpu_clone(optimizer.state_dict())
            def verify(observation):
                return compare_update(model,optimizer,common_model,expected_model,expected_optimizer,
                                      observation,expected_observation,args.parameter_atol,args.parameter_rtol)
            row={'dataset':ds,'relation':relation,'batch':args.batch,'group':args.group,
                 'model_config':{'type':'lstm','aggregator':'sum','hidden_dim':16,'num_layers':3,'entity_feature':'bias'},
                 'rules':len(runner.bodies),'checkpoint':str(runner.checkpoint),'checkpoint_sha256':checkpoint_hash,
                 'setup_seconds':setup_seconds,'preparation':prep,'backend_metadata':runner.metadata,
                 'optimizer_defaults':dict(optimizer.defaults),'untimed_upstream_bootstrap_losses':bootstrap,
                 'reference_next_step':expected_observation,'cpu_dataset_batch_index':batch_index,
                 'queries':[{'h':int(h_cpu[i]),'r':relation,'t':int(t_cpu[i]),'removed_edge':int(removed_cpu[i])} for i in range(args.batch)],
                 'target_positive_counts':target_cpu.sum(1).tolist(),
                 'variants':{v:{'warmup_checks':[],'measured_steps':[]} for v in args.variants}}
            result['cases'].append(row);save()
            for variant in args.variants:
                for iteration in range(args.warmup):
                    restore();observation=step(variant);sync(device)
                    row['variants'][variant]['warmup_checks'].append({'iteration':iteration,'observation':observation,**verify(observation)})
                print('training warmup/update verified',spec,variant,flush=True)
                save()
            for iteration in range(args.repeats):
                order=list(args.variants);random.Random(20261014+iteration).shuffle(order)
                for variant in order:
                    restore();sync(device);started=time.perf_counter()
                    observation=step(variant);sync(device)
                    elapsed_ms=(time.perf_counter()-started)*1000
                    check=verify(observation)
                    row['variants'][variant]['measured_steps'].append({'iteration':iteration,'elapsed_ms':elapsed_ms,'observation':observation,**check})
                save()
            for variant in args.variants:
                item=row['variants'][variant]
                item['training_batch']=stats([x['elapsed_ms'] for x in item['measured_steps']])
                item['all_steps_verified']=True
                restore();gc.collect();sync(device)
                base=torch.cuda.memory_allocated(device);reserved=torch.cuda.memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
                observation=step(variant);sync(device)
                absolute_peak=torch.cuda.max_memory_allocated(device)
                item['training_memory']={'base_allocated_bytes':base,'absolute_peak_allocated_bytes':absolute_peak,
                    'incremental_peak_allocated_bytes':absolute_peak-base,'base_reserved_bytes':reserved,
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                    'end_allocated_bytes':torch.cuda.memory_allocated(device),'measurement_separate_from_timed_repeats':True,
                    'update_check':verify(observation)}
                save()
            row['checkpoint_unchanged']=hashlib.sha256(runner.checkpoint.read_bytes()).hexdigest()==checkpoint_hash
            if not row['checkpoint_unchanged']:raise AssertionError('checkpoint changed')
            print(json.dumps({'case':spec,'train_ms':{v:row['variants'][v]['training_batch']['median_ms'] for v in args.variants},
                              'training_incremental_peak_MiB':{v:row['variants'][v]['training_memory']['incremental_peak_allocated_bytes']/2**20 for v in args.variants},
                              'all_steps_verified':True}),flush=True)
            save();runner.close()
            del runner,model,graph,optimizer,train_set,common_model,common_optimizer,expected_model,expected_optimizer
            gc.collect();torch.cuda.empty_cache()
        result['complete']=True;save()
    except Exception as error:
        result['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()};save();raise


if __name__=='__main__':main()
