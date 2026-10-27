"""Official WN18RR emb/PNA: real GPU forward, gradients and Adam checks.

Run only through the allocated stage-3 worker. Frozen rule/graph checkpoint
controls; explicit cache preloading is separate from warm replay. No epoch,
EM convergence or paper accuracy claim is implied by this sampled experiment.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
S2 = ROOT.parent / '2026-10-14-rnnlogic-sparse-stage2'
S1 = ROOT.parent / '2026-10-14-rnnlogic-strong-baselines'
OLD = ROOT.parent / '2026-10-14-candidate-tests/kg'
sys.path[:0] = [str(S2), str(S1), str(OLD/'deps'), str(OLD/'upstream/RNNLogic/src')]
import torch
from data import KnowledgeGraph
from predictors import PredictorPlus
from prefix_cache import PrefixGrounder
from gpu_backend import GPUBackend
from official_native_adapter import OfficialNativeBackend, OfficialNativeController
from bench_stage2 import CPUInputController, sample
from bench_memoized import mixed_groups, assert_outputs
from bench_cached_sparse import verify_terminals, gradient_check, train_benchmark
from run_bench import sync, stats
from pna_predictor import PNAPredictor, CachedPNAPredictor


class OfficialPNARunner:
    def __init__(self, device, threads, relation=12):
        self.device = device; self.relation = relation
        self.g = KnowledgeGraph(str(OLD/'upstream/RNNLogic/data/wn18rr'))
        g = self.g
        self.original_g = g.grounding; self.original_p = g.propagate
        self.adj = g.relation2adjacency
        self.config = {'type':'emb','num_layers':3,'hidden_dim':16,'entity_feature':'bias','aggregator':'pna'}
        self.m = PredictorPlus(g, **self.config)
        self.m.set_rules(str(OLD/'upstream/RNNLogic/data/wn18rr/rnnlogic_rules.txt'))
        self.checkpoint = S1/'checkpoints/wn18rr_12_official_officialloss_v2_fit5.pt'
        checkpoint = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        self.m.load_state_dict(checkpoint['state'], strict=True)
        self.m.to(device).eval(); self.m.rule_features = self.m.rule_features.to(device)
        self.entries = self.m.relation2rules[relation]
        self.bodies = [tuple(body) for _, (_,body) in self.entries]
        self.native = OfficialNativeBackend(g, threads=threads, bits=64)
        self.gpu = GPUBackend(g, device=device, strategy='csr')
        self.ctrl = PrefixGrounder(g, self.bodies, policy='last_use', collect_stats=False)
        self.native_adapter = PNAPredictor(self.m,
            lambda graph,bodies: CPUInputController(OfficialNativeController(self.native,bodies)), candidate_block_size=128)
        self.resident_adj = [[z.to(device) for z in pair] for pair in self.adj]
        self.g.relation2adjacency = self.resident_adj

    def forward(self, variant, h, r, removed):
        self.g.relation2adjacency = self.resident_adj
        self.g.propagate = self.original_p
        self.g.grounding = self.original_g
        if variant == 'upstream':
            return self.m(h,r,removed)
        if variant == 'csr':
            self.g.propagate = self.gpu.propagate
            self.g.grounding = self.ctrl.grounding
            self.ctrl.begin(h,self.relation,removed)
            complete = False
            try:
                output = self.m(h,r,removed)
                complete = True
                return output
            finally:
                self.ctrl.finish(require_complete=complete)
        if variant == 'native64_t128':
            return self.native_adapter(h,r,removed)
        raise ValueError(variant)

    def close(self):
        self.g.grounding = self.original_g
        self.g.propagate = self.original_p
        self.g.relation2adjacency = self.adj
        self.native.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--build-batch', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--splits', nargs='+', default=['train','valid'])
    parser.add_argument('--candidate-blocks', type=int, nargs='+', default=[0,128,512,2048])
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('real PNA benchmark requires allocated stage-3 Slurm worker')
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    device = torch.device('cuda', torch.cuda.current_device())
    output = Path(args.output)
    result = {'complete':False,'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'config':vars(args),
              'pinned_linux_cpus':os.getenv('BENCH_PINNED_LINUX_CPUS'), 'affinity':sorted(os.sched_getaffinity(0)),
              'scope':'official WN18RR emb/PNA checkpoint; sampled relation-12 real GPU forward/gradient/Adam controls',
              'timing':'synchronized wall clock; interleaved randomized variants; cold cache construction separate',
              'memory_note':'per-variant incremental CUDA allocation peak over common resident graph/model/cache state; excludes allocator reserve and host native state',
              'upstream_baseline':'unchanged upstream neural/propagation arithmetic with GPU-resident adjacency; stronger control avoiding repeated adjacency H2D; differs from stage1 raw upstream CPU-adjacency timing',
              'not_measured':['full epoch/EM','paper accuracy','automatic graph/rule mutation detection','full dataset cache'],
              'splits':[]}
    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2)); temporary.replace(output)
    save()
    runner = None
    try:
        start = time.perf_counter()
        runner = OfficialPNARunner(device,args.threads)
        result['setup_seconds'] = time.perf_counter()-start
        result['checkpoint'] = str(runner.checkpoint)
        result['checkpoint_sha256'] = hashlib.sha256(runner.checkpoint.read_bytes()).hexdigest()
        result['model_config'] = runner.config
        result['state_shapes'] = {key:list(tensor.shape) for key,tensor in runner.m.state_dict().items()}
        result['native_metadata'] = runner.native.metadata
        result['gpu_csr_metadata'] = {'setup_seconds':runner.gpu.setup_seconds,'resident_bytes':runner.gpu.resident_bytes,'jit_metadata':runner.gpu.jit_metadata}
        result['relation_rule_count'] = len(runner.entries)
        save()
        for split in args.splits:
            groups = [sample(runner.g,runner.relation,args.build_batch,split,i,device) for i in [0,1]]
            if any(query is None for query in groups):
                raise ValueError('insufficient sampled query groups')
            version = hashlib.sha256(repr(runner.entries).encode()).hexdigest()
            adapter = CachedPNAPredictor(runner.m, namespace='wn18rr:c890b025:static',rule_version=version,candidate_block_size=128)
            bodies = [body for _,(_,body) in adapter.rules_for_relation(runner.relation)]
            prepared = runner.native.prepare_rules(bodies)
            row = {'split':split,'rule_plan_fingerprint':adapter.rule_fingerprint,'cold_build':[],'mixed_groups':[]}
            result['splits'].append(row);save()
            for number, query in enumerate(groups):
                h,r,t,removed,target = query
                sync(device); start = time.perf_counter()
                hc=h.cpu(); rc=r.cpu(); ec=None if removed is None else removed.cpu()
                copied = time.perf_counter()
                terminals = runner.native.batch(prepared,hc,runner.relation,ec)
                built = time.perf_counter()
                preload = adapter.preload_from_terminals(hc,rc,ec,terminals)
                sync(device); end = time.perf_counter()
                exact = verify_terminals(runner,query,bodies,terminals)
                row['cold_build'].append({'group':number,'query_D2H_ms':(copied-start)*1000,
                    'native_count_ms':(built-copied)*1000,'cache_fill_ms':(end-built)*1000,'cold_total_ms':(end-start)*1000,
                    'preload':preload,'exact_count_checks':exact})
                del terminals;save()
            row['cache'] = adapter.cache_statistics()
            for label, query in mixed_groups(*groups):
                h,r,t,removed,target = query
                base_variants = ['upstream','csr','native64_t128']
                funcs = {name:(lambda name=name:runner.forward(name,h,r,removed)) for name in base_variants}
                def cached(block):
                    adapter.candidate_block_size=block
                    return adapter(h,r,removed)
                funcs.update({f'cache_t{block}':(lambda block=block:cached(block)) for block in args.candidate_blocks})
                with torch.no_grad():
                    reference = runner.forward('upstream',h,r,removed)
                    errors = {name:assert_outputs(fn(),reference) for name,fn in funcs.items()}
                    for _ in range(args.warmup):
                        for fn in funcs.values():
                            fn();sync(device)
                    samples = {name:[] for name in funcs}
                    for repeat in range(args.repeats):
                        order=list(funcs); random.Random(903+repeat).shuffle(order)
                        for name in order:
                            sync(device); start=time.perf_counter(); funcs[name]();sync(device)
                            samples[name].append((time.perf_counter()-start)*1000)
                    memory = {}
                    for name,fn in funcs.items():
                        gc.collect();sync(device)
                        base=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
                        fn();sync(device)
                        memory[name]={'base_allocated_bytes':base,'incremental_peak_bytes':torch.cuda.max_memory_allocated(device)-base}
                group = {'label':label,'batch':len(h),'forward':{name:stats(value) for name,value in samples.items()},
                         'max_logit_error':errors,'incremental_memory':memory}
                row['mixed_groups'].append(group);save()
                adapter.candidate_block_size=128
                if split=='train':
                    group['cache_gradients'] = gradient_check(runner,adapter,query)
                    group['native_gradients'] = gradient_check(runner,runner.native_adapter,query)
                    if args.training and len(h)==32:
                        saved_training_start = {name:tensor.detach().cpu().clone() for name,tensor in runner.m.state_dict().items()}
                        try:
                            group['training_batch'] = train_benchmark(runner,adapter,query,
                                ['upstream','csr','native64_t128','cached_gpu_csr'],args.warmup,args.repeats)
                        finally:
                            runner.m.load_state_dict(saved_training_start)
                            runner.m.zero_grad(set_to_none=True)
                            runner.m.eval()
                print(json.dumps({'split':split,'group':label,'forward_ms':{name:timing['median_ms'] for name,timing in group['forward'].items()},
                                  'gradient_checks':split=='train','training':bool(group.get('training_batch'))}),flush=True)
                save()
            del funcs, cached, adapter, reference
            gc.collect();sync(device)
        result['complete']=True;save()
    except Exception as error:
        result['error']=repr(error);result['traceback']=traceback.format_exc();save()
        raise
    finally:
        if runner is not None:
            runner.close()


if __name__=='__main__':
    main()
