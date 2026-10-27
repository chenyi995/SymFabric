"""CPU-only export of exact FuncToNode boundary tensors for runtime comparison."""
import argparse, gc, hashlib, json, os, socket, time, traceback
from pathlib import Path
from runner import ROOT, OLD, S1, torch, KnowledgeGraph, PredictorPlus
from query_stream import iter_trace_batches, sha256_file
from spa_native_adapter import SPANativeBackend
from official_native_adapter import OfficialNativeBackend
from test_pna_predictor import TinyGraph, SparseInterface
from layers import FuncToNode
from pna_predictor import CachedPNAPredictor
import pna_predictor


class Captured(Exception): pass


def tensor_sha(tensor):
    return hashlib.sha256(memoryview(tensor.detach().contiguous().numpy())).hexdigest()


def capture(adapter,h,r,removed):
    result={};original=pna_predictor.pna_aggregate
    def intercept(layer,counts,embeddings,batch_ids,block=128):
        result.update(counts=counts.detach(),embeddings=embeddings.detach(),batch_ids=batch_ids.detach(),
          layer_state={n:t.detach().clone() for n,t in layer.state_dict().items()},
          hidden_dim=embeddings.size(1),layer_norm_kind='layer_norm')
        raise Captured()
    pna_predictor.pna_aggregate=intercept
    try:
        with torch.no_grad():adapter(h,r,removed)
    except Captured:pass
    finally:pna_predictor.pna_aggregate=original
    if not result:raise AssertionError('no candidates at layer boundary')
    return result


def save_fixture(directory,name,fixture,metadata):
    fixture['name']=name;fixture['format_version']=1;fixture['metadata']=metadata
    fixture['tensor_sha256']={key:tensor_sha(fixture[key]) for key in ['counts','embeddings','batch_ids']}
    path=directory/(name+'.pt');temporary=path.with_suffix('.tmp')
    torch.save(fixture,temporary);temporary.replace(path)
    # Round trip only pure tensors/primitive metadata; modern process needs no custom pickle classes.
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    assert loaded['name']==name and all(tensor_sha(loaded[k])==fixture['tensor_sha256'][k] for k in fixture['tensor_sha256'])
    del loaded
    return {'name':name,'path':str(path),'file_sha256':sha256_file(path),'file_bytes':path.stat().st_size,
      'shape':{'rules':fixture['counts'].size(0),'candidates':fixture['counts'].size(1),'hidden':fixture['hidden_dim']},
      'layer_norm_kind':fixture['layer_norm_kind'],'tensor_sha256':fixture['tensor_sha256'],'metadata':metadata}


def tiny_fixtures(directory,source):
    rows=[];torch.manual_seed(337)
    model=PredictorPlus(TinyGraph(),type='emb',hidden_dim=8,entity_feature='bias',aggregator='pna')
    model.set_rules([[0],[0,0],[0,0],[0,1],[0,0,1],[0,3]])
    with torch.no_grad():model.rule_emb[2].copy_(model.rule_emb[1])
    adapter=CachedPNAPredictor(model,SparseInterface,namespace='fixture-tiny',rule_version='v1',candidate_block_size=0)
    heads=torch.tensor([0,1,2,3,4,256]);edges=torch.tensor([0,1,2,3,4,5])
    adapter.preload(heads,torch.zeros_like(heads),edges)
    entries=adapter.rules_for_relation(0)
    for name,selected in [('tiny01_mixed5',[5,0,4,1,4]),('tiny02_mixed3',[5,2,5]),('tiny03_singleton',[5])]:
        h=heads[selected];r=torch.zeros_like(h);removed=edges[selected]
        fixture=capture(adapter,h,r,removed)
        dense=torch.stack([model.graph.grounding(h,0,body,removed).reshape(-1) for _,(_,body) in entries])
        candidates=torch.nonzero(dense.sum(0)).flatten()
        assert torch.equal(fixture['counts'],dense[:,candidates].float())
        assert torch.equal(fixture['batch_ids'],candidates//model.graph.entity_size)
        with torch.no_grad():
            fixture['eager_cpu_output']=model.rule_to_entity(fixture['counts'],fixture['embeddings'],fixture['batch_ids'])
            score,_=model(h,r,removed)
        fixture['candidate_ids']=candidates;fixture['queries']=torch.stack([h,r,removed],1)
        fixture['global_rule_ids']=torch.tensor([i for i,_ in entries])
        metadata={**source,'kind':'tiny_known_compile_failure_family','seed':337,'selected_head_indices':selected,
          'heads':h.tolist(),'removed_edges':removed.tolist(),'exact_original_counts_checked':True,
          'query_sha256':tensor_sha(fixture['queries']),
          'initialization_note':'All parameter creation is CPU in original failing script; moving the model to CUDA before CPU-created set_rules parameters consumes no random draws. Exact same CPU seed337 construction and duplicate embedding copy.',
          'original_model_score_first_head_candidate':float(score[0,int(h[0])]),
          'cpu_output_scope':'original pure FuncToNode CPU eager result; GPU eager reference must still be computed in each runtime'}
        rows.append(save_fixture(directory,name,fixture,metadata));del fixture,dense
    for feature,label in [(1,'min'),(2,'max')]:
        layer=FuncToNode(4);layer.layer_norm=torch.nn.Identity()
        with torch.no_grad():
            linear=layer.add_model.layers[0];linear.weight.zero_();linear.bias.zero_()
            for dim in range(4):linear.weight[dim,(feature*4+dim)*3]=1
        counts=torch.tensor([[9,0,0],[2,4,0],[0,8,6],[3,1,5]]).float()
        embeddings=torch.ones(4,4);batch_ids=torch.tensor([0,0,2])
        fixture={'counts':counts,'embeddings':embeddings,'batch_ids':batch_ids,'hidden_dim':4,'layer_norm_kind':'identity',
          'layer_state':{n:t.detach().clone() for n,t in layer.state_dict().items()},'expected_output':torch.ones(3,4),
          'expected_embeddings_gradient':torch.tensor([[1]*4,[1]*4,[1]*4,[0]*4]).float()}
        x=embeddings.clone().requires_grad_();out=layer(counts,x,batch_ids);out.sum().backward()
        assert torch.equal(out,fixture['expected_output']) and torch.equal(x.grad,fixture['expected_embeddings_gradient'])
        rows.append(save_fixture(directory,'tiny_tie_'+label,fixture,{**source,'kind':'isolated_first_index_tie',
          'loss':'output.sum()','normalizer_override':'Identity for exact isolated extrema channel check; explicit test fixture, not real checkpoint'}))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);a=ap.parse_args()
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('allocated CPU-only job required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('explicitly disable CUDA for fixture producer')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42)
    directory=Path(a.directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'manifest.json';started=time.perf_counter()
    layers=OLD/'upstream/RNNLogic/src/layers.py';data_dir=OLD/'upstream/RNNLogic/data/wn18rr'
    checkpoint=S1/'checkpoints/wn18rr_12_official_officialloss_v2_fit5.pt'
    source={'legacy_torch':str(torch.__version__),'official_layers_path':str(layers),'official_layers_sha256':sha256_file(layers)}
    result={'complete':False,'job':os.getenv('SLURM_JOB_ID'),'host':socket.gethostname(),'affinity':sorted(os.sched_getaffinity(0)),
      'scope':'pure FuncToNode runtime fixtures; exact integer grounding then original float32 boundary; no real neural work or CUDA on producer',
      'schema':{'required':['counts','embeddings','batch_ids','hidden_dim','layer_state','layer_norm_kind','metadata'],
        'state_loading':'FuncToNode(hidden_dim); replace layer_norm with Identity only when layer_norm_kind=identity; strict load_state_dict(layer_state)',
        'serialization':'torch.save plain tensor/primitive dict; torch.load(map_location=cpu,weights_only=True); no external class imports'},
      'source':source,'fixtures':[]}
    def save():
        result['elapsed_seconds']=time.perf_counter()-started
        tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(manifest)
    save();engines={}
    try:
        result['fixtures']+=tiny_fixtures(directory,source);save()
        graph=KnowledgeGraph(str(data_dir));model=PredictorPlus(graph,type='emb',num_layers=3,hidden_dim=16,entity_feature='bias',aggregator='pna')
        rules=data_dir/'rnnlogic_rules.txt';model.set_rules(str(rules));model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state'],strict=True);model.eval()
        source_real={**source,'dataset':'wn18rr','checkpoint':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint),
          'rules_sha256':sha256_file(rules),'graph_file_sha256':{name:sha256_file(data_dir/name) for name in ['entities.dict','relations.dict','train.txt','valid.txt','test.txt']},
          'checkpoint_scope':'official emb/PNA model after five short fitting steps; no accuracy/convergence claim',
          'model_config':{'type':'emb','num_layers':3,'hidden_dim':16,'entity_feature':'bias','aggregator':'pna'}}
        trace=ROOT/'query_stream_wn18rr.npz';source_real['trace_sha256']=sha256_file(trace)
        engines={'spa':SPANativeBackend(graph,threads=16,bits=64),'stdmap':OfficialNativeBackend(graph,threads=16,bits=64)}
        result['count_backends']={name:engine.metadata for name,engine in engines.items()}
        batches={}
        for b in iter_trace_batches(trace,epoch=0):
            rr=int(b['queries'][0,1])
            if rr in [12,8,7] and rr not in batches and len(b['queries'])==32:batches[rr]=b
            if len(batches)==3:break
        screen=json.loads((ROOT/'results/allrelation_wn18rr_tile2048.json').read_text())
        screen_by_relation={row['relation']:row for row in screen['cases']}
        for relation in [12,8,7]:
            b=batches[relation];query=torch.from_numpy(b['queries'].copy());h=query[:,0].contiguous();r=query[:,1].contiguous();removed=query[:,3].contiguous()
            entries=model.relation2rules[relation];bodies=[tuple(body) for _,(_,body) in entries]
            outputs={name:engine.batch(engine.prepare_rules(bodies),h,relation,removed) for name,engine in engines.items()}
            digest=hashlib.sha256();checked=0
            for (index,value),(other_index,other_value) in zip(outputs['spa'],outputs['stdmap']):
                assert torch.equal(index,other_index) and torch.equal(value,other_value)
                digest.update(memoryview(index.numpy()));digest.update(memoryview(value.numpy()));checked+=1
            assert checked==len(entries)
            screen_row=screen_by_relation[relation]
            assert screen_row['query_sha256']==b['query_sha256'] and screen_row['integer_terminals_equal'] and screen_row['gpu_native_integer_terminals_equal']
            adapter=CachedPNAPredictor(model,namespace='fixture-real',rule_version=source_real['rules_sha256'],candidate_block_size=0)
            adapter.preload_from_terminals(h,r,removed,outputs['spa'])
            fixture=capture(adapter,h,r,removed)
            candidate_ids=torch.unique(torch.cat([i for i,_ in outputs['spa']]),sorted=True)
            assert torch.equal(fixture['batch_ids'],candidate_ids//graph.entity_size)
            # Check every compact rule column against exact source terminals at the float boundary.
            for slot,(indices,values) in enumerate(outputs['spa']):
                positions=torch.searchsorted(candidate_ids,indices)
                assert torch.equal(fixture['counts'][slot,positions],values.float())
                assert int(torch.count_nonzero(fixture['counts'][slot]))==indices.numel()
            fixture['candidate_ids']=candidate_ids;fixture['queries']=query
            fixture['global_rule_ids']=torch.tensor([i for i,_ in entries])
            metadata={**source_real,'kind':'real_first_epoch0_B32','relation':relation,'actual_batch_index':b['batch_index'],
              'query_sha256':b['query_sha256'],'exact_spa_stdmap_terminal_rules_checked':checked,'exact_int64_terminal_sha256':digest.hexdigest(),
              'matching_prior_allrelation_integer_oracles':'both original upstream and GPU CSR matched SPA for this exact query hash in allrelation_wn18rr_tile2048.json',
              'original_float_boundary_checked_every_rule':True,'candidate_order':'ascending query_flat_entity; original rule order with distinct global IDs',
              'query_columns':['head','relation','tail','removed_edge_id']}
            row=save_fixture(directory,f'wn18rr_r{relation}',fixture,metadata);result['fixtures'].append(row);save()
            print(json.dumps({'fixture':row['name'],'shape':row['shape'],'bytes':row['file_bytes'],'exact_rules':checked}),flush=True)
            del adapter,fixture,outputs,candidate_ids;gc.collect()
        result['cuda_initialized']=torch.cuda.is_initialized();assert not result['cuda_initialized']
        result['complete']=True;save()
    except Exception:
        result['error']=traceback.format_exc();save();raise
    finally:
        for engine in engines.values():engine.close()
if __name__=='__main__':main()
