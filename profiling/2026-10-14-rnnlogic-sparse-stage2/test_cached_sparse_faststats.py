"""Incremental statistics must equal the base physical-storage scan exactly."""
import argparse
import json
from pathlib import Path
import torch
from test_streaming_predictor import TinyGraph, SparseInterface, evaluate, compare
from cached_sparse_predictor import CachedSparsePredictor
from cached_sparse_faststats_predictor import CachedSparseFastStatsPredictor
from predictors import PredictorPlus


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device(args.device)
    graph=TinyGraph();model=PredictorPlus(graph,type='emb',hidden_dim=8,entity_feature='bias',aggregator='sum')
    model.set_rules([[0],[0,0],[0,0],[0,0,1],[0,1],[0,3],[1,1]])
    model.to(device)
    cache=CachedSparseFastStatsPredictor(model,SparseInterface,namespace='tiny',rule_version='v1',collect_stats=True)
    checks=[]
    def check(label):
        fast=cache.cache_statistics();scan=CachedSparsePredictor.cache_statistics(cache)
        assert fast==scan,(label,fast,scan)
        checks.append({'label':label,'stats':fast})
    check('initial_empty')
    h=torch.tensor([0,1,2,256]);r=torch.zeros_like(h);removed=torch.tensor([0,1,2,3])
    cache.preload(h,r,removed);check('four_entries_share_four_packed_storage_arrays')
    h2=torch.tensor([0,4,4]);r2=torch.zeros_like(h2);removed2=torch.tensor([0,5,5])
    before=cache.cache_statistics();fill=cache.preload(h2,r2,removed2)
    assert fill['new_keys']==1
    check('duplicate_existing_and_same_batch_keys_only_add_one_entry')
    cache.preload(h,r,removed);assert cache.cache_statistics()==checks[-1]['stats'];check('no_op_duplicate_preload')
    cache.preload(h,r,None);check('separate_valid_query_keys')
    hh=torch.tensor([4,0,256,4],device=device);rr=torch.zeros_like(hh);ee=torch.tensor([5,0,3,5],device=device)
    error=compare(evaluate(model,cache,hh,rr,ee),evaluate(model,model,hh,rr,ee))
    check('forward_backward_do_not_change_persistent_counters')
    key=next(iter(cache.cache));entry=cache.cache[key]
    rejected=[]
    for label,operation in [('replace',lambda:cache.cache.__setitem__(key,entry)),('delete',lambda:cache.cache.__delitem__(key)),
                            ('pop',lambda:cache.cache.pop(key)),('update',lambda:cache.cache.update({key:entry})),
                            ('setdefault',lambda:cache.cache.setdefault(key,entry)),('union',lambda:cache.cache.__ior__({key:entry}))]:
        try:operation()
        except (KeyError,TypeError):rejected.append(label)
        else:raise AssertionError(label+' mutation was not rejected')
    check('unsupported_mutations_leave_counters_unchanged')
    cache.reset(namespace='tiny2',rule_version='v2');check('reset_clears_storage_identity_map_and_all_counters')
    cache.preload(h,r,removed);check('refill_after_reset_and_possible_allocator_address_reuse')
    # Entirely empty terminal states still store rowptr entries; tensor storages
    # with data_ptr==0 must not corrupt physical-storage accounting.
    model2=PredictorPlus(graph,type='emb',hidden_dim=8,entity_feature='bias',aggregator='sum')
    model2.set_rules([[0,3]]);model2.to(device)
    cache2=CachedSparseFastStatsPredictor(model2,SparseInterface,namespace='empty',rule_version='v1')
    cache2.preload(h,r,None)
    assert cache2.cache_statistics()==CachedSparsePredictor.cache_statistics(cache2)
    assert cache2.cache_statistics()['persistent_tensor_payload_bytes']==len(h)*8
    result={'passed':True,'device':str(device),'statistics_checks':checks,'empty_states_passed':True,
            'unsupported_mutations_rejected':rejected,'forward_max_abs_error':error[0],'gradient_max_abs_error':error[1],
            'scope':'metadata-only adaptation compared against base full physical-storage scan; mixed/repeated-query forward and parameter gradients checked'}
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'device':str(device),'stats_checks':len(checks)+1,'forward_error':error[0],'gradient_error':error[1]}))

if __name__=='__main__':main()
