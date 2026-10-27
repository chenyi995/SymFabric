"""Offline block-policy oracle: execute the real tiny CPU allocation manager."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'2026-10-14-rnnlogic-sparse-stage2'))
from cached_sparse_predictor import CachedSparsePredictor
from cache_lru import AllocationBlockLRU
from test_cache_lru import Model
from capacity_policy_replay import simulate_block_lru, simulate_query_lru


def per_key_coordinates(key):
    return [list(range(key % 4)), list(range(1, 1 + (key * 3) % 5))]


def terminal_batch(keys):
    result=[]
    for rule in range(2):
        coordinates=[]; values=[]
        for query,key in enumerate(keys):
            destinations=per_key_coordinates(int(key))[rule]
            coordinates.extend(query*16+entity for entity in destinations)
            values.extend([int(key)+1]*len(destinations))
        result.append((torch.tensor(coordinates,dtype=torch.int64),torch.tensor(values,dtype=torch.int64)))
    return result


def main():
    torch.set_num_threads(1)
    rng=np.random.default_rng(81071)
    payload=np.asarray([12*sum(map(len,per_key_coordinates(k)))+16*len(set(sum(per_key_coordinates(k),[])))+8 for k in range(9)],dtype=np.int64)
    fields=['lookups','hits','misses','unique_misses','admitted_blocks','admitted_keys','evicted_blocks','evicted_keys','evicted_bytes','rejected_admissions','peak_persistent_bytes']
    cases=[]
    for budget in [0,8,36,64,128,256,512]:
        for repetition in range(4):
            batches=[rng.integers(0,9,size=int(rng.integers(1,7))) for _ in range(14)]
            ids=np.concatenate(batches)
            offsets=np.concatenate(([0],np.cumsum([len(b) for b in batches])))
            epoch_offsets=np.array([0,7,14])
            modeled=simulate_block_lru(ids,offsets,payload,budget,epoch_offsets)
            adapter=CachedSparsePredictor(Model(),namespace='tiny-replay',rule_version='v1')
            manager=AllocationBlockLRU(adapter,budget)
            rejected_keys=0
            checkpoints=[]
            for i,keys in enumerate(batches):
                h=torch.tensor(keys,dtype=torch.int64);r=torch.zeros_like(h);removed=h.clone()
                context=manager.lookup(h,r,removed)
                positions=context['miss_indices']
                if positions:
                    missing=keys[positions]
                    fit,report=manager.admit(context,h[positions],r[positions],removed[positions],terminal_batch(missing))
                    if not fit:rejected_keys+=len(positions)
                manager.verify_invariants()
                if i+1 in epoch_offsets:
                    checkpoints.append(manager.statistics())
            actual=manager.statistics()
            for field in fields:
                assert modeled[field]==actual[field],(budget,repetition,field,modeled[field],actual[field])
            assert modeled['rejected_keys']==rejected_keys
            assert modeled['resident_bytes']==actual['persistent_unique_storage_bytes']
            assert modeled['resident_keys']==actual['cached_query_keys']
            assert modeled['resident_blocks']==actual['resident_blocks']
            previous={field:0 for field in fields}
            for epoch,point in zip(modeled['epochs'],checkpoints):
                for field in fields:
                    if field=='peak_persistent_bytes':continue
                    assert epoch[field]==point[field]-previous[field],(budget,repetition,field)
                assert epoch['resident_bytes']==point['persistent_unique_storage_bytes']
                previous=point
            cases.append(dict(budget_bytes=budget,repetition=repetition,batches=len(batches),accesses=len(ids),hits=actual['hits'],rejected_admissions=actual['rejected_admissions'],passed=True))
            manager.clear()
    # Separate serial ideal policy: known trace, differently sized entries, and
    # a rejected oversize item must not flush existing cache entries.
    ideal=simulate_query_lru(np.array([0,1,0,2,0,1]),np.array([8,16,32]),24)
    assert ideal['hits']==3 and ideal['misses']==3 and ideal['rejected_admissions']==1
    assert ideal['resident_bytes']==24 and ideal['evicted_keys']==0
    result=dict(complete=True,seed=81071,cases=cases,case_count=len(cases),
                checked_batches=sum(c['batches'] for c in cases),
                reference='Real CachedSparsePredictor CPU preloader plus AllocationBlockLRU actual tensor storages; every batch checks ownership/unique-storage invariants. Offline modeled counters and epoch deltas compared exactly.',
                ideal_query_lru_hand_case_passed=True)
    (ROOT/'capacity_policy_independent_tiny.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['complete','case_count','checked_batches','ideal_query_lru_hand_case_passed']}))


if __name__=='__main__':main()
