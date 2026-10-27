"""Small exact cross-batch/namespace/LRU checks, not performance measurements."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from sparse_cpu_backend import SparseCPUBackend
from sparse_cpu_backend_optimized import SparseCPUBackendOptimized
from memoized_grounder import MemoizedGrounder
from test_sparse_cpu import dense_step


def run(controller, graph, queries, bodies, *, head=0, remove=True):
    h = torch.tensor([q[0] for q in queries], dtype=torch.int64)
    removed = torch.tensor([q[1] for q in queries], dtype=torch.int64) if remove else None
    controller.begin(h, head, removed)
    for body in bodies:
        ids, values = controller.grounding(h, head, body, removed)
        assert ids.dtype == torch.int64 and values.dtype == torch.int64
        assert torch.all(values != 0) and len(ids) == len(torch.unique(ids))
        actual = torch.zeros(len(queries) * graph.entity_size, dtype=torch.int64)
        actual[ids] = values
        expected = torch.nn.functional.one_hot(h, graph.entity_size).T
        for rel in body:
            expected = dense_step(graph, expected, rel, removed if rel == head else None)
        torch.testing.assert_close(actual.reshape(len(queries), graph.entity_size), expected.T, rtol=0, atol=0)
        # Mutating a consumer's terminal cannot corrupt cached counts.
        values.zero_()
    return controller.finish()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output');args=p.parse_args()
    torch.set_num_threads(1)
    # Parallel edges e0,e1; deleting e2 instead must leave a different result.
    a=torch.tensor([[2,2,3,4,1,4,5],[1,1,1,2,4,3,5]],dtype=torch.int64)
    b=torch.tensor([[3,4,5,1],[2,3,4,5]],dtype=torch.int64)
    graph=SimpleNamespace(entity_size=7,relation2adjacency=[(a,None),(b,None)])
    bodies=[(),(0,),(0,0),(0,1),(0,1,0),(1,),(1,0)]
    groups=[[(1,0),(1,2),(2,3)], [(2,3),(1,0),(4,4)], [(4,4),(1,2),(1,0)]]
    results=[]
    for library in ['scipy','graphblas_optimized']:
        backend=SparseCPUBackend(graph,library='scipy') if library=='scipy' else SparseCPUBackendOptimized(graph)
        ctrl=MemoizedGrounder(graph,bodies,backend,namespace='graph-v1/rules-v1',collect_stats=True)
        first=run(ctrl,graph,groups[0],bodies)
        second=run(ctrl,graph,groups[1],bodies)
        third=run(ctrl,graph,groups[2],bodies)
        assert first['misses']==21 and first['hits']==0
        assert second['misses']==7 and second['hits']==14
        assert third['misses']==0 and third['hits']==21
        # All-pairs warm replay differs in order AND group from both prior batches.
        mixed=run(ctrl,graph,[(1,2),(4,4),(2,3),(1,0)],bodies)
        assert mixed['hits']==28 and mixed['builds']==0
        # Missing-edge key must distinguish evaluation from training.
        valid=run(ctrl,graph,groups[0],bodies,remove=False)
        assert valid['misses']==14 and valid['hits']==7  # two distinct head entities, repeated h=1
        valid_again=run(ctrl,graph,list(reversed(groups[0])),bodies,remove=False)
        assert valid_again['hits']==21 and valid_again['misses']==0
        # Different head relation also has a separate key (valid removes nothing).
        relation=run(ctrl,graph,groups[0],bodies,head=1,remove=False)
        assert relation['misses']==14
        # Disabled diagnostics must not change hit behavior or output/capacity.
        ctrl.collect_stats=False
        silent=run(ctrl,graph,groups[2],bodies)
        assert silent['hits'] is None and silent['misses'] is None
        ctrl.collect_stats=True
        # Namespace reset invalidates graph counts. Update graph and backend.
        graph2=SimpleNamespace(entity_size=7,relation2adjacency=[(torch.cat([a,torch.tensor([[6],[1]])],dim=1),None),(b,None)])
        backend2=SparseCPUBackend(graph2,library='scipy') if library=='scipy' else SparseCPUBackendOptimized(graph2)
        ctrl.reset_namespace('graph-v2/rules-v1',backend=backend2)
        reset=run(ctrl,graph2,groups[0],bodies)
        assert reset['misses']==21 and reset['start_cache_entries']==0
        # Capacity zero excludes even zero-entry terminals and bounds memory.
        uncached=MemoizedGrounder(graph,bodies,backend,namespace='g/r',capacity_bytes=0,collect_stats=True)
        c0=run(uncached,graph,groups[0],bodies)
        c1=run(uncached,graph,groups[0],bodies)
        assert c0['current_cache_entries']==0 and c1['hits']==0 and c1['misses']==21
        # LRU bounded to exactly two one-hot entries, with real eviction.
        single=[()]
        bounded=MemoizedGrounder(graph,single,backend,namespace='lru',capacity_bytes=230,collect_stats=True)
        r1=run(bounded,graph,groups[0],single)
        r2=run(bounded,graph,[groups[0][1]],single)
        r3=run(bounded,graph,[groups[0][0]],single)
        assert r1['current_cache_entries']==2 and r1['evictions']==1
        assert r2['hits']==1 and r3['misses']==1
        for row in [r1,r2,r3]:assert row['current_cache_charged_bytes']<=230
        # Explicit bulk priming uses two independently grouped native-style
        # terminal sets; a differently ordered four-query batch must then hit.
        primed=MemoizedGrounder(graph,bodies,backend,namespace='primed-g/r',collect_stats=True)
        priming=[]
        for queries in groups[:2]:
            hh=torch.tensor([q[0] for q in queries],dtype=torch.int64)
            rr=torch.tensor([q[1] for q in queries],dtype=torch.int64)
            terminals=[]
            for body in bodies:
                dense=torch.nn.functional.one_hot(hh,graph.entity_size).T
                for relation_id in body:
                    dense=dense_step(graph,dense,relation_id,rr if relation_id==0 else None)
                flat=dense.T.contiguous().reshape(-1)
                ids=torch.nonzero(flat,as_tuple=True)[0]
                terminals.append((ids,flat[ids]))
            priming.append(primed.prime_batch(hh,0,rr,terminals))
        assert priming[0]['new_entries']==21
        assert priming[1]['new_entries']==7 and priming[1]['existing_verified_entries']==14
        mixed_primed=run(primed,graph,[(1,2),(4,4),(2,3),(1,0)],bodies)
        assert mixed_primed['hits']==28 and mixed_primed['builds']==0
        # Reject duplicate indices; no silent summation or truncated duplicate.
        bad=MemoizedGrounder(graph,[()],backend,namespace='invalid-prime')
        try:
            bad.prime_batch(torch.tensor([1]),0,torch.tensor([0]),[(torch.tensor([1,1]),torch.tensor([1,1]))])
        except ValueError:
            pass
        else:raise AssertionError('duplicate priming indices accepted')
        # No query case is still valid, with empty terminal outputs.
        empty=run(ctrl,graph2,[],bodies)
        assert empty['query_lookups']==0
        results.append({'library':library,'first':first,'second_reordered_mixed':second,
                        'third_different_mixed':third,'all_query_mixed_warm':mixed,
                        'evaluation_key_separate':valid,'namespace_reset':reset,
                        'bounded_lru':r3,'bulk_priming':priming,'bulk_primed_new_group':mixed_primed})
    result={'passed':True,'tested_backends':2,
            'checks':['exact dense reference counts','same head different removed edge','reordered and mixed batches',
                      'None removal is a distinct key','head relation is a distinct key','consumer mutation isolation',
                      'namespace and backend replacement','bounded LRU and oversized entries','zero-capacity cache',
                      'disabled diagnostic counters','empty query batch'],
            'cases':results}
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'backends':len(results),'first_misses':21,'mixed_second_hits':14,
                      'mixed_second_misses':7,'all_query_new_group_hits':28},indent=2))

if __name__=='__main__':main()
