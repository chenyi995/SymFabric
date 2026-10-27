"""Tiny exact count checks for local SPA against std::map and original Python."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from query_stream import upstream_data
from spa_native_adapter import SPANativeBackend
from official_native_adapter import OfficialNativeBackend

KnowledgeGraph=upstream_data().KnowledgeGraph
class Graph:
    grounding=KnowledgeGraph.grounding
    propagate=KnowledgeGraph.propagate
    def __init__(self,duplicate=False):
        self.entity_size=9;self.relation_size=3
        edges=[[(0,1),(0,2),(1,3),(2,3),(3,0),(1,0),(2,0),(3,4)],
               [(0,0),(0,1),(1,0),(1,1)],[]]
        if duplicate:edges[0].append(edges[0][0])
        self.relation2adjacency=[]
        for relation in edges:
            index=torch.tensor([[dst for src,dst in relation],[src for src,dst in relation]],dtype=torch.int64)
            self.relation2adjacency.append([index,torch.ones(len(relation))])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('requires allocated CPU worker')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.perf_counter();graph=Graph()
    bodies=[[],[0],[0,0],[0,0,0,0,0],[1,0,1],[1],[2],[0,2],[0,0],[2,0],[1]*33]
    h=torch.tensor([0,3,1,8,0]);removed=torch.tensor([0,4,5,1,1])
    checks=[];bounds=[]
    for threads in [1,4,16]:
        engines={'stdmap':OfficialNativeBackend(graph,threads=threads,bits=64),
                 'spa':SPANativeBackend(graph,threads=threads)}
        try:
            prepared={name:engine.prepare_rules(bodies) for name,engine in engines.items()}
            for relation,head,edges in [(0,h,None),(0,h,removed),(0,h.flip(0),removed.flip(0)),
                                        (1,torch.tensor([0,1,8]),torch.tensor([0,1,2]))]:
                outputs={name:engine.batch(prepared[name],head,relation,edges) for name,engine in engines.items()}
                for rule,body in enumerate(bodies):
                    expected=graph.grounding(head,relation,body,edges)
                    for name in engines:
                        index,value=outputs[name][rule]
                        actual=torch.zeros(expected.numel(),dtype=torch.int64);actual[index]=value
                        assert torch.equal(actual.reshape_as(expected),expected),(threads,name,relation,body)
                        assert index.numel()<2 or bool((index[1:]>index[:-1]).all())
                        assert bool((value>0).all())
                    assert torch.equal(outputs['stdmap'][rule][0],outputs['spa'][rule][0])
                    assert torch.equal(outputs['stdmap'][rule][1],outputs['spa'][rule][1])
                checks.append({'threads':threads,'query_relation':relation,'removed':edges is not None,'batch':len(head),'exact_rules':len(bodies)})
            for name,engine in engines.items():
                assert engine.batch(engine.prepare_rules([]),h,0,None)==[]
                try:engine.prepare_rules([[1]*64])
                except OverflowError:bounds.append(name+':'+str(threads))
                else:raise AssertionError('overflow guard missing')
        finally:
            for engine in engines.values():engine.close()
    assert int(graph.grounding(torch.tensor([0]),0,[0,0],None)[0,3])==2
    large=graph.grounding(torch.tensor([0]),0,[1]*33,None)
    assert int(large[0,0])==2**32 and int(large[0,1])==2**32
    before=graph.grounding(torch.tensor([0,3]),0,[0]*5,None)
    after=graph.grounding(torch.tensor([0,3]),0,[0]*5,torch.tensor([0,0]))
    assert bool((before!=after).any()),'deletion case did not change any count'
    duplicate_rejections=[]
    for cls in [OfficialNativeBackend,SPANativeBackend]:
        try:cls(Graph(duplicate=True),threads=1,bits=64)
        except ValueError:duplicate_rejections.append(cls.__name__)
        else:raise AssertionError('duplicate triples accepted')
    result={'passed':True,'complete':True,'checks':checks,'overflow_rejections':bounds,
            'duplicate_triples_rejected':duplicate_rejections,'path_multiplicity':2,'large_exact_path_count':2**32,
            'empty_rule_identity':True,'every_occurrence_removed_edge':True,'empty_frontiers_and_reused_buffers':True,
            'seconds':time.perf_counter()-start}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({'passed':True,'exact_cases':len(checks),'seconds':result['seconds']}))

if __name__=='__main__':main()
