"""Small semantic/gradient checks, no claim of a representative runtime benchmark."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / '2026-10-14-candidate-tests/kg'
sys.path[:0] = [str(OLD/'deps'), str(OLD/'upstream/RNNLogic/src')]
import torch
from data import KnowledgeGraph
from predictors import PredictorPlus
from streaming_predictor import StreamingPredictor, PrefixGrounder


class TinyGraph:
    # Use the original propagation and original rule execution as the oracle.
    propagate = KnowledgeGraph.propagate
    grounding = KnowledgeGraph.grounding

    def __init__(self):
        self.entity_size = 257
        self.relation_size = 4
        src_dst = [
            [(0,1),(1,2),(2,3),(3,4),(0,2),(1,3),(2,4),(4,0),(0,1)],
            [(0,2),(1,3),(2,4),(3,5),(4,0),(5,1),(2,0)],
            [(0,0),(1,1),(2,2),(0,1),(1,0)],
            [],
        ]
        self.relation2adjacency = []
        for edges in src_dst:
            adj = torch.tensor([[dst for src,dst in edges], [src for src,dst in edges]], dtype=torch.int64)
            self.relation2adjacency.append([adj, torch.ones(len(edges))])


class SparseInterface:
    """Dense oracle converted to sparse only to test the consumer interface."""
    def __init__(self, graph, bodies):
        self.inner = PrefixGrounder(graph, bodies, policy='last_use', collect_stats=False)
    def begin(self, *args):
        return self.inner.begin(*args)
    def grounding(self, *args):
        count = self.inner.grounding(*args)
        nz = torch.nonzero(count, as_tuple=True)
        return nz[0] * count.shape[1] + nz[1], count[nz]
    def finish(self, **kwargs):
        return self.inner.finish(**kwargs)


def evaluate(model, runner, h, r, removed):
    model.zero_grad(set_to_none=True)
    out, mask = runner(h, r, removed)
    weight = torch.linspace(.01, .1, out.numel(), device=out.device).view_as(out)
    loss = ((out * weight)[mask].sum() + .03 * out[mask].square().sum()) / max(1, int(mask.sum()))
    loss.backward()
    grads = {k: None if p.grad is None else p.grad.detach().clone() for k,p in model.named_parameters()}
    return out.detach(), mask.detach(), loss.detach(), grads


def compare(actual, expected):
    out, mask, loss, grads = actual
    ref, refmask, refloss, refgrads = expected
    assert torch.equal(mask, refmask)
    torch.testing.assert_close(out, ref, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(loss, refloss, rtol=2e-4, atol=2e-5)
    errors=[]
    for name, want in refgrads.items():
        got=grads[name]
        assert (got is None) == (want is None), name
        if want is not None:
            torch.testing.assert_close(got, want, rtol=5e-4, atol=2e-5, msg=name)
            errors.append(float((got-want).abs().max()))
    finite=torch.isfinite(ref)
    return float((out[finite]-ref[finite]).abs().max()), max(errors, default=0.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output')
    args=parser.parse_args()
    device=torch.device(args.device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    graph=TinyGraph()
    patterns=[[], [0], [1], [3], [0,0], [0,1], [1,0], [2,0], [0,1,0], [2,2,0], [3,0], [1,2]]
    rules=[[0]+patterns[(i*7)%len(patterns)] for i in range(137)] + [[1,1]]
    h=torch.tensor([0,1,2,256], dtype=torch.int64, device=device)
    r=torch.zeros_like(h)
    removed=torch.tensor([0,1,2,3], dtype=torch.int64, device=device)
    original_counts=graph.grounding(h,0,[0,0],None)
    deleted_counts=graph.grounding(h,0,[0,0],removed)
    assert (original_counts != deleted_counts).any()
    # Parallel 0->1 edges create true integer path multiplicity (not reachability).
    assert int(graph.grounding(h,0,[0],None)[0,1]) == 2
    results=[]
    started=time.perf_counter()
    for kind in ('emb','lstm'):
        for feature in ('bias','none'):
            torch.manual_seed(1943)
            model=PredictorPlus(graph,type=kind,num_layers=2,hidden_dim=8,entity_feature=feature,aggregator='sum')
            model.set_rules(rules)
            model.to(device)
            with torch.no_grad():
                if feature=='bias': model.bias.uniform_(-.1,.1)
            for split, removed_arg in [('train',removed), ('valid',None)]:
                expected=evaluate(model,model,h,r,removed_arg)
                for order in ('original','prefix'):
                    for block in (8,32,128):
                        for terminal_format in ('dense','sparse'):
                            adapter=StreamingPredictor(model,block_size=block,rule_order=order,
                                grounder_factory=SparseInterface if terminal_format=='sparse' else None,
                                terminal_format=terminal_format,collect_stats=True)
                            errors=compare(evaluate(model,adapter,h,r,removed_arg),expected)
                            results.append(dict(kind=kind,feature=feature,split=split,order=order,block=block,
                                                terminal_format=terminal_format,max_logit_error=errors[0],
                                                max_gradient_error=errors[1],stats=adapter.last_stats))
    # No-rules and all-zero terminal cases preserve the original empty behavior.
    empty_checks=0
    for feature in ('bias','none'):
        model=PredictorPlus(graph,type='emb',hidden_dim=8,entity_feature=feature,aggregator='sum')
        model.set_rules([[0,3],[0,3,0]])
        model.to(device)
        for relation in (0,2):
            rr=torch.full_like(h,relation)
            adapter=StreamingPredictor(model)
            with torch.no_grad():
                want=model(h,rr,None); got=adapter(h,rr,None)
            assert torch.equal(got[0],want[0]) and torch.equal(got[1],want[1])
            empty_checks+=1
    # Reject PNA rather than reinterpreting its moments/min/max as a sum.
    pna=PredictorPlus(graph,type='emb',hidden_dim=8,aggregator='pna')
    pna.set_rules([[0,0]])
    try:
        StreamingPredictor(pna)
        raise AssertionError('PNA incorrectly accepted')
    except ValueError as error:
        assert 'PNA' in str(error)
    # saved_tensors_hooks inspect actual autograd state, rather than merely
    # asserting that Python's local count variables were released.
    model=PredictorPlus(graph,type='emb',hidden_dim=8,entity_feature='bias',aggregator='sum')
    model.set_rules(rules); model.to(device)
    adapter=StreamingPredictor(model,block_size=32,collect_stats=True)
    saved=[]
    def pack(tensor):
        saved.append(dict(shape=list(tensor.shape),dtype=str(tensor.dtype),bytes=tensor.numel()*tensor.element_size()))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor:tensor):
        out,mask=adapter(h,r,removed)
        out.square().mean().backward()
    full_count_bytes=137*h.numel()*graph.entity_size*4
    compact_bytes=adapter.last_stats['total_compact_float_bytes']
    assert compact_bytes < full_count_bytes/20
    assert not any(x['shape'] in [[137, h.numel()*graph.entity_size], [h.numel()*graph.entity_size,137]] for x in saved)
    assert not any(x['dtype']=='torch.int64' and len(x['shape'])==2 and set(x['shape'])=={h.numel(), graph.entity_size} for x in saved)
    result={'device':str(device),'cases':results,'empty_cases_passed':empty_checks,
        'pna_rejected':True,'masked_path_count_changes_verified':True,'parallel_edge_integer_multiplicity_verified':True,
        'autograd_saved_tensor_check':{'dense_full_count_bytes':full_count_bytes,'compact_saved_count_bytes':compact_bytes,
                                     'saved_tensor_metadata':saved,'passed':True},
        'seconds':time.perf_counter()-started,'passed':True,
        'scope':'tiny graph semantic tests; sparse terminal wrapper only tests interface, NOT sparse engine or representative performance'}
    if args.output: Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'forward_gradient_cases':len(results),'empty_cases':empty_checks,
        'max_logit_error':max(x['max_logit_error'] for x in results),
        'max_gradient_error':max(x['max_gradient_error'] for x in results),
        'dense_full_count_bytes':full_count_bytes,'compact_saved_count_bytes':compact_bytes,
        'seconds':result['seconds']}))


if __name__=='__main__':main()
