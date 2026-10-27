"""Tiny exact-semantics PNA checks; CPU/GPU, all gradients, ties and Adam."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
STAGE2 = ROOT.parent / '2026-10-14-rnnlogic-sparse-stage2'
sys.path.insert(0, str(STAGE2))
import torch
from test_streaming_predictor import TinyGraph, SparseInterface, evaluate, compare
from predictors import PredictorPlus
from layers import FuncToNode
from pna_predictor import PNAPredictor, CachedPNAPredictor, pna_aggregate


def layer_evaluate(layer, counts, embeddings, groups, block):
    layer.zero_grad(set_to_none=True)
    x = embeddings.detach().clone().requires_grad_()
    out = layer(counts, x, groups) if block is None else pna_aggregate(layer, counts, x, groups, block)
    weights = torch.linspace(.01, .7, out.numel(), device=out.device).reshape_as(out)
    loss = (out * weights).sum() + .03 * out.square().sum()
    loss.backward()
    grads = {'input': x.grad.clone(), **{n: p.grad.clone() for n, p in layer.named_parameters()}}
    return out.detach(), grads


def layer_checks(device):
    torch.manual_seed(20)
    layer = FuncToNode(8).to(device)
    counts = torch.randint(0, 11, (9, 197), device=device).float()
    counts[0] = torch.arange(197, device=device) % 5 + 1
    groups = torch.tensor([0] * 3 + [2] * 65 + [4] * 129, device=device)
    embeddings = torch.randn(9, 8, device=device)
    embeddings[1] = embeddings[0]
    embeddings[3, ::2] = embeddings[2, ::2]
    expected, expected_grad = layer_evaluate(layer, counts, embeddings, groups, None)
    rows = []
    for block in [0, 1, 7, 128, 512]:
        out, gradients = layer_evaluate(layer, counts, embeddings, groups, block)
        torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-6)
        for name, ref in expected_grad.items():
            torch.testing.assert_close(gradients[name], ref, rtol=3e-4, atol=2e-5, msg=name)
        rows.append({'block': block, 'max_output_error': float((out - expected).abs().max()),
                     'max_gradient_error': max(float((gradients[n] - ref).abs().max()) for n, ref in expected_grad.items())})
    # Prove this input detects the wrong per-tile scale normalization.
    naive = torch.cat([layer(counts[:, s:s+7], embeddings, groups[s:s+7]) for s in range(0, 197, 7)])
    assert float((naive - expected).abs().max()) > 1e-3
    # Isolate min/max channels: count magnitudes must not scale extrema, and
    # ties route all gradient to the first supported local rule, never split.
    ties = []
    for feature, name in [(1, 'min'), (2, 'max')]:
        tied_layer = FuncToNode(4).to(device)
        tied_layer.layer_norm = torch.nn.Identity()
        with torch.no_grad():
            linear = tied_layer.add_model.layers[0]
            linear.weight.zero_(); linear.bias.zero_()
            for dim in range(4):
                linear.weight[dim, (feature * 4 + dim) * 3] = 1
        tied = torch.ones(4, 4, device=device, requires_grad=True)
        c = torch.tensor([[9, 0, 0], [2, 4, 0], [0, 8, 6], [3, 1, 5]], device=device).float()
        g = torch.tensor([0, 0, 2], device=device)
        expected_tie = torch.tensor([[1]*4, [1]*4, [1]*4, [0]*4], dtype=torch.float32, device=device)
        for block in [0, 1, 2, 128]:
            tied.grad = None
            result = pna_aggregate(tied_layer, c, tied, g, block)
            torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)
            result.sum().backward()
            torch.testing.assert_close(tied.grad, expected_tie, rtol=0, atol=0)
        ties.append(name)
    return {'cases': rows, 'first_supported_tie_gradients_exact': ties,
            'incorrect_per_tile_scale_detected': True, 'missing_query_ids_tested': True}


def model_checks(device):
    graph = TinyGraph()
    # Duplicate bodies have distinct global ids; an unrelated head interleaves
    # them. Empty-body identity and absent terminals both appear in this plan.
    rules = [[0], [0,0], [1,1], [0,0], [0,0,1], [0,1], [0,3], [0,2,0], [0,1,0]]
    preload_groups = [(torch.tensor([0,1,2,256]), torch.tensor([0,1,2,3])),
                      (torch.tensor([3,4,0,0]), torch.tensor([4,5,0,8]))]
    queries = [([4,1,0,4,256,2], [5,1,0,5,3,2]), ([256,3,0,0], [3,4,8,0])]
    rows = []
    for kind in ['emb', 'lstm']:
        for feature in ['bias', 'none']:
            torch.manual_seed(331)
            model = PredictorPlus(graph, type=kind, num_layers=2, hidden_dim=8, entity_feature=feature, aggregator='pna')
            model.set_rules(rules); model.to(device)
            if kind == 'emb':
                with torch.no_grad():
                    model.rule_emb[3].copy_(model.rule_emb[1])
                    model.rule_emb[5, ::2].copy_(model.rule_emb[4, ::2])
            for split in ['train', 'valid']:
                for block in [0, 1, 7, 128]:
                    fresh = PNAPredictor(model, SparseInterface, candidate_block_size=block, collect_stats=True)
                    cached = CachedPNAPredictor(model, SparseInterface, namespace='tiny-v1', rule_version='rules-v1',
                                                candidate_block_size=block, collect_stats=True)
                    for hh, ee in preload_groups:
                        cached.preload(hh, torch.zeros_like(hh), ee if split == 'train' else None)
                    for query, (heads, removed) in enumerate(queries):
                        h = torch.tensor(heads, device=device); r = torch.zeros_like(h)
                        edge = torch.tensor(removed, device=device) if split == 'train' else None
                        expected = evaluate(model, model, h, r, edge)
                        for name, adapter in [('fresh', fresh), ('cache', cached)]:
                            error = compare(evaluate(model, adapter, h, r, edge), expected)
                            rows.append({'kind':kind,'feature':feature,'split':split,'block':block,'query_group':query,
                                         'consumer':name,'max_logit_error':error[0],'max_gradient_error':error[1]})
    # No identity rule: query 256 has no candidates while others do. This also
    # verifies scale normalization when the first and last query groups vanish.
    model = PredictorPlus(graph, type='emb', hidden_dim=8, entity_feature='bias', aggregator='pna')
    model.set_rules([[0,0],[0,0],[0,1],[0,3]]); model.to(device)
    h = torch.tensor([256,0,4,256], device=device); r = torch.zeros_like(h)
    fresh = PNAPredictor(model, SparseInterface, candidate_block_size=1)
    cached = CachedPNAPredictor(model, SparseInterface, namespace='empty-queries', rule_version='v1', candidate_block_size=1)
    cached.preload(h.cpu(), r.cpu(), None)
    for adapter in [fresh, cached]:
        compare(evaluate(model, adapter, h, r, None), evaluate(model, model, h, r, None))
    empty_checks = 0
    for feature in ['bias', 'none']:
        model = PredictorPlus(graph, type='emb', hidden_dim=8, entity_feature=feature, aggregator='pna')
        model.set_rules([[0,3],[1,0]]); model.to(device)
        for relation in [0, 2]:
            r = torch.full_like(h, relation)
            fresh = PNAPredictor(model, SparseInterface, candidate_block_size=1)
            cached = CachedPNAPredictor(model, SparseInterface, namespace='empty', rule_version='v1', candidate_block_size=1)
            cached.preload(h.cpu(), r.cpu(), None)
            for adapter in [fresh, cached]:
                with torch.no_grad():
                    got = adapter(h, r, None); ref = model(h, r, None)
                assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
                if feature == 'bias':
                    compare(evaluate(model, adapter, h, r, None), evaluate(model, model, h, r, None))
                empty_checks += 1
    # Original graph counts explicitly distinguish the two removed duplicate
    # edges; either single removed edge leaves one 0->1 path, not zero.
    h_cpu = torch.tensor([0,0]); removal = torch.tensor([0,8])
    assert graph.grounding(h_cpu,0,[0],None)[:,1].tolist() == [2,2]
    assert graph.grounding(h_cpu,0,[0],removal)[:,1].tolist() == [1,1]
    return {'cases': rows, 'empty_result_checks': empty_checks,
            'mixed_candidate_empty_queries': True, 'individual_parallel_edge_removal': True}


def adam_checks(device):
    torch.manual_seed(772)
    graph = TinyGraph()
    original = PredictorPlus(graph, type='emb', hidden_dim=8, entity_feature='bias', aggregator='pna')
    original.set_rules([[0], [0,0], [0,0], [0,1], [0,0,1], [0,3]])
    original.to(device)
    model = copy.deepcopy(original)
    adapter = CachedPNAPredictor(model, SparseInterface, namespace='adam', rule_version='v1', candidate_block_size=2)
    heads = torch.tensor([0,1,2,3,4,256]); edges = torch.tensor([0,1,2,3,4,5])
    adapter.preload(heads, torch.zeros_like(heads), edges)
    reference_optimizer = torch.optim.Adam(original.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rows = []
    for step, selected in enumerate([[4,1,0,5], [3,0,2], [5,0,4,1,4]]):
        h = heads[selected].to(device); r = torch.zeros_like(h); removed = edges[selected].to(device)
        target = torch.nn.functional.one_hot((h + 1) % graph.entity_size, graph.entity_size).float()
        results = []
        for current, opt in [(original, reference_optimizer), (adapter, optimizer)]:
            opt.zero_grad(set_to_none=True)
            score, mask = current(h, r, removed)
            loss = -((torch.softmax(score, 1) + 1e-8).log()[mask] * target[mask]).sum() / target[mask].sum().clamp(min=1)
            loss.backward()
            results.append((score.detach(), loss.detach()))
            opt.step()
        torch.testing.assert_close(results[0][0], results[1][0], rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(results[0][1], results[1][1], rtol=2e-5, atol=2e-6)
        parameter_error = moment_error = 0.
        for (name, expected), (other_name, actual) in zip(original.named_parameters(), model.named_parameters()):
            assert name == other_name
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5, msg=name)
            parameter_error = max(parameter_error, float((actual - expected).abs().max()))
            want_state = reference_optimizer.state.get(expected, {})
            got_state = optimizer.state.get(actual, {})
            assert want_state.keys() == got_state.keys()
            for state_key in want_state:
                torch.testing.assert_close(got_state[state_key], want_state[state_key], rtol=5e-4, atol=2e-6, msg=name + '/' + state_key)
                moment_error = max(moment_error, float((got_state[state_key] - want_state[state_key]).abs().max()))
        rows.append({'step':step + 1, 'max_parameter_error':parameter_error, 'max_optimizer_state_error':moment_error,
                     'loss_error':float((results[0][1]-results[1][1]).abs())})
    return rows


def cache_gates(device):
    model = PredictorPlus(TinyGraph(), type='emb', hidden_dim=8, aggregator='pna')
    model.set_rules([[0,0],[0,0],[0,1]]); model.to(device)
    adapter = CachedPNAPredictor(model, SparseInterface, namespace='gates', rule_version='v1')
    h = torch.tensor([0,0]); r = torch.zeros_like(h); removed = torch.tensor([0,8])
    adapter.preload(h,r,removed)
    assert len(adapter.cache) == 2
    tests = []
    def rejects(name, error, function):
        try:
            function()
        except error:
            tests.append(name)
        else:
            raise AssertionError('missing rejection: ' + name)
    rejects('train_vs_valid_key', KeyError, lambda: adapter(h.to(device), r.to(device), None))
    rejects('rule_reordering', ValueError, lambda: PNAPredictor(model, SparseInterface, rule_order='prefix'))
    rejects('changed_namespace', ValueError, lambda: adapter.preload(h,r,removed,namespace='wrong'))
    # Distinct exact ints that round to identical float32 must still be detected.
    adapter.reset(namespace='gates',rule_version='v2')
    term = [(torch.tensor([1]),torch.tensor([2**24])), (torch.tensor([1]),torch.tensor([7])),
            (torch.empty(0,dtype=torch.int64),torch.empty(0,dtype=torch.int64))]
    adapter.preload_from_terminals(h[:1],r[:1],removed[:1],term)
    changed = [(i.clone(), v.clone()) for i,v in term]
    changed[0][1][0] += 1
    rejects('exact_integer_digest_before_float', ValueError,
            lambda: adapter.preload_from_terminals(h[:1],r[:1],removed[:1],changed))
    adapter.reset(namespace='gates',rule_version='v3')
    assert not adapter.cache and adapter.cache_statistics()['persistent_unique_storage_bytes'] == 0
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    start = time.perf_counter()
    layers = layer_checks(device)
    models = model_checks(device)
    adam = adam_checks(device)
    gates = cache_gates(device)
    result = {'passed':True, 'device':str(device), 'torch':torch.__version__, 'layer_checks':layers,
              'model_checks':models, 'adam_steps':adam, 'cache_validation_gates':gates,
              'seconds':time.perf_counter()-start,
              'scope':'tiny CPU/GPU exact-semantics tests against original dense RNNLogic; performance not representative'}
    if args.output:
        Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True, 'device':str(device), 'predictor_cases':len(models['cases']),
                      'max_logit_error':max(x['max_logit_error'] for x in models['cases']),
                      'max_gradient_error':max(x['max_gradient_error'] for x in models['cases']),
                      'adam_steps':len(adam), 'seconds':result['seconds']}))


if __name__ == '__main__':
    main()
