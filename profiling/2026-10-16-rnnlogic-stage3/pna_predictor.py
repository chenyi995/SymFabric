"""Local exact-semantics PNA consumers; upstream RNNLogic is never patched.

PNAPredictor consumes fresh sparse int64 terminals from a stage-2 controller.
CachedPNAPredictor reuses the stage-2 explicit per-query CSR count cache, with
its exact-int64 preload validation and keys including each removed edge.
Both preserve original rule order and distinct global rule ids, including
duplicate rule bodies. Counts convert to float32 at the upstream boundary.

Each forward builds compact [rules, active candidates] counts. Candidate tiling
bounds the largest [candidates, rules, hidden] PNA temporary, but does NOT bound
the compact counts or all saved autograd state. Degree-scale normalization is
computed once across ALL candidates of each query before tiling. First-index
min/max gradients use the unchanged torch.min/max rule-axis reduction. Neural
weights, encoding, layer normalization and scoring are the original modules.

candidate_block_size=0 invokes the original FuncToNode directly, as an oracle
and a simple baseline. Positive sizes tile candidates only; rules never reorder.
No stochastic/training changes, count approximation or grounding fallback exist.
"""
from __future__ import annotations

import operator
from pathlib import Path
import sys

import torch

_STAGE2 = Path(__file__).resolve().parent.parent / '2026-10-14-rnnlogic-sparse-stage2'
if str(_STAGE2) not in sys.path:
    sys.path.insert(0, str(_STAGE2))
from cached_sparse_faststats_predictor import CachedSparseFastStatsPredictor, _IncrementalCountCache


def pna_aggregate(layer, counts, embeddings, batch_ids, candidate_block_size=128):
    """Original FuncToNode arithmetic, with global query scales and row tiling.

    counts is float32 [R,C], embeddings [R,H], and batch_ids [C]. Each column
    must have positive support, as in upstream PredictorPlus's candidate_set.
    An empty candidate set is handled by the predictor's original empty branch.
    """
    block = operator.index(candidate_block_size)
    if block < 0:
        raise ValueError('candidate_block_size must be nonnegative')
    if block == 0:
        return layer(counts, embeddings, batch_ids)
    if not batch_ids.numel():
        raise ValueError('empty candidates must use PredictorPlus empty-result branch')
    degree = counts.sum(0) + 1
    degree_out = degree.unsqueeze(-1)
    scale = degree_out.log()
    batch_size = batch_ids.max().item() + 1
    sum_scale = torch.zeros(batch_size, device=embeddings.device)
    cn_scale = torch.zeros(batch_size, device=embeddings.device)
    ones = torch.ones(scale.size(0), device=embeddings.device)
    sum_scale.scatter_add_(0, batch_ids, scale.squeeze(-1))
    cn_scale.scatter_add_(0, batch_ids, ones)
    mean_scale = sum_scale / cn_scale.clamp(min=layer.eps)
    scale = scale / mean_scale[batch_ids].unsqueeze(-1).clamp(min=layer.eps)
    scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=layer.eps)], dim=-1)
    output = []
    message = embeddings.unsqueeze(0)
    for start in range(0, counts.size(1), block):
        stop = min(start + block, counts.size(1))
        weight = counts[:, start:stop].transpose(0, 1).unsqueeze(-1)
        weight_zero = weight == 0
        total = (message * weight).sum(1)
        sq_total = ((message ** 2) * weight).sum(1)
        # torch.amin/amax and scatter-reduce have different tie gradients.
        # Retaining min/max(dim=1) also preserves distinct duplicate-rule ids.
        minimum = message.expand(weight.size(0), -1, -1).masked_fill(weight_zero, float('inf')).min(1)[0]
        maximum = message.expand(weight.size(0), -1, -1).masked_fill(weight_zero, float('-inf')).max(1)[0]
        divisor = degree_out[start:stop].clamp(min=layer.eps)
        mean = total / divisor
        sq_mean = sq_total / divisor
        std = (sq_mean - mean ** 2).clamp(min=layer.eps).sqrt()
        features = torch.cat([mean, minimum, maximum, std], dim=-1)
        update = (features.unsqueeze(-1) * scales[start:stop].unsqueeze(-2)).flatten(-2)
        output.append(torch.relu(layer.layer_norm(layer.add_model(update))))
    return torch.cat(output, dim=0)


def _validate_model(model, candidate_block_size, rule_order):
    if model.aggregator != 'pna':
        raise ValueError('this adapter requires the original PNA aggregator')
    if rule_order != 'original':
        raise ValueError('PNA requires original rule order to preserve first-index tie gradients')
    block = operator.index(candidate_block_size)
    if block < 0:
        raise ValueError('candidate_block_size must be nonnegative')
    return block


def _queries(model, h, r, removed):
    device = next(model.parameters()).device
    if h.device != device or r.device != device or h.dtype != torch.int64 or r.dtype != torch.int64:
        raise ValueError('queries must be int64 on the model device')
    if h.ndim != 1 or r.shape != h.shape or not h.numel():
        raise ValueError('queries must be nonempty equally shaped 1D tensors')
    if removed is not None and (removed.device != device or removed.dtype != torch.int64 or removed.shape != h.shape):
        raise ValueError('removed edges must match query shape/device and int64 dtype')
    relation = int(r[0].item())
    if bool((r != relation).any()):
        raise ValueError('a PredictorPlus batch must contain a single relation')
    return relation, device


def _predict(model, h, r, candidates, counts, rule_ids, block):
    batch, entities = h.numel(), model.graph.entity_size
    device = h.device
    if not candidates.numel():
        mask = torch.zeros((batch, entities), dtype=torch.float32, device=device)
        if model.entity_feature == 'bias':
            return mask + model.bias.unsqueeze(0), (1 - mask).bool()
        if model.entity_feature == 'RotatE':
            return mask + model.RotatE(h, r), (1 - mask).bool()
        return mask - float('-inf'), mask.bool()
    if model.rule_features.device != device:
        model.rule_features = model.rule_features.to(device)
    embeddings = model.rule_emb[rule_ids] if model.type == 'emb' else model.encode_rules(model.rule_features[rule_ids])
    output = pna_aggregate(model.rule_to_entity, counts, embeddings, candidates // entities, block)
    rel = model.relation_emb(r[0]).unsqueeze(0).expand(output.size(0), -1)
    output = model.score_model(torch.cat([output, rel], dim=-1)).squeeze(-1)
    score = torch.zeros(batch * entities, device=device)
    score.scatter_(0, candidates, output)
    score = score.view(batch, entities)
    if model.entity_feature == 'bias':
        score = score + model.bias.unsqueeze(0)
        mask = torch.ones_like(score, dtype=torch.bool)
    elif model.entity_feature == 'RotatE':
        score = score + model.RotatE(h, r)
        mask = torch.ones_like(score, dtype=torch.bool)
    else:
        mask = torch.zeros(batch * entities, dtype=torch.bool, device=device)
        mask.index_fill_(0, candidates, True)
        mask = mask.view(batch, entities)
        score = score.masked_fill(~mask, float('-inf'))
    return score, mask


def _compact(terminals, device):
    """Input contract: one unique positive int64 count per flat index per rule.

    Grounders must aggregate repeated paths in integer arithmetic. Shape/type
    checks run here; canonical positive/unique checks run in cache preload.
    Fresh native/GPU grounder exact-count tests establish that same contract.
    """
    if not terminals:
        return (torch.empty(0, dtype=torch.int64, device=device),
                torch.empty((0, 0), dtype=torch.float32, device=device))
    source = terminals[0][0].device
    if source != device and source.type != 'cpu':
        raise ValueError('terminal source must be CPU or model device')
    for index, value in terminals:
        if index.dtype != torch.int64 or value.dtype != torch.int64:
            raise ValueError('terminal indices and values must be exact int64')
        if index.ndim != 1 or value.shape != index.shape or index.device != source or value.device != source:
            raise ValueError('terminals must be matching 1D tensors on one source device')
    flat = torch.cat([index for index, _ in terminals])
    candidates, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    rule = torch.cat([torch.full_like(index, slot) for slot, (index, _) in enumerate(terminals)])
    values = torch.cat([value for _, value in terminals]).float()
    counts = torch.zeros((len(terminals), candidates.numel()), dtype=torch.float32, device=source)
    counts.index_put_((rule, inverse), values)
    return candidates.to(device), counts.to(device)


class PNAPredictor(torch.nn.Module):
    """Fresh-count PNA with stage-2 sparse controller begin/grounding/finish API."""
    def __init__(self, model, grounder_factory, *, candidate_block_size=128,
                 rule_order='original', collect_stats=False):
        super().__init__()
        self.candidate_block_size = _validate_model(model, candidate_block_size, rule_order)
        self.model = model
        self.grounder_factory = grounder_factory
        self.collect_stats = bool(collect_stats)
        self.plans = {r: tuple((int(i), (int(head), tuple(map(int, body))))
                              for i, (head, body) in entries)
                      for r, entries in enumerate(model.relation2rules)}
        self.controllers = {}
        self._rule_indices = {}
        self.last_stats = None

    def rules_for_relation(self, relation):
        return self.plans[int(relation)]

    def forward(self, h, r, edges_to_remove=None):
        relation, device = _queries(self.model, h, r, edges_to_remove)
        plan = self.plans[relation]
        if relation not in self.controllers:
            self.controllers[relation] = self.grounder_factory(self.model.graph, [body for _, (_, body) in plan])
        controller = self.controllers[relation]
        controller.begin(h, relation, edges_to_remove)
        complete = False
        try:
            terminals = [controller.grounding(h, head, body, edges_to_remove) for _, (head, body) in plan]
            complete = True
        finally:
            grounder_stats = controller.finish(require_complete=complete)
        candidates, counts = _compact(terminals, device)
        key = (relation, device)
        if key not in self._rule_indices:
            self._rule_indices[key] = torch.tensor([i for i, _ in plan], dtype=torch.int64, device=device)
        self.last_stats = None
        if self.collect_stats:
            self.last_stats = {'relation': relation, 'rules': len(plan), 'candidates': candidates.numel(),
                'compact_count_bytes': counts.numel() * counts.element_size(),
                'terminal_int64_payload_bytes': sum((i.numel() + v.numel()) * 8 for i, v in terminals),
                'candidate_block_size': self.candidate_block_size, 'global_per_query_scale': True,
                'original_rule_order': True, 'cross_forward_count_cache': False, 'grounder': grounder_stats,
                'dense_full_entity_feature_pool_created': False}
        del terminals
        return _predict(self.model, h, r, candidates, counts, self._rule_indices[key], self.candidate_block_size)


class CachedPNAPredictor(CachedSparseFastStatsPredictor):
    """Stage-2 validated CSR preload/reset interface with a PNA-only forward.

    Cache inheritance deliberately reuses only count handling and metadata.
    Its sum-only constructor and neural forward are never called or modified.
    Persistent storage remains sparse; each forward densifies compact [R,C].
    """
    def __init__(self, model, grounder_factory=None, *, namespace, rule_version,
                 candidate_block_size=128, rule_order='original', collect_stats=False):
        torch.nn.Module.__init__(self)
        self.candidate_block_size = _validate_model(model, candidate_block_size, rule_order)
        self.model = model
        self.grounder_factory = grounder_factory
        self.rule_order = rule_order
        self.collect_stats = bool(collect_stats)
        self.last_stats = None
        self.cache = _IncrementalCountCache()
        self.reset(namespace=namespace, rule_version=rule_version)

    def forward(self, h, r, edges_to_remove=None):
        self._check_version(None, None)
        relation, device = _queries(self.model, h, r, edges_to_remove)
        heads = h.detach().cpu().tolist()
        removed = [None] * len(heads) if edges_to_remove is None else edges_to_remove.detach().cpu().tolist()
        entries = []
        for head, edge in zip(heads, removed):
            key = self._key(relation, head, edge)
            if key not in self.cache:
                raise KeyError(f'explicit preloaded-set miss: {key}; no grounding fallback')
            entries.append(self.cache[key])
        plan = self.plans[relation]
        candidates = torch.cat([entry.candidates + query * self.model.graph.entity_size
                                for query, entry in enumerate(entries)])
        crow_parts = [torch.zeros(1, dtype=torch.int64, device=device)]
        offset = 0
        for entry in entries:
            crow_parts.append(entry.crow[1:] + offset)
            offset += entry.nnz
        crow = torch.cat(crow_parts)
        columns = torch.cat([entry.columns for entry in entries])
        values = torch.cat([entry.values for entry in entries])
        csr = torch.sparse_csr_tensor(crow, columns, values, (candidates.numel(), len(plan)), device=device)
        counts = csr.to_dense().transpose(0, 1).contiguous()
        if relation not in self._rule_indices:
            self._rule_indices[relation] = torch.tensor([i for i, _ in plan], dtype=torch.int64, device=device)
        self.last_stats = None
        if self.collect_stats:
            self.last_stats = {'relation': relation, 'rules': len(plan), 'candidates': candidates.numel(),
                'compact_count_bytes': counts.numel() * counts.element_size(),
                'candidate_block_size': self.candidate_block_size, 'global_per_query_scale': True,
                'original_rule_order': True, 'cross_forward_count_cache': True,
                'query_d2h_key_lookup_included': True, 'dense_full_entity_feature_pool_created': False,
                **self.cache_statistics()}
        return _predict(self.model, h, r, candidates, counts, self._rule_indices[relation], self.candidate_block_size)
