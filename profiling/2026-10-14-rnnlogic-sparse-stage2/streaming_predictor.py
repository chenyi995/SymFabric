"""Local blockwise consumer of *unchanged* RNNLogic PredictorPlus sum weights.

The original source is never patched. This adapter retains dense integer graph
states only within a block, then saves compact float counts for autograd. It does
not retain the full [rule, batch, entity] float count tensor. A sparse terminal
interface also avoids dense terminal scans and integer block storage.

Usage::
    adapter = StreamingPredictor(model, block_size=32, rule_order='prefix')
    logits, mask = adapter(h, r, removed)

The default grounder is stage 1 PrefixGrounder(policy='last_use'). The caller
chooses graph.propagate (upstream/native/Triton), exactly as in stage 1. An external
factory(graph, ordered_rule_bodies) can return another controller implementing
begin(h, relation, removed), grounding(h, relation, body, removed), finish(). The
controller must honor every-query edge exclusion and exact int64 path counts.
With terminal_format='dense', grounding returns int64 [batch, entity]. With
'sparse', it returns (flat_indices, int64_values), where flat_indices are unique,
nonzero entries in row-major [batch, entity] layout. Sparse terminals may be on
CPU for a CUDA model: the entire block is compacted on CPU and then transfers
only compact counts and candidate indices once per block. A CPU controller must
accept/convert the passed query tensors itself, including that cost in forward. Duplicate flat indices are
unsupported: combine them in integer arithmetic inside the grounder.

Semantics/limitations:
- sum only; PNA is explicitly rejected. Counts are nonnegative int64 path counts
  with overflow excluded by the caller, exactly as for upstream graph execution.
- Rules may be original or stable lexicographic body order; global rule ids stay
  unchanged. Rule permutations/reduction grouping can change floating-point
  rounding; logits and all parameter gradients need tolerance-based comparison.
- Default controllers retain no count cache across forward. Explicit external
  memoized controllers can retain counts and must report cold/hit costs separately.
  Graph/plan preparation happens at adapter construction.
- Encoding and scoring use the original trainable modules. The graph is discrete
  and has no upstream gradient; count conversion remains float32.
- The adapter still owns [batch*entity, hidden] float features and a dense output.
  It is not an entirely sparse PredictorPlus. Dense terminal mode can hold up to
  block_size dense int64 terminal states; nonzero() introduces CUDA synchronization.
- Timed use must set collect_stats=False. Diagnostic counters report logical
  tensors, not allocator peaks or retained prefix-cache aliases.
"""
from __future__ import annotations

import operator
from pathlib import Path
import sys

import torch

_STAGE1 = Path(__file__).resolve().parent.parent / '2026-10-14-rnnlogic-strong-baselines'
if str(_STAGE1) not in sys.path:
    sys.path.insert(0, str(_STAGE1))
from prefix_cache import PrefixGrounder


class StreamingPredictor(torch.nn.Module):
    def __init__(self, model, block_size=32, rule_order='prefix',
                 grounder_factory=None, collect_stats=False,
                 terminal_format='dense'):
        super().__init__()
        if model.aggregator != 'sum':
            raise ValueError('StreamingPredictor supports sum aggregation only; PNA is unimplemented')
        if rule_order not in ('original', 'prefix'):
            raise ValueError('rule_order must be original or prefix')
        if terminal_format not in ('dense', 'sparse'):
            raise ValueError('terminal_format must be dense or sparse')
        self.block_size = operator.index(block_size)
        if self.block_size <= 0:
            raise ValueError('block_size must be positive')
        self.model = model
        self.rule_order = rule_order
        self.terminal_format = terminal_format
        self.collect_stats = bool(collect_stats)
        self.last_stats = None
        self.controllers = {}
        self.plans = {}
        self._index_cache = {}
        if grounder_factory is None:
            if terminal_format != 'dense':
                raise ValueError('sparse terminal_format requires an explicit grounder_factory')
            def grounder_factory(graph, bodies):
                return PrefixGrounder(graph, bodies, policy='last_use', collect_stats=self.collect_stats)
        for relation, entries in enumerate(model.relation2rules):
            plan = [(int(index), (int(head), tuple(int(x) for x in body)))
                    for index, (head, body) in entries]
            if rule_order == 'prefix':
                plan.sort(key=lambda entry: entry[1][1])
            self.plans[relation] = tuple(plan)
            self.controllers[relation] = grounder_factory(model.graph, [body for _, (_, body) in plan])

    def _indices(self, relation, device):
        key = (relation, device.type, device.index)
        if key not in self._index_cache:
            self._index_cache[key] = torch.tensor(
                [index for index, _ in self.plans[relation]], dtype=torch.int64, device=device)
        return self._index_cache[key]

    def _dense_block(self, terminals, batch, entities, device):
        support = torch.zeros((batch, entities), dtype=torch.bool, device=device)
        for count in terminals:
            if count.dtype != torch.int64 or count.shape != (batch, entities) or count.device != device:
                raise ValueError('dense terminal must be int64 [batch, entity] on query device')
            support.logical_or_(count != 0)
        candidates = torch.nonzero(support.reshape(-1), as_tuple=True)[0]
        # Integer gathers precede float conversion and stacking. In particular,
        # do not flatten a non-contiguous dense [B,N] tensor: that would copy BN.
        rows, columns = candidates // entities, candidates % entities
        compact = torch.stack([count[rows, columns].float() for count in terminals], dim=0)
        return candidates, compact

    def _sparse_block(self, terminals, batch, entities, device):
        source_device = terminals[0][0].device
        if source_device != device and source_device.type != 'cpu':
            raise ValueError('sparse terminal source must be CPU or query device')
        for index, value in terminals:
            if index.dtype != torch.int64 or value.dtype != torch.int64:
                raise ValueError('sparse indices and values must be int64')
            if index.ndim != 1 or value.shape != index.shape or index.device != source_device or value.device != source_device:
                raise ValueError('sparse terminals must be equally shaped 1D tensors on one source device')
        flat = torch.cat([index for index, _ in terminals])
        candidates, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        rule = torch.cat([torch.full_like(index, slot) for slot, (index, _) in enumerate(terminals)])
        values = torch.cat([value for _, value in terminals]).float()
        compact = torch.zeros((len(terminals), candidates.numel()), dtype=torch.float32, device=source_device)
        compact.index_put_((rule, inverse), values)
        return candidates.to(device), compact.to(device)

    def forward(self, all_h, all_r, edges_to_remove=None):
        if all_h.dtype != torch.int64 or all_r.dtype != torch.int64:
            raise ValueError('queries must be int64')
        if all_h.ndim != 1 or all_r.shape != all_h.shape or all_h.device != all_r.device or not all_h.numel():
            raise ValueError('queries must be nonempty equally shaped 1D tensors on the same device')
        relation = int(all_r[0].item())
        if bool((all_r != relation).any()):
            raise ValueError('a PredictorPlus batch must contain a single query relation')
        model = self.model
        graph = model.graph
        batch, entities = all_h.numel(), graph.entity_size
        device = all_h.device
        plan = self.plans[relation]
        indices = self._indices(relation, device)
        if model.rule_features.device != device:
            model.rule_features = model.rule_features.to(device)
        # Encode once, with exactly the scheduled global ids. No feature or
        # parameter is synthesized, trained separately, or detached here.
        if len(plan):
            embeddings = model.rule_emb[indices] if model.type == 'emb' else model.encode_rules(model.rule_features[indices])
        else:
            embeddings = model.relation_emb.weight.new_empty((0, model.hidden_dim))
        features = embeddings.new_zeros((batch * entities, model.hidden_dim))
        support = torch.zeros(batch * entities, dtype=torch.bool, device=device)
        controller = self.controllers[relation]
        diagnostic = {'block_size': self.block_size, 'rule_order': self.rule_order,
                      'terminal_format': self.terminal_format, 'rules': len(plan),
                      'blocks': 0, 'peak_terminal_block_bytes_logical': 0,
                      'peak_compact_float_bytes': 0, 'total_compact_float_bytes': 0,
                      'dense_feature_bytes': features.numel() * features.element_size(),
                      'prefix_stats': None, 'no_cross_forward_count_cache': not bool(getattr(controller,'cross_forward_cache',getattr(getattr(controller,'controller',None),'cross_forward_cache',False)))}
        controller.begin(all_h, relation, edges_to_remove)
        complete = False
        try:
            for start in range(0, len(plan), self.block_size):
                stop = min(start + self.block_size, len(plan))
                terminals = []
                for _, (head, body) in plan[start:stop]:
                    if head != relation:
                        raise ValueError('rule head differs from query relation')
                    terminals.append(controller.grounding(all_h, head, body, edges_to_remove))
                if self.terminal_format == 'dense':
                    candidates, compact = self._dense_block(terminals, batch, entities, device)
                else:
                    candidates, compact = self._sparse_block(terminals, batch, entities, device)
                if self.collect_stats:
                    tensors = terminals if self.terminal_format == 'dense' else [t for pair in terminals for t in pair]
                    diagnostic['peak_terminal_block_bytes_logical'] = max(
                        diagnostic['peak_terminal_block_bytes_logical'], sum(t.numel() * t.element_size() for t in tensors))
                    compact_bytes = compact.numel() * compact.element_size()
                    diagnostic['peak_compact_float_bytes'] = max(diagnostic['peak_compact_float_bytes'], compact_bytes)
                    diagnostic['total_compact_float_bytes'] += compact_bytes
                    diagnostic['blocks'] += 1
                del terminals
                if candidates.numel():
                    # index_add_ does not need prior feature values for backward.
                    # Matmul saves only compact counts, never the dense terminals.
                    features.index_add_(0, candidates, compact.t() @ embeddings[start:stop])
                    support.index_fill_(0, candidates, True)
                del compact, candidates
            complete = True
        finally:
            # An aborted forward must release states without hiding its error.
            if complete:
                diagnostic['prefix_stats'] = controller.finish()
            else:
                try:
                    controller.finish(require_complete=False)
                except TypeError:
                    controller.finish()
        self.last_stats = diagnostic if self.collect_stats else None
        candidates = torch.nonzero(support, as_tuple=True)[0]
        if not candidates.numel():
            # Match upstream's empty-result branch, including its +inf behavior
            # when no entity feature exists. This is a compatibility baseline.
            mask = torch.zeros((batch, entities), dtype=torch.float32, device=device)
            if model.entity_feature == 'bias':
                return mask + model.bias.unsqueeze(0), (1 - mask).bool()
            if model.entity_feature == 'RotatE':
                return mask + model.RotatE(all_h, all_r), (1 - mask).bool()
            return mask - float('-inf'), mask.bool()
        output = features[candidates]
        output = torch.relu(model.rule_to_entity.layer_norm(model.rule_to_entity.add_model(output)))
        rel = model.relation_emb(all_r[0]).unsqueeze(0).expand(output.size(0), -1)
        output = model.score_model(torch.cat([output, rel], dim=-1)).squeeze(-1)
        score = torch.zeros(batch * entities, device=device)
        score.scatter_(0, candidates, output)
        score = score.view(batch, entities)
        mask = support.view(batch, entities)
        if model.entity_feature == 'bias':
            score = score + model.bias.unsqueeze(0)
            mask = torch.ones_like(mask)
        elif model.entity_feature == 'RotatE':
            score = score + model.RotatE(all_h, all_r)
            mask = torch.ones_like(mask)
        else:
            score = score.masked_fill(~mask, float('-inf'))
        return score, mask
