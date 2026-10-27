"""Optional sparse float consumption of exact upstream-style integer rule counts.

This changes only the rule-count-to-neural-embedding contraction in the stage-2
adapter. Exact int64 grounding remains the external controller's responsibility.
Counts convert to float32 at the same semantic boundary as upstream PredictorPlus.
PyTorch sparse COO multiplication computes the weighted rule-embedding sum and
its dense embedding gradients. No dense [rule,union_candidate] matrix is built.

The sparse terminal contract is inherited from StreamingPredictor: per-rule
unique nonzero int64 flat_indices=query*entity_count+entity and int64 values.
By default CPU terminals compact on CPU, then a coalesced COO tensor and
candidate ids transfer to the model device within forward. With
compact_device='model', block-concatenated flat ids, rule ids and integer values
transfer first, and unique/coalesce/float conversion run on the model device. There is no offline count preprocessing or cross-forward cache.
The parent still maintains dense [batch*entity,hidden] features and dense scores.
"""
from __future__ import annotations

import torch
from streaming_predictor import StreamingPredictor


class SparseMMPredictor(StreamingPredictor):
    """Same constructor/forward as StreamingPredictor, sparse terminals only.

    ``no_dense_compact`` is an explicit contract flag, not an alternative path:
    False is rejected. Unsupported sparse PyTorch operators fail visibly; this
    adapter never silently densifies or substitutes another grounding engine.
    """
    def __init__(self, model, block_size=32, rule_order='prefix',
                 grounder_factory=None, collect_stats=False,
                 terminal_format='sparse', no_dense_compact=True, compact_device='source'):
        if compact_device not in ('source', 'model'):
            raise ValueError('compact_device must be source or model')
        self.compact_device = compact_device
        if terminal_format != 'sparse':
            raise ValueError('SparseMMPredictor requires sparse terminal_format')
        if not no_dense_compact:
            raise ValueError('SparseMMPredictor always requires no_dense_compact=True')
        super().__init__(model, block_size=block_size, rule_order=rule_order,
                         grounder_factory=grounder_factory,
                         collect_stats=collect_stats, terminal_format=terminal_format)
        self.no_dense_compact = True
        self._coo_diagnostics = []

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
        rule = torch.cat([torch.full_like(index, slot) for slot, (index, _) in enumerate(terminals)])
        raw_values = torch.cat([value for _, value in terminals])
        execution_device = device if self.compact_device == 'model' else source_device
        # Exactly three block-level transfers in model-side compaction mode;
        # never a per-rule copy. Integer values cross before the original float
        # conversion boundary, then unique/coalesce run on the selected device.
        flat = flat.to(execution_device)
        rule = rule.to(execution_device)
        values = raw_values.to(execution_device).float()
        candidates, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        coordinates = torch.stack((rule, inverse), dim=0)
        compact = torch.sparse_coo_tensor(coordinates, values,
                    (len(terminals), candidates.numel()), device=execution_device,
                    dtype=torch.float32).coalesce()
        # Coalescing handles order only under the unique-terminal input contract.
        # It is not a float approximation to integer duplicate-path reduction.
        compact = compact.to(device)
        candidates = candidates.to(device)
        if self.collect_stats:
            self._coo_diagnostics.append({
                'shape': list(compact.shape),
                'nnz': compact._nnz(),
                'value_bytes': compact.values().numel() * compact.values().element_size(),
                'index_bytes': compact.indices().numel() * compact.indices().element_size(),
                'logical_dense_float_bytes': compact.numel() * compact.element_size(),
                'source_device': str(source_device),
                'compaction_device': str(execution_device),
                'compaction_placement': self.compact_device,
                'model_device': str(device),
            })
        return candidates, compact

    def forward(self, *args, **kwargs):
        self._coo_diagnostics = []
        output = super().forward(*args, **kwargs)
        if self.collect_stats:
            rows = self._coo_diagnostics
            stats = self.last_stats
            # The parent uses Tensor.numel(), which reports logical size for a
            # sparse tensor. Rename that figure and provide actual COO payloads.
            stats['peak_compact_logical_dense_float_bytes'] = stats['peak_compact_float_bytes']
            stats['total_compact_logical_dense_float_bytes'] = stats['total_compact_float_bytes']
            stats['peak_compact_float_bytes'] = max((r['value_bytes'] for r in rows), default=0)
            stats['total_compact_float_bytes'] = sum(r['value_bytes'] for r in rows)
            stats['peak_compact_sparse_storage_bytes'] = max((r['value_bytes'] + r['index_bytes'] for r in rows), default=0)
            stats['total_compact_sparse_storage_bytes'] = sum(r['value_bytes'] + r['index_bytes'] for r in rows)
            stats['compact_layout'] = 'torch.sparse_coo'
            stats['dense_compact_materialized'] = False
            stats['coo_blocks'] = rows
            stats['compact_memory_note'] = 'COO index/value payload only, not allocator peak, transpose copies, temporary coalescing storage, or all autograd saved tensors'
        return output
