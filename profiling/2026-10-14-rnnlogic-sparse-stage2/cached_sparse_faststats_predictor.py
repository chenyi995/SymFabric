"""Same explicit count-cache execution, with incremental O(1) metadata queries.

The base cached_sparse_predictor.py is unchanged. This module changes only
bookkeeping during insertion/reset: no integer count, CSR matrix, neural forward,
cache key, or gradient calculation changes. Deletion/replacement is unsupported;
reset() is the sole eviction operation, as for this explicit preloaded-set test.
"""
from cached_sparse_predictor import CachedSparsePredictor


class _IncrementalCountCache(dict):
    def __init__(self):
        super().__init__()
        self._storage_sizes = {}
        self.unique_storage_bytes = 0
        self.payload_bytes = 0
        self.nonzeros = 0
        self.active_entities = 0

    def __setitem__(self, key, entry):
        if key in self:
            raise KeyError('count-cache entries are immutable; reset before replacing keys')
        if key != entry.key:
            raise ValueError('cache dictionary key differs from entry key')
        # Inspect exactly the same backing storages as the base full scan.
        storages = {}
        payload = 0
        for tensor in (entry.candidates, entry.crow, entry.columns, entry.values):
            payload += tensor.numel() * tensor.element_size()
            storage = tensor.untyped_storage()
            storage_key = (tensor.device.type, tensor.device.index, storage.data_ptr())
            size = storage.nbytes()
            if storage_key in self._storage_sizes and self._storage_sizes[storage_key] != size:
                raise RuntimeError('live tensor storage changed size unexpectedly')
            storages[storage_key] = size
        super().__setitem__(key, entry)
        for storage_key, size in storages.items():
            if storage_key not in self._storage_sizes:
                self._storage_sizes[storage_key] = size
                self.unique_storage_bytes += size
        self.payload_bytes += payload
        self.nonzeros += entry.nnz
        self.active_entities += entry.n_candidates

    def clear(self):
        super().clear()
        self._storage_sizes.clear()
        self.unique_storage_bytes = 0
        self.payload_bytes = 0
        self.nonzeros = 0
        self.active_entities = 0

    def __delitem__(self, key):
        raise TypeError('individual count-cache deletion is unsupported; use adapter.reset()')

    def pop(self, *args, **kwargs):
        raise TypeError('individual count-cache deletion is unsupported; use adapter.reset()')

    def popitem(self):
        raise TypeError('individual count-cache deletion is unsupported; use adapter.reset()')

    def update(self, *args, **kwargs):
        raise TypeError('bulk dictionary mutation bypasses count-cache accounting; insert through preload()')

    def setdefault(self, *args, **kwargs):
        raise TypeError('dictionary defaults are unsupported; insert through preload()')

    def __ior__(self, other):
        raise TypeError('dictionary union mutation is unsupported; insert through preload()')


class CachedSparseFastStatsPredictor(CachedSparsePredictor):
    """Base behavior with exact incremental counters for cached tensor storage."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cache:
            raise RuntimeError('base constructor unexpectedly preloaded entries')
        self.cache = _IncrementalCountCache()

    def cache_statistics(self):
        if not isinstance(self.cache, _IncrementalCountCache):
            raise TypeError('cache container was replaced outside the supported preload/reset API')
        return {'cached_query_keys': len(self.cache), 'persistent_device': str(self.device),
                'persistent_unique_storage_bytes': self.cache.unique_storage_bytes,
                'persistent_tensor_payload_bytes': self.cache.payload_bytes,
                'cached_nonzeros': self.cache.nonzeros,
                'cached_active_entities': self.cache.active_entities,
                'storage_note': 'count-cache tensor backing storage only; excludes graph/model, Python metadata, allocator reserve and warm-forward workspace'}
