"""Whole-preload-allocation LRU with exact tensor backing-storage accounting.

The manager exclusively owns the adapter cache. Requests execute serially; callers
must complete/synchronize consumption before admitting the next request. Block
metadata holds keys and integer sizes, never Tensor references. The budget covers
persistent count tensors, not graph/model, allocator reserve, or construction and
forward scratch. GPU terminal capacity checks may allocate temporary tensors; the
CPU/GPU capacity checks and scalar synchronization stay inside admit(). Blocks
touched by the current request cannot be evicted. Trusted-producer mode is only
for a separately verified source guaranteeing per-rule unique in-range flat
indices, strictly positive exact-int64 counts, and a proved absence of overflow.
It skips redundant content validation, never exact global candidate counting.
"""
from collections import OrderedDict
from dataclasses import dataclass
import time
import torch


def _entry_storage(entry):
    result = {}
    for tensor in (entry.candidates, entry.crow, entry.columns, entry.values):
        storage = tensor.untyped_storage()
        identity = (tensor.device.type, tensor.device.index, storage.data_ptr())
        size = storage.nbytes()
        if identity in result and result[identity] != size:
            raise RuntimeError('inconsistent backing storage size')
        result[identity] = size
    return result


class _StorageDict(dict):
    """Actual reference counts updated during the inherited preloader itself."""
    def __init__(self):
        super().__init__()
        self.refs = {}
        self.signatures = {}
        self.storage_bytes = self.payload_bytes = self.nnz = self.candidates = 0

    def __setitem__(self, key, entry):
        if key in self or key != entry.key:
            raise KeyError('immutable cache entries require a fresh matching key')
        signature = _entry_storage(entry)
        for identity, size in signature.items():
            if identity in self.refs and self.refs[identity][0] != size:
                raise RuntimeError('live storage changed size')
        payload = sum(t.numel() * t.element_size() for t in
                      (entry.candidates, entry.crow, entry.columns, entry.values))
        super().__setitem__(key, entry)
        self.signatures[key] = (signature, payload, entry.nnz, entry.n_candidates)
        for identity, size in signature.items():
            if identity not in self.refs:
                self.refs[identity] = [size, 0]
                self.storage_bytes += size
            self.refs[identity][1] += 1
        self.payload_bytes += payload
        self.nnz += entry.nnz
        self.candidates += entry.n_candidates

    def __delitem__(self, key):
        signature, payload, nnz, candidates = self.signatures.pop(key)
        super().__delitem__(key)
        for identity, size in signature.items():
            self.refs[identity][1] -= 1
            if self.refs[identity][1] == 0:
                self.storage_bytes -= size
                del self.refs[identity]
        self.payload_bytes -= payload
        self.nnz -= nnz
        self.candidates -= candidates

    def clear(self):
        super().clear()
        self.refs.clear()
        self.signatures.clear()
        self.storage_bytes = self.payload_bytes = self.nnz = self.candidates = 0

    def pop(self, *args, **kwargs):
        raise TypeError('evict through AllocationBlockLRU')
    def popitem(self):
        raise TypeError('evict through AllocationBlockLRU')
    def update(self, *args, **kwargs):
        raise TypeError('preload individual entries through adapter')
    def setdefault(self, *args, **kwargs):
        raise TypeError('preload individual entries through adapter')
    def __ior__(self, other):
        raise TypeError('preload individual entries through adapter')


@dataclass(frozen=True)
class _Block:
    keys: tuple
    storage_bytes: int


class AllocationBlockLRU:
    def __init__(self, adapter, budget_bytes, *, trusted_unique_positive_terminals=False):
        if adapter.cache:
            raise ValueError('manager requires an empty adapter cache')
        if int(budget_bytes) != budget_bytes or budget_bytes < 0:
            raise ValueError('nonnegative integer budget required')
        self.adapter = adapter
        self.budget_bytes = int(budget_bytes)
        self.trusted_unique_positive_terminals = bool(trusted_unique_positive_terminals)
        self.cache = _StorageDict()
        adapter.cache = self.cache
        # This remains exact while the base preload is inserting new entries.
        adapter.cache_statistics = self._adapter_statistics
        self.blocks = OrderedDict()
        self.key_to_block = {}
        self._next_block = self._serial = 0
        self._version = (adapter.namespace, adapter.rule_version)
        self.totals = dict(lookups=0, hits=0, misses=0, unique_misses=0,
                           admitted_blocks=0, admitted_keys=0, evicted_blocks=0,
                           evicted_keys=0, evicted_bytes=0, rejected_admissions=0)
        self.peak_bytes = 0

    def _check_version(self):
        if self.adapter.cache is not self.cache:
            raise RuntimeError('adapter cache replaced outside manager')
        if (self.adapter.namespace, self.adapter.rule_version) != self._version:
            raise RuntimeError('call manager.clear after adapter version reset')

    def _adapter_statistics(self):
        c = self.cache
        return dict(cached_query_keys=len(c), persistent_device=str(self.adapter.device),
                    persistent_unique_storage_bytes=c.storage_bytes,
                    persistent_tensor_payload_bytes=c.payload_bytes,
                    cached_nonzeros=c.nnz, cached_active_entities=c.candidates,
                    storage_note='count-cache backing storage only; excludes graph/model, allocator reserve, construction/forward scratch and external references')

    def statistics(self):
        self._check_version()
        return dict(**self._adapter_statistics(), budget_bytes=self.budget_bytes,
                    resident_blocks=len(self.blocks), peak_persistent_bytes=self.peak_bytes,
                    trusted_unique_positive_terminals=self.trusted_unique_positive_terminals,
                    terminal_validation=('producer guarantees unique in-range indices, positive exact-int64 counts and overflow safety; metadata and exact global candidate count checked' if self.trusted_unique_positive_terminals else 'metadata, index range, positive counts and per-rule uniqueness checked'),
                    policy='whole-preload-allocation LRU, current-request hit blocks pinned',
                    **self.totals)

    def _keys(self, h, r, removed):
        relation = self.adapter._cpu_queries(h, r, removed)
        heads = h.tolist()
        edges = [None] * len(heads) if removed is None else removed.tolist()
        return tuple(self.adapter._key(relation, head, edge) for head, edge in zip(heads, edges))

    def lookup(self, h_cpu, r_cpu, removed_cpu=None):
        self._check_version()
        keys = self._keys(h_cpu, r_cpu, removed_cpu)
        misses, seen_missing, hit_keys, pinned = [], set(), set(), set()
        hits = 0
        for index, key in enumerate(keys):
            block_id = self.key_to_block.get(key)
            if block_id is None:
                if key in self.cache:
                    raise RuntimeError('unregistered cache insertion')
                if key not in seen_missing:
                    seen_missing.add(key)
                    misses.append(index)
            else:
                if key not in self.cache:
                    raise RuntimeError('unregistered cache deletion')
                hits += 1
                hit_keys.add(key)
                pinned.add(block_id)
                self.blocks.move_to_end(block_id)
        self._serial += 1
        self.totals['lookups'] += 1
        self.totals['hits'] += hits
        self.totals['misses'] += len(keys) - hits
        self.totals['unique_misses'] += len(misses)
        return dict(serial=self._serial, namespace=self._version[0], rule_version=self._version[1],
                    keys=keys, miss_indices=misses, missing_keys=tuple(keys[i] for i in misses),
                    hits=hits, misses=len(keys)-hits, unique_hits=len(hit_keys),
                    unique_misses=len(misses), pinned_block_ids=tuple(sorted(pinned)),
                    pinned_bytes=sum(self.blocks[b].storage_bytes for b in pinned))

    def _prospective(self, h, r, removed, terminals):
        relation = self.adapter._cpu_queries(h, r, removed)
        batch = h.numel()
        entities = self.adapter.model.graph.entity_size
        if len(terminals) != len(self.adapter.rules_for_relation(relation)):
            raise ValueError('terminal count differs from rule plan')
        flat = []
        nnz = 0
        source_device = None
        for indices, values in terminals:
            if indices.dtype != torch.int64 or values.dtype != torch.int64 or indices.device != values.device:
                raise TypeError('matching-device int64 sparse terminal pairs required')
            if indices.device.type != 'cpu' and indices.device != self.adapter.device:
                raise ValueError('terminal source must be CPU or adapter device')
            if source_device is None:
                source_device = indices.device
            elif source_device != indices.device:
                raise ValueError('all terminal pairs must share one source device')
            if indices.ndim != 1 or values.shape != indices.shape:
                raise ValueError('invalid sparse terminal shape')
            if indices.numel():
                if not self.trusted_unique_positive_terminals:
                    if int(indices.min()) < 0 or int(indices.max()) >= batch * entities or bool((values <= 0).any()):
                        raise ValueError('invalid sparse terminal indices/counts')
                    if torch.unique(indices).numel() != indices.numel():
                        raise ValueError('duplicate sparse terminal indices')
                flat.append(indices)
            nnz += indices.numel()
        active = int(torch.unique(torch.cat(flat)).numel()) if flat else 0
        return dict(bytes=int(12 * nnz + 16 * active + 8 * batch), nnz=int(nnz), active=int(active))

    def admit(self, context, missing_h, missing_r, missing_removed, terminals):
        self._check_version()
        if context['serial'] != self._serial or (context['namespace'], context['rule_version']) != self._version:
            raise ValueError('stale request context')
        if not context['missing_keys']:
            if missing_h.numel() or len(terminals):
                raise ValueError('all-hit admission takes empty queries and terminals')
            return True, dict(admitted=False, reason='all_hit', evicted_blocks=0, evicted_keys=0, evicted_bytes=0)
        keys = self._keys(missing_h, missing_r, missing_removed)
        if keys != tuple(context['missing_keys']) or len(set(keys)) != len(keys):
            raise ValueError('admission must contain exactly first-occurrence unique misses in context order')
        if any(key in self.cache for key in keys):
            raise ValueError('request misses changed before admission')
        started = time.perf_counter()
        prospective = self._prospective(missing_h, missing_r, missing_removed, terminals)
        pinned = set(context['pinned_block_ids'])
        pinned_bytes = sum(self.blocks[b].storage_bytes for b in pinned)
        report = dict(prospective_bytes=prospective['bytes'], prospective_nnz=prospective['nnz'],
                      prospective_active_entities=prospective['active'], pinned_bytes=pinned_bytes,
                      eviction_blocks=[], evicted_blocks=0, evicted_keys=0, evicted_bytes=0,
                      capacity_check_seconds=time.perf_counter()-started)
        if pinned_bytes + prospective['bytes'] > self.budget_bytes:
            self.totals['rejected_admissions'] += 1
            report.update(admitted=False, reason='pinned_plus_new_block_exceeds_budget')
            return False, report
        # Admission feasibility is proven before touching existing cache entries.
        for block_id in tuple(self.blocks):
            if self.cache.storage_bytes + prospective['bytes'] <= self.budget_bytes:
                break
            if block_id in pinned:
                continue
            block = self.blocks.pop(block_id)
            for key in block.keys:
                del self.cache[key]
                del self.key_to_block[key]
            report['eviction_blocks'].append(block_id)
            report['evicted_blocks'] += 1
            report['evicted_keys'] += len(block.keys)
            report['evicted_bytes'] += block.storage_bytes
        before = self.cache.storage_bytes
        try:
            preload = self.adapter.preload_from_terminals(missing_h, missing_r, missing_removed, terminals)
            if any(key not in self.cache for key in keys):
                raise RuntimeError('preload failed to insert required keys')
            delta = self.cache.storage_bytes - before
            if delta != prospective['bytes']:
                raise RuntimeError(f'actual block storage {delta} differs from exact prospective {prospective["bytes"]}')
            if self.cache.storage_bytes > self.budget_bytes:
                raise RuntimeError('persistent count-cache budget exceeded')
        except BaseException:
            for key in keys:
                if key in self.cache:
                    del self.cache[key]
            raise
        block_id = self._next_block
        self._next_block += 1
        self.blocks[block_id] = _Block(keys, delta)
        for key in keys:
            self.key_to_block[key] = block_id
        self.peak_bytes = max(self.peak_bytes, self.cache.storage_bytes)
        self.totals['admitted_blocks'] += 1
        self.totals['admitted_keys'] += len(keys)
        for field in ('evicted_blocks', 'evicted_keys', 'evicted_bytes'):
            self.totals[field] += report[field]
        self._serial += 1  # A context cannot be admitted twice.
        report.update(admitted=True, reason='fits', block_id=block_id,
                      actual_new_storage_bytes=delta, preload=preload,
                      persistent_storage_bytes=self.cache.storage_bytes)
        return True, report

    def clear(self):
        """Drop owned entries and accept a version reset performed by the adapter."""
        if self.adapter.cache is not self.cache:
            raise RuntimeError('adapter cache replaced outside manager')
        self.cache.clear()
        self.blocks.clear()
        self.key_to_block.clear()
        self._version = (self.adapter.namespace, self.adapter.rule_version)
        self._serial += 1

    def verify_invariants(self):
        """Slow explicit audit; never needed inside timed request loops."""
        self._check_version()
        all_keys = [key for block in self.blocks.values() for key in block.keys]
        if len(all_keys) != len(set(all_keys)) or set(all_keys) != set(self.cache) or set(all_keys) != set(self.key_to_block):
            raise AssertionError('block/key ownership mismatch')
        scan = {}
        for block_id, block in self.blocks.items():
            block_scan = {}
            for key in block.keys:
                if self.key_to_block[key] != block_id:
                    raise AssertionError('wrong block ownership')
                block_scan.update(_entry_storage(self.cache[key]))
            if sum(block_scan.values()) != block.storage_bytes:
                raise AssertionError('block storage mismatch')
            for identity, size in block_scan.items():
                if size and identity in scan:
                    raise AssertionError('nonempty storage shared across allocation blocks')
                scan[identity] = size
        if sum(scan.values()) != self.cache.storage_bytes or sum(b.storage_bytes for b in self.blocks.values()) != self.cache.storage_bytes:
            raise AssertionError('incremental unique-storage accounting mismatch')
        if self.cache.storage_bytes > self.budget_bytes:
            raise AssertionError('persistent storage budget exceeded')
        return dict(passed=True, audited_blocks=len(self.blocks), audited_keys=len(self.cache),
                    audited_unique_storage_bytes=sum(scan.values()))
