"""Exact query-level memoization of fixed RNNLogic integer rule counts.

A persistent CPU cache is explicitly scoped to an immutable graph/rule namespace.
Every key also includes head relation, head entity, original removed edge ID (or
None), and the complete rule body. Thus batches may be reordered or mixed, and
query-specific edge deletion remains valid. NN weights never enter this cache.
A miss executes one query's complete body with the supplied sparse backend.

No automatic graph/version discovery or invalidation is implemented. The caller
must reset_namespace when graph/rule versions change and supply a freshly built
backend if adjacency changes. Cache contents are terminal counts, not logits.
"""
from collections import OrderedDict
import operator
import time

import torch


class MemoizedGrounder:
    """Persistent sparse-terminal cache compatible with StreamingPredictor.

    ``capacity_bytes`` limits explicitly charged logical storage: int64 entity
    IDs and values plus a fixed key-field charge. It is not a Python RSS bound.
    Empty outputs also incur a key charge, so a bounded cache cannot accumulate
    unlimited zero-valued entries. Actual process overhead must be measured
    separately. None means unbounded logical capacity and is reported as such.

    With collect_stats=False (the timing default), hit/miss/build/lookup counters
    and timers are disabled. Capacity accounting and LRU policy remain active.
    Set collect_stats=True for diagnostic runs; stats are available from finish.
    Returned sparse tensors are fresh copies; consumers cannot mutate the cache.
    """

    cross_forward_cache = True

    def __init__(self, graph, rule_bodies, backend=None, *, namespace,
                 capacity_bytes=None, collect_stats=False, threads=1):
        if not isinstance(namespace, str) or not namespace:
            raise ValueError('an explicit nonempty graph/rule namespace is required')
        if capacity_bytes is not None:
            capacity_bytes = operator.index(capacity_bytes)
            if capacity_bytes < 0:
                raise ValueError('capacity_bytes must be nonnegative or None')
        if backend is None:
            from sparse_cpu_backend_optimized import SparseCPUBackendOptimized
            backend = SparseCPUBackendOptimized(graph, threads=threads)
        self.graph = graph
        self.backend = backend
        self.entity_size = int(graph.entity_size)
        if backend.entity_size != self.entity_size:
            raise ValueError('backend entity count differs from graph')
        self.rule_bodies = self._bodies(rule_bodies)
        self.namespace = namespace
        self.capacity_bytes = capacity_bytes
        self.collect_stats = bool(collect_stats)
        self._cache = OrderedDict()
        self._charged_bytes = 0
        self._payload_bytes = 0
        self._active = False
        self._totals = self._empty_counters()
        self._forward = self._empty_counters()
        self._namespace_reset_count = 0
        self.metadata = {
            'backend': 'exact_query_rule_terminal_memoization',
            'namespace': namespace,
            'capacity_bytes': capacity_bytes,
            'capacity_accounting': 'tensor payload plus logical key-field charge; excludes Python object/allocator/process overhead',
            'key': ['namespace', 'head_relation', 'head_entity', 'removed_original_edge_id_or_None', 'body_tuple'],
            'miss_execution': 'complete single-query body on supplied CPU sparse backend',
            'reuse_scope': 'across forwards, batch orders, batch groupings, and NN weight updates within fixed namespace',
            'automatic_graph_invalidation': False,
            'count_semantics': 'exact int64 path multiplicity; one original edge removed per query at every head-relation step',
        }

    @staticmethod
    def _integer(value):
        if torch.is_tensor(value):
            raise TypeError('relations must be Python integers')
        return operator.index(value)

    @classmethod
    def _bodies(cls, bodies):
        return tuple(tuple(cls._integer(r) for r in body) for body in bodies)

    @staticmethod
    def _signature(x):
        if x is None:
            return None
        try:
            version = x._version
        except RuntimeError:
            version = None
        return id(x), version

    @staticmethod
    def _empty_counters():
        return {name: 0 for name in ['query_lookups', 'hits', 'misses', 'builds',
                                     'propagations', 'evictions', 'oversized_skips', 'insertions']}

    def _inc(self, name, value=1):
        if self.collect_stats:
            self._forward[name] += value
            self._totals[name] += value

    def _add_time(self, name, seconds):
        if self.collect_stats:
            self._forward[name] = self._forward.get(name, 0.0) + seconds
            self._totals[name] = self._totals.get(name, 0.0) + seconds

    def _entry_charge(self, body, ids, values):
        payload = (ids.numel() + values.numel()) * 8
        # Logical namespace and key fields plus a fixed 64-byte entry charge.
        # This is deliberately explicit, not an estimate of Python object size.
        logical_key = 64 + len(self.namespace.encode('utf-8')) + 8 * (4 + len(body))
        return payload + logical_key, payload

    def begin(self, h, head, removed=None):
        if self._active:
            raise RuntimeError('finish the preceding forward before begin')
        if h.device.type != 'cpu' or h.dtype != torch.int64 or h.ndim != 1:
            raise TypeError('h must be a CPU int64 vector; use a CPU input adapter for hybrid execution')
        if removed is not None and (removed.device.type != 'cpu' or removed.dtype != torch.int64 or removed.shape != h.shape):
            raise TypeError('removed must be a CPU int64 vector matching h')
        self._head = self._integer(head)
        self._h, self._removed = h, removed
        self._h_signature, self._removed_signature = self._signature(h), self._signature(removed)
        self._heads = h.tolist()
        self._removed_ids = [None] * len(h) if removed is None else removed.tolist()
        self._index = 0
        self._rule_calls = 0
        self._forward = self._empty_counters()
        self._forward_start_entries = len(self._cache)
        self._forward_start_bytes = self._charged_bytes
        self._active = True
        return self

    def _build(self, entity, head, body, removed):
        started = time.perf_counter() if self.collect_stats else None
        h = torch.tensor([entity], dtype=torch.int64)
        removal = None if removed is None else torch.tensor([removed], dtype=torch.int64)
        x = self.backend.initial(h)
        for relation in body:
            x = self.backend.propagate(x, relation, removal if relation == head else None)
        ids, values = self.backend.to_sparse(x)
        # Own exactly these payloads rather than retaining an oversized view or
        # exporting library buffers that another object may subsequently mutate.
        ids, values = ids.clone(), values.clone()
        self._inc('builds')
        self._inc('propagations', len(body))
        if started is not None:
            self._add_time('build_seconds', time.perf_counter() - started)
        return ids, values

    def _retain(self, key, ids, values, record_stats=True):
        charged, payload = self._entry_charge(key[-1], ids, values)
        if self.capacity_bytes is not None and charged > self.capacity_bytes:
            if record_stats:
                self._inc('oversized_skips')
            return False
        while self.capacity_bytes is not None and self._cache and self._charged_bytes + charged > self.capacity_bytes:
            _, (_, _, old_charge, old_payload) = self._cache.popitem(last=False)
            self._charged_bytes -= old_charge
            self._payload_bytes -= old_payload
            if record_stats:
                self._inc('evictions')
        self._cache[key] = (ids, values, charged, payload)
        self._charged_bytes += charged
        self._payload_bytes += payload
        if record_stats:
            self._inc('insertions')
        return True

    @torch.no_grad()
    def grounding(self, h, head, body, removed=None):
        if not self._active:
            raise RuntimeError('call begin before grounding')
        if self._signature(h) != self._h_signature or self._signature(removed) != self._removed_signature:
            raise ValueError('query tensor changed within forward')
        if self._integer(head) != self._head:
            raise ValueError('query relation changed within forward')
        body = tuple(self._integer(r) for r in body)
        if self._index >= len(self.rule_bodies) or body != self.rule_bodies[self._index]:
            raise ValueError('grounding calls must follow declared rule order')
        output_ids, output_values = [], []
        for query, (entity, removal) in enumerate(zip(self._heads, self._removed_ids)):
            lookup_started = time.perf_counter() if self.collect_stats else None
            key = (self.namespace, self._head, entity, removal, body)
            entry = self._cache.get(key)
            self._inc('query_lookups')
            if entry is not None:
                self._cache.move_to_end(key)
            if lookup_started is not None:
                self._add_time('lookup_seconds', time.perf_counter() - lookup_started)
            if entry is None:
                self._inc('misses')
                ids, values = self._build(entity, self._head, body, removal)
                self._retain(key, ids, values)
            else:
                self._inc('hits')
                ids, values = entry[0], entry[1]
            output_ids.append(ids + query * self.entity_size)
            output_values.append(values)
        self._index += 1
        self._rule_calls += 1
        if not output_ids:
            return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
        return torch.cat(output_ids), torch.cat(output_values)

    grounding_sparse = grounding

    @property
    def stats(self):
        counters = dict(self._forward) if self.collect_stats else {k: None for k in self._empty_counters()}
        return {
            **counters,
            'policy': 'persistent_query_terminal_lru',
            'namespace': self.namespace,
            'collect_stats': self.collect_stats,
            'capacity_bytes': self.capacity_bytes,
            'current_cache_entries': len(self._cache),
            'current_cache_charged_bytes': self._charged_bytes,
            'current_cache_payload_bytes': self._payload_bytes,
            'namespace_reset_count': self._namespace_reset_count,
            'rule_calls': getattr(self, '_rule_calls', 0),
            'rule_count': len(self.rule_bodies),
            'cumulative_counters': dict(self._totals) if self.collect_stats else None,
            'memory_accounting': self.metadata['capacity_accounting'],
        }

    def finish(self, require_complete=True):
        if not self._active:
            raise RuntimeError('no active forward')
        complete = self._index == len(self.rule_bodies)
        result = self.stats
        result.update(complete=complete, start_cache_entries=self._forward_start_entries,
                      start_cache_charged_bytes=self._forward_start_bytes)
        self._active = False
        self._h = self._removed = self._heads = self._removed_ids = None
        if require_complete and not complete:
            raise RuntimeError('forward did not consume all declared rule bodies')
        return result

    @torch.no_grad()
    def prime_batch(self, h, head, removed, terminals):
        """Explicitly charge bulk count construction separately from cache fill.

        ``terminals`` follows rule_bodies order and contains CPU int64
        (flat_indices, counts), unique nonzero entries with flat=query*N+entity.
        The caller must verify the producing integer backend; this method checks
        structure and consistency with any existing key, not graph semantics.
        Its returned elapsed time includes validation, splitting, copies and LRU
        insertion. It neither hides that time inside setup nor computes logits.
        """
        started = time.perf_counter()
        if self._active:
            raise RuntimeError('cannot prime during an active forward')
        if h.device.type != 'cpu' or h.dtype != torch.int64 or h.ndim != 1:
            raise TypeError('h must be a CPU int64 vector')
        if removed is not None and (removed.device.type != 'cpu' or removed.dtype != torch.int64 or removed.shape != h.shape):
            raise TypeError('removed must be a CPU int64 vector matching h')
        head = self._integer(head)
        heads = h.tolist()
        removals = [None] * len(h) if removed is None else removed.tolist()
        if heads and (min(heads) < 0 or max(heads) >= self.entity_size):
            raise ValueError('head entity out of range')
        terminals = iter(terminals)
        report = {'input_queries': len(heads), 'input_rules': len(self.rule_bodies),
                  'input_nonzeros': 0, 'new_entries': 0, 'existing_verified_entries': 0,
                  'evictions': 0, 'oversized_skips': 0,
                  'start_cache_entries': len(self._cache),
                  'start_cache_charged_bytes': self._charged_bytes}
        for body in self.rule_bodies:
            try:
                ids, values = next(terminals)
            except StopIteration as error:
                raise ValueError('fewer terminals than declared rule bodies') from error
            for tensor in (ids, values):
                if tensor.device.type != 'cpu' or tensor.dtype != torch.int64 or tensor.ndim != 1:
                    raise TypeError('terminals must be CPU int64 vectors')
            if ids.shape != values.shape:
                raise ValueError('terminal index/count shapes differ')
            if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= len(heads) * self.entity_size):
                raise ValueError('terminal flat index out of range')
            if not bool(torch.all(values != 0)):
                raise ValueError('terminal must not include explicit zero counts')
            # A sorted flat index groups queries and allows one bulk split.
            if ids.numel() > 1 and not bool(torch.all(ids[1:] > ids[:-1])):
                ids, order = ids.sort()
                values = values[order]
                if bool(torch.any(ids[1:] == ids[:-1])):
                    raise ValueError('terminal flat indices must be unique')
            report['input_nonzeros'] += ids.numel()
            boundaries = torch.searchsorted(ids, torch.arange(len(heads) + 1, dtype=torch.int64) * self.entity_size).tolist()
            for query, (entity, removal) in enumerate(zip(heads, removals)):
                key = (self.namespace, head, entity, removal, body)
                lo, hi = boundaries[query:query + 2]
                entity_ids = (ids[lo:hi] - query * self.entity_size).clone()
                counts = values[lo:hi].clone()
                existing = self._cache.get(key)
                if existing is not None:
                    old_ids, old_values = existing[:2]
                    if not torch.equal(old_ids, entity_ids):
                        order = old_ids.argsort()
                        old_ids, old_values = old_ids[order], old_values[order]
                    if not torch.equal(old_ids, entity_ids) or not torch.equal(old_values, counts):
                        raise ValueError('provided terminal disagrees with an existing exact cache key')
                    self._cache.move_to_end(key)
                    report['existing_verified_entries'] += 1
                    continue
                before_count = len(self._cache)
                inserted = self._retain(key, entity_ids, counts, record_stats=False)
                if inserted:
                    report['new_entries'] += 1
                    report['evictions'] += before_count + 1 - len(self._cache)
                else:
                    report['oversized_skips'] += 1
        try:
            next(terminals)
        except StopIteration:
            pass
        else:
            raise ValueError('more terminals than declared rule bodies')
        report.update(prime_seconds=time.perf_counter() - started,
                      final_cache_entries=len(self._cache),
                      final_cache_charged_bytes=self._charged_bytes,
                      final_cache_payload_bytes=self._payload_bytes,
                      capacity_bytes=self.capacity_bytes,
                      namespace=self.namespace,
                      source_verification='caller responsibility; structural and existing-key checks performed here')
        return report

    def reset_namespace(self, namespace, *, backend=None, rule_bodies=None):
        """Explicitly invalidate cached counts; no graph mutation is inferred."""
        if self._active:
            raise RuntimeError('cannot reset namespace during a forward')
        if not isinstance(namespace, str) or not namespace:
            raise ValueError('an explicit nonempty namespace is required')
        if backend is not None:
            if backend.entity_size != self.entity_size:
                raise ValueError('new backend entity count differs; create a new controller')
            self.backend = backend
        if rule_bodies is not None:
            self.rule_bodies = self._bodies(rule_bodies)
        self._cache.clear()
        self._charged_bytes = self._payload_bytes = 0
        self._totals = self._empty_counters()
        self.namespace = namespace
        self.metadata['namespace'] = namespace
        self._namespace_reset_count += 1
