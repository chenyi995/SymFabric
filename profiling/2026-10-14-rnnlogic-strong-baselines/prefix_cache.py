"""Per-forward prefix sharing for the unchanged RNNLogic grounding interface.

Only integer states are cached. The graph's existing ``propagate`` method can be
replaced independently with a semantically equivalent native or fused backend.
No cross-forward or cross-query reuse is permitted by this controller.

Memory statistics describe logical tensor storage owned by this controller,
including overlapping propagation input/output. They exclude propagate-internal
messages, graph/NN tensors, allocator overhead and output aliases retained by the
caller. Measure allocator peaks separately. The LRU budget limits cached storage;
transient input/output storage can temporarily exist outside that budget.
"""
from collections import OrderedDict
import operator

import torch


class PrefixGrounder:
    """Execute an ordered list of rule bodies with safe per-forward sharing.

    Call ``begin(h, head, removed)`` before each predictor forward, replace
    ``graph.grounding`` with this object's ``grounding`` method, and call
    ``finish()`` afterwards. ``head`` and body entries must be Python integers.
    Each grounding call must use the same h/removed tensor objects, unmodified,
    and follow rule_bodies order (including duplicate and empty rules).
    Set ``collect_stats=False`` for timed forwards and enable it between
    forwards for diagnostics. Disabled statistics never change cache policy
    or capacity accounting; maximum-memory counters are then unmeasured.

    policies:
      all: retain every computed prefix, including the initial one-hot state.
      last_use: retain only prefixes needed by a later rule, release at last use.
      lru: evict least-recently-used states to enforce budget_bytes; recompute
           missing prefixes from their deepest available cached ancestor.
      last_use_lru: apply the same LRU capacity limit, but do not cache states
           without future consumers and release states at their last use.
    """

    def __init__(self, graph, rule_bodies, policy="all", budget_bytes=None, collect_stats=True):
        if policy not in {"all", "last_use", "lru", "last_use_lru"}:
            raise ValueError("policy must be all, last_use, lru, or last_use_lru")
        self._bounded = policy in {"lru", "last_use_lru"}
        self._release_last_use = policy in {"last_use", "last_use_lru"}
        if self._bounded:
            if budget_bytes is None or operator.index(budget_bytes) < 0:
                raise ValueError("bounded policies require nonnegative budget_bytes")
            budget_bytes = operator.index(budget_bytes)
        elif budget_bytes is not None:
            raise ValueError("budget_bytes applies only to lru or last_use_lru")
        self.graph = graph
        self.policy = policy
        self.budget_bytes = budget_bytes
        self.collect_stats = bool(collect_stats)
        self.rule_bodies = tuple(tuple(self._integer(r) for r in b) for b in rule_bodies)
        self._plans = tuple(tuple(b[:i] for i in range(len(b) + 1)) for b in self.rule_bodies)
        self._last_use = {}
        for index, prefixes in enumerate(self._plans):
            for prefix in prefixes:
                self._last_use[prefix] = index
        self._release_at = [[] for _ in self.rule_bodies]
        for prefix, index in self._last_use.items():
            self._release_at[index].append(prefix)
        self._cache = OrderedDict()
        self._cache_meta = {}
        self._cache_storages = {}
        self._cache_bytes = 0
        self._active = False
        self._stats = {}
        self._h = self._removed = None

    @staticmethod
    def _integer(value):
        if torch.is_tensor(value):
            raise TypeError("head/body relations must be Python integers; extract outside the timed loop")
        return operator.index(value)

    @staticmethod
    def _signature(tensor):
        if tensor is None:
            return None
        try:
            version = tensor._version
        except RuntimeError:
            # Inference tensors have no version counter. Callers must not mutate
            # them during the forward, as required for all inputs by this API.
            version = None
        return id(tensor), version

    @staticmethod
    def _storage(tensor):
        storage = tensor.untyped_storage()
        return (tensor.device.type, tensor.device.index, storage.data_ptr()), storage.nbytes()

    def begin(self, h, head, removed=None):
        if self._active:
            raise RuntimeError("finish the preceding forward before begin")
        if h.dtype != torch.int64 or h.ndim != 1:
            raise ValueError("h must be a one-dimensional int64 tensor")
        if removed is not None and (removed.dtype != torch.int64 or removed.shape != h.shape or removed.device != h.device):
            raise ValueError("removed must be int64 with the same shape/device as h")
        self._head = self._integer(head)
        self._h, self._removed = h, removed
        self._h_signature = self._signature(h)
        self._removed_signature = self._signature(removed)
        self._cache.clear()
        self._cache_meta.clear()
        self._cache_storages.clear()
        self._cache_bytes = 0
        self._seen = set()
        self._index = 0
        self._active = True
        self._stats = {
            "policy": self.policy,
            "budget_bytes": self.budget_bytes,
            "collect_stats": self.collect_stats,
            "memory_accounting": "logical controller state storage; excludes propagate-internal temporaries and externally retained outputs",
            "rule_count": len(self.rule_bodies),
            "rule_calls": 0,
            "propagations": 0,
            "hits": 0,
            "misses": 0,
            "cache_lookup_hits": 0,
            "cache_lookup_misses": 0,
            "identity_builds": 0,
            "recomputes": 0,
            "insertions": 0,
            "evictions": 0,
            "last_use_releases": 0,
            "max_cache_states": 0,
            "max_cache_bytes": 0,
            "max_working_bytes_outside_cache": 0,
            "max_live_bytes": 0,
        }
        return self

    def _observe(self, *working):
        if not self.collect_stats:
            return
        extra = {}
        for tensor in working:
            key, size = self._storage(tensor)
            if key not in self._cache_storages:
                extra[key] = size
        working_bytes = sum(extra.values())
        s = self._stats
        s["max_cache_states"] = max(s["max_cache_states"], len(self._cache))
        s["max_cache_bytes"] = max(s["max_cache_bytes"], self._cache_bytes)
        s["max_working_bytes_outside_cache"] = max(s["max_working_bytes_outside_cache"], working_bytes)
        s["max_live_bytes"] = max(s["max_live_bytes"], self._cache_bytes + working_bytes)

    def _drop(self, prefix, reason):
        del self._cache[prefix]
        key, _ = self._cache_meta.pop(prefix)
        entry = self._cache_storages[key]
        entry[0] -= 1
        if entry[0] == 0:
            self._cache_bytes -= entry[1]
            del self._cache_storages[key]
        if self.collect_stats:
            self._stats[reason] += 1

    def _retain(self, prefix, tensor):
        if self._release_last_use and self._last_use[prefix] <= self._index:
            return
        key, size = self._storage(tensor)
        incremental = 0 if key in self._cache_storages else size
        if self._bounded:
            if size > self.budget_bytes:
                return
            while self._cache and self._cache_bytes + incremental > self.budget_bytes:
                self._drop(next(iter(self._cache)), "evictions")
                incremental = 0 if key in self._cache_storages else size
        self._cache[prefix] = tensor
        self._cache_meta[prefix] = key, size
        if key in self._cache_storages:
            self._cache_storages[key][0] += 1
        else:
            self._cache_storages[key] = [1, size]
            self._cache_bytes += size
        if self.collect_stats:
            self._stats["insertions"] += 1
        self._observe(tensor)

    @torch.no_grad()
    def grounding(self, h, head, body, removed=None):
        if not self._active:
            raise RuntimeError("call begin before grounding")
        if self._signature(h) != self._h_signature or self._signature(removed) != self._removed_signature:
            raise ValueError("query tensors changed within a forward; begin a new forward")
        if self._integer(head) != self._head:
            raise ValueError("query relation changed within a forward")
        body = tuple(self._integer(r) for r in body)
        if self._index >= len(self.rule_bodies) or body != self.rule_bodies[self._index]:
            raise ValueError("grounding calls must follow the declared rule order")
        prefixes = self._plans[self._index]
        x = None
        start = 0
        for depth in range(len(prefixes) - 1, -1, -1):
            prefix = prefixes[depth]
            if prefix in self._cache:
                x = self._cache[prefix]
                self._cache.move_to_end(prefix)
                if self.collect_stats:
                    self._stats["hits"] += 1
                    self._stats["cache_lookup_hits"] += 1
                start = depth
                break
            if self.collect_stats:
                self._stats["misses"] += 1
                self._stats["cache_lookup_misses"] += 1
        if x is None:
            x = torch.nn.functional.one_hot(h, self.graph.entity_size).transpose(0, 1).unsqueeze(-1)
            if self.collect_stats:
                self._stats["identity_builds"] += 1
            self._observe(x)
            self._retain((), x)
        for depth in range(start, len(body)):
            relation = body[depth]
            prefix = prefixes[depth + 1]
            y = self.graph.propagate(x, relation, removed if relation == self._head else None)
            if self.collect_stats:
                self._stats["propagations"] += 1
                if prefix in self._seen:
                    self._stats["recomputes"] += 1
                self._seen.add(prefix)
            self._observe(x, y)
            x = y
            # Clear the extra local alias before the next propagation, so the
            # runtime lifetime matches the logical input/output accounting.
            del y
            self._retain(prefix, x)
        if self._release_last_use:
            for prefix in self._release_at[self._index]:
                if prefix in self._cache:
                    self._drop(prefix, "last_use_releases")
        self._observe(x)
        self._index += 1
        self._stats["rule_calls"] += 1
        return x.squeeze(-1).transpose(0, 1)

    @property
    def stats(self):
        result = dict(self._stats)
        result["current_cache_bytes"] = self._cache_bytes
        result["current_cache_states"] = len(self._cache)
        return result

    def finish(self, require_complete=True):
        if not self._active:
            raise RuntimeError("no active forward")
        result = self.stats
        result["complete"] = self._index == len(self.rule_bodies)
        result["final_cache_bytes"] = self._cache_bytes
        result["final_cache_states"] = len(self._cache)
        self._cache.clear()
        self._cache_meta.clear()
        self._cache_storages.clear()
        self._cache_bytes = 0
        self._h = self._removed = None
        self._active = False
        self._stats = result
        if require_complete and not result["complete"]:
            raise RuntimeError("forward ended before all declared rule bodies were executed")
        return dict(result)

    end = finish
