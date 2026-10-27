"""Explicit preloaded query-count cache, using real PyTorch CSR float32 MM.

This is a restricted software control, not a cache scheduler. Every forward key
must have been explicitly preloaded. Counts are independent of neural weights;
all rule encoding and neural processing are recomputed and remain differentiable.
Graph/rule changes require reset(namespace=..., rule_version=...).

Cache key: (namespace, relation, query_head, removed_edge_or_None, rule_version).
Each query stores compact entity ids and CSR [active_entities, local_rule_count]
float32 counts on the model device. Duplicate rules keep separate local columns
and original global parameter indices. No [batch*entities,hidden] pool is built.

preload_from_terminals accepts exact CPU int64 sparse terminals, one per scheduled
rule, using flat_index=query*entity_count+entity. Source semantic equivalence must
be checked externally. This method validates unique/nonzero entries and exact
int64 equality of duplicate/preexisting keys before float conversion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time

import torch


@dataclass
class _Entry:
    key: tuple
    candidates: torch.Tensor
    crow: torch.Tensor
    columns: torch.Tensor
    values: torch.Tensor
    csr: torch.Tensor
    n_candidates: int
    nnz: int
    integer_digest: str


class CachedSparsePredictor(torch.nn.Module):
    def __init__(self, model, grounder_factory=None, *, namespace, rule_version,
                 rule_order='original', collect_stats=False):
        super().__init__()
        if model.aggregator != 'sum':
            raise ValueError('CachedSparsePredictor supports sum only; PNA is unimplemented')
        if rule_order not in ('original', 'prefix'):
            raise ValueError('rule_order must be original or prefix')
        self.model = model
        self.grounder_factory = grounder_factory
        self.rule_order = rule_order
        self.collect_stats = bool(collect_stats)
        self.last_stats = None
        self.cache = {}
        self.reset(namespace=namespace, rule_version=rule_version)

    @property
    def namespace(self):
        return self._namespace

    @property
    def rule_version(self):
        return self._rule_version

    def reset(self, *, namespace, rule_version):
        """Invalidate every count before adopting the explicitly supplied version."""
        if not isinstance(namespace, str) or not namespace or not isinstance(rule_version, str) or not rule_version:
            raise ValueError('namespace and rule_version must be nonempty explicit strings')
        self.cache.clear()
        self._namespace = namespace
        self._rule_version = rule_version
        self.device = next(self.model.parameters()).device
        self.plans = {}
        for relation, entries in enumerate(self.model.relation2rules):
            plan = [(int(index), (int(head), tuple(map(int, body)))) for index, (head, body) in entries]
            if self.rule_order == 'prefix':
                plan.sort(key=lambda entry: entry[1][1])
            self.plans[relation] = tuple(plan)
        self.rule_fingerprint = hashlib.sha256(json.dumps(self.plans, sort_keys=True).encode()).hexdigest()
        self._rule_indices = {}
        self._controllers = {}
        self.last_stats = None

    def rules_for_relation(self, relation):
        """Immutable (global_id, (head, body)) plan required by terminal input."""
        return self.plans[int(relation)]

    def _check_version(self, namespace, rule_version):
        if namespace is not None and namespace != self.namespace:
            raise ValueError('namespace changed: call reset before preloading new counts')
        if rule_version is not None and rule_version != self.rule_version:
            raise ValueError('rule_version changed: call reset before preloading new counts')
        if next(self.model.parameters()).device != self.device:
            raise ValueError('model device changed: reset and preload the cache on its new device')

    def _key(self, relation, head, removed):
        return (self.namespace, int(relation), int(head), None if removed is None else int(removed), self.rule_version)

    @staticmethod
    def _cpu_queries(h, r, removed):
        if h.device.type != 'cpu' or r.device.type != 'cpu' or h.dtype != torch.int64 or r.dtype != torch.int64:
            raise ValueError('preload queries must be CPU int64')
        if h.ndim != 1 or r.shape != h.shape or not h.numel():
            raise ValueError('preload queries must be nonempty equally shaped 1D tensors')
        relation = int(r[0])
        if bool((r != relation).any()):
            raise ValueError('one preload call must use a single relation')
        if removed is not None and (removed.device.type != 'cpu' or removed.dtype != torch.int64 or removed.shape != h.shape):
            raise ValueError('preload removed edges must be CPU int64 matching query shape')
        return relation

    def preload(self, h_cpu, r_cpu, removed_cpu=None, *, namespace=None, rule_version=None):
        """Build the entire query batch once with the supplied exact grounder."""
        self._check_version(namespace, rule_version)
        relation = self._cpu_queries(h_cpu, r_cpu, removed_cpu)
        if self.grounder_factory is None:
            raise ValueError('preload requires grounder_factory; use preload_from_terminals for externally built counts')
        if relation not in self._controllers:
            self._controllers[relation] = self.grounder_factory(self.model.graph, [body for _, (_, body) in self.plans[relation]])
        controller = self._controllers[relation]
        start = time.perf_counter()
        controller.begin(h_cpu, relation, removed_cpu)
        complete = False
        try:
            terminals = [controller.grounding(h_cpu, head, body, removed_cpu) for _, (head, body) in self.plans[relation]]
            complete = True
        finally:
            try:
                grounder_stats = controller.finish(require_complete=complete)
            except TypeError:
                grounder_stats = controller.finish()
        count_build_seconds = time.perf_counter() - start
        report = self.preload_from_terminals(h_cpu, r_cpu, removed_cpu, terminals,
                                             namespace=namespace, rule_version=rule_version)
        report['count_build_seconds'] = count_build_seconds
        report['count_build_plus_preload_seconds'] = count_build_seconds + report['preload_seconds']
        report['grounder_stats'] = grounder_stats
        return report

    def preload_from_terminals(self, h_cpu, r_cpu, removed_cpu, terminals, *, namespace=None, rule_version=None):
        """Validate/convert/upload a batch of precomputed exact integer terminals.

        ``terminals`` follows rules_for_relation(), each pair is CPU int64
        (unique flat query/entity indices, strictly positive path counts).
        Count building is external and must be reported separately from this call.
        Existing/duplicate query keys must have identical exact int64 contents.
        All CPU preparation, verification and H2D transfers here are timed.
        """
        self._check_version(namespace, rule_version)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        relation = self._cpu_queries(h_cpu, r_cpu, removed_cpu)
        entities = self.model.graph.entity_size
        batch = h_cpu.numel()
        if bool(((h_cpu < 0) | (h_cpu >= entities)).any()):
            raise ValueError('query head outside graph')
        plan = self.plans[relation]
        if len(terminals) != len(plan):
            raise ValueError('terminal list does not match this relation rule plan')
        for index, value in terminals:
            if index.device.type != 'cpu' or value.device.type != 'cpu' or index.dtype != torch.int64 or value.dtype != torch.int64:
                raise ValueError('terminals must contain CPU int64 index/count tensors')
            if index.ndim != 1 or value.shape != index.shape:
                raise ValueError('terminal indices and counts must be matching 1D tensors')
            if index.numel():
                if int(index.min()) < 0 or int(index.max()) >= batch * entities or bool((value <= 0).any()):
                    raise ValueError('terminal indices out of range or counts not strictly positive')
                if torch.unique(index).numel() != index.numel():
                    raise ValueError('duplicate terminal index: aggregate paths in exact integer grounding first')
        validation_seconds = time.perf_counter() - started
        build_start = time.perf_counter()
        if terminals:
            flat = torch.cat([index for index, _ in terminals])
            local_rule = torch.cat([torch.full_like(index, slot) for slot, (index, _) in enumerate(terminals)])
            integer_values = torch.cat([value for _, value in terminals])
        else:
            flat = torch.empty(0, dtype=torch.int64)
            local_rule = torch.empty_like(flat)
            integer_values = torch.empty_like(flat)
        query = flat // entities
        entity = flat % entities
        order = torch.argsort(query, stable=True)
        sizes = torch.bincount(query, minlength=batch).tolist()
        entity = entity[order]
        local_rule = local_rule[order]
        integer_values = integer_values[order]
        h_list = h_cpu.tolist()
        removed_list = [None] * batch if removed_cpu is None else removed_cpu.tolist()
        pending = {}
        seen = {}
        cursor = 0
        for slot, size in enumerate(sizes):
            key = self._key(relation, h_list[slot], removed_list[slot])
            stop = cursor + size
            candidates, rows = torch.unique(entity[cursor:stop], sorted=True, return_inverse=True)
            coordinates = torch.stack((rows, local_rule[cursor:stop]), dim=0)
            int_coo = torch.sparse_coo_tensor(coordinates, integer_values[cursor:stop],
                                             (candidates.numel(), len(plan)), dtype=torch.int64).coalesce()
            int_csr = int_coo.to_sparse_csr()
            crow = int_csr.crow_indices()
            columns = int_csr.col_indices()
            exact_values = int_csr.values()
            digest = hashlib.sha256()
            for tensor in (candidates, crow, columns, exact_values):
                digest.update(tensor.contiguous().numpy().tobytes())
            integer_digest = digest.hexdigest()
            if key in seen and seen[key] != integer_digest:
                raise ValueError('duplicate query key has different exact integer counts')
            seen[key] = integer_digest
            existing = self.cache.get(key)
            if existing is not None:
                if existing.integer_digest != integer_digest:
                    raise ValueError('preloaded key changed counts: reset namespace/rule_version explicitly')
            elif key not in pending:
                pending[key] = (candidates, crow, columns, exact_values.float(), integer_digest)
            cursor = stop
        cpu_build_seconds = time.perf_counter() - build_start
        # Concatenate each storage kind once and transfer four block tensors.
        # Each cache entry owns views into that preload's shared backing arrays.
        pack_start = time.perf_counter()
        entries = list(pending.items())
        if entries:
            packed_candidates = torch.cat([values[0] for _, values in entries])
            packed_crow = torch.cat([values[1] for _, values in entries])
            packed_columns = torch.cat([values[2] for _, values in entries])
            packed_values = torch.cat([values[3] for _, values in entries])
        cpu_pack_seconds = time.perf_counter() - pack_start
        transfer_start = time.perf_counter()
        if entries:
            packed_candidates = packed_candidates.to(self.device)
            packed_crow = packed_crow.to(self.device)
            packed_columns = packed_columns.to(self.device)
            packed_values = packed_values.to(self.device)
            candidate_at = crow_at = nnz_at = 0
            for key, (candidate_cpu, crow_cpu, columns_cpu, values_cpu, digest) in entries:
                count, nnz = candidate_cpu.numel(), columns_cpu.numel()
                candidate = packed_candidates[candidate_at:candidate_at + count]
                crow = packed_crow[crow_at:crow_at + count + 1]
                columns = packed_columns[nnz_at:nnz_at + nnz]
                values = packed_values[nnz_at:nnz_at + nnz]
                csr = torch.sparse_csr_tensor(crow, columns, values, (count, len(plan)), device=self.device)
                self.cache[key] = _Entry(key, candidate, crow, columns, values, csr, count, nnz, digest)
                candidate_at += count
                crow_at += count + 1
                nnz_at += nnz
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        transfer_and_bind_seconds = time.perf_counter() - transfer_start
        return {'namespace': self.namespace, 'rule_version': self.rule_version,
                'rule_fingerprint': self.rule_fingerprint, 'relation': relation,
                'input_queries': batch, 'unique_input_keys': len(seen), 'new_keys': len(entries),
                'validation_seconds': validation_seconds, 'cpu_csr_build_seconds': cpu_build_seconds,
                'cpu_pack_seconds': cpu_pack_seconds, 'transfer_and_bind_seconds': transfer_and_bind_seconds,
                'preload_seconds': time.perf_counter() - started,
                'count_build_included': False, 'integer_duplicate_key_check': 'canonical exact-int64 CSR SHA256',
                **self.cache_statistics()}

    def cache_statistics(self):
        storages = {}
        payload = 0
        for entry in self.cache.values():
            for tensor in (entry.candidates, entry.crow, entry.columns, entry.values):
                payload += tensor.numel() * tensor.element_size()
                storage = tensor.untyped_storage()
                storages[(tensor.device.type, tensor.device.index, storage.data_ptr())] = storage.nbytes()
        return {'cached_query_keys': len(self.cache), 'persistent_device': str(self.device),
                'persistent_unique_storage_bytes': sum(storages.values()), 'persistent_tensor_payload_bytes': payload,
                'cached_nonzeros': sum(entry.nnz for entry in self.cache.values()),
                'cached_active_entities': sum(entry.n_candidates for entry in self.cache.values()),
                'storage_note': 'count-cache tensor backing storage only; excludes graph/model, Python metadata, allocator reserve and warm-forward workspace'}

    def forward(self, all_h, all_r, edges_to_remove=None):
        self.last_stats = None
        self._check_version(None, None)
        if all_h.device != self.device or all_r.device != self.device or all_h.dtype != torch.int64 or all_r.dtype != torch.int64:
            raise ValueError('queries must be int64 on the cache/model device')
        if all_h.ndim != 1 or all_r.shape != all_h.shape or not all_h.numel():
            raise ValueError('queries must be nonempty equally shaped 1D tensors')
        if edges_to_remove is not None and (edges_to_remove.device != self.device or edges_to_remove.dtype != torch.int64 or edges_to_remove.shape != all_h.shape):
            raise ValueError('removed edges must match query shape/device and int64 dtype')
        # Deliberate host key lookup, including the real D2H transfer/sync cost.
        heads = all_h.detach().cpu().tolist()
        relations = all_r.detach().cpu().tolist()
        removed = [None] * len(heads) if edges_to_remove is None else edges_to_remove.detach().cpu().tolist()
        relation = relations[0]
        if any(value != relation for value in relations):
            raise ValueError('a PredictorPlus batch must contain a single relation')
        entries = []
        for head, edge in zip(heads, removed):
            key = self._key(relation, head, edge)
            if key not in self.cache:
                raise KeyError(f'explicit preloaded-set miss: {key}; no grounding fallback is implemented')
            entries.append(self.cache[key])
        batch = len(heads)
        model = self.model
        entities = model.graph.entity_size
        total_candidates = sum(entry.n_candidates for entry in entries)
        if not total_candidates:
            if self.collect_stats:
                self.last_stats = {'batch': batch, 'relation': relation, 'csr_rows': 0, 'csr_nonzeros': 0,
                    'csr_payload_bytes': 0, 'candidate_features_bytes': 0,
                    'dense_full_entity_feature_pool_created': False, 'all_queries_preloaded': True,
                    'query_d2h_key_lookup_included': True, 'csr_concatenation_included': False, **self.cache_statistics()}
            mask = torch.zeros((batch, entities), dtype=torch.float32, device=self.device)
            if model.entity_feature == 'bias':
                return mask + model.bias.unsqueeze(0), (1 - mask).bool()
            if model.entity_feature == 'RotatE':
                return mask + model.RotatE(all_h, all_r), (1 - mask).bool()
            return mask - float('-inf'), mask.bool()
        crow_parts = [torch.zeros(1, dtype=torch.int64, device=self.device)]
        nnz_offset = 0
        for entry in entries:
            crow_parts.append(entry.crow[1:] + nnz_offset)
            nnz_offset += entry.nnz
        crow = torch.cat(crow_parts)
        columns = torch.cat([entry.columns for entry in entries])
        values = torch.cat([entry.values for entry in entries])
        candidates = torch.cat([entry.candidates + query * entities for query, entry in enumerate(entries)])
        plan = self.plans[relation]
        counts = torch.sparse_csr_tensor(crow, columns, values, (total_candidates, len(plan)), device=self.device)
        if relation not in self._rule_indices:
            self._rule_indices[relation] = torch.tensor([index for index, _ in plan], dtype=torch.int64, device=self.device)
        indices = self._rule_indices[relation]
        if model.rule_features.device != self.device:
            model.rule_features = model.rule_features.to(self.device)
        embeddings = model.rule_emb[indices] if model.type == 'emb' else model.encode_rules(model.rule_features[indices])
        features = torch.sparse.mm(counts, embeddings)
        output = torch.relu(model.rule_to_entity.layer_norm(model.rule_to_entity.add_model(features)))
        rel = model.relation_emb(all_r[0]).unsqueeze(0).expand(output.size(0), -1)
        output = model.score_model(torch.cat((output, rel), dim=-1)).squeeze(-1)
        score = torch.zeros(batch * entities, device=self.device)
        score.scatter_(0, candidates, output)
        score = score.view(batch, entities)
        if model.entity_feature == 'bias':
            score = score + model.bias.unsqueeze(0)
            mask = torch.ones_like(score, dtype=torch.bool)
        elif model.entity_feature == 'RotatE':
            score = score + model.RotatE(all_h, all_r)
            mask = torch.ones_like(score, dtype=torch.bool)
        else:
            mask = torch.zeros(batch * entities, dtype=torch.bool, device=self.device)
            mask.index_fill_(0, candidates, True)
            mask = mask.view(batch, entities)
            score = score.masked_fill(~mask, float('-inf'))
        if self.collect_stats:
            self.last_stats = {'batch': batch, 'relation': relation, 'csr_rows': total_candidates,
                'csr_nonzeros': nnz_offset, 'csr_payload_bytes': sum(t.numel() * t.element_size() for t in (crow, columns, values)),
                'candidate_features_bytes': features.numel() * features.element_size(),
                'dense_full_entity_feature_pool_created': False, 'all_queries_preloaded': True,
                'query_d2h_key_lookup_included': True, 'csr_concatenation_included': True,
                **self.cache_statistics()}
        return score, mask
