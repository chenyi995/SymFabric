"""Exact one-rank upstream PredictorPlus query order, without dense targets.

The trace replays upstream dataset constructors and make_batches, then uses the
actual PyTorch DistributedSampler and DataLoader with an index-only view. No
grounding, neural inference, target assembly, or CUDA work is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / '2026-10-14-candidate-tests/kg'
UPSTREAM = OLD / 'upstream/RNNLogic'
STAGE2 = ROOT.parent / '2026-10-14-rnnlogic-sparse-stage2'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_data():
    """Import the unchanged upstream module, without importing a benchmark."""
    name = '_stage3_upstream_query_data'
    if name not in sys.modules:
        dependency = str(OLD / 'deps')
        if dependency not in sys.path:
            sys.path.append(dependency)
        spec = importlib.util.spec_from_file_location(name, UPSTREAM / 'src/data.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class ParsedGraph(SimpleNamespace):
    """Only graph fields read by upstream dataset constructors and __getitem__."""
    def encode_hr(self, head, relation):
        return relation * self.entity_size + head

    def encode_ht(self, head, tail):
        return tail * self.entity_size + head


def parse_graph(dataset):
    """Preserve file order and relation-local original edge IDs exactly."""
    directory = UPSTREAM / 'data' / dataset
    def dictionary(filename):
        with (directory / filename).open() as stream:
            rows = [line.rstrip('\n').split('\t') for line in stream]
        return {name: int(index) for index, name in rows}
    entities = dictionary('entities.dict')
    relations = dictionary('relations.dict')
    graph = ParsedGraph(entity_size=len(entities), relation_size=len(relations),
                        entity2id=entities, relation2id=relations,
                        id2relation={index: name for name, index in relations.items()},
                        relation2ht2index=[{} for _ in relations], hr2o={},
                        max_indegree=[0 for _ in relations])
    destinations = [Counter() for _ in relations]
    for split in ('train', 'valid', 'test'):
        facts = []
        with (directory / (split + '.txt')).open() as stream:
            for line in stream:
                head, relation, tail = line.strip().split('\t')
                head, relation, tail = entities[head], relations[relation], entities[tail]
                facts.append((head, relation, tail))
                if split == 'train':
                    lookup = graph.relation2ht2index[relation]
                    edge = graph.encode_ht(head, tail)
                    if edge in lookup:
                        raise ValueError('duplicate training triple rejected by upstream loader')
                    lookup[edge] = len(lookup)
                    graph.hr2o.setdefault(graph.encode_hr(head, relation), []).append(tail)
                    destinations[relation][tail] += 1
        setattr(graph, split + '_facts', facts)
    graph.max_indegree = [max(counts.values(), default=0) for counts in destinations]
    return graph


class _IndexOnlyDataset(Dataset):
    def __init__(self, original):
        self.original = original

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        return index


def build_training_trace(graph, batch_size=32, epochs=3, seed=1):
    """Return compact arrays for exact all-relation one-rank train invocations.

    Initialization reproduces run_predictorplus.py: train, valid, test dataset
    constructors, in that order. Each epoch then reproduces Trainer.train's
    make_batches and DistributedSampler.set_epoch(0). Python and CPU torch RNG
    states are restored on exit, so calling this helper does not perturb a run.
    """
    if batch_size <= 0 or epochs <= 0:
        raise ValueError('positive batch size and epoch count required')
    module = upstream_data()
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    query_chunks, batch_offsets, epoch_offsets, source_indices = [], [0], [0], []
    batch_hashes, sampler_hashes = [], []
    try:
        train_set = module.TrainDataset(graph, batch_size)
        # These consume Python shuffles before the first training call.
        module.ValidDataset(graph, batch_size)
        module.TestDataset(graph, batch_size)
        for _epoch in range(epochs):
            train_set.make_batches()
            sampler = DistributedSampler(train_set, num_replicas=1, rank=0)
            sampler.set_epoch(0)
            # The official loader has batch_size=1 and no custom collation.
            # Index-only fetching avoids its unnecessary dense [B,N] target.
            loader = DataLoader(_IndexOnlyDataset(train_set), 1, sampler=sampler,
                                num_workers=0)
            order = []
            for item in loader:
                index = int(item.item())
                order.append(index)
                facts = train_set.batches[index]
                queries = np.asarray([(h, r, t, graph.relation2ht2index[r][graph.encode_ht(h, t)])
                                      for h, r, t in facts], dtype='<i4')
                if not np.all(queries[:, 1] == queries[0, 1]):
                    raise AssertionError('upstream batch is not relation homogeneous')
                query_chunks.append(queries)
                batch_offsets.append(batch_offsets[-1] + len(queries))
                source_indices.append(index)
                batch_hashes.append(hashlib.sha256(queries.tobytes()).hexdigest())
            sampler_hashes.append(hashlib.sha256(np.asarray(order, dtype='<i8').tobytes()).hexdigest())
            epoch_offsets.append(len(source_indices))
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
    return {
        'queries': np.concatenate(query_chunks),
        'batch_offsets': np.asarray(batch_offsets, dtype='<i8'),
        'epoch_batch_offsets': np.asarray(epoch_offsets, dtype='<i8'),
        'source_batch_indices': np.asarray(source_indices, dtype='<i4'),
        'batch_query_sha256': np.asarray(batch_hashes, dtype='S64'),
        'sampler_order_sha256': np.asarray(sampler_hashes, dtype='S64'),
        'seed': np.asarray(seed, dtype='<i8'),
        'batch_size': np.asarray(batch_size, dtype='<i8'),
    }


def iter_trace_batches(trace, epoch=None):
    """Yield metadata and NumPy int64 [B,4] h/r/t/removed arrays.

    ``trace`` is an NPZ path or a mapping returned by build_training_trace.
    ``batch_index`` is global across epochs; ``epoch_batch_index`` starts at 0.
    Dense targets should be assembled by the caller inside its measured scope.
    """
    owned = not hasattr(trace, 'keys')
    arrays = np.load(trace, allow_pickle=False) if owned else trace
    try:
        epochs = arrays['epoch_batch_offsets']
        queries = arrays['queries']
        offsets = arrays['batch_offsets']
        sources = arrays['source_batch_indices']
        hashes = arrays['batch_query_sha256']
        selected = range(len(epochs) - 1) if epoch is None else [int(epoch)]
        for epoch_index in selected:
            if not 0 <= epoch_index < len(epochs) - 1:
                raise IndexError(epoch_index)
            for batch_index in range(int(epochs[epoch_index]), int(epochs[epoch_index + 1])):
                lo, hi = offsets[batch_index:batch_index + 2]
                yield {
                    'epoch': epoch_index, 'batch_index': batch_index,
                    'epoch_batch_index': batch_index - int(epochs[epoch_index]),
                    'source_batch_index': int(sources[batch_index]),
                    'queries': queries[lo:hi].astype(np.int64),
                    'query_sha256': bytes(hashes[batch_index]).decode(),
                }
    finally:
        if owned:
            arrays.close()


def _inversions_to_left(values):
    """Exact count of previous values greater than each value, using NumPy.

    Vectorized merge levels avoid a Python loop over millions of Fenwick updates.
    Extra space is linear; the input is a permutation, so ties never occur.
    """
    size = len(values)
    padded_size = 1 << (size - 1).bit_length()
    padded = np.full(padded_size, size, dtype=np.int64)
    padded[:size] = values
    answer = np.zeros(padded_size, dtype=np.int64)
    width = 1
    while width < padded_size:
        blocks = padded.reshape(-1, 2 * width)
        order = np.argsort(blocks, axis=1, kind='stable')
        left = order < width
        greater_left = width - np.cumsum(left, axis=1)
        greater_left[left] = 0
        original = order + np.arange(0, padded_size, 2 * width)[:, None]
        answer[original.ravel()] += greater_left.ravel()
        width *= 2
    return answer[:size]


def _summary(values):
    values = np.asarray(values)
    if not len(values):
        return {'count': 0}
    return {'count': int(len(values)), 'min': int(values.min()),
            'median': float(np.median(values)), 'mean': float(values.mean()),
            'p90': float(np.percentile(values, 90)), 'p99': float(np.percentile(values, 99)),
            'max': int(values.max())}


def add_reuse_census(trace, graph):
    """Add exact distinct-key stack distances and access gaps to trace arrays."""
    query = trace['queries']
    relation_counts = np.asarray([len(x) for x in graph.relation2ht2index], dtype=np.int64)
    bases = np.concatenate(([0], np.cumsum(relation_counts)))
    key_ids = bases[query[:, 1]] + query[:, 3]
    size = int(bases[-1])
    offsets = trace['batch_offsets'][trace['epoch_batch_offsets']]
    if not np.all(np.diff(offsets) == size):
        raise AssertionError('this census requires full one-rank epochs')
    gaps = np.full(len(query), -1, dtype='<i4')
    distances = np.full(len(query), -1, dtype='<i4')
    batch_gaps = np.full(len(query), -1, dtype='<i4')
    batch_ids = np.repeat(np.arange(len(trace['batch_offsets']) - 1), np.diff(trace['batch_offsets']))
    epochs = []
    previous_positions = None
    for epoch, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        ids = key_ids[lo:hi]
        if not np.array_equal(np.sort(ids), np.arange(size)):
            raise AssertionError('epoch did not cover every training key exactly once')
        positions = np.empty(size, dtype=np.int64)
        positions[ids] = np.arange(size)
        if previous_positions is not None:
            previous_order = key_ids[lo - size:lo]
            previous_before_and_current_after = _inversions_to_left(positions[previous_order])
            per_key_distance = np.empty(size, dtype=np.int64)
            per_key_distance[previous_order] = size - 1 - previous_before_and_current_after
            gaps[lo:hi] = size + np.arange(size) - previous_positions[ids] - 1
            distances[lo:hi] = per_key_distance[ids]
            batch_gaps[lo:hi] = batch_ids[lo:hi] - batch_ids[lo - size + previous_positions[ids]] - 1
        epochs.append({
            'epoch': epoch, 'queries': int(hi - lo), 'unique_full_query_keys': size,
            'coverage_fraction': 1.0, 'new_keys': size if epoch == 0 else 0,
            'within_epoch_full_key_reuses': 0,
            'reuses_of_prior_epoch_keys': 0 if epoch == 0 else size,
            'intervening_query_accesses': _summary(gaps[lo:hi][gaps[lo:hi] >= 0]),
            'intervening_distinct_full_keys_lru_stack_distance': _summary(distances[lo:hi][distances[lo:hi] >= 0]),
            'intervening_batches': _summary(batch_gaps[lo:hi][batch_gaps[lo:hi] >= 0]),
            'ordered_query_sha256': hashlib.sha256(query[lo:hi].tobytes()).hexdigest(),
        })
        previous_positions = positions
    trace['full_key_reuse_gap_queries'] = gaps
    trace['full_key_reuse_distance_distinct_keys'] = distances
    trace['full_key_reuse_gap_batches'] = batch_gaps
    finite_distances = distances[distances >= 0]
    capacities = sorted(set([0, 1, 32, size // 100, size // 10, size // 4, size // 2,
                             3 * size // 4, size]))
    return {
        'key_fields': ['head_entity', 'head_relation', 'original_removed_edge_id'],
        'namespace': 'fixed graph content and complete ordered rule set; neural weights excluded',
        'distance_definition': 'number of distinct full query keys strictly between consecutive accesses; -1=cold',
        'access_gap_definition': 'number of query accesses strictly between consecutive accesses',
        'unique_full_query_keys': size, 'total_accesses': int(len(query)),
        'epochs': epochs,
        'equal_entry_count_lru_capacity_reference': [
            {'capacity_entries': capacity,
             'hits_over_all_epochs': int(np.count_nonzero(finite_distances < capacity)),
             'misses_over_all_epochs': int(len(query) - np.count_nonzero(finite_distances < capacity))}
            for capacity in capacities],
        'lru_reference_limitation': 'Exact key-count LRU only, not byte capacity or performance. Entries have different payload sizes. Cache construction, misses, copies, lookup, assembly, eviction and learned-feature recomputation remain real work.',
    }


def rule_and_bound_audit(dataset, graph, trace):
    rules = [[] for _ in range(graph.relation_size)]
    path = UPSTREAM / 'data' / dataset / 'rnnlogic_rules.txt'
    with path.open() as stream:
        for index, line in enumerate(stream):
            row = list(map(int, line.split()))
            rules[row[0]].append((index, tuple(row[1:])))
    first_epoch_batches = int(trace['epoch_batch_offsets'][1])
    first_offsets = trace['batch_offsets'][:first_epoch_batches + 1]
    first_relations = trace['queries'][first_offsets[:-1], 1]
    sizes = np.diff(first_offsets)
    rows, rejections = [], []
    for relation, entries in enumerate(rules):
        max_bound, rejected32, rejected64 = 1, [], []
        max_length = 0
        for index, body in entries:
            bound = 1
            max_length = max(max_length, len(body))
            first32, first64 = None, None
            for depth, edge_relation in enumerate(body, 1):
                bound *= graph.max_indegree[edge_relation]
                max_bound = max(max_bound, bound)
                if bound > 2**31 - 1 and first32 is None:
                    first32 = depth
                if bound > 2**63 - 1 and first64 is None:
                    first64 = depth
            if first32 is not None or not body:
                rejected32.append(index)
            if first64 is not None:
                rejected64.append(index)
                rejections.append({'head_relation': relation, 'global_rule_index': index,
                                   'body': list(body), 'first_rejected_prefix_length': first64,
                                   'final_bound': str(bound)})
        relation_sizes = sizes[first_relations == relation]
        row = {
            'relation': relation, 'relation_name': graph.id2relation[relation],
            'train_facts': len(graph.relation2ht2index[relation]),
            'fraction_of_queries': len(graph.relation2ht2index[relation]) / len(graph.train_facts),
            'batches_per_epoch': int(len(relation_sizes)),
            'batch_sizes_histogram': {str(k): v for k, v in sorted(Counter(map(int, relation_sizes)).items())},
            'rule_rows': len(entries), 'distinct_rule_bodies': len(set(body for _, body in entries)),
            'empty_rule_rows': sum(not body for _, body in entries), 'max_rule_body_length': max_length,
            'query_rule_rows_per_epoch': len(entries) * len(graph.relation2ht2index[relation]),
            'max_graph_indegree': graph.max_indegree[relation],
            'native_guard_max_prefix_bound': str(max_bound),
            'native64_guard_pass': not rejected64, 'native64_rejected_rule_rows': len(rejected64),
            'native32_guard_pass': not rejected32, 'native32_rejected_rule_rows_including_empty': len(rejected32),
        }
        rows.append(row)
    return rows, {
        'method': 'Exact static reproduction of OfficialNativeBackend.prepare_rules: every rule prefix multiplies max destination indegree of each body relation, in unbounded Python integers.',
        'int64_limit': str(2**63 - 1), 'int32_limit': str(2**31 - 1),
        'native64_supported_relations': [row['relation'] for row in rows if row['native64_guard_pass']],
        'native64_rejected_relations': [row['relation'] for row in rows if not row['native64_guard_pass']],
        'native64_rejected_rule_rows': rejections,
        'meaning': 'Pass proves all-query intermediate count safety under this sufficient bound. Rejection is failure of the conservative proof, not evidence of actual overflow. The current backend refuses the whole relation plan; do not run rejected plans or count them as supported.',
        'count_execution_performed': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['FB15k-237', 'wn18rr'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'complete': False,
        'scope': 'All-relation full one-rank training query stream for fixed-rule PredictorPlus; no grounding, target assembly, optimizer execution or full training performed.',
        'configuration': {'seed': args.seed, 'batch_size': args.batch_size, 'epochs': args.epochs,
                          'rank': 0, 'world_size': 1, 'sampler_seed': 0, 'sampler_epoch_each_call': 0,
                          'loader_batch_size': 1, 'num_workers': 0},
        'upstream_configuration_difference': 'Official supplied PredictorPlus configs use four GPUs. This trace explicitly uses one rank; seed=1 and batch_size=32 match the supplied configs by default.',
        'initialization_order': ['TrainDataset constructor (includes make_batches)', 'ValidDataset constructor', 'TestDataset constructor', 'each train invocation: make_batches; DistributedSampler; DataLoader; sampler.set_epoch(0)'],
        'loader_method': 'Actual DataLoader and DistributedSampler order over an index-only view of the official TrainDataset; original TrainDataset.batches supplies every fact. Index-only fetching intentionally skips dense target materialization.',
        'query_columns': ['h', 'r', 't', 'original_removed_edge_id'],
        'query_hash_encoding': 'SHA256 of row-major little-endian int32 [queries,4]; batch boundaries separately in int64 batch_offsets.',
        'versions': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__},
        'source_sha256': {str(path): sha256_file(path) for path in [
            UPSTREAM / 'src/data.py', UPSTREAM / 'src/trainer.py',
            UPSTREAM / 'src/run_predictorplus.py', STAGE2 / 'official_native_adapter.py']},
        'datasets': [],
    }
    previous = json.loads((STAGE2 / 'exact_cache_key_counts.json').read_text())
    previous = {row['dataset']: row for row in previous['datasets']}
    destination = args.output_dir / 'query_stream_manifest.json'
    for dataset in args.datasets:
        print('Parsing and tracing', dataset, flush=True)
        graph = parse_graph(dataset)
        trace = build_training_trace(graph, args.batch_size, args.epochs, args.seed)
        reuse = add_reuse_census(trace, graph)
        relations, bounds = rule_and_bound_audit(dataset, graph, trace)
        if reuse['unique_full_query_keys'] != previous[dataset]['total_train_unique_query_keys']:
            raise AssertionError('stage2 key census disagrees')
        old_relations = {r['relation']: r for r in previous[dataset]['per_relation']}
        for relation in relations:
            for field in ('train_facts', 'rule_rows', 'distinct_rule_bodies', 'empty_rule_rows'):
                if relation[field] != old_relations[relation['relation']][field]:
                    raise AssertionError('stage2 per-relation census disagrees: ' + field)
        path = args.output_dir / ('query_stream_' + dataset + '.npz')
        np.savez_compressed(path, **trace)
        row = {
            'dataset': dataset, 'entities': graph.entity_size, 'relations': graph.relation_size,
            'trace_path': str(path.resolve()), 'trace_sha256': sha256_file(path),
            'trace_bytes': path.stat().st_size,
            'batches_per_epoch': np.diff(trace['epoch_batch_offsets']).tolist(),
            'queries_per_epoch': np.diff(trace['batch_offsets'][trace['epoch_batch_offsets']]).tolist(),
            'ordered_query_sha256_all_epochs': hashlib.sha256(trace['queries'].tobytes()).hexdigest(),
            'input_sha256': {name: sha256_file(UPSTREAM / 'data' / dataset / name) for name in
                             ['entities.dict', 'relations.dict', 'train.txt', 'valid.txt', 'test.txt', 'rnnlogic_rules.txt']},
            'stage2_unique_key_census_match': True, 'stage2_all_relation_rule_census_match': True, 'reuse': reuse,
            'relation_census': relations, 'native_overflow_guard': bounds,
        }
        result['datasets'].append(row)
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'dataset': dataset, 'queries_per_epoch': row['queries_per_epoch'],
                          'batches_per_epoch': row['batches_per_epoch'],
                          'native64_rejected_relations': bounds['native64_rejected_relations'],
                          'trace_bytes': row['trace_bytes']}), flush=True)
        del graph, trace
    result['complete'] = True
    destination.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
