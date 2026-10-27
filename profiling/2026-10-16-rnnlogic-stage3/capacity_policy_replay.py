"""Exact offline byte-policy replay on measured payloads; no GPU performance model."""
from collections import OrderedDict
import json
from pathlib import Path
import time
import numpy as np
from query_stream import ROOT, sha256_file


def simulate_block_lru(key_ids, batch_offsets, payload_by_key, budget_bytes, epoch_batch_offsets=None):
    """Match AllocationBlockLRU lookup/admit recency, pinning and block ownership."""
    blocks, key_to_block = OrderedDict(), {}
    used = peak = next_block = 0
    totals = dict(lookups=0, hits=0, misses=0, unique_misses=0,
        admitted_blocks=0, admitted_keys=0, evicted_blocks=0, evicted_keys=0,
        evicted_bytes=0, rejected_admissions=0, rejected_keys=0)
    epochs = []
    boundaries = {} if epoch_batch_offsets is None else {int(b): i for i, b in enumerate(epoch_batch_offsets[1:])}
    previous = totals.copy()
    for batch_index, (lo, hi) in enumerate(zip(batch_offsets[:-1], batch_offsets[1:])):
        pinned, missing, seen_missing = set(), [], set()
        hits = 0
        for key in key_ids[lo:hi]:
            key = int(key)
            block = key_to_block.get(key)
            if block is None:
                if key not in seen_missing:
                    missing.append(key)
                    seen_missing.add(key)
            else:
                hits += 1
                pinned.add(block)
                blocks.move_to_end(block)
        totals['lookups'] += 1
        totals['hits'] += hits
        totals['misses'] += int(hi-lo)-hits
        totals['unique_misses'] += len(missing)
        if missing:
            new_bytes = sum(int(payload_by_key[key]) for key in missing)
            pinned_bytes = sum(blocks[block][1] for block in pinned)
            if pinned_bytes + new_bytes > budget_bytes:
                totals['rejected_admissions'] += 1
                totals['rejected_keys'] += len(missing)
            else:
                while used + new_bytes > budget_bytes:
                    victim = next(block for block in blocks if block not in pinned)
                    old_keys, old_bytes = blocks.pop(victim)
                    for key in old_keys:
                        del key_to_block[key]
                    used -= old_bytes
                    totals['evicted_blocks'] += 1
                    totals['evicted_keys'] += len(old_keys)
                    totals['evicted_bytes'] += old_bytes
                blocks[next_block] = (tuple(missing), new_bytes)
                for key in missing:
                    key_to_block[key] = next_block
                next_block += 1
                used += new_bytes
                peak = max(peak, used)
                totals['admitted_blocks'] += 1
                totals['admitted_keys'] += len(missing)
        if batch_index + 1 in boundaries:
            epoch = {k: totals[k]-previous[k] for k in totals}
            epoch.update(epoch=boundaries[batch_index+1], resident_bytes=used,
                         resident_keys=len(key_to_block), resident_blocks=len(blocks))
            epochs.append(epoch)
            previous = totals.copy()
    return dict(policy='whole-preload-allocation LRU; current-request hit blocks pinned; one missing-key block per batch',
        budget_bytes=int(budget_bytes), **totals, hit_fraction=totals['hits']/len(key_ids),
        resident_bytes=used, peak_persistent_bytes=peak,
        resident_keys=len(key_to_block), resident_blocks=len(blocks), epochs=epochs)


def simulate_query_lru(key_ids, payload_by_key, budget_bytes, epoch_query_offsets=None):
    """Ideal independently owned per-query entries, consumed immediately in order."""
    cache = OrderedDict()
    used = peak = 0
    totals = dict(lookups=0, hits=0, misses=0, admitted_keys=0,
                  evicted_keys=0, evicted_bytes=0, rejected_admissions=0)
    epochs = []
    boundaries = {} if epoch_query_offsets is None else {int(q): i for i, q in enumerate(epoch_query_offsets[1:])}
    previous = totals.copy()
    for position, key in enumerate(key_ids):
        key = int(key)
        totals['lookups'] += 1
        if key in cache:
            totals['hits'] += 1
            cache.move_to_end(key)
        else:
            totals['misses'] += 1
            size = int(payload_by_key[key])
            if size > budget_bytes:
                totals['rejected_admissions'] += 1
            else:
                while used + size > budget_bytes:
                    _, old_size = cache.popitem(last=False)
                    used -= old_size
                    totals['evicted_keys'] += 1
                    totals['evicted_bytes'] += old_size
                cache[key] = size
                used += size
                peak = max(peak, used)
                totals['admitted_keys'] += 1
        if position + 1 in boundaries:
            epoch = {k: totals[k]-previous[k] for k in totals}
            epoch.update(epoch=boundaries[position+1], resident_bytes=used, resident_keys=len(cache))
            epochs.append(epoch)
            previous = totals.copy()
    return dict(policy='ideal independent per-query byte LRU; immediate serial query consumption, no batch pinning or shared storage',
        budget_bytes=int(budget_bytes), **totals, hit_fraction=totals['hits']/len(key_ids),
        resident_bytes=used, peak_persistent_bytes=peak, resident_keys=len(cache), epochs=epochs)


def replay_dataset(dataset, budgets_gib=(4, 8, 16)):
    started = time.perf_counter()
    capacity = ROOT / ('count_capacity_' + dataset + '.npz')
    trace = ROOT / ('query_stream_' + dataset + '.npz')
    with np.load(capacity, allow_pickle=False) as archive:
        query0 = archive['queries']
        measured = archive['payload_bytes']
    if np.any(measured < 0):
        raise ValueError('incomplete first-pass payload census cannot support full byte-policy replay')
    with np.load(trace, allow_pickle=False) as archive:
        queries = archive['queries']
        offsets = archive['batch_offsets']
        epoch_offsets = archive['epoch_batch_offsets']
    relation_count = int(queries[:, 1].max()) + 1
    counts = np.bincount(query0[:, 1], minlength=relation_count)
    bases = np.concatenate(([0], np.cumsum(counts)))
    ids0 = bases[query0[:, 1]] + query0[:, 3]
    if not np.array_equal(np.sort(ids0), np.arange(len(query0))):
        raise ValueError('first pass is not a complete unique full-key census')
    payload = np.empty(len(query0), dtype=np.int64)
    payload[ids0] = measured
    ids = bases[queries[:, 1]] + queries[:, 3]
    result = dict(dataset=dataset, complete=True, modeled_not_measured_gpu=True,
        scope='Exact offline cache-policy state replay over actual three-epoch query order and measured CPU native64 count payloads. No GPU execution, throughput, latency or cache-fill-cost inference.',
        full_cache_payload_bytes=int(payload.sum()), unique_query_keys=len(payload), accesses=len(ids),
        capacity_census_sha256=sha256_file(capacity), trace_sha256=sha256_file(trace),
        payload_namespace='fixed graph and complete ordered rule plan; relation/original-edge ID is a bijective compact encoding of the full (h,r,removed) training key',
        policy_match='AllocationBlockLRU lookup touches each hit block in request order even when admission is rejected. Current hit blocks are pinned. All unique misses form one allocation block. Reject before eviction if pinned+new exceeds budget; otherwise evict whole oldest unpinned blocks.',
        ideal_query_lru_caveat='Ideal means independently owned allocation only, not optimal replacement or an upper bound on hits. Whole-block hit touches refresh every key in that block, so block LRU can have a higher hit rate than independent per-query LRU.',
        budgets=[])
    for gib in budgets_gib:
        budget = int(gib * 2**30)
        result['budgets'].append(dict(budget_GiB=gib,
            block_lru=simulate_block_lru(ids, offsets, payload, budget, epoch_offsets),
            ideal_query_lru=simulate_query_lru(ids, payload, budget, offsets[epoch_offsets])))
    result['offline_replay_seconds'] = time.perf_counter()-started
    destination = ROOT / ('count_capacity_' + dataset + '_policy.json')
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(destination)
    return result
