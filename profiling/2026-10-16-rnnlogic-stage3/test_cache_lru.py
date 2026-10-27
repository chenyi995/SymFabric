"""Small CPU checks of real base-preloader storage ownership and LRU admission."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / '2026-10-14-rnnlogic-sparse-stage2'))
from cached_sparse_predictor import CachedSparsePredictor
from cache_lru import AllocationBlockLRU


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.aggregator = 'sum'
        self.graph = SimpleNamespace(entity_size=16)
        self.relation2rules = [[(0, (0, (0,))), (1, (0, (1,)))]]


def tensors(heads, removed=True):
    h = torch.tensor(heads, dtype=torch.int64)
    return h, torch.zeros_like(h), h.clone() if removed else None


def terms(rows):
    return [(torch.tensor(index, dtype=torch.int64), torch.tensor(value, dtype=torch.int64)) for index, value in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    adapter = CachedSparsePredictor(Model().to(args.device), namespace='test', rule_version='v1')
    manager = AllocationBlockLRU(adapter, 160)
    checks = []
    def check(label):
        audit = manager.verify_invariants()
        scan = CachedSparsePredictor.cache_statistics(adapter)
        fast = adapter.cache_statistics()
        for key in ('cached_query_keys', 'persistent_unique_storage_bytes', 'persistent_tensor_payload_bytes', 'cached_nonzeros', 'cached_active_entities'):
            assert scan[key] == fast[key], (key, scan, fast)
        checks.append(dict(label=label, **audit))
    def admit(heads, data, removed=True):
        q = tensors(heads, removed)
        context = manager.lookup(*q)
        ix = context['miss_indices']
        unique = (q[0][ix], q[1][ix], None if q[2] is None else q[2][ix])
        return context, manager.admit(context, *unique, terms(data))

    check('empty')
    _, (ok, report) = admit([0, 1], [([0, 17], [1, 2]), ([1, 17], [3, 4])])
    assert ok and report['actual_new_storage_bytes'] == 112
    assert len({adapter.cache[k].values.untyped_storage().data_ptr() for k in adapter.cache}) == 1
    check('two_keys_share_one_allocation_block')
    _, (ok, report) = admit([2], [([2], [1]), ([], [])])
    assert ok and manager.statistics()['persistent_unique_storage_bytes'] == 148
    check('second_block')

    context = manager.lookup(*tensors([0, 3]))
    assert context['hits'] == 1 and context['miss_indices'] == [1] and context['pinned_bytes'] == 112
    ok, report = manager.admit(context, *tensors([3]), terms([([3], [1]), ([], [])]))
    assert ok and report['evicted_keys'] == 1 and report['evicted_bytes'] == 36
    assert adapter._key(0, 0, 0) in adapter.cache and adapter._key(0, 1, 1) in adapter.cache
    check('pinned_block_preserved_and_lru_unpinned_block_evicted')

    context = manager.lookup(*tensors([0, 4]))
    before = dict(manager.statistics())
    block_order = tuple(manager.blocks)
    keys_before = set(adapter.cache)
    ok, report = manager.admit(context, *tensors([4]), terms([([4, 5], [1, 1]), ([], [])]))
    assert not ok and report['prospective_bytes'] == 64
    assert set(adapter.cache) == keys_before and tuple(manager.blocks) == block_order
    assert manager.statistics()['persistent_unique_storage_bytes'] == before['persistent_unique_storage_bytes']
    check('oversized_pinned_admission_rejected_before_eviction_or_upload')

    manager.lookup(*tensors([3]))  # Make the two-key block least recently used.
    context, (ok, report) = admit([5, 5], [([5], [2]), ([], [])])
    assert context['miss_indices'] == [0] and context['misses'] == 2 and context['unique_misses'] == 1
    assert ok and report['evicted_keys'] == 2 and report['evicted_bytes'] == 112
    assert adapter._key(0, 0, 0) not in adapter.cache and adapter._key(0, 1, 1) not in adapter.cache
    check('duplicate_misses_count_once_and_entire_shared_block_evicted')

    _, (ok, report) = admit([6], [([], []), ([], [])])
    assert ok and report['actual_new_storage_bytes'] == 8
    check('empty_result_still_charges_csr_row_pointer')
    _, (ok, _) = admit([5], [([5], [2]), ([], [])], removed=False)
    assert ok and adapter._key(0, 5, None) in adapter.cache and adapter._key(0, 5, 5) in adapter.cache
    check('valid_no_deletion_key_distinct_from_training_key')

    context = manager.lookup(*tensors([5, 5]))
    assert context['hits'] == 2 and not context['miss_indices']
    empty = torch.empty(0, dtype=torch.int64)
    ok, report = manager.admit(context, empty, empty, empty, [])
    assert ok and report['reason'] == 'all_hit'
    check('all_hit_duplicate_request')

    context = manager.lookup(*tensors([7]))
    manager.lookup(*tensors([5]))
    try:
        manager.admit(context, *tensors([7]), terms([([7], [1]), ([], [])]))
    except ValueError:
        pass
    else:
        raise AssertionError('stale context accepted')
    context = manager.lookup(*tensors([7]))
    try:
        manager.admit(context, *tensors([7]), terms([([7, 7], [1, 2]), ([], [])]))
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate sparse indices accepted')
    context = manager.lookup(*tensors([7]))
    try:
        manager.admit(context, *tensors([7]), terms([([7], [0]), ([], [])]))
    except ValueError:
        pass
    else:
        raise AssertionError('nonpositive sparse count accepted in default checked mode')
    check('stale_context_duplicate_terminal_and_nonpositive_count_rejected')

    manager.clear()
    check('clear_releases_all_owned_blocks')
    _, (ok, _) = admit([0], [([0], [1]), ([], [])])
    assert ok
    adapter.reset(namespace='test', rule_version='v2')
    try:
        manager.lookup(*tensors([0]))
    except RuntimeError:
        pass
    else:
        raise AssertionError('unacknowledged version reset accepted')
    manager.clear()
    assert manager.lookup(*tensors([0]))['miss_indices'] == [0]
    check('version_reset_requires_explicit_manager_clear')
    from cached_sparse_faststats_predictor import CachedSparseFastStatsPredictor
    from pna_predictor import CachedPNAPredictor
    for kind, cls in [('faststats_sum', CachedSparseFastStatsPredictor), ('pna', CachedPNAPredictor)]:
        model = Model().to(args.device)
        if kind == 'pna':
            model.aggregator = 'pna'
        wrapped = cls(model, namespace='compatibility', rule_version='v1')
        owned = AllocationBlockLRU(wrapped, 64)
        q = tensors([0])
        context = owned.lookup(*q)
        ok, report = owned.admit(context, *q, terms([([0], [1]), ([], [])]))
        assert ok and wrapped.cache_statistics()['persistent_unique_storage_bytes'] == 36
        checks.append(dict(label='inherited_' + kind + '_statistics_override', **owned.verify_invariants()))
        owned.clear()
    trusted_adapter = CachedSparsePredictor(Model().to(args.device), namespace='trusted', rule_version='v1')
    trusted = AllocationBlockLRU(trusted_adapter, 64, trusted_unique_positive_terminals=True)
    q = tensors([0])
    context = trusted.lookup(*q)
    ok, report = trusted.admit(context, *q, terms([([0], [1]), ([], [])]))
    assert ok and report['actual_new_storage_bytes'] == 36
    assert trusted.statistics()['trusted_unique_positive_terminals']
    checks.append(dict(label='trusted_producer_retains_exact_byte_accounting', **trusted.verify_invariants()))
    q = tensors([1])
    context = trusted.lookup(*q)
    wrong_dtype = [(torch.tensor([1], dtype=torch.int64), torch.tensor([1.0])), (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))]
    try:
        trusted.admit(context, *q, wrong_dtype)
    except TypeError:
        pass
    else:
        raise AssertionError('trusted mode skipped dtype contract')
    checks.append(dict(label='trusted_producer_still_checks_metadata', **trusted.verify_invariants()))
    trusted.clear()
    result = dict(complete=True, device=args.device, checks=checks, final=manager.statistics(),
                  scope='tiny real base-preloader storage accounting; not timing or full-workload capacity evidence')
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
