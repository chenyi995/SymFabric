"""Actual first-pass native64 count payload census; no cache or GPU execution."""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import numpy as np
import torch
from query_stream import ROOT, UPSTREAM, STAGE2, sha256_file, iter_trace_batches, upstream_data
sys.path[:0] = [str(STAGE2)]
from official_native_adapter import OfficialNativeBackend



def validate_spa():
    real_path, tiny_path = ROOT/'results/spa_native_real.json', ROOT/'results/spa_tiny_cpu.json'
    real, tiny = json.loads(real_path.read_text()), json.loads(tiny_path.read_text())
    if not (real.get('complete') is True and len(real.get('cases', [])) == 30
            and all(row.get('exact_terminal_rules_checked', 0) > 0 for row in real['cases'])
            and tiny.get('complete') is True and tiny.get('passed') is True):
        raise ValueError('SPA requires completed 30-case real exact-terminal validation and passed tiny validation')
    return {'real_path': str(real_path), 'real_sha256': sha256_file(real_path),
            'tiny_path': str(tiny_path), 'tiny_sha256': sha256_file(tiny_path),
            'real_exact_cases': len(real['cases']), 'tiny_passed': True}


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def census(dataset, args, global_deadline):
    started = time.perf_counter()
    output = ROOT / ('count_capacity_' + dataset + '.json')
    archive = ROOT / ('count_capacity_' + dataset + '.npz')
    source = ROOT / ('query_stream_' + dataset + '.npz')
    with np.load(source, allow_pickle=False) as arrays:
        count_batches = int(arrays['epoch_batch_offsets'][1])
        offsets = arrays['batch_offsets'][:count_batches + 1].copy()
        queries = arrays['queries'][:int(offsets[-1])].copy()
    count = len(queries)
    payload = np.full(count, -1, dtype=np.int64)
    nonzeros = payload.copy()
    candidates = payload.copy()
    ground_seconds = np.full(count_batches, np.nan)
    size_seconds = ground_seconds.copy()
    batch_backend = np.full(count_batches, -1, dtype=np.int8)
    batch_run_index = np.full(count_batches, -1, dtype=np.int32)
    relation_rows = {}
    manifest_path = ROOT / 'query_stream_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    dataset_manifest = next(d for d in manifest['datasets'] if d['dataset'] == dataset)
    manifest_sha256 = sha256_file(manifest_path)
    input_sha256 = {name: sha256_file(UPSTREAM/'data'/dataset/name) for name in dataset_manifest['input_sha256']}
    if input_sha256 != dataset_manifest['input_sha256']:
        raise ValueError('current graph/dictionary/rule inputs differ from frozen trace manifest')
    if sha256_file(source) != dataset_manifest['trace_sha256']:
        raise ValueError('trace archive differs from its frozen manifest')
    if dataset_manifest['native_overflow_guard']['native64_rejected_relations']:
        raise RuntimeError('native64 plan lacks complete overflow proof')
    result = dict(dataset=dataset, complete=False, censored=False,
        host=socket.gethostname(), job=os.getenv('SLURM_JOB_ID'),
        affinity=sorted(os.sched_getaffinity(0)), threads=args.threads,
        scope='Actual native64 grounding for every observed first-epoch training query in upstream trace order; exact prospective immutable CSR payload census. No GPU timing or cache admission measured.',
        trace_sha256=sha256_file(source), query_count=count, batch_count=count_batches,
        all_relation_native64_guard_proven=True,
        payload_formula='12 * rule_entity_nonzeros + 16 * distinct_destination_entities + 8 per query',
        payload_components='float32 values, int64 rule columns, int64 candidates, int64 CSR row offsets including one sentinel per query',
        persistent_payload_excludes=['graph/model', 'allocator reserve', 'Python metadata', 'count-building scratch', 'assembly/forward scratch'],
        grounding_scope='OfficialNativeBackend.batch including native result extraction/copy; complete relation rules including duplicate bodies as distinct columns',
        size_calculation_scope='concatenate sparse terminal flat IDs; bincount(flat//N); unique(flat) and candidate bincount; payload arithmetic',
        no_matrices_retained_between_batches=True, completed_batches=0, processed_queries=0,
        known_payload_bytes=0, native_grounding_seconds=0.0, size_calculation_seconds=0.0,
        dataset_soft_limit_seconds=args.max_dataset_seconds, total_soft_limit_seconds=args.max_total_seconds,
        relation_progress=relation_rows, output_npz=str(archive))
    prior_elapsed = 0.0
    resumed_batches = 0
    if args.resume and (output.exists() or archive.exists()):
        if not output.exists() or not archive.exists():
            raise ValueError('resume requires both JSON and NPZ outputs')
        previous = json.loads(output.read_text())
        if previous['trace_sha256'] != result['trace_sha256']:
            raise ValueError('resume trace hash changed')
        if previous.get('verified_input_sha256', input_sha256) != input_sha256:
            raise ValueError('resume graph/dictionary/rule input hashes changed')
        if previous.get('query_manifest_sha256', manifest_sha256) != manifest_sha256:
            raise ValueError('resume query manifest changed')
        if previous['dataset'] != dataset or not previous['all_relation_native64_guard_proven']:
            raise ValueError('resume dataset or overflow-proof mismatch')
        with np.load(archive, allow_pickle=False) as saved:
            if not np.array_equal(saved['queries'], queries) or not np.array_equal(saved['batch_offsets'], offsets):
                raise ValueError('resume query rows or batch boundaries changed')
            payload = saved['payload_bytes'].copy()
            nonzeros = saved['nonzeros'].copy()
            candidates = saved['candidates'].copy()
            ground_seconds = saved['native_grounding_seconds'].copy()
            size_seconds = saved['size_calculation_seconds'].copy()
            if 'batch_backend' in saved and 'batch_run_index' in saved:
                batch_backend = saved['batch_backend'].copy()
                batch_run_index = saved['batch_run_index'].copy()
            else:
                legacy_count = int(previous['completed_batches'])
                batch_backend[:legacy_count] = 0
                batch_run_index[:legacy_count] = 0
        if any(a.shape != (count,) for a in (payload, nonzeros, candidates)) or any(a.shape != (count_batches,) for a in (ground_seconds, size_seconds, batch_backend, batch_run_index)):
            raise ValueError('resume payload or timing array shape changed')
        resumed_batches = int(previous['completed_batches'])
        if not 0 <= resumed_batches <= count_batches:
            raise ValueError('resume completed batch count is out of range')
        stop = int(offsets[resumed_batches])
        if not (np.all(payload[:stop] >= 8) and np.all(payload[stop:] == -1)
                and np.all(nonzeros[:stop] >= 0) and np.all(candidates[:stop] >= 0)
                and np.array_equal(payload[:stop], 12*nonzeros[:stop]+16*candidates[:stop]+8)
                and np.all(np.isfinite(ground_seconds[:resumed_batches]))
                and np.all(np.isfinite(size_seconds[:resumed_batches]))
                and np.all(nonzeros[stop:] == -1) and np.all(candidates[stop:] == -1)
                and np.all(np.isnan(ground_seconds[resumed_batches:]))
                and np.all(np.isnan(size_seconds[resumed_batches:]))
                and np.all(np.isin(batch_backend[:resumed_batches], [0, 1]))
                and np.all(batch_run_index[:resumed_batches] >= 0)
                and np.all(batch_backend[resumed_batches:] == -1)
                and np.all(batch_run_index[resumed_batches:] == -1)):
            raise ValueError('resume arrays are not a consistent complete-batch prefix')
        if previous['complete']:
            print('Existing census complete; no grounding repeated', dataset, flush=True)
            return previous
        prior_elapsed = float(previous['elapsed_seconds'])
        legacy_run = dict(job=previous['job'], host=previous['host'], affinity=previous['affinity'],
            threads=previous['threads'], completed_batches=previous['completed_batches'],
            elapsed_seconds=previous['elapsed_seconds'],
            native_grounding_seconds=previous['native_grounding_seconds'],
            size_calculation_seconds=previous['size_calculation_seconds'],
            backend_selection='stdmap', backend=previous.get('backend'),
            provenance_note='Initial run predated resume-support metadata; values preserved from its final JSON. Original run did not record input hashes. Its graph/rule identity is anchored to the original frozen query manifest, whose current input files are now verified before resuming.',
            legacy_input_identity_anchor=dict(query_manifest_sha256=manifest_sha256, input_sha256=input_sha256))
        result.update(previous)
        result['runs'] = list(previous.get('runs', [legacy_run]))
        result.update(host=socket.gethostname(), job=os.getenv('SLURM_JOB_ID'),
            affinity=sorted(os.sched_getaffinity(0)), threads=args.threads,
            complete=False, censored=False, dataset_soft_limit_seconds=args.max_dataset_seconds,
            total_soft_limit_seconds=args.max_total_seconds)
        for stale in ('archive_sha256', 'censor_reason', 'error', 'policy_replay'):
            result.pop(stale, None)
        relation_rows = result['relation_progress']
    else:
        result['runs'] = []
    result['grounding_timing_warning'] = 'Cumulative observed execution across the recorded runs/hosts/backends, not a single-platform benchmark or backend speedup. Per-batch backend and run IDs permit separation.'
    result['batch_backend_codes'] = {'-1': 'unprocessed', '0': 'stdmap', '1': 'spa'}
    result['query_manifest_sha256'] = manifest_sha256
    result['verified_input_sha256'] = input_sha256
    initial_ground_seconds = float(np.nansum(ground_seconds))
    initial_size_seconds = float(np.nansum(size_seconds))
    result['runs'].append(dict(job=os.getenv('SLURM_JOB_ID'), host=socket.gethostname(),
        affinity=sorted(os.sched_getaffinity(0)), threads=args.threads, backend_selection=args.backend,
        resumed_after_batch=resumed_batches, resumed_after_query=int(offsets[resumed_batches]),
        trace_sha256=result['trace_sha256'], query_manifest_sha256=manifest_sha256,
        verified_input_sha256=input_sha256, script_sha256=sha256_file(Path(__file__)),
        native_adapter_sha256=sha256_file(STAGE2/'official_native_adapter.py')))
    if args.backend == 'spa':
        result['runs'][-1]['spa_validation'] = validate_spa()
        result['runs'][-1]['spa_source_sha256'] = {name: sha256_file(ROOT/name) for name in ('spa_native_adapter.py', 'spa_native_adapter.cpp')}
    current_run_index = len(result['runs'])-1
    def save():
        known = payload >= 0
        run_elapsed = time.perf_counter()-started
        result['runs'][-1].update(elapsed_seconds=run_elapsed,
            completed_batches=result['completed_batches']-resumed_batches,
            native_grounding_seconds=float(np.nansum(ground_seconds))-initial_ground_seconds,
            size_calculation_seconds=float(np.nansum(size_seconds))-initial_size_seconds)
        result.update(elapsed_seconds=prior_elapsed+run_elapsed,
            current_run_elapsed_seconds=run_elapsed,
            known_payload_bytes=int(payload[known].sum()),
            known_rule_entity_nonzeros=int(nonzeros[known].sum()),
            known_distinct_candidate_entities=int(candidates[known].sum()),
            processed_queries=int(known.sum()),
            complete=bool(known.all()),
            full_cache_payload_bytes=int(payload.sum()) if known.all() else None,
            full_cache_payload_lower_bound_bytes=int(payload[known].sum()),
            native_grounding_seconds=float(np.nansum(ground_seconds)),
            size_calculation_seconds=float(np.nansum(size_seconds)))
        temporary = archive.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, queries=queries, batch_offsets=offsets,
                payload_bytes=payload, nonzeros=nonzeros, candidates=candidates,
                native_grounding_seconds=ground_seconds, size_calculation_seconds=size_seconds,
                batch_backend=batch_backend, batch_run_index=batch_run_index)
        temporary.replace(archive)
        save_json(output, result)
    save()
    graph = backend = None
    try:
        graph = upstream_data().KnowledgeGraph(str(UPSTREAM / 'data' / dataset))
        backend_class = OfficialNativeBackend
        if args.backend == 'spa':
            from spa_native_adapter import SPANativeBackend
            backend_class = SPANativeBackend
        backend = backend_class(graph, threads=args.threads, bits=64)
        result['backend'] = backend.metadata
        result['runs'][-1]['backend'] = backend.metadata
        setup_seconds = time.perf_counter()-started
        result['runs'][-1]['graph_and_backend_setup_seconds'] = setup_seconds
        result['graph_and_backend_setup_seconds'] = result.get('graph_and_backend_setup_seconds', 0.0) + setup_seconds
        plans = [[] for _ in range(graph.relation_size)]
        with (UPSTREAM / 'data' / dataset / 'rnnlogic_rules.txt').open() as stream:
            for line in stream:
                head, *body = map(int, line.split())
                plans[head].append(tuple(body))
        prepared = {}
        deadline = min(started + args.max_dataset_seconds, global_deadline)
        last_save = time.perf_counter()
        for item in iter_trace_batches(source, epoch=0):
            if item['epoch_batch_index'] < resumed_batches:
                continue
            if time.perf_counter() >= deadline:
                result.update(censored=True, censor_reason='declared soft time limit reached before next batch')
                break
            index = item['epoch_batch_index']
            q = item['queries']
            relation = int(q[0, 1])
            if relation not in prepared:
                prepared[relation] = backend.prepare_rules(plans[relation])
            h = torch.from_numpy(q[:, 0].copy())
            removed = torch.from_numpy(q[:, 3].copy())
            before = time.perf_counter()
            terminals = backend.batch(prepared[relation], h, relation, removed)
            ground_seconds[index] = time.perf_counter() - before
            before = time.perf_counter()
            flat = torch.cat([ids for ids, _ in terminals]) if terminals else torch.empty(0, dtype=torch.int64)
            nnz = torch.bincount(flat // graph.entity_size, minlength=len(q))
            unique = torch.unique(flat)
            active = torch.bincount(unique // graph.entity_size, minlength=len(q))
            sizes = 12 * nnz + 16 * active + 8
            lo, hi = offsets[index:index + 2]
            payload[lo:hi], nonzeros[lo:hi], candidates[lo:hi] = sizes.numpy(), nnz.numpy(), active.numpy()
            size_seconds[index] = time.perf_counter() - before
            batch_backend[index] = 0 if args.backend == 'stdmap' else 1
            batch_run_index[index] = current_run_index
            result['completed_batches'] = index + 1
            row = relation_rows.setdefault(str(relation), dict(batches=0, queries=0, payload_bytes=0, native_grounding_seconds=0.0, size_calculation_seconds=0.0))
            row['batches'] += 1
            row['queries'] += len(q)
            row['payload_bytes'] += int(sizes.sum())
            row['native_grounding_seconds'] += ground_seconds[index]
            row['size_calculation_seconds'] += size_seconds[index]
            del terminals, flat, unique, nnz, active, sizes, h, removed
            if time.perf_counter() - last_save >= 30:
                save()
                print(json.dumps(dict(dataset=dataset, batches=result['completed_batches'], queries=result['processed_queries'], known_GiB=result['known_payload_bytes']/2**30, seconds=result['elapsed_seconds'])), flush=True)
                last_save = time.perf_counter()
        save()
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save()
        raise
    finally:
        if backend is not None:
            backend.close()
        del backend, graph
        gc.collect()
    result['archive_sha256'] = sha256_file(archive)
    if result['complete']:
        from capacity_policy_replay import replay_dataset
        result['policy_replay'] = replay_dataset(dataset, [4, 8, 16])
    else:
        result['policy_replay'] = dict(complete=False, reason='Full stream byte-policy replay requires payloads for every query; censored prefixes are not imputed.')
    save_json(output, result)
    print(json.dumps(dict(dataset=dataset, complete=result['complete'], batches=result['completed_batches'], known_GiB=result['known_payload_bytes']/2**30, grounding_seconds=result['native_grounding_seconds'], size_seconds=result['size_calculation_seconds'])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['wn18rr', 'FB15k-237'])
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--backend', choices=['stdmap', 'spa'], default='stdmap')
    parser.add_argument('--resume', action='store_true', help='Continue a trace-hash-validated completed-batch prefix without recomputing it')
    parser.add_argument('--max-dataset-seconds', type=float, default=1200)
    parser.add_argument('--max-total-seconds', type=float, default=1500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    deadline = time.perf_counter() + args.max_total_seconds
    for dataset in args.datasets:
        census(dataset, args, deadline)


if __name__ == '__main__':
    main()
