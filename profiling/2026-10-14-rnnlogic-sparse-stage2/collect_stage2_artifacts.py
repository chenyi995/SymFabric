"""Read existing evidence and refresh reproducibility artifacts; runs no benchmarks."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent
S1 = ROOT.parent / '2026-10-14-rnnlogic-strong-baselines'
UP = ROOT.parent / '2026-10-14-candidate-tests/kg/upstream/RNNLogic'
FORWARD = [
 'cpu_sparse_t1.json', 'cpu_sparse_t16.json',
 'cpu_sparse_hybrid_gpu_nn_v2.json', 'gpu_stream_screen_v2.json',
 'gpu_graph_stream_v2.json', 'hybrid_pool16.json', 'gbopt_hybrid16.json',
 'official_native_hybrid1.json', 'official_native_hybrid16.json',
 'native_sparse_mm16.json']
TRAIN = ['train_stream.json', 'train_native16.json']
EARLY = ['cpu_sparse_hybrid_gpu_nn.json', 'gpu_stream_screen.json', 'gpu_graph_stream.json']


def read(path):
    return json.loads(path.read_text())


def record(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def cmd(*args):
    return subprocess.check_output(args, text=True).strip()


def dataset_counts(dataset, selected_relation):
    base = UP / 'data' / dataset
    entities = {line.rstrip('\n').split('\t')[1]: int(line.split('\t')[0])
                for line in (base / 'entities.dict').read_text().splitlines()}
    relations = {line.rstrip('\n').split('\t')[1]: int(line.split('\t')[0])
                 for line in (base / 'relations.dict').read_text().splitlines()}
    ids_to_relation = {v: k for k, v in relations.items()}
    facts = {}
    for split in ['train', 'valid', 'test']:
        facts[split] = [tuple(line.split('\t')) for line in (base / f'{split}.txt').read_text().splitlines()]
        facts[split] = [(entities[h], relations[r], entities[t]) for h, r, t in facts[split]]
    edge_maps = collections.defaultdict(dict)
    train_keys = set()
    duplicates = 0
    for h, r, t in facts['train']:
        if (h, t) in edge_maps[r]:
            duplicates += 1
        else:
            edge_maps[r][h, t] = len(edge_maps[r])
        train_keys.add((h, r, edge_maps[r][h, t]))
    assert duplicates == 0, 'Would violate official Python loader duplicate assertion'
    valid_keys = {(h, r, None) for h, r, t in facts['valid']}
    test_keys = {(h, r, None) for h, r, t in facts['test']}
    rules = collections.defaultdict(list)
    for line in (base / 'rnnlogic_rules.txt').read_text().splitlines():
        vals = tuple(map(int, line.split()))
        rules[vals[0]].append(vals[1:])
    fact_counts = {split: collections.Counter(r for h, r, t in values) for split, values in facts.items()}
    key_counts = {split: collections.Counter(r for h, r, e in keys) for split, keys in [('train', train_keys), ('valid', valid_keys), ('test', test_keys)]}
    train_heads = collections.defaultdict(set)
    for h, r, t in facts['train']:
        train_heads[r].add(h)
    rows = []
    for r in sorted(ids_to_relation):
        rows.append({'relation': r, 'relation_name': ids_to_relation[r],
                     'train_facts': fact_counts['train'][r],
                     'train_unique_query_keys': key_counts['train'][r],
                     'train_unique_heads': len(train_heads[r]),
                     'valid_facts': fact_counts['valid'][r],
                     'valid_unique_query_keys': key_counts['valid'][r],
                     'test_facts': fact_counts['test'][r],
                     'test_unique_query_keys': key_counts['test'][r],
                     'rule_rows': len(rules[r]), 'distinct_rule_bodies': len(set(rules[r])),
                     'empty_rule_rows': sum(len(b) == 0 for b in rules[r])})
    return {'dataset': dataset, 'entity_count': len(entities), 'relation_count': len(relations),
            'total_train_facts': len(facts['train']), 'total_train_unique_query_keys': len(train_keys),
            'total_train_duplicate_triples': duplicates,
            'total_valid_facts': len(facts['valid']), 'total_valid_unique_query_keys': len(valid_keys),
            'total_test_facts': len(facts['test']), 'total_test_unique_query_keys': len(test_keys),
            'total_valid_test_union_unique_query_keys': len(valid_keys | test_keys),
            'selected_relation': selected_relation,
            'selected': next(x for x in rows if x['relation'] == selected_relation),
            'per_relation': rows}


def dump_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    counts = {'created_utc': now,
              'method': 'Exact parsing of the official files in their existing relation ID space; relation-local original edge IDs assigned in train.txt insertion order, matching src/data.py. No graph execution, synthetic inverse edges, timing or memory extrapolation.',
              'train_key': ['head_entity', 'head_relation', 'removed_original_edge_id'],
              'valid_test_key': ['head_entity', 'head_relation', None],
              'namespace_requirement': 'Cache namespace must additionally identify fixed graph and complete rule set; learned neural weights may change without invalidating integer counts.',
              'datasets': [dataset_counts('FB15k-237', 272), dataset_counts('wn18rr', 12)]}
    (ROOT / 'exact_cache_key_counts.json').write_text(json.dumps(counts, indent=2) + '\n')
    forward, training, accepted, exclusions = [], [], [], []
    all_results = {}
    for path in sorted(ROOT.glob('*.json')):
        if path.name in ['manifest_stage2.json', 'exact_cache_key_counts.json']:
            continue
        try:
            data = read(path)
        except (ValueError, OSError):
            continue
        all_results[path.name] = data
        if data.get('complete') is False or path.name in EARLY:
            failures = {'top_level_error': data.get('error'), 'failure': data.get('failure'), 'traceback': data.get('traceback'),
                        'case_failures': [{'dataset': c.get('dataset', c.get('case')), 'failures': c.get('failures', {})}
                                          for c in data.get('cases', []) if c.get('failures')]}
            exclusions.append({'file': path.name, 'complete': data.get('complete'),
                               'reason': 'superseded early non-v2 run' if path.name in EARLY else 'incomplete run; not a completed performance result',
                               'failures': failures, 'file_record': record(path)})
    for name in FORWARD + TRAIN:
        data = all_results.get(name)
        if not data or data.get('complete') is not True:
            continue
        accepted.append(name)
        for case in data['cases']:
            for variant, item in case['variants'].items():
                common = {'result_file': name, 'dataset': case['dataset'], 'relation': case['relation'],
                          'batch': case['batch'], 'split': case.get('split', 'train'), 'group': case.get('group', 0),
                          'threads': case.get('threads', data['config'].get('threads')),
                          'device': data['config'].get('device', 'cuda'), 'variant': variant,
                          'result_complete': data['complete'], 'warmup': data['config'].get('warmup'),
                          'declared_repeats': data['config'].get('repeats'),
                          'case_failures_json': json.dumps(case.get('failures', {}), sort_keys=True),
                          'scope': data.get('scope', '')}
                if name in FORWARD:
                    s = item.get('forward', {})
                    if not s:
                        continue
                    row = dict(common, raw_sample_count=len(s.get('samples_ms', [])), median_ms=s.get('median_ms'),
                               min_ms=s.get('min_ms'), max_ms=s.get('max_ms'),
                               samples_ms_json=json.dumps(s.get('samples_ms', [])),
                               mask_equal=item.get('mask_equal'), logit_max_abs_error=item.get('logit_max_abs_error'),
                               gradient_allclose=item.get('gradient', {}).get('allclose', 'not recorded'),
                               gradient_max_abs_error=item.get('gradient', {}).get('max_gradient_abs_error', ''),
                               variant_failure_json=json.dumps(case.get('failures', {}).get(variant, {}), sort_keys=True))
                    forward.append(row)
                else:
                    s = item['training_batch']
                    checks = item['warmup_checks'] + item['measured_steps']
                    row = dict(common, raw_sample_count=len(item['measured_steps']), median_ms=s['median_ms'],
                               min_ms=s.get('min_ms'), max_ms=s.get('max_ms'), samples_ms_json=json.dumps(s['samples_ms']),
                               all_steps_verified=item.get('all_steps_verified'),
                               verified_check_count=len(checks),
                               parameter_update_allclose=all(c.get('parameter_and_update_allclose') is True for c in checks),
                               adam_state_allclose=all(c.get('adam_state_allclose') is True for c in checks),
                               loss_max_abs_error=max(c.get('loss_abs_error', 0) for c in checks),
                               parameter_max_abs_error=max(c.get('max_parameter_abs_error', 0) for c in checks),
                               separate_memory_update_passed=item.get('training_memory', {}).get('update_check', {}).get('parameter_and_update_allclose'),
                               checkpoint_unchanged=case.get('checkpoint_unchanged'))
                    training.append(row)
    cache_rows = []
    cache_name = 'cached_sparse_real.json'
    cache_data = all_results.get(cache_name, {})
    if cache_data.get('complete') is True:
        accepted.append(cache_name)
        for case in cache_data['cases']:
            dataset, relation = case['case'].split(':')
            cache_info = case['cache']
            cache_common = {'cache_query_keys': cache_info['cached_query_keys'],
                            'cache_persistent_unique_storage_bytes': cache_info['persistent_unique_storage_bytes'],
                            'cache_cold_total_ms': case['cold_total_ms'],
                            'cache_scope': 'preloaded sampled keys; no online misses/eviction/full-dataset fit claim'}
            cache_rows.append(dict(result_file=cache_name, dataset=dataset, relation=int(relation), split=case['split'],
                                   **cache_common,
                                   query_D2H_ms=sum(x['query_D2H_ms'] for x in case['cold_build']),
                                   native_count_ms=sum(x['native_count_ms'] for x in case['cold_build']),
                                   cache_fill_ms=sum(x['cache_fill_ms'] for x in case['cold_build']),
                                   exact_terminal_tensor_checks=sum(x['correctness']['exact_terminal_tensor_checks'] for x in case['cold_build']),
                                   mixed_groups=len(case['mixed_groups'])))
            for group in case['mixed_groups']:
                common = {'result_file': cache_name, 'dataset': dataset, 'relation': int(relation),
                          'batch': group['batch'], 'split': case['split'], 'group': group['label'],
                          'threads': cache_data['config']['threads'], 'device': 'cuda', 'result_complete': True,
                          'warmup': cache_data['config']['warmup'], 'declared_repeats': cache_data['config']['repeats'],
                          'case_failures_json': '{}', 'scope': cache_data['scope'], **cache_common}
                for variant, timing in group['forward'].items():
                    grad = group.get('gradient', {}) if variant == 'cached_gpu_csr' else {}
                    forward.append(dict(common, variant=variant, raw_sample_count=len(timing['samples_ms']),
                                        median_ms=timing['median_ms'], min_ms=timing['min_ms'], max_ms=timing['max_ms'],
                                        samples_ms_json=json.dumps(timing['samples_ms']),
                                        mask_equal='passed via assert_outputs; not separate JSON field',
                                        logit_max_abs_error=group['logit_max_abs_error'][variant],
                                        gradient_allclose=grad.get('passed', 'not recorded'),
                                        gradient_max_abs_error=grad.get('max_gradient_abs_error', ''), variant_failure_json='{}'))
                for variant, item in group.get('training_batch', {}).get('variants', {}).items():
                    timing=item['timing']; checks=item['update_checks']
                    training.append(dict(common, variant=variant, raw_sample_count=len(timing['samples_ms']),
                                         median_ms=timing['median_ms'], min_ms=timing['min_ms'], max_ms=timing['max_ms'],
                                         samples_ms_json=json.dumps(timing['samples_ms']),
                                         all_steps_verified=item['all_updates_verified'], verified_check_count=len(checks),
                                         parameter_update_allclose=all(x['parameter_and_update_allclose'] for x in checks),
                                         adam_state_allclose=all(x['adam_state_allclose'] for x in checks),
                                         loss_max_abs_error=max(x['loss_abs_error'] for x in checks),
                                         parameter_max_abs_error=max(x['max_parameter_abs_error'] for x in checks),
                                         separate_memory_update_passed='not measured', checkpoint_unchanged='not recorded',
                                         cache_training_check_note='Five timed update checks recorded; two warmups asserted by code but records not retained. Mixed query batches, not TrainDataset batches.'))
    assert forward and training
    dump_csv(ROOT / 'forward_stage2.csv', forward, list(dict.fromkeys(k for row in forward for k in row)))
    dump_csv(ROOT / 'training_stage2.csv', training, list(dict.fromkeys(k for row in training for k in row)))
    if cache_rows:
        dump_csv(ROOT / 'cache_stage2.csv', cache_rows, list(cache_rows[0]))
    dump_csv(ROOT / 'excluded_results_stage2.csv',
             [{'file': x['file'], 'complete': x['complete'], 'reason': x['reason'],
               'failures_json': json.dumps(x['failures'], sort_keys=True)} for x in exclusions],
             ['file', 'complete', 'reason', 'failures_json'])
    capacity_rows = []
    for name, data in all_results.items():
        if not name.startswith('cache_capacity_') or data.get('complete') is not True:
            continue
        for c in data['cases']:
            capacity_rows.append(dict(result_file=name, case=c['case'], budget_mib=data['config']['budget_mib'],
                total_actual_keys=c['unique_train_key_count'], loaded_keys=c['loaded_keys'],
                fully_cached=c['fully_cached_selected_relation'], persistent_unique_storage_bytes=c['cache']['persistent_unique_storage_bytes'],
                native_count_seconds=c['native_count_seconds'], cache_fill_seconds=c['cache_fill_seconds'],
                extra_capacity_accounting_seconds=c['capacity_accounting_seconds'],
                cold_build_plus_fill_seconds=c['cold_build_plus_fill_seconds'],
                spotcheck_passed=c['correctness_spotcheck']['passed'], stop_reason=c['stop_reason']))
    if capacity_rows:
        dump_csv(ROOT / 'capacity_stage2.csv', capacity_rows, list(capacity_rows[0]))
    versions = collections.defaultdict(lambda: collections.defaultdict(list))
    def scan_versions(value, source, prefix=''):
        if isinstance(value, dict):
            for key, item in value.items():
                p = f'{prefix}.{key}' if prefix else key
                if key in ['torch', 'triton', 'numpy', 'scipy', 'python_graphblas', 'suitesparse_graphblas', 'suitesparse_library'] and not isinstance(item, dict):
                    versions[key][json.dumps(item, sort_keys=True)].append(f'{source}:{p}')
                scan_versions(item, source, p)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    scan_versions(item, source, f'{prefix}[{i}]')
    for name in accepted + ['sparse_gpu_correctness.json', 'sparse_cpu_correctness.json']:
        if name in all_results:
            scan_versions(all_results[name], name)
    for name in ['gpu_correctness.json']:
        scan_versions(read(S1 / name), f'stage1/{name}')
    stage1_names = ['run_bench.py', 'prefix_cache.py', 'cpu_backend.py', 'cpu_native.cpp',
                    'gpu_backend.py', 'bench_cuda_graph.py', 'bench_train_step.py']
    source_files = sorted(p for p in ROOT.iterdir() if p.suffix in ['.py', '.cpp', '.sh', '.c'])
    data_files = []
    for ds, rel in [('FB15k-237', 272), ('wn18rr', 12)]:
        data_files.extend(UP / 'data' / ds / n for n in ['entities.dict', 'relations.dict', 'train.txt', 'valid.txt', 'test.txt', 'rnnlogic_rules.txt'])
        data_files.append(S1 / 'checkpoints' / f'{ds}_{rel}_lstm_sum_officialloss_v2_fit5.pt')
    gpu_text = (ROOT / 'gpu_metadata.txt').read_text()
    driver = re.search(r'Driver Version\s*:\s*(\S+)', gpu_text)
    cuda_max = re.search(r'CUDA Version\s*:\s*(\S+)', gpu_text)
    statuses = []
    for p in sorted((ROOT / 'done').glob('*.status')):
        script = p.with_suffix('.sh')
        statuses.append({'status_file': p.name, 'exit_status': p.read_text().strip(),
                         'command': script.read_text().strip() if script.exists() else None})
    commits = {'path': str(UP), 'url': cmd('git', '-C', str(UP), 'remote', 'get-url', 'origin'),
               'commit': cmd('git', '-C', str(UP), 'rev-parse', 'HEAD'),
               'status_porcelain': cmd('git', '-C', str(UP), 'status', '--porcelain'),
               'tracked_diff': cmd('git', '-C', str(UP), 'diff', 'HEAD', '--'),
               'snapshot_observed_utc': now}
    manifest = {
        'created_utc': now, 'scope': 'Read-only snapshot of completed real CPU/GPU stage2 software evidence, with incomplete failures preserved; no accelerator performance result.',
        'collector': str(Path(__file__).resolve()),
        'upstream': commits,
        'environment_from_existing_evidence': {'allocation': read(ROOT / 'allocation.json'),
             'cpu_lscpu_text': (ROOT / 'lscpu.txt').read_text(),
             'gpu_names_from_complete_results': sorted({all_results[n].get('gpu') for n in accepted if all_results[n].get('gpu')}),
             'driver_version_from_nvidia_smi': driver.group(1) if driver else None,
             'driver_max_cuda_version_from_nvidia_smi': cuda_max.group(1) if cuda_max else None,
             'driver_cuda_note': 'nvidia-smi CUDA reports driver compatibility, not Torch build CUDA; Torch version +cu124 is independently recorded.',
             'hardware_evidence_files': [record(ROOT / n) for n in ['allocation.json', 'gpu_metadata.txt', 'lscpu.txt', 'topology.xml', 'runtime_team_pool16.json']],
             'benchmark_python_executable_from_launch_script': '/home/cw636/venvs/epic/bin/python'},
        'dependency_versions_from_results': {key: [{'version': json.loads(v), 'evidence_locations': locations} for v, locations in values.items()] for key, values in versions.items()},
        'dependency_version_limit': 'No fresh installation/import/version probe; only observed result metadata is asserted. SPARSE_CPU_NOTES.txt additionally documents installed packages, but unrecorded versions are not inferred as measured.',
        'current_stage2_root_sources': [record(p) for p in source_files],
        'imported_stage1_sources': [record(S1 / n) for n in stage1_names],
        'upstream_imported_sources': [record(p) for p in sorted((UP / 'src').glob('*.py'))] + [record(UP / 'miner' / n) for n in ['rnnlogic.cpp', 'rnnlogic.h']],
        'source_snapshot_note': 'Current file hashes are a snapshot, not a claim all past jobs ran current files. Existing launch-time source snapshots preserve Python content and hashes; C++ build identity is separately recorded in native backend metadata and library hashes.',
        'launch_time_source_snapshots': [record(p) for p in sorted((ROOT / 'source_snapshots').glob('*.json'))],
        'native_binaries': [record(p) for p in sorted(ROOT.glob('*.so'))] + [record(p) for p in sorted(S1.glob('cpu_native_*.so'))],
        'data_rules_and_checkpoints': [record(p) for p in data_files],
        'exact_key_counts': record(ROOT / 'exact_cache_key_counts.json'),
        'accepted_forward_training_results': [dict(record(ROOT / n), complete=True) for n in accepted],
        'other_result_records': [dict(record(ROOT / n), complete=d.get('complete'), passed=d.get('passed')) for n, d in all_results.items() if n not in accepted],
        'completed_capacity_results': [record(ROOT / n) for n,d in all_results.items() if n.startswith('cache_capacity_') and d.get('complete') is True],
        'capacity_table': dict(record(ROOT / 'capacity_stage2.csv'), rows=len(capacity_rows)) if capacity_rows else None,
        'reports_and_figures': [record(p) for p in sorted(ROOT.iterdir()) if p.suffix in ['.txt','.png','.pdf'] and p.is_file()],
        'completed_phase_diagnostics': [record(ROOT / n) for n in ['phase_stage2.json', 'phase_native16.json'] if all_results.get(n, {}).get('complete') is True],
        'excluded_performance_results': exclusions,
        'table_policy': 'Only explicitly selected complete forward/training JSON files; exclude superseded early non-v2 and incomplete runs. Profiler-only data stays separate. cached_sparse_real warm rows explicitly carry cold construction and persistent sampled-cache metadata.',
        'tables': [dict(record(ROOT / n), rows=count) for n, count in [('forward_stage2.csv', len(forward)), ('training_stage2.csv', len(training)), ('excluded_results_stage2.csv', len(exclusions))] + ([('cache_stage2.csv', len(cache_rows))] if cache_rows else [])],
        'job_step_statuses_preserved': statuses,
        'failed_step_log_evidence': [dict(record(p), final_lines=p.read_text(errors='replace').splitlines()[-30:]) for p in sorted((ROOT / 'logs').glob('*.log')) if any(x['exit_status'] != '0' and p.stem == Path(x['status_file']).stem for x in statuses)],
        'simulator': {'manifest': record(ROOT / 'simulator/manifest.json'), 'build_identity': read(ROOT / 'simulator/manifest.json'),
                      'scope': 'Official Ramulator2.1 build and synthetic/example DRAM smoke only; no RNNLogic workload or custom accelerator simulation.'},
        'limitations': ['Resident sampled PredictorPlus lstm/sum and fixed brief-fit checkpoints; no complete epoch/EM/convergence claim.',
                        'Forward/training configurations use five timing samples; capacity/cold-build censuses are single runs per implementation. No strong p95/tail-latency claim.',
                        'Forward and actual TrainDataset batches differ; do not subtract their timings to infer backward cost.',
                        'Logical query keys do not establish full-cache bytes or hit rate; no whole-dataset memory extrapolation.',
                        'GPU allocation increments exclude already resident backends/captures and CPU storage.',
                        'An included complete file can still contain variant failures; CSV preserves them rather than silently treating completion as correctness.']}
    (ROOT / 'manifest_stage2.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'forward_rows': len(forward), 'training_rows': len(training), 'excluded_files': [x['file'] for x in exclusions],
                      'datasets': [{k: v for k, v in d.items() if k != 'per_relation'} for d in counts['datasets']],
                      'upstream_clean': not commits['status_porcelain'] and not commits['tracked_diff']}, indent=2))


if __name__ == '__main__':
    main()
