from pathlib import Path
import json, csv, hashlib, subprocess, datetime
p=Path(__file__).resolve().parent
up=p.parent/'2026-10-14-candidate-tests/kg/upstream/RNNLogic'
# Retain distinct timing scopes in separate exports.
rows=[]
d=json.loads((p/'train_step.json').read_text())
assert d['complete']
for c in d['cases']:
 for v,r in c['variants'].items():
  t=r['training_batch'];rows.append({'file':'train_step.json','dataset':c['dataset'],'relation':c['relation'],'batch':c['batch'],'model':'lstm_sum','variant':v,'median_ms':t['median_ms'],'min_ms':t['min_ms'],'max_ms':t['max_ms'],'samples_ms':json.dumps(t['samples_ms']),'all_steps_verified':r['all_steps_verified']})
with (p/'training_steps.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[]
d=json.loads((p/'cuda_graph.json').read_text())
assert d['complete']
for c in d['cases']:
 for g in c['query_groups']:
  for v,r in g['variants'].items():
   t=r['forward'];rows.append({'file':'cuda_graph.json','dataset':c['dataset'],'relation':c['relation'],'batch':c['batch'],'model':'lstm_sum','query_group':g['group_id'],'variant':v,'median_ms':t['median_ms'],'min_ms':t['min_ms'],'max_ms':t['max_ms'],'samples_ms':json.dumps(t['samples_ms']),'capture_persistent_allocated_bytes_common_to_case':c['capture']['persistent_allocated_delta_bytes'],'every_rule_int64_equal':g['every_rule_int64_equal_upstream_and_csr']})
with (p/'cuda_graph_forwards.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(p/'REPRODUCE.txt').write_text('''RNNLogic real-hardware baseline reproduction

Measured job: Slurm222516, node4; finished COMPLETED, exit0, elapsed47m47s.
The Slurm allocation is released. Do not attempt to attach new work to it.

Environment used
  /home/cw636/venvs/epic/bin/python (PyTorch2.5.1+cu124, Triton3.1.0)
  PYTHONPATH=/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-candidate-tests/kg/deps
  TRITON_CACHE_DIR=<this directory>/triton-cache
  Upstream directory: ../2026-10-14-candidate-tests/kg/upstream/RNNLogic
  Expected upstream HEAD: c890b025a14d6294d93222388a449fda8a851009

Run on a new Slurm compute allocation, not a login node. Request one GPU,
16 physical CPU cores, and sufficient RAM. Verify that allocation's abstract
Slurm CPU IDs against its actual Linux topology. AFFINITY_NOTES.txt records
why taskset and explicit OpenMP places were necessary for the measured run.
Set BENCH_CPU_LIST to the Linux CPU IDs assigned to the new allocation;
run_pinned.sh refuses to assume the old job's8-23 mapping for a different job.
Use OMP_NUM_THREADS=16, MKL_NUM_THREADS=16 and OMP_PROC_BIND=spread;
run_pinned.sh sets explicit OMP_PLACES and OPENBLAS_NUM_THREADS=1.

Actual executable commands are preserved verbatim in done/*.sh, and their
outputs in logs/*.log. They assume cwd is this experiment directory. Each
script names its output JSON. Copy the experiment directory or change output
paths before reproducing if the recorded evidence must be retained. Execute
GPU benchmark tasks sequentially; do not contend multiple benchmark jobs on
one device. The task worker is no longer required for reproduction.

Recommended measurement order (all already completed here)
  001 /012: GPU kernel unit and dynamic-edge correctness.
  013 /014 /042: thread affinity, topology and OpenMP runtime verification.
  020 /030: main GPU and batch-size forward comparisons.
  040 /140: CPU1/8/16 and16-thread PASSIVE waiting-policy comparison.
  050: exact real-graph intermediate/terminal integer comparisons.
  060: official WN emb/PNA configuration.
  080: bounded last-use/LRU original rule order.
  090: sampled real training batch, common model/Adam/RNG next-step checks.
  100: low-rule-count control.
  110: grounding CUDA Graph, two real query groups.
  120: corrected profiler diagnostic and prefix-order GPU comparison.
  121: prefix-order CPU comparison.

Important provenance and corrections
  010 gpu_smoke.json is harness development, excluded from formal tables.
  Only checkpoints whose names include officialloss_v2 are formal fit states;
  earlier loss experiments are not used. The model is briefly fitted, not a
  converged paper-quality checkpoint.
  Task070 initially failed profile postprocessing. Task120 successfully reran
  profiling. A later trace-category audit corrected the final JSON summary to
  count cat==kernel, excluding gpu_user_annotation ranges. Raw trace files are
  retained; phase_profile.invalid_device_annotation_classification.json is
  explicitly INVALID and must not be used as measured kernel work.
  allocation_metadata.json reflects the worker's initial main-thread affinity,
  not benchmark worker teams. Use explicit_affinity.json,
  slurm_mapping_verified.json and pinned_linux_cpus in the formal results.
  BENCHMARK_REVIEW.txt is an intermediate audit: its 'queued/unmeasured' claims
  about later tests are superseded by REPORT_zh.txt and completed JSON files.
  Final script hashes describe the final source snapshot. During this session
  scripts gained new experiments and profiler postprocessing fixes; they are
  not immutable per-job source snapshots. Use JSON config, job scripts and
  logs for each run's executed options; old runs are not claimed to have used
  every subsequently added option.

Scopes and statistics
  results.csv: resident PredictorPlus forwards (280 rows).
  training_steps.csv: sampled training batch including H2D/loss/backward/Adam;
  excludes data loading, epoch and EM. Each repeat restores the same state.
  cuda_graph_forwards.csv: resident forward with only grounding captured;
  setup/H2D excluded and capture allocations common to all variants in case.
  These files must not be pooled as one timing boundary.
  GPU5/CPU3 repeats are preliminary; p95 fields are interpolated values, not
  reliable tail-latency estimates. Raw samples remain in JSON/CSV.
  Prefix-cache logical budgets exclude external outputs/working tensors.
  No simulator, FPGA/ASIC, power, bandwidth saturation, full-epoch or complete
  test-set ranking/MRR result was produced by this experiment.

Regenerate generic forward export:
  python3 collect_results.py
Regenerate scientific figure (no GPU needed):
  /home/cw636/venvs/epic/bin/python plot_results.py
Regenerate scope-specific CSV, reproduction notes, and final manifest:
  python3 finalize_results.py
The two scope-specific CSV files are direct projections of train_step.json
and cuda_graph.json and preserve all original repeat samples.
''')
# Final source/data/result evidence hashes. Never list credentials or environment secrets.
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
paths=set()
for pattern in ['*.py','*.cpp','*.sh','*.so','*.json','*.csv','*.txt','*.pdf','*.png','done/*.sh','done/*.status','checkpoints/*officialloss_v2*','traces/*.json']:
 paths.update(p.glob(pattern))
paths.discard(p/'manifest.json')
files=[]
for q in sorted(paths):
 if q.is_file():files.append({'path':str(q.relative_to(p)),'bytes':q.stat().st_size,'sha256':digest(q)})
data=[]
for ds in ['FB15k-237','wn18rr']:
 for q in sorted((up/'data'/ds).glob('*')):
  if q.is_file():data.append({'path':str(q.relative_to(up)),'bytes':q.stat().st_size,'sha256':digest(q)})
formal=['gpu_main.json','gpu_batches.json','cpu_main.json','cpu_passive.json','gpu_official_pna.json','gpu_bounded_lifetime.json','gpu_small_relation.json','gpu_prefix_schedule.json','cpu_prefix_schedule.json','train_step.json','cuda_graph.json','real_graph_correctness.json','phase_profile.json']
assert all(json.loads((p/n).read_text()).get('complete') for n in formal)
status=subprocess.check_output(['sacct','-j','222516','--format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,NodeList','-P'],text=True)
commit=subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip()
clean=subprocess.check_output(['git','-C',str(up),'status','--porcelain'],text=True).strip()
assert not clean
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'real CPU/GPU RNNLogic strong-software-baseline experiment; no simulator/accelerator measurements','upstream_path':str(up),'upstream_commit':commit,'upstream_worktree_clean':not clean,'remote_HEAD_verified_during_session':commit,'job_accounting':status,'actual_benchmark_linux_cpus':list(range(8,24)),'formal_complete_results':formal,'final_snapshot_not_per_job_immutable_sources':True,'excluded':['gpu_smoke.json','phase_profile.invalid_device_annotation_classification.json','checkpoints without officialloss_v2','earlier unpinned affinity timings'],'files':files,'upstream_data_and_rules':data}
(p/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'formal_result_files':len(formal),'hashed_files':len(files),'data_files':len(data),'training_rows':8,'graph_rows':12,'upstream_clean':not clean,'job_accounting':status},indent=2))
