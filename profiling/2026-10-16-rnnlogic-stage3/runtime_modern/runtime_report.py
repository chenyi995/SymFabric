"""Small stdlib-only report; safe after any queued run or partial checkpoint."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(name):
    p=ROOT/name
    return json.loads(p.read_text()) if p.exists() else None

def write_report():
    legacy=read('legacy_eager.json');modern=read('modern_compile.json')
    status='gpu_pending'
    if legacy or modern: status='gpu_run_partial_or_running'
    if modern and modern.get('complete'):
        status='complete_pass' if modern.get('all_cases_passed') else 'complete_checks_failed'
    elif modern and modern.get('error'):status='gpu_run_failed'
    cases=[]
    for row in (modern or {}).get('cases',[]):
        cases.append({'fixture':row['fixture'],'passed':row.get('passed',False),
                      'compiler_correctness':row.get('compiler_correctness'),
                      'three_adam_steps':row.get('three_adam_steps'),
                      'cross_version_eager_diagnostic':row.get('cross_version_eager_diagnostic'),
                      'forward_timing':row.get('forward_timing'),
                      'first_compiled_forward_backward_seconds':row.get('first_compiled_forward_backward_seconds'),
                      'error':row.get('error'),'oom':row.get('oom',False)})
    summary={'updated_utc':datetime.now(timezone.utc).isoformat(),'status':status,
             'legacy_runtime':(legacy or {}).get('torch','2.5.1+cu124'),
             'modern_runtime':(modern or {}).get('torch','2.14.1+cu126'),
             'modern_job':(modern or {}).get('job','223341'),
             'modern_host':(modern or {}).get('host'),'modern_gpu':(modern or {}).get('gpu'),
             'modern_driver':(modern or {}).get('driver_version'),
             'modern_complete':bool((modern or {}).get('complete')),
             'all_compiler_checks_passed':(modern or {}).get('all_compiler_checks_passed',False),
             'accepted_performance_available':status=='complete_pass',
             'legacy_complete':bool((legacy or {}).get('complete')),
             'legacy_all_cases_passed':(legacy or {}).get('all_cases_passed'),
             'cuda_eager_smoke_passed':(modern or {}).get('cuda_eager_smoke_passed',False),
             'cases':cases,'error':(modern or {}).get('error'),
             'scope':'Standalone unchanged official FuncToNode/MLP with fixed exact tiny and actual WN relation12/8/7 inputs; not full RNNLogic/latest dependency stack or E2E.',
             'limits':['No inference that the PyTorch2.5.1 compiler failure persists in modern versions.',
                       'Same-runtime eager is the compiler oracle;2.14 changed clamp-boundary gradients.',
                       'Performance is accepted only after every required fixture/gradient/tie/Adam gate passes.',
                       'Forward timings exclude graph grounding, cache construction/transfers, scoring head and full-model training.',
                       'GPU driver compatibility is established only for the actual recorded allocated GPU/driver, not other hardware.']}
    tmp=ROOT/'modern_runtime_summary.tmp';tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(ROOT/'modern_runtime_summary.json')
    lines=['Modern PyTorch standalone aggregation control', '', 'Status: '+status,
           'Legacy runtime: '+summary['legacy_runtime'], 'Modern runtime: '+summary['modern_runtime'],
           'Modern job: '+str(summary['modern_job']),
           'Recorded host/GPU/driver: '+str(summary['modern_host'])+' / '+str(summary['modern_gpu'])+' / '+str(summary['modern_driver']),
           'Scope: '+summary['scope'], '',
           'Correctness: same-runtime eager versus compiled; output rtol2e-4, gradient rtol3e-4, atol2e-5; isolated tie values/embedding gradients exact; three Adam steps on tiny mixed-query layer+rule embeddings.',
           'Cross-version eager gradients are diagnostic because PyTorch2.14 changed clamp-boundary subgradient semantics.', '']
    if not modern:
        lines.append('No modern GPU result exists yet. Official cu126 wheel installation and CPU import passed; GPU execution awaits allocation.')
    else:
        for row in cases:
            line=row['fixture']+': '+('PASS' if row['passed'] else 'not passed / incomplete')
            if row['oom']:line+=' (out of memory)'
            lines.append(line)
            if status=='complete_pass' and row['forward_timing']:
                timing=row['forward_timing'];eager=timing['eager']['median_ms'];compiled=timing['compiled']['median_ms']
                lines.append('  forward median eager %.6f ms, compiled %.6f ms, within-runtime ratio %.4fx'%(eager,compiled,eager/compiled))
            if row['error']:lines.append('  '+row['error'].strip().splitlines()[-1])
        if summary['error']:lines.append('Run error: '+summary['error'].strip().splitlines()[-1])
    lines+=['','Interpretation limits:']+['- '+v for v in summary['limits']]
    lines+=['','Sources: version_feasibility.json, environment_import.json, legacy_eager.json, modern_compile.json, ../pna_runtime_fixtures/manifest.json, gpu_runs/JOBID/.']
    tmp=ROOT/'MODERN_RUNTIME_REPORT.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(ROOT/'MODERN_RUNTIME_REPORT.txt')
    return summary

if __name__=='__main__':
    print(json.dumps({k:v for k,v in write_report().items() if k in ['status','modern_job','modern_complete','all_compiler_checks_passed']}))
