"""Summarize completed evidence; pending jobs are never counted as measurements."""
from pathlib import Path
import hashlib,json,datetime,subprocess
ROOT=Path(__file__).resolve().parent

def read(path):
 p=ROOT/path
 return json.loads(p.read_text()) if p.exists() else None

def main():
 from runtime_modern.runtime_report import write_report
 write_report()
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 evidence={}
 for pattern in ('*.json','results/*.json','runtime_modern/*.json'):
  for p in ROOT.glob(pattern):
   if p.name=='manifest_stage3.json':continue
   evidence[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 sources={}
 for pattern in ('*.py','*.cpp','*.sh','queue/*.sh','done/*.sh','runtime_modern/*.py','runtime_modern/*.sh'):
  for p in ROOT.glob(pattern):sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 jobs=subprocess.run(['squeue','-u','cw636','-o','%.12i %.10T %.10M %.12l %R'],capture_output=True,text=True).stdout
 manifest={'updated_utc':now,'scope':'Stage3 live evidence manifest; individual complete flags determine valid scope','evidence':evidence,'current_sources_sha256':sources,'slurm_snapshot':jobs,'statuses':{p.stem:p.read_text().strip() for p in (ROOT/'done').glob('*.status')}}
 (ROOT/'manifest_stage3.json').write_text(json.dumps(manifest,indent=2))
 lines=['RNNLogic第三轮：真实查询顺序、全关系容量和官方PNA','更新时间（UTC）：'+now,'','判断边界','本轮区分四类证据：真实CPU计数；容量策略的离线回放；真实GPU执行；硬件时序仿真。后两类只有取得相应结果才报告。容量模型不等于GPU加速比。','固定图、固定发布规则、训练时逐查询删边；数据集和上游源码不变。所有新功能均为本地实验适配，不冒称上游已有。','本轮上游HEAD只读核验仍为c890b025a14d6294d93222388a449fda8a851009。','']
 trace=read('query_stream_manifest.json')
 if trace and trace['complete']:
  lines+=['实际执行顺序','使用官方TrainDataset/ValidDataset/TestDataset构造顺序和训练期make_batches，再经过DistributedSampler与DataLoader。声明采用单rank、B32、seed1，区别于发布配置四GPU。生成三个连续训练轮次；每轮查询与原始关系内删边ID一一对应。']
  for d in trace['datasets']:
   lines.append(f"{d['dataset']}：每轮{d['queries_per_epoch'][0]:,}条查询、{d['batches_per_epoch'][0]:,}批；native64所有关系的保守溢出界检查通过。")
  lines+=['每轮每个完整训练查询key只出现一次，因此冷缓存首轮命中率为零。后续轮次必须计入重排、容量和失效条件。','']
 lines+=['全关系计数存储实测','存储公式：每query为12×规则-实体非零数 +16×活跃候选实体数 +8字节。对应float32 values和int64 columns/candidates/CSR row pointers。计数在原有转换边界前仍为int64。','这里只统计计数缓存，不包含图、模型、Python元数据、临时工作区或CUDA保留内存。没有保存全部矩阵，逐批生成后累计逐key准确大小。']
 for ds in ('wn18rr','FB15k-237'):
  d=read(f'count_capacity_{ds}.json')
  if not d:lines.append(ds+'：未取得测量。');continue
  if d['complete']:
   lines.append(f"{ds}：已完成{d['processed_queries']:,}/{d['query_count']:,}条；完整计数缓存{d['known_payload_bytes']:,}字节（{d['known_payload_bytes']/2**30:.6f} GiB）。")
  else:
   lines.append(f"{ds}：目前只完成{d['processed_queries']:,}/{d['query_count']:,}条；已观测{d['known_payload_bytes']/2**30:.6f} GiB，是完整容量的下界，不能当作全量结果或外推值。")
  lines.append(f"观测部分native计数{d['native_grounding_seconds']:.3f}秒，大小统计{d['size_calculation_seconds']:.3f}秒。它们不是GPU填充成本，也不是神经模型E2E；若续跑更换主机/后端，累计执行时间不能当作同一平台的性能。")
  policy=read(f'count_capacity_{ds}_policy.json')
  if policy and policy['complete']:
   for b in policy['budgets']:
    fractions=[e['hits']/max(1,e['hits']+e['misses']) for e in b['block_lru']['epochs']]
    lines.append(f"  {b['budget_GiB']} GiB分配块LRU，三轮命中率："+' / '.join(f'{x:.2%}' for x in fractions)+'。')
 spa=read('results/spa_native_real.json')
 if spa and spa['complete']:
  lines+=['','新增标准CPU累加器对照','保持原有邻接结构、独立规则执行、逐查询删边和int64安全界，只以两个稠密计数缓冲区和touched列表替换std::map；终点排序保持相同输出合同。上游文件不变。']
  for ds in ('FB15k-237','wn18rr'):
   ratios=[x['spa_speedup'] for x in spa['cases'] if x['dataset']==ds]
   lines.append(f"{ds}所测{len(ratios)}个实际B32批次：计数阶段加速{min(ratios):.2f}–{max(ratios):.2f}倍；不是完整模型E2E或全关系加权收益。")
  lines+=['全部30个真实批次、10308个规则末端计数精确相等；另有空规则、重复路径、多次删边与大整数等小图检查。平台node6分配16物理CPU核，同场交错计时。','']
 lines+=['以上命中率来自真实顺序+逐key实测大小的离线策略回放，不是GPU吞吐测试。块LRU按底层共享分配整体释放，当前查询命中的块固定到本批完成；已与真实缓存管理器逐批对照。独立query LRU仅假定独立分配，不是命中率最优上界。','','真实GPU进度']
 for name,title in [('results/pna_tiny_gpu.json','PNA GPU语义检查'),('results/pna_official_wn18rr.json','官方WN emb/PNA实测'),('results/gpu_cache_tiny.json','GPU直接构建缓存正确性'),('results/gpu_cache_real.json','CPU/GPU冷构建对照'),('results/screen_wn.json','WN全22关系前向筛查'),('results/screen_fb_combined.json','FB全474关系前向筛查'),('results/allrelation_wn18rr_tile2048.json','WN强PNA分块2048全关系对照'),('results/pna_compile.json','原PNA函数torch.compile对照')]:
  d=read(name);lines.append(title+'：'+('尚无GPU结果。' if d is None else 'complete='+str(d.get('complete',False))+'；详见'+name+'。'))
 lines+=['全关系筛查使用每个关系实际loader顺序中的首个完整批；2次预热、3次交错重复。候选包括上游、GPU CSR+前缀共享、native CPU+GPU稀疏消费、显式计数缓存；WN保留官方PNA并比较候选分块。','前向模型使用短拟合检查点，不代表收敛或全测试集MRR。前向抽样、固定权重查询流及完整训练必须分别报告。','真实查询流回放包括查询装配/H2D、miss构建、入缓存和换出；完成状态与完整逐轮实测见下方。','','硬件立项门槛','WN全查询计数在8 GiB预算内可驻留，重复grounding首先应由软件复用处理。FB完整query缓存超过预算也不等于计数不可复用：不含query关系的规则不受删边影响，含该关系的规则可研究精确删边增量。参见DELETION_DELTA_AUDIT.txt（若已生成）。','硬件仿真须等待强软件、真实未命中成本和具体残余算子证据。既有Ramulator smoke只是接口验证，本轮没有据此推断CPU/GPU或专用加速器周期、功耗和speedup。','','可复现文件','query_stream_manifest.json / query_stream_*.npz：真实顺序、数据和源码哈希。','count_capacity_*.json/.npz：逐query计数payload与逐batch CPU计时。','count_capacity_*_policy.json：完整数据集的容量策略回放。','AUDIT_PLAN.txt：测量边界、语义陷阱、仿真门槛。','manifest_stage3.json：结果和当前源码哈希、调度状态。','运行时来源另见results/sources_*.json；当前源码哈希不替代各实验启动时快照。','','Slurm快照',jobs]
 from gpu_evidence_summary import lines as gpu_lines
 index=lines.index('硬件立项门槛');lines[index:index]=gpu_lines()+['']
 (ROOT/'REPORT_zh.txt').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'report':str(ROOT/'REPORT_zh.txt'),'evidence_files':len(evidence),'updated_utc':now}))
if __name__=='__main__':main()
