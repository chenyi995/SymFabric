"""Evidence-only report snippets; never extrapolate a sampled screen into a run."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(name):
 p=ROOT/'results'/name
 return json.loads(p.read_text()) if p.exists() else None

def lines():
 out=['','GPU实测细节','平台：Slurm node3，RTX A5000 24 GiB，16个分配且固定的EPYC 7352物理核；PyTorch 2.5.1+CUDA 12.4。CPU软件对照中的node6结果另列，不能与本段绝对耗时直接相除。']
 cold=read('gpu_cache_real.json')
 if cold and cold['complete']:
  out+=['冷构建对照：相同真实B32查询，1次预热/3次交错重复，下列均为计数+缓存构建总时间中位数，不包含神经网络前向。所有684个逐规则终端计数、缓存四数组以及模型输出/梯度检查通过。']
  labels={'native_cpu':'std::map+CPU整理','native_gpu_pack':'std::map+GPU整理','csr_gpu':'GPU计数+GPU整理','spa_cpu':'SPA+CPU整理','spa_gpu_pack':'SPA+GPU整理'}
  for c in cold['cases']:
   out.append(c['case']+'：'+'；'.join(labels[k]+f" {v['total_ms']['median_ms']:.3f} ms" for k,v in c['timing'].items())+'。')
  out+=['这两项关系样本表明构建路径应按工作负载选择；不能将CPU逐规则整理开销直接归因于硬件。']
 screen=read('allrelation_wn18rr_tile2048.json')
 if screen and screen['complete']:
  cases=screen['cases'];out+=[f"WN强软件筛查完成{len(cases)}个关系，每关系1个真实B32组，全部逐规则计数及输出检查通过；PNA分块2048。"]
  estimates={v:sum(c['epoch_batches_for_relation']*c['variants'][v]['forward']['median_ms']/1000 for c in cases) for v in ['upstream','csr','native','cached']}
  out.append('按各关系批次数加权的首组估计秒数：'+', '.join(f'{v}={t:.3f}' for v,t in estimates.items())+'；这是分层样本估计，不是完整轮次测量。')
 profile=read('pna_profile.json')
 if profile and profile['complete']:
  out+=['PNA局部profiling：在真实计数上比较原始完整聚合与分块512/2048，独立计时不启用profiler；CUDA轨迹只用于定位算子，不代替墙钟时间。']
  for c in profile['cases']:
   out.append(f"关系{c['relation']}，活跃候选{c['shape']['candidates']:,}："+', '.join(f"tile{k}={v['wall']['median_ms']:.3f} ms" for k,v in c['warm'].items() if 'wall' in v)+'。')
  out+=['原始PNA中间张量与masked_fill/mul/reduction仍有标准编译融合的优化空间；尚不能称为专用硬件不可替代的瓶颈。']
 training=read('pna_training_control.json')
 if training and training['complete']:
  out+=['','加强后的真实训练batch对照','WN关系12，两个预加载组重新混合的B32；2次预热、5次交错计时，每次恢复同一模型与Adam状态并检查更新。此处是单批训练，冷缓存构建、DataLoader/数据集构造和完整epoch/EM不在计时内。']
  out.append('中位数：'+', '.join(f"{k}={v['timing']['median_ms']:.3f}ms" for k,v in training['training_batch']['variants'].items())+'。')
  out+=['计时包含CPU onehot、查询/目标H2D、平滑、前向、loss、backward、Adam、zero_grad与同步。全部816个计数终端、11个参数梯度及5次Adam更新检查通过。先前tile128约107ms不是最强训练软件对照。']
 compiled=read('pna_compile.json')
 if compiled and not compiled['complete'] and 'error' in compiled:
  out+=['','编译对照的失败结果','在已安装PyTorch2.5.1+cu124中，直接torch.compile原FuncToNode.forward未通过第三个小图检查（单个活跃候选），最大score误差0.0293978，超过原阈值；未进入真实工作负载计时，未计入任何编译加速。','生成的单候选kernel在scatter_add更新前读取归一化sum/count，见results/pna_compile_failure_analysis.json和原生成代码。这仅诊断该版本/配置；不能视作新版本编译器或GPU无法正确优化PNA的证据。最新运行库兼容性仍需单独验证，原2.5.1环境未覆盖升级。']
 modern_path=ROOT/'runtime_modern/modern_runtime_summary.json'
 if modern_path.exists():
  modern=json.loads(modern_path.read_text())
  out+=['','新版运行库隔离对照',f"旧版{modern['legacy_runtime']}；新版{modern['modern_runtime']}；Slurm作业{modern['modern_job']}；当前状态{modern['status']}。"]
  out+=['新版环境安装和官方层CPU导入已通过。GPU对照固定使用8个输入，包括原失败单候选、精确min/max并列梯度以及真实WN12/8/7；直接调用未修改的官方FuncToNode/MLP。正确性以同版本eager为准，包含输出、全部学习参数梯度和三次连续Adam更新。', '这是独立聚合层对照；不包含grounding、缓存构建、打分头或完整模型训练，也不代表整个RNNLogic依赖栈已升级。']
  if modern['accepted_performance_available']:
   out.append('全部8个输入及编译检查通过；以下仅为该层驻留输入的前向中位数。首次编译开销另存于逐例结果。')
   for row in modern['cases']:
    if row.get('forward_timing'):
     t=row['forward_timing'];a=t['eager']['median_ms'];b=t['compiled']['median_ms']
     out.append(f"{row['fixture']}：eager {a:.6f}ms，compiled {b:.6f}ms，比值{a/b:.3f}x。")
  else:
   out.append('尚无通过全部检查的新版GPU性能结论；不接受部分通过输入的加速比。若仍为gpu_pending，驱动兼容性及旧编译问题在新版中是否存在均未验证。')
  out.append('详细结果与失败信息：runtime_modern/MODERN_RUNTIME_REPORT.txt、modern_runtime_summary.json及原始逐例JSON。')
 delta=read('deletion_delta_fb272.json')
 if delta and delta['complete']:
  out+=['','FB272精确基结果与删边增量的软件对照','本地针对性适配：209条不受删边影响的规则共用六个head的基结果，67条query关系仅出现在首步的规则以精确int64基结果减suffix计数；规则顺序、输出终端格式和原有图均保持一致。上游未原生实现这个适配。']
  out.append(f"全部{delta['queries']}查询、38个实际批次（含B26尾批）在三次配对遍历中进行{sum(p['terminal_checks'] for p in delta['passes']):,}个终端精确检查；全部通过。CPU node6、16物理核，同场比较SPA。")
  out.append(f"六个head基结果的int64 owning数组{delta['base']['owning_array_bytes']:,}字节，构建{delta['base']['total_seconds']*1000:.3f} ms；每次单轮比较都加计一次构建成本。")
  for p in delta['passes']:
   out.append(f"pass{p['pass']}：SPA {p['sum_measured_batch_seconds']['spa_masked']:.3f}s，base+delta含基结果 {p['base_delta_plus_once_base_seconds']:.3f}s，比值{p['speedup_including_once_base']:.3f}x；合并/完整终端物化{p['phase_seconds']['merge_and_materialization_seconds']:.3f}s。")
  out+=['结果证明当前完整query矩阵缓存并非唯一合法表示。这里只验证一个满足公式前提的关系；不是全关系/GPU/模型E2E加速，也不代表已经实现更一般的增量引擎。']
 out+=['','完整实际查询流回放','采用固定权重、官方emb/PNA、单rank实际训练查询顺序、B32含尾批；两种模式使用相同的epoch0/1查询序列。每批计时包含CPU输入装配、H2D、计数/缓存操作、模型前向与同步；不包含训练目标生成、loss、backward、Adam。检查点仅短拟合5步，不主张收敛精度。']
 streams={}
 for name,title in [('stream_wn_uncached.json','按关系选择的无跨批计数缓存'),('stream_wn_cache8g.json','8 GiB完整query计数缓存')]:
  d=read(name)
  if not d:out.append(title+'：尚未取得结果。');continue
  out.append(title+f"：complete={d['complete']}；已保存{len(d['epochs'])}轮，错误数{len(d['errors'])}。")
  for e in d['epochs']:
   counter=e['counter'];cache=e['cache_after'];fraction=counter.get('hits',0)/max(1,counter.get('hits',0)+counter.get('misses',0))
   out.append(f"  epoch{e['epoch']}：完整={e['trace_complete']}，{counter.get('queries',0):,}查询/{counter.get('batches',0):,}批；测量批次总时间{e['batch_forward_sum_seconds']:.3f} s，包含记录/显式oracle的轮次墙钟{e['wall_seconds_including_trace_access_jsonl_oracles']:.3f} s；命中率{fraction:.2%}；生产峰值GPU分配{e['production_peak_gpu_allocated_bytes']/2**30:.3f} GiB。")
   if cache:out.append(f"  轮末实际缓存分配{cache['persistent_unique_storage_bytes']:,}字节；phase_seconds={e['phase_seconds']}。")
  out.append(f"  周期性oracle检查{len(d['oracle_checks'])}次，全部通过={all(c['pass'] for c in d['oracle_checks'])}；首次构建/JIT在测量区间内，静态图/模型初始化及显式oracle在区间外。")
  if d['complete']:streams[name]=d
 if len(streams)==2:
  u=streams['stream_wn_uncached.json'];c=streams['stream_wn_cache8g.json']
  for a,b in zip(u['epochs'],c['epochs']):
   assert a['epoch']==b['epoch'] and a['counter']['queries']==b['counter']['queries']
   out.append(f"同一epoch{a['epoch']}查询流，无缓存/缓存批次总时间比={a['batch_forward_sum_seconds']/b['batch_forward_sum_seconds']:.3f}倍。单次完整遍历测量，没有重复整轮置信区间。")
  out.append('两轮合计无缓存/缓存总时间比='+f"{sum(e['batch_forward_sum_seconds'] for e in u['epochs'])/sum(e['batch_forward_sum_seconds'] for e in c['epochs']):.3f}倍；已经计入首轮冷构建。")
 return out
