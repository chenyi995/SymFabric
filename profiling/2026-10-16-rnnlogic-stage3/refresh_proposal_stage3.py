"""Refresh a bounded dated evidence block without rewriting older literature."""
from pathlib import Path
import json
from runtime_modern.runtime_report import write_report
write_report()
ROOT=Path(__file__).resolve().parent
P=ROOT.parent/'2026-10-14-rnnlogic-proposal/PROPOSAL_zh.txt'
s=P.read_text()
def read(name):
 p=ROOT/name
 return json.loads(p.read_text()) if p.exists() else None
start='【第三轮新增实测汇总】';end='【第三轮新增实测汇总结束】'
block=[start,'真实GPU平台为node3 RTX A5000、分配16个EPYC7352物理核。当前仍不报告硬件仿真speedup；下面区分CPU计数、GPU前向和完整训练。']
f=read('count_capacity_FB15k-237.json')
if f and f['complete']:
 block.append(f"FB全部544230查询、474关系计数容量实测完成：{f['known_payload_bytes']:,}字节（{f['known_payload_bytes']/2**30:.6f}GiB）。这是完整query物化CSR的容量，不是任何合法实现的存储下界；CPU累计计数时间跨std::map/SPA和续跑，不能视作共同平台加速比。")
cold=read('results/gpu_cache_real.json')
if cold and cold['complete']:
 for c in cold['cases']:
  t={k:v['total_ms']['median_ms'] for k,v in c['timing'].items()};best=min(t,key=t.get)
  block.append(f"相同B32冷计数+缓存构建，{c['case']}：最优所测路径{best}，{t[best]:.3f}ms；std::map+CPU整理{t['native_cpu']:.3f}ms。均不含神经前向；684个精确终端及输出/梯度检查全部通过。")
d=read('results/deletion_delta_fb272.json')
if d and d['complete']:
 ratios=[x['speedup_including_once_base'] for x in d['passes']]
 block.append(f"FB272基结果+删边delta已实现为本地有限适配并实测：全部1210查询/38实际批次，三次配对遍历31,464个规则终端精确一致。六个head基数组{d['base']['owning_array_bytes']:,}B（int64），构建{d['base']['total_seconds']*1000:.2f}ms；计入一次构建后比SPA计数快{min(ratios):.3f}–{max(ratios):.3f}x。剩余大头是完整终端合并/物化。没有声称全关系通用化、GPU或神经E2E收益。")
u=read('results/stream_wn_uncached.json');c=read('results/stream_wn_cache8g.json')
if u and c and u['complete'] and c['complete']:
 block.append('WN已完成两轮相同真实查询流的无缓存/8GiB缓存对照：每轮173670查询、5438批，含尾批，官方emb/PNA和分块2048。')
 for a,b in zip(u['epochs'],c['epochs']):
  block.append(f"epoch{a['epoch']}：无缓存{a['batch_forward_sum_seconds']:.3f}s，缓存{b['batch_forward_sum_seconds']:.3f}s；同轮比值{a['batch_forward_sum_seconds']/b['batch_forward_sum_seconds']:.3f}x。计时包含输入装配/H2D、未命中构建、换入换出与神经前向；显式oracle另计。")
 block.append('两轮合计比值'+f"{sum(e['batch_forward_sum_seconds'] for e in u['epochs'])/sum(e['batch_forward_sum_seconds'] for e in c['epochs']):.3f}x，首轮构建已计入。固定权重、短拟合检查点、单次整轮遍历；不等于完整训练、EM或收敛精度。")
else:block.append('WN完整GPU查询流结果尚在收集；抽样或完整标志为false的数据不作为完整轮次结果。')
t=read('results/pna_training_control.json')
if t and t['complete']:
 v=t['training_batch']['variants']
 block.append('更强的WN12真实B32训练batch：'+', '.join(f"{k}={a['timing']['median_ms']:.3f}ms" for k,a in v.items())+'；包括loss/backward/Adam，所有梯度和更新检查通过。是预构建缓存后的混合查询单批，冷构建、完整epoch/EM另计；之前tile128约107ms不再作为最强训练软件残余。')
k=read('results/pna_compile.json')
if k and not k['complete'] and k.get('error'):
 block.append('直接编译原PNA函数的PyTorch2.5.1对照失败：单个活跃候选小图score误差0.0293978，实际工作负载计时未执行。生成代码存在scatter归一化写入前读取；失败配置不计入加速，也不能推断最新编译器同样失败。须继续验证当前运行库或局部合法融合。')
m=read('runtime_modern/modern_runtime_summary.json')
if m:
 block.append(f"新版隔离对照：{m['legacy_runtime']}与{m['modern_runtime']}，GPU作业{m['modern_job']}，状态{m['status']}。固定8个微型及真实WN输入，使用未修改的官方聚合层；包含输出/梯度、并列值和三次Adam检查。只有全部通过才接受编译性能；范围不包括完整模型、grounding和缓存构建。细节见runtime_modern/MODERN_RUNTIME_REPORT.txt。")
 if not m['accepted_performance_available']:block.append('尚未取得全部正确性检查通过的新版GPU性能结果；不可把旧运行库的编译失败当作现代软件不可优化的证据。')
f=read('results/screen_fb_combined.json')
if f:block.append(f"FB全关系GPU抽样筛查：已保存{len(f['cases'])}/474个完整关系批次，complete={f['complete']}，逐例保留主机/分配来源；不是FB完整查询流。")
block+=['PNA真实残余主要是大中间张量、mask/mul/reduction，继续检查原函数编译融合；FB需将delta与GPU构建/神经消费放在同一平台继续比较。已有经典增量计算和软件局部融合的优化空间不能作为架构创新。', '数据、方法限制和运行状态以../2026-10-16-rnnlogic-stage3/REPORT_zh.txt及results/*.json为准。',end]
if start in s:s=s[:s.index(start)]+ '\n'.join(block)+s[s.index(end)+len(end):]
else:s=s.replace('新增必读的最接近软件工作：','\n'.join(block)+'\n\n新增必读的最接近软件工作：')
s=s.replace('更新日期：2026-10-16（第三轮CPU证据与软件缺口修订）','更新日期：2026-10-16（第三轮CPU/GPU实测与软件缺口修订）')
s=s.replace('这是待实现/测量的软件基线，不是当前开源库已经支持的功能。','这一针对FB272的本地软件基线已实现和实测（见下方新增汇总），并非上游库已支持的功能。')
s=s.replace('FB已观测部分即可严格证明完整query缓存超过16GiB；全量计数与准确字节策略回放继续执行，不能按这个前缀外推全量数字。详细进度以第三轮结果文件为准。','FB完整query缓存容量已实测为185.973791GiB；这是当前完整矩阵表示的代价，不能据此排除规则列或基结果共享。准确字节策略回放以第三轮结果文件为准。')
s=s.replace('当前研究优先级：WN先验证官方PNA在合法缓存后的剩余成本；FB先测更快CPU累加器、GPU构建/缓存和删除不变列/精确delta复用。','当前研究优先级：WN已有完整查询流缓存和加强训练batch实测，继续测新版编译器能消除的PNA剩余成本；FB已有CPU累加器和有限关系精确delta对照，继续完成全关系GPU筛查，并测GPU构建、delta与神经消费的联合成本。')
P.write_text(s)
print(str(P))
