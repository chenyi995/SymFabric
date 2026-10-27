import json,os,re,socket,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
p=Path(__file__).resolve().parent
job=os.environ['SLURM_JOB_ID']
s=subprocess.check_output(['scontrol','show','job','-dd',job],text=True)
abstract=re.search(r'CPU_IDs=([0-9,-]+)',s).group(1)
ids=set()
for term in abstract.split(','):
 v=list(map(int,term.split('-')));ids.update(range(v[0],v[-1]+1))
xml=Path('/var/spool/slurmd/hwloc_topo_whole.xml').read_text()
root=ET.fromstring(xml);mapping=[]
for socket_ in root.iter('object'):
 if socket_.get('type') not in ['Package','Socket']:continue
 for core in socket_.iter('object'):
  if core.get('type')!='Core':continue
  for pu in core.iter('object'):
   if pu.get('type')=='PU':mapping.append((int(pu.get('os_index')),socket_.get('os_index'),core.get('os_index')))
cpus=[];seen=set()
for i,(linux,pack,core) in enumerate(mapping):
 if i in ids and (pack,core) not in seen:cpus.append(linux);seen.add((pack,core))
assert len(cpus)>=int(os.environ['SLURM_CPUS_PER_TASK'])
cpus=cpus[:int(os.environ['SLURM_CPUS_PER_TASK'])]
(p/'allocation.json').write_text(json.dumps({'host':socket.gethostname(),'job':job,'abstract_ids':abstract,'linux_cpus':cpus,'scontrol':s,'source':'actual slurmd hwloc XML; socket/core/PU traversal'},indent=2))
(p/'topology.xml').write_text(xml)
print(','.join(map(str,cpus)))
