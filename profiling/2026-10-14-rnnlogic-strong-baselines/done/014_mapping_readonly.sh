#!/bin/bash
set -u
TASK_ROOT=/zpool-00/home/cw636/chenyi/SymFabric/profiling/2026-10-14-rnnlogic-strong-baselines
cd "$TASK_ROOT"
/home/cw636/venvs/epic/bin/python - <<'PY'
import json, pathlib, shutil, subprocess, xml.etree.ElementTree as ET
result={}
xml=None
path=pathlib.Path('/var/spool/slurmd/hwloc_topo_whole.xml')
try:
 xml=path.read_text();result['source']=str(path)
except OSError as e:
 result['daemon_xml_read_error']=str(e)
if xml is None:
 for name in ['lstopo-no-graphics','lstopo','hwloc-ls']:
  tool=shutil.which(name)
  if tool:
   p=subprocess.run([tool,'--of','xml'],capture_output=True,text=True,timeout=15)
   result['topology_command']={'argv':[tool,'--of','xml'],'returncode':p.returncode,'stderr':p.stderr}
   if p.returncode==0:xml=p.stdout;result['source']=tool;break
if xml:
 pathlib.Path('node4_hwloc_topology.xml').write_text(xml)
 root=ET.fromstring(xml)
 mapping=[]
 for package in root.iter('object'):
  if package.get('type') not in ['Package','Socket']:continue
  for core in package.iter('object'):
   if core.get('type')!='Core':continue
   for pu in core.iter('object'):
    if pu.get('type')!='PU':continue
    mapping.append({'abstract':len(mapping),'linux_cpu':int(pu.get('os_index')),'core_os_index':core.get('os_index'),'package_os_index':package.get('os_index')})
 result['mapping_socket_core_thread_order']=mapping
 result['allocated_abstract_16_47']=[row for row in mapping if 16<=row['abstract']<=47]
 result['linux_allocated_cpus']=sorted(row['linux_cpu'] for row in result['allocated_abstract_16_47'])
 result['linux_one_per_core']=sorted({next(item['linux_cpu'] for item in result['allocated_abstract_16_47'] if (item['package_os_index'],item['core_os_index'])==(row['package_os_index'],row['core_os_index'])) for row in result['allocated_abstract_16_47']})
for candidate in ['/usr/sbin/slurmd','/usr/sbin/slurmd-wlm','/usr/local/sbin/slurmd',shutil.which('slurmd')]:
 if candidate and pathlib.Path(candidate).is_file():
  p=subprocess.run([candidate,'-vvvvv','-C'],capture_output=True,text=True,timeout=15)
  pathlib.Path('slurm_cpu_map_verbose.txt').write_text(p.stdout+p.stderr)
  result['slurmd_config_only']={'argv':[candidate,'-vvvvv','-C'],'returncode':p.returncode}
  break
pathlib.Path('slurm_mapping_verified.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['mapping_socket_core_thread_order','allocated_abstract_16_47']},indent=2))
PY
