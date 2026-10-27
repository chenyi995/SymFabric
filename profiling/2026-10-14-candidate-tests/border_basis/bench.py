import sys,os,json,time,statistics,socket,math,subprocess,cProfile,pstats
from pathlib import Path
b=Path(__file__).resolve().parent;up=b/'upstream/OracleBorderBasis';sys.path.insert(0,str(up/'src'))
from sage.all import PolynomialRing,QQ,matrix
from border_basis_lib.border_basis import BorderBasisCalculator
out={'host':socket.gethostname(),'job':os.getenv('SLURM_JOB_ID'),'commit':subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip(),'scope':'current official Gaussian elimination kernel; QQ fixtures; no neural oracle; no broken lstable wrapper repair','cases':[]}
def call(c,V,G,fast):return c.gaussian_elimination(list(V),list(G),use_fast_elimination=fast)[0]
def spaces(ring,groups):
 mons=sorted({m for group in groups for p in group for m in p.dict()})
 return [matrix(QQ,len(ps),len(mons),[p.dict().get(m,0) for p in ps for m in mons]).row_space() for ps in groups]
for path in sorted((up/'tests/instances').glob('*.json')):
 row={'fixture':path.name}
 try:
  data=json.loads(path.read_text());R=PolynomialRing(QQ,data['variables'],order='degrevlex');V=[R(p) for p in data['V']];G=[R(p) for p in data['F']];E=[R(p) for p in data['expected_output']];c=BorderBasisCalculator(R)
  W={f:call(c,V,G,f) for f in [False,True]};ss=spaces(R,[V+G,V+W[False],V+W[True],V+E]);row.update({'nV':len(V),'nG':len(G),'terms':sum(len(p.dict()) for p in V+G),'span_equal':ss[0]==ss[1]==ss[2],'expected_span_equal':ss[0]==ss[3],'nW':[len(W[False]),len(W[True])]});assert row['span_equal']
  samples={False:[],True:[]}
  for rep in range(6):
   for fast in ([False,True] if rep%2==0 else [True,False]):
    n=100;start=time.perf_counter()
    for _ in range(n):call(c,V,G,fast)
    ms=(time.perf_counter()-start)*1000/n
    if rep:samples[fast].append(ms)
  row['timing']={str(f):{'median_ms':statistics.median(samples[f]),'samples_ms':samples[f]} for f in samples}
  for f in [False,True]:
   pr=cProfile.Profile();pr.enable();call(c,V,G,f);pr.disable()
   with (b/(path.stem+'_'+str(f)+'_profile.txt')).open('w') as stream:pstats.Stats(pr,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(20)
 except Exception as e:row['error']=repr(e)
 out['cases'].append(row);(b/'raw.json').write_text(json.dumps(out,indent=2));print(json.dumps(row),flush=True)
