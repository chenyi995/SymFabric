import urllib.request,re,json
from html.parser import HTMLParser
from pathlib import Path
b=Path(__file__).resolve().parent/'tptp';urls={}
class Pre(HTMLParser):
 def __init__(self):super().__init__();self.active=False;self.text=[]
 def handle_starttag(self,t,a):
  if t=='pre':self.active=True
 def handle_endtag(self,t):
  if t=='pre':self.active=False
 def handle_data(self,d):
  if self.active:self.text.append(d)
def fetch(rel):
 p=b/rel
 if p.exists():return
 u='https://seetptp.tptp.org/file/TPTPWorld/TPTP/'+rel
 parser=Pre();parser.feed(urllib.request.urlopen(u,timeout=30).read().decode());s=''.join(parser.text);assert 'fof(' in s or 'cnf(' in s or 'include(' in s
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s);urls[rel]=u;print(rel,len(s),flush=True)
 for x in re.findall(r"include\('([^']+)'",s):fetch(x)
fetch('Problems/CSR/CSR188+1.p');(b/'sources.json').write_text(json.dumps(urls,indent=2))
