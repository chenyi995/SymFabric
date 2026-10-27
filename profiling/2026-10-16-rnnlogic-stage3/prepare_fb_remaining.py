import json
from pathlib import Path
p=Path(__file__).resolve().parent
x=json.loads((p/'results/screen_fb.json').read_text())
completed={c['relation'] for c in x['cases'] if any('forward' in d for d in c['variants'].values())}
manifest=json.loads((p/'query_stream_manifest.json').read_text())
# FB15k-237 has237original+237inverse relation IDs; verify coverage in trace below.
import numpy as np
z=np.load(p/'query_stream_FB15k-237.npz')
relations=set(map(int,np.unique(z['queries'][:,1])))
assert relations==set(range(474)) and completed<=relations
print(' '.join(str(r) for r in sorted(relations-completed)))
