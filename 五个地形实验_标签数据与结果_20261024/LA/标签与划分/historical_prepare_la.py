"""Full author-shared LA grid, metre units, author linspace spacing; no resampling."""
from pathlib import Path
import numpy as np,json,hashlib,resource
R=Path(__file__).resolve().parents[1]
resource.setrlimit(resource.RLIMIT_AS,(90*1024**3,90*1024**3))
p=R/'data/LA.off'
if not p.exists():
 z=np.load(R/'data/la-4kx4k.npy',mmap_mode='r');assert z.shape==(4000,4000) and np.isfinite(z).all()
 n=4000;x=np.linspace(0,28.34*n,n);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:
  f.write(f'OFF\n{n*n} {2*(n-1)**2} 0\n')
  for i in range(n):np.savetxt(f,np.column_stack((x,np.full(n,x[i]),z[i])),fmt='%.9f')
  for i in range(n-1):
   a=i*n+np.arange(n-1);b=a+n;c=a+1;d=b+1
   np.savetxt(f,np.column_stack((np.full(n-1,3),a,b,c)),fmt='%d')
   np.savetxt(f,np.column_stack((np.full(n-1,3),c,b,d)),fmt='%d')
 tmp.replace(p)
 (R/'LA_mesh.json').write_text(json.dumps({'vertices':n*n,'faces':2*(n-1)**2,'nominal_spacing_m':28.34,'actual_spacing_m':28.34*n/(n-1),'zero_policy':'retain as author code','units':'metres','resampling':False,'author_commit':'7d1e193e0e685053447f69baee2e8eec0b936b7d'},indent=2))
if not (R/'LA_qa.json').exists():
 from prepare import prepare
 prepare('LA')
