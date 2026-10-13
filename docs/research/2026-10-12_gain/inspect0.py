from common import *
import json
L=Labels(0)
for k,(o,h) in L.off.items(): print(k,o,h.strip())
z=np.load(L.path); print('name',z['name'])
meta=json.load(open(ROOT+'/cache/twin/full_meta.json')); print(meta.keys() if isinstance(meta,dict) else type(meta)); print(meta['names'][:4]); print({k:(v if not isinstance(v,list) else len(v)) for k,v in meta.items()})
t=np.load(ROOT+'/cache/twin/full_truth.npy',mmap_mode='r'); print('truth',t.shape,t.dtype)
mu=L.vec('mu'); var=L.vec('var'); D3=L.vec('D3'); dG=L.vec('dG')
print('mu0 rms',np.sqrt((mu[0]**2).mean()),'var0 mean',var[0].mean())
W0=weights(0,0)
print('var0 vs col norms',np.abs(var[0]-(W0**2).sum(0)).max(),' vs row norms',np.abs(var[0]-(W0**2).sum(1)).max())
C0=L.mat('C11',0); G=W0.T@W0; np.fill_diagonal(G,0)
print('C11_0 diag',np.abs(np.diag(C0)).max(),'C0 vs W0^T W0 offdiag maxerr',np.abs(C0-G).max(),'rms C0',np.sqrt((C0**2).mean()),'rms err',np.sqrt(((C0-G)**2).mean()))
print('C sym',np.abs(C0-C0.T).max())
tr=truth(0)
for l in [1,5,15]:
    W=weights(0,l)
    print(l,'mu_l vs W^T truth_{l-1}: rms err',np.sqrt(((mu[l]-W.T@tr[l-1])**2).mean()),'rms mu',np.sqrt((mu[l]**2).mean()),' wrong orient err',np.sqrt(((mu[l]-W@tr[l-1])**2).mean()))
for l in [0,1,5,15]:
    print(l,'layer0 D3 rms/sig^3',np.sqrt(((D3[l]/var[l]**1.5)**2).mean()),'dG rms/sig^4',np.sqrt(((dG[l]/var[l]**2)**2).mean()))
D21=L.mat('D21',5); print('D21 diag max',np.abs(np.diag(D21)).max(),'asym rms',np.sqrt(((D21-D21.T)**2).mean()),'rms',np.sqrt((D21**2).mean()))
