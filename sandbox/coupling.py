"""kappa3 x kappa4 coupling test on exact towers vs MC (n=64, L=8, seeds 0,1).
Variants: k3 age<=2 or full  x  k4 off / scalar (lambda-law-like) / full tensor."""
import numpy as np, json, sys, time
from tower import make_net, run_tower
res={}
for seed in (0,1):
    n,L=64,8
    W=make_net(n,L,seed); mc=np.load(f'results/mc_n{n}_L{L}_s{seed}.npz')['y']
    for k3name,k3o in (('k3age2',dict(direct_max_age=2,indirect_max_age=2)),('k3age4',dict(direct_max_age=4,indirect_max_age=4)),('k3full',{})):
        for k4name,k4o in (('k4off',dict(k4=False)),('k4scalar',dict(k4=True,k4mode='scalar')),('k4full',dict(k4=True,k4mode='full'))):
            o=dict(k3o); o.update(k4o); t=time.time()
            try:
                r=run_tower(W,o); m=float(np.mean((r['y']-mc)**2))
            except Exception as e:
                m=None; print('ERR',k3name,k4name,e)
            res[f's{seed}_{k3name}_{k4name}']=m
            print(seed,k3name,k4name,m,'%.0fs'%(time.time()-t),flush=True)
json.dump(res,open('results/coupling_n64.json','w'),indent=1)
