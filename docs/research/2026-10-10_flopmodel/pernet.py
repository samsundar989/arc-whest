import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.expanduser("~/scratch"))
from variants4 import *
d = json.load(open('fit_ctl.json')); st = d['stats']; nets = list(st)
n = 1024
def tabs(net, mode):
    R=[];H=[]
    for l in range(16):
        kr = n - st[net][l]['dead2.0']*n; kh = n - st[net][l]['dead2.5']*n
        f = (lambda x: min(1024, 32*math.ceil(x/32 - 1e-9))) if mode == 'up' else (lambda x: min(1024, 32*round(x/32)))
        R.append(f(kr)); H.append(f(kh))
    return R, H
res = {}
for mode in ('up', 'near'):
    Fs = []
    for net in nets[::5]:
        R, H = tabs(net, mode)
        mod, tot, ops = rc.run("estimator_k3_v81d_starall2.py", patch=lambda m: (set_kas("R", R)(m), set_kas("H", H)(m), set_kas("D", H)(m)))
        Fs.append(tot + 32768)
    Fs = np.array(Fs)
    res[mode] = Fs.tolist()
    print(mode, "nets", len(Fs), "mean F %.0f (%+.3f%% vs fixed)  sd %.3f%%  min %+.2f%% max %+.2f%%" % (Fs.mean(), 100*(Fs.mean()-BASE)/BASE, 100*Fs.std()/Fs.mean(), 100*(Fs.min()-BASE)/BASE, 100*(Fs.max()-BASE)/BASE))
json.dump(res, open('pernet_F.json','w'))
