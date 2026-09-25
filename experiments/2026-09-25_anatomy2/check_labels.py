import json, sys, numpy as np, cupy as cp
sys.path.insert(0, "tools/labels")
from cumulants import NetLabels
fit = json.load(open("data/panels/fit.json"))
bad = [(i, fit["names"][k]) for k, i in enumerate(fit["indices"][:20])
       if NetLabels(i).name != fit["names"][k]]
print("name mismatches:", bad)
L = NetLabels(0, cp)
print(L.name, "keys", list(L.z.keys()))
mg = L.pre_marginals()
for l in (0, 1, 8, 15):
    d = L.layer(l)
    C = d["C"]; sd = cp.sqrt(cp.diagonal(C))
    s3 = sd[:, None] ** 2 * sd[None, :]; s4 = sd[:, None] ** 3 * sd[None, :]; s22 = sd[:, None]**2 * sd[None,:]**2
    off = ~cp.eye(C.shape[0], dtype=bool)
    r = lambda a, s: float(cp.sqrt(cp.mean((a / s)[off] ** 2)))
    print(f"L{l}: std k3/s^3 marg {np.sqrt(np.mean((mg['k3'][l]/mg['var'][l]**1.5)**2)):.2e}"
          f" k4/s^4 marg {np.sqrt(np.mean((mg['k4'][l]/mg['var'][l]**2)**2)):.2e}"
          f" | rms norm k3_iij {r(d['k3_iij'], s3):.2e} k4_iijj {r(d['k4_iijj'], s22):.2e}"
          f" k4_iiij {r(d['k4_iiij'], s4):.2e} | rms corr {r(C, sd[:,None]*sd[None,:]):.3e}"
          f" | mu/sd rms {float(cp.sqrt(cp.mean((cp.asarray(mg['mu'][l])/sd)**2))):.2e}")
