"""Sphere-vs-Gauss gain vs n for the full tower (kappa3 only) and variant C. python3 sphere_scaling.py n"""
import numpy as np, sys, json, time
from tower import make_net, run_tower
from relu_hermite import e1_chi
def mse(a, b): return float(np.mean((a-b)**2))
n = int(sys.argv[1]); L = 8; seed = 0
W = make_net(n, L, seed); mc = np.load(f'/home/claude/sandbox/results/mc_n{n}_L{L}_s{seed}.npz'); ymc = mc['y']
out = dict(n=n, L=L, seed=seed, floor=float(mc['y_var'].mean()), v=float(1-e1_chi(n)**2))
k2 = run_tower(W, dict(k3=False))['y']; out['K2'] = mse(k2, ymc)
for name, kw in [('full(k3)', dict(k4=False)), ('C diag+pair+hub gate', dict(k4=False, distinct_maxorder=2, transport='gate'))]:
    t = time.time()
    g = run_tower(W, dict(kw))['y']
    s = run_tower(W, dict(kw, sphere=True, k4face=True))['y']
    out[name] = dict(gauss=mse(g, ymc), sphere_face=mse(s, ymc), ratio=mse(s, ymc)/mse(g, ymc), t=time.time()-t)
    print(n, name, out[name], flush=True)
    if n <= 64:
        sf = run_tower(W, dict(kw, sphere=True, k4=True, k4_born=False))['y']; out[name]['sphere_fullface'] = mse(sf, ymc)
        print('   full-face', out[name]['sphere_fullface'], flush=True)
json.dump(out, open(f'/home/claude/sandbox/results/sphere_n{n}.json', 'w'), indent=1)
