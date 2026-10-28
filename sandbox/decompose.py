"""Tower-internal decomposition (no MC needed): fraction of the kappa3 effect by birth age, direct vs indirect,
plus pattern and channel ablations. python3 decompose.py n L seed [k4]"""
import numpy as np, sys, json, time
from tower import make_net, run_tower
from relu_hermite import readout_c3

def mse(a, b): return float(np.mean((a-b)**2))

def decompose(n, L, seed, use_k4=False):
    W = make_net(n, L, seed)
    t = time.time(); full = run_tower(W, dict(k4=use_k4)); yf = full['y']
    k2 = run_tower(W, dict(k3=False, k4=False))['y']
    tot = mse(k2, yf)
    out = dict(n=n, L=L, seed=seed, k4=use_k4, total_k3_effect_norm2=tot, t_full=time.time()-t)
    print(f"n={n} L={L} seed={seed}: total (K2 -> full) norm2 {tot:.3e}  rms {np.sqrt(tot):.3e}  ({out['t_full']:.0f}s)", flush=True)
    def R(name, **kw):
        o = dict(k4=use_k4); o.update(kw); y = run_tower(W, o)['y']; d = mse(y, yf); out[name] = d
        print(f"  {name:40s} dropped norm2 {d:.3e}  fraction {d/tot:.3f}", flush=True); return y
    R('direct only (no indirect)', indirect_mean=False, indirect_cov=False)
    R('no indirect cov', indirect_cov=False)
    R('no indirect mean', indirect_mean=False)
    R('transport gate', transport='gate')
    R('born diag only', born_pair=False, born_distinct=False)
    R('born diag+pair', born_distinct=False)
    R('born diag+pair+hub2', distinct_maxorder=2)
    R('born upto order3', distinct_maxorder=3)
    R('no k3^2 readout', readout_k33=False)
    if use_k4:
        R('no k4', k4=False)
        R('k4 transport gate (k3 exact)', transport='gate')
    for a in range(1, L-1):
        R(f'age<={a} both', direct_max_age=a, indirect_max_age=a)
        R(f'indirect age<={a} (direct all)', indirect_max_age=a)
        R(f'direct age<={a} (indirect all)', direct_max_age=a)
    R('indirect mean all ages, indirect cov age<=1', indirect_cov=True, indirect_max_age=None)  # placeholder same as full
    rec = full['rec'][-1]; sig = np.sqrt(np.diag(rec['C'])); al = rec['mu']/sig; c3 = readout_c3(al)
    out['direct_by_birth_norm2'] = {int(b): float(np.mean((c3*d/sig**2)**2)) for b, d in rec['d3_by_birth'].items()}
    out['proj3_by_layer'] = [dict(l=i+1, **{k: (v if not isinstance(v, dict) else {int(b): e for b, e in v.items()}) for k, v in r.get('proj3', {}).items()}) for i, r in enumerate(full['rec'])]
    json.dump(out, open(f'/home/claude/sandbox/results/decomp_n{n}_L{L}_s{seed}{"_k4" if use_k4 else ""}.json', 'w'), indent=1)
    return out

if __name__ == '__main__':
    n, L, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    decompose(n, L, seed, use_k4=(len(sys.argv) > 4 and sys.argv[4] == 'k4'))
