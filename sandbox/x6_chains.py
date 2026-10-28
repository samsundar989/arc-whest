"""X6 sandbox, step 1: run the reference tower and several truncated chains on the same net and dump
per-layer first/second moments (pre-activation mu_l, C_l; post-ReLU m_h,l, Cov_h,l) + final prediction.
   python3 x6_chains.py n L seed [variants,comma,separated]
-> results/x6_chains_n{n}_L{L}_s{seed}.npz  and a .log with the MSE-vs-MC ratios (if the MC file exists).
"""
import numpy as np, sys, time, os
from tower import make_net, run_tower

VARIANTS = {
    'full': dict(),
    'age2': dict(direct_max_age=2, indirect_max_age=2),
    'age3': dict(direct_max_age=3, indirect_max_age=3),
    'age4': dict(direct_max_age=4, indirect_max_age=4),
    'ind3': dict(indirect_max_age=3),
    'dp':   dict(born_distinct=False),                 # diag+pair born only (no hub)
    'gate': dict(transport='gate'),
    'k2':   dict(k3=False),
}

def run(n, L, seed, names):
    W = make_net(n, L, seed)
    mcf = f'/home/claude/sandbox/results/mc_n{n}_L{L}_s{seed}.npz'
    mc = np.load(mcf) if os.path.exists(mcf) else None
    out = dict(n=n, L=L, seed=seed)
    log = open(f'/home/claude/sandbox/results/x6_chains_n{n}_L{L}_s{seed}.log', 'a')
    ys = {}
    for name in names:
        t = time.time()
        o = dict(VARIANTS[name], k4=False)
        if n*n*n*8*L > 1.5e9 and 'direct_max_age' not in o and 'indirect_max_age' not in o:
            o['merge_births'] = True        # identical chain (transport is linear in K3), one n^3 tensor instead of L
        r = run_tower(W, o)
        dt = time.time() - t
        ys[name] = r['y']
        out[f'{name}_y'] = r['y']
        for l in range(1, L+1):
            rc = r['rec'][l-1]
            out[f'{name}_mu{l}'] = rc['mu']; out[f'{name}_C{l}'] = rc['C']
            if 'Cov_h' in rc:
                out[f'{name}_mh{l}'] = rc['m_h']; out[f'{name}_Covh{l}'] = rc['Cov_h']
        msg = f'{name:6s} t={dt:6.1f}s'
        if mc is not None:
            m = float(np.mean((r['y'] - mc['y'])**2)); out[f'{name}_mse'] = m
            msg += f'  MSE vs MC {m:.3e}'
            if 'full' in ys:
                msg += f'  ratio to full {m/out["full_mse"]:.3f}'
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
    fn = f'/home/claude/sandbox/results/x6_chains_n{n}_L{L}_s{seed}.npz'
    if os.path.exists(fn):   # merge with earlier variants
        old = dict(np.load(fn)); old.update(out); out = old
    np.savez(fn, **out)
    return out

if __name__ == '__main__':
    n, L, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    names = sys.argv[4].split(',') if len(sys.argv) > 4 else ['full', 'age2', 'age3', 'age4', 'ind3', 'dp', 'gate', 'k2']
    run(n, L, seed, names)
