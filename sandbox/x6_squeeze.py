"""Squeezed-GOE null for the X6 extrapolation.  python3 x6_squeeze.py n L seed
Model: an *incoherent* covariance-error innovation R_{l0} (GOE, zero diagonal, unit Frobenius norm) is injected at pre-activation
layer l0 and transported by the gate-linearised chain,  dC_{l+1} = W_{l+1} G_l dC_l G_l W_{l+1}^T  (G_l = diag Phi(mu/sigma) from the
K2 chain of the same random net).  Reports, at every readout layer l, the fraction of ||dC_l||_F^2 and the R^2 of the in-layer
quadratic readout diag(T_k) against dvar_l = diag(dC_l) for the top-k eigenpairs, for
   (a) a single injection at l0 = 2 (oldest possible error),  (b) a mixture: equal-norm independent injections at every layer 2..l-1.
This is a pure geometry statement about the 1024 x 16 He-init chain and needs no cumulant tower, so it runs at n = 1024 directly.
"""
import numpy as np, sys, json, time
from tower import make_net, run_tower
from x6_analysis import eig_sorted, frob_fractions, participation_ratio, r2

def goe(n, rng):
    R = rng.standard_normal((n, n)); R = (R + R.T)/np.sqrt(2); np.fill_diagonal(R, 0.0)
    return R/np.linalg.norm(R)

def run(n, L, seed, ks=(1, 2, 4, 8, 16, 32, 64, 128), reps=2, inj_scale=1.0):
    rng = np.random.default_rng(100 + seed)
    W = make_net(n, L, seed)
    t = time.time()
    k2 = run_tower(W, dict(k3=False, k4=False, pmax=12))      # K2 chain only (gates); pmax=12 is enough for the gates
    gates = []
    for l in range(1, L):
        rc = k2['rec'][l-1]; sig = np.sqrt(np.diag(rc['C'])); gates.append(0.5*(1+__import__('scipy.special', fromlist=['erf']).erf(rc['mu']/sig/np.sqrt(2))))
    print(f'K2 chain n={n} L={L}: {time.time()-t:.1f}s', flush=True)
    ks = [k for k in ks if k < n]
    out = dict(n=n, L=L, seed=seed, single={}, mixture={})
    for mode in ['single', 'mixture']:
        acc = {}
        for rep in range(reps):
            S = None
            for l in range(2, L+1):
                if S is not None:
                    g = gates[l-2]
                    S = W[l-1] @ (np.outer(g, g)*S) @ W[l-1].T
                if mode == 'mixture' or l == 2:
                    R = goe(n, rng)*(1.0 if S is None else inj_scale)
                    S = R if S is None else S + R
                if l < 4: continue
                lam, U = eig_sorted(S); fr = frob_fractions(lam, ks); pr = participation_ratio(lam)
                dv = np.diag(S); c = np.cumsum(lam[None, :]*U**2, axis=1)
                R2 = {k: r2(dv, c[:, k-1]) for k in ks}
                a = acc.setdefault(l, dict(PR=0.0, fro={k: 0.0 for k in ks}, R2={k: 0.0 for k in ks}, gain=0.0, trans=0.0))
                a['PR'] += pr/reps; a['gain'] += np.linalg.norm(S)**2/reps
                if mode == 'mixture': a['trans'] += (1 - inj_scale**2/np.linalg.norm(S)**2)/reps   # transported share of ||dC_l||^2
                for k in ks: a['fro'][k] += fr[k]/reps; a['R2'][k] += R2[k]/reps
        out[mode] = acc
        print(f'\n## {mode} injection, n={n}, L={L}   (fro = Frobenius fraction in top-k; R2 = in-layer diag readout R^2)')
        print('l    PR     |F|^2  trans  ' + ' '.join(f'fro{k:<3d}' for k in ks) + ' | ' + ' '.join(f'R2_{k:<3d}' for k in ks))
        for l, a in acc.items():
            print(f'{l:<3d} {a["PR"]:6.1f} {a["gain"]:8.3f} {a["trans"]:5.2f}  ' + ' '.join(f'{a["fro"][k]:6.3f}' for k in ks) + ' | ' + ' '.join(f'{a["R2"][k]:6.3f}' for k in ks), flush=True)
    json.dump(out, open(f'/home/claude/sandbox/results/x6_squeeze_n{n}_L{L}_s{seed}' + (f'_inj{inj_scale}' if inj_scale != 1.0 else '') + '.json', 'w'), indent=1, default=float)
    return out

if __name__ == '__main__':
    n, L, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    run(n, L, seed, inj_scale=float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
