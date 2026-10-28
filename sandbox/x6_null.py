"""GOE null: fraction of ||G||_F^2 (and of the induced variance error diag(W G W^T)) carried by the top-k eigenpairs (by |lambda|)
of a random symmetric n x n matrix, for k = 16, 32 and for fixed k/n.  python3 x6_null.py"""
import numpy as np, json
from x6_analysis import eig_sorted, frob_fractions, sandwich_r2
rng = np.random.default_rng(1)
out = {}
for n in [64, 128, 256, 512, 1024]:
    ks = sorted(set([k for k in [4, 8, 16, 32, 64] if k <= n] + [max(1, n//64), max(1, n//32), n//16, n//8]))
    fr = {k: 0.0 for k in ks}; R2 = {k: 0.0 for k in ks}; reps = 3 if n <= 512 else 2
    for r in range(reps):
        G = rng.standard_normal((n, n)); G = (G + G.T)/np.sqrt(2); np.fill_diagonal(G, 0.0)
        lam, U = eig_sorted(G); f = frob_fractions(lam, ks)
        W = rng.standard_normal((n, n))*np.sqrt(2.0/n)
        t = np.einsum('ij,jk,ik->i', W, G, W); s = sandwich_r2(lam, U, W, t, ks, remove_diag=True)
        for k in ks: fr[k] += f[k]/reps; R2[k] += s[k]/reps
    out[n] = dict(fro={k: round(fr[k], 4) for k in ks}, R2nd={k: round(R2[k], 4) for k in ks})
    print(f'n={n:5d}  frob fraction: ' + '  '.join(f'k={k}:{fr[k]:.3f}' for k in ks))
    print(f'         sandwich R2nd: ' + '  '.join(f'k={k}:{R2[k]:.3f}' for k in ks), flush=True)
json.dump(out, open('/home/claude/sandbox/results/x6_null.json', 'w'), indent=1)
