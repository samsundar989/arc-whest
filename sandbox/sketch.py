"""Q5: variance of randomized estimators for the direct kappa3 contractions (sandbox n=64/128)."""
import numpy as np
from adjoint import k2_chain, backward_adjoint
from tower import make_net
def run(n, L, seed, ks=(8, 16, 32, 64, 128, 256), reps=20):
    W = make_net(n, L, seed); layers = k2_chain(W)
    # legs of the oldest birth (h_1) to z_L
    V = W[L-1].T.copy()
    for l in range(L-1, 1, -1):
        V = W[l-1].T@(layers[l-1]['g'][:, None]*V)
    lay = layers[0]; d, P, rt, a1, a2 = lay['d'], lay['P'], lay['rho'], lay['A1'], lay['A2']
    rng = np.random.default_rng(0)
    exact_diag = (V**3).T@d
    S1 = rt@(a1[:, None]*V); exact_hub_main = np.sum(a2[:, None]*V*S1*S1, axis=0)
    out = []
    for k in ks:
        if k > n: continue
        e_sub = []; e_hub = []
        for r in range(reps):
            # (i) subset sampling of diag-born lineage columns with uniform weights
            idx = rng.choice(n, k, replace=False)
            est = (n/k)*(V[idx]**3).T@d[idx]; e_sub.append(np.mean((est-exact_diag)**2))
            # (ii) two independent Gaussian sketches for (rt x)_i^2 in the hub term
            O1 = rng.standard_normal((n, k))/np.sqrt(k); O2 = rng.standard_normal((n, k))/np.sqrt(k)
            X = a1[:, None]*V
            y1 = (rt@O1)@(O1.T@X); y2 = (rt@O2)@(O2.T@X)
            est = np.sum(a2[:, None]*V*y1*y2, axis=0); e_hub.append(np.mean((est-exact_hub_main)**2))
        out.append(dict(k=k, rel_var_subset=float(np.mean(e_sub)/np.mean(exact_diag**2)), rel_var_hub=float(np.mean(e_hub)/np.mean(exact_hub_main**2)), n_over_k=n/k))
    return out
if __name__ == '__main__':
    for n in [64, 128]:
        res = run(n, 8, 0)
        print(f"n={n}: oldest birth, direct term. relative MSE of estimator vs k:")
        for r in res: print(f"   k={r['k']:4d}  subset-sampling (diag legs): {r['rel_var_subset']:.3f}   two-sketch hub: {r['rel_var_hub']:.3f}   (n/k = {r['n_over_k']:.1f})")
