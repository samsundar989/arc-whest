"""Q2 ablation tables: python3 experiments.py n L seed [k4=1]"""
import numpy as np, sys, json, time
from tower import make_net, run_tower

def mse(a, b): return float(np.mean((a-b)**2))

def ablations(n, L, seed, use_k4=True, mcfile=None):
    W = make_net(n, L, seed)
    mc = np.load(mcfile or f'/home/claude/sandbox/results/mc_n{n}_L{L}_s{seed}.npz')
    ymc = mc['y']; floor = float(mc['y_var'].mean())
    res = dict(n=n, L=L, seed=seed, N=int(mc['N']), mc_floor=floor)
    base = dict(k4=use_k4)
    runs = {}
    def R(name, **kw):
        o = dict(base); o.update(kw)
        t = time.time(); r = run_tower(W, o); dt = time.time()-t
        runs[name] = r
        res[name] = dict(mse=mse(r['y'], ymc), t=dt)
        print(f"  {name:34s} MSE vs MC {res[name]['mse']:.3e}   ({dt:.0f}s)", flush=True)
        return r
    full = R('full')
    yf = full['y']
    for name, kw in [('K2 only', dict(k3=False, k4=False)),
                     ('K2+k4 only (no k3)', dict(k3=False)),
                     ('no k4', dict(k4=False)),
                     ('direct only (no indirect)', dict(indirect_mean=False, indirect_cov=False)),
                     ('indirect mean only', dict(indirect_cov=False)),
                     ('indirect cov only', dict(indirect_mean=False)),
                     ('transport gate', dict(transport='gate')),
                     ('born diag only', dict(born_pair=False, born_distinct=False)),
                     ('born diag+pair', dict(born_distinct=False)),
                     ('born diag+pair+hub2', dict(distinct_maxorder=2)),
                     ('born upto order3', dict(distinct_maxorder=3)),
                     ('no k3^2 readout', dict(readout_k33=False)),
                     ('no k4->k3', dict(k4_to_k3=False)),
                     ]:
        if not use_k4 and ('k4' in name and name != 'no k4'):
            continue
        r = R(name, **kw)
        res[name]['removed_norm2'] = mse(r['y'], yf)
    for a in range(1, L):
        r = R(f'age<={a} both', direct_max_age=a, indirect_max_age=a); res[f'age<={a} both']['removed_norm2'] = mse(r['y'], yf)
        r = R(f'hybrid direct all, indirect age<={a}', indirect_max_age=a); res[f'hybrid direct all, indirect age<={a}']['removed_norm2'] = mse(r['y'], yf)
        r = R(f'direct age<={a}, indirect all', direct_max_age=a); res[f'direct age<={a}, indirect all']['removed_norm2'] = mse(r['y'], yf)
    # per-birth direct contributions at the readout (from full run)
    rec = full['rec'][-1]
    sig = np.sqrt(np.diag(rec['C'])); al = rec['mu']/sig
    from relu_hermite import readout_c3
    c3 = readout_c3(al)
    res['direct_by_birth_norm2'] = {int(b): float(np.mean((c3*d/sig**2)**2)) for b, d in rec['d3_by_birth'].items()}
    # per-layer diagnostics vs MC
    diag = []
    for l in range(1, L+1):
        rc = full['rec'][l-1]
        diag.append(dict(l=l, mu_err=float(np.sqrt(mse(rc['mu'], mc[f'mu{l}']))), mu_rms=float(np.sqrt(np.mean(mc[f'mu{l}']**2))),
                         sig2_err=float(np.sqrt(mse(np.diag(rc['C']), np.diag(mc[f'C{l}'])))), sig2_rms=float(np.sqrt(np.mean(np.diag(mc[f'C{l}'])**2))),
                         k3d_err=float(np.sqrt(mse(rc['d3'], mc[f'k3d{l}']))), k3d_rms=float(np.sqrt(np.mean(mc[f'k3d{l}']**2))),
                         k3p_err=float(np.sqrt(mse(rc['P3'], mc[f'k3p{l}']))), k3p_rms=float(np.sqrt(np.mean(mc[f'k3p{l}']**2))),
                         k4d_err=float(np.sqrt(mse(rc['d4'], mc[f'k4d{l}']))), k4d_rms=float(np.sqrt(np.mean(mc[f'k4d{l}']**2))),
                         k4p_err=float(np.sqrt(mse(rc['K22'], mc[f'k4p{l}']))), k4p_rms=float(np.sqrt(np.mean(mc[f'k4p{l}']**2)))))
    res['layer_diag'] = diag
    # K2-only per-layer errors for reference
    k2 = runs['K2 only']
    res['layer_diag_k2'] = [dict(l=l, mu_err=float(np.sqrt(mse(k2['rec'][l-1]['mu'], mc[f'mu{l}']))),
                                 sig2_err=float(np.sqrt(mse(np.diag(k2['rec'][l-1]['C']), np.diag(mc[f'C{l}']))))) for l in range(1, L+1)]
    np.savez(f'/home/claude/sandbox/results/tower_n{n}_L{L}_s{seed}.npz', y_full=yf, y_mc=ymc,
             **{f'var_{k}': v['y'] for k, v in runs.items()})
    json.dump(res, open(f'/home/claude/sandbox/results/ablation_n{n}_L{L}_s{seed}.json', 'w'), indent=1)
    return res

if __name__ == '__main__':
    n, L, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    use_k4 = (len(sys.argv) < 5) or sys.argv[4] == '1'
    ablations(n, L, seed, use_k4)
