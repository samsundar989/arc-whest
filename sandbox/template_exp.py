"""Section 7(b): truncated chains +/- radial template, Gaussian vs sphere formulation.
python3 template_exp.py n L seed"""
import numpy as np, sys, json, time
from tower import make_net, run_tower, sym_mC
from relu_hermite import e1_chi, cov_relu, relu_moments

def mse(a, b): return float(np.mean((a-b)**2))

def main(n, L, seed):
    W = make_net(n, L, seed); mc = np.load(f'/home/claude/sandbox/results/mc_n{n}_L{L}_s{seed}.npz'); ymc = mc['y']
    v = 1 - e1_chi(n)**2
    usek4 = n <= 64
    out = dict(n=n, L=L, seed=seed, v=float(v), mc_floor=float(mc['y_var'].mean()))
    full = run_tower(W, dict(k4=usek4)); yf = full['y']
    out['full'] = mse(yf, ymc)
    print(f"full tower MSE {out['full']:.3e}  (floor {out['mc_floor']:.1e})")
    variants = {
        'A diag-born, gate': dict(born_pair=False, born_distinct=False, transport='gate'),
        'B diag+pair, gate': dict(born_distinct=False, transport='gate'),
        'C diag+pair+hub, gate': dict(distinct_maxorder=2, transport='gate'),
        'D diag+pair+hub, exact transport': dict(distinct_maxorder=2),
    }
    res = {}
    for name, kw in variants.items():
        base = dict(k4=False); base.update(kw)
        r0 = run_tower(W, base)
        # oracle per-layer missing template coefficient: project (K3_full - K3_variant)(z_l) onto Sym(mu,C) of the variant
        eps_fit = {}
        for l in range(2, L+1):
            rf = full['rec'][l-1]; rv = r0['rec'][l-1]
            # pair+diag entries only (what MC can see); Sym template entries
            muv, Cv = rv['mu'], rv['C']; Vv = np.diag(Cv)
            S = (2*muv[:, None]*Cv + muv[None, :]*Vv[:, None])/3.0; np.fill_diagonal(S, muv*Vv)
            dK = (rf['P3'] + np.diag(rf['d3'])) - (rv['P3'] + np.diag(rv['d3']))
            eps_fit[l] = float(np.sum(dK*S)/np.sum(S*S))
        # template inserted into born kappa3 of h_l == Sym(mu_{l+1}, C_{l+1}) at z_{l+1}: use eps_fit[l+1]
        rad = run_tower(W, dict(base, template_eps=lambda l, n_: 6*v))
        orc = run_tower(W, dict(base, template_eps=lambda l, n_: eps_fit.get(l+1, 0.0)))
        # sphere formulation with cheap face carrier (sqrt T2 gating), final e1
        sph = run_tower(W, dict(base, sphere=True, k4face=True))
        sphg = run_tower(W, dict(base, sphere=True, k4face=True, face_gate='g'))
        row = dict(gauss=mse(r0['y'], ymc), gauss_vs_full=mse(r0['y'], yf), radius_template=mse(rad['y'], ymc), oracle_template=mse(orc['y'], ymc),
                   oracle_vs_full=mse(orc['y'], yf), sphere_face_sqrtT2=mse(sph['y'], ymc), sphere_face_g=mse(sphg['y'], ymc),
                   n_eps_fit={l: n*e for l, e in eps_fit.items()})
        if usek4:
            sphfull = run_tower(W, dict(base, sphere=True, k4=True, k4_born=False))
            row['sphere_fullface_noborn_k4'] = mse(sphfull['y'], ymc)
        res[name] = row
        print(f"{name:36s} gauss {row['gauss']:.3e} | +radius tmpl {row['radius_template']:.3e} | +oracle tmpl {row['oracle_template']:.3e} | sphere(face sqrtT2) {row['sphere_face_sqrtT2']:.3e} sphere(face g) {row['sphere_face_g']:.3e}" + (f" sphere(full face) {row['sphere_fullface_noborn_k4']:.3e}" if usek4 else '') + f" | n*eps_fit by layer {[round(e,2) for e in row['n_eps_fit'].values()]}", flush=True)
    if usek4:
        sf = run_tower(W, dict(k4=True, sphere=True)); out['full_sphere_k4'] = mse(sf['y'], ymc)
        sfc = run_tower(W, dict(k4=True, sphere=False, k4face=True, face_lam=-6.0/(n+2), final_scale=e1_chi(n))); out['full_gauss_k4_plus_carrier'] = mse(sfc['y'], ymc)
        print(f"full tower, sphere formulation (full k4 tensor incl. face): {out['full_sphere_k4']:.3e};  gaussian full k4 + face carrier: {out['full_gauss_k4_plus_carrier']:.3e}")
    out['variants'] = res
    json.dump(out, open(f'/home/claude/sandbox/results/template_n{n}_L{L}_s{seed}.json', 'w'), indent=1)

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
