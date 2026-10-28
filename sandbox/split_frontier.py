"""SPLIT-family error-vs-cost frontier on exact first-order hub-star kappa3 lineages.

Reuses pooled_carrier's machinery (k2_chain, readouts, werr, leg_at, phi) but builds the legs ONCE per (n, seed)
and loops the (E, r) grid inside one process.  For each layer l in 9..15 and each E the stacked old legs
(ages > E, KAS hubs |alpha| < kas) are SVD'd once; every r then takes the leading r left-singular vectors.

Per (n, E, r) we report
  cost   : SPLIT cost model 1, 4*nex + 6r          (pooled_carrier), in n^3 units  (= n^2-units / n), mean over layers 9..15
  cost2  : exact 4*nex ; band 2r + 4r + 0.5*r^2/n*(n_old/n)   (readout-heavy, Khatri-Rao style), in n^3 units
  e3w    : rms over layers 9..15 of the sensitivity-weighted D3 error
  e21w   : rms over layers 9..15 of the sensitivity-weighted D21 error
Errors are averaged over seeds (rms over seeds of the per-seed rms) in the printed table / frontier.
Usage: python split_frontier.py <n> [seeds...]      (defaults: seeds 1 2 ; L = 16)
"""
import numpy as np, json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pooled_carrier import k2_chain, readouts, werr, leg_at, phi

L = 16
LAYERS = list(range(9, 16))
E_LIST = [2, 3, 4, 5, 6, 7, 8]
RFRAC = [0.14, 0.21, 0.28, 0.375, 0.5, 0.56, 0.75]
KAS = 2.5

def build(n, seed):
    """W, moments, full readouts + sensitivities at the layers of interest, kept legs per (l, s)."""
    W = [np.random.default_rng(seed + 1000 * k).standard_normal((n, n)) * np.sqrt(2.0 / n) for k in range(L)]
    st = k2_chain(W)
    full, sens, legs = {}, {}, {}
    for l in LAYERS:
        a, sg = st[l]['alpha'], st[l]['sigma']
        sens[l] = dict(s3=(a * phi(a) / sg**2)**2, t21=np.outer(phi(a), phi(a))**2)
        Uall, Vall, wall = [], [], []
        per = {}
        for s in range(l):                           # birth layer s, age = l - s
            U, V = leg_at(W, st, s, l)
            p = st[s]
            w = p['sigma'] * phi(p['alpha'])
            Uall.append(U); Vall.append(V); wall.append(w)
            keep = np.abs(p['alpha']) < KAS
            per[s] = (U[:, keep], V[:, keep], w[keep])
        full[l] = readouts(np.concatenate(Uall, 1), np.concatenate(Vall, 1), np.concatenate(wall))
        legs[l] = per
    return full, sens, legs

def frontier_one(n, seed):
    t0 = time.time()
    full, sens, legs = build(n, seed)
    tb = time.time() - t0
    r_list = [int(round(f * n)) for f in RFRAC]
    out = {}
    for E in E_LIST:
        # per layer: exact tier + SVD basis of the old tier, computed once per (E, l)
        per_l = {}
        for l in LAYERS:
            per = legs[l]
            ex = [per[s] for s in sorted(per) if l - s <= E]
            old = [per[s] for s in sorted(per) if l - s > E]
            Ue = np.concatenate([u for u, _, _ in ex], 1); Ve = np.concatenate([v for _, v, _ in ex], 1)
            we = np.concatenate([w for _, _, w in ex])
            nex = Ue.shape[1]
            if old:
                oU = np.concatenate([u for u, _, _ in old], 1); oV = np.concatenate([v for _, v, _ in old], 1)
                ow = np.concatenate([w for _, _, w in old])
                sw = np.sqrt(ow)[None, :]
                Q, _, _ = np.linalg.svd(np.concatenate([oU * sw, oV * sw], 1), full_matrices=False)
                # exact-tier contribution to the readouts is fixed; band contribution depends on r
                per_l[l] = (nex, readouts(Ue, Ve, we), oU, oV, ow, Q)
            else:
                per_l[l] = (nex, readouts(Ue, Ve, we), None, None, None, None)
        for r, rf in zip(r_list, RFRAC):
            rows, c1, c2 = [], [], []
            for l in LAYERS:
                nex, (D3e, D21e), oU, oV, ow, Q = per_l[l]
                if Q is not None:
                    Qr = Q[:, :r]
                    pU = Qr @ (Qr.T @ oU); pV = Qr @ (Qr.T @ oV)
                    D3b, D21b = readouts(pU, pV, ow)
                    D3, D21 = D3e + D3b, D21e + D21b
                    n_old = oU.shape[1]
                    band1 = 6.0 * r
                    band2 = 2.0 * r + 4.0 * r + 0.5 * r * r / n * (n_old / n)
                else:
                    D3, D21 = D3e, D21e
                    n_old = 0; band1 = band2 = 0.0
                F3, F21 = full[l]
                rows.append(dict(l=l, nex=int(nex), n_old=int(n_old),
                                 e3w=werr(D3, F3, sens[l]['s3']), e21w=werr(D21, F21, sens[l]['t21']),
                                 e3=werr(D3, F3, np.ones(n)), e21=werr(D21, F21, np.ones((n, n))),
                                 cost_n2=4.0 * nex + band1, cost2_n2=4.0 * nex + band2))
                c1.append(4.0 * nex + band1); c2.append(4.0 * nex + band2)
            rms = lambda k: float(np.sqrt(np.mean([x[k]**2 for x in rows])))
            out[f'E{E}_r{r}'] = dict(E=E, r=r, rfrac=rf, cost=float(np.mean(c1)) / n, cost2=float(np.mean(c2)) / n,
                                     cost_n2=float(np.mean(c1)), cost2_n2=float(np.mean(c2)),
                                     e3w=rms('e3w'), e21w=rms('e21w'), e3=rms('e3'), e21=rms('e21'), rows=rows)
    return out, dict(build_s=tb, total_s=time.time() - t0)

def pareto(points, ck, ek):
    """indices of points not dominated in (cost ck, error ek), sorted by cost."""
    idx = sorted(range(len(points)), key=lambda i: (points[i][ck], points[i][ek]))
    front, best = [], np.inf
    for i in idx:
        if points[i][ek] < best - 1e-12:
            front.append(i); best = points[i][ek]
    return front

def aggregate(per_seed):
    keys = list(next(iter(per_seed.values())).keys())
    agg = {}
    for k in keys:
        rs = [per_seed[s][k] for s in per_seed]
        agg[k] = dict(E=rs[0]['E'], r=rs[0]['r'], rfrac=rs[0]['rfrac'],
                      cost=float(np.mean([x['cost'] for x in rs])), cost2=float(np.mean([x['cost2'] for x in rs])),
                      e3w=float(np.sqrt(np.mean([x['e3w']**2 for x in rs]))),
                      e21w=float(np.sqrt(np.mean([x['e21w']**2 for x in rs]))),
                      e3=float(np.sqrt(np.mean([x['e3']**2 for x in rs]))),
                      e21=float(np.sqrt(np.mean([x['e21']**2 for x in rs]))))
    return agg

def table(agg, n, ck='cost'):
    pts = list(agg.values())
    f3 = set(pareto(pts, ck, 'e3w')); f21 = set(pareto(pts, ck, 'e21w'))
    lines = [f'# n={n}  sorted by {ck} (n^3 units/layer, mean over layers 9-15); errors = rms over layers 9-15, seeds pooled',
             f'{"E":>2} {"r":>4} {"r/n":>5} {"cost":>7} {"cost2":>7} {"e3w":>8} {"e21w":>8} {"e3":>8} {"e21":>8}  front']
    for p in sorted(pts, key=lambda p: p[ck]):
        i = pts.index(p)
        tag = ('3' if i in f3 else '-') + ('21' if i in f21 else '--')
        lines.append(f'{p["E"]:>2} {p["r"]:>4} {p["rfrac"]:>5.3f} {p["cost"]:>7.3f} {p["cost2"]:>7.3f} '
                     f'{p["e3w"]:>8.4f} {p["e21w"]:>8.4f} {p["e3"]:>8.4f} {p["e21"]:>8.4f}  {tag}')
    return '\n'.join(lines)

if __name__ == '__main__':
    n = int(sys.argv[1]); seeds = [int(x) for x in sys.argv[2:]] or [1, 2]
    per_seed, timing = {}, {}
    for sd in seeds:
        per_seed[sd], timing[sd] = frontier_one(n, sd)
        print(f'seed {sd}: build {timing[sd]["build_s"]:.1f}s total {timing[sd]["total_s"]:.1f}s', flush=True)
    agg = aggregate(per_seed)
    os.makedirs('results', exist_ok=True)
    res = dict(n=n, L=L, seeds=seeds, layers=LAYERS, E_list=E_LIST, rfrac=RFRAC, kas=KAS,
               cost_model_1='4*nex + 6r   (n^2 units; cost field divided by n -> n^3 units)',
               cost_model_2='4*nex + 2r + 4r + 0.5*r^2/n*(n_old/n)   (n^2 units; cost2 field divided by n)',
               agg=agg, per_seed={str(s): per_seed[s] for s in per_seed}, timing={str(s): timing[s] for s in timing})
    json.dump(res, open(f'results/split_frontier_n{n}.json', 'w'))
    print(table(agg, n, 'cost')); print(); print(table(agg, n, 'cost2'))
