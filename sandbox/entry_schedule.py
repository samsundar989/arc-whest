"""Birth-layer-dependent entry schedule E(s) for the exact tier of the SPLIT kappa3 carrier.

Today the exact tier keeps every lineage for a uniform E = 6 ages whatever its birth layer s.  Here the exact tier
keeps lineage s for E(s) ages (exact if age = l - s <= E(s)); older lineages go to the rank-r band (best rank-r basis
of their stacked legs at layer l, r = 0.28 n fixed), exactly as in pooled_carrier.SPLIT.  Everything else (random
He nets, exact Gaussian chain, first-order hub-star lineages, KAS hubs |alpha| < 2.5, readouts D3 / D21, sensitivity
weights (alpha phi / sigma^2)^2 and (phi_i phi_k)^2) is reused from pooled_carrier; legs are built once per (n, seed)
through its leg_at cache and all schedules are evaluated on the same legs.

Per schedule and layer l = 4..15 : e3w, e21w (sensitivity-weighted relative errors vs the full lineage sum), e3, e21,
nex (exact hub columns alive), cost = 4 nex + 6 r (n^2 units; reported / n = n^3 units per layer).
Summary over layers 9..15 : rms errors, mean cost, per-layer cost profile, late-layer (11..15) cost share, max cost.
Each schedule is compared with uniform E = 6 and with the uniform schedule of equal mean cost (log-linear
interpolation of error against cost along the uniform E = 2..12 curve).

Usage: python entry_schedule.py <n> [seeds...] [--rfrac f] [--ref full|kas]     (defaults: seeds 1 2 ; L = 16 ; r = 0.28 n)
  --ref kas  : measure against the KAS-pruned lineage sum (isolates the exact-vs-band error from the hub-pruning floor)
  --rfrac f  : band rank fraction (diagnostic; the main run is f = 0.28)
"""
import numpy as np, json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pooled_carrier import k2_chain, readouts, werr, leg_at, phi

L = 16
LAYERS = list(range(4, L))          # layers evaluated
SUMM = list(range(9, L))            # layers summarised
LATE = list(range(11, L))           # late layers (cost share)
KAS = 2.5
RFRAC = 0.28

def step(lo, hi, s0):
    """E(s) = lo for s < s0, hi for s >= s0."""
    return lambda s: lo if s < s0 else hi

SCHEDULES = {
    'U6':        (lambda s: 6,                 'uniform E = 6 (reference)'),
    'S4/8':      (step(4, 8, 5),               'E = 4 for s <= 4, 8 for s >= 5'),
    'S3/9':      (step(3, 9, 5),               'E = 3 for s <= 4, 9 for s >= 5'),
    'S5/7':      (step(5, 7, 5),               'E = 5 for s <= 4, 7 for s >= 5'),
    'S2/10':     (step(2, 10, 5),              'E = 2 for s <= 4, 10 for s >= 5'),
    'R3+s/2':    (lambda s: 3 + 0.5 * s,       'linear ramp E = 3 + s/2'),
    'R2+s/2':    (lambda s: 2 + 0.5 * s,       'linear ramp E = 2 + s/2'),
    'S4/8@6':    (step(4, 8, 6),               'E = 4 for s <= 5, 8 for s >= 6 (mean-cost matched to U6)'),
    'S3/9@6':    (step(3, 9, 6),               'E = 3 for s <= 5, 9 for s >= 6 (mean-cost matched to U6)'),
    'S5/7@6':    (step(5, 7, 6),               'E = 5 for s <= 5, 7 for s >= 6 (mean-cost matched to U6)'),
    'S4/8@7':    (step(4, 8, 7),               'E = 4 for s <= 6, 8 for s >= 7'),
    'S8/4':      (step(8, 4, 5),               'reverse control: E = 8 for s <= 4, 4 for s >= 5'),
}
UNIFORM_E = list(range(2, 13))

REF = 'full'   # 'full' = all hubs (default) ; 'kas' = KAS-pruned lineage sum (removes the common hub-pruning floor)

def build(n, seed):
    W = [np.random.default_rng(seed + 1000 * k).standard_normal((n, n)) * np.sqrt(2.0 / n) for k in range(L)]
    st = k2_chain(W)
    full, sens, legs = {}, {}, {}
    for l in LAYERS:
        a, sg = st[l]['alpha'], st[l]['sigma']
        sens[l] = dict(s3=(a * phi(a) / sg**2)**2, t21=np.outer(phi(a), phi(a))**2)
        Uall, Vall, wall, per = [], [], [], {}
        for s in range(l):
            U, V = leg_at(W, st, s, l)
            p = st[s]
            w = p['sigma'] * phi(p['alpha'])
            Uall.append(U); Vall.append(V); wall.append(w)
            keep = np.abs(p['alpha']) < KAS
            per[s] = (U[:, keep], V[:, keep], w[keep])
        if REF == 'kas':
            full[l] = readouts(np.concatenate([per[s][0] for s in sorted(per)], 1),
                               np.concatenate([per[s][1] for s in sorted(per)], 1),
                               np.concatenate([per[s][2] for s in sorted(per)]))
        else:
            full[l] = readouts(np.concatenate(Uall, 1), np.concatenate(Vall, 1), np.concatenate(wall))
        legs[l] = per
    return full, sens, legs

def evaluate(Efun, n, r, full, sens, legs):
    rows = []
    for l in LAYERS:
        per = legs[l]
        ex = [per[s] for s in sorted(per) if l - s <= Efun(s)]
        old = [per[s] for s in sorted(per) if l - s > Efun(s)]
        if ex:
            Ue = np.concatenate([u for u, _, _ in ex], 1); Ve = np.concatenate([v for _, v, _ in ex], 1)
            we = np.concatenate([w for _, _, w in ex])
            D3, D21 = readouts(Ue, Ve, we); nex = Ue.shape[1]
        else:
            D3 = np.zeros(n); D21 = np.zeros((n, n)); nex = 0
        n_old = 0
        if old:
            oU = np.concatenate([u for u, _, _ in old], 1); oV = np.concatenate([v for _, v, _ in old], 1)
            ow = np.concatenate([w for _, _, w in old]); n_old = oU.shape[1]
            sw = np.sqrt(ow)[None, :]
            Q, _, _ = np.linalg.svd(np.concatenate([oU * sw, oV * sw], 1), full_matrices=False)
            Qr = Q[:, :r]
            pU = Qr @ (Qr.T @ oU); pV = Qr @ (Qr.T @ oV)
            D3b, D21b = readouts(pU, pV, ow)
            D3 = D3 + D3b; D21 = D21 + D21b
        F3, F21 = full[l]
        rows.append(dict(l=l, nex=int(nex), n_old=int(n_old), n_exact_layers=len(ex),
                         e3w=werr(D3, F3, sens[l]['s3']), e21w=werr(D21, F21, sens[l]['t21']),
                         e3=werr(D3, F3, np.ones(n)), e21=werr(D21, F21, np.ones((n, n))),
                         cost=(4.0 * nex + (6.0 * r if old else 0.0)) / n))      # n^3 units per layer
    return rows

def summarise(rows):
    S = [x for x in rows if x['l'] in SUMM]
    rms = lambda k, R: float(np.sqrt(np.mean([x[k]**2 for x in R])))
    costs = [x['cost'] for x in S]
    late = sum(x['cost'] for x in S if x['l'] in LATE)
    return dict(cost=float(np.mean(costs)), cost_max=float(np.max(costs)), late_share=float(late / sum(costs)),
                cost_profile={str(x['l']): x['cost'] for x in rows},
                e3w=rms('e3w', S), e21w=rms('e21w', S), e3=rms('e3', S), e21=rms('e21', S),
                e3w_late=rms('e3w', [x for x in S if x['l'] in LATE]),
                e21w_late=rms('e21w', [x for x in S if x['l'] in LATE]))

def pool(summaries):
    """pool seeds: mean cost, rms-over-seeds of the per-seed rms errors."""
    out = {}
    for k in ('cost', 'cost_max', 'late_share'):
        out[k] = float(np.mean([s[k] for s in summaries]))
    out['cost_profile'] = {l: float(np.mean([s['cost_profile'][l] for s in summaries]))
                           for l in summaries[0]['cost_profile']}
    for k in ('e3w', 'e21w', 'e3', 'e21', 'e3w_late', 'e21w_late'):
        out[k] = float(np.sqrt(np.mean([s[k]**2 for s in summaries])))
    return out

def interp_uniform(uni, cost, key):
    """log-linear interpolation of uniform-E error at the given mean cost; returns (value, fractional E)."""
    pts = sorted(uni.values(), key=lambda p: p['cost'])
    cs = np.array([p['cost'] for p in pts]); es = np.array([p[key] for p in pts]); Es = np.array([p['E'] for p in pts])
    if cost <= cs[0]: return float(es[0]), float(Es[0])
    if cost >= cs[-1]: return float(es[-1]), float(Es[-1])
    i = int(np.searchsorted(cs, cost)) - 1
    t = (cost - cs[i]) / (cs[i + 1] - cs[i])
    val = float(np.exp((1 - t) * np.log(max(es[i], 1e-300)) + t * np.log(max(es[i + 1], 1e-300))))
    return val, float(Es[i] + t * (Es[i + 1] - Es[i]))

def main(n, seeds, rfrac=RFRAC, tag=''):
    r = int(round(rfrac * n))
    t0 = time.time()
    per_seed = {}
    for sd in seeds:
        t1 = time.time()
        full, sens, legs = build(n, sd)
        tb = time.time() - t1
        res = {}
        for name, (Ef, desc) in SCHEDULES.items():
            rows = evaluate(Ef, n, r, full, sens, legs)
            res[name] = dict(desc=desc, E_of_s=[float(Ef(s)) for s in range(L - 1)], rows=rows, **summarise(rows))
        for E in UNIFORM_E:
            name = f'U{E}'
            if name in res: continue
            rows = evaluate(lambda s, E=E: E, n, r, full, sens, legs)
            res[name] = dict(desc=f'uniform E = {E}', E_of_s=[float(E)] * (L - 1), rows=rows, **summarise(rows))
        per_seed[sd] = res
        print(f'n={n} seed {sd}: build {tb:.1f}s, schedules {time.time() - t1 - tb:.1f}s', flush=True)
    # pool seeds
    names = list(per_seed[seeds[0]].keys())
    agg = {nm: pool([per_seed[sd][nm] for sd in seeds]) for nm in names}
    for nm in names:
        agg[nm]['desc'] = per_seed[seeds[0]][nm]['desc']; agg[nm]['E_of_s'] = per_seed[seeds[0]][nm]['E_of_s']
    uni = {E: dict(E=E, **agg[f'U{E}']) for E in UNIFORM_E}
    ref = agg['U6']
    comp = {}
    for nm in SCHEDULES:
        a = agg[nm]
        b3, Eeq = interp_uniform(uni, a['cost'], 'e3w'); b21, _ = interp_uniform(uni, a['cost'], 'e21w')
        b3l, _ = interp_uniform(uni, a['cost'], 'e3w_late'); b21l, _ = interp_uniform(uni, a['cost'], 'e21w_late')
        comp[nm] = dict(cost=a['cost'], cost_vs_U6=a['cost'] / ref['cost'] - 1.0, E_equal_cost=Eeq,
                        e3w=a['e3w'], e21w=a['e21w'], e3w_U6=ref['e3w'], e21w_U6=ref['e21w'],
                        e3w_eq=b3, e21w_eq=b21, gain3=1.0 - a['e3w'] / b3, gain21=1.0 - a['e21w'] / b21,
                        e3w_late=a['e3w_late'], e21w_late=a['e21w_late'], e3w_late_eq=b3l, e21w_late_eq=b21l,
                        gain3_late=1.0 - a['e3w_late'] / b3l, gain21_late=1.0 - a['e21w_late'] / b21l,
                        late_share=a['late_share'], cost_max=a['cost_max'],
                        profile=[a['cost_profile'][str(l)] for l in SUMM])
    out = dict(n=n, L=L, seeds=seeds, r=r, rfrac=rfrac, kas=KAS, reference=REF, layers=LAYERS, summary_layers=SUMM, late_layers=LATE,
               cost_model='per layer (4*nex + 6r)/n, n^3 units; nex = exact KAS hub columns alive; band cost only when old legs exist',
               schedules={nm: SCHEDULES[nm][1] for nm in SCHEDULES}, uniform_E=UNIFORM_E,
               agg=agg, comparison=comp,
               per_seed={str(sd): per_seed[sd] for sd in per_seed}, time=time.time() - t0)
    os.makedirs('results', exist_ok=True)
    json.dump(out, open(f'results/entry_schedule_n{n}{tag}.json', 'w'))
    # print
    print(f'\n# n={n} r={r} ref={REF}  cost = n^3 units/layer, mean over layers 9-15; errors rms over layers 9-15 (seeds pooled)')
    print(f'{"schedule":>9} {"cost":>7} {"dcost":>6} {"max":>6} {"late%":>5} {"e3w":>7} {"e21w":>7} | {"E_eq":>5} {"e3w_eq":>7} {"e21w_eq":>7} {"gain3":>6} {"gain21":>6} | {"e3wL":>7} {"e21wL":>7} {"g3L":>6} {"g21L":>6}')
    for nm, c in comp.items():
        print(f'{nm:>9} {c["cost"]:>7.2f} {100*c["cost_vs_U6"]:>+5.1f}% {c["cost_max"]:>6.2f} {100*c["late_share"]:>5.1f} '
              f'{c["e3w"]:>7.4f} {c["e21w"]:>7.4f} | {c["E_equal_cost"]:>5.2f} {c["e3w_eq"]:>7.4f} {c["e21w_eq"]:>7.4f} '
              f'{100*c["gain3"]:>+5.1f}% {100*c["gain21"]:>+5.1f}% | {c["e3w_late"]:>7.4f} {c["e21w_late"]:>7.4f} '
              f'{100*c["gain3_late"]:>+5.1f}% {100*c["gain21_late"]:>+5.1f}%')
    print('\nuniform curve:')
    for E in UNIFORM_E:
        u = uni[E]
        print(f'  U{E:<2} cost {u["cost"]:>6.2f} max {u["cost_max"]:>6.2f} e3w {u["e3w"]:.4f} e21w {u["e21w"]:.4f} e3wL {u["e3w_late"]:.4f} e21wL {u["e21w_late"]:.4f}')
    print('\ncost profile layers 9..15 (n^3/layer):')
    for nm in SCHEDULES:
        print(f'  {nm:>9} ' + ' '.join(f'{x:6.2f}' for x in comp[nm]['profile']))
    print('\nper-layer e3w / e21w layers 4..15:')
    for nm in list(SCHEDULES):
        a = agg[nm]
        rows3 = [float(np.sqrt(np.mean([per_seed[sd][nm]['rows'][i]['e3w']**2 for sd in seeds]))) for i in range(len(LAYERS))]
        rows21 = [float(np.sqrt(np.mean([per_seed[sd][nm]['rows'][i]['e21w']**2 for sd in seeds]))) for i in range(len(LAYERS))]
        print(f'  {nm:>9} e3w  ' + ' '.join(f'{x:.3f}' for x in rows3))
        print(f'  {"":>9} e21w ' + ' '.join(f'{x:.3f}' for x in rows21))
    print(f'\ntotal {time.time() - t0:.1f}s')

if __name__ == '__main__':
    # python entry_schedule.py <n> [seeds...] [--rfrac f] [--ref full|kas]
    args = sys.argv[1:]; rfrac = RFRAC; tag = ''
    if '--rfrac' in args:
        i = args.index('--rfrac'); rfrac = float(args[i + 1]); del args[i:i + 2]; tag += f'_r{rfrac:g}'
    if '--ref' in args:
        i = args.index('--ref'); REF = args[i + 1]; del args[i:i + 2]; tag += f'_ref{REF}' if REF != 'full' else ''
    n = int(args[0]); seeds = [int(x) for x in args[1:]] or [1, 2]
    main(n, seeds, rfrac, tag)
