"""Markdown tables for I_x6_sandbox.md from the x6_an_*.json / x6_squeeze_*.json files.  python3 x6_tables.py > results/x6_tables.md"""
import json, os, glob
R = '/home/claude/sandbox/results'
def J(v, n, L, s, ref='full'):
    f = f'{R}/x6_an_{v}_vs_{ref}_n{n}_L{L}_s{s}.json'
    return json.load(open(f)) if os.path.exists(f) else None
def g(d, *keys, default=float('nan')):
    for k in keys:
        if d is None: return default
        d = d.get(str(k), d.get(k)) if isinstance(d, dict) else None
    return default if d is None else d
def f3(x):
    try: return f'{x:.3f}' if x == x else '–'
    except Exception: return '–'
def f2(x):
    try: return f'{x:.2f}' if x == x else '–'
    except Exception: return '–'

print('## T1. Chains: final-layer MSE vs MC and ratio to the full first-order tower (L = 8)\n')
print('| chain | n=64 s0 | n=64 s1 | n=128 s0 | n=256 s0 |')
print('|---|---|---|---|---|')
names = dict(full='full tower (reference)', age2='ages ≤ 2 (direct+indirect)', age3='ages ≤ 3', age4='ages ≤ 4', ind3='direct all, indirect ages ≤ 3', dp='born diag+pair only (no hub)', gate='gate transport', k2='K2 only')
for v in ['full', 'age2', 'age3', 'age4', 'ind3', 'dp', 'gate', 'k2']:
    cells = []
    for n, s in [(64, 0), (64, 1), (128, 0), (256, 0)]:
        d = dict(__import__('numpy').load(f'{R}/x6_chains_n{n}_L8_s{s}.npz')) if os.path.exists(f'{R}/x6_chains_n{n}_L8_s{s}.npz') else {}
        if f'{v}_mse' in d and 'full_mse' in d:
            cells.append(f'{float(d[f"{v}_mse"]):.2e} ({float(d[f"{v}_mse"])/float(d["full_mse"]):.2f}×)')
        else: cells.append('–')
    print(f'| {names[v]} | ' + ' | '.join(cells) + ' |')

def layer_table(v, n, L, s, layers, ref='full', title=None):
    r = J(v, n, L, s, ref)
    if r is None: print(f'(missing {v} n={n} L={L} s={s})'); return
    print(f'\n### {title or v}: n={n}, L={L}, seed={s}, reference={ref}\n')
    print('| l | off-ch share | PR(O_l) | fro1 | fro2 | fro4 | fro8 | fro16 | fro32 | ndR²4 | ndR²16 | ndR²32 | GOE R²16/32 | scr R²16/32 | D2 k=8/16/32 | D4 k=16 (loc/rnd) | I/dC | span1_16 | core1_16 | R²(loc) | tpl | win4 / null |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for l in layers:
        Ly = r['layers'].get(str(l))
        if not Ly or not Ly.get('B'): continue
        B = Ly['B']; D2 = r.get('D2', {}).get(str(l), {}); D4 = r.get('D4', {}).get(str(l), {}); G = Ly.get('G', {}); D = Ly.get('D', {}); E = Ly.get('E', {}); F = Ly.get('F', {}); H = Ly.get('H', {})
        print(f"| {l} | {f3(Ly.get('off_ch'))} | {B['PR']:.1f} | " + ' | '.join(f3(g(B['fro'], k)) for k in [1, 2, 4, 8, 16, 32]) + ' | ' + ' | '.join(f3(g(B['R2nd'], k)) for k in [4, 16, 32])
              + f" | {f2(g(B['goe_R2'], 16))}/{f2(g(B['goe_R2'], 32))} | {f2(g(B['scr_R2'], 16))}/{f2(g(B['scr_R2'], 32))} | {f2(g(D2, 8))}/{f2(g(D2, 16))}/{f2(g(D2, 32))} | {f2(g(D4, 'loc', 16))}/{f2(g(D4, 'rnd', 16))} | {f2(D.get('I_frac'))} | {f2(g(G, 'span1', 16))} | {f2(g(G, 'core1', 16))} | {f2(E.get('R2_loc'))} | {f2(F.get('tpl'))} | {f2(H.get('win4', [float('nan')])[0])} / {f2(H.get('wrnd4', [float('nan')])[0])} |")

print('\n## T2. Per-layer rank / transport diagnostics, n = 128, L = 16 (the depth of the real chain)\n')
print('Columns: off-ch share = share of ‖δvar_{l+1}‖² from the off-diagonal sandwich; PR = participation ratio of the spectrum of O_l; fro_k = fraction of ‖O_l‖_F² in the top-k eigenpairs; ndR²_k = R² of the rank-k sandwich readout (diagonal of the truncation removed) against δvar^off_{l+1}; GOE/scr = same R² for a GOE matrix / sign-scrambled O_l; D2 = R² on δvar_l of a rank-k state with oracle innovation every layer; D4 = same when new directions may enter only through 10 local vectors per layer (loc) or 10 random vectors (rnd), oracle core; I/dC = innovation share of ‖δC_l‖²; span1/core1 = one-step persistence of the top-16 directions (span fraction / oracle-core R²); R²(loc) = OLS R² of δvar_l on per-neuron features; tpl = share of ‖O_l‖² in the m mᵀ + Cov templates; win4 = share of ‖δC_l‖² in the span of 4 local vectors × last 6 layers transported (null = random span of the same dimension).')
for v, t in [('dp', 'missing hub-born terms (incoherent born pattern)'), ('age3', 'lineages older than 3 layers dropped (coherent lineage content)'), ('k2', 'K2-only chain (everything κ3 missing)')]:
    layer_table(v, 128, 16, 0, [3, 4, 6, 8, 10, 12, 13, 14, 15], title=t)

print('\n## T3. Same diagnostics at L = 8 for all chain variants (n = 128) — layers 5–7\n')
for v in ['age3', 'age4', 'ind3', 'dp', 'gate', 'k2']:
    layer_table(v, 128, 8, 0, [3, 5, 6, 7], title=names[v])
print('\n### full tower vs MC truth (second-order residual), n = 128, L = 8\n')
layer_table('full', 128, 8, 0, [3, 4, 5, 6, 7], ref='mc', title='full tower vs MC')

print('\n## T4. n-scaling at L = 8 (layer 7 = last post-ReLU covariance): fraction of ‖O_7‖_F² in top-k and GOE null\n')
null = json.load(open(f'{R}/x6_null.json'))
print('| error | n | PR | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | ndR²_16 | ndR²_32 | GOE null fro16 | fro32 |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for v, ref, label in [('dp', 'full', 'dp (hub missing)'), ('age3', 'full', 'age ≤ 3'), ('k2', 'full', 'K2 only'), ('full', 'mc', 'full vs MC (2nd order)')]:
    for n, s in [(64, 0), (64, 1), (128, 0), (256, 0)]:
        r = J(v, n, 8, s, ref)
        if r is None: continue
        B = r['layers']['7']['B']; nl = null[str(n)]['fro']
        print(f"| {label} | {n} (s{s}) | {B['PR']:.1f} | " + ' | '.join(f3(g(B['fro'], k)) for k in [1, 2, 4, 8, 16, 32, 64]) + f" | {f3(g(B['R2nd'], 16))} | {f3(g(B['R2nd'], 32))} | {f3(g(nl, 16))} | {f3(g(nl, 32))} |")

print('\n## T5. n-scaling at L = 16 (layers 12, 14, 15)\n')
print('| error | n | l | PR | fro1 | fro2 | fro4 | fro8 | fro16 | fro32 | D2 k=16 | D2 k=32 | D4 k=16 | I/dC | span1_16 | R²(loc) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for v in ['dp', 'age3']:
    for n, s in [(64, 0), (64, 1), (128, 0), (256, 0)]:
        r = J(v, n, 16, s)
        if r is None: continue
        for l in [12, 14, 15]:
            Ly = r['layers'][str(l)]; B = Ly['B']; D2 = r['D2'].get(str(l), {}); D4 = r.get('D4', {}).get(str(l), {})
            print(f"| {v} | {n} (s{s}) | {l} | {B['PR']:.1f} | " + ' | '.join(f3(g(B['fro'], k)) for k in [1, 2, 4, 8, 16, 32]) + f" | {f3(g(D2, 16))} | {f3(g(D2, 32))} | {f3(g(D4, 'loc', 16))} | {f2(Ly.get('D', {}).get('I_frac'))} | {f2(g(Ly.get('G', {}), 'span1', 16))} | {f2(Ly.get('E', {}).get('R2_loc'))} |")

print('\n## T6. Squeezed-GOE null (pure transport geometry of the He-init chain, no cumulants): incoherent unit-norm innovation(s) transported by W G, L = 16\n')
print('| injection | n | l | PR | transported share | fro8 | fro16 | fro32 | fro64 | R²_16 | R²_32 | R²_64 |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for n in [128, 256, 512, 1024]:
    f = f'{R}/x6_squeeze_n{n}_L16_s0.json'
    if not os.path.exists(f): continue
    q = json.load(open(f))
    for mode, lab in [('single', 'single, at l=2'), ('mixture', 'one per layer (equal norm)')]:
        for l in [8, 12, 14, 16]:
            a = q[mode].get(str(l))
            if a is None: continue
            print(f"| {lab} | {n} | {l} | {a['PR']:.1f} | {f2(a.get('trans', float('nan'))) if mode == 'mixture' else '1'} | " + ' | '.join(f3(g(a['fro'], k)) for k in [8, 16, 32, 64]) + ' | ' + ' | '.join(f3(g(a['R2'], k)) for k in [16, 32, 64]) + ' |')
