"""Compact cross-run summary of the x6_analysis json files.  python3 x6_summary.py n L seed trunc [ref]"""
import json, sys
def show(n, L, seed, trunc, ref='full'):
    r = json.load(open(f'/home/claude/sandbox/results/x6_an_{trunc}_vs_{ref}_n{n}_L{L}_s{seed}.json'))
    ratio = (r['mse_tr']/r['mse_full']) if r.get('mse_tr') and r.get('mse_full') else float('nan')
    print(f"\n=== n={n} L={L} s={seed} {trunc} vs {ref}: MSE ratio {ratio:.2f}")
    print("l  off_ch | PR_O  fro16 fro32  R2_16 R2_32 | GOE16 GOE32 | scr16 scr32 | D2_16 D2_32 | R2loc | I/dC | tpl  offE PR_R  rR2_16 rR2_32 rscr16 rscr32")
    for l in range(1, L+1):
        Ly = r['layers'][str(l)]; B = Ly.get('B')
        if not B: continue
        D2 = r['D2'].get(str(l), {}); E = Ly.get('E', {}); D = Ly.get('D', {}); F = Ly.get('F', {})
        print(f"{l:<2d} {Ly['off_ch']:.3f} | {B['PR']:5.1f} {B['fro']['16']:.3f} {B['fro']['32']:.3f}  {B['R2']['16']:.3f} {B['R2']['32']:.3f} | {B['goe_R2']['16']:.3f} {B['goe_R2']['32']:.3f} | {B['scr_R2']['16']:.3f} {B['scr_R2']['32']:.3f} | "
              f"{D2.get('16', float('nan')):.3f} {D2.get('32', float('nan')):.3f} | {E.get('R2_loc', float('nan')):.3f} | {D.get('I_frac', float('nan')):.3f} | "
              f"{F.get('tpl', float('nan')):.3f} {F.get('offE', float('nan')):.3f} {F.get('PR_R', float('nan')):5.1f}  {F.get('R2', {}).get('16', float('nan')):.3f}  {F.get('R2', {}).get('32', float('nan')):.3f}  {F.get('scr_R2', {}).get('16', float('nan')):.3f}  {F.get('scr_R2', {}).get('32', float('nan')):.3f}")
    if 'C_gate' in r:
        print('C (gate transport from l0): l -> pure16 core16 fullT :', {l: (round(x['pure']['16'], 3), round(x['core']['16'], 3), round(x['fullT'], 3)) for l, x in r['C_gate'].items()})
if __name__ == '__main__':
    n, L, seed, trunc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    show(n, L, seed, trunc, sys.argv[5] if len(sys.argv) > 5 else 'full')
