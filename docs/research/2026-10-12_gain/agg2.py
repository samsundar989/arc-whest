from agg import *
import warnings; warnings.simplefilter('ignore')
for obj,models in (('D3',['fix_va','fix_vc','lin_int','free2','free3','free2N']),('D4',['fix_va','fix_own_const','fix_own','ln2_own','free_clos','free_clos2','free_poly'])):
    for w in (False,True):
        table(f'{obj}: R2 (1-SSE/SST uncentered, sigma-normalised) {"WEIGHTED" if w else "unweighted"} all neurons',[(m,R2(f'{obj}/{m}','all',w)) for m in models],LY,fmt='%.4f',width=14)
    for t in (2.5,):
        for m in ([models[0],'free2','free2N'] if obj=='D3' else ['fix_va','fix_own_const','free_poly']):
            table(f'{obj}/{m}: R2 by class tau={t} (unweighted | weighted) + energy share',[(c,R2(f'{obj}/{m}',f'{t}/{c}')) for c in ('dead','kink','on')]+[('w:'+c,R2(f'{obj}/{m}',f'{t}/{c}',True)) for c in ('dead','kink','on')]+[('Esh:'+c,shareE(f'{obj}/{m}',f'{t}/{c}')) for c in ('dead','kink','on')]+[('wEsh:'+c,shareE(f'{obj}/{m}',f'{t}/{c}',True)) for c in ('dead','kink','on')],LY,fmt='%.4f',width=11)
for t in (2.0,3.0):
    table(f'D3/free2: R2 by class tau={t} (unw, weighted)',[(c,R2('D3/free2',f'{t}/{c}')) for c in ('dead','kink','on')]+[('w:'+c,R2('D3/free2',f'{t}/{c}',True)) for c in ('dead','kink','on')],LY,fmt='%.4f')
