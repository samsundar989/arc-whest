from agg import *
import warnings; warnings.simplefilter('ignore')
for obj,models in (('D21',['fix_va','fix_vc','free2','free3']),('Q31',['fix_va','fix_own','free_clos','free_poly']),('D22',['fix_va','fix_own','free_clos','free_poly'])):
    table(f'{obj}: R2 all offdiag: unweighted | WEIGHTED',[(m,R2(f'{obj}/{m}','all',False)) for m in models]+[('w:'+m,R2(f'{obj}/{m}','all',True)) for m in models],LY,fmt='%.4f',width=12)
    t=2.5
    for m in ((models[1],models[-1]) if obj!='D21' else ('fix_vc','free3')):
        table(f'{obj}/{m}: R2 by ROW class and COL class tau={t} (unweighted) + energy shares',[(rc+':'+c,R2(f'{obj}/{m}',f'{t}/{rc}/{c}')) for rc in ('row','col') for c in ('dead','kink','on')]+[('Esh:'+rc[0]+':'+c,shareE(f'{obj}/{m}',f'{t}/{rc}/{c}')) for rc in ('row','col') for c in ('dead','kink','on')],LY,fmt='%.4f',width=12)
        table(f'{obj}/{m}: R2 by ROW class and COL class tau={t} (WEIGHTED) + weighted energy shares',[(rc+':'+c,R2(f'{obj}/{m}',f'{t}/{rc}/{c}',True)) for rc in ('row','col') for c in ('dead','kink','on')]+[('wEsh:'+rc[0]+':'+c,shareE(f'{obj}/{m}',f'{t}/{rc}/{c}',True)) for rc in ('row','col') for c in ('dead','kink','on')],LY,fmt='%.4f',width=13)
for t in (2.0,3.0):
    table(f'D21/free3: R2 by ROW class tau={t} (unw, weighted)',[(c,R2('D21/free3',f'{t}/row/{c}')) for c in ('dead','kink','on')]+[('w:'+c,R2('D21/free3',f'{t}/row/{c}',True)) for c in ('dead','kink','on')],LY,fmt='%.4f')
