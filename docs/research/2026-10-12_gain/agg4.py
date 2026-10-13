from agg import *
import warnings; warnings.simplefilter('ignore')
table('PRUNE: rms_b(dvar_b/var_b) full effect / everywhere-replaced residuals',[('D21 full',P('D21/all')),('+diag',P('D21+diag/all')),('noise',P('noise/all')),('fix_va',P('fix_va/all')),('fix_vc',P('fix_vc/all')),('ln2_vc',P('ln2_vc/all')),('free2',P('free2/all')),('free3',P('free3/all')),('free3N',P('free3N/all')),('defl1',P('free3_defl1/all')),('defl4',P('free3_defl4/all')),('defl16',P('free3_defl16/all'))],PL,fmt='%.3g',width=11)
table('PRUNE: mean_b(dvar_b/var_b) (bias)',[('D21 full',P('D21/all',1)),('fix_vc',P('fix_vc/all',1)),('free3',P('free3/all',1)),('free3N',P('free3N/all',1))],PL,fmt='%.3g',width=11)
for t in (2.0,2.5,3.0):
    for cn in ('on','dead','dead+on'):
        cols=[('ZERO',P(f'D21/rows/{t}/{cn}')),('fix_va',P(f'fix_va/rows/{t}/{cn}')),('fix_vc',P(f'fix_vc/rows/{t}/{cn}')),('vc_kinkfit',P(f'fix_vckink/{t}/rows/{t}/{cn}')),('ln2_vc',P(f'ln2_vc/rows/{t}/{cn}')),('free2',P(f'free2/rows/{t}/{cn}')),('free3',P(f'free3/rows/{t}/{cn}')),('f3_kinkfit',P(f'free3_kinkfit/{t}/rows/{t}/{cn}')),('f3_own',P(f'free3_own/{t}/rows/{t}/{cn}')),('f3N_kinkfit',P(f'free3N_kinkfit/{t}/rows/{t}/{cn}')),('defl1',P(f'free3_defl1/rows/{t}/{cn}')),('defl16',P(f'free3_defl16/rows/{t}/{cn}')),('noise',P(f'noise/rows/{t}/{cn}')),('biasZERO',P(f'D21/rows/{t}/{cn}',1)),('bias_f3k',P(f'free3_kinkfit/{t}/rows/{t}/{cn}',1))]
        r=table(f'PRUNE rows "{cn}" tau={t}: rms_b(dvar/var) error when rows are ZEROED vs REPLACED by gain model',cols,PL,fmt='%.3g',width=12)
    cols=[(f'{nm}:{cn}',P(f'{nm}/cols/{t}/{cn}')) for cn in ('dead','on') for nm in ('D21','fix_va','fix_vc','free3','noise')]
    table(f'PRUNE cols tau={t}: zero (D21) vs replaced',cols,PL,fmt='%.3g',width=13)
