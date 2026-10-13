from agg import *
import warnings; warnings.simplefilter('ignore')
table('MEAN channel: rms_a(w3 D3/6) full, and k4 (w4 k4/24) full; [next-layer mu rms]',[('k3 full',Mq('full')),('k3 full->mu',Mq('full',1)),('k4 full',Mq('full_k4')),('k4 full->mu',Mq('full_k4',1))],PL,fmt='%.3g',width=13)
for t in (2.0,2.5,3.0):
    for sn in ('dead','on','dead+on','all'):
        cols=[('n',lambda d,t=t,sn=sn:d['P']['mean'][f'{t}/{sn}/n'])]+[(m,Mq(f'{t}/{sn}/{m}')) for m in ('zero','fix_va','fix_vc','fix_vkeep','free2','free2_keep','free2N_keep','k4_zero','k4_fixva','k4_poly')]+[('zero->mu',Mq(f'{t}/{sn}/zero',1)),('f2keep->mu',Mq(f'{t}/{sn}/free2_keep',1))]
        if sn=='all' and t!=2.5: continue
        r=table(f'MEAN D3 set "{sn}" tau={t}: rms_a(dm_a) abs',cols,PL,fmt='%.3g',width=12)
