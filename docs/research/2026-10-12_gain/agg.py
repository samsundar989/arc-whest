import json, glob, sys, numpy as np
files=sorted(glob.glob('out_gain_0??.json'))
D=[json.load(open(f)) for f in files]
NL=16
print('nets:',[f[9:12] for f in files])
def get(fn,layers=range(NL)):
    M=np.full((len(D),NL),np.nan)
    for i,d in enumerate(D):
        for l in layers:
            try:
                x=fn(d[str(l)])
                if x is not None: M[i,l]=x
            except (KeyError,ZeroDivisionError,TypeError): pass
    return np.nanmean(M,0),np.nanstd(M,0)
def table(title,cols,layers=range(NL),fmt='%9.3g',sd=True,width=None):
    print('\n## '+title)
    w=width or max(10,max(len(c[0]) for c in cols)+1)
    print('  l '+''.join(c[0].rjust(w) for c in cols))
    res=[get(c[1],layers) for c in cols]
    for l in layers:
        print('%3d '%l+''.join((fmt%r[0][l]).rjust(w) for r in res))
    ll=list(layers)
    print('avg '+''.join((fmt%np.nanmean(r[0][ll])).rjust(w) for r in res))
    print('max '+''.join((fmt%np.nanmax(r[0][ll])).rjust(w) for r in res))
    if sd:
        print('rsd '+''.join(('%.2g'%np.nanmedian(r[1][ll]/np.abs(r[0][ll]))).rjust(w) for r in res)+'   (net-to-net sd/mean, median over layers)')
    return res
LY=list(range(1,16)); PL=list(range(1,15))
def R2(key,sub,w=False):
    def f(d):
        x=d['R'][key][sub]
        return (1-x[2]/x[3]) if w else (1-x[0]/x[1])
    return f
def shareE(key,sub,w=False):
    def f(d):
        x=d['R'][key][sub]; a=d['R'][key]['all']
        return x[3]/a[3] if w else x[1]/a[1]
    return f
def P(key,i=0): return lambda d:d['P'][key][i]
def Mq(key,i=0): return lambda d:d['P']['mean'][key][i]
def Sq(*ks):
    def f(d):
        x=d['S']
        for k in ks: x=x[k]
        return x
    return f
