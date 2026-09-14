"""Dense E14 recurrence from Gaussian input, independent of active carried state.

Only the frozen Wick-coefficient evaluator/table are shared. Every moment,
fourth tensor, third tensor, projection and scalar state is rebuilt from this
oracle's own previous state. Active debug records are comparison targets only.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import flopscope.numpy as fnp
from flopscope import BudgetContext

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
import estimator_e14_tree as active
import e14_tree_oracle as tree
from e14_tree_layer_oracle import tensor_q, centered


def arr(x): return np.asarray(x,dtype=np.float64)


def wick(mu,cov,table,const):
    var=np.maximum(np.diag(cov),1e-12)
    sigma=np.sqrt(var)
    alpha=np.clip(mu/sigma,-active.ALPHA_CAP,active.ALPHA_CAP)
    # Coefficients are the supplied F21 nonlinearity; no active state is read.
    with BudgetContext(10**8):
        w=active._wick(fnp.array(mu.astype('float32')),
                       fnp.array(sigma.astype('float32')),
                       fnp.array(alpha.astype('float32')),
                       table,fnp.float32,const['wick'])
    return {k:arr(v) for k,v in w.items()}


def moment_table(S,w,table,last):
    pk={}
    for row in table['terms']:
        ip=tuple(row['int_part']); d=len(ip)
        if last and ip!=(1,): continue
        term=float(row['scalar'])
        for block in row['blocks']:
            term=term*S[tuple(block['vec'])]
        for axis,kp in enumerate(row['wick']):
            wk=w[(int(kp[0]),int(kp[1]))]
            term=term*(wk if d==1 else (wk[:,None] if axis==0 else wk[None,:]))
        pk[ip]=term if ip not in pk else pk[ip]+term
    return pk


def birth_path(A,w2):
    n=A.shape[0]; I=np.eye(n); C=A*w2[None,:]
    out=np.zeros((n,n,n))
    for f0,f1,f2 in ((A,3*I,C),):
        for perm in ((f0,f1,f2),(f0,f2,f1),(f1,f0,f2),
                     (f1,f2,f0),(f2,f0,f1),(f2,f1,f0)):
            out+=np.einsum('it,jt,kt->ijk',*perm)/6
    return out


def u_tensor(y,S):
    return (np.einsum('i,jk->ijk',y,S)+np.einsum('j,ik->ijk',y,S)
            +np.einsum('k,ij->ijk',y,S))


def oracle_layer(W,state,table,const,last):
    mu,cov,T,c4,core=state; n=len(mu); B=W.T
    pre=B@mu; cpre=B@cov@B.T
    w=wick(pre,cpre,table,const)
    old=np.einsum('ia,jb,kc,abc->ijk',B,B,B,T)
    D3=np.einsum('iii->i',old)
    D21=np.array([[old[i,i,j] if i!=j else 0.0 for j in range(n)] for i in range(n)])
    if core is None:
        dG=np.zeros(n); lam=0.0; Goff=np.zeros((n,n))
    else:
        g,lam,Cout=core
        dG=(B*B)@(g-lam*np.diag(Cout))+lam*np.diag(cpre)
        Goff=lam*((cpre+cpre.T)*0.5-np.diag(np.diag(cpre)))
    G=np.diag(dG)+Goff
    Q=tensor_q(G,2.0)
    S={(1,):pre,(3,):D3,(4,):np.full(n,c4*4.0)}
    if not last:
        mask=1-np.eye(n); C11=cpre*mask
        S.update({(2,):np.maximum(np.diag(cpre),1e-12),
                  (1,1):C11,(2,1):D21,(1,2):D21.T,
                  (2,2):np.full((n,n),c4*4/3)*mask})
    pk=moment_table(S,w,table,last)
    for p in ((1,) if last else (1,2,3,4)):
        pk[(p,)]=pk[(p,)]+np.einsum('iiii->i',Q)*w[(4,p)]/24
    if not last:
        for p,q in ((1,1),(2,1),(2,2)):
            delta=np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    if i==j: continue
                    delta[i,j]=(Q[i,i,i,j]*w[(3,p)][i]*w[(1,q)][j]/6
                               +Q[i,i,j,j]*w[(2,p)][i]*w[(2,q)][j]/4
                               +Q[i,j,j,j]*w[(1,p)][i]*w[(3,q)][j]/6)
            pk[(p,q)]=pk[(p,q)]+delta
        for key in list(pk):
            if len(key)==2 and key[0]==key[1]: pk[key]=(pk[key]+pk[key].T)*0.5
    K=centered(pk,table,last)
    if last:
        return (K[(1,)],None,None,0.0,None),{'pre':pre,'cpre':cpre,'w':w,
                                            'S':S,'pK':pk,'K':K,'Q':Q}
    cout=np.diag(K[(2,)])+K[(1,1)]
    b,a,d=w[(1,1)],w[(2,1)],w[(3,1)]
    old=old*np.einsum('i,j,k->ijk',b,b,b)
    A=b[:,None]*C11
    path=birth_path(A,a)
    y=2.0*lam/12.0*a
    X=A*b[None,:]; U=u_tensor(y,(X+X.T)*0.5)
    Tnew=old+path+U
    for i in range(n):
        Tnew[i,i,i]=K[(3,)][i]
        for j in range(n):
            if i!=j:
                Tnew[i,i,j]=K[(2,1)][i,j]
                Tnew[i,j,i]=K[(2,1)][i,j]
                Tnew[j,i,i]=K[(2,1)][i,j]
    Ctree=(cpre+cpre.T)*0.5
    dense=tree.dense_tree(Ctree,b,a,d)
    k22=((K[(2,2)]+K[(2,2)].T)*0.5)*mask
    anchored=tree.anchored(dense,K[(4,)],k22)
    H=np.einsum('abcc->ab',anchored)
    _,gnew,lamnew,_=tree.projected(H,cout)
    core_new=(gnew,lamnew,(cout+cout.T)*0.5)
    lap=table['lap2_d4']
    scalar=1.0/(8*n*(n+2))* (
        float(lap['(4,)']['terms'][0]['coef'])*
        float(lap['(4,)']['int_partition_coef'])*K[(4,)].sum()
        +float(lap['(2, 2)']['terms'][0]['coef'])*
        float(lap['(2, 2)']['int_partition_coef'])*K[(2,2)].sum())
    detail={'pre':pre,'cpre':cpre,'w':w,'S':S,'pK':pk,'K':K,'Q':Q,
            'H':H,'g':gnew,'lambda':lamnew,'T3':Tnew,'scalar':scalar}
    return (K[(1,)],cout,Tnew,scalar,core_new),detail


def controlled_weights(n,kind):
    if kind=='admissible_controlled':
        x=np.eye(n)+0.03*np.ones((n,n))
        return [x.copy() for _ in range(4)]
    rng=np.random.default_rng(818+n*10+4)
    return [rng.normal(size=(n,n))*np.sqrt(2/n) for _ in range(4)]


def compare_case(n,kind):
    # The candidate receives f32 weights; share those exact rounded operands.
    weights=[x.astype('float32').astype('float64')
             for x in controlled_weights(n,kind)]
    table=active._tables()
    est=active.Estimator(); est.setup(None); const=est._constants(n,fnp.float32)
    fs=(fnp.zeros(n,dtype=fnp.float32),fnp.eye(n,dtype=fnp.float32),
        None,0.0,None,None,None)
    active._E14_DEBUG_CAPTURE=[]; active._E14_QUERY_FUSED=True
    try:
        with BudgetContext(10**11):
            fstates=[]
            for i,W in enumerate(weights):
                fs=active._layer_tree(fnp.array(W.astype('float32')),fs,table,
                                      fnp.float32,const,2.0,i==3,None)
                fstates.append(fs)
        traces=active._E14_DEBUG_CAPTURE
    finally: active._E14_DEBUG_CAPTURE=None
    os=(np.zeros(n),np.eye(n),np.zeros((n,n,n)),0.0,None)
    details=[]
    for i,W in enumerate(weights):
        os,detail=oracle_layer(W,os,table,const,i==3)
        ft=fstates[i]; tr=traces[i]
        got={k:arr(v) for k,v in tr['K'].items()}
        gaps={'mean':float(np.max(abs(os[0]-arr(ft[0]))))}
        for key,v in detail['K'].items(): gaps['K'+str(key)]=float(np.max(abs(v-got[key])))
        for key,v in detail['pK'].items():
            gaps['pK'+str(key)]=float(np.max(abs(v-arr(tr['pK'][key]))))
        if i<3:
            from e14_tree_layer_oracle import dense_tower
            dense=dense_tower(ft[2],ft[5],ft[6])
            gaps.update({'cov':float(np.max(abs(os[1]-arr(ft[1])))),
                         'K3_all':float(np.max(abs(os[2]-dense))),
                         'scalar':abs(os[3]-float(ft[3])),
                         'g':float(np.max(abs(os[4][0]-arr(ft[4][0])))),
                         'lambda':abs(os[4][1]-float(ft[4][1]))})
        details.append({'layer':i,'errors':gaps,'max_error':max(gaps.values()),
            'oracle':{'pre':detail['pre'].tolist(),'cpre':detail['cpre'].tolist(),
                      'w':{str(k):v.tolist() for k,v in detail['w'].items()},
                      'S':{str(k):v.tolist() for k,v in detail['S'].items()},
                      'Q':detail['Q'].tolist(),
                      'mean':os[0].tolist(),'cov':None if os[1] is None else os[1].tolist(),
                      'T3':None if os[2] is None else os[2].tolist(),
                      'c4':os[3],'core':None if os[4] is None else
                      {'g':os[4][0].tolist(),'lambda':os[4][1],'Cout':os[4][2].tolist()},
                      'pK':{str(k):v.tolist() for k,v in detail['pK'].items()},
                      'K':{str(k):v.tolist() for k,v in detail['K'].items()}},
            'active':{'mean':arr(ft[0]).tolist(),
                      'pK':{str(k):arr(v).tolist() for k,v in tr['pK'].items()},
                      'K':{str(k):arr(v).tolist() for k,v in tr['K'].items()},
                      'cov':None if ft[1] is None else arr(ft[1]).tolist(),
                      'c4':float(ft[3])}})
    return {'n':n,'kind':kind,'weights':[x.tolist() for x in weights],
            'layers':details,'max_error':max(x['max_error'] for x in details)}


if __name__=='__main__':
    out=ROOT/'experiments'/'e14_tree_candidate'/'full_oracle_v4.json'
    result={'cases':[],'scope':'independent dense Gaussian-to-depth4 state; common Wick table evaluator only; no labels'}
    try:
        for n in (3,4):
            for kind in ('admissible_controlled','negative_variance'):
                result['cases'].append(compare_case(n,kind))
        result['max_error']=max(c['max_error'] for c in result['cases'])
        result['declared_f32_bound']=2e-5
        result['passed']=result['max_error']<=result['declared_f32_bound']
        first=next(((c,layer) for c in result['cases'] for layer in c['layers']
                    if layer['max_error']>result['declared_f32_bound']),None)
        if first is not None:
            case,layer=first
            field=max(layer['errors'],key=layer['errors'].get)
            aa=np.array(layer['active']['pK']['(4,)'])
            oo=np.array(layer['oracle']['pK']['(4,)'])
            j=int(np.argmax(abs(aa-oo)))
            result['first_falsifier']={'n':case['n'],'kind':case['kind'],
                'layer':layer['layer'],'field':field,'index':j,'active':float(aa[j]),
                'oracle':float(oo[j]),'absolute_gap':float(abs(aa[j]-oo[j])),
                'pre':layer['oracle']['pre'],'cpre':layer['oracle']['cpre'],
                'S3':layer['oracle']['S']['(3,)'],
                'S4':layer['oracle']['S']['(4,)'],
                'w_0_4':layer['oracle']['w']['(0, 4)'],
                'w_3_4':layer['oracle']['w']['(3, 4)'],
                'w_4_4':layer['oracle']['w']['(4, 4)']}
    except Exception as exc:
        result['first_exception']=repr(exc)
        out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        raise
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'max_error':result['max_error'],'passed':result['passed'],
                      'first_falsifier':result.get('first_falsifier')},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
