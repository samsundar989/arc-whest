"""Independent dense n=3/4 audit of E14's first differing nonlinear layer.

Uses the active layer's supplied F21 Wick/table state, then independently
enumerates the six-term Q, finite hyperedge response, centering partitions,
and every entry of the resulting third-cumulant tensor. No labels are read.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import flopscope.numpy as fnp
from flopscope import BudgetContext

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import estimator_e14_tree as e


def tensor_q(g, metric):
    n=g.shape[0]; q=np.zeros((n,)*4,dtype=np.float64)
    eye=np.eye(n)
    for i in range(n):
      for j in range(n):
       for k in range(n):
        for l in range(n):
         q[i,j,k,l]=metric*(g[i,j]*eye[k,l]+g[i,k]*eye[j,l]+g[i,l]*eye[j,k]
                    +g[j,k]*eye[i,l]+g[j,l]*eye[i,k]+g[k,l]*eye[i,j])/6
    return q


def centered(pk, table, last=False):
    def block(b,d):
        nz=[i for i,x in enumerate(b) if x>0]
        key=tuple(sorted((b[i] for i in nz),reverse=True))
        x=pk[key]
        if len(nz)==2 and b[nz[0]]<b[nz[1]]: x=x.T
        if len(nz)!=d: x=x[:,None] if nz==[0] else x[None,:]
        return x
    result={}
    for key, rows in table['pK_to_K'].items():
        ip=ast.literal_eval(key); d=len(ip)
        if last and ip!=(1,): continue
        acc=None
        for row in rows:
            prod=1.0
            for b in row['blocks']: prod=prod*block(tuple(int(x) for x in b),d)
            term=float(row['coef'])*prod
            acc=term if acc is None else acc+term
        result[ip]=acc
    return result


def dense_tower(tw,Y,R):
    n=tw.A_D.shape[0]; k=tw.k
    AD=np.asarray(tw.A_D,dtype=np.float64)
    AL=np.asarray(tw.A_L,dtype=np.float64)
    P=np.eye(n) if tw.M is None else np.concatenate([np.asarray(tw.M),np.eye(n)],axis=1)
    w=np.asarray(tw.w2b,dtype=np.float64)
    yy=np.asarray(Y,dtype=np.float64); rr=np.asarray(R,dtype=np.float64)
    out=np.zeros((n,n,n),dtype=np.float64)
    for age in range(k):
        sl=slice(age*n,(age+1)*n)
        D,A,M=AD[:,sl],AL[:,sl],P[:,sl]
        C=A*w[sl][None,:]
        # All factor permutations are enumerated; no reduced readout formula.
        factors=((D,M,M),(A,3*M,C))
        for f0,f1,f2 in factors:
            for perm in ((f0,f1,f2),(f0,f2,f1),(f1,f0,f2),
                         (f1,f2,f0),(f2,f0,f1),(f2,f1,f0)):
                out+=np.einsum('it,jt,kt->ijk',*perm)/6
        X=(A*rr[:,age][None,:])@M.T
        S=(X+X.T)*0.5; y=yy[:,age]
        out+=np.einsum('i,jk->ijk',y,S)+np.einsum('j,ik->ijk',y,S)+np.einsum('k,ij->ijk',y,S)
    return out


def dense_birth(A, w2, y, r):
    n=A.shape[0]; I=np.eye(n); C=A*w2[None,:]
    out=np.zeros((n,n,n),dtype=np.float64)
    for f0,f1,f2 in ((A,3*I,C),):
        for perm in ((f0,f1,f2),(f0,f2,f1),(f1,f0,f2),
                     (f1,f2,f0),(f2,f0,f1),(f2,f1,f0)):
            out+=np.einsum('it,jt,kt->ijk',*perm)/6
    X=A*r[None,:]; S=(X+X.T)*0.5
    out+=np.einsum('i,jk->ijk',y,S)+np.einsum('j,ik->ijk',y,S)+np.einsum('k,ij->ijk',y,S)
    return out


def run_case(n,depth,layer,kind):
    if kind=='admissible_controlled':
        x=np.eye(n)+0.03*np.ones((n,n))
        W=[fnp.array(x.astype('float32')) for _ in range(depth)]
    else:
        rng=np.random.default_rng(818+n*10+depth)
        W=[fnp.array((rng.normal(size=(n,n))*np.sqrt(2/n)).astype('float32'))
           for _ in range(depth)]
    est=e.Estimator(); est.setup(None); const=est._constants(n,fnp.float32)
    state=(fnp.zeros(n,dtype=fnp.float32),fnp.eye(n,dtype=fnp.float32),
           None,0.0,None,None,None)
    e._E14_DEBUG_CAPTURE=[]
    try:
        with BudgetContext(10**11):
            states=[]
            for l in range(depth):
                state=e._layer_tree(W[l],state,e._tables(),fnp.float32,const,2.0,
                                    l==depth-1,None)
                states.append(state)
        trace=e._E14_DEBUG_CAPTURE
    finally:
        e._E14_DEBUG_CAPTURE=None
    # Each boundary is checked from its independently supplied prior state.
    tr=trace[layer]; out=states[layer]; prev=states[layer-1]
    conv=lambda x: np.asarray(x,dtype=np.float64)
    core=tr['incoming_core']; g,lam,Cout=(conv(core[0]),float(core[1]),conv(core[2]))
    B=conv(W[layer]).T; cpre=conv(tr['cov_pre']); n=B.shape[0]
    gout=np.diag(g)+lam*(Cout-np.diag(np.diag(Cout)))
    dslow=np.diag(B@gout@B.T)
    dfast=(B*B)@(g-lam*np.diag(Cout))+lam*np.diag(cpre)
    G=np.diag(dfast)+lam*((cpre+cpre.T)*0.5-np.diag(np.diag(cpre)))
    Q=tensor_q(G,2.0)
    w={key:conv(value) for key,value in tr['w'].items()}
    base={key:conv(value).copy() for key,value in tr['pK_base'].items()}
    pk={key:value.copy() for key,value in base.items()}
    for p in (1,2,3,4):
        pk[(p,)]+=np.einsum('iiii->i',Q)*w[(4,p)]/24
    for p,q in ((1,1),(2,1),(2,2)):
        delta=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i==j: continue
                delta[i,j]=(Q[i,i,i,j]*w[(3,p)][i]*w[(1,q)][j]/6
                           +Q[i,i,j,j]*w[(2,p)][i]*w[(2,q)][j]/4
                           +Q[i,j,j,j]*w[(1,p)][i]*w[(3,q)][j]/6)
        pk[(p,q)]+=delta
    for key in list(pk):
        if len(key)==2 and key[0]==key[1]: pk[key]=(pk[key]+pk[key].T)*0.5
    want=centered(pk,e._tables())
    got={key:conv(value) for key,value in tr['K'].items()}
    errs={'dG_identity':float(np.max(abs(dslow-dfast))),
          'q4':float(np.max(abs(conv(tr['q4'])-np.einsum('iiii->i',Q)))),
          'q31':float(np.max(abs(conv(tr['q31'])-np.array([[Q[i,i,i,j] if i!=j else 0
                                               for j in range(n)] for i in range(n)])))),
          'q22':float(np.max(abs(conv(tr['q22'])-np.array([[Q[i,i,j,j] if i!=j else 0
                                               for j in range(n)] for i in range(n)]))))}
    for key in pk:
        errs['pK'+str(key)]=float(np.max(abs(pk[key]-conv(tr['pK'][key]))))
    for key in want:
        errs['K'+str(key)]=float(np.max(abs(want[key]-got[key])))
    # Full dense tower: every i,j,k, including all-distinct and repeats.
    full=dense_tower(out[2],out[5],out[6])
    errs['K3_repeated3']=float(np.max(abs(np.einsum('iii->i',full)-got[(3,)])))
    errs['K3_repeated21']=float(max(abs(full[i,i,j]-got[(2,1)][i,j])
                                       for i in range(n) for j in range(n) if i!=j))
    old=dense_tower(prev[2],prev[5],prev[6])
    old=np.einsum('ia,jb,kc,abc->ijk',B,B,B,old)
    b=w[(1,1)]; old*=np.einsum('i,j,k->ijk',b,b,b)
    C11=cpre*(1-np.eye(n))
    A0=b[:,None]*C11
    newborn=dense_birth(A0,w[(2,1)],conv(tr['y0']),b)
    expected=old+newborn
    for i in range(n):
        expected[i,i,i]=want[(3,)][i]
        for j in range(n):
            if i!=j:
                expected[i,i,j]=want[(2,1)][i,j]
                expected[i,j,i]=want[(2,1)][i,j]
                expected[j,i,i]=want[(2,1)][i,j]
    errs['K3_all_entries']=float(np.max(abs(expected-full)))
    return {'n':n,'depth':depth,'layer':layer,'kind':kind,
            'errors':errs,'max_error':max(errs.values()),
            'source_ages':out[2].k,
            'first_operands':{'W':conv(W[layer]).tolist(),'Cpre':cpre.tolist(),
                'Cout_s':Cout.tolist(),'g':g.tolist(),'lambda':lam,
                'dG_direct':dslow.tolist(),'dG_identity':dfast.tolist(),
                'Q':Q.tolist(),'pK_base':{str(k):v.tolist() for k,v in base.items()},
                'pK_oracle':{str(k):v.tolist() for k,v in pk.items()},
                'K_oracle':{str(k):v.tolist() for k,v in want.items()},
                'K_active':{str(k):v.tolist() for k,v in got.items()},
                'K3_dense_tower':full.tolist(),
                'K3_independent_expected':expected.tolist(),
                'K3_old_transport_wick':old.tolist(),
                'K3_newborn_path_plus_U':newborn.tolist()}}


if __name__=='__main__':
    out=ROOT/'experiments'/'e14_tree_candidate'/'layer_oracle_v3.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    result={'cases':[],'scope':'independent dense Q/manifest/centering and full K3 across two successive active nonlinear layers on supplied closure states'}
    try:
        for n in (3,4):
            for kind in ('admissible_controlled','negative_variance'):
                for layer in (1,2): result['cases'].append(run_case(n,4,layer,kind))
        result['max_error']=max(c['max_error'] for c in result['cases'])
        result['passed']=result['max_error']<2e-5
    except Exception as exc:
        result['passed']=False; result['first_exception']=repr(exc)
        out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        raise
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'max_error':result['max_error'],'passed':result['passed']},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
