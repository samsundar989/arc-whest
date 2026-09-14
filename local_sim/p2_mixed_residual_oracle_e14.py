"""Independent dense f64 source oracle for exposed mixed-kernel traces."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import p2_compact_fullrank_f32_e14 as fixture

ROOT=Path(__file__).resolve().parents[1]


def values(x):
    return None if x is None else np.asarray(x['values'],dtype=np.float64)


def dense_source_slices(D,A,P,w):
    """Enumerate all permutations, independently of the direct query grouping."""
    n=A.shape[0];C=A*w[None,:]
    T=np.zeros((n,n,n),dtype=np.float64)
    from itertools import permutations
    for i in range(n):
        for j in range(n):
            for k in range(n):
                out=0.0
                for q in range(A.shape[1]):
                    legs=(D[:,q],P[:,q],P[:,q])
                    out+=sum(legs[p[0]][i]*legs[p[1]][j]*legs[p[2]][k]
                             for p in ((0,1,2),(1,0,2),(1,2,0)))/3.0
                    legs=(A[:,q],P[:,q],C[:,q])
                    out+=sum(legs[p[0]][i]*legs[p[1]][j]*legs[p[2]][k]
                             for p in permutations((0,1,2)))/2.0
                T[i,j,k]=out
    return np.array([T[i,i,i] for i in range(n)]),np.array([[T[i,i,j] for j in range(n)] for i in range(n)])


def factorized_f64_slices(D,A,P,w):
    C=A*w[None,:]
    d3=np.sum(D*P*P,axis=1)+3*np.sum(A*P*C,axis=1)
    d21=(2*(D*P)@P.T+(P*P)@D.T+3*(A*C)@P.T+6*(A*P)@C.T)/3
    return d3,d21


def check(trace_path,weights):
    data=json.loads(Path(trace_path).read_text())
    rows=[]
    for layer in range(2,len(weights)):
        tw=data['rows'][layer-1]['tower'];n=weights.shape[1];k=tw['k']
        W=np.asarray(weights[layer],dtype=np.float32).astype(np.float64).T
        D=W@values(tw['D'])
        Q=values(tw['Q']);FA=values(tw['FA']);FP=values(tw['FP'])
        Aold=np.empty((n,0)) if Q is None else W@(Q@FA)
        Pold=np.empty((n,0)) if Q is None else W@(Q@FP)
        Ay=W@values(tw['young_A'])
        Py=W if tw['young_P'] is None else W@values(tw['young_P'])
        A=np.concatenate([Aold,Ay],axis=1)
        P=np.concatenate([Pold,Py],axis=1)
        d3,d21=dense_source_slices(D,A,P,values(tw['w']))
        f3,f21=factorized_f64_slices(D,A,P,values(tw['w']))
        got3=values(data['rows'][layer]['trace']['before_pk']['S']['(3,)'])
        got21=values(data['rows'][layer]['trace']['before_pk']['S'].get('(2, 1)'))
        np.fill_diagonal(d21,0.0)
        rows.append({'layer':layer,'oracle_D3':d3.tolist(),'oracle_D21':d21.tolist(),
                     'active_D3':got3.tolist(),'active_D21':None if got21 is None else got21.tolist(),
                     'max_dense_factor_D3_abs':float(np.max(np.abs(d3-f3))),
                     'max_dense_factor_D21_abs':float(np.max(np.abs(d21-f21*(1-np.eye(n))))),
                     'max_D3_abs':float(np.max(np.abs(d3-got3))),
                     'max_D21_abs':None if got21 is None else float(np.max(np.abs(d21-got21)))})
    return rows


def main():
    out=ROOT/'experiments'/'e14_slice_a';summary=[]
    for label,weights in [('b3',fixture.b1.WEIGHTS),('b4',fixture.fixture_weights())]:
        rows=check(out/f'{label}_f21_forced_mixed.json',weights)
        path=out/f'{label}_dense_f64_oracle.json'
        path.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        summary.append({'fixture':label,'max_D3_abs':max(x['max_D3_abs'] for x in rows),
                        'max_D21_abs':max(x['max_D21_abs'] for x in rows if x['max_D21_abs'] is not None),
                        'max_dense_factor_D3_abs':max(x['max_dense_factor_D3_abs'] for x in rows),
                        'max_dense_factor_D21_abs':max(x['max_dense_factor_D21_abs'] for x in rows),
                        'path':str(path.relative_to(ROOT))})
    (out/'oracle_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
