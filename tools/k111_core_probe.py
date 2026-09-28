"""Covariance-basis symmetric K111 core representability, diagnostic only.

No state mutation, no prediction claim, no score/cost claim. Core coefficients
are oracle fits; covariance modes use only the current model state. Matrix-free
CG fits a symmetric q^3 core against the normalized D21 error with D3 restored
on its diagonal. Diagonal weights explicitly separate the D3 compromise.

Resource review: reuse existing labels/weights, stream one net/layer, no autograd,
no dense n^3 tensor, two CPU threads, checkpoint each net. Max additional arrays
are n*q^2 and the n*n residual. q=32 quadratic basis is at most 528 columns.
Queue the substantial run; proposed RAM16/VRAM10/CPU2. Tiny CPU test is direct.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sym(G):
    return sum(G.permute(p) for p in itertools.permutations(range(3))) / 6.0


def fit_core(U, E, weight, steps=100, tol=1e-7):
    """Solve on symmetric full-core coordinates, whose inner product is Frobenius.

    A(G) = H(G) U^T, H(G)[i,c] = sum_ab U[i,a] U[i,b] G[a,b,c].
    U is orthonormal. Thus A* A is formed without an n*n product per step.
    Only its RHS requires E@U. Sym is the self-adjoint orthogonal projector
    onto symmetric cores. A diagonal weight w adds (w-1) diag(A)*diag(B).
    """
    import torch
    n, q = U.shape
    H = (U[:, :, None] * U[:, None, :]).reshape(n, q*q)
    ed = torch.diagonal(E)
    rhs = sym((H.T @ (E @ U + (weight-1)*ed[:, None]*U)).reshape(q,q,q))

    def normal(G):
        V = H @ G.reshape(q*q,q)
        dv = (V*U).sum(1)
        return sym((H.T @ (V + (weight-1)*dv[:,None]*U)).reshape(q,q,q))

    G = torch.zeros_like(rhs)
    r = rhs.clone(); p = r.clone()
    rr = (r*r).sum(); r0 = float(rr.sqrt())
    history = []
    reason = 'iterations'
    for it in range(steps):
        if float(rr.sqrt()) <= tol*max(r0,1e-30):
            reason='converged'; break
        Ap = normal(p)
        curvature = (p*Ap).sum()
        if float(curvature) <= 0:
            reason='nonpositive_curvature'; break
        alpha = rr/curvature
        G = G + alpha*p
        r = r-alpha*Ap
        rn=(r*r).sum()
        p=r+(rn/rr)*p
        rr=rn
        if it in (0,9,24,49,99) or it==steps-1:
            history.append({'iteration':it+1,'relative_normal_residual':float(rr.sqrt())/max(r0,1e-30)})
    D = (H @ G.reshape(q*q,q)) @ U.T
    off = ~torch.eye(n, dtype=torch.bool,device=U.device)
    def removed(y,pred):
        den=float((y*y).sum())
        return None if den==0 else 1-float(((y-pred)**2).sum())/den
    eoff=E[off]; poff=D[off]; pd=torch.diagonal(D)
    err=(eoff-poff).square().sum()+weight*(ed-pd).square().sum()
    den=eoff.square().sum()+weight*ed.square().sum()
    return G, D, {'diagonal_weight':weight,'reason':reason,'iterations':it+1,
                  'relative_normal_residual':float(rr.sqrt())/max(r0,1e-30),
                  'history':history,'offdiag_energy_removed':removed(eoff,poff),
                  'D3_energy_removed':removed(ed,pd),
                  'weighted_energy_removed':1-float(err)/float(den),
                  'core_symmetry_max':float((G-G.permute(2,1,0)).abs().max())}


def tiny(out):
    """Independent explicit symmetric-orbit LS vs matrix-free CG, CPU float64."""
    import torch
    torch.set_num_threads(1)
    rng=np.random.default_rng(20260928)
    n,q=11,3
    U=np.linalg.qr(rng.normal(size=(n,q)))[0]
    E=rng.normal(size=(n,n))
    cores=[]
    for ijk in itertools.combinations_with_replacement(range(q),3):
        G=np.zeros((q,q,q))
        ps=set(itertools.permutations(ijk))
        for p in ps:
            G[p]=1/np.sqrt(len(ps))
        cores.append(G)
    design=np.stack([np.einsum('ia,ib,jc,abc->ij',U,U,U,G).ravel() for G in cores],1)
    result={}
    for w in (0.,1.,float(n)):
        W=np.ones((n,n)); np.fill_diagonal(W,w)
        sd=np.sqrt(W.ravel())
        theta=np.linalg.lstsq(sd[:,None]*design,sd*E.ravel(),rcond=None)[0]
        pred=(design@theta).reshape(n,n)
        G,D,info=fit_core(torch.from_numpy(U),torch.from_numpy(E),w,steps=150,tol=1e-12)
        full=np.einsum('ia,jb,kc,abc->ijk',U,U,U,G.numpy())
        got=full[np.arange(n)[:,None],np.arange(n)[:,None],np.arange(n)[None,:]]
        info['dense_ls_max_abs']=float(np.max(np.abs(pred-D.numpy())))
        info['full_tensor_slice_max_abs']=float(np.max(np.abs(got-D.numpy())))
        result[str(w)]=info
    # Exact representable signal must recover both off-diagonal and D3 slices.
    core=sum(cores[i]*(i+1) for i in range(len(cores)))
    signal=np.einsum('ia,ib,jc,abc->ij',U,U,U,core)
    _,D,info=fit_core(torch.from_numpy(U),torch.from_numpy(signal),1,steps=150,tol=1e-12)
    result['exact_signal_max_abs']=float(np.max(np.abs(D.numpy()-signal)))
    result['passed']=all(result[str(w)]['dense_ls_max_abs']<1e-9 and result[str(w)]['full_tensor_slice_max_abs']<1e-10 for w in (0.,1.,float(n))) and result['exact_signal_max_abs']<1e-9
    out.mkdir(parents=True,exist_ok=True)
    (out/'tiny.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit('FAILED tiny algebra')


def run(a):
    from k111_diag import Diag, rep3, rep21
    import torch
    torch.set_num_threads(2)
    args=SimpleNamespace(cand=str(ROOT/'estimator_k3_v62_s1.py'),params=str(ROOT/a.params),h=.05,max_layer=99,gain_file='')
    d=Diag(args)
    layers={int(x) for x in a.layers.split(',')}
    ranks=[int(x) for x in a.ranks.split(',')]
    weights=[float(x) for x in a.diag_weights.split(',')]
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    meta={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'candidate_sha256':hashlib.sha256(Path(args.cand).read_bytes()).hexdigest(),
          'params':a.params,'params_sha256':hashlib.sha256(Path(args.params).read_bytes()).hexdigest(),
          'layers':sorted(layers),'ranks':ranks,'diag_weights':weights,'cg_steps':a.cg_steps,'nets':a.nets,
          'target':'normalized AD label-model residual orthogonal to ONE model-AD direction',
          'interpretation':'core coefficients are per-net oracle fits; finite CG capture is a feasible lower bound, unconstrained product-space projection is an upper bound'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    start,end=map(int,a.nets.split('-'))
    off=d.off64; eye=torch.eye(d.n,device=d.dev,dtype=torch.float64)
    for g in range(start,end+1):
        dest=out/f'net_{g:03d}.json'
        if dest.exists():
            continue
        t0=time.time(); torch.cuda.reset_peak_memory_stats()
        w,truth=d.data.net('full',g); labs=d.labels(g)
        result={'net':g,'layers':{}}
        rec={}
        def hook(l,name,value,*aux):
            if name in ('mu','var','C11','D3','D21'):
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f'nonfinite {name} {g}/{l}')
                rec[name]=value.detach()
            return value
        d.mod._A2=hook
        state=(d.T.fnp.zeros(d.n),d.T.fnp.eye(d.n),None,0.0,d.T.fnp.zeros(d.n))
        with torch.no_grad():
            for l in range(d.depth):
                prev=state;rec.clear();state=d.tw._step(l,w[l],state)
                if l not in layers:
                    continue
                if 'C11' not in rec:
                    rec['C11']=(w[l].T@prev[1]@w[l])*off.float()
                if 'D21' not in rec:
                    ad=d.mod._advance(w[l].T.contiguous(),prev[2])
                    rec['D21']=d.mod._dslice21(ad,off.float());del ad
                W=w[l].double()
                t,S=d.slices(d.mod._advance(d.I32,prev[2]),off.float())
                lp=d.label_parts(labs,l-1,W)
                md3=rec['D3'].double()-rep3(t,S*off,W)
                md21=(rec['D21'].double()-rep21(t,S*off,W))*off
                sig=rec['var'].double().clamp_min(1e-20).sqrt()
                scale=sig[:,None]**2*sig[None,:]
                E=(lp['AD21']-md21)/scale*off+torch.diag((lp['AD3']-md3)/sig**3)
                Mod=md21/scale+torch.diag(md3/sig**3)
                beta=(E*Mod).sum()/(Mod*Mod).sum()
                E=E-beta*Mod
                R=rec['C11'].double()/(sig[:,None]*sig[None,:])+eye
                _,Q=torch.linalg.eigh(R);Q=Q.flip(1)
                rr={}
                den=float((E*E).sum())
                for q in ranks:
                    U=Q[:,:q].contiguous()
                    ij=torch.triu_indices(q,q,device=d.dev)
                    H=U[:,ij[0]]*U[:,ij[1]]
                    # SVD avoids assuming all quadratic columns are independent.
                    Qh,sh,_=torch.linalg.svd(H,full_matrices=False)
                    keep=sh>sh[0]*1e-10
                    Qh=Qh[:,keep]
                    right=E@U;core=Qh.T@right
                    upper={'quadratic_columns':int(H.shape[1]),'numerical_rank':int(keep.sum()),
                           'right_capture':float((right*right).sum())/den,
                           'left_capture':float((Qh.T@E).square().sum())/den,
                           'bilateral_capture':float((core*core).sum())/den}
                    del Qh,sh,H,core,right
                    fits=[]
                    for weight in weights:
                        G,D,info=fit_core(U,E,weight,steps=a.cg_steps,tol=a.cg_tol)
                        fits.append(info);del G,D
                    rr[q]={'unconstrained_product_space':upper,'symmetric_core':fits}
                    print(f'net={g} layer={l} q={q} upper={upper["bilateral_capture"]:.4f} elapsed={time.time()-t0:.1f}s',flush=True)
                result['layers'][l]={'removed_model_scaling':float(beta),'orth_error_energy':den,'ranks':rr}
                del E,Mod,R,Q,prev,lp
        d.mod._A2=None
        m=state[0].clamp_min(0).double()
        result['baseline_final_mse']=float(((m-torch.from_numpy(truth[-1]).to(d.dev))**2).mean())
        result['seconds']=time.time()-t0;result['peak_vram_gib']=torch.cuda.max_memory_allocated()/2**30
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result));os.replace(tmp,dest)
        progress=os.environ.get('WORKERQ_PROGRESS')
        if progress:Path(progress).write_text(str((g-start+1)/(end-start+1)))
        print(f'COMPLETE net={g} seconds={result["seconds"]:.1f} peak={result["peak_vram_gib"]:.2f}',flush=True)
        del labs,w,state


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('cmd',choices=['tiny','run'])
    p.add_argument('--nets',default='0-1');p.add_argument('--layers',default='9,12,15')
    p.add_argument('--ranks',default='16,32');p.add_argument('--diag-weights',default='1,1024')
    p.add_argument('--cg-steps',type=int,default=100);p.add_argument('--cg-tol',type=float,default=1e-7)
    p.add_argument('--params',default='experiments/2026-09-27_S2t3/params_step736.npz')
    p.add_argument('--out',default=str(ROOT/'experiments/2026-09-28_review_arch/core_probe'))
    a=p.parse_args()
    tiny(Path(a.out)) if a.cmd=='tiny' else run(a)
