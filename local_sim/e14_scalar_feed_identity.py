"""Label-free, tiny dense oracle for the proposed scalar-metric K4 feed.

The direct side enumerates four- and three-index tensors. The reduced side uses
U factors. These are deliberately separate implementations; no closure rule is
selected here. Run as a script to retain full tensors and mutation witnesses.
"""
from __future__ import annotations

import itertools as it
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/e14_slice_b/identity_results.json"
TOL = 2e-12


def frozen_backend():
    """Load immutable F21 against the installed flopscope, not local shims."""
    here = ROOT / "local_sim"
    sys.path = [v for v in sys.path if Path(v or ".").resolve() != here]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    path = ROOT / "estimator_k3_strassen_L4_f21.py"
    spec = importlib.util.spec_from_file_location("e14_b_frozen_f21", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod._STRASSEN_DEPTH = 0  # tiny shape; real active _mm dispatch remains.
    return mod, hashlib.sha256(path.read_bytes()).hexdigest()


def query_with_extra(mod, tw, mask, enabled, y=None, r=None):
    if not enabled:
        return mod._dslice21(tw, mask)
    a,p,d,w = tw.A_L,tw.M,tw.A_D,tw.w2b
    assert y is not None and r is not None
    x = mod._mm(a*r,p.T)
    t = mod.fnp.diag(x)
    l1 = 2*(d*p)+3*(a*(a*w))+3*(y[:,None]*a)*r
    l3 = 6*(a*p)*w+3*(y[:,None]*p)*r
    return ((mod._mm(l1,p.T)+mod._mm(p*p,d.T)+mod._mm(l3,a.T))/3
            +t[:,None]*y[None,:])*mask


def dense_f21_tower(tw):
    """Enumerate frozen tower factors, with newest M=I, into a symmetric K3."""
    n=tw.A_D.shape[0]
    out=np.zeros((n,n,n),dtype=np.float64)
    for age in range(tw.k):
        sl=slice(age*n,(age+1)*n)
        d=np.asarray(tw.A_D[:,sl],dtype=np.float64)
        a=np.asarray(tw.A_L[:,sl],dtype=np.float64)
        p=(np.eye(n) if tw.M is None or age*n>=tw.M.shape[1]
           else np.asarray(tw.M[:,sl],dtype=np.float64))
        c=a*np.asarray(tw.w2b[sl],dtype=np.float64)[None,:]
        for i,j,k in it.product(range(n),repeat=3):
            for r in range(n):
                out[i,j,k] += (d[i,r]*p[j,r]*p[k,r]
                               +d[j,r]*p[i,r]*p[k,r]
                               +d[k,r]*p[i,r]*p[j,r])/3
                out[i,j,k] += (a[i,r]*p[j,r]*c[k,r]
                               +a[i,r]*p[k,r]*c[j,r]
                               +a[j,r]*p[i,r]*c[k,r]
                               +a[j,r]*p[k,r]*c[i,r]
                               +a[k,r]*p[i,r]*c[j,r]
                               +a[k,r]*p[j,r]*c[i,r])/2
    return out


def frozen_birth_check(mod,n,results):
    """Trace unchanged F21 _layer and independently reconstruct its full birth."""
    source,first=inspect.getsourcelines(mod._layer)
    markers={"after_K":"    mu_out = K[(1,)]",
             "before_wick":"    if adv is not None:",
             "after_birth":"    lap = table[\"lap2_d4\"]"}
    linenos={}
    for label,marker in markers.items():
        hits=[first+i for i,line in enumerate(source) if line.startswith(marker)]
        if label=="before_wick": hits=[h for h in hits if h>linenos["after_K"]]
        assert len(hits)==1,(label,hits)
        linenos[label]=hits[0]
    reverse={v:k for k,v in linenos.items()}
    dtype=mod.fnp.float32
    table=mod._tables()
    fnp=mod.fnp
    const={"off_mask":fnp.ones((n,n),dtype=dtype)-fnp.eye(n,dtype=dtype),
           "ones_n":fnp.ones(n,dtype=dtype),
           "wick":mod._wick_program(table,dtype)}
    state=(fnp.zeros(n,dtype=dtype),fnp.eye(n,dtype=dtype),None,0.0)
    prior=sys.gettrace()
    for layer in range(2):
        snapshots={}
        def tracer(frame,event,arg):
            if frame.f_code is mod._layer.__code__ and event=="line" and frame.f_lineno in reverse:
                snapshots[reverse[frame.f_lineno]]=dict(frame.f_locals)
            return tracer
        weight=np.array([[0.31+0.09*i-0.07*j+0.014*layer
                          for j in range(n)] for i in range(n)],dtype=np.float32)
        try:
            sys.settrace(tracer)
            state=mod._layer(fnp.asarray(weight),state,table,dtype,const,2.0,False,None)
        finally:
            sys.settrace(prior)
        post=snapshots["after_birth"]
        pre=snapshots["before_wick"]
        k=snapshots["after_K"]["K"]
        w=pre["w"]
        c11=np.asarray(pre["C11"])
        w1=np.asarray(w[(1,1)])
        w2=np.asarray(w[(2,1)])
        fac1=w1[:,None]*c11
        d3=np.asarray(pre["D3"])
        d21=np.asarray(pre["D21"])
        p111_3=(w1**3)*d3
        p111_21=((w1**2)[:,None]*w1[None,:])*d21+(fac1**2)*w2[None,:]*(1-np.eye(n))
        compare(f"frozen_n{n}_layer{layer}_p111_3",p111_3,np.asarray(pre["p111_3"]),results["checks"],1e-6)
        compare(f"frozen_n{n}_layer{layer}_p111_21",p111_21,np.asarray(pre["p111_21"]),results["checks"],1e-6)
        # Independent two-leg birth tensor, not read from F21's appended A_L.
        path=np.zeros((n,n,n))
        for i,j,h in it.product(range(n),repeat=3):
            for r in range(n):
                vals=(fac1[i,r]*fac1[h,r]*w2[r]*(j==r),
                      fac1[i,r]*fac1[j,r]*w2[r]*(h==r),
                      fac1[j,r]*fac1[h,r]*w2[r]*(i==r),
                      fac1[j,r]*fac1[i,r]*w2[r]*(h==r),
                      fac1[h,r]*fac1[j,r]*w2[r]*(i==r),
                      fac1[h,r]*fac1[i,r]*w2[r]*(j==r))
                path[i,j,h]+=sum(vals)/2
        old=np.zeros_like(path) if post["adv"] is None else dense_f21_tower(post["adv"])
        expected=birth(old,path,np.asarray(k[(3,)]),np.asarray(k[(2,1)]))
        actual=dense_f21_tower(state[2])
        compare(f"frozen_n{n}_layer{layer}_S3",np.asarray(k[(3,)])-p111_3,
                np.asarray(pre["S3"]),results["checks"],1e-6)
        compare(f"frozen_n{n}_layer{layer}_S21",np.asarray(k[(2,1)])-p111_21,
                np.asarray(pre["S21"]),results["checks"],1e-6)
        compare(f"frozen_n{n}_layer{layer}_whole_birth",expected,actual,results["checks"],1e-6)
        # Inject a supplied nonzero scalar-metric Q at the captured production
        # consumer boundary. The frozen estimator itself remains unchanged.
        # First independently replay its unmodified SIMPLE baseline.
        lap_table=table["lap2_d4"]
        lap=tuple(float(lap_table[z]["terms"][0]["coef"])
                  *float(lap_table[z]["int_partition_coef"])
                  for z in ("(4,)","(2, 2)"))
        proj=mod._proj_coef_d4_r2(n)
        observed_s={z:np.asarray(v) for z,v in pre["S"].items()}
        observed_w={z:np.asarray(v) for z,v in w.items()}
        zeroq=np.zeros((n,)*4)
        _,_,_,baseline_k,_=manifest_reference(
            zeroq,n,layer,table,lap,proj,0,2.0,old,
            supplied_s=observed_s,supplied_w_override=observed_w)
        for ip in ((1,),(2,),(1,1),(3,),(2,1),(4,),(2,2)):
            compare(f"frozen_n{n}_layer{layer}_manifest_K{ip}",
                    baseline_k[ip],np.asarray(k[ip]),results["checks"],1e-5)
        core=np.array([[((-1)**(i+j))*(0.11+0.04*i+0.03*j)
                        for j in range(n)] for i in range(n)])
        core=(core+core.T)/2
        extra_q=lift(core,2.0)
        _,_,_,corrected_k,corrected_scalar=manifest_reference(
            extra_q,n,layer,table,lap,proj,0,2.0,old,
            supplied_s=observed_s,supplied_w_override=observed_w)
        a,b=observed_w[(2,1)],observed_w[(1,1)]
        direct=direct_j(extra_q,a,b)
        reduced=u_tensor(a/6,(b[:,None]*core)*b[None,:])
        direct_born=birth(old+path,direct,corrected_k[(3,)],corrected_k[(2,1)])
        reduced_born=birth(old+path,reduced,corrected_k[(3,)],corrected_k[(2,1)])
        compare(f"frozen_n{n}_layer{layer}_supplied_Q_whole_birth",
                direct_born,reduced_born,results["checks"],1e-6)
        results["complete_tensors"][f"frozen_n{n}_layer{layer}"]={
            "old_after_wick":old.tolist(),"two_leg_path":path.tolist(),
            "expected":expected.tolist(),"frozen_tower":actual.tolist(),
            "K3":np.asarray(k[(3,)]).tolist(),"K21":np.asarray(k[(2,1)]).tolist(),
            "p111_3":np.asarray(pre["p111_3"]).tolist(),
            "p111_21":np.asarray(pre["p111_21"]).tolist(),
            "S3":np.asarray(pre["S3"]).tolist(),"S21":np.asarray(pre["S21"]).tolist(),
            "supplied_Q":extra_q.tolist(),"corrected_K3":np.asarray(corrected_k[(3,)]).tolist(),
            "corrected_K21":np.asarray(corrected_k[(2,1)]).tolist(),
            "corrected_scalar":corrected_scalar,
            "direct_Q_birth":direct_born.tolist(),"reduced_Q_birth":reduced_born.tolist()}


def lift(g, m):
    n = len(g)
    q = np.empty((n,) * 4, dtype=np.float64)
    for i, j, k, l in it.product(range(n), repeat=4):
        q[i, j, k, l] = m * (
            g[i, j] * (k == l) + g[i, k] * (j == l)
            + g[i, l] * (j == k) + g[j, k] * (i == l)
            + g[j, l] * (i == k) + g[k, l] * (i == j)
        ) / 6.0
    return q


def direct_j(q, a, b):
    n = len(a)
    jten = np.empty((n,) * 3)
    for i, j, k in it.product(range(n), repeat=3):
        jten[i, j, k] = (
            a[i] * b[j] * b[k] * q[i, i, j, k]
            + b[i] * a[j] * b[k] * q[i, j, j, k]
            + b[i] * b[j] * a[k] * q[i, j, k, k]
        ) / 2.0
    return jten


def u_tensor(y, s):
    n = len(y)
    out = np.empty((n,) * 3)
    for i, j, k in it.product(range(n), repeat=3):
        out[i, j, k] = y[i] * s[j, k] + y[j] * s[i, k] + y[k] * s[i, j]
    return out


def repeated(t):
    """Independently reconstruct F21's repeated slices at each index."""
    n = len(t)
    out = np.zeros_like(t)
    for i, j, k in it.product(range(n), repeat=3):
        if i == j == k:
            out[i, j, k] = t[i, i, i]
        elif i == j:
            out[i, j, k] = t[i, i, k]
        elif i == k:
            out[i, j, k] = t[i, i, j]
        elif j == k:
            out[i, j, k] = t[j, j, i]
    return out


def transport(t, p):
    return np.einsum("ia,jb,kc,abc->ijk", p, p, p, t, optimize=False)


def birth(base, feed, k3, k21):
    t = base + feed
    n = len(t)
    correction = np.zeros_like(t)
    for i, j, k in it.product(range(n), repeat=3):
        if i == j == k:
            correction[i, j, k] = k3[i] - t[i, i, i]
        elif i == j:
            correction[i, j, k] = k21[i, k] - t[i, i, k]
        elif i == k:
            correction[i, j, k] = k21[i, j] - t[i, i, j]
        elif j == k:
            correction[i, j, k] = k21[j, i] - t[j, j, i]
    return t + correction


def supplied_w(n, step):
    return {(order,p): np.array([0.21+0.037*order+0.029*p+0.013*i+0.007*step
                                 for i in range(n)])
            for order in range(5) for p in range(1,5)}


def manifest_reference(q,n,step,table,lap,proj,scalar_in,metric,incoming,
                       supplied_s=None,supplied_w_override=None):
    """Evaluate frozen SIMPLE rows and pK_to_K partitions from supplied S.

    This does not call the handwritten `consumers` formulas or its baseline.
    The table, including centering signs, is the independently frozen F21 rule.
    """
    w = supplied_w(n,step) if supplied_w_override is None else supplied_w_override
    idx = np.arange(n)
    off = 1-np.eye(n)
    if supplied_s is None:
        c11 = off*(0.019+0.004*idx[:,None]-0.003*idx[None,:])
        c11 = (c11+c11.T)/2
        d3 = np.array([incoming[i,i,i] for i in range(n)])
        d21 = np.array([[incoming[i,i,j] if i!=j else 0
                         for j in range(n)] for i in range(n)])
        s = {(3,):0.14+0.017*idx+d3,
             (4,):0.073+0.011*idx+scalar_in*metric**2,
             (1,1):c11,
             (2,1):off*(0.016+0.002*idx[:,None]+0.005*idx[None,:])+d21,
             (2,2):off*(0.012+0.003*idx[:,None]+0.001*idx[None,:]
                        +scalar_in*metric**2/3)}
        s[(1,2)] = s[(2,1)].T
    else:
        s={key:np.asarray(value,dtype=np.float64) for key,value in supplied_s.items()}
    pk = {}
    for row in table["terms"]:
        ip = tuple(row["int_part"])
        degree = len(ip)
        val = float(row["scalar"])
        for block in row["blocks"]:
            val = val*s[tuple(block["vec"])]
        for axis,kp in enumerate(row["wick"]):
            wk = w[tuple(kp)]
            val = val*(wk if degree==1 else (wk[:,None] if axis==0 else wk[None,:]))
        pk[ip] = pk.get(ip,0)+val
    for ip in list(pk):
        if len(ip)==2 and ip[0]==ip[1]:
            pk[ip] = (pk[ip]+pk[ip].T)/2
    baseline = {k:np.array(v,copy=True) for k,v in pk.items()}
    # Enumerate Q slices independently of the analytic mG shortcuts.
    for p in range(1,5):
        for i in range(n):
            pk[(p,)][i] += q[i,i,i,i]*w[4,p][i]/24
    for p,r in ((1,1),(2,1),(2,2)):
        for i,j in it.product(range(n),repeat=2):
            if i==j: continue
            pk[p,r][i,j] += (
                q[i,i,i,j]*w[3,p][i]*w[1,r][j]/6
                +q[i,i,j,j]*w[2,p][i]*w[2,r][j]/4
                +q[i,j,j,j]*w[1,p][i]*w[3,r][j]/6)
    for ip in ((1,1),(2,2)):
        pk[ip] = (pk[ip]+pk[ip].T)/2
    def block(b,degree):
        positions = tuple(i for i,v in enumerate(b) if v)
        key = tuple(b[i] for i in positions)
        value = pk[tuple(sorted(key,reverse=True))]
        if len(key)==2 and key[0]<key[1]: value=value.T
        if len(positions)!=degree:
            value=value[:,None] if positions==(0,) else value[None,:]
        return value
    centered={}
    for raw_key,rows in table["pK_to_K"].items():
        ip=tuple(int(z) for z in raw_key.strip("()").split(",") if z.strip())
        val=0
        for row in rows:
            term=float(row["coef"])
            for b in row["blocks"]: term=term*block(b,len(ip))
            val=val+term
        centered[ip]=val
    scalar=proj*(lap[0]*float(centered[(4,)].sum())
                 +lap[1]*float(centered[(2,2)].sum()))
    return s,baseline,pk,centered,scalar


def final_mean_reference(q,n,step,table,s,scalar_in,metric):
    """Final F21 mean-only rows, with Q4 added before K1 is read."""
    w=supplied_w(n,step)
    out=np.zeros(n)
    for row in table["terms"]:
        if tuple(row["int_part"])!=(1,): continue
        value=float(row["scalar"])
        for block in row["blocks"]: value=value*s[tuple(block["vec"])]
        value=value*w[tuple(row["wick"][0])]
        out=out+value
    out=out+np.array([q[i,i,i,i] for i in range(n)])*w[4,1]/24
    return out


def consumers(q, n, step, lap, proj, scalar_in, metric, baseline, *, mutation=None):
    """Independent finite-hyperedge SIMPLE moment and centering calculation.

    All supplied base moments and response coefficients are explicit; this does
    not regenerate G. Connected pair P is not converted to raw moments before
    centering. A mutation changes this live path, not an unrelated number.
    """
    w = supplied_w(n,step)
    base_a = {p: baseline[(p,)] for p in range(1,5)}
    base_p = {(p,r):baseline[(p,r)] for p,r in ((1,1),(2,1),(2,2))}
    q4 = np.array([q[i,i,i,i] for i in range(n)])
    q31 = np.array([[q[i,i,i,j] for j in range(n)] for i in range(n)])
    q22 = np.array([[q[i,i,j,j] for j in range(n)] for i in range(n)])
    # Baseline moments already contain F21's scalar D4 and D22 via S. Add only
    # the extra Q here; duplicating the scalar is a live-path mutation below.
    aa = {p: base_a[p]+q4*w[4,p]/24 for p in range(1,5)}
    if mutation == "double_scalar":
        aa = {p: aa[p]+scalar_in*metric**2*w[4,p]/24 for p in range(1,5)}
    pp = {}
    for p,r in ((1,1),(2,1),(2,2)):
        left = q31
        right = q31.T
        right_w1 = w[1,p][:,None]*w[3,r][None,:]
        if mutation == "wrong_q31_orientation":
            right_w1 = w[3,p][:,None]*w[1,r][None,:]
        delta = (left*w[3,p][:,None]*w[1,r][None,:]/6
                 +q22*w[2,p][:,None]*w[2,r][None,:]/4
                 +right*right_w1/6)
        pp[p,r] = base_p[p,r]+delta*(1-np.eye(n))
    # Cumulant centering at exactly the F21 consumers relevant to K3 birth.
    k3 = aa[3]-3*aa[2]*aa[1]+2*aa[1]**3
    k21 = pp[2,1]-2*aa[1][:,None]*pp[1,1]
    np.fill_diagonal(k21,0)
    k4 = aa[4]-4*aa[3]*aa[1]-3*aa[2]**2+12*aa[2]*aa[1]**2-6*aa[1]**4
    mu, a2 = aa[1],aa[2]
    raw11 = pp[1,1]+np.outer(mu,mu)
    raw21 = pp[2,1]+np.outer(a2,mu)
    raw12 = raw21.T
    raw22 = pp[2,2]+np.outer(a2,a2)
    central22 = (raw22-2*mu[None,:]*raw21-2*mu[:,None]*raw12
                 +4*np.outer(mu,mu)*raw11
                 +mu[None,:]**2*a2[:,None]+mu[:,None]**2*a2[None,:]
                 -3*np.outer(mu**2,mu**2))
    variance = a2-mu**2
    covariance = pp[1,1]
    k22 = central22-np.outer(variance,variance)-2*covariance**2
    np.fill_diagonal(k22,0)
    scalar = proj*(lap[0]*float(k4.sum())+lap[1]*float(k22.sum()))
    return {"a":aa,"P":pp,"K3":k3,"K21":k21,"K4":k4,
            "K22":k22,"scalar":scalar,"q4":q4,"q31":q31,"q22":q22}


def compare(name, left, right, results, tolerance=TOL):
    err = np.asarray(left) - np.asarray(right)
    worst = np.unravel_index(int(np.argmax(np.abs(err))), err.shape)
    entry = {"max_abs": float(np.max(np.abs(err))), "worst_index": [int(v) for v in worst],
             "left_at_worst": float(np.asarray(left)[worst]),
             "right_at_worst": float(np.asarray(right)[worst])}
    results[name] = entry
    if entry["max_abs"] > tolerance:
        raise AssertionError(f"{name}: {entry}")


def run():
    frozen, frozen_sha = frozen_backend()
    lap_table = frozen._tables()["lap2_d4"]
    lap = tuple(float(lap_table[k]["terms"][0]["coef"])
                *float(lap_table[k]["int_partition_coef"])
                for k in ("(4,)","(2, 2)"))
    results = {"scope": "supplied cores only; no labels, regeneration, or seed block",
               "tolerance_absolute": TOL, "frozen_f21_sha256": frozen_sha,
               "diagnostic_correction": "Initial active-_mm comparison included the diagonal of the fused expression while frozen _dslice21 masks it; first n3_age1 apparent max difference 0.11090882122516632. Both branches now apply the same offdiagonal mask.",
               "birth_diagnostic_correction": "Initial frozen n3 layer0 full-birth difference 3.6880916653991846e-9 was f32 roundoff under an f64 oracle threshold. Initial layer1 all-distinct difference 0.011918990019543454 came from treating the advanced oldest M as symbolic I in the test decoder; advanced M is explicit. Corrected decoder checks M shape for each age.",
               "checks": {}, "mutations": {}, "complete_tensors": {}}
    for small_n in (3,4):
        frozen_birth_check(frozen,small_n,results)
    for n in (3, 4):
        # Explicit signed symmetric core, unequal response coefficients.
        g = np.array([[((-1) ** (i + j)) * (0.13 + 0.07 * i + 0.11 * j)
                       for j in range(n)] for i in range(n)], dtype=np.float64)
        g = (g + g.T) / 2
        a = np.array([0.41 + 0.19 * i for i in range(n)])
        b = np.array([0.72 - 0.09 * i for i in range(n)])
        m = 1.7
        for kind, core in (("signed", g), ("diagonal", np.diag(np.diag(g)))):
            key = f"n{n}_{kind}"
            q = lift(core, m)
            jd = direct_j(q, a, b)
            yr = m * a / 12
            sr = (b[:, None] * core) * b[None, :]
            jr = u_tensor(yr, sr)
            compare(key + "_cancel", jd - repeated(jd),
                    jr - repeated(jr), results["checks"])
            base = np.array([[[0.013 * (i+j+k+1) for k in range(n)]
                              for j in range(n)] for i in range(n)])
            base = sum(np.transpose(base, p) for p in it.permutations(range(3))) / 6
            k3 = np.array([0.22 + i * 0.04 for i in range(n)])
            k21 = np.array([[0 if i == j else 0.018 * (i-j)
                             for j in range(n)] for i in range(n)])
            td = birth(base, jd, k3, k21)
            tr = birth(base, jr, k3, k21)
            tm = tr.copy()
            compare(key + "_whole_birth", td, tr, results["checks"])
            results["complete_tensors"][key] = {
                "G": core.tolist(), "Q": q.tolist(), "J_dense": jd.tolist(),
                "J_reduced": jr.tolist(), "D_J_dense": repeated(jd).tolist(),
                "D_J_reduced": repeated(jr).tolist(),
                "birth_dense": td.tolist(), "birth_reduced": tr.tolist()}
            # Two distinct supplied closure states; both paths receive exactly
            # the same next moment slices and the same linear/Wick maps.
            scalar_in = 0.052
            for step in range(2):
                p = np.array([[((i + 2*j + step) % 4 - 1.5) * 0.17
                               + (0.83 if i == j else 0)
                               for j in range(n)] for i in range(n)])
                w = np.array([0.67 + 0.08*i + 0.03*step for i in range(n)])
                td = transport(td, p) * w[:, None, None] * w[None, :, None] * w[None, None, :]
                tr = transport(tr, p) * w[:, None, None] * w[None, :, None] * w[None, None, :]
                tm = transport(tm, p) * w[:, None, None] * w[None, :, None] * w[None, None, :]
                compare(key + f"_transport_wick_{step}", td, tr, results["checks"])
                # Supplied, nonzero next closure and moment consumers, equal on
                # both sides. This tests repeated correction persistence.
                next_core = core * (0.81 + 0.06*step)
                aa, bb = a * (0.88 + 0.02*step), b * (0.91 + 0.01*step)
                jnext = direct_j(lift(next_core, m), aa, bb)
                unext = u_tensor(m*aa/12, (bb[:, None]*next_core)*bb[None, :])
                qnext = lift(next_core,m)
                projection = frozen._proj_coef_d4_r2(n)
                supplied_s,baseline,mpk,mk,mscalar = manifest_reference(
                    qnext,n,step,frozen._tables(),lap,projection,scalar_in,m,tm)
                cm = consumers(qnext,n,step,lap,projection,scalar_in,m,baseline)
                for p in range(1,5):
                    compare(key+f"_manifest_a{p}_{step}",cm["a"][p],mpk[(p,)],results["checks"])
                for ip in ((1,1),(2,1),(2,2)):
                    compare(key+f"_manifest_P{ip}_{step}",cm["P"][ip],mpk[ip],results["checks"])
                compare(key+f"_manifest_K1_{step}",mk[(1,)],mpk[(1,)],results["checks"])
                compare(key+f"_manifest_K2_{step}",mk[(2,)],mpk[(2,)]-mpk[(1,)]**2,results["checks"])
                compare(key+f"_manifest_cov_{step}",mk[(1,1)],mpk[(1,1)],results["checks"])
                for name,ip in (("K3",(3,)),("K21",(2,1)),("K4",(4,)),("K22",(2,2))):
                    compare(key+f"_manifest_{name}_{step}",cm[name],mk[ip],results["checks"])
                compare(key+f"_final_mean_pruning_{step}",
                        final_mean_reference(qnext,n,step,frozen._tables(),supplied_s,scalar_in,m),
                        mk[(1,)],results["checks"])
                compare(key+f"_manifest_scalar_{step}",np.array(cm["scalar"]),np.array(mscalar),results["checks"])
                if step==0 and kind=="signed" and n==3:
                    _,_,without_scalar_pk,without_scalar_k,_=manifest_reference(
                        qnext,n,step,frozen._tables(),lap,projection,0.0,m,tm)
                    results["mutations"]["omitted_scalar_D22"] = float(np.max(np.abs(
                        mpk[(2,2)]-without_scalar_pk[(2,2)])))
                kk3,kk21 = cm["K3"],cm["K21"]
                compare(key+f"_q4_consumer_{step}",cm["q4"],m*np.diag(next_core),results["checks"])
                compare(key+f"_q31_consumer_{step}",cm["q31"]*(1-np.eye(n)),
                        m*next_core*(1-np.eye(n))/2,results["checks"])
                compare(key+f"_q22_consumer_{step}",cm["q22"]*(1-np.eye(n)),
                        m*(np.diag(next_core)[:,None]+np.diag(next_core)[None,:])
                        *(1-np.eye(n))/6,results["checks"])
                tr_before = tr
                td = birth(td, jnext, kk3, kk21)
                tr = birth(tr, unext, kk3, kk21)
                tm = birth(tm,unext,mk[(3,)],mk[(2,1)])
                compare(key + f"_whole_next_{step}", td, tr, results["checks"])
                compare(key + f"_manifest_whole_next_{step}",td,tm,results["checks"])
                results["complete_tensors"][key][f"next_{step}_dense"] = td.tolist()
                results["complete_tensors"][key][f"next_{step}_reduced"] = tr.tolist()
                results["complete_tensors"][key][f"next_{step}_manifest"] = tm.tolist()
                results["complete_tensors"][key][f"consumer_{step}"] = {
                    "supplied_S":{str(p):v.tolist() for p,v in supplied_s.items()},
                    "a":{str(p):v.tolist() for p,v in cm["a"].items()},
                    "P":{str(p):v.tolist() for p,v in cm["P"].items()},
                    "K3":kk3.tolist(),"K21":kk21.tolist(),
                    "K4":cm["K4"].tolist(),"K22":cm["K22"].tolist(),
                    "scalar":cm["scalar"],"q31":cm["q31"].tolist(),
                    "q22":cm["q22"].tolist(),"q4":cm["q4"].tolist()}
                if step==0 and kind=="signed" and n==3:
                    for mutation in ("wrong_q31_orientation","double_scalar"):
                        bad = consumers(qnext,n,step,lap,projection,scalar_in,m,baseline,mutation=mutation)
                        bad_tensor = birth(tr_before,unext,bad["K3"],bad["K21"])
                        results["mutations"][mutation+"_live_consumer"] = float(np.max(np.abs(
                            bad_tensor-tr)))
                scalar_in = cm["scalar"]
            if kind == "signed" and n == 3:
                wrong = u_tensor(m*a/6, sr)
                results["mutations"]["wrong_1_over_6"] = float(abs((wrong-repeated(wrong)-jd+repeated(jd))[0,1,2]))
                results["mutations"]["missing_birth_subtraction"] = float(np.max(np.abs(jd-jr)))
                results["mutations"]["diagonal_feed_without_birth_subtraction"] = float(np.max(np.abs(direct_j(lift(np.diag(np.diag(g)),m),a,b))))
                # The scalar fourth-order account is an independent consumer.
                # Subtracting it a second time changes q4 rather than the
                # birth-cancelled all-distinct feed.
                q4 = m*np.diag(g)
                scalar_account = np.array([0.037,-0.024,0.053])
                results["mutations"]["double_scalar_subtraction"] = float(np.max(np.abs((q4-scalar_account)-q4)))
        # Exact normalization witness G_23=t, all others zero (zero-based 1,2).
        if n == 3:
            witness = np.zeros((3, 3)); witness[1,2] = witness[2,1] = 0.39
            q = lift(witness, m)
            results["checks"]["n3_witness"] = {"q_1123": float(q[0,0,1,2]),
                "j_123": float(direct_j(q,a,b)[0,1,2]),
                "expected": float(m*a[0]*b[1]*b[2]*0.39/12)}
            assert abs(results["checks"]["n3_witness"]["j_123"] - results["checks"]["n3_witness"]["expected"]) < TOL

        # Shared F21 A/P/y/r factors, all ages including newborn. Materialize
        # an independent dense U tensor from transported birth S0.
        c = np.array([[0 if i == j else 0.09*(1+i-j) for j in range(n)] for i in range(n)])
        a0 = b[:, None] * c
        y0 = m * 0.31 * a / 12
        r = b.copy()
        s0 = (a0*r[None, :] + (a0*r[None, :]).T) / 2
        for age in range(4):
            p = np.eye(n)
            yy = y0.copy()
            for step in range(age):
                lin = np.array([[0.05*(i-j+step) + (0.82 if i==j else 0)
                                 for j in range(n)] for i in range(n)])
                wick = np.array([0.64+0.06*i+0.02*step for i in range(n)])
                p = wick[:,None] * (lin @ p)
                yy = wick * (lin @ yy)
            aa = p @ a0
            x = (aa*r[None,:]) @ p.T
            s = (x+x.T)/2
            tdense = transport(u_tensor(y0,s0), p)
            tshare = u_tensor(yy,s)
            prefix = f"n{n}_age{age}"
            compare(prefix+"_full_U", tdense, tshare, results["checks"])
            t = np.diag(x)
            d3 = 3*yy*t
            d21 = yy[:,None]*(x+x.T) + np.outer(t,yy)
            compare(prefix+"_D3", d3, np.array([tdense[i,i,i] for i in range(n)]), results["checks"])
            compare(prefix+"_D21", d21, np.array([[tdense[i,i,j] for j in range(n)] for i in range(n)]), results["checks"])
            compare(prefix+"_D12", d21.T, np.array([[tdense[i,j,j] for j in range(n)] for i in range(n)]), results["checks"])
            if age == 0:
                compare(prefix+"_birth_D3_zero", d3, np.zeros(n), results["checks"])
                newborn_wick = np.array([0.66+0.04*i for i in range(n)])
                results["mutations"][prefix+"_newborn_extra_wick"] = float(np.max(np.abs(
                    tshare-u_tensor(newborn_wick*yy,
                                    newborn_wick[:,None]*s*newborn_wick[None,:]))))
            if age == 2:
                results["mutations"][prefix+"_omit_transpose"] = float(np.max(np.abs(d21-(yy[:,None]*x+np.outer(t,yy)))))
                results["mutations"][prefix+"_omit_outer"] = float(np.max(np.abs(np.outer(t,yy))))
                results["mutations"][prefix+"_double_wick"] = float(np.max(np.abs(tshare-u_tensor(yy*0.71,s*0.71**2))))
                results["mutations"][prefix+"_missing_y_transport"] = float(np.max(np.abs(tshare-u_tensor(y0,s))))
                results["mutations"][prefix+"_missing_old_y_wick"] = float(np.max(np.abs(tshare-u_tensor(yy/0.71,s))))
            # Active F21 three-product query with and without the U feed.
            # This deliberately uses f32 operands on both sides. It diagnoses
            # reassociation, rather than claiming bit parity.
            if age > 0:
                d = (0.04 + 0.03*np.arange(n*n).reshape(n,n)).astype(np.float32)
                af, pf, yf, rf = (v.astype(np.float32) for v in (aa,p,yy,r))
                wf, df = (0.2+0.05*np.arange(n)).astype(np.float32), d
                cf = af*wf[None,:]
                base_f = ((2*(df*pf)+3*(af*cf))@pf.T
                          +(pf*pf)@df.T+6*(af*pf)@cf.T)/np.float32(3)
                xf = (af*rf[None,:])@pf.T
                tf = np.diag(xf)
                unfused = base_f + yf[:,None]*(xf+xf.T)+np.outer(tf,yf)
                l1 = 2*(df*pf)+3*(af*cf)+3*(yf[:,None]*af)*rf[None,:]
                l3 = 6*(af*pf)*wf[None,:]+3*(yf[:,None]*pf)*rf[None,:]
                fused = (l1@pf.T+(pf*pf)@df.T+l3@af.T)/np.float32(3)+np.outer(tf,yf)
                gap = float(np.max(np.abs(fused-unfused)))
                scale = max(1.0, float(np.max(np.abs(unfused))))
                results["checks"][prefix+"_active_f32_fusion"] = {
                    "max_abs": gap, "max_scaled": gap/scale,
                    "fused": fused.tolist(), "unfused": unfused.tolist()}
                if gap/scale > 1e-5:
                    raise AssertionError(f"{prefix} f32 fusion gap {gap/scale}")
                fnp = frozen.fnp
                wrap = lambda v: fnp.asarray(v, dtype=fnp.float32)
                tw = frozen._Tower(wrap(df), wrap(af), wrap(pf), wrap(wf), 1)
                mask = wrap(np.ones((n,n),dtype=np.float32)-np.eye(n,dtype=np.float32))
                old = np.asarray(frozen._dslice21(tw,mask))
                disabled = np.asarray(query_with_extra(frozen,tw,mask,False))
                compare(prefix+"_disabled_frozen", disabled, old, results["checks"])
                results["checks"][prefix+"_disabled_frozen"]["frozen_output"] = old.tolist()
                # Actual F21 _mm operations, with the specified f32 arrays.
                f_a, f_p, f_d, f_w, f_y, f_r = map(wrap, (af,pf,df,wf,yf,rf))
                f_c = f_a * f_w
                f_x = frozen._mm(f_a*f_r, f_p.T)
                f_t = fnp.diag(f_x)
                feed = f_y[:,None]*(f_x+f_x.T)+f_t[:,None]*f_y[None,:]
                unfrozen = (old + np.asarray(feed))*np.asarray(mask)
                factive = np.asarray(query_with_extra(frozen,tw,mask,True,f_y,f_r))
                ferr = float(np.max(np.abs(factive-unfrozen)))
                fscale = max(1.0,float(np.max(np.abs(unfrozen))))
                results["checks"][prefix+"_actual_f21_mm"] = {
                    "max_abs": ferr, "max_scaled": ferr/fscale,
                    "fused": factive.tolist(), "unfused": unfrozen.tolist()}
                if ferr/fscale > 1e-5:
                    raise AssertionError(f"{prefix} actual F21 _mm fusion {ferr/fscale}")
    for name, value in results["mutations"].items():
        if value <= 1e-5:
            raise AssertionError(f"mutation {name} not detected: {value}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {len(results['checks'])} checks, {len(results['mutations'])} mutation witnesses: {OUT}")


if __name__ == "__main__":
    run()
