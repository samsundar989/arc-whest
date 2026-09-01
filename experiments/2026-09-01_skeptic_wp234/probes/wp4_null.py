"""NULL CONTROL for WP4's harness.

A PROVABLY LINEAR, PROVABLY ADDITIVE state-space system with a kappa3-like
channel, driven through EXACTLY the interventions p2_adjoint_k3.run_arm applies,
and scored with EXACTLY the statistics response_curve()/per_layer_response()
compute.  If a linear system fails Q3/Q4/Q5, the statistics cannot detect
nonlinearity and the WP4 KILL is unfounded.
"""
import numpy as np
LAD = (0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)

def run(A,C,D,F,s0,k0,L,arm_at=None,eps=1.0):
    s,k = s0.copy(), k0.copy()
    for l in range(L):
        s,k = A@s + F@k, C@k + D@s          # ONE LINEAR STEP. no nonlinearity.
        f = eps if (arm_at is None or l==arm_at) else 0.0
        k = f*k
    return s

def stats(A,C,D,F,s0,k0,L):
    m = {e: run(A,C,D,F,s0,k0,L,None,e) for e in LAD}
    m0 = m[0.0]; slope=(m[LAD[1]]-m0)/LAD[1]
    worst = max(np.linalg.norm(m[e]-(m0+slope*e))/(np.linalg.norm(m[e]-m0) or 1.0)
                for e in LAD if e>0)
    total = m[1.0]-m0
    deltas=[run(A,C,D,F,s0,k0,L,l,1.0)-m0 for l in range(L)]
    summed=np.sum(deltas,axis=0)
    exp = 1.0 - float((total-summed)@(total-summed))/float(total@total)
    cos = float(summed@total/(np.linalg.norm(summed)*np.linalg.norm(total)))
    sq=[float(d@d) for d in deltas]
    return worst, exp, cos, max(sq)/sum(sq)

rng=np.random.default_rng(0)
n=32
print(f"{'L':>3} {'Q3 worst_lin_dev':>18} {'Q4 explained':>14} {'Q5 cosine':>10} {'Q6 maxshare':>12}  verdict-by-WP4-rules")
for L in (3,6,10,16):
    A=rng.normal(0,1/np.sqrt(n),(n,n)); C=rng.normal(0,1/np.sqrt(n),(n,n))
    D=rng.normal(0,0.3/np.sqrt(n),(n,n)); F=rng.normal(0,0.3/np.sqrt(n),(n,n))
    s0=rng.normal(size=n); k0=np.zeros(n)
    w,e,c,ms=stats(A,C,D,F,s0,k0,L)
    v = "KILL(lin)" if w>0.25 else ("KILL(add)" if e<0.5 else ("KILL(cos)" if c<0.5 else "not killed"))
    print(f"{L:>3} {w:>18.4f} {e:>14.4f} {c:>10.4f} {ms:>12.4f}  {v}")
