import numpy as np
import flopscope as flops, flopscope.numpy as fnp
def bill(name, fn):
    with flops.BudgetContext(flop_budget=2**50, quiet=True) as b:
        out = fn()
    print(f"{name:45s} flops={int(b.flops_used):>12d} type={type(out).__name__} shape={getattr(out,'shape',None)}")
    return out
rng=np.random.default_rng(0)
n=10
X=fnp.asarray(rng.standard_normal((n,n)).astype(np.float32))
v=fnp.asarray(rng.standard_normal((n,)).astype(np.float32))
S=bill("X + X.T", lambda: X + X.T)
bill("(X+X.T)*0.5", lambda: S*0.5)
om = bill("ones-eye", lambda: fnp.ones((n,n),dtype=fnp.float32)-fnp.eye(n,dtype=fnp.float32))
bill("X*om", lambda: X*om)
bill("v[:,None]+v[None,:]", lambda: v[:,None]+v[None,:])
D=bill("diag(v)", lambda: fnp.diag(v))
bill("diag(v)+X", lambda: D+X)
bill("om*om", lambda: om*om)
E=bill("eye", lambda: fnp.eye(n,dtype=fnp.float32))
W=fnp.asarray(rng.standard_normal((n,n)).astype(np.float32))
bill("W.T @ eye", lambda: W.T @ E)
bill("W.T @ W", lambda: W.T @ W)
bill("diagonal(eye)", lambda: fnp.diagonal(E))
bill("X @ X.T", lambda: X @ X.T)
