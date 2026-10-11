import sys, os
# needs the real flopscope 0.12.1 on the path (the repo venv)
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
print("flopscope", getattr(flops, "__version__", "?"))
def bill(name, fn):
    with flops.BudgetContext(flop_budget=2**50, quiet=True) as b:
        out = fn()
    print(f"{name:52s} flops={int(b.flops_used):>10d} dtype={getattr(out,'dtype',None)} shape={getattr(out,'shape',None)}")
    return out
rng = np.random.default_rng(0)
n = 64
Ar = rng.standard_normal((n, n)).astype(np.float32)
Br = rng.standard_normal((n, n)).astype(np.float32)
Cr = rng.standard_normal((n, n)).astype(np.float32)
A = fnp.asarray(Ar); B = fnp.asarray(Br); C = fnp.asarray(Cr)
Z = fnp.asarray((Br + 1j * Cr).astype(np.complex64))
Ac = fnp.asarray(Ar.astype(np.complex64))
print("expected real matmul n*n*(2n-1) =", n * n * (2 * n - 1))
bill("f32 @ f32", lambda: A @ B)
o = bill("f32 @ c64 (A @ (B+iC))", lambda: A @ Z)
bill("c64 @ c64", lambda: Ac @ Z)
bill("matmul(f32, c64)", lambda: fnp.matmul(A, Z))
bill("einsum ij,jk f32,c64", lambda: fnp.einsum("ij,jk->ik", A, Z))
bill("c64 * c64 elementwise", lambda: Z * Z)
bill("c64 + c64", lambda: Z + Z)
bill("real(Z) ", lambda: fnp.real(Z) if hasattr(fnp, "real") else None)
bill("imag(Z) ", lambda: fnp.imag(Z) if hasattr(fnp, "imag") else None)
bill("B + 1j*C construct", lambda: B + C * np.complex64(1j))
o = np.asarray(o) if not isinstance(o, np.ndarray) else o
print("max |Re(A@Z) - A@B| =", float(np.abs(np.real(np.asarray(o)) - Ar @ Br).max()),
      " max |Im - A@C| =", float(np.abs(np.imag(np.asarray(o)) - Ar @ Cr).max()))
