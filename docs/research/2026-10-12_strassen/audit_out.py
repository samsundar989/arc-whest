"""Audit: (1) every out= write in the prototypes is billed for numel(out) (so a fix of flopscope #275,
'bill the output shape', cannot raise our bill); (2) no element of an fnp.empty buffer is read before
it is written (NaN poison)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "site"))
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
from flopscope._budget import get_active_budget
import swproto as sp

class Shim:
    """Stands in for `fnp` inside swproto: forwards everything, checks out= calls."""
    def __init__(self): self.n_out = 0; self.under = 0; self.poison = False
    def __getattr__(self, name):
        real = getattr(fnp, name)
        if name == "empty" and self.poison:
            def empty(shape, dtype=float):
                e = real(shape, dtype=dtype); np.asarray(e).fill(np.nan); return e
            return empty
        if name not in ("add", "subtract", "matmul", "concatenate", "copyto"):
            return real
        def f(*a, **kw):
            b = get_active_budget(); before = b.flops_used
            r = real(*a, **kw)
            billed = b.flops_used - before
            out = kw.get("out", a[0] if name == "copyto" else None)
            if out is not None:
                self.n_out += 1
                if name == "matmul":
                    k = a[0].shape[-1]; want = out.size * (2 * k - 1)
                else:
                    want = out.size
                if billed != want: self.under += 1; print("  MISMATCH", name, billed, want, out.shape)
            return r
        return f
shim = Shim(); sp.fnp = shim
rng = np.random.default_rng(3)
for (m, k, n, d) in [(256, 256, 256, 5), (128, 104, 1312, 3), (216, 2784, 216, 3), (256, 64, 1408, 4), (512, 448, 320, 6)]:
    An = rng.standard_normal((m, k)).astype(np.float32); Bn = rng.standard_normal((k, n)).astype(np.float32)
    A = fnp.asarray(An); B = fnp.asarray(Bn); ref = An.astype(np.float64) @ Bn.astype(np.float64)
    for poison in (False, True):
        shim.poison = poison
        with flops.BudgetContext(flop_budget=2**60, quiet=True):
            outs = {"out": sp.sw_out(A, B, d), "fused": sp.sw_out(A, B, d, True), "ks": sp.sw_ks(A, B, d)}
        for nm, o in outs.items():
            o = np.asarray(o)
            assert np.isfinite(o).all(), (nm, "NaN leaked from an unwritten buffer slot")
            assert np.linalg.norm(o - ref) / np.linalg.norm(ref) < 1e-3
    print((m, k, n, d), "ok: finite with NaN-poisoned buffers, numerically correct")
print("out= writes checked:", shim.n_out, " billed != numel(out) [or (2k-1)*numel(out) for matmul]:", shim.under)
