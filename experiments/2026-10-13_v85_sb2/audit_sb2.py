"""S1 check 5: every `out=` write in the built sb2 estimator's `_swo` has out.shape == the broadcast
shape of its operands (flopscope #275), is billed numel(out) [(2k-1)*numel(out) for matmul], and no
NaN-poisoned `empty` element is ever read. Also bit-identity vs `_sb` on C- and F-ordered operands.

    python experiments/2026-10-13_s/audit_sb2.py <built estimator.py>
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "local_sim"))
import numpy as np  # noqa: E402
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
from flopscope._budget import get_active_budget  # noqa: E402

spec = importlib.util.spec_from_file_location("cand", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Shim:
    def __init__(self):
        self.n_out, self.bad, self.poison = 0, 0, False

    def __getattr__(self, name):
        real = getattr(fnp, name)
        if name == "empty":
            def empty(shape, dtype=float):
                e = real(shape, dtype=dtype)
                if self.poison:
                    np.asarray(e).fill(np.nan)
                return e
            return empty
        if name not in ("add", "subtract", "matmul", "concatenate", "copyto"):
            return real

        def f(*a, **kw):
            out = kw.get("out", a[0] if name == "copyto" else None)
            b = get_active_budget()
            before = b.flops_used
            r = real(*a, **kw)
            billed = b.flops_used - before
            if out is not None:
                self.n_out += 1
                if name == "matmul":
                    want_shape = tuple(np.broadcast_shapes(a[0].shape[:-2], a[1].shape[:-2])) + (a[0].shape[-2], a[1].shape[-1])
                    want_bill = out.size * (2 * a[0].shape[-1] - 1)
                elif name == "concatenate":
                    parts = a[0]
                    want_shape = (sum(p.shape[0] for p in parts),) + tuple(parts[0].shape[1:])
                    want_bill = out.size
                elif name == "copyto":
                    want_shape = tuple(a[1].shape)
                    want_bill = out.size
                else:
                    want_shape = tuple(np.broadcast_shapes(a[0].shape, a[1].shape))
                    want_bill = out.size
                if tuple(out.shape) != want_shape or billed != want_bill:
                    self.bad += 1
                    print("  BAD", name, tuple(out.shape), want_shape, billed, want_bill)
            return r
        return f


shim = Shim()
mod.fnp = shim
rng = np.random.default_rng(3)
cases = [(256, 256, 256, 5), (128, 104, 1312, 3), (216, 2784, 216, 3), (256, 64, 1408, 4), (512, 448, 320, 5),
         (1024, 1024, 1056, 5), (800, 1984, 256, 3)]
ok = True
for (m, k, n, d) in cases:
    for order in ("CC", "FC", "CF", "FF"):
        An = rng.standard_normal((m, k)).astype(np.float32)
        Bn = rng.standard_normal((k, n)).astype(np.float32)
        if order[0] == "F":
            An = np.asfortranarray(An)
        if order[1] == "F":
            Bn = np.asfortranarray(Bn)
        A, B = fnp.asarray(An), fnp.asarray(Bn)
        for poison in (False, True):
            shim.poison = poison
            with flops.BudgetContext(flop_budget=2 ** 60, quiet=True):
                C = fnp.empty((1, m, n), dtype=A.dtype)
                mod._swo(A[None], B[None], C, d)
                ref = mod._sb(A[None], B[None], d)
            c, r = np.asarray(C), np.asarray(ref)
            fin = bool(np.isfinite(c).all())
            eq = bool(np.array_equal(c, r))
            ok &= fin and eq
            if not (fin and eq):
                print("  FAIL", (m, k, n, d), order, "poison" if poison else "", "finite", fin, "equal", eq)
    print((m, k, n, d), "CC/FC/CF/FF x poison: done", flush=True)
print(f"out= writes checked: {shim.n_out}; shape or bill != expected: {shim.bad}; "
      f"bit-identical and NaN-free on all cases: {ok}")
sys.exit(0 if ok and shim.bad == 0 else 1)
