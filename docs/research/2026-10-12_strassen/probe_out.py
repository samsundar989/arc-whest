"""Task 1: what out= does and bills on REAL flopscope 0.12.1."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "site"))
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
print("flopscope", flops.__version__)
def bill(name, fn):
    try:
        with flops.BudgetContext(flop_budget=2**50, quiet=True) as b:
            out = fn()
        sd = b.summary_dict()
        ops = {k: (v["calls"], v["flop_cost"]) for k, v in sd.get("operations", {}).items()}
        print(f"OK   {name:58s} flops={int(b.flops_used):>9d} ops={ops}")
        return out
    except Exception as e:
        print(f"FAIL {name:58s} {type(e).__name__}: {str(e)[:110]}")
rng = np.random.default_rng(0)
b_, m, k = 3, 8, 6
an = rng.standard_normal((b_, m, k)).astype(np.float32); bn = rng.standard_normal((b_, m, k)).astype(np.float32)
a = fnp.asarray(an); b = fnp.asarray(bn)
print("numel =", an.size)
with flops.BudgetContext(flop_budget=2**50, quiet=True) as bb:
    pass
print("summary_dict keys:", list(bb.summary_dict().keys()))
bill("a + b", lambda: a + b)
bill("fnp.empty((3,8,6),f32)", lambda: fnp.empty((b_, m, k), dtype=np.float32))
bill("fnp.zeros((3,8,6),f32)", lambda: fnp.zeros((b_, m, k), dtype=np.float32))
buf = fnp.empty((b_, m, k), dtype=np.float32)
r = bill("(a) add(a,b,out=buf)", lambda: fnp.add(a, b, out=buf))
print("     returns buf itself:", r is buf, " equal:", np.array_equal(np.asarray(buf), an + bn))
big = fnp.zeros((7 * b_, m, k), dtype=np.float32)
r = bill("(b) add(a,b,out=big[3:6])  [batch-slice view]", lambda: fnp.add(a, b, out=big[3:6]))
print("     read-back ok:", np.array_equal(np.asarray(big)[3:6], an + bn), " rest zero:", float(np.abs(np.asarray(big)[:3]).max()), float(np.abs(np.asarray(big)[6:]).max()))
quad = fnp.zeros((b_, 2 * m, 2 * k), dtype=np.float32)
r = bill("(b) add(a,b,out=quad[:, :m, k:]) [non-contig quadrant]", lambda: fnp.add(a, b, out=quad[:, :m, k:]))
print("     read-back ok:", np.array_equal(np.asarray(quad)[:, :m, k:], an + bn), " others zero:", float(np.abs(np.asarray(quad)[:, m:, :]).max()), float(np.abs(np.asarray(quad)[:, :m, :k]).max()))
r = bill("(b) add(a,b,out=quad[:, m:, :k]) second quadrant", lambda: fnp.add(a, b, out=quad[:, m:, :k]))
r = bill("(c) subtract(a,b,out=big[0:3])", lambda: fnp.subtract(a, b, out=big[0:3]))
print("     ok:", np.array_equal(np.asarray(big)[0:3], an - bn))
r = bill("(c) multiply(a,b,out=big[6:9])", lambda: fnp.multiply(a, b, out=big[6:9]))
print("     ok:", np.array_equal(np.asarray(big)[6:9], an * bn))
# in-place aliasing: out is one of the inputs' views
al = fnp.zeros((b_, m, k), dtype=np.float32); fnp.add(a, 0, out=al)
r = bill("(c) add(al, b, out=al)  [aliasing in-place]", lambda: fnp.add(al, b, out=al))
print("     ok:", np.array_equal(np.asarray(al), an + bn))
# input operands that are views of a written buffer
r = bill("    add(big[3:6], big[0:3]) reading written views", lambda: fnp.add(big[3:6], big[0:3]))
print("     ok:", np.array_equal(np.asarray(r), (an + bn) + (an - bn)))
# (d) matmul out=
A = fnp.asarray(rng.standard_normal((5, 8, 6)).astype(np.float32)); B = fnp.asarray(rng.standard_normal((5, 6, 10)).astype(np.float32))
print("expected batched matmul 5*8*10*(2*6-1) =", 5 * 8 * 10 * 11)
bill("(d) A @ B batched", lambda: A @ B)
mo = fnp.zeros((9, 8, 10), dtype=np.float32)
r = bill("(d) matmul(A,B,out=mo[2:7])", lambda: fnp.matmul(A, B, out=mo[2:7]))
print("     ok:", np.allclose(np.asarray(mo)[2:7], np.asarray(A) @ np.asarray(B), atol=1e-5), " rest zero:", float(np.abs(np.asarray(mo)[:2]).max()))
mq = fnp.zeros((5, 16, 20), dtype=np.float32)
r = bill("(d) matmul(A,B,out=mq[:, 8:, :10]) quadrant view", lambda: fnp.matmul(A, B, out=mq[:, 8:, :10]))
print("     ok:", np.allclose(np.asarray(mq)[:, 8:, :10], np.asarray(A) @ np.asarray(B), atol=1e-5))
# multi-dim batch matmul (no reshape)
A4 = fnp.asarray(rng.standard_normal((7, 7, 8, 6)).astype(np.float32)); B4 = fnp.asarray(rng.standard_normal((7, 7, 6, 10)).astype(np.float32))
print("expected 49*8*10*11 =", 49 * 8 * 10 * 11)
bill("(d) 4-D batch A4 @ B4 (7,7,8,6)@(7,7,6,10)", lambda: A4 @ B4)
bill("(d) strided-view operands A4[:, :4] @ B4[:, :4]", lambda: A4[:, :4] @ B4[:, :4])
bill("(d) transposed-view operand A4.transpose(1,0,2,3) @ B4", lambda: A4.transpose(1, 0, 2, 3) @ B4)
# (e) copies into a view
r = bill("(e) copyto(big[9:12], a)", lambda: fnp.copyto(big[9:12], a))
print("     ok:", np.array_equal(np.asarray(big)[9:12], an))
r = bill("(e) add(a, 0, out=big[12:15])", lambda: fnp.add(a, 0, out=big[12:15]))
r = bill("(e) positive(a, out=big[15:18])", lambda: fnp.positive(a, out=big[15:18]))
print("     ok:", np.array_equal(np.asarray(big)[15:18], an))
r = bill("(e) concatenate([a,b],axis=0,out=big[0:6])", lambda: fnp.concatenate([a, b], axis=0, out=big[0:6]))
r = bill("(e) stack([a,b],axis=0, out=view)", lambda: fnp.stack([a, b], axis=0, out=fnp.zeros((2, b_, m, k), dtype=np.float32)))
# things known/expected to fail or bill
def setit():
    big[0:3] = a
bill("    big[0:3] = a  (slice assignment)", setit)
def iadd():
    x = big[0:3]; x += a
bill("    view += a", iadd)
bill("    a[:, :4, :3] slice", lambda: a[:, :4, :3])
bill("    a[None]", lambda: a[None])
bill("    a[::-1] negative-step view", lambda: a[::-1])
bill("    a.transpose(0,2,1)", lambda: a.transpose(0, 2, 1))
bill("    fnp.transpose(a,(0,2,1))", lambda: fnp.transpose(a, (0, 2, 1)))
bill("    fnp.swapaxes(a,1,2)", lambda: fnp.swapaxes(a, 1, 2))
bill("    fnp.moveaxis(a,0,2)", lambda: fnp.moveaxis(a, 0, 2))
bill("    a.reshape(3,2,4,6) method", lambda: a.reshape(3, 2, 4, 6))
bill("    fnp.reshape(a,(3,2,4,6))", lambda: fnp.reshape(a, (3, 2, 4, 6)))
bill("    a.view() / a.T", lambda: a.T)
bill("    fnp.expand_dims(a,0)", lambda: fnp.expand_dims(a, 0))
bill("    fnp.split(a,3,axis=0)", lambda: fnp.split(a, 3, axis=0))
bill("    fnp.block([[q,q],[q,q]])", lambda: fnp.block([[a, a], [a, a]]))
bill("    fnp.broadcast_to(a,(2,3,8,6))", lambda: fnp.broadcast_to(a, (2, 3, 8, 6)))
bill("    as_strided", lambda: fnp.lib.stride_tricks.as_strided(a, (3, 8, 6), a.strides))
bill("    fnp.negative(a, out=big[18:21])", lambda: fnp.negative(a, out=big[18:21]))
# out= LARGER than broadcast inputs (issue #275 pattern) -- we must NOT rely on this
bill("    [#275] add(a[0], b[0], out=buf) out larger than inputs", lambda: fnp.add(a[0], b[0], out=buf))
# shape mismatch
bill("    add(a,b,out=big[0:2]) wrong shape", lambda: fnp.add(a, b, out=big[0:2]))
# dtype
bill("    add(a,b,out=float64 buf)", lambda: fnp.add(a, b, out=fnp.zeros((b_, m, k), dtype=np.float64)))
bill("    3-term: add(add(a,b),a,out=..) ", lambda: fnp.add(fnp.add(a, b), a, out=buf))
print("type of big[3:6]:", type(big[3:6]).__name__, " writeable flag:", big[3:6].flags.writeable, " base is big:", big[3:6].base is not None)
