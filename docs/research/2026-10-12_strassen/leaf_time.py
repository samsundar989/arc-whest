"""VM leaf-kernel timing: batched fnp.matmul ps/FLOP versus leaf shape (1 BLAS thread)."""
import sys, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "site"))
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
rng = np.random.default_rng(0)
print("summary op fields:", end=" ")
with flops.BudgetContext(flop_budget=2**60, quiet=True) as b:
    x = fnp.asarray(np.ones((4, 4), np.float32)); y = x @ x
print(b.summary_dict()["operations"])
def t(bsz, m, k, n, reps=3):
    A = fnp.asarray(rng.standard_normal((bsz, m, k)).astype(np.float32)); B = fnp.asarray(rng.standard_normal((bsz, k, n)).astype(np.float32))
    out = fnp.empty((bsz, m, n), dtype=np.float32)
    best = 1e9
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as bb:
        fnp.matmul(A, B, out=out)
        for _ in range(reps):
            t0 = time.perf_counter(); fnp.matmul(A, B, out=out); best = min(best, time.perf_counter() - t0)
    f = bsz * m * n * (2 * k - 1)
    # elementwise add pass over the output-sized array, for the glue rate
    t0 = time.perf_counter()
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as bb:
        fnp.add(out, out, out=out)
    ta = time.perf_counter() - t0
    print(f"leaf ({m:3d},{k:3d},{n:3d}) batch={bsz:7d} flops={f:.3e}  matmul {1e12*best/f:7.1f} ps/FLOP ({1e6*best/bsz:6.2f} us/leaf)   add pass {1e9*ta/out.size:5.2f} ns/elt")
for L, s in ((4, 64), (5, 32), (6, 16), (7, 8)):
    t(min(7 ** L, 120000), s, s, s)
t(16807, 32, 26, 166); t(117649, 16, 13, 83)
t(16807, 27, 174, 27); t(117649, 14, 87, 14)
t(2401, 64, 16, 350); t(16807, 32, 8, 175); t(117649, 16, 4, 88)
