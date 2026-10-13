"""Wall-time probe (VM, 1 BLAS thread): one (shape, design, depth) per process."""
import sys, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "site"))
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
import swproto as sp
m, k, n, d, name = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 2
rng = np.random.default_rng(0)
A = fnp.asarray(rng.standard_normal((m, k)).astype(np.float32)); B = fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
fn = {"ref": lambda: sp.mm_ref(A, B, d), "out": lambda: sp.sw_out(A, B, d), "fused": lambda: sp.sw_out(A, B, d, True), "ks": lambda: sp.sw_ks(A, B, d)}[name]
best = None
for r in range(reps):
    t = time.perf_counter()
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as b:
        out = fn()
    dt = time.perf_counter() - t
    sd = b.summary_dict()
    del out
    rec = (dt, sd["flopscope_backend_time_s"], sd["flopscope_overhead_time_s"], sd["residual_wall_time_s"], int(b.flops_used),
           sum(v["calls"] for v in sd["operations"].values()), sd["operations"]["matmul"]["flop_cost"])
    if best is None or rec[0] < best[0]: best = rec
dt, be, ov, res, f, o, leaf = best
import resource
print(f"({m},{k})@({k},{n}) d={d} {name:5s} flops={f:>13d} frac={f/(m*n*(2*k-1)):.4f} ops={o:4d} wall={dt:7.3f}s backend={be:7.3f}s overhead={ov:.3f}s "
      f"ps/FLOP={1e12*dt/f:6.1f} maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB", flush=True)
