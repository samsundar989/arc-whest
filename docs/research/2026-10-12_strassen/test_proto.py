import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "site"))
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
import swproto as sp

def meter(fn):
    t = time.perf_counter()
    with flops.BudgetContext(flop_budget=2**60, quiet=True) as b:
        out = fn()
    dt = time.perf_counter() - t
    sd = b.summary_dict()
    nops = sum(v["calls"] for v in sd["operations"].values())
    return out, int(b.flops_used), nops, dt, sd

def run(m, k, n, depths, seed=0, which=("ref", "out", "fused", "ks")):
    rng = np.random.default_rng(seed)
    An = rng.standard_normal((m, k)).astype(np.float32); Bn = rng.standard_normal((k, n)).astype(np.float32)
    ref64 = An.astype(np.float64) @ Bn.astype(np.float64)
    nrm = np.linalg.norm(ref64)
    A = fnp.asarray(An); B = fnp.asarray(Bn)
    dense = m * n * (2 * k - 1)
    e0 = np.linalg.norm((An @ Bn) - ref64) / nrm
    print(f"\n=== ({m},{k})@({k},{n})  dense={dense}  plain f32 rel.err={e0:.2e}")
    for d in depths:
        if (m % (1 << d)) or (k % (1 << d)) or (n % (1 << d)):
            print(f" d={d}: dims not divisible by 2^{d}, skipped"); continue
        for name in which:
            fn = {"ref": lambda: sp.mm_ref(A, B, d), "out": lambda: sp.sw_out(A, B, d),
                  "fused": lambda: sp.sw_out(A, B, d, True), "ks": lambda: sp.sw_ks(A, B, d)}[name]
            model = {"ref": lambda: sp.cost_ref(m, k, n, d), "out": lambda: sp.cost_out(m, k, n, d),
                     "fused": lambda: sp.cost_out(m, k, n, d, True), "ks": lambda: sp.cost_ks(m, k, n, d)}[name]()
            out, f, o, dt, sd = meter(fn)
            err = np.linalg.norm(np.asarray(out).astype(np.float64) - ref64) / nrm
            leaf = sd["operations"]["matmul"]["flop_cost"]
            ok = "MODEL=METER" if (f, o) == tuple(model) else f"MODEL MISMATCH {model}"
            print(f" d={d} {name:5s} flops={f:>14d} frac_dense={f/dense:.4f} glue={f-leaf:>12d} ({100*(f-leaf)/f:4.1f}%) ops={o:4d} "
                  f"relerr={err:.2e} ({err/e0:5.1f}x) wall={dt:6.2f}s {ok}")
            del out

if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        m, k, n = map(int, args[:3]); depths = [int(x) for x in args[3].split(",")]
        which = tuple(args[4].split(",")) if len(args) > 4 else ("ref", "out", "fused", "ks")
        run(m, k, n, depths, which=which)
    else:
        run(64, 64, 64, [1, 2, 3])
        run(96, 64, 160, [1, 2, 3, 4])
        run(256, 256, 256, [3, 4, 5, 6])
