"""STAGE-0 PROBE P0.2 -- the Strassen carrier, measured end to end on real flopscope.

Design, chosen 2026-08-19:

  * tau = -2.5 is kept.  After the shipped pruning, the LEAST-alive neurons are
    dropped until the surviving width divides by 4, so both matmul dimensions
    split.  They move from `alive` to `dead` and the existing offset machinery
    carries their closed-form rectified-Gaussian mean exactly, so this costs the
    MSE of at most three near-dead neurons.

  * BLOCK-RESIDENT.  The naive recursion concatenates the four output quadrants
    and re-slices them next layer; flopscope bills concatenate at 1/element and
    it costs ~2.7x that again in residual.  But the next layer's four input
    blocks ARE this layer's four output blocks, so the whole 32-layer chain can
    stay in block form and never concatenate once.  Only the final mean is
    reassembled.

  * The seven weight-side combinations are formed ONCE per MLP (they are
    (k/2, n/2) and do not depend on the sample), not once per sample batch.

Measures, against the dense pruned carrier on identical mints and point sets:
counted FLOPs, operation count, and the numerical deviation of the prediction.
Strassen is exact in exact arithmetic but reorders float32 summation, so the
deviation is measured rather than assumed.

DESIGN TIER.  No gate, no decision block, estimator.py untouched.
Run:  uv run python local_sim/probe_strassen_carrier.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# TRAP: Python puts the script's own directory at sys.path[0], and local_sim/
# contains a numpy SHIM named `flopscope`.  A probe living here that needs the
# REAL cost model must evict its own directory before importing, or it silently
# measures the shim -- whose BudgetContext has none of the real attributes.
sys.path = [q for q in sys.path
            if os.path.abspath(q or os.getcwd()) != _HERE]
import flopscope as flops              # noqa: E402  -- the real one
import flopscope.numpy as fnp          # noqa: E402
assert "local_sim" not in flops.__file__, f"got the shim: {flops.__file__}"

import numpy as np                     # noqa: E402
from scipy.special import ndtr         # noqa: E402
from scipy.stats import norm           # noqa: E402

for _p in (_HERE, _ROOT):              # now safe: flopscope is already cached
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mint_provenance import canonical_mint  # noqa: E402

BITS, DIM, DEPTH, N = 30, 256, 32, 16384
B_BUDGET = 2.72e11
LAMBDA = 1e11
US_PER_OP = 7.0          # calibrated from graded 327743: 1416 ops, 9.845 ms
TAU = -2.5
SEED0 = 30_806_016_500

_dn = np.load(os.path.join(_HERE, "sobol_dirnum.npz"))
DIRNUM = _dn[_dn.files[0]]


def sobol_points(seed):
    idx = np.arange(N, dtype=np.uint32)
    idx = idx ^ (idx >> np.uint32(1))
    idx = idx.reshape((N, 1))
    pts = np.zeros((N, DIM), dtype=np.uint32)
    for k in range(BITS):
        pts ^= ((idx >> np.uint32(k)) & np.uint32(1)) * DIRNUM[k][:DIM].astype(np.uint32)
    sh = np.random.default_rng(seed).integers(0, 1 << BITS, size=DIM, dtype=np.uint32)
    u = ((pts ^ sh).astype(np.float64) + 0.5) * (1.0 / float(1 << BITS))
    return norm.ppf(u).astype(np.float32)


def closure(w64):
    mu, cov = np.zeros(DIM), np.eye(DIM)
    off = np.ones((DIM, DIM)) - np.eye(DIM)
    alphas, mus = [], []
    for w in w64:
        mp = w.T @ mu
        cp = w.T @ cov @ w
        vp = np.maximum(np.diag(cp), 1e-12)
        sg = np.sqrt(vp)
        a = mp / sg
        Phi = ndtr(a)
        phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
        mu = mp * Phi + sg * phi
        ez2 = (mp * mp + vp) * Phi + mp * sg * phi
        cov = np.outer(Phi, Phi) * cp * off + np.diag(np.maximum(ez2 - mu * mu, 0.0))
        alphas.append(a.copy())
        mus.append(mu.copy())
    return alphas, mus


def build(w64, alphas, mus, multiple):
    """Shipped pruning, then trim the least-alive neurons to a multiple of `multiple`."""
    alive, dead = [], []
    for li in range(DEPTH):
        if li == DEPTH - 1:
            alive.append(np.arange(DIM))
            dead.append(None)
            continue
        a = alphas[li]
        idx = np.flatnonzero(a > TAU)
        if idx.size == 0:
            idx = np.flatnonzero(a >= a.max())
        drop = idx.size % multiple
        if drop:                                     # shed the least-alive
            idx = idx[np.argsort(a[idx])[drop:]]
            idx = np.sort(idx)
        alive.append(idx)
        d = np.setdiff1d(np.arange(DIM), idx)
        dead.append(d if d.size else None)

    sliced, offsets, prev = [], [], None
    for li in range(DEPTH):
        w32 = w64[li].astype(np.float32)
        cur = alive[li]
        base = w32 if prev is None else w32[prev]
        sliced.append(np.ascontiguousarray(base[:, cur]))
        dp = dead[li - 1] if li > 0 else None
        offsets.append(None if dp is None
                       else (mus[li - 1][dp].astype(np.float32) @ w32[np.ix_(dp, cur)]))
        prev = cur
    return sliced, offsets, [len(a) for a in alive]


# --------------------------------------------------------------------------- #
def dense_chain(X, sliced, offsets):
    h = X
    for li in range(DEPTH):
        z = h @ sliced[li]
        if offsets[li] is not None:
            z = z + offsets[li]
        h = fnp.maximum(z, fnp.float32(0.0))
    return h.mean(axis=0, dtype=fnp.float64)


def weight_combos(w):
    """The seven Strassen right-hand sides, formed once per MLP per layer."""
    k, n = w.shape
    kh, nh = k // 2, n // 2
    b11, b12 = w[:kh, :nh], w[:kh, nh:]
    b21, b22 = w[kh:, :nh], w[kh:, nh:]
    #        M1          M2    M3          M4          M5   M6          M7
    return (b11 + b22, b11, b12 - b22, b21 - b11, b22, b11 + b12, b21 + b22)


def strassen_chain(X, sliced, offsets):
    """Block-resident depth-1 Strassen: the chain never concatenates."""
    combos = [weight_combos(w) for w in sliced]
    m = X.shape[0] // 2
    k = X.shape[1] // 2
    a11, a12 = X[:m, :k], X[:m, k:]
    a21, a22 = X[m:, :k], X[m:, k:]

    for li in range(DEPTH):
        c1, c2, c3, c4, c5, c6, c7 = combos[li]
        m1 = (a11 + a22) @ c1
        m2 = (a21 + a22) @ c2
        m3 = a11 @ c3
        m4 = a22 @ c4
        m5 = (a11 + a12) @ c5
        m6 = (a21 - a11) @ c6
        m7 = (a12 - a22) @ c7
        t11 = m1 + m4 - m5 + m7
        t12 = m3 + m5
        t21 = m2 + m4
        t22 = m1 - m2 + m3 + m6
        off = offsets[li]
        if off is not None:
            nh = t11.shape[1]
            o1, o2 = off[:nh], off[nh:]
            t11 = t11 + o1
            t12 = t12 + o2
            t21 = t21 + o1
            t22 = t22 + o2
        z = fnp.float32(0.0)
        a11 = fnp.maximum(t11, z)
        a12 = fnp.maximum(t12, z)
        a21 = fnp.maximum(t21, z)
        a22 = fnp.maximum(t22, z)

    top = (a11.sum(axis=0, dtype=fnp.float64) + a21.sum(axis=0, dtype=fnp.float64))
    bot = (a12.sum(axis=0, dtype=fnp.float64) + a22.sum(axis=0, dtype=fnp.float64))
    return fnp.concatenate([top, bot]) / float(X.shape[0])


# --------------------------------------------------------------------------- #
def to_fnp(X, sliced, offsets):
    """Marshal outside the budget: on the grader, weights arrive as fnp arrays."""
    return (fnp.asarray(X),
            [fnp.asarray(w) for w in sliced],
            [None if o is None else fnp.asarray(o) for o in offsets])


def run(fn, X, sliced, offsets):
    Xf, sf, of = to_fnp(X, sliced, offsets)
    with flops.BudgetContext(flop_budget=int(1e14)) as ctx:
        out = fn(Xf, sf, of)
        return np.asarray(out), int(ctx.flops_used), len(ctx.op_log)


def main():
    print(f"N={N}, depth={DEPTH}, tau={TAU}, "
          f"residual price {US_PER_OP} us/op = {US_PER_OP*1e-6*LAMBDA:,.0f} FLOP-eq\n")
    rows = []
    for mi in range(6):
        Wn, _, _ = canonical_mint(DIM, DEPTH, SEED0 + mi)
        w64 = [x.astype(np.float64) for x in Wn]
        alphas, mus = closure(w64)
        X = sobol_points(900_000 + mi)

        sd, od, wd = build(w64, alphas, mus, 1)
        ss, os_, ws = build(w64, alphas, mus, 4)
        yd, fd, nd = run(dense_chain, X, sd, od)
        ys, fs, ns = run(strassen_chain, X, ss, os_)
        yd4, fd4, nd4 = run(dense_chain, X, ss, os_)      # same pruning, dense

        rel = float(np.abs(ys - yd4).max() / np.abs(yd4).max())
        eff_d = fd4 + nd4 * US_PER_OP * 1e-6 * LAMBDA
        eff_s = fs + ns * US_PER_OP * 1e-6 * LAMBDA
        rows.append((np.mean(wd), np.mean(ws), fd, fd4, fs, nd4, ns, rel,
                     eff_d / eff_s, fd / fs))
        print(f"mlp {mi}: alive {np.mean(wd):.1f} -> {np.mean(ws):.1f} (mult of 4) | "
              f"dense {fd4/1e10:.3f}e10 F / {nd4} ops | strassen {fs/1e10:.3f}e10 F / "
              f"{ns} ops | max rel dev {rel:.2e}", flush=True)

    a = np.array(rows, dtype=float)
    print(f"\n{'quantity':<44}{'value':>18}")
    print(f"{'mean alive width, tau=-2.5':<44}{a[:,0].mean():>18.2f}")
    print(f"{'mean alive width, trimmed to multiple of 4':<44}{a[:,1].mean():>18.2f}")
    print(f"{'counted FLOPs, dense (shipped pruning)':<44}{a[:,2].mean():>18,.0f}")
    print(f"{'counted FLOPs, dense (trimmed pruning)':<44}{a[:,3].mean():>18,.0f}")
    print(f"{'counted FLOPs, strassen depth 1':<44}{a[:,4].mean():>18,.0f}")
    print(f"{'ops, dense':<44}{a[:,5].mean():>18,.0f}")
    print(f"{'ops, strassen depth 1':<44}{a[:,6].mean():>18,.0f}")
    print(f"{'max relative prediction deviation':<44}{a[:,7].max():>18.3e}")
    print(f"{'counted-FLOP ratio (dense/strassen)':<44}{a[:,9].mean():>17.4f}x")
    print(f"{'EFFECTIVE ratio at 7.0 us/op  <-- the number':<44}{a[:,8].mean():>17.4f}x")
    print(f"\nC/B  dense {a[:,3].mean()/B_BUDGET:.4f}  ->  strassen "
          f"{a[:,4].mean()/B_BUDGET:.4f}")
    be = (a[:,3].mean() - a[:,4].mean()) / LAMBDA * 1e6 / (a[:,6].mean() - a[:,5].mean())
    print(f"break-even residual price: {be:.1f} us/op   (we measure 7.0)")


if __name__ == "__main__":
    main()
