"""Fast gates for the stage-1 v32 arms, run BEFORE any P11 read.

Each claim can fail on its own; thresholds are fixed here, before running.

  1. NESTING: v32 generated with `_HUB_ROUTES = ()` is frozen v31 bit for bit,
     in output and in counted FLOPs, at two tiny shapes.
  2. BIRTH TENSOR (f64, real tiny chain, identical incoming state): the newborn
     full tensor of v32_dense minus v31_dense equals the oracle's independently
     formed compensated routes (i)+(ii) -- zero on repeated entries, the
     derived hub on all-distinct entries -- at every birth. rel <= 1e-10.
  3. CONSUMED SLICES (f64): the reconstructed v32_dense tower tensor (hubs
     included), transported by the next layer's weights, reproduces the file's
     own `_dslice3`/`_dslice21`. rel <= 1e-10. This is the implementation of
     `D3 += 2 y.R`, `D21 += ...` checked against a dense contraction.
  4. EXACT LAW: at one birth from the scale mixture, v32_dense's all-distinct
     d^2*eps response matches the exact quadrature response (5%).
  5. FLOAT32: `predict` (float32) vs the float64 layer loop, v32 vs v31 --
     v32's worst relative row error <= 2x v31's own + 1e-5.
  6. EQUIVARIANCE: relabelling a hidden layer moves v32's prediction by no
     more than 100x v31's own float32 floor (the v31 gate).
  7. COST: counted FLOPs v32/v31 at 96x6 and 128x8 printed; the budget guard
     over-predicts measured FLOPs for every arm; at 1024x16 the guard picks the
     same `gens` for v32 and v31.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

import k4feed_oracle as K  # noqa: E402
import s1_build_v32 as B  # noqa: E402

OK = []


def claim(name, ok, detail=""):
    OK.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<62} {detail}")


def _predict(mod, w, width, depth, seed, budget=2 ** 41):
    import flopscope as flops
    import harness
    mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=width, depth=depth, seed=seed)
    ctx = flops.BudgetContext(flop_budget=budget, quiet=True)
    with ctx:
        p = mod.Estimator().predict(mlp, budget)
    return np.asarray(p, dtype=np.float64), float(ctx.flops_used)


def main():
    import flopscope as flops
    import flopscope.numpy as fnp
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    tmp = tempfile.mkdtemp(prefix="s1v32_")
    with open(B.SOURCE, "rb") as fh:
        src_raw = fh.read()
    nest = os.path.join(tmp, "v32_noroutes.py")
    with open(nest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(B.build((), True, 32, "nesting arm", src_raw.decode("utf-8"),
                         hashlib.sha256(src_raw).hexdigest()))
    m31 = K._load(os.path.join(_ROOT, "estimator_k3_v31.py"))
    m31d = K._load(os.path.join(_ROOT, "estimator_k3_v31_dense.py"))
    m32 = K._load(os.path.join(_ROOT, "estimator_k3_v32.py"))
    m32d = K._load(os.path.join(_ROOT, "estimator_k3_v32_dense.py"))
    m32i = K._load(os.path.join(_ROOT, "estimator_k3_v32_i.py"))
    m32u = K._load(os.path.join(_ROOT, "estimator_k3_v32_uncomp.py"))
    mnest = K._load(nest)

    # ---- 1. nesting -------------------------------------------------------
    print("1. nesting: _HUB_ROUTES = () is v31")
    for (wd, dp, sd) in ((64, 5, 90000000501), (96, 6, 90000000502)):
        w, es, _ = canonical_mint(wd, dp, sd)
        p31, f31 = _predict(m31, w, wd, dp, es)
        pn, fn = _predict(mnest, w, wd, dp, es)
        p32, f32 = _predict(m32, w, wd, dp, es)
        s = max(np.abs(p31).max(), 1e-300)
        claim(f"{wd}x{dp}: no-route arm == v31 bitwise and same FLOPs",
              np.array_equal(p31, pn) and f31 == fn, f"flops {fn:.0f} vs {f31:.0f}")
        claim(f"{wd}x{dp}: v32 is not a no-op", np.abs(p32 - p31).max() / s > 1e-7,
              f"rel {np.abs(p32 - p31).max() / s:.2e}  cost x{f32 / f31:.5f}")

    # ---- 2 & 3. birth tensor and consumed slices, f64 chain --------------
    print("\n2-3. f64 chain, n=9, depth=6, lam patched to 0.15")
    n, depth = 9, 6
    rng = np.random.default_rng(11)
    Ws = [rng.normal(size=(n, n)) * np.sqrt(2.0 / n) for _ in range(depth + 1)]
    lam = 0.15
    state = (np.zeros(n), np.eye(n), None, 0.0, np.zeros(n))
    worst_birth, worst_slices, max_hub = 0.0, 0.0, 0.0
    tw31 = None
    for l in range(depth):
        mu, cov, tw, c4, gk4 = state
        li = max(l, 1)
        # IDENTICAL incoming state, incoming hubs included: the same module is
        # run twice on the same tower, once with births' routes switched off.
        # (Stripping the hubs instead would change the consumed D3/D21 and so
        # the table's repeated slices -- a different birth, not the same one.)
        out32 = K.run_layer(m32d, Ws[l], mu, cov, gk4, lam, layer_index=li, tw=tw)
        saved = m32d._HUB_ROUTES
        m32d._HUB_ROUTES = ()
        try:
            out31 = K.run_layer(m32d, Ws[l], mu, cov, gk4, lam, layer_index=li, tw=tw)
        finally:
            m32d._HUB_ROUTES = saved
        # newborn lineage only: take the youngest block of each tower
        T32 = _youngest_tensor(out32[2], n)
        T31 = _youngest_tensor(out31[2], n)
        # the oracle's routes, from THIS layer's own coefficients (f64, independent formulas)
        W = Ws[l]
        cov_pre = W.T @ np.asarray(cov) @ W
        var = np.maximum(np.diag(cov_pre), 1e-12)
        sig = np.sqrt(var)
        alpha = np.clip((W.T @ np.asarray(mu)) / sig, -40.0, 40.0)
        ph = K._phi(alpha)
        w1, w2, w3 = ndtr(alpha), ph / sig, -alpha * ph / var
        C_off = cov_pre * (1.0 - np.eye(n))
        lam_l = lam if l >= 1 else 0.0
        if l == 0:
            lam_l = lam  # layer_index clamps to 1, so lam is live at l=0 too
        dG = (W.T ** 2) @ (np.asarray(gk4) - np.diag(np.asarray(cov)) * lam_l) + var * lam_l
        D22 = (2.0 / 6.0) * (dG[:, None] + dG[None, :]) * (1.0 - np.eye(n))
        expect = K._compensated(K.feed_tensor(lam_l, 2.0, w1, w2, C_off)
                                + K.link_tensor(D22, w1, w2, w3, C_off))
        diff = T32 - T31
        sc = max(np.abs(T31).max(), 1e-300)
        worst_birth = max(worst_birth, np.abs(diff - expect).max() / sc)
        max_hub = max(max_hub, np.abs(expect).max() / sc)
        # consumed slices after one transport
        WT = Ws[l + 1].T
        Tfull = K.tower_tensor(out32[2], n)
        Tt = np.einsum("ai,bj,ck,ijk->abc", WT, WT, WT, Tfull)
        with flops.BudgetContext(flop_budget=2 ** 50, quiet=True):
            adv = m32d._advance(fnp.asarray(WT, dtype=fnp.float64), out32[2])
            off = fnp.ones((n, n), dtype=fnp.float64) - fnp.eye(n, dtype=fnp.float64)
            d3 = np.asarray(m32d._dslice3(adv), dtype=np.float64)
            d21 = np.asarray(m32d._dslice21(adv, off), dtype=np.float64)
        e3, e21 = K.slices(Tt)
        s2 = max(np.abs(e21).max(), np.abs(e3).max(), 1e-300)
        worst_slices = max(worst_slices, np.abs(d3 - e3).max() / s2, np.abs(d21 - e21).max() / s2)
        state = (np.asarray(out32[0], dtype=np.float64), np.asarray(out32[1], dtype=np.float64),
                 out32[2], out32[3], np.asarray(out32[4], dtype=np.float64))
    claim("2. newborn(v32_dense) - newborn(v31_dense) == oracle routes (i)+(ii)",
          worst_birth <= 1e-10, f"worst rel {worst_birth:.2e}  (hub size {max_hub:.2e} rel)")
    claim("2. the routes are not vacuous at this fixture", max_hub > 1e-4, f"{max_hub:.2e}")
    claim("3. transported full tensor == _dslice3/_dslice21 (hubs included)",
          worst_slices <= 1e-10, f"worst rel {worst_slices:.2e}")

    # ---- 4. exact law ------------------------------------------------------
    print("\n4. exact scale-mixture law, n=3")
    mu = np.array([0.35, -0.2, 0.1])
    C = np.random.default_rng(20260916).normal(size=(3, 3))
    C = (C + C.T) * 0.5
    np.fill_diagonal(C, 0.0)

    def resp(fn, d=0.02, eps=0.01):
        return ((fn(d, eps) - fn(0.0, eps)) - (fn(d, -eps) - fn(0.0, -eps))) / (2 * d * d * eps)

    def exact(d, eps):
        return K._mixture_cumulants(mu, np.eye(3) + eps * C, d)["k3_012"]

    def est(mod):
        def f(d, eps):
            out = K.run_layer(mod, np.eye(3), mu, np.eye(3) + eps * C, np.full(3, 1.5 * d * d),
                              3.0 * d * d)
            return K.tower_tensor(out[2], 3)[0, 1, 2]
        return f
    ex = resp(exact)
    r32 = resp(est(m32d))
    claim("4. v32_dense all-distinct response == exact (5%)",
          abs(r32 - ex) <= 0.05 * abs(ex), f"exact {ex:+.4e}  v32 {r32:+.4e}")

    # ---- 5. float32 vs float64 ---------------------------------------------
    print("\n5. float32 predict vs float64 layer loop, 64x5")
    wd, dp = 64, 5
    w, es, _ = canonical_mint(wd, dp, 90000000503)
    errs = {}
    for name, mod in (("v31", m31), ("v32", m32)):
        p32, _ = _predict(mod, w, wd, dp, es)
        st = (np.zeros(wd), np.eye(wd), None, 0.0, np.zeros(wd))
        rows = []
        for l in range(dp):
            import flopscope as _fl
            dt = fnp.float64
            with _fl.BudgetContext(flop_budget=2 ** 50, quiet=True):
                const = mod.Estimator()._constants(wd, dt)
                o = mod._layer(fnp.asarray(np.asarray(w[l], dtype=np.float64), dtype=dt),
                               (fnp.asarray(st[0], dtype=dt), fnp.asarray(st[1], dtype=dt), st[2], st[3],
                                fnp.asarray(st[4], dtype=dt)),
                               mod._tables(), dt, const, 2.0, last_layer=(l == dp - 1),
                               gens=None, layer_index=l)
            rows.append(np.asarray(o[0], dtype=np.float64))
            if l < dp - 1:
                st = (np.asarray(o[0], dtype=np.float64), np.asarray(o[1], dtype=np.float64), o[2], o[3],
                      np.asarray(o[4], dtype=np.float64))
        f64 = np.maximum(np.stack(rows), 0.0)
        e = max(np.abs(p32[l] - f64[l]).max() / max(np.abs(f64[l]).max(), 1e-300)
                for l in range(1, dp))
        errs[name] = e
    claim("5. v32 float32 error <= 2x v31's own + 1e-5",
          errs["v32"] <= 2.0 * errs["v31"] + 1e-5, f"v31 {errs['v31']:.2e}  v32 {errs['v32']:.2e}")

    # ---- 6. equivariance ---------------------------------------------------
    print("\n6. permutation equivariance, 96x6")
    wd, dp = 96, 6
    rng = np.random.default_rng(20260916)
    w, es, _ = canonical_mint(wd, dp, 90000000504)
    for lay in (1, dp - 2):
        perm = rng.permutation(wd)
        wp = [np.array(x, copy=True) for x in w]
        wp[lay] = wp[lay][:, perm]
        wp[lay + 1] = wp[lay + 1][perm, :]
        err = {}
        for name, mod in (("v31", m31), ("v32", m32)):
            base, _ = _predict(mod, w, wd, dp, es)
            pout, _ = _predict(mod, wp, wd, dp, es)
            moved = base.copy()
            moved[lay] = base[lay][perm]
            err[name] = float(np.mean((moved - pout) ** 2)) / max(float(np.mean(base ** 2)), 1e-300)
        ratio = err["v32"] / max(err["v31"], 1e-300)
        claim(f"6. layer {lay}: v32 equivariance error <= 100x v31's", ratio <= 100.0,
              f"v31 {err['v31']:.2e}  v32 {err['v32']:.2e}  ratio {ratio:.1f}x")

    # ---- 7. cost -----------------------------------------------------------
    print("\n7. cost, metered")
    for cw, cd in ((96, 6), (128, 8)):
        w, es, _ = canonical_mint(cw, cd, 90000000506)
        cost = {}
        for name, mod in (("v31", m31), ("v32", m32), ("v32_i", m32i), ("v32_uncomp", m32u),
                          ("v32_dense", m32d)):
            _, cost[name] = _predict(mod, w, cw, cd, es)
            g = mod._fit_generations(cw, cd, 2 ** 41)
            model = mod._projected_flops(cw, cd, g)
            claim(f"7. {cw}x{cd} {name}: guard over-predicts", model >= cost[name],
                  f"model/measured {model / cost[name]:.4f}")
        print(f"      {cw}x{cd}: v32/v31 counted FLOPs x{cost['v32'] / cost['v31']:.5f}")
    g31 = m31._fit_generations(1024, 16, 2 ** 41)
    g32 = m32._fit_generations(1024, 16, 2 ** 41)
    claim("7. 1024x16: guard picks the same gens for v32 and v31", g31 == g32,
          f"{g31} / {g32}; projected x{m32._projected_flops(1024, 16, g32) / m31._projected_flops(1024, 16, g31):.5f}")

    print(f"\n{sum(OK)}/{len(OK)} claims pass")
    return 0 if all(OK) else 1


def _youngest_tensor(tw, n):
    """Full tensor of the YOUNGEST lineage only (P = I, its hubs included)."""
    class _One:
        pass
    o = _One()
    k = tw.k
    sl = slice((k - 1) * n, k * n)
    o.k = 1
    o.A_L = np.asarray(tw.A_L)[:, sl]
    o.M = None
    o.w2b = np.asarray(tw.w2b)[sl]
    o.A_D = None if tw.A_D is None else np.asarray(tw.A_D)[:, sl]
    r = np.asarray(tw.L).shape[2]
    o.Z = np.asarray(tw.Z)[:, (k - 1) * r:k * r]
    o.L = np.asarray(tw.L)[k - 1:k]
    o.sb = np.asarray(tw.sb)[sl]
    o.eb = np.asarray(tw.eb)[sl]
    Y = getattr(tw, "Y", None)
    o.Y = None if Y is None else np.asarray(Y)[:, 2 * (k - 1):2 * k]
    o.CA = None if Y is None else np.asarray(tw.CA)[sl]
    o.CB = None if Y is None else np.asarray(tw.CB)[sl]
    return K.tower_tensor(o, n)


if __name__ == "__main__":
    raise SystemExit(main())
