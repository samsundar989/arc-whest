"""Fast label-free gates for the handoff-09-17 v35 arms, run BEFORE any truth read.

Thresholds are fixed here, before the first run.

  1. NESTING: v35_off == v32 bit for bit, output AND counted FLOPs (64x5, 96x6).
  2. READOUT ISOLATION: v35_R's rows 0..depth-2 are v32's bit for bit; the final
     row moves (not a null edit).
  3. CLOSURE PARITY (f64, injected boundary, live `_layer`): v35_Q's mu_out,
     cov_out, g_out, S3 and K21 == `closure_audit.ref_step` on the independently
     generated BUDGET-4 table. rel <= 1e-10.
  4. PROTECTED CONTENT (f64, injected boundary): the (U V^T)*off that v35_P
     (resp. v35_QP) removes from the residual equals the reference's D22-driven
     K21 families of the SIMPLE (resp. BUDGET-4) table. rel <= 1e-10.
  5. FULL-RANK IDENTITY (f64 layer loop on a real tiny chain, every row):
     v35_P_dense == v32_dense and v35_QP_dense == v35_Q_dense, rel <= 1e-9.
  6. FLOAT32: `predict` vs the f64 layer loop; each arm's worst relative row error
     <= 2x v32's own + 1e-5.
  7. EQUIVARIANCE: relabelling hidden layer 1 moves each rank-32 arm's prediction
     by <= 100x v32's own float32 floor.
  8. COST + GUARD: counted FLOPs ratio vs v32 printed at 96x6 and 128x8; the budget
     guard over-predicts the measured FLOPs for every arm; the guard picks the
     same `gens` as v32 at 1024x16.
  9. FINITE: no non-finite prediction at a large-variance 64x6 fixture.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_AUD = os.path.join(_HERE, "closure_audit")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in (_HERE, _AUD)] + [_ROOT, _HERE, _AUD]

import numpy as np  # noqa: E402

OK = []


def claim(name, ok, detail=""):
    OK.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<66} {detail}")


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def layer_loop(mod, w, n, depth, perm=None):
    import flopscope as flops
    import flopscope.numpy as fnp
    st = (np.zeros(n), np.eye(n), None, 0.0, np.zeros(n))
    rows = []
    const = mod.Estimator()._constants(n, fnp.float64)
    for l in range(depth):
        W = np.asarray(w[l], dtype=np.float64)
        with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
            o = mod._layer(fnp.asarray(W, dtype=fnp.float64),
                           (fnp.asarray(st[0], dtype=fnp.float64), fnp.asarray(st[1], dtype=fnp.float64),
                            st[2], st[3], fnp.asarray(st[4], dtype=fnp.float64)),
                           mod._tables(), fnp.float64, const, 2.0,
                           last_layer=(l == depth - 1), gens=None, layer_index=l)
        rows.append(np.asarray(o[0], dtype=np.float64))
        st = (o[0], o[1], o[2], o[3], o[4])
    return np.stack(rows)


def main():
    import flopscope as flops
    import k4feed_oracle as K
    import s1_parity_v32 as P1
    import test_ref_step as T
    import ref_step
    import canon
    import s6_build_closure as B
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    L = lambda name: K._load(os.path.join(_ROOT, name))
    m32, m32d = L("estimator_k3_v32.py"), L("estimator_k3_v32_dense.py")
    arms = {a: L(f"estimator_k3_v35_{a}.py") for a in ("off", "R", "Q", "P", "QP")}
    dense = {a: L(f"estimator_k3_v35_{a}_dense.py") for a in ("P", "QP", "Q")}
    b4 = [dict(output=list(k[0]), factors=list(k[1]), derivs=list(k[2]), coef=str(c))
          for k, c in B.load_table(os.path.join(_HERE, "closure", "closure_budget4.json")).items()]
    v32_terms, _ = canon.canon_ours(os.path.join(_ROOT, "estimator_k3_v32.py"))
    b2 = canon.to_records(v32_terms)

    print("\n1-2. nesting and readout isolation")
    for wd, dp, sd in ((64, 5, 90000000801), (96, 6, 90000000802)):
        w, es, _ = canonical_mint(wd, dp, sd)
        pb, fb = P1._predict(m32, w, wd, dp, es)
        po, fo = P1._predict(arms["off"], w, wd, dp, es)
        claim(f"{wd}x{dp} v35_off == v32 bitwise + FLOPs", np.array_equal(pb, po) and fb == fo,
              f"flops {fo:.0f} vs {fb:.0f}")
        pr, fr = P1._predict(arms["R"], w, wd, dp, es)
        claim(f"{wd}x{dp} v35_R rows 0..d-2 == v32 bitwise", np.array_equal(pr[:-1], pb[:-1]))
        claim(f"{wd}x{dp} v35_R final row moves", rel(pr[-1], pb[-1]) > 1e-7, "rel %.2e" % rel(pr[-1], pb[-1]))

    print("\n3. closure parity, live _layer vs reference (BUDGET 4), f64 injected boundary")
    for n, seed, vs in ((7, 1, 1.0), (9, 2, 0.15), (8, 3, 3.0)):
        bd = T.fixture(n, 20260917_00 + seed, vs)
        live = T.run_live(L("estimator_k3_v35_Q.py"), bd)
        ref = ref_step.step(bd, b4, want_families=True)
        Kr = ref["K"]
        mu, sigma = bd["mu_pre"], np.sqrt(np.diag(bd["cov_pre"]))
        w1, w2 = ref_step.wick(1, 1, mu, sigma), ref_step.wick(2, 1, mu, sigma)
        off = 1.0 - np.eye(n)
        C11 = bd["cov_pre"] * off
        tag = f"n={n} s={vs}"
        for key in ("mu_out", "cov_out", "g_out"):
            claim(f"{tag} Q {key}", rel(live[key], ref[key]) <= 1e-10, "%.1e" % rel(live[key], ref[key]))
        claim(f"{tag} Q S3", rel(live["sb"], Kr[(3,)] - w1 ** 3 * bd["D3"]) <= 1e-10,
              "%.1e" % rel(live["sb"], Kr[(3,)] - w1 ** 3 * bd["D3"]))
        fac1 = w1[:, None] * C11
        p111 = (w1[:, None] ** 2 * w1[None, :]) * bd["D21"] + ((fac1 * fac1) * w2[None, :]) * off
        K21 = live["R"] + (live["eb"][:, None] * C11) * w1[None, :] + p111
        claim(f"{tag} Q K21", rel(K21, Kr[(2, 1)]) <= 1e-10, "%.1e" % rel(K21, Kr[(2, 1)]))

    print("\n4. protected content equals the reference D22-driven K21 families")
    for arm, table in (("P", b2), ("QP", b4)):
        for n, seed, vs in ((7, 4, 1.0), (9, 5, 0.3)):
            bd = T.fixture(n, 20260917_00 + seed, vs)
            # hubs ON: the protection lives in the hub birth branch; their compensation
            # is identical in both runs, so the difference is exactly what was protected
            with_p = T.run_live(L(f"estimator_k3_v35_{arm}.py"), bd, keep_hubs=True)
            without = T.run_live(L("estimator_k3_v35_Q.py" if arm == "QP" else "estimator_k3_v35_off.py"),
                                 bd, keep_hubs=True)
            removed = without["R"] - with_p["R"]
            ref = ref_step.step(bd, table, want_families=True)
            fam = ref["families"]
            p1 = ref["pK"][(1,)]
            E = np.zeros((n, n))
            for (out, fs, dv), val in fam.items():
                if "D22" in fs and set(fs) <= B._LOWRANK:
                    if out == (2, 1):
                        E += val
                    elif out == (1, 1):
                        E += -2.0 * p1[:, None] * val
            claim(f"{arm} n={n} removed == D22-driven K21", rel(removed, E) <= 1e-10, "%.1e" % rel(removed, E))

    print("\n5. full-rank identity, f64 layer loop on a real tiny chain")
    for wd, dp, sd in ((24, 5, 90000000803), (32, 6, 90000000804)):
        w, es, _ = canonical_mint(wd, dp, sd)
        r32d = layer_loop(m32d, w, wd, dp)
        rPd = layer_loop(dense["P"], w, wd, dp)
        rQd = layer_loop(dense["Q"], w, wd, dp)
        rQPd = layer_loop(dense["QP"], w, wd, dp)
        claim(f"{wd}x{dp} P_dense == v32_dense (all rows)", rel(rPd, r32d) <= 1e-9, "%.1e" % rel(rPd, r32d))
        claim(f"{wd}x{dp} QP_dense == Q_dense (all rows)", rel(rQPd, rQd) <= 1e-9, "%.1e" % rel(rQPd, rQd))
        claim(f"{wd}x{dp} Q_dense differs from v32_dense", rel(rQd, r32d) > 1e-8, "%.1e" % rel(rQd, r32d))

    print("\n6-7. float32 and equivariance (96x6, layer 1 relabelled -- s1_parity_v32 spelling)")
    wd, dp = 96, 6
    w, es, _ = canonical_mint(wd, dp, 90000000504)
    lay = 1
    perm = np.random.default_rng(20260916).permutation(wd)
    wp = [np.array(x, copy=True) for x in w]
    wp[lay] = wp[lay][:, perm]
    wp[lay + 1] = wp[lay + 1][perm, :]
    base_err = base_eq = None
    for name, mod in [("v32", m32)] + [(a, arms[a]) for a in ("R", "Q", "P", "QP")]:
        p, _ = P1._predict(mod, w, wd, dp, es)
        f64 = layer_loop(mod, w, wd, dp)
        err = max(rel(p[l], f64[l]) for l in range(1, dp))
        pout, _ = P1._predict(mod, wp, wd, dp, es)
        moved = p.copy()
        moved[lay] = p[lay][perm]
        eq = float(np.mean((moved - pout) ** 2)) / max(float(np.mean(p ** 2)), 1e-300)
        if name == "v32":
            base_err, base_eq = err, eq
            print(f"        v32 float32 floor {err:.2e}, equivariance floor {eq:.2e}")
            continue
        claim(f"{name} float32 worst row <= 2x v32 + 1e-5", err <= 2 * base_err + 1e-5, "%.2e" % err)
        claim(f"{name} equivariance <= 100x v32 floor", eq <= 100 * max(base_eq, 1e-300), "%.2e" % eq)

    print("\n8. cost and guard")
    for wd, dp, sd in ((96, 6, 90000000807), (128, 8, 90000000808)):
        w, es, _ = canonical_mint(wd, dp, sd)
        _, fb = P1._predict(m32, w, wd, dp, es)
        for a in ("R", "Q", "P", "QP"):
            _, fa = P1._predict(arms[a], w, wd, dp, es)
            g = arms[a]._projected_flops(wd, dp, arms[a]._fit_generations(wd, dp, 2 ** 41))
            claim(f"{wd}x{dp} {a} guard over-predicts", g >= fa, "cost x%.4f  guard/meas %.3f" % (fa / fb, g / fa))
    for a in ("R", "Q", "P", "QP"):
        claim(f"1024x16 {a} guard picks v32's gens",
              arms[a]._fit_generations(1024, 16, 2 ** 41) == m32._fit_generations(1024, 16, 2 ** 41))

    print("\n9. finite at a large-variance fixture")
    wd, dp = 64, 6
    w, es, _ = canonical_mint(wd, dp, 90000000809)
    wl = [np.asarray(x, dtype=np.float64) * 1.8 for x in w]
    for a in ("R", "Q", "P", "QP"):
        p, _ = P1._predict(arms[a], wl, wd, dp, es)
        claim(f"{a} finite, non-negative", bool(np.isfinite(p).all() and (p >= 0).all()))

    print(f"\n{sum(OK)}/{len(OK)} claims passed")
    return 0 if all(OK) else 1


if __name__ == "__main__":
    sys.exit(main())
