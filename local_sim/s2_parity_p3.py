"""Fast gates for the P3 arms (`s2_build_p3.py`), before any truth read.

  1. NESTING: the generated file with `_P3_ON = False` is its base (v32 or v31)
     bit for bit in output and counted FLOPs.
  2. BIRTH TENSOR (f64 dense chain, identical incoming state): newborn(P3 on) -
     newborn(P3 off) equals the independently formed beta1 + beta2 routes on
     all-distinct entries and 0 on repeated entries (compensated), <= 1e-9 rel.
  3. SLICES: dense chain == factored chain (<= 1e-10), which checks the folded
     LA/LP terms and the thin contraction against a dense L-block transport.
  4. FLOAT32 vs FLOAT64 (<= 2x base + 1e-5), EQUIVARIANCE (<= 100x base),
     GUARD over-predicts, same `gens` at 1024x16, cost ratio printed.
"""

from __future__ import annotations

import itertools
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

import dense_tower as DT  # noqa: E402
import k4feed_oracle as K  # noqa: E402
import s1_parity_v32 as P1  # noqa: E402
import s2_build_p3 as B  # noqa: E402

OK = []


def claim(name, ok, detail=""):
    OK.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<62} {detail}")


def beta_routes(D21, C_off, w1, w2, w3):
    """beta1 + beta2 as full tensors; only all-distinct entries are meaningful."""
    n = D21.shape[0]
    # beta1: 1/2 D21_ds C_dt w3_d w1_s w1_t over role perms; beta2: 1/2 D21_ds C_st w2_d w2_s w1_t
    Q1 = 0.5 * np.einsum("ds,dt,d,s,t->dst", D21, C_off, w3, w1, w1)
    Q2 = 0.5 * np.einsum("ds,st,d,s,t->dst", D21, C_off, w2, w2, w1)
    Q = Q1 + Q2
    return sum(Q.transpose(pm) for pm in itertools.permutations(range(3)))


def main(base_arm="estimator_k3_v32.py", p3_arm="estimator_k3_v33.py",
         p3_dense="estimator_k3_v33_dense.py"):
    import flopscope as flops
    import flopscope.numpy as fnp
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    args = B.ARMS[p3_arm]
    tmp = tempfile.mkdtemp(prefix="s2p3_")
    off_path = os.path.join(tmp, "p3_off.py")
    with open(off_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(B.build(*args).replace("_P3_ON = True", "_P3_ON = False", 1))
    mbase = K._load(os.path.join(_ROOT, base_arm))
    moff = K._load(off_path)
    mp3 = K._load(os.path.join(_ROOT, p3_arm))

    print("1. nesting")
    for (wd, dp, sd) in ((64, 5, 90000000601), (96, 6, 90000000602)):
        w, es, _ = canonical_mint(wd, dp, sd)
        pb, fb = P1._predict(mbase, w, wd, dp, es)
        po, fo = P1._predict(moff, w, wd, dp, es)
        pp, fp = P1._predict(mp3, w, wd, dp, es)
        s = max(np.abs(pb).max(), 1e-300)
        claim(f"{wd}x{dp}: P3-off == base bitwise and same FLOPs",
              np.array_equal(pb, po) and fb == fo, f"{fo:.0f} vs {fb:.0f}")
        claim(f"{wd}x{dp}: P3 is not a no-op", np.abs(pp - pb).max() / s > 1e-7,
              f"rel {np.abs(pp - pb).max() / s:.2e}  cost x{fp / fb:.5f}")

    print("\n2. birth tensor, f64 dense chain n=8 depth=6")
    n, depth = 8, 6
    rng = np.random.default_rng(29)
    Ws = [rng.normal(size=(n, n)) * np.sqrt(2.0 / n) for _ in range(depth + 1)]
    dmod, _ = DT.load_dense(os.path.join(_ROOT, p3_dense))
    worst, size = 0.0, 0.0
    caps = []
    DT.run_chain(dmod, Ws[:depth], lam=[0.15] * 16,
                 capture=lambda l, si, so: caps.append((l, si, so)))
    for l, si, so in caps:
        if l == 0 or l == depth - 1:
            continue
        mu, cov, tw, c4, gk4 = si
        # BOTH arms re-run here under the SAME patched lambda table: the capture
        # above ran inside `run_chain`, which restores the table on exit, so
        # differencing against it compared two different kappa4 channels (the
        # first version of this check did exactly that and failed at 2.7e-2).

        def _run(on):
            saved = (dmod._P3_ON, list(dmod._K4_LAM))
            dmod._P3_ON, dmod._K4_LAM = on, [0.15] * 16
            try:
                with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
                    return dmod._layer(fnp.asarray(Ws[l], dtype=fnp.float64),
                                       (fnp.asarray(mu, dtype=fnp.float64),
                                        fnp.asarray(cov, dtype=fnp.float64), tw, c4,
                                        fnp.asarray(gk4, dtype=fnp.float64)),
                                       dmod._tables(), fnp.float64,
                                       dmod.Estimator()._constants(n, fnp.float64), 2.0,
                                       last_layer=False, gens=None, layer_index=l)
            finally:
                dmod._P3_ON, dmod._K4_LAM = saved
        so = (None, None, _run(True)[2])
        out_off = _run(False)
        diff = so[2].T - out_off[2].T
        W = Ws[l]
        cov_pre = W.T @ cov @ W
        var = np.maximum(np.diag(cov_pre), 1e-12)
        sig = np.sqrt(var)
        al = np.clip((W.T @ mu) / sig, -40.0, 40.0)
        ph = K._phi(al)
        w1, w2, w3 = ndtr(al), ph / sig, -al * ph / var
        C_off = cov_pre * (1.0 - np.eye(n))
        Tin = tw.T
        Tadv = np.einsum("ai,bj,ck,ijk->abc", W.T, W.T, W.T, Tin)
        D21 = np.einsum("aab->ab", Tadv) * (1.0 - np.eye(n))
        expect = K._compensated(beta_routes(D21, C_off, w1, w2, w3))
        sc = max(np.abs(so[2].T).max(), 1e-300)
        worst = max(worst, np.abs(diff - expect).max() / sc)
        size = max(size, np.abs(expect).max() / sc)
    claim("2. newborn difference == beta1+beta2 (distinct), 0 (repeated)", worst <= 1e-9,
          f"worst rel {worst:.2e}  route size {size:.2e}")
    claim("2. routes not vacuous", size > 1e-5, f"{size:.2e}")

    print("\n3. dense chain == factored chain")
    wv = DT.self_check(os.path.join(_ROOT, p3_dense), n=10, depth=6, seed=5)
    claim("3. dense == factored (hubs + P3 blocks)", wv <= 1e-10, f"{wv:.2e}")

    print("\n4. float32, equivariance, guard, cost")
    wd, dp = 64, 5
    w, es, _ = canonical_mint(wd, dp, 90000000603)
    errs = {}
    for name, mod in (("base", mbase), ("p3", mp3)):
        p32, _ = P1._predict(mod, w, wd, dp, es)
        st = (np.zeros(wd), np.eye(wd), None, 0.0, np.zeros(wd))
        rows = []
        for l in range(dp):
            dt = fnp.float64
            with flops.BudgetContext(flop_budget=2 ** 55, quiet=True):
                o = mod._layer(fnp.asarray(np.asarray(w[l], dtype=np.float64), dtype=dt),
                               (fnp.asarray(st[0], dtype=dt), fnp.asarray(st[1], dtype=dt), st[2],
                                st[3], fnp.asarray(st[4], dtype=dt)),
                               mod._tables(), dt, mod.Estimator()._constants(wd, dt), 2.0,
                               last_layer=(l == dp - 1), gens=None, layer_index=l)
            rows.append(np.asarray(o[0], dtype=np.float64))
            if l < dp - 1:
                st = (rows[-1], np.asarray(o[1], dtype=np.float64), o[2], o[3],
                      np.asarray(o[4], dtype=np.float64))
        f64 = np.maximum(np.stack(rows), 0.0)
        errs[name] = max(np.abs(p32[l] - f64[l]).max() / max(np.abs(f64[l]).max(), 1e-300)
                         for l in range(1, dp))
    claim("4. float32 error <= 2x base + 1e-5", errs["p3"] <= 2 * errs["base"] + 1e-5,
          f"base {errs['base']:.2e}  p3 {errs['p3']:.2e}")
    wd, dp = 96, 6
    rng = np.random.default_rng(20260917)
    w, es, _ = canonical_mint(wd, dp, 90000000604)
    for lay in (1, dp - 2):
        perm = rng.permutation(wd)
        wp = [np.array(x, copy=True) for x in w]
        wp[lay] = wp[lay][:, perm]
        wp[lay + 1] = wp[lay + 1][perm, :]
        err = {}
        for name, mod in (("base", mbase), ("p3", mp3)):
            b0, _ = P1._predict(mod, w, wd, dp, es)
            b1, _ = P1._predict(mod, wp, wd, dp, es)
            mv = b0.copy()
            mv[lay] = b0[lay][perm]
            err[name] = float(np.mean((mv - b1) ** 2)) / max(float(np.mean(b0 ** 2)), 1e-300)
        ratio = err["p3"] / max(err["base"], 1e-300)
        claim(f"4. equivariance layer {lay} <= 100x base", ratio <= 100.0,
              f"base {err['base']:.2e}  p3 {err['p3']:.2e}  x{ratio:.1f}")
    for cw, cd in ((96, 6), (128, 8)):
        w, es, _ = canonical_mint(cw, cd, 90000000605)
        _, fb = P1._predict(mbase, w, cw, cd, es)
        _, fp = P1._predict(mp3, w, cw, cd, es)
        model = mp3._projected_flops(cw, cd, mp3._fit_generations(cw, cd, 2 ** 41))
        claim(f"4. {cw}x{cd} guard over-predicts", model >= fp, f"model/measured {model / fp:.4f}  "
              f"p3/base x{fp / fb:.5f}")
    gb = mbase._fit_generations(1024, 16, 2 ** 41)
    gp = mp3._fit_generations(1024, 16, 2 ** 41)
    claim("4. 1024x16 same gens", gb == gp, f"{gb}/{gp} projected x"
          f"{mp3._projected_flops(1024, 16, gp) / mbase._projected_flops(1024, 16, gb):.5f}")
    print(f"\n{sum(OK)}/{len(OK)} claims pass")
    return 0 if all(OK) else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    raise SystemExit(main(*a) if a else main())
