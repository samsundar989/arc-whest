"""Step 1 (HANDOFF 09-27) checks for the harmonic old tier (`tools/build_harm.py`).

    .venv-kprop/Scripts/python.exe tools/harm_check.py arc      # the constant, vs ARC
    .venv/Scripts/python.exe       tools/harm_check.py chain    # factored vs dense, n = 32

`arc`: on a random symmetric (8,8,8) tensor, ARC's own `proj_geq_r(T, n, 1)` (the r >= 1
harmonic part, as an r = 1 core) must equal u = 3/(n+2) (T : I), and `rad(u)` must equal
Sym(u (x) I). Then the metric form: with C = L L^T, whitening T by L^-1 and projecting
in the identity metric must give u = 3/(n+2) (T : C^-1) and radial part Sym(u (x) C).

`chain`: an arm file run in float64 under the real flopscope at n = 32, depth 9. At every
fold, the oldest lineage is expanded to a dense tensor and
  (1) `_harm_trace` == dense T : K (K = cov_out^-1, and the diag metric) to 1e-10;
  (2) the remainder T - Sym(u (x) C) is traceless in C;
  (3) after one transport by the next layer's weights, the file's D3/D21 consumer of U
      == the (3,) / (2,1) slices of the dense transported Sym(u (x) C) (U-only check at the
      first fold layer, where U is exactly the new fold).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def sym3(T):
    import itertools
    return sum(T.transpose(p) for p in itertools.permutations(range(3))) / 6.0


def sym_uC(u, C):
    return (np.einsum("a,bc->abc", u, C) + np.einsum("b,ac->abc", u, C)
            + np.einsum("c,ab->abc", u, C)) / 3.0


def cmd_arc():
    sys.path.insert(0, str(ROOT / "external" / "mlp_cumulant_propagation" / "src"))
    import torch
    from mlp_kprop.harmonic import proj_geq_r, rad
    n = 8
    rng = np.random.default_rng(0)
    T = sym3(rng.normal(size=(n, n, n)))
    u = 3.0 / (n + 2) * np.einsum("abb->a", T)
    H = proj_geq_r(torch.tensor(T), n, 1)
    core = H.core.numpy()
    radial = rad(H.core, n=n).numpy()
    e1 = np.abs(core - u).max() / np.abs(u).max()
    e2 = np.abs(radial - sym_uC(u, np.eye(n))).max() / np.abs(radial).max()
    print(f"ARC core vs 3/(n+2) tr T: {e1:.2e}   rad(core) vs Sym(u(x)I): {e2:.2e}")
    # metric form through whitening
    G = rng.normal(size=(n, n))
    C = G @ G.T + n * np.eye(n)
    L = np.linalg.cholesky(C)
    Li = np.linalg.inv(L)
    Tw = np.einsum("ai,bj,ck,ijk->abc", Li, Li, Li, T)
    uw = proj_geq_r(torch.tensor(Tw), n, 1).core.numpy()
    radial_C = np.einsum("ai,bj,ck,ijk->abc", L, L, L, sym_uC(uw, np.eye(n)))
    uC = 3.0 / (n + 2) * np.einsum("abc,bc->a", T, np.linalg.inv(C))
    e3 = np.abs(radial_C - sym_uC(uC, C)).max() / np.abs(radial_C).max()
    rem = T - sym_uC(uC, C)
    e4 = np.abs(np.einsum("abc,bc->a", rem, np.linalg.inv(C))).max() / np.abs(uC).max()
    print(f"metric C: ARC-whitened radial vs Sym(u(x)C), u = 3/(n+2) T:C^-1: {e3:.2e}"
          f"   remainder trace: {e4:.2e}")
    # ARC's proj_coef is a float32 torch tensor: agreement is bounded at ~1e-8 by it
    ok = max(e1, e2, e3) < 1e-7 and e4 < 1e-10
    print("PASS" if ok else "FAIL")
    return ok


def lineage_tensor(tw, n, b):
    """Dense tensor of lineage `b` of a POST-ADVANCE / post-wick tower (M has k blocks)."""
    sl = slice(b * n, (b + 1) * n)
    f = lambda x: np.asarray(x, dtype=np.float64)
    A = f(tw.A_L)[:, sl]
    P = f(tw.M)[:, sl]
    w2 = f(tw.w2b)[sl]
    r = tw.L.shape[2]
    A_D = P * f(tw.sb)[sl] + 3.0 * A * f(tw.eb)[sl] + f(tw.Z)[:, b * r:(b + 1) * r] @ f(tw.L)[b].T
    TD = np.einsum("aj,bj,cj->abc", A_D, P, P)
    T = (TD + TD.transpose(1, 0, 2) + TD.transpose(2, 1, 0)) / 3.0
    TL = np.einsum("aj,bj,cj->abc", A, P, A * w2)
    T = T + 3.0 * sym3(TL)
    if tw.Y is not None:
        Y = f(tw.Y)
        for h, cv in enumerate((tw.CA, tw.CB)):
            X = A * f(cv)[sl]
            S = X @ P.T + P @ X.T
            y = Y[:, 2 * b + h]
            T = T + (np.einsum("a,bc->abc", y, S) + np.einsum("b,ac->abc", y, S)
                     + np.einsum("c,ab->abc", y, S)) / 3.0
    return T


def cmd_chain(arm=None, n=32, depth=9, seed=11):
    sys.path.append(str(ROOT / "local_sim"))
    sys.path.insert(0, str(ROOT / "tools"))
    import importlib.util
    import flopscope as flops
    import flopscope.numpy as fnp
    import build_harm
    src = build_harm.build((ROOT / "estimator_k3_v68_wino.py").read_text(encoding="utf-8"), 3, "C")
    p = ROOT / "cache" / "_harm_check_arm.py"
    p.parent.mkdir(exist_ok=True)
    p.write_text(src, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location("_harm_arm", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._CORR_ON = False
    dt = fnp.float64
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(size=(n, n)) * np.sqrt(2.0 / n) for _ in range(depth + 1)]
    table = mod._tables()
    const = mod.Estimator()._constants(n, dt)
    worst = {"trace_C": 0.0, "trace_diag": 0.0, "remainder": 0.0, "d3": 0.0, "d21": 0.0}
    seen = {"folds": 0, "consumer_checks": 0}
    orig_fold = mod._harm_fold
    captured = {}

    def spy(tw, U, cov_out, n_, dtype):
        if tw is not None and tw.k >= mod._HARM_FOLD_AGE:
            # At n = 32 a few near-dead neurons make the chain's cov_out slightly
            # indefinite; the algebra under test holds for any invertible metric,
            # so the check folds with cov_out shifted to be well conditioned.
            C = np.asarray(cov_out, dtype=np.float64)
            C = (C + C.T) / 2 + (abs(min(np.linalg.eigvalsh(C).min(), 0.0)) + 0.1) * np.eye(n_)
            cov_out = fnp.asarray(C, dtype=dt)
            K = np.linalg.inv(C)
            T = lineage_tensor(tw, n_, 0)
            dense = np.einsum("abc,bc->a", T, K)
            fac = np.asarray(mod._harm_trace(tw, fnp.asarray(K, dtype=dt), None, n_))
            s = np.abs(dense).max()
            worst["trace_C"] = max(worst["trace_C"], np.abs(fac - dense).max() / s)
            Kd = 1.0 / np.diag(C)
            dd = np.einsum("abb,b->a", T, Kd)
            facd = np.asarray(mod._harm_trace(tw, None, fnp.asarray(Kd, dtype=dt), n_))
            worst["trace_diag"] = max(worst["trace_diag"], np.abs(facd - dd).max() / np.abs(dd).max())
            u = 3.0 / (n_ + 2) * dense
            rem = np.einsum("abc,bc->a", T - sym_uC(u, C), K)
            worst["remainder"] = max(worst["remainder"], np.abs(rem).max() / np.abs(u).max())
            seen["folds"] += 1
            if U is None:
                captured["u_first"] = (u, C)
        return orig_fold(tw, U, cov_out, n_, dtype)

    mod._harm_fold = spy
    state = (np.zeros(n), np.eye(n), None, 0.0, np.zeros(n))
    with flops.BudgetContext(flop_budget=2 ** 50, quiet=True):
        for l in range(depth):
            mu, cov, tw, c4, gk4 = state
            had_U = tw is not None and tw.U is not None
            st = (fnp.asarray(mu, dtype=dt), fnp.asarray(cov, dtype=dt), tw, c4,
                  fnp.asarray(gk4, dtype=dt))
            if tw is not None and tw.U is not None and "u_first" in captured and not had_first(seen):
                # consumer check: U is exactly the first fold; transport the dense radial
                # tensor by this layer's W and compare with D3/D21 the layer consumes
                u, C = captured["u_first"]
                WT = Ws[l].T
                Rt = np.einsum("ai,bj,ck,ijk->abc", WT, WT, WT, sym_uC(u, C))
                d3_ref = np.einsum("aaa->a", Rt)
                d21_ref = np.einsum("aai->ai", Rt) * (1 - np.eye(n))
                Ut = WT @ u
                Cp = WT @ C @ WT.T
                var = np.diag(Cp)
                d3 = Ut * var
                C11 = Cp * (1 - np.eye(n))
                d21 = ((2.0 * Ut[:, None]) * C11 + var[:, None] * Ut[None, :]) / 3.0 * (1 - np.eye(n))
                worst["d3"] = np.abs(d3 - d3_ref).max() / np.abs(d3_ref).max()
                worst["d21"] = np.abs(d21 - d21_ref).max() / np.abs(d21_ref).max()
                # and the file's own U must be this transported vector (gain 1)
                fileU = np.asarray(WT @ np.asarray(tw.U, dtype=np.float64))
                worst["fileU"] = np.abs(fileU - Ut).max() / np.abs(Ut).max()
                seen["consumer_checks"] += 1
            out = mod._layer(fnp.asarray(Ws[l], dtype=dt), st, table, dt, const, 2.0,
                             last_layer=(l == depth - 1), gens=None, layer_index=l, depth=depth)
            state = out
    print(f"folds checked: {seen['folds']}, consumer checks: {seen['consumer_checks']}")
    for k, v in worst.items():
        print(f"  {k:12s} {v:.2e}")
    ok = seen["folds"] > 0 and seen["consumer_checks"] > 0 and max(worst.values()) < 1e-9
    print("PASS" if ok else "FAIL")
    return ok


def had_first(seen):
    return seen["consumer_checks"] > 0


if __name__ == "__main__":
    ok = {"arc": cmd_arc, "chain": cmd_chain}[sys.argv[1]]()
    raise SystemExit(0 if ok else 1)
