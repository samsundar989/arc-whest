"""P2's reconstruction identity, checked exactly before any estimator is touched.

Slice 6b, mechanism **P2**. `experiments/2026-09-14_slice6a_p1/p2_design_note.md`
has the derivation and the price; this file is the algebra, on real chain state,
at full rank where the identity must be EXACT rather than approximate.

THE IDENTITY
------------
F21 births its dense leg as

    A_D = diag(S3) + 3 * S21.T                  (estimator_k3_strassen_L4_f21.py:944)

P2 splits `S21` into a separable part and a residual:

    e     = w(1,2) - 2 * pK1 * w(1,1)
    S_sep = zero_diag( e_i * C_off[i,j] * w1_j )
    R     = S21 - S_sep

and because `C_off` is symmetric and `a_b = fac1 = w1[:,None] * C11` is a leg
F21 ALREADY carries,

    (3 * S_sep).T[i,j] = 3 * e_j * C_off[j,i] * w1_i = 3 * fac1[i,j] * e_j

so the whole separable half is a **column scale on an existing leg**:

    A_D = diag(S3) + 3 * a_b * d(e) + 3 * R.T

WHAT IS EXACT AND WHAT IS NOT — the distinction the whole mechanism rests on
---------------------------------------------------------------------------
The rewrite above is **exact**. It is a regrouping of terms and introduces no
error at any rank. What is approximate is the *later* step of carrying `R` at
rank `r` instead of densely, and that approximation must never be conflated
with the identity. This file tests only the exact part, at full rank, where a
failure means the algebra is wrong rather than the rank being small.

The rank's cost is separately measured: `p2_split_n128.json` gives the
Frobenius picture, and the oracle's own end-to-end figures are +1.6% / +0.9% /
+0.4% raw MSE at rank 16 / 32 / 64.

Run: uv run python local_sim/p2_p2_identity_test.py
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

SOURCE = os.path.join(_ROOT, "estimator_k3_strassen_L4_f21.py")

RESULTS = []

_HOOK = '''    _P2_CAP.append({
        "S21": np.asarray(S21, dtype=np.float64),
        "S3": np.asarray(S3, dtype=np.float64),
        "A_D": np.asarray(fnp.diag(S3) + 3.0 * S21.T, dtype=np.float64),
        "C11": np.asarray(C11, dtype=np.float64),
        "fac1": np.asarray(fac1, dtype=np.float64),
        "w11": np.asarray(w[(1, 1)], dtype=np.float64),
        "w12": np.asarray(w[(1, 2)], dtype=np.float64),
        "pK1": np.asarray(pK[(1,)], dtype=np.float64),
    })
'''


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<64} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def _instrumented():
    with open(SOURCE, "rb") as fh:
        raw = fh.read()
    src = raw.decode("utf-8")
    anchor = "    out_tw = _append(adv, fnp.diag(S3) + 3.0 * S21.T, fac1, w2b_vec, n)"
    if src.count(anchor) != 1:
        raise SystemExit("F21 has moved: the birth anchor is not unique")
    src = src.replace(anchor, _HOOK + anchor, 1)
    src = src.replace("import flopscope as flops",
                      "import numpy as np\nimport flopscope as flops", 1)
    src = src.replace("_TABLES = None", "_P2_CAP = []\n_TABLES = None", 1)
    return src, hashlib.sha256(raw).hexdigest()


def _capture(width=64, depth=6, seed=40901003000):
    import importlib.util

    import flopscope as flops
    import harness
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    src, sha = _instrumented()
    d = tempfile.mkdtemp(prefix="p2ident_")
    path = os.path.join(d, "f21_probe.py")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("f21_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    est = mod.Estimator()

    class _Ctx:
        pass
    _Ctx.width, _Ctx.depth = width, depth
    _Ctx.flop_budget, _Ctx.seed, _Ctx.n_mlps = 2 ** 41, 0, 1
    est.setup(_Ctx())
    w, es, _ = canonical_mint(width, depth, seed)
    mlp = harness.MLP(weights=[np.asarray(x) for x in w], width=width,
                      depth=depth, seed=es)
    with flops.BudgetContext(flop_budget=2 ** 41, quiet=True):
        est.predict(mlp, 2 ** 41)
    return mod._P2_CAP, sha


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b))) / s


def t_identity_is_exact():
    head("1. the rewrite is EXACT at full rank, on real chain state")
    caps, sha = _capture()
    check("F21 state captured at its own birth site", len(caps) > 0,
          f"{len(caps)} layers, F21 sha {sha[:16]}...")
    worst_sep, worst_full = 0.0, 0.0
    for l, c in enumerate(caps):
        S21, C11, fac1 = c["S21"], c["C11"], c["fac1"]
        w11, w12, pK1, S3 = c["w11"], c["w12"], c["pK1"], c["S3"]
        e = w12 - 2.0 * pK1 * w11

        # (a) the separable transpose really IS a column scale on `a_b`
        S_sep = e[:, None] * C11 * w11[None, :]
        lhs = 3.0 * S_sep.T
        rhs = 3.0 * fac1 * e[None, :]
        worst_sep = max(worst_sep, _rel(lhs, rhs))

        # (b) the FULL reconstruction, residual carried densely
        R = S21 - S_sep
        rebuilt = np.diag(S3) + 3.0 * fac1 * e[None, :] + 3.0 * R.T
        worst_full = max(worst_full, _rel(rebuilt, c["A_D"]))
    # THE TOLERANCE IS IN FLOAT32 ULPS, AND THE REASON IS NOT A CONVENIENCE.
    # F21 computes the captured state in float32, so a rewrite that is exact in
    # real arithmetic still differs by round-off once it is regrouped -- F21's
    # own `_sb` docstring calls five float32 ulps "reassociation" for exactly
    # this reason. The first version of this test demanded 1e-12 against
    # float32 state and duly reported a failure at 3.7 ulps, which would have
    # read as an algebra error. `t_identity_in_float64` below is what actually
    # proves the algebra; this check proves the IMPLEMENTATION agrees with it
    # to the precision the implementation has.
    eps32 = float(np.finfo(np.float32).eps)
    check("3*S_sep.T == 3*a_b*d(e), every layer",
          worst_sep < 8 * eps32,
          f"worst {worst_sep:.3e} = {worst_sep / eps32:.2f} float32 ulps")
    check("diag(S3) + 3*a_b*d(e) + 3*R.T == A_D, every layer",
          worst_full < 8 * eps32,
          f"worst {worst_full:.3e} = {worst_full / eps32:.2f} float32 ulps")
    return caps


def t_identity_in_float64():
    """The algebra itself, in a precision where round-off cannot hide it.

    Synthetic float64 operands with the same shapes and symmetries as the real
    ones. If this passes at 1e-14 and the float32 check passes at a few ulps,
    the rewrite is exact and the implementation is faithful to it. If THIS
    failed, no tolerance anywhere would make the mechanism sound.
    """
    head("1b. the same identity in float64, where round-off cannot hide it")
    rng = np.random.default_rng(20260915)
    worst_sep = worst_full = 0.0
    for n in (7, 33, 64):
        C = rng.normal(size=(n, n))
        C11 = (C + C.T) * 0.5
        np.fill_diagonal(C11, 0.0)            # C_off is symmetric, zero diagonal
        w11 = rng.normal(size=n)
        e = rng.normal(size=n)
        S3 = rng.normal(size=n)
        fac1 = w11[:, None] * C11             # F21's `a_b`, by construction
        S_sep = e[:, None] * C11 * w11[None, :]
        R = rng.normal(size=(n, n))           # an ARBITRARY residual
        S21 = S_sep + R
        A_D = np.diag(S3) + 3.0 * S21.T
        worst_sep = max(worst_sep, _rel(3.0 * S_sep.T, 3.0 * fac1 * e[None, :]))
        worst_full = max(worst_full,
                         _rel(np.diag(S3) + 3.0 * fac1 * e[None, :] + 3.0 * R.T,
                              A_D))
    check("3*S_sep.T == 3*a_b*d(e) exactly, widths 7/33/64",
          worst_sep < 1e-14, f"worst rel {worst_sep:.3e}")
    check("the full reconstruction is exact for an ARBITRARY residual",
          worst_full < 1e-14,
          f"worst rel {worst_full:.3e} -- it is a regrouping, not a fit")
    check("...so the rewrite introduces NO error at full rank",
          worst_sep < 1e-14 and worst_full < 1e-14,
          "the approximation is the RANK, and it is a separate step")


def t_the_rank_is_where_error_enters(caps):
    head("2. and the rank is where error enters — separately, and measurably")
    rows = []
    for l, c in enumerate(caps):
        S21, C11 = c["S21"], c["C11"]
        w11, w12, pK1, S3, fac1 = (c["w11"], c["w12"], c["pK1"], c["S3"],
                                   c["fac1"])
        e = w12 - 2.0 * pK1 * w11
        R = S21 - (e[:, None] * C11 * w11[None, :])
        U, sv, Vt = np.linalg.svd(R, full_matrices=False)
        for r in (8, 16, 32):
            rr = min(r, sv.size)
            Rr = (U[:, :rr] * sv[:rr]) @ Vt[:rr]
            rebuilt = np.diag(S3) + 3.0 * fac1 * e[None, :] + 3.0 * Rr.T
            rows.append((l, r, _rel(rebuilt, c["A_D"])))
    n = caps[0]["S21"].shape[0]
    print(f"  reconstruction error of A_D at rank r, n = {n}")
    print(f"  {'layer':>5} {'r=8':>12} {'r=16':>12} {'r=32':>12}")
    for l in sorted({x[0] for x in rows}):
        e8 = [x[2] for x in rows if x[0] == l and x[1] == 8][0]
        e16 = [x[2] for x in rows if x[0] == l and x[1] == 16][0]
        e32 = [x[2] for x in rows if x[0] == l and x[1] == 32][0]
        print(f"  {l:>5} {e8:>12.3e} {e16:>12.3e} {e32:>12.3e}")
    mono = all(
        [x[2] for x in rows if x[0] == l and x[1] == 8][0]
        >= [x[2] for x in rows if x[0] == l and x[1] == 16][0]
        >= [x[2] for x in rows if x[0] == l and x[1] == 32][0]
        for l in sorted({x[0] for x in rows}))
    check("more rank is monotonically less error, every layer", mono, "")
    check("rank 32 is not yet exact -- it is an approximation and says so",
          max(x[2] for x in rows if x[1] == 32) > 1e-12,
          "an 'exact' low-rank reconstruction would mean the residual was "
          "already thin, which would make P2 free rather than a trade")
    print()
    print("  NOTE the units. This is reconstruction error of A_D in Frobenius")
    print("  norm, NOT score. The oracle's own end-to-end figures are +1.6% /")
    print("  +0.9% / +0.4% raw MSE at rank 16 / 32 / 64, an order of magnitude")
    print("  smaller than the Frobenius picture, because what the chain")
    print("  consumes is a contraction. Only a score sets the rank.")


def main():
    print("=" * 96)
    print("P2's RECONSTRUCTION IDENTITY — exact at full rank, approximate by rank")
    print("=" * 96)
    caps = t_identity_is_exact()
    t_identity_in_float64()
    if caps:
        t_the_rank_is_where_error_enters(caps)
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
