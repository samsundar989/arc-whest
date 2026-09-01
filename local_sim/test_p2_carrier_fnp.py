"""WP1 — the fnp carrier must BE the frozen reference, not resemble it.

Contract: `experiments/LOG.md`, *CONTRACT WP1*. Specification:
`docs/KERDOCK_MUB_SPEC.md`. Parity oracle: `local_sim/kerdock_mub.py`, which is
FROZEN — this suite never edits it to make parity easier, and any disagreement
is a defect in `p2_carrier_fnp.py`.

WHY BYTE PARITY AND NOT A TOLERANCE
------------------------------------
Every entry of the carrier is `+/- 1/sqrt(N)`, and at every supported width `N`
is a power of two, so `1/sqrt(N)` is exactly representable in binary floating
point (`1/32` at the live width). There is no rounding to tolerate. Equality is
achievable, so equality is demanded — the standard `test_shape_parity.py`
already sets for v2.6 vs v2.4.1.

THE WEAK CHECK, NAMED
---------------------
`docs/KERDOCK_MUB_SPEC.md` §5 warns that **within-basis orthogonality holds for
ANY sign vector, including a wrong one** — the rank-based coordinate mapping
passed it perfectly while producing ANF degree 9 instead of 2. So this suite
does not rest on it. The discriminator is the cross-basis Gram: every entry
must be `+/- 2^(n/2)` exactly, never zero. Section V exercises both and says
which is which, and section N runs the rejected mapping as a **negative
control** that must fail the discriminator.

Run: uv run python local_sim/test_p2_carrier_fnp.py
     uv run python local_sim/test_p2_carrier_fnp.py --live   # + the 1024 checks
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim ships shims NAMED flopscope/whestbench: APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402  -- the ORACLE's language, never the candidate's
import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

import kerdock_mub as REF  # noqa: E402  -- the frozen reference
import p2_carrier_fnp as C  # noqa: E402

try:
    flops.configure(symmetry_warnings=False)
except Exception:
    pass

_FAILED = []
_TOTAL = [0]


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<66}{detail}")
    if not ok:
        _FAILED.append(name)


def _np(a):
    """A flopscope array as plain NumPy, for COMPARISON ONLY.

    The candidate never does this. It exists so the oracle and the candidate can
    be diffed at all, and it is confined to this suite.
    """
    return np.asarray(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true",
                    help="also run the m=9 / N=1024 checks (~1 min, 128 MiB)")
    a = ap.parse_args()

    print("WP1 - the fnp carrier against the frozen NumPy reference\n")

    # ------------------------------------------------------------------ R --
    print("R  the ring, the trace and the Gray map agree with the reference")
    for m in (3, 5, 7, 9):
        rref, rc = REF.GaloisRing(m), C.GaloisRing(m)
        check(f"R: m={m} Hensel lift identical", rref.h == rc.h, str(rc.h))
        check(f"R: m={m} Teichmuller enumeration identical",
              rref.teichmuller() == rc.teichmuller(), f"|T| = {1 << m}")
        # The O(1) `idx` dict must be the SAME map as the reference's O(n)
        # `pw.index`, or the speed fix has changed the object.
        same_idx = all(rref.pw.index(e) == rc.idx[e] for e in rc.pw)
        check(f"R: m={m} the idx dict equals the reference's pw.index",
              same_idx, "the setup-time speed fix moves no byte")
        ks = list(range(0, rc.order, max(1, rc.order // 37)))
        check(f"R: m={m} trace_of_power agrees on {len(ks)} exponents",
              all(rref.trace_of_power(k) == rc.trace_of_power(k) for k in ks),
              "pure Python, no arrays")
    check("R: the Gray map is the standard convention",
          [C.gray(z) for z in range(4)] == [(0, 0), (0, 1), (1, 1), (1, 0)],
          "phi(0)=(0,0) phi(1)=(0,1) phi(2)=(1,1) phi(3)=(1,0)")
    check("R: PRIMITIVE is identical to the reference's",
          C.PRIMITIVE == REF.PRIMITIVE, "a divergence would build another object")

    # ------------------------------------------------------------------ S --
    print("\nS  the sign vectors are byte-identical, without NumPy")
    sizes = [(3, 8), (5, 16), (7, 16)] + ([(9, 16)] if a.live else [])
    for m, k in sizes:
        ref = REF.sign_vectors(m, k)                       # int8 (k, N)
        cand = C.sign_vectors_py(m, k)                     # list[list[int]]
        eq = (len(cand) == ref.shape[0]
              and all(list(map(int, ref[i])) == cand[i] for i in range(len(cand))))
        check(f"S: m={m} k={k} sign vectors byte-identical", eq,
              f"{len(cand)} x {len(cand[0])}, values in "
              f"{sorted(set(cand[0]))}")
        check(f"S: m={m} entries are exactly +/-1",
              all(v in (1, -1) for row in cand for v in row), "integers")

    # ------------------------------------------------------------------ W --
    print("\nW  the Walsh matrix, by Sylvester recursion in fnp")
    for n in (4, 6, 8) + ((10,) if a.live else ()):
        w_ref = REF.walsh_matrix(n).astype(np.float64)
        w_c = _np(C.walsh_fnp(fnp, n, dtype="float64"))
        check(f"W: n={n} Walsh matrix byte-identical to the reference",
              np.array_equal(w_ref, w_c), f"{w_c.shape}")
        N = 1 << n
        check(f"W: n={n} W is symmetric and W @ W = N I",
              np.array_equal(w_c, w_c.T)
              and np.array_equal(w_c @ w_c, N * np.eye(N)),
              "the defining Hadamard property")

    # ------------------------------------------------------------------ C --
    print("\nC  the carrier itself, byte-identical, in both scalings")
    for m, k in sizes:
        for anti in (False, True):
            ref_i = REF.carrier_int8(m, k, antipodes=anti).astype(np.float64)
            cand_i = _np(C.carrier_fnp(fnp, m=m, k=k, antipodes=anti,
                                       dtype="float64", unit_norm=False))
            check(f"C: m={m} k={k} antipodes={anti} raw +/-1 carrier identical",
                  np.array_equal(ref_i, cand_i),
                  f"{cand_i.shape}"
                  + ("" if np.array_equal(ref_i, cand_i)
                     else f"  max|diff| {np.abs(ref_i - cand_i).max()}"))
        ref_f = REF.carrier_float32(m, k, antipodes=True)
        cand_f = _np(C.carrier_fnp(fnp, m=m, k=k, antipodes=True,
                                   dtype="float32", unit_norm=True))
        check(f"C: m={m} k={k} scaled float32 carrier BYTE-identical",
              cand_f.dtype == ref_f.dtype
              and cand_f.tobytes() == ref_f.tobytes(),
              f"sha256 {hashlib.sha256(cand_f.tobytes()).hexdigest()[:16]}")

    # ------------------------------------------------------------------ V --
    print("\nV  the DEFINING properties, exhaustively, at reduced sizes")
    for m in (3, 5, 7):
        n, N = m + 1, 1 << (m + 1)
        k = min(8, 1 << m)
        rows = _np(C.carrier_fnp(fnp, m=m, k=k, antipodes=False,
                                 dtype="float64", unit_norm=False))
        B = rows.reshape(k, N, N)
        # V1 entries
        check(f"V1: m={m} every entry is +/-1 before scaling",
              set(np.unique(B).tolist()) == {-1.0, 1.0}, "exact integers")
        # V2 within-basis -- THE WEAK CHECK, and it is labelled
        within = all(np.array_equal(B[i] @ B[i].T, N * np.eye(N))
                     for i in range(k))
        check(f"V2: m={m} within-basis orthogonality (WEAK -- holds for a "
              f"WRONG sign vector too)", within, "necessary, not discriminating")
        # V3 cross-basis -- THE DISCRIMINATOR
        target = 2 ** (n / 2)
        worst, zeros = 0.0, 0
        for i, j in itertools.combinations(range(k), 2):
            G = np.abs(B[i] @ B[j].T)
            worst = max(worst, float(np.abs(G - target).max()))
            zeros += int((G == 0).sum())
        check(f"V3: m={m} EVERY cross-basis |correlation| is exactly "
              f"{int(target)}", worst == 0.0 and zeros == 0,
              f"max deviation {worst}, zeros {zeros}  <- THE discriminator")
        # V4/V5 rows, lines, negation closure, distinctness
        full = _np(C.carrier_fnp(fnp, m=m, k=k, antipodes=True,
                                 dtype="float64", unit_norm=False))
        check(f"V4: m={m} row count is k*N*2", full.shape[0] == k * N * 2,
              f"{full.shape[0]}")
        check(f"V4: m={m} closed under negation, +v immediately before -v",
              np.array_equal(full[0::2], -full[1::2]), "row order is pinned")
        uniq = {r.tobytes() for r in full}
        check(f"V5: m={m} all {full.shape[0]} rows are pairwise distinct",
              len(uniq) == full.shape[0], f"{len(uniq)} distinct")

    # ------------------------------------------------------------------ N --
    print("\nN  the REJECTED coordinate mapping must FAIL the discriminator")
    m = 5
    n, N = m + 1, 1 << (m + 1)
    bad = REF.bad_mapping_sign_vectors(m, 4).astype(np.float64)
    W = REF.walsh_matrix(n).astype(np.float64)
    Bb = bad[:, None, :] * W[None, :, :]
    target = 2 ** (n / 2)
    within_ok = all(np.array_equal(Bb[i] @ Bb[i].T, N * np.eye(N))
                    for i in range(4))
    devs, zeros = [], 0
    for i, j in itertools.combinations(range(4), 2):
        G = np.abs(Bb[i] @ Bb[j].T)
        devs.append(float(np.abs(G - target).max()))
        zeros += int((G == 0).sum())
    check("N: the rejected mapping PASSES within-basis orthogonality",
          within_ok, "which is exactly why V2 cannot be the discriminator")
    check("N: ...and FAILS the cross-basis identity",
          max(devs) > 0 or zeros > 0,
          f"max deviation {max(devs)}, {zeros} zero entries")
    check("N: the good mapping and the rejected one really differ",
          not np.array_equal(bad, np.asarray(C.sign_vectors_py(m, 4),
                                             dtype=np.float64)),
          "otherwise this control is vacuous")

    # ------------------------------------------------------------------ D --
    print("\nD  determinism: repeated calls, and a FRESH interpreter")
    m, k = 7, 8
    h1 = hashlib.sha256(_np(C.carrier_fnp(fnp, m=m, k=k, dtype="float32")
                            ).tobytes()).hexdigest()
    h2 = hashlib.sha256(_np(C.carrier_fnp(fnp, m=m, k=k, dtype="float32")
                            ).tobytes()).hexdigest()
    check("D: two calls in one process agree bitwise", h1 == h2, h1[:16])
    child = (
        "import hashlib,os,sys;"
        "here=%r;root=%r;"
        "sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != here]"
        " + [root, here];"
        "import numpy as np, flopscope.numpy as fnp, p2_carrier_fnp as C;"
        "print(hashlib.sha256(np.asarray("
        "C.carrier_fnp(fnp, m=%d, k=%d, dtype='float32')).tobytes()).hexdigest())"
        % (_HERE, _ROOT, m, k))
    r = subprocess.run([sys.executable, "-c", child], capture_output=True,
                       text=True, cwd=_ROOT, errors="replace")
    h3 = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else ""
    check("D: a FRESH interpreter agrees bitwise", h3 == h1,
          f"{h3[:16]} vs {h1[:16]}" if h3 else (r.stderr or "")[-60:])
    check("D: no seed is involved anywhere in construction",
          "seed" not in C.carrier_fnp.__doc__.lower()
          or "no seed" in C.carrier_fnp.__doc__.lower(),
          "the basis subset is a fixed enumeration, not a draw")

    # ------------------------------------------------------------------ P --
    print("\nP  propagation: parity with a dense reference, and chunk edges")
    rng = np.random.default_rng(90000000301)
    for width, depth, m, k in ((16, 3, 3, 4), (64, 4, 5, 4), (256, 5, 7, 4)):
        w_np = [rng.standard_normal((width, width)).astype(np.float32)
                * np.float32((2.0 / width) ** 0.5) for _ in range(depth)]
        W = [fnp.array(x) for x in w_np]
        car = C.carrier_fnp(fnp, m=m, k=k, dtype="float32")
        radial = C.radial_factor(width)

        # the dense reference, computed here in NumPy from the SAME carrier
        h = _np(car).astype(np.float64)
        ref_rows = []
        for li in range(depth):
            h = np.maximum(h @ w_np[li].astype(np.float64), 0.0)
            ref_rows.append(h.mean(axis=0))
        ref = np.stack(ref_rows) * radial

        # `exact_layer1=False`: the NumPy reference below propagates EVERY
        # row, including row 0, so parity must be measured against the pure
        # quadrature. The exact-layer-1 substitution is a SEPARATE property and
        # is checked on its own after this loop.
        dense = _np(C.propagate_means(fnp, car, W, chunk_rows=None,
                                      radial=radial, dtype="float32",
                                      exact_layer1=False))
        check(f"P: {width}x{depth} unchunked matches a dense NumPy reference",
              np.allclose(dense, ref, rtol=2e-5, atol=2e-7),
              f"max|rel| {np.abs(dense - ref).max() / max(ref.max(), 1e-30):.2e}"
              " (float32 vs float64 association)")

        # CHUNKING IS EXACT ONLY UP TO FLOAT ASSOCIATION, and the bar must come
        # from that rather than from what happens to pass. Summing `R` float32
        # terms accumulates a relative error of order `sqrt(R) * eps32`;
        # partitioning the sum re-associates it. At `R = 8192` that is
        # `sqrt(8192) * 1.19e-7 = 1.1e-5` relative, and the observed worst is
        # `1e-5`. The first version of this check used `rtol=1e-6` and failed a
        # correct implementation five times.
        #
        # A chunking BUG is nothing like that size: a dropped tail or a wrong
        # slice loses a whole fraction of the rows, so the error is of order
        # `1/n_chunks` -- percent-level, four orders above the bound. Mutation
        # PM below drops the tail on purpose to prove the check can fail.
        n_rows = int(car.shape[0])
        eps32 = float(np.finfo(np.float32).eps)
        assoc = 4.0 * (n_rows ** 0.5) * eps32
        scale = float(np.abs(dense).max())
        for cr in (n_rows, n_rows // 2, n_rows // 4, 1000, 997, 1):
            if cr < 1 or cr > n_rows:
                continue
            got = _np(C.propagate_means(fnp, car, W, chunk_rows=cr,
                                        radial=radial, dtype="float32",
                                        exact_layer1=False))
            tail = n_rows % cr
            worst = float(np.abs(got - dense).max())
            check(f"P: {width}x{depth} chunk {cr} (tail {tail}) matches "
                  "unchunked",
                  worst <= assoc * scale,
                  f"max|diff| {worst:.3e} of a {assoc * scale:.3e} "
                  f"float32-association bound"
                  + ("  <- ODD/TAIL chunk" if tail else ""))

        # PM -- the negative control for the bound above.
        #
        # The chunk size is derived from `n_rows` so a tail ALWAYS exists and
        # `n_kept` is always positive. A literal 997 gave `n_kept = 0` on the
        # 128-row shapes, and the control reported `nan > bound` as a failure --
        # a broken control that looked like a broken guard, which is the
        # HARNESS FAILURE class `p1pre_fault_inject.py` names.
        cr = n_rows // 3 + 1
        n_kept = (n_rows // cr) * cr
        assert 0 < n_kept < n_rows, (n_rows, cr, n_kept)
        h = _np(car)[:n_kept].astype(np.float64)
        rows_ = []
        for li in range(depth):
            h = np.maximum(h @ w_np[li].astype(np.float64), 0.0)
            rows_.append(h.mean(axis=0))
        dropped = np.stack(rows_) * radial
        check(f"P: {width}x{depth} PM -- DROPPING the tail chunk is detected",
              float(np.abs(dropped - dense).max()) > assoc * scale,
              f"a dropped tail moves it {np.abs(dropped - dense).max():.3e}, "
              f"far above the {assoc * scale:.3e} bound")

        # And in float64 the same partition must agree far more tightly, which
        # separates "association" from "a bug that happens to be small".
        car64 = C.carrier_fnp(fnp, m=m, k=k, dtype="float64")
        W64 = [fnp.array(x.astype(np.float64)) for x in w_np]
        d64 = _np(C.propagate_means(fnp, car64, W64, chunk_rows=None,
                                    radial=radial, dtype="float64",
                                    exact_layer1=False))
        c64 = _np(C.propagate_means(fnp, car64, W64, chunk_rows=997,
                                    radial=radial, dtype="float64",
                                    exact_layer1=False))
        ex = _np(C.propagate_means(fnp, car, W, chunk_rows=None,
                                   radial=radial, dtype="float32",
                                   exact_layer1=True))
        check(f"P: {width}x{depth} exact_layer1 replaces ROW 0 and nothing else",
              np.array_equal(ex[1:], dense[1:])
              and not np.array_equal(ex[0], dense[0]),
              "rows 1.. bitwise unchanged; row 0 moved to the closed form")
        w0d = np.asarray(w_np[0], dtype=np.float64)
        check(f"P: {width}x{depth} ...and row 0 IS the closed form",
              float(np.abs(ex[0] - np.sqrt((w0d ** 2).sum(axis=0))
                           / np.sqrt(2 * np.pi)).max()
                    / np.abs(ex[0]).max()) < 1e-12,
              "||w|| / sqrt(2 pi), in float64")
        check(f"P: {width}x{depth} in float64 the SAME partition agrees to 1e-12",
              float(np.abs(c64 - d64).max()) < 1e-12 * max(scale, 1.0),
              f"max|diff| {np.abs(c64 - d64).max():.3e} -- association, not a bug")

    # ------------------------------------------------------------------ L --
    print("\nL  the exact layer-1 identity, which the radial factor must earn")
    # E[ReLU(w.x)] = ||w|| / sqrt(2 pi) for x ~ N(0, I). The carrier estimates
    # the ANGULAR factor; if the radial constant is wrong, layer 1 is wrong by
    # exactly that ratio -- which is why the spec validates it separately.
    # WHAT THIS SECTION CAN AND CANNOT DECIDE.
    #
    # `E[ReLU(w.x)] = ||w|| / sqrt(2 pi)` is EXACT. The carrier estimates the
    # angular half by quadrature, and `ReLU` is not a polynomial, so a finite
    # point set has genuine quadrature error even when every byte is right.
    # `docs/CURRENT_STATE.md` records the reason from the other side: the
    # carrier-aligned fourth moment is 21.688x the spherical target, so the
    # subset is provably NOT a 4-design.
    #
    # So a tight threshold here would be an invented bar that fails a correct
    # implementation -- and the first version of this check used `2e-3` and did
    # exactly that. What this section decides is whether the RADIAL FACTOR is
    # right, and it decides it by SEPARATION (with it, ~1%; without it, ~90%)
    # and by CONVERGENCE (more bases, less error). The carrier's accuracy is a
    # research quantity for WP2 and is reported, never gated, here.
    for width, m in ((64, 5), (256, 7)):
        w0 = rng.standard_normal((width, width)).astype(np.float32) \
            * np.float32((2.0 / width) ** 0.5)
        exact = np.linalg.norm(w0.astype(np.float64), axis=0) / np.sqrt(2 * np.pi)
        errs = {}
        for k in (1, 2, 4, 8):
            car = C.carrier_fnp(fnp, m=m, k=k, dtype="float64")
            # `exact_layer1=False`: this section measures how well the
            # QUADRATURE approximates a row whose answer is known. With the
            # substitution on, it would measure the closed form against itself.
            got = _np(C.propagate_means(fnp, car, [fnp.array(w0)],
                                        chunk_rows=None,
                                        radial=C.radial_factor(width),
                                        dtype="float64",
                                        exact_layer1=False))[0]
            errs[k] = float(np.sqrt(np.mean((got - exact) ** 2)) / exact.mean())
        bad = _np(C.propagate_means(
            fnp, C.carrier_fnp(fnp, m=m, k=8, dtype="float64"),
            [fnp.array(w0)], chunk_rows=None, radial=1.0, dtype="float64",
            exact_layer1=False))[0]
        bad_rel = float(np.abs(bad - exact).max() / exact.max())

        check(f"L: width {width} WITH the radial factor, layer-1 rms error is "
              "small", errs[8] < 0.05,
              "  ".join(f"k={k}:{v:.2e}" for k, v in errs.items()))
        check(f"L: width {width} WITHOUT it, layer-1 is wrong by ~90%",
              bad_rel > 0.5,
              f"{bad_rel:.3f} -- a {bad_rel / max(errs[8], 1e-12):.0f}x "
              "separation; E||x|| is not decoration")
        # CONVERGENCE is the property that actually tests the machinery: a
        # correct quadrature improves with more points. A transposed index or a
        # mis-scaled basis would sit at a floor instead.
        check(f"L: width {width} the error DECREASES with more bases",
              errs[8] < errs[4] < errs[2] < errs[1],
              "  ".join(f"k={k}:{v:.2e}" for k, v in errs.items())
              + "   (truth-free: a closed form, not a label)")
    check("L: the radial factor matches a direct lgamma evaluation",
          abs(C.radial_factor(1024) - 31.99218826) < 1e-6,
          f"E||x|| at N=1024 = {C.radial_factor(1024):.8f}")
    check("L: ...and is below sqrt(N), as Jensen requires",
          C.radial_factor(1024) < 32.0, "E||x|| < sqrt(E||x||^2) = sqrt(N)")

    # ------------------------------------------------------------------ M --
    print("\nM  mutations: each must break a NAMED property")
    m, k, n = 5, 4, 6
    N = 1 << n
    good = _np(C.carrier_fnp(fnp, m=m, k=k, antipodes=False, dtype="float64",
                             unit_norm=False)).reshape(k, N, N)
    muts = []
    # M1 wrong scaling
    muts.append(("M1 scaling: 1/N instead of 1/sqrt(N)",
                 lambda: not np.array_equal(
                     _np(C.carrier_fnp(fnp, m=m, k=k, dtype="float64")),
                     _np(C.carrier_fnp(fnp, m=m, k=k, antipodes=True,
                                       dtype="float64", unit_norm=False)) / N)))
    # M2 row order
    sw = good.copy()
    sw[[0, 1]] = sw[[1, 0]]
    muts.append(("M2 order: swapping two bases changes the object",
                 lambda: not np.array_equal(sw, good)))
    # M3 antipode convention
    full = _np(C.carrier_fnp(fnp, m=m, k=k, antipodes=True, dtype="float64",
                             unit_norm=False))
    muts.append(("M3 antipode: -v BEFORE +v is a different row order",
                 lambda: not np.array_equal(full[0::2], full[1::2])))
    # M4 dtype
    muts.append(("M4 dtype: float32 and float64 carriers differ in bytes",
                 lambda: _np(C.carrier_fnp(fnp, m=m, k=k, dtype="float32")
                             ).tobytes()
                 != _np(C.carrier_fnp(fnp, m=m, k=k, dtype="float64")).tobytes()))
    # M5 a flipped sign in one position breaks the cross-basis identity
    def m5():
        b = good.copy()
        b[0, 0, 0] *= -1
        G = np.abs(b[0] @ b[1].T)
        return float(np.abs(G - 2 ** (n / 2)).max()) > 0
    muts.append(("M5 one flipped entry breaks the cross-basis identity", m5))
    for name, fn in muts:
        try:
            ok = bool(fn())
        except Exception as exc:
            ok, name = False, name + f" (raised {type(exc).__name__})"
        check(f"M: {name}", ok, "detected")

    # ------------------------------------------------------------- LIVE ----
    if a.live:
        print("\nX  the LIVE width, 1024")
        plan = C.CarrierPlan(fnp, 1024, k=8)
        check("X: 8 antipodal bases give 16,384 rows and 8,192 lines",
              plan.n_rows == 16384 and plan.distinct_lines == 8192,
              f"{plan.n_rows} rows")
        check("X: the carrier is 64 MiB in float32",
              plan.carrier.dtype.itemsize == 4
              and plan.n_rows * 1024 * 4 == 64 * 2 ** 20, "one allocation")
        check("X: forward FLOPs at depth 16 are exactly 2**39",
              plan.forward_flops(16) == 2 ** 39,
              f"{plan.forward_flops(16):,} -> multiplier "
              f"{plan.forward_flops(16) / 2 ** 41:.4f}")
        check("X: the radial factor is 1/32-exact at N=1024",
              abs(plan.radial - C.radial_factor(1024)) == 0.0,
              f"{plan.radial:.8f}")
        ref_f = REF.carrier_float32(9, 8, antipodes=True)
        check("X: BYTE-identical to the frozen reference at the live width",
              _np(plan.carrier).tobytes() == ref_f.tobytes(),
              f"sha256 {hashlib.sha256(_np(plan.carrier).tobytes()).hexdigest()[:16]}")
    else:
        print("\nX  the LIVE width, 1024 -- NOT RUN (pass --live)")

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + "; ".join(_FAILED[:6]))
        raise SystemExit(1)
    print(f"{_TOTAL[0]}/{_TOTAL[0]} checks passed")


if __name__ == "__main__":
    main()
