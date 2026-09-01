"""WP2 — the objective's algebra, its nulls, and whether it can see anything.

Contract: `experiments/LOG.md`, *CONTRACT WP2* (2026-08-31), committed at
`6059c2c` before the module existed. Predictions P1-P9 appear below as named
checks.

THE CHECK THAT MATTERS MOST IS P6.
An objective that ranks the REJECTED `rank(x)` coordinate mapping at or above
the correct carrier cannot select anything, and every number it produces is
decoration. `kerdock_mub.bad_mapping_sign_vectors` preserves that mapping as an
executable negative control precisely so this can be run rather than argued.

Run: uv run python local_sim/test_p2_carrier_objective.py
     uv run python local_sim/test_p2_carrier_objective.py --live  # + N=1024
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import kerdock_mub as REF  # noqa: E402
import p2_carrier_objective as O  # noqa: E402

_FAILED = []
_TOTAL = [0]

#: Outside every registered block. This suite reads no truth.
SEED0 = 90000000601


def check(name, ok, detail=""):
    _TOTAL[0] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<64}{detail}")
    if not ok:
        _FAILED.append(name)


def carrier(m, k):
    return REF.carrier_float32(m, k, antipodes=True).astype(np.float64)


def bad_carrier(m, k):
    """The REJECTED `rank(x)` mapping, built into a full antipodal point set.

    A perfectly valid unit-norm point set: same row count, same block structure,
    same antipodal pairing. Only the coordinate identification is wrong. Nothing
    but an objective that actually measures angular quality can tell them apart.
    """
    n = m + 1
    N = 1 << n
    s = REF.bad_mapping_sign_vectors(m, k).astype(np.float64)
    W = REF.walsh_matrix(n).astype(np.float64)
    rows = (s[:, None, :] * W[None, :, :]).reshape(k * N, N) / math.sqrt(N)
    out = np.empty((2 * k * N, N), dtype=np.float64)
    out[0::2], out[1::2] = rows, -rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true",
                    help="also run the N=1024 general-path check (~2 GiB, chunked)")
    a = ap.parse_args()
    print("WP2 - the truth-free carrier objective\n")

    # ------------------------------------------------------------------ K --
    print("K  the kernel's endpoints and its composition")
    check("K: kappa(1) = 1 exactly (the sphere is preserved)",
          O.kappa(1.0) == 1.0, repr(float(O.kappa(1.0))))
    check("K: kappa(0) = 1/pi", abs(O.kappa(0.0) - 1 / math.pi) < 1e-15,
          "%.12f" % O.kappa(0.0))
    check("K: kappa(-1) = 0", abs(O.kappa(-1.0)) < 1e-15, "%.2e" % O.kappa(-1.0))
    # THE TRAP THAT PRODUCED NEGATIVE DISCREPANCIES DURING DESIGN.
    check("K: kappa^L(-1) is 0 only at L=1; at L>=2 it is kappa(0)",
          float(O.kappa_L(-1.0, 1)) == 0.0
          and abs(float(O.kappa_L(-1.0, 2)) - 1 / math.pi) < 1e-15
          and abs(float(O.kappa_L(-1.0, 16)) - float(O.kappa_L(0.0, 15))) < 1e-15,
          "assuming 0 gave NEGATIVE squared discrepancies")
    check("K: kappa maps [-1,1] into [0,1] and is non-decreasing",
          bool(np.all(np.diff(O.kappa(np.linspace(-1, 1, 4001))) >= -1e-15))
          and bool(np.all(O.kappa(np.linspace(-1, 1, 4001)) >= -1e-15)),
          "a valid zonal kernel on the sphere")
    check("K: kappa^L(1) = 1 at every depth",
          all(float(O.kappa_L(1.0, L)) == 1.0 for L in (1, 2, 8, 16, 64)),
          "depth cannot shrink the diagonal")
    # c_L convergence, pinned rather than trusted
    for n in (64, 1024):
        lo, hi = O.c_L(n, 16, degree=1500), O.c_L(n, 16, degree=6000)
        check(f"K: c_L(n={n}) is converged in the quadrature degree",
              abs(lo - hi) / hi < 1e-12, f"1500 vs 6000 nodes: {abs(lo-hi):.2e}")
    check("K: c_L -> kappa^L(0) as n grows (rho concentrates at 0)",
          abs(O.c_L(65536, 16) - float(O.kappa_L(0.0, 16)))
          < abs(O.c_L(16, 16) - float(O.kappa_L(0.0, 16))),
          "the uniform measure's inner product has sd 1/sqrt(n)")

    # ------------------------------------------------------------- P1/P2 --
    print("\nP1/P2  non-negativity, and the closed form against the general path")
    cases = [(3, 2), (3, 4), (3, 8), (5, 4), (5, 8), (7, 4), (7, 8), (7, 16)]
    for m, k in cases:
        N = 1 << (m + 1)
        U = carrier(m, k)
        gen = O.mmd2(U, depth=16)
        clo = O.mmd2_carrier_closed(k, N, depth=16)
        check(f"P1: m={m} k={k:<2} MMD^2 > 0 and finite",
              gen > 0 and math.isfinite(gen), "%.6e" % gen)
        rel = abs(gen - clo) / abs(gen)
        check(f"P2: m={m} k={k:<2} closed form == chunked general path",
              rel <= 1e-7, "rel %.2e  (closed %.6e)" % (rel, clo))
    # the census must close, and it is the closed form's whole basis
    for m, k in ((3, 4), (7, 8)):
        N = 1 << (m + 1)
        cen = O.carrier_gram_census(k, N)
        G = np.round(carrier(m, k) @ carrier(m, k).T * N).astype(np.int64)
        import collections
        obs = collections.Counter(G.ravel().tolist())
        want = {int(round(v * N)): c for v, c in cen.items()}
        check(f"P2: m={m} k={k} the Gram census is EXACT, not assumed",
              dict(obs) == want, f"{len(obs)} distinct values, counts match")

    # ------------------------------------------------------------- P3/P4 --
    print("\nP3/P4  rotation and permutation invariance")
    rng = np.random.default_rng(SEED0)
    for m, k in ((3, 4), (5, 4)):
        N = 1 << (m + 1)
        U = carrier(m, k)
        base = O.mmd2(U, depth=16)
        q, r = np.linalg.qr(rng.standard_normal((N, N)))
        Q = q * np.sign(np.diag(r))
        rot = O.mmd2(U @ Q, depth=16)
        check(f"P3: m={m} k={k} rotation invariant",
              abs(rot - base) / base <= 1e-10, "rel %.2e" % (abs(rot-base)/base))
        perm = O.mmd2(U[rng.permutation(U.shape[0])], depth=16)
        # P4 AS CONTRACTED SAID "EXACTLY (bitwise)". THAT IS REFUTED, and the
        # prediction was wrong rather than the code. Reordering rows reorders
        # NumPy's pairwise summation of the Gram, which moves the total by one
        # ULP: measured `2.22e-16` absolute, `7.9e-13` relative, identically at
        # `chunk = R` so it is not the chunking. It is exact at m=5 only because
        # that sum happens to reassociate cleanly.
        #
        # The bar is set by the RESOLUTION THE OBJECTIVE NEEDS, not by what
        # passes: the tightest separation it must call is P6 at m=7 k=4, which
        # is `1.006x` -- `6e-3` relative. A `7.9e-13` reordering residual is ten
        # orders below that, so it cannot flip a ranking.
        rel = abs(perm - base) / abs(base)
        check(f"P4: m={m} k={k} permutation invariant to summation order",
              rel <= 1e-11,
              ("bitwise identical" if perm == base else "rel %.2e" % rel)
              + " -- vs a 6e-3 separation to resolve")
    # the randomized-carrier NULL control must be indistinguishable
    N = 64
    rc = O.control_randomized_carrier(2 * 4 * N, N, 4, SEED0 + 1)
    check("P3: the randomized-orientation control scores as the carrier",
          abs(O.mmd2(rc, depth=16) - O.mmd2(carrier(5, 4), depth=16))
          / O.mmd2(carrier(5, 4), depth=16) <= 1e-10,
          "a NULL control: if it differed, every ranking would be suspect")

    # ---------------------------------------------------------------- P8 --
    print("\nP8  the two EXACT algebraic nulls")
    for m, k in ((3, 4), (5, 8), (7, 4)):
        N = 1 << (m + 1)
        U = carrier(m, k)
        R = U.shape[0]
        for power in (1, 3, 5):
            s = O.gram_power_sum(U, power)
            check(f"P8: m={m} k={k} sum <u_i,u_j>^{power} = 0 (antipodal)",
                  abs(s) < 1e-6 * R, "%.3e over %d pairs" % (s, R * R))
        s2 = O.gram_power_sum(U, 2)
        want = R * R / float(N)
        check(f"P8: m={m} k={k} sum <u_i,u_j>^2 = R^2/N (tight frame)",
              abs(s2 - want) / want < 1e-12,
              "%.6f vs %.6f" % (s2, want))

    # ---------------------------------------------------------------- P5 --
    print("\nP5  MMD^2 strictly decreases in k, and roughly as 1/k")
    panel = O.build_panel(n=1024, depth=16)
    vals = [p["mmd2"] for p in panel["policies"]]
    ks = [p["k"] for p in panel["policies"]]
    check("P5: strictly decreasing over k = " + ", ".join(map(str, ks)),
          all(b < a for a, b in zip(vals, vals[1:])),
          " > ".join("%.3e" % v for v in vals[:4]) + " > ...")
    prods = [v * k for v, k in zip(vals, ks)]
    spread = max(prods) / min(prods)
    check("P5: k * MMD^2 is constant to <1% -- an almost exact 1/k law",
          spread < 1.01, "spread %.5f, k*MMD^2 ~ %.4e" % (spread, prods[0]))

    # ---------------------------------------------------------------- P6 --
    print("\nP6  THE DECISIVE CHECK: can it see a known-bad carrier?")
    for m, k in ((3, 4), (5, 4), (5, 8), (7, 4)):
        good, bad = carrier(m, k), bad_carrier(m, k)
        gm, bm = O.mmd2(good, depth=16), O.mmd2(bad, depth=16)
        check(f"P6: m={m} k={k} the REJECTED rank(x) mapping scores WORSE",
              bm > gm, "bad %.6e vs good %.6e  (%.3fx)" % (bm, gm, bm / gm))
        # and it really is a valid point set, or the comparison is unfair
        nrm = np.sqrt((bad * bad).sum(axis=1))
        check(f"P6: m={m} k={k} ...and the bad set is a VALID unit-norm set",
              float(np.abs(nrm - 1).max()) < 1e-6
              and bad.shape == good.shape,
              "same shape, unit rows -- only the mapping differs")

    # ---------------------------------------------------------------- P7 --
    print("\nP7  the carrier against matched-row controls")
    for m, k, draws in ((5, 4, 8), (7, 4, 8)):
        N = 1 << (m + 1)
        rows = 2 * k * N
        base = O.mmd2(carrier(m, k), depth=16)
        for name, fn in O.CONTROLS.items():
            vs = [O.mmd2(fn(rows, N, SEED0 + 100 * d + hash(name) % 97,
                            **({} if name != "x" else {})), depth=16)
                  for d in range(draws)]
            mean = float(np.mean(vs))
            check(f"P7: m={m} k={k} carrier beats '{name}' in the mean of "
                  f"{draws} draws",
                  base < mean,
                  "carrier %.4e vs %.4e  (%.3fx better)  [min %.4e]"
                  % (base, mean, mean / base, min(vs)))

    # ---------------------------------------------------------------- P9 --
    print("\nP9  the objective reports the carrier as IMPERFECT")
    check("P9: MMD^2 > 0 at every panel k -- it is not a spherical design",
          all(p["mmd2"] > 0 for p in panel["policies"]),
          "consistent with the recorded 21.688x fourth-moment excess")
    check("P9: ...and an objective returning 0 would be measuring nothing",
          min(p["mmd2"] for p in panel["policies"]) > 1e-12,
          "smallest panel value %.3e" % min(p["mmd2"] for p in panel["policies"]))

    # ----------------------------------------------------------- PANEL ----
    print("\nF  the panel is FROZEN, content-addressed, and its rule is total")
    d1, d2 = O.panel_digest(panel), O.panel_digest(O.build_panel(1024, 16))
    check("F: the panel digest is deterministic", d1 == d2, d1[:16])
    moved = O.build_panel(1024, 16)
    moved["policies"][0]["mmd2"] *= 1.0000001
    check("F: ...and moves if ANY policy number moves",
          O.panel_digest(moved) != d1, "content-addressed, not name-addressed")
    moved2 = O.build_panel(1024, 16)
    moved2["policy_rule"] = moved2["policy_rule"] + " "
    check("F: ...and moves if the RULE moves",
          O.panel_digest(moved2) != d1, "the rule is inside the digest")
    for ceiling, want_k in ((O.BUDGET, 16), (0.25 * O.BUDGET, 8),
                            (0.1 * O.BUDGET, 2)):
        got = O.select_policy(panel, ceiling)
        check(f"F: the rule picks k={want_k} at a {ceiling/O.BUDGET:.2f}B ceiling",
              got["k"] == want_k, f"k={got['k']}, MMD^2 {got['mmd2']:.4e}")
    try:
        O.select_policy(panel, 1)
        check("F: the rule REFUSES when nothing fits", False, "it returned")
    except ValueError as e:
        check("F: the rule REFUSES when nothing fits", True, str(e)[:52])
    for key in ("design_falsifier", "proposed_score_gate", "limitation"):
        check(f"F: the panel carries its '{key}' in the digested content",
              bool(panel.get(key)) and len(panel[key]) > 40,
              panel[key][:46] + "...")

    # ---------------------------------------------------------------- X ----
    if a.live:
        print("\nX  the LIVE width: the general path must confirm the closed form")
        U = carrier(9, 2)                      # 4096 x 1024, chunked
        gen = O.mmd2(U, depth=16, chunk=1024)
        clo = O.mmd2_carrier_closed(2, 1024, depth=16)
        check("X: N=1024 k=2 closed form == chunked general path",
              abs(gen - clo) / abs(gen) <= 1e-7,
              "gen %.8e  closed %.8e  rel %.2e"
              % (gen, clo, abs(gen - clo) / abs(gen)))
        check("X: N=1024 k=2 sum <u_i,u_j>^2 = R^2/N exactly",
              abs(O.gram_power_sum(U, 2, chunk=1024)
                  - U.shape[0] ** 2 / 1024.0) / (U.shape[0] ** 2 / 1024.0) < 1e-12,
              "the tight-frame identity at the shipping width")
    else:
        print("\nX  the LIVE width -- NOT RUN (pass --live)")

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + "; ".join(_FAILED[:6]))
        raise SystemExit(1)
    print(f"{_TOTAL[0]}/{_TOTAL[0]} checks passed")


if __name__ == "__main__":
    main()
