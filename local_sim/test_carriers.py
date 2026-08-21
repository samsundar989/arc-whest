"""
DESIGN STRENGTH — what degree is each carrier actually exact to?

Stage 1 / item 5. Contract and predictions S1-S5: `experiments/LOG.md`,
2026-08-19 (night).

THE QUESTION
------------
`docs/CURRENT_STATE.md` open question #3 leaves one input-space lane alive: a true
degree-4/5 angular design, external ceiling ~1.46x from the layer-32 harmonic
spectrum. That ceiling is quoted against an IDEAL baseline.

The carrier audit's governing finding was SUBSTITUTION — two devices that
annihilate the same low-order structure do not compose. So the number that decides
whether to build a Kerdock/BCH design is not "how much variance lives in degrees
<= 5" but **how much of it the shipped carrier already removes**. Nobody measured
that: the audit measured outcomes, never the carriers' design strength.

THE TEST
--------
For unit vectors `{u_i}` and a random unit `a`, a spherical t-design satisfies

    mean_i (a . u_i)^k  ==  E_{u~S^{d-1}}[(a . u)^k]   for every k <= t

    E[(a.u)^k] = 0                                     k odd
               = (k-1)!! / (d (d+2) ... (d+k-2))       k even

Ridge powers `(a.u)^k` generate the degree-k polynomials as `a` varies, so
agreement over many random `a` is a strong practical test — **not a proof**. A set
that fails it is definitely not a t-design; a set that passes it is very likely
one. That asymmetry is the right way round for a kill test.

Reported per carrier and per degree: the RMS relative deviation from the exact
moment over many random directions, and the ratio to iid's. **A carrier is doing
work at degree k exactly insofar as that ratio is small.**

Run: uv run python local_sim/test_carriers.py
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

from carriers import CARRIERS, build, mean_radius, to_sphere  # noqa: E402

RESULTS = []
DIM = 256
N = 16384          # the shipped operating point
N_DIRS = 4000      # random probe directions a
DEGREES = (1, 2, 3, 4, 5, 6)


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {name:<58} {'ok ' if ok else 'FAILED'}  {detail}")
    return ok


def head(t):
    print(f"\n{t}\n{'-' * len(t)}")


def exact_moment(k: int, d: int) -> float:
    """E[(a.u)^k] for u uniform on S^{d-1}, ||a|| = 1."""
    if k % 2:
        return 0.0
    m = k // 2
    num = 1.0
    for j in range(m):
        num *= (2 * j + 1)
    den = 1.0
    for j in range(m):
        den *= (d + 2 * j)
    return num / den


def strength_profile(u: np.ndarray, dim: int, seed: int, n_dirs: int = N_DIRS):
    """RMS relative deviation from the exact spherical moment, per degree.

    `u` is projected to the sphere first, so a carrier's radial choice cannot
    contaminate a purely ANGULAR measurement. Odd degrees have exact moment 0, so
    they are normalised by the degree-2 scale (1/d) raised to k/2 — otherwise the
    relative error is undefined.
    """
    us = to_sphere(u)
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, n_dirs))
    a /= np.linalg.norm(a, axis=0, keepdims=True)
    proj = us @ a                                  # (n, n_dirs)
    out = {}
    for k in DEGREES:
        got = (proj ** k).mean(axis=0)             # (n_dirs,)
        want = exact_moment(k, dim)
        scale = abs(want) if want != 0.0 else exact_moment(2, dim) ** (k / 2.0)
        out[k] = float(np.sqrt(np.mean(((got - want) / scale) ** 2)))
    return out


def t_exact_moments_are_right():
    head("0. the reference moments themselves")
    d = DIM
    check("E[(a.u)^2] == 1/d", abs(exact_moment(2, d) - 1.0 / d) < 1e-18,
          f"{exact_moment(2, d):.6e}")
    check("E[(a.u)^4] == 3/(d(d+2))", abs(exact_moment(4, d) - 3.0 / (d * (d + 2))) < 1e-20,
          f"{exact_moment(4, d):.6e}")
    check("E[(a.u)^6] == 15/(d(d+2)(d+4))",
          abs(exact_moment(6, d) - 15.0 / (d * (d + 2) * (d + 4))) < 1e-22)
    check("odd moments are zero", all(exact_moment(k, d) == 0.0 for k in (1, 3, 5)))
    # An independent check of the whole machinery: the deviation must fall at the
    # Monte-Carlo rate as the point count grows. A fixed tolerance would be the
    # wrong test — the odd degrees are normalised by (1/d)^(k/2), a deliberately
    # tiny scale, so their relative deviation is large in absolute terms at ANY
    # n. What must hold is 1/sqrt(n): 200,000 points against 16,384 is a factor
    # sqrt(200000/16384) = 3.49 in every degree.
    small = to_sphere(np.random.default_rng(7).standard_normal((16_384, d)))
    big = to_sphere(np.random.default_rng(8).standard_normal((200_000, d)))
    ps = strength_profile(small, d, seed=11, n_dirs=400)
    pb = strength_profile(big, d, seed=11, n_dirs=400)
    want = math.sqrt(200_000 / 16_384)
    ratios = {k: ps[k] / pb[k] for k in DEGREES}
    check("iid deviation falls at the Monte-Carlo rate 1/sqrt(n)",
          all(0.6 * want < r < 1.7 * want for r in ratios.values()),
          f"expect {want:.2f}x;  " + "  ".join(f"k{k}={ratios[k]:.2f}" for k in DEGREES))


def t_radial_identity():
    head("1. the radial identity — exact, not a model")
    d = DIM
    er = mean_radius(d)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((400_000, d))
    got = float(np.linalg.norm(x, axis=1).mean())
    check("mean_radius(d) matches a 400k-sample E||x||",
          abs(got - er) / er < 2e-3, f"exact {er:.6f}  sampled {got:.6f}")
    check("E||x|| ~ sqrt(d - 1/2) to 1e-4 at d=256",
          abs(er - math.sqrt(d - 0.5)) / er < 1e-4,
          f"{er:.6f} vs {math.sqrt(d - 0.5):.6f}")
    # Independence of R and u is what licenses replacing R by its mean.
    r = np.linalg.norm(x, axis=1)
    u0 = x[:, 0] / r
    corr = float(np.corrcoef(r, np.abs(u0))[0, 1])
    check("R and u are uncorrelated (the independence the identity needs)",
          abs(corr) < 0.01, f"corr(R, |u_0|) = {corr:+.5f}")


def t_design_strength():
    head(f"2. MEASURED design strength — d = {DIM}, n = {N}, {N_DIRS} probe directions")
    profiles = {}
    for name in CARRIERS:
        n = N
        if name == "haar frames":
            n = (N // (2 * DIM)) * 2 * DIM      # 16384 = 32 bases of 256, exactly
        elif name.startswith("kerdock+axial"):
            # An exact design has ONE size: 2^16 sign vectors + 2d axial points.
            # Subsampling restores the S-dependence the axial correction removes,
            # which is the whole property being measured, so the carrier refuses.
            n = (1 << 16) + 2 * DIM
        elif name.startswith("kerdock"):
            n = 1 << 16
        elif name.startswith("bch OA half"):
            n = 1 << 16
        elif name.startswith("bch OA"):
            # An orthogonal array has ONE size. Subsampling it destroys the
            # balance that makes it an array at all, so it is measured whole:
            # 2^17 = 131,072 rows, i.e. 65,536 antipodal pairs.
            n = 1 << 17
        pts = build(name, n, DIM, seed=20260819)
        nd = 800 if n > 40000 else N_DIRS       # the big sets are 8x the work
        profiles[name] = strength_profile(pts, DIM, seed=4242, n_dirs=nd)
        profiles[name]["_n"] = n

    base = profiles["iid"]
    print(f"\n  {'carrier':<22}{'n':>7}" + "".join(f"{'k=' + str(k):>12}" for k in DEGREES))
    print("  " + "-" * (29 + 12 * len(DEGREES)))
    for name, p in profiles.items():
        print(f"  {name:<22}{p['_n']:>7}"
              + "".join(f"{p[k]:>12.3e}" for k in DEGREES))
    print(f"\n  ratio to iid (smaller = the carrier is doing more work at that degree)")
    print(f"  {'carrier':<22}{'':>7}" + "".join(f"{'k=' + str(k):>12}" for k in DEGREES))
    print("  " + "-" * (29 + 12 * len(DEGREES)))
    for name, p in profiles.items():
        print(f"  {name:<22}{'':>7}"
              + "".join(f"{p[k] / base[k]:>12.3e}" for k in DEGREES))
    return profiles


def t_oa_is_a_cube_design_not_a_sphere_design(profiles):
    """The OA's strength is combinatorial. Sphericality is a separate question.

    Every coordinate of every OA point is +-1/sqrt(d), so along a COORDINATE
    direction `(a.u)^4` is identically 1/d^2, against the spherical 3/(d(d+2)).
    The array can therefore never be a spherical 4-design however high its
    combinatorial strength. What a random rotation buys is that a generic
    direction stops being special — which is exactly what the profile measures.
    """
    head("2b. the OA is a design for the CUBE. Is it one for the sphere?")
    d = DIM
    rot = profiles.get("bch OA (rotated)")
    unrot = profiles.get("bch OA (unrotated)")
    if rot is None:
        return
    axis = 1.0 / d ** 2
    sphere = exact_moment(4, d)
    check("along a coordinate axis the OA's 4th moment is 1/d^2, not 3/(d(d+2))",
          abs(axis / sphere - (d + 2) / (3.0 * d)) < 1e-12,
          f"axis {axis:.4e}  vs sphere {sphere:.4e}  ratio {axis / sphere:.4f}")

    # My first version of this check asserted "unrotated is worse than rotated"
    # and it is FALSE: 7.550e-04 against 7.772e-04, i.e. identical. The reason is
    # that rotating the POINT SET is the same as rotating the PROBE DIRECTIONS,
    # and the probes are already isotropic — so an isotropic moment test is
    # rotation-invariant by construction and CANNOT see axis alignment. The
    # rotation matters for the real integrand, not for this measurement. The
    # defect is visible only by probing along the axes themselves.
    check("the isotropic probe is rotation-invariant, so it cannot see alignment",
          abs(unrot[4] - rot[4]) / rot[4] < 0.10,
          f"unrotated {unrot[4]:.3e}  rotated {rot[4]:.3e} — agree, as they must")

    from carriers import bch_oa_unrotated
    pts = to_sphere(bch_oa_unrotated(1 << 17, d, 0))
    eye = np.eye(d)
    got_axis = float(np.mean((pts @ eye) ** 4))
    check("probed along the AXES, the unrotated OA is badly wrong at degree 4",
          abs(got_axis - axis) / axis < 1e-9
          and abs(got_axis - sphere) / sphere > 0.5,
          f"axis-probe {got_axis:.4e} == 1/d^2, which is {got_axis / sphere:.3f}x "
          f"the spherical value")
    check("probed along RANDOM directions the same points are near-exact",
          unrot[4] < 1e-3,
          f"{unrot[4]:.3e} relative — {1.0 / (unrot[4] / (axis / sphere - 1) ):.0f}x "
          f"closer than the axis probe")


def t_confinement(profiles):
    """Is the degree-4 error a property of the ARRAY, or of the DIRECTION?

    This is the test that killed the subspace route. The estimator's response
    lives in a rank-32 subspace (`_alignment_basis`), so it looked as though a
    design confined to 32 dimensions would deliver the same gain at 1/64th the
    point count. It cannot, and the reason is arithmetic rather than
    constructional.

    For a `+-1/16` point set with all 4-subsets balanced,

        E[(a.u)^4] = (3 - 2S) / 65536,     S = sum_i a_i^4

    against the spherical `3/(256*258)`. The ratio is `(3 - 2S)*258/(3*256)`,
    which is **exactly 1 when S = 3/258** — precisely the expected S for a random
    unit vector in R^256. So the array is exact in expectation for SPREAD
    directions and increasingly wrong for confined ones, whatever its size.
    """
    head("2c. is the degree-4 error the ARRAY's or the DIRECTION's?")
    rng = np.random.default_rng(20260819)
    want = exact_moment(4, DIM)

    def dirs(k):
        a = np.zeros((DIM, 1200))
        a[:k] = rng.standard_normal((k, 1200))
        return a / np.linalg.norm(a, axis=0, keepdims=True)

    def err(pts, d):
        got = ((to_sphere(pts) @ d) ** 4).mean(axis=0)
        return float(np.sqrt(np.mean(((got - want) / want) ** 2)))

    ks = (256, 128, 64, 32)
    # The UNROTATED array is the one to probe: "confined to the first k
    # coordinates" then means confined RELATIVE TO THE ARRAY'S OWN BALANCE
    # STRUCTURE, which is exactly the subspace route's proposal (align the
    # balanced columns with the response's subspace). The ROTATED array is the
    # deployed case and is shown alongside, because the contrast IS the finding.
    ks = (256, 128, 64, 32)
    sets = {"sobol (SHIPPED)": build("sobol (SHIPPED)", N, DIM, 20260819),
            "bch OA, UNROTATED": build("bch OA (unrotated)", 1 << 17, DIM, 20260819),
            "bch OA, rotated": build("bch OA (rotated)", 1 << 17, DIM, 20260819)}
    D = {k: dirs(k) for k in ks}
    print(f"  {'point set':<22}" + "".join(f"{'conf ' + str(k):>14}" for k in ks))
    print("  " + "-" * (22 + 14 * len(ks)))
    out = {}
    for nm, P in sets.items():
        out[nm] = {k: err(P, D[k]) for k in ks}
        print(f"  {nm:<22}" + "".join(f"{out[nm][k]:>14.3e}" for k in ks))
    print("\n  predicted (3-2S)*258/(3*256), S = 3/(k+2):")
    for k in ks:
        S = 3.0 / (k + 2)
        print(f"    confined to k={k:>3}  ratio {(3 - 2*S)*258/(3*256):.5f}  "
              f"-> {(1 - (3 - 2*S)*258/(3*256))*100:.2f}% low")

    raw, rot, sob = (out["bch OA, UNROTATED"], out["bch OA, rotated"],
                     out["sobol (SHIPPED)"])
    check("the array beats Sobol by >10x on SPREAD directions",
          sob[256] / raw[256] > 10, f"{sob[256] / raw[256]:.1f}x")
    check("UNROTATED, it LOSES to Sobol on directions confined to 32 dims",
          raw[32] > sob[32], f"OA {raw[32]:.3e} vs sobol {sob[32]:.3e} "
          f"({raw[32] / sob[32]:.1f}x worse)")
    check("and that confined error matches the analytic (3-2S) prediction",
          abs(raw[32] / 0.0515 - 1) < 0.20, f"{raw[32]:.3e} vs predicted 5.15e-02")
    check("ROTATION removes the confinement penalty entirely",
          rot[32] < raw[32] / 10,
          f"rotated {rot[32]:.3e} vs unrotated {raw[32]:.3e} — a random rotation "
          f"turns any confined direction into a spread one")
    check("SO the design cannot be shrunk to the rank-32 subspace",
          raw[32] > sob[32] and rot[32] < sob[32],
          "aligning the balanced columns with the response costs more than it saves")


def t_predictions(profiles):
    head("3. the pre-registered predictions")
    base = profiles["iid"]
    haar = profiles["haar frames"]
    sob = profiles["sobol (SHIPPED)"]
    anti = profiles["sobol+anti"]

    # S1 — Haar frames exact at degree 2, not at degree 4.
    check("S1: Haar frames are EXACT at degree 2", haar[2] < 1e-12,
          f"rel err {haar[2]:.3e}")
    check("S1: Haar frames are NOT exact at degree 4", haar[4] > 1e-6,
          f"rel err {haar[4]:.3e}, {haar[4] / base[4]:.3f}x iid")

    # S2 — the shipped carrier is not exact at degree 2, but is well below iid.
    check("S2: shipped Sobol is NOT exact at degree 2", sob[2] > 1e-12,
          f"rel err {sob[2]:.3e}")
    check("S2: shipped Sobol beats iid at degree 2", sob[2] < base[2],
          f"{base[2] / sob[2]:.2f}x better than iid")

    # S3 — THE DECISIVE ONE.
    r4 = sob[4] / base[4]
    verdict = ("LANE ALIVE — a true degree-4 design still has room"
               if r4 > 0.1 else
               "LANE DEAD — Sobol already collects degree 4")
    check("S3: shipped Sobol's degree-4 error is within 10x of iid's",
          r4 > 0.1, f"{r4:.4f}x iid  ->  {verdict}")

    # S4 — antipodal symmetry kills odd moments exactly.
    check("S4: antipodal makes every ODD moment exactly zero",
          all(anti[k] < 1e-12 for k in (1, 3, 5)),
          "  ".join(f"k{k}={anti[k]:.2e}" for k in (1, 3, 5)))
    check("S4: and the shipped carrier alone does NOT",
          any(sob[k] > 1e-9 for k in (1, 3, 5)),
          "  ".join(f"k{k}={sob[k]:.2e}" for k in (1, 3, 5)))

    # S5 — the MUB arithmetic.
    d = DIM
    k = int(round(math.log(d, 4)))
    check("S5: 256 = 4^4, so the real-MUB bound d/2+1 is attainable",
          4 ** k == d and d // 2 + 1 == 129, f"4^{k} = {4 ** k}, bound = {d // 2 + 1}")
    check("S5: 129 x 256 x 2 == 66,048 signed directions (Team Puffi's figure)",
          129 * 256 * 2 == 66048, f"{129 * 256 * 2:,}")
    check("S5: a 66,048-direction bank is 4.03x the shipped operating point",
          abs(66048 / N - 4.03) < 0.01, f"{66048 / N:.2f}x N = {N:,}")


def t_kerdock():
    """CP1.1 — the non-linear construction, its fingerprint, and the identity.

    Contract and predictions K1-K5: `experiments/LOG.md`, 2026-08-20, committed as
    b2a92a2 BEFORE any of this ran.
    """
    import collections
    from carriers import (_kerdock_bits, kerdock_signs, kerdock_axial,
                          kerdock_axial_unrotated)

    head("6. KERDOCK / MUB-129 — the non-linear escape from the 131,072 floor")

    # ---- K1: the fingerprint. If this fails, nothing measured below counts. --
    bits = _kerdock_bits()
    uniq = np.unique(np.packbits(bits.astype(np.uint8), axis=1), axis=0)
    compl = np.unique(np.packbits((1 - bits).astype(np.uint8), axis=1), axis=0)
    w = bits.sum(axis=1)
    counts = dict(sorted(collections.Counter(w.tolist()).items()))
    check("K1: exactly 65,536 distinct codewords of length 256",
          len(uniq) == 65536 and bits.shape[1] == 256, f"{len(uniq):,} x {bits.shape[1]}")
    check("K1: closed under complement", np.array_equal(uniq, compl))
    check("K1: weight support is EXACTLY the Kerdock signature",
          sorted(counts) == [0, 120, 128, 136, 256],
          f"{sorted(counts)}   min distance 2^7 - 2^3 = 120")
    check("K1: and the weight COUNTS are Kerdock's",
          counts == {0: 1, 120: 32512, 128: 510, 136: 32512, 256: 1}, f"{counts}")

    # ---- the axial correction is an IDENTITY, not a tuning ------------------
    d = DIM
    sg = kerdock_signs(d).astype(np.float64)
    rng = np.random.default_rng(20260820)
    tgt4 = exact_moment(4, d)
    worst_pred, worst_exact = 0.0, 0.0
    for _ in range(24):
        a = rng.standard_normal(d); a /= np.linalg.norm(a)
        Sv = float((a ** 4).sum())
        p = sg @ a / math.sqrt(d)
        worst_pred = max(worst_pred,
                         abs(float((p ** 4).mean()) - (3.0 - 2.0 * Sv) / d ** 2)
                         / ((3.0 - 2.0 * Sv) / d ** 2))
        tot = float((p ** 4).sum()) + 2.0 * Sv          # axial: (a.e_i)^4 = a_i^4
        worst_exact = max(worst_exact, abs(tot / (sg.shape[0] + 2 * d) - tgt4) / tgt4)
    check("the sign set obeys (3 - 2S)/d^2 for EVERY direction",
          worst_pred < 1e-12, f"worst rel dev {worst_pred:.2e}")
    check("+ the 2d axial points makes degree 4 EXACT, at uniform weights",
          worst_exact < 1e-12,
          f"worst rel dev {worst_exact:.2e}   [(3-2S) + 2S]/(2^16+2d) = 3/(d(d+2))")

    # ---- the deployment case: CONFINED directions ---------------------------
    # docs/GOTCHAS: a bare cube array confined to 32 dims reads 5.15% low and is
    # 4.6x WORSE than Sobol unrotated. That is what closed the subspace route.
    pts = kerdock_axial_unrotated(66048, d, 0)
    for kdim in (256, 64, 32):
        errs = []
        for _ in range(8):
            a = np.zeros(d)
            idx = rng.choice(d, kdim, replace=False)
            a[idx] = rng.standard_normal(kdim)
            a /= np.linalg.norm(a)
            pr = to_sphere(pts) @ a
            errs.append(abs(float((pr ** 4).mean()) - tgt4) / tgt4)
        e = float(np.sqrt(np.mean(np.square(errs))))
        check(f"exact at degree 4 with directions confined to {kdim:>3} dims",
              e < 1e-12, f"rel err {e:.2e}   (a bare cube array degrades here)")

    # ---- K2/K3/K4: the measured profile against the recorded baselines ------
    prof = {}
    for name, n in (("iid", 65536), ("sobol (SHIPPED)", 65536),
                    ("haar frames", 65536), ("kerdock (rotated)", 65536),
                    ("kerdock+axial (rotated)", 66048),
                    ("bch OA (rotated)", 1 << 17)):
        prof[name] = strength_profile(build(name, n, d, seed=20260820), d,
                                      seed=4242, n_dirs=400)
        prof[name]["_n"] = n
    base = prof["iid"]
    print(f"\n  {'carrier':<26}{'n':>8}"
          + "".join(f"{'k=' + str(k):>10}" for k in DEGREES) + "   (ratio to iid)")
    print("  " + "-" * (34 + 10 * len(DEGREES)))
    for nm, p in prof.items():
        print(f"  {nm:<26}{p['_n']:>8}"
              + "".join(f"{p[k] / base[k]:>10.4f}" for k in DEGREES))

    k4 = prof["kerdock (rotated)"][4] / base[4]
    k4a = prof["kerdock+axial (rotated)"][4] / base[4]
    k4b = prof["bch OA (rotated)"][4] / base[4]
    check("K2: rotated Kerdock alone clears the 0.10x GO gate at degree 4",
          k4 <= 0.10, f"{k4:.4f}x iid at 65,536 points "
                      f"(the BCH array reads {k4b:.4f}x at 131,072)")
    check("K2: and it matches the BCH array's degree-4 error at HALF the points",
          abs(prof["kerdock (rotated)"][4] - prof["bch OA (rotated)"][4])
          / prof["bch OA (rotated)"][4] < 0.05,
          f"{prof['kerdock (rotated)'][4]:.4e} vs {prof['bch OA (rotated)'][4]:.4e}")
    check("K3: the axial points do not merely help, they make degree 4 EXACT",
          k4a < 1e-8, f"{prof['kerdock+axial (rotated)'][4]:.3e} absolute")
    check("K4: degree 6 is below iid but NOT exact (a 5-design, not a 7-design)",
          0.0 < prof["kerdock+axial (rotated)"][6] / base[6] < 1.0,
          f"{prof['kerdock+axial (rotated)'][6] / base[6]:.4f}x iid")
    check("degrees 1-5 are all at machine precision",
          all(prof["kerdock+axial (rotated)"][k] < 1e-10 for k in (1, 2, 3, 4, 5)),
          ", ".join(f"k{k}={prof['kerdock+axial (rotated)'][k]:.1e}"
                    for k in (1, 2, 3, 4, 5)))

    # ---- the count, three ways ---------------------------------------------
    check("2^16 + 2d == 129 x 256 x 2 == 66,048 — the three accountings agree",
          (1 << 16) + 2 * d == 129 * 256 * 2 == 66048,
          f"Kerdock+axial = Kuperberg 2^16+2n = K129/MUB")


def main():
    print("=" * 104)
    print("CARRIER DESIGN STRENGTH — how exact is each point set, and to what degree?")
    print("=" * 104)
    t_exact_moments_are_right()
    t_radial_identity()
    profiles = t_design_strength()
    t_oa_is_a_cube_design_not_a_sphere_design(profiles)
    t_confinement(profiles)
    t_predictions(profiles)
    t_kerdock()
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
