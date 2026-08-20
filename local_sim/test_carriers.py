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


def main():
    print("=" * 104)
    print("CARRIER DESIGN STRENGTH — how exact is each point set, and to what degree?")
    print("=" * 104)
    t_exact_moments_are_right()
    t_radial_identity()
    profiles = t_design_strength()
    t_oa_is_a_cube_design_not_a_sphere_design(profiles)
    t_predictions(profiles)
    n_ok = sum(1 for r in RESULTS if r)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
