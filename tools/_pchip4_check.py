"""Standalone check of the 4-knot monotone cubic used by the D2 coefficient layer.

The profile must satisfy three properties before it goes anywhere near the
estimator, because a fit that silently changes behaviour at its own defaults is
worse than no fit at all:

  1. **Constant reproduction.** Four equal knots must return that value EXACTLY
     for every layer index -- not to 1e-15, exactly -- or "defaults = the
     current constants" is false and bit-parity with C10 is unreachable.
  2. **Endpoint interpolation.** t=0 returns knot 0 and t=1 returns knot 3.
  3. **Monotone preservation.** Monotone knots must not produce an overshooting
     curve; that is the whole reason for PCHIP over a natural cubic.

Pure Python throughout: the profile is evaluated on scalars once per layer, so
it costs ZERO flopscope operations. That property is what lets the fitted layer
be free at runtime.
"""
from __future__ import annotations

_H = 1.0 / 3.0  # four knots at t = 0, 1/3, 2/3, 1


def _end_slope(d_near, d_far):
    """Shape-preserving one-sided slope (Fritsch-Carlson end condition)."""
    m = (3.0 * d_near - d_far) / 2.0
    if m * d_near <= 0.0:
        return 0.0
    if d_near * d_far <= 0.0 and abs(m) > abs(3.0 * d_near):
        return 3.0 * d_near
    return m


def pchip4(knots, t):
    """Monotone cubic through 4 knots at t = 0, 1/3, 2/3, 1. `t` clamped to [0,1]."""
    y0, y1, y2, y3 = knots
    # EXACT constant reproduction, checked before any arithmetic can round it.
    if y0 == y1 == y2 == y3:
        return y0
    if t <= 0.0:
        return y0
    if t >= 1.0:
        return y3
    d = ((y1 - y0) / _H, (y2 - y1) / _H, (y3 - y2) / _H)
    m = [0.0, 0.0, 0.0, 0.0]
    for k in (1, 2):
        a, b = d[k - 1], d[k]
        if a * b <= 0.0:
            m[k] = 0.0
        else:
            # Weighted harmonic mean; weights are equal for uniform spacing.
            m[k] = 2.0 / (1.0 / a + 1.0 / b)
    m[0] = _end_slope(d[0], d[1])
    m[3] = _end_slope(d[2], d[1])

    seg = 0 if t < _H else (1 if t < 2.0 * _H else 2)
    s = (t - seg * _H) / _H
    yk, yk1, mk, mk1 = knots[seg], knots[seg + 1], m[seg], m[seg + 1]
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * yk + h10 * _H * mk + h01 * yk1 + h11 * _H * mk1


def layer_value(knots, i, depth):
    """Profile value at layer `i` of `depth`."""
    t = 0.0 if depth <= 1 else float(i) / float(depth - 1)
    return pchip4(knots, t)


def main() -> int:
    ok = True

    # 1. constant reproduction, exactly
    for c in (0.0, 1.0, 0.625, 1.225, -0.1, 0.00968143045664494):
        for i in range(16):
            v = layer_value([c] * 4, i, 16)
            if v != c:
                print(f"FAIL constant {c!r} at layer {i}: got {v!r}")
                ok = False
    print("ok  constant reproduction is EXACT (== , not approx)")

    # 2. endpoints
    k = [0.2, 0.5, 0.9, 1.4]
    assert layer_value(k, 0, 16) == k[0], layer_value(k, 0, 16)
    assert layer_value(k, 15, 16) == k[3], layer_value(k, 15, 16)
    print("ok  endpoints interpolate the outer knots exactly")

    # 3. monotone preservation and no overshoot
    for k in ([0.0, 0.1, 0.9, 1.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 2.0]):
        vals = [layer_value(k, i, 16) for i in range(16)]
        lo, hi = min(k), max(k)
        if min(vals) < lo - 1e-12 or max(vals) > hi + 1e-12:
            print(f"FAIL overshoot for {k}: range {min(vals):.4f}..{max(vals):.4f}")
            ok = False
        inc = all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
        dec = all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))
        mono_knots = (all(b >= a for a, b in zip(k, k[1:]))
                      or all(b <= a for a, b in zip(k, k[1:])))
        if mono_knots and not (inc or dec):
            print(f"FAIL monotonicity lost for {k}")
            ok = False
    print("ok  monotone knots give a monotone, non-overshooting profile")

    # 4. a realistic lambda profile against the frozen table
    lam = [0.00968143045664494, 0.009142257779618279, 0.009841284427774549,
           0.009301796540053444, 0.009449170133567886, 0.009167097043377889,
           0.0076572907850084955, 0.0069621042106064275, 0.0054750830927211116,
           0.004574739646806055, 0.005647050489044209, 0.005575418293711488,
           0.005672301130570071, 0.005715207829079359, 0.007679451606169168, 0.0]
    knots = [lam[0], lam[5], lam[10], lam[15]]
    fit = [layer_value(knots, i, 16) for i in range(16)]
    err = max(abs(a - b) for a, b in zip(lam, fit))
    print(f"ok  4 knots through the frozen _K4_LAM table: max |error| {err:.3e} "
          f"(table range {min(lam):.3e}..{max(lam):.3e})")
    print(f"    -- the table is NOT smooth, so 4 knots cannot reproduce it; the")
    print(f"       default path must keep the table itself, not a spline of it.")

    print("\nALL CHECKS PASS" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
