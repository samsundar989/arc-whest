"""Label-free gates for the step-3 shared old-lineage coordinates. Thresholds fixed here.

  1. NESTING: v36_v32_off == v32 bit for bit, output and counted FLOPs (64x5, 96x6).
  2. FULL-RANK IDENTITY (f64 layer loop, real tiny chains, every row): forced
     rank = n, ages 2 and 3 (2 passes) on the dense base == v32_dense, rel <= 1e-9,
     AND joins actually happened (old-lineage count > 0).
  3. APPROXIMATE PATH RUNS (temp arms, rank 16 at n = 64, age 3): joins happen, output
     finite and non-negative; relative change vs exact printed (no gate -- accuracy is
     decided on truth, not here).
  4. EQUIVARIANCE of the approximate path (96x6, rank 24, age 3, layer 1 relabelled):
     <= 100x v32's own float32 floor.
  5. The budget guard still over-predicts measured FLOPs (96x6, rank 24 arm).
"""
from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

OK = []


def claim(name, ok, detail=""):
    OK.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<64} {detail}")


def rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300))


def count_joins(mod):
    box = {"joins": 0, "max_ko": 0}
    orig = mod._shared_join

    def wrapped(tw):
        out = orig(tw)
        if out is not tw:
            box["joins"] += 1
        box["max_ko"] = max(box["max_ko"], out.ko)
        return out
    mod._shared_join = wrapped
    return box


def main():
    import flopscope as flops
    import k4feed_oracle as K
    import s1_parity_v32 as P1
    import s6_parity as P6
    import s7_build_shared as S
    from mint_provenance import canonical_mint

    flops.configure(symmetry_warnings=False)
    L = lambda name: K._load(os.path.join(_ROOT, name))
    m32 = L("estimator_k3_v32.py")
    m32d = L("estimator_k3_v32_dense.py")

    print("\n1. nesting")
    moff = L("estimator_k3_v36_v32_off.py")
    for wd, dp, sd in ((64, 5, 90000000801), (96, 6, 90000000802)):
        w, es, _ = canonical_mint(wd, dp, sd)
        pb, fb = P1._predict(m32, w, wd, dp, es)
        po, fo = P1._predict(moff, w, wd, dp, es)
        claim(f"{wd}x{dp} v36_off == v32 bitwise + FLOPs", np.array_equal(pb, po) and fb == fo)

    print("\n2. full-rank identity (forced rank n) vs v32_dense, f64")
    for tag in ("full_a2", "full_a3"):
        for wd, dp, sd in ((24, 6, 90000000803), (32, 7, 90000000804)):
            mod = L(f"estimator_k3_v36_v32_dense_{tag}.py")
            box = count_joins(mod)
            w, es, _ = canonical_mint(wd, dp, sd)
            r_ref = P6.layer_loop(m32d, w, wd, dp)
            r_sh = P6.layer_loop(mod, w, wd, dp)
            claim(f"{tag} {wd}x{dp} == v32_dense (all rows)", rel(r_sh, r_ref) <= 1e-9 and box["joins"] > 0,
                  "rel %.1e  joins %d  max old %d" % (rel(r_sh, r_ref), box["joins"], box["max_ko"]))

    tmp = tempfile.mkdtemp(prefix="s7_")
    src = open(os.path.join(_ROOT, "estimator_k3_v32.py"), encoding="utf-8").read()

    def temp_arm(age, rank, passes=1):
        p = os.path.join(tmp, f"v36_a{age}r{rank}.py")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(S.build(src, age, rank, passes, False, True))
        return K._load(p)

    print("\n3. approximate path runs")
    wd, dp = 64, 7
    w, es, _ = canonical_mint(wd, dp, 90000000805)
    pb, fb = P1._predict(m32, w, wd, dp, es)
    mod = temp_arm(3, 16)
    box = count_joins(mod)
    pa, fa = P1._predict(mod, w, wd, dp, es)
    claim("64x7 rank16 age3: joins happen", box["joins"] > 0, "joins %d max old %d" % (box["joins"], box["max_ko"]))
    claim("64x7 rank16 age3: finite, non-negative", bool(np.isfinite(pa).all() and (pa >= 0).all()),
          "final-row rel change vs exact %.2e, FLOPs x%.4f" % (rel(pa[-1], pb[-1]), fa / fb))

    print("\n4-5. equivariance and guard, 96x6 rank 24 age 3")
    wd, dp = 96, 6
    w, es, _ = canonical_mint(wd, dp, 90000000504)
    lay = 1
    perm = np.random.default_rng(20260916).permutation(wd)
    wp = [np.array(x, copy=True) for x in w]
    wp[lay] = wp[lay][:, perm]
    wp[lay + 1] = wp[lay + 1][perm, :]
    floors = {}
    for name, m in (("v32", m32), ("shared", temp_arm(3, 24))):
        p, f = P1._predict(m, w, wd, dp, es)
        pout, _ = P1._predict(m, wp, wd, dp, es)
        moved = p.copy()
        moved[lay] = p[lay][perm]
        floors[name] = float(np.mean((moved - pout) ** 2)) / max(float(np.mean(p ** 2)), 1e-300)
        if name == "shared":
            g = m._projected_flops(wd, dp, m._fit_generations(wd, dp, 2 ** 41))
            claim("96x6 guard over-predicts", g >= f, "guard/meas %.3f" % (g / f))
    claim("96x6 equivariance <= 100x v32 floor", floors["shared"] <= 100 * max(floors["v32"], 1e-300),
          "v32 %.2e shared %.2e" % (floors["v32"], floors["shared"]))
    print(f"\n{sum(OK)}/{len(OK)} claims passed")
    return 0 if all(OK) else 1


if __name__ == "__main__":
    sys.exit(main())
