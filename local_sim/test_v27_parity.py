"""v2.7 PARITY — the calibration must touch the Phase-2 analytic path and NOTHING else.

v2.7 = v2.6.1 plus a per-row output rescale applied inside `_analytic_only`. That
method is reached only when the sampling stack declines the network — in practice
`width > _SOBOL_DIM`, which is the Phase-2 shape. Everywhere else v2.7 must be
v2.6.1 **bit for bit**, because the two execute the identical program on identical
inputs; equality is achievable, so equality is demanded (the standard
`local_sim/test_shape_parity.py` already sets for v2.6 vs v2.4.1).

Three things are checked, and each has a way to fail:

  T1  On Phase-1-shaped networks (256x32) and small shapes, v2.7 == v2.6.1
      EXACTLY, and the counted FLOPs are equal too. A calibration that leaks
      into the sampling path would break this.
  T2  On the Phase-2 shape (1024x16), v2.7 == v2.6.1 * _CAL_G row-wise, to
      float64 rounding. A table that is misindexed, transposed, or silently
      broadcast would break this.
  T3  At a depth the table does not cover, v2.7 == v2.6.1 EXACTLY -- the
      uncalibrated rows, not a truncated or recycled table. This is the
      shape-generic requirement (AGENTS.md hard rule 7), and it is the clause a
      careless `_CAL_G[:depth]` would silently violate.

Run: uv run python local_sim/test_v27_parity.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim ships shims NAMED flopscope/whestbench: APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import flopscope as flops  # noqa: E402
import whestbench as wb  # noqa: E402

from mint_provenance import canonical_mint  # noqa: E402

_FAILED = []

# --------------------------------------------------------------------------
# Seeds: DELIBERATELY OUTSIDE EVERY REGISTERED BLOCK.
#
# The first version of this file typed four seeds drawn from D19 -- a LOCKED
# SINGLE_USE *decision* block -- and four from the P01 DESIGN block. (Their exact
# values are deliberately NOT reproduced here: `scan_sources` reads comments as
# well as code, so quoting them would re-trip the very check that caught them.
# They are in the LOG entry and in git history.)
# An independent Skeptic caught it: `local_sim/test_seed_ledger.py`
# dropped to 59/60 on `A3: the tree has no unrecorded seed use`, which turns
# `scripts/preflight.ps1` step 7 RED, and D19's entire value is that it has
# never been drawn.
#
# The deeper error was conceptual. This file performs a BITWISE PARITY check
# between two versions of one program. It needs arbitrary deterministic inputs,
# not experimental provenance: no truth is measured, no decision is made, no
# statistic is computed. Spending block seeds on it buys nothing and burns
# something. These values lie outside every range in `seed_blocks.py`, which is
# what makes them free to type as literals -- and this comment is here so that a
# future reader does not "tidy" them back into a block.
# --------------------------------------------------------------------------
_S = (90000000001, 90000000002, 90000000003, 90000000004,
      90000000005, 90000000006, 90000000007, 90000000008)



def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<58}{detail}")
    if not ok:
        _FAILED.append(name)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(mod, width, depth, seed):
    w, est_seed, _ = canonical_mint(width, depth, seed)
    mlp = wb.MLP(width=width, depth=depth,
                 weights=[np.asarray(x) for x in w], seed=est_seed)
    est = mod.Estimator()
    est.setup(None)
    with flops.BudgetContext(flop_budget=2 ** 41, quiet=True) as ctx:
        out = np.asarray(est.predict(mlp, 2 ** 41))
    return out, int(ctx.flops_used)


def main():
    old = _load("estimator_v2_6_1.py", "_v261")
    new = _load("estimator_v2_7.py", "_v27")
    G = np.array(new._CAL_G, dtype=np.float64)
    print(f"v2.7 parity vs v2.6.1   |_CAL_G| = {len(G)}   G[0] = {G[0]!r}\n")

    check("G[0] is EXACTLY 1.0 (layer 1's mean is exact in closed form)",
          G[0] == 1.0, f"{G[0]!r}")
    check("every constant lies in (0.99, 1.0]", bool(((G > 0.99) & (G <= 1.0)).all()),
          f"[{G.min():.9f}, {G.max():.9f}]")
    check("the table is non-increasing in depth (compounding bias)",
          bool((np.diff(G) <= 0).all()), "monotone")

    print("\nT1  off the analytic path, v2.7 must be v2.6.1 bit for bit")
    for width, depth, seed in ((256, 32, _S[0]), (256, 8, _S[1]),
                               (64, 16, _S[2]), (128, 4, _S[3])):
        a, fa = _run(old, width, depth, seed)
        b, fb = _run(new, width, depth, seed)
        same = a.shape == b.shape and np.array_equal(a, b)
        check(f"{width}x{depth}: identical output", same,
              "bitwise equal" if same else f"max |diff| {np.max(np.abs(a-b)):.3e}")
        check(f"{width}x{depth}: identical counted FLOPs", fa == fb, f"{fa:,} == {fb:,}")

    print("\nT2  on the Phase-2 shape, v2.7 == v2.6.1 * _CAL_G, row-wise")
    for seed in (_S[4], _S[5]):
        a, fa = _run(old, 1024, 16, seed)
        b, fb = _run(new, 1024, 16, seed)
        want = a * G[:, None]
        dev = float(np.max(np.abs(b - want)))
        check(f"seed {seed}: rows match the table", dev <= 1e-15,
              f"max |diff| {dev:.2e}")
        check(f"seed {seed}: all finite", bool(np.isfinite(b).all()), "finite")
        check(f"seed {seed}: the rescale is the ONLY cost added", fb - fa == 32832,
              f"delta {fb-fa:,} FLOPs (f64 billing: 16,384 elements at 2x, plus the table)")

    print("\nT3  at an uncovered depth the rows must be UNCALIBRATED, not truncated")
    for width, depth, seed in ((1024, 8, _S[6]), (1024, 20, _S[7])):
        a, _ = _run(old, width, depth, seed)
        b, _ = _run(new, width, depth, seed)
        same = np.array_equal(a, b)
        check(f"{width}x{depth} (depth != {len(G)}): identical output", same,
              "bitwise equal -- no table applied"
              if same else f"max |diff| {np.max(np.abs(a-b)):.3e}")
        # the negative control: a truncated table WOULD have changed these rows,
        # so this check can actually fail rather than being vacuously true.
        would = float(np.max(np.abs(a * G[:depth, None] - a))) if depth <= len(G) else None
        if would is not None:
            check(f"{width}x{depth}: and a truncated table would NOT have been a no-op",
                  would > 1e-12, f"it would have moved rows by {would:.3e}")

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)}: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("v2.7 parity: ALL CHECKS PASS")


if __name__ == "__main__":
    main()
