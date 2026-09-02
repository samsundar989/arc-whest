"""WP9 gate clause G2 — what does the corrector actually COST under flopscope?

Contract: `experiments/LOG.md`, *CONTRACT WP9* (2026-09-01), commit `9c961ce`.
Its prediction **P6** commits to `<= 5e8` counted FLOPs for the corrector, and
its gate clause **G2** requires the whole corrected estimator, including the
grading server's `+2**24` per-MLP offset, to stay at or below `2.199e11` so the
multiplier remains pinned at the `0.1` floor.

WHY THIS EXISTS SEPARATELY FROM THE DESIGN STUDY
------------------------------------------------
`p2_corrector.py` fits and scores in torch, whose meter is not flopscope's.
This repository has already had to retract a `5.49x` "affordability target"
that compared two different meters, and `AGENTS.md` records the rule that came
out of it: **only an `fnp` spelling billed by flopscope is a price.** So the
corrector's cost is measured here, in the grader's own meter, on the grader's
own shape, against the frozen K2p1 arm as the control.

The weights are ARBITRARY. Cost under flopscope is a function of shapes and
dtypes, not of values, and fitting has to happen before real weights exist. A
number produced here is a **price**, never an accuracy claim.

FLOAT32 DISCIPLINE
------------------
`docs/GOTCHAS.md`: `flops.stats.norm.cdf` and `.pdf` silently PROMOTE float32
to float64 and emit a `FlopscopeWarning` saying so. An arm that does not cast
back is float32 in name only -- the recorded `3.1%` saving that turned out to
be `50%` once two `.astype` calls were added. Both spellings here cast back
immediately, so the comparison is like for like.

Run:
    uv run python local_sim/p2_corrector_price.py
    uv run python local_sim/p2_corrector_price.py --dtype float32 --json <out>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import numpy as np  # noqa: E402

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

WIDTH, DEPTH = 1024, 16
BUDGET = 2 ** 41
FLOOR_FLOPS = int(0.1 * BUDGET)
#: WP6's conservative upper bound on the grading server's unexplained per-MLP
#: additive offset. `official_config` carries the provenance; it is repeated as
#: a literal here only so this module prices without importing the round.
SERVER_OFFSET = 2 ** 24

N_FEATURES, HIDDEN = 8, 32
_INV_SQRT_2PI = 0.3989422804014327


def _weights(seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2.0 / WIDTH)
    return [fnp.array((rng.standard_normal((WIDTH, WIDTH)) * scale).astype(dtype))
            for _ in range(DEPTH)]


def _corr_params(seed=1, dtype=np.float64):
    rng = np.random.default_rng(seed)
    return (fnp.array((rng.standard_normal((N_FEATURES, HIDDEN))
                       * 0.3).astype(dtype)),
            fnp.array(np.zeros(HIDDEN, dtype=dtype)),
            fnp.array((rng.standard_normal((HIDDEN, 2)) * 0.01).astype(dtype)),
            fnp.array(np.zeros(2, dtype=dtype)))


def closure(w, dtype, corrector=None, mode="none"):
    """K2p1, in the shipped operation order, with an optional correction.

    The uncorrected path is `local_sim/p2_arms/arm_k2p1.py` line for line. The
    corrected path adds ONLY the feature block and three matmuls, so the
    difference the meter reports is the corrector and nothing else.
    """
    dt = dtype
    width = WIDTH
    mu = fnp.zeros(width, dtype=dt)
    cov = fnp.eye(width, dtype=dt)
    off = fnp.ones((width, width), dtype=dt) - fnp.eye(width, dtype=dt)
    rows = []
    for k in range(len(w)):
        wk = w[k]
        mu_pre = wk.T @ mu
        cov_pre = wk.T @ cov @ wk
        var_pre = fnp.maximum(fnp.diagonal(cov_pre), 1e-12)
        sig = fnp.sqrt(var_pre)
        alpha = mu_pre / sig
        # GOTCHAS: these two PROMOTE to float64 and warn. Cast back at once, or
        # the whole downstream covariance accumulator silently runs at f64 and
        # the arm is float32 in name only.
        Phi = flops.stats.norm.cdf(alpha).astype(dt)
        phi = flops.stats.norm.pdf(alpha).astype(dt)

        mu_raw = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var_raw = fnp.maximum(ez2 - mu_raw * mu_raw, 0.0)

        if corrector is not None and k > 0 and (
                mode == "full" or (mode == "last" and k == len(w) - 1)):
            W1, b1, W2, b2 = corrector
            a = fnp.clip(alpha, -8.0, 8.0)
            m = fnp.mean(a)
            s = fnp.sqrt(fnp.mean((a - m) * (a - m))) + 1e-8
            mp = fnp.mean(Phi)
            lay = fnp.full(width, k / (len(w) - 1.0), dtype=dt)
            f = fnp.stack([a, Phi, phi, (a - m) / s, lay,
                           fnp.full(width, 0.0, dtype=dt) + m,
                           fnp.full(width, 0.0, dtype=dt) + s,
                           fnp.full(width, 0.0, dtype=dt) + mp], axis=1)
            h = fnp.tanh(f @ W1 + b1)
            c = h @ W2 + b2
            cm = fnp.clip(c[:, 0], -0.25, 0.25)
            cv = fnp.clip(c[:, 1], -0.5, 0.5)
            mu_new = mu_raw + sig * cm
            var_new = fnp.maximum(var_raw * (1.0 + cv), 0.0)
        else:
            mu_new, var_new = mu_raw, var_raw

        cov = fnp.outer(Phi, Phi) * cov_pre * off + fnp.diag(var_new)
        mu = mu_new
        rows.append(mu)
    return fnp.stack(rows, axis=0)


def price(mode, dtype, seed=0):
    w = _weights(seed, dtype)
    corr = _corr_params(seed + 1, dtype) if mode != "none" else None
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
        out = closure(w, dtype, corr, mode)
        # Touch the result so nothing can be elided. Reading a scalar OUT of
        # the budget context would be a materialisation; `shape` is metadata.
        assert out.shape == (DEPTH, WIDTH)
        total = float(ctx.flops_used)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dtype", default="float64",
                    choices=("float64", "float32"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    dt = np.float64 if a.dtype == "float64" else np.float32

    print("=" * 78)
    print("WP9 G2 -- the corrector's price in the GRADER'S meter (flopscope)")
    print("=" * 78)
    print(f"  shape {WIDTH} x {DEPTH}   dtype {a.dtype}   budget {BUDGET:,}")
    print(f"  free zone at the 0.1 floor: {FLOOR_FLOPS:,} FLOPs")
    print()

    base = price("none", dt)
    last = price("last", dt)
    full = price("full", dt)
    d_last, d_full = last - base, full - base

    rows = [("K2p1, uncorrected (arm B0)", base, 0.0),
            ("+ corrector at the FINAL layer (arm A)", last, d_last),
            ("+ corrector at EVERY layer (arm B)", full, d_full)]
    print(f"  {'arm':<40}{'counted FLOPs':>18}{'delta':>14}")
    for name, tot, d in rows:
        print(f"  {name:<40}{tot:>18,.0f}{d:>14,.0f}")

    graded = full + SERVER_OFFSET
    cb = graded / BUDGET
    print()
    print(f"  arm B at the grader's meter (+2**24 offset): {graded:,.0f}")
    print(f"  C/B = {cb:.7f}   multiplier = {max(0.1, cb):.4f}")
    print(f"  free-zone headroom remaining: {FLOOR_FLOPS - graded:,.0f} FLOPs "
          f"({100 * (FLOOR_FLOPS - graded) / FLOOR_FLOPS:.2f}%)")
    print()

    # The two claims the contract made before any of this existed.
    p6 = d_full <= 5e8
    g2 = graded <= FLOOR_FLOPS
    print(f"  P6  corrector costs <= 5e8 counted FLOPs        "
          f"{'HELD' if p6 else 'REFUTED'}   ({d_full:,.0f})")
    print(f"  G2  multiplier stays pinned at the 0.1 floor    "
          f"{'PASS' if g2 else 'FAIL'}   (C/B {cb:.7f})")
    print(f"      per-layer corrector cost: {d_full / (DEPTH - 1):,.0f} FLOPs "
          f"over {DEPTH - 1} corrected layers")
    print(f"      arm A / arm B cost ratio: "
          f"{(d_full / d_last) if d_last else float('nan'):.2f}x "
          f"(expect ~{DEPTH - 1} -- one layer versus {DEPTH - 1})")

    rep = {"artifact": "wp9_corrector_price", "artifact_version": 1,
           "contract": "experiments/LOG.md CONTRACT WP9, commit 9c961ce",
           "meter": "flopscope (the grader's meter), NOT torch",
           "width": WIDTH, "depth": DEPTH, "dtype": a.dtype,
           "budget": BUDGET, "floor_flops": FLOOR_FLOPS,
           "server_offset_per_mlp": SERVER_OFFSET,
           "n_features": N_FEATURES, "hidden": HIDDEN,
           "flops": {"B0_uncorrected": base, "A_final_layer_only": last,
                     "B_every_layer": full},
           "delta": {"A": d_last, "B": d_full,
                     "B_per_corrected_layer": d_full / (DEPTH - 1)},
           "graded_flops_arm_B": graded, "c_over_b_arm_B": cb,
           "multiplier_arm_B": max(0.1, cb),
           "P6_corrector_under_5e8": bool(p6),
           "G2_multiplier_pinned_at_floor": bool(g2),
           "SCOPE": ("A PRICE, not an accuracy claim. The corrector weights "
                     "here are arbitrary: flopscope bills shapes and dtypes, "
                     "not values. Fitted weights change no counted FLOP.")}
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        print(f"\n  wrote {os.path.relpath(a.json, _ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
