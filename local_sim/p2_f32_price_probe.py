"""Is the repository's "float32 K2 saves only 3.1%" measurement an artifact?

## The claim under test

`docs/CURRENT_STATE.md` records, as the reason coupled residual sampling was
deprioritised, that a float32 K2 arm measured only a **3.1%** counted-cost
reduction rather than the ~50% a float64 -> float32 change should buy, since
flopscope bills float64 at twice the float32 rate.

## The mechanism this probe proposes instead

flopscope 0.12 promotes `stats.norm.cdf` / `stats.norm.pdf` inputs from float32
to float64 "to match scipy.stats", and warns about it. `local_sim/p2_arms/
arm_k2p1_f32.py` calls both and never casts the results back. `Phi` and `phi` are
therefore float64 from layer 0 onward, and every downstream expression that
touches them -- the mean update, the second-moment update, and the covariance
accumulator `outer(Phi, Phi) * cov_pre * off`, which is where nearly all the cost
lives -- is float64 too. The arm is float32 in name only.

If that is the mechanism, casting the two stats results straight back to float32
recovers the missing saving, and the deprioritisation was made on a measurement of
a mislabelled arm.

## What this probe does NOT claim

Only counted cost. A float32 closure is a different numerical object and its
accuracy must be measured separately against truth before anything is built on
it; `--accuracy` reports the mean-square divergence from the float64 control on
the scored row so the cost saving is never quoted alone.

Nothing here ships and `estimator.py` is untouched.

Run:
    uv run python local_sim/p2_f32_price_probe.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# local_sim holds shims named `flopscope` and `whestbench`; APPEND, never insert.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_ROOT, _HERE]

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
import whestbench as wb  # noqa: E402

from mint_provenance import canonical_mint  # noqa: E402

_INV_SQRT_2PI = 0.3989422804014327


def _closure(w, width, depth, dtype, cast_back):
    """The k2p1 closure, parameterised by dtype and by whether stats output is cast back."""
    mu = fnp.zeros(width, dtype=dtype)
    cov = fnp.eye(width, dtype=dtype)
    off = fnp.ones((width, width), dtype=dtype) - fnp.eye(width, dtype=dtype)
    rows = []
    for k in range(depth):
        wk = w[k]
        mu_pre = wk.T @ mu
        cov_pre = wk.T @ cov @ wk
        var_pre = fnp.maximum(fnp.diagonal(cov_pre), 1e-12)
        sig = fnp.sqrt(var_pre)
        alpha = mu_pre / sig
        Phi = flops.stats.norm.cdf(alpha)
        phi = flops.stats.norm.pdf(alpha)
        if cast_back:
            # THE one-line intervention. flopscope 0.12 promotes stats output to
            # float64; without this the arm is float32 in name only.
            Phi = Phi.astype(dtype)
            phi = phi.astype(dtype)
        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var = fnp.maximum(ez2 - mu * mu, 0.0)
        acc = fnp.outer(Phi, Phi) * cov_pre * off
        cov = acc + fnp.diag(var)
        rows.append(mu)
    return fnp.stack(rows, axis=0)


def run(mlp, dtype, cast_back):
    width, depth = mlp.width, len(mlp.weights)
    with flops.BudgetContext(flop_budget=2 ** 41, quiet=True) as ctx:
        w = [x.astype(dtype) for x in mlp.weights]
        rows = _closure(w, width, depth, dtype, cast_back)
        var0 = (w[0] * w[0]).sum(axis=0, dtype=fnp.float64)
        exact = fnp.sqrt(var0) * _INV_SQRT_2PI
        out = fnp.concatenate(
            [fnp.reshape(exact.astype(rows.dtype), (1, width)), rows[1:]], axis=0
        )
        arr = np.asarray(out)
    return int(ctx.flops_used), arr, str(rows.dtype)


def main():
    import warnings
    warnings.filterwarnings("ignore")

    seed = 40821001000
    w, est_seed, _ = canonical_mint(1024, 16, seed)
    mlp = wb.MLP(width=1024, depth=16, weights=[np.asarray(x) for x in w], seed=est_seed)

    cases = [
        ("f64 (the shipped spelling)", fnp.float64, False),
        ("f32, stats NOT cast back (arm_k2p1_f32 as written)", fnp.float32, False),
        ("f32, stats cast back to f32", fnp.float32, True),
    ]
    res, base_arr, base_f = [], None, None
    B = 2 ** 41
    print(f"seed {seed}   1024 x 16   budget B = {B:,}\n")
    print(f"{'arm':<52}{'counted FLOPs':>18}{'F/B':>10}{'mult':>7}  state dtype")
    for name, dt, cb in cases:
        f, arr, sdt = run(mlp, dt, cb)
        if base_arr is None:
            base_arr, base_f = arr, f
        rel = float(np.mean((arr[-1] - base_arr[-1]) ** 2) / np.mean(base_arr[-1] ** 2))
        res.append({"arm": name, "flops": f, "f_over_b": f / B,
                    "multiplier": max(0.1, f / B), "state_dtype": sdt,
                    "rel_ms_vs_f64_final_layer": rel,
                    "cost_ratio_vs_f64": f / base_f})
        print(f"{name:<52}{f:>18,}{f/B:>10.5f}{max(0.1, f/B):>7.3f}  {sdt}")

    print("\ncost relative to the float64 control, and divergence on the scored row:")
    for r in res:
        print(f"  {r['arm']:<52}{r['cost_ratio_vs_f64']:>8.4f}x   "
              f"rel mean-square {r['rel_ms_vs_f64_final_layer']:.3e}")

    free = int(0.1 * B)
    print(f"\nThe multiplier floor buys {free:,} counted FLOPs for free.")
    for r in res:
        print(f"  {r['arm']:<52} leaves {free - r['flops']:>16,} free")

    out = os.path.join(_ROOT, "experiments", "2026-08-26_p2_cal", "f32_price_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"seed": seed, "budget": B, "free_flops_at_floor": free,
                   "cases": res}, fh, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
