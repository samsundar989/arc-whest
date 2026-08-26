"""P2-CAL, part 2: the corrections that do NOT commute with the closure.

## Why this file exists

P2-CAL's first pass measured `C0`, `C1`, `C2` and `P2` and found the last three
**bit-equivalent** -- 12 significant figures on the scored row. That is not a bug.
The K2 closure is **exactly positively homogeneous of degree 1** in its state:
scaling `(mu, cov) -> (g*mu, g^2*cov)` leaves `alpha = mu_pre/sigma` unchanged,
hence leaves `Phi` and `phi` unchanged, hence multiplies every downstream mean by
exactly `g`. Verified directly at `1.6e-30` relative mean square.

So the pre-registered mechanism prediction -- that applying the gain *in the loop*
beats applying it *after* -- is **REFUTED**, and refuted for a reason that is
structural rather than empirical. Every purely multiplicative state correction,
however many parameters it is given, is a single output rescale in disguise.

That is a strong constraint on the whole design space, and it says exactly where
the remaining power is: **corrections that change `alpha`**. Those do not commute.
Two of them are measured here.

| arm | correction | changes `alpha`? | registered? |
|---|---|---|---|
| `C3` | per-layer gain on the post-ReLU **variance** (covariance diagonal) | yes | yes -- in the 2026-08-26 contract |
| `C4` | per-layer gain on the post-ReLU **off-diagonal** covariance | yes | **NO -- exploratory, added after seeing C0-C2** |

`C4` is declared unregistered on purpose. It is the correction the mechanism
points at -- the off-diagonal is where the Price-order-1 truncation actually lives,
and it is the one part of layer 0's state that is *not* exact -- but it was chosen
after seeing results, so it cannot bear the contract's continuation gate. It is
reported to size the next contract, and it is labelled everywhere it appears.

## Fitting

Sequential, one layer at a time, on the fit half only, by a 1-D scan against the
**final-layer** mean-square error -- the scored quantity. Layer 0 is included for
both arms, unlike the mean gain: layer 0's post-ReLU *mean* is exact so scaling it
can only hurt, but its *covariance* is not, so there is something there to fix.

Run (GPU venv):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_cal_nc.py \\
        --out experiments/2026-08-26_p2_cal/cal_nc.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.append(_p)

from p2_cal_gpu import (  # noqa: E402
    TRUTH, run_all, stage_weights, xmse_final, per_mlp_xmse_final, paired_t,
)


def fit_gain_sequence(Wmm, dev, fit_idx, target_final, kind, base_mean,
                      depth=16, chunk=16, grid=None, base_var=None, base_off=None):
    """Sequential 1-D scan of a per-layer gain, minimising FINAL-layer MSE.

    `kind` is 'var' or 'off'. Every evaluation runs the full closure, so the cost
    is `depth * len(grid)` passes over the fit half.
    """
    if grid is None:
        grid = np.array([0.94, 0.97, 0.99, 1.0, 1.01, 1.03, 1.06])
    g = np.ones(depth) if base_var is None else np.array(base_var)
    o = np.ones(depth) if base_off is None else np.array(base_off)
    cur = g if kind == "var" else o
    n_fit = len(fit_idx)
    for k in range(depth):
        best, bv = cur[k], np.inf
        for cand in grid:
            trial = cur.copy()
            trial[k] = cand
            kw = {"mean_gain": base_mean,
                  "var_gain": trial if kind == "var" else g,
                  "offdiag_gain": o if kind == "var" else trial}
            P = run_all(Wmm, dev, n_fit, chunk=chunk, **kw)
            v = float(np.mean((P[fit_idx, -1].numpy() - target_final) ** 2))
            if v < bv:
                bv, best = v, float(cand)
        cur[k] = best
        print(f"  {kind} layer {k:>2}  g = {best:.4f}   fit-set final MSE {bv:.4e}", flush=True)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(TRUTH, allow_pickle=True)
    n = args.n
    seeds = z["seeds"][:n]
    y1 = z["means_r1"][:n].astype(np.float64)
    y2 = z["means_r2"][:n].astype(np.float64)
    T = 0.5 * (y1 + y2)
    Wmm = stage_weights(seeds)

    prev = json.load(open(os.path.join(_ROOT, "experiments", "2026-08-26_p2_cal", "cal.json")))
    mean_gain = np.array(prev["arms"]["C2"]["layer_gains"])

    nfit = int(args.fit_frac * n)
    fit_idx = np.arange(0, nfit)
    hold_idx = np.arange(nfit, n)
    print(f"P2-CAL/nc  {n} networks, fit {nfit}, held out {n-nfit}, device={dev}", flush=True)
    print(f"mean gains carried over from C2: {np.round(mean_gain,6).tolist()}\n", flush=True)

    res = {"n": n, "n_fit": nfit,
           "note": "C3 is registered; C4 is EXPLORATORY and was chosen after seeing C0-C2",
           "mean_gain_from_C2": mean_gain.tolist(), "arms": {}}
    t0 = time.time()

    def record(name, P, extra=None):
        r = {"xmse_fit": xmse_final(P, y1, y2, fit_idx),
             "xmse_hold": xmse_final(P, y1, y2, hold_idx),
             "per_mlp_xmse_hold": per_mlp_xmse_final(P, y1, y2, hold_idx)}
        if extra:
            r.update(extra)
        res["arms"][name] = r
        print(f"  {name:<4} fit {r['xmse_fit']:.4e}  held-out {r['xmse_hold']:.4e}", flush=True)
        return r

    print("C2 baseline (mean gain only) ...", flush=True)
    P2c = run_all(Wmm, dev, n, chunk=args.chunk, mean_gain=mean_gain)
    record("C2", P2c)

    print("\nC3  per-layer VARIANCE gain (registered arm) ...", flush=True)
    vg = fit_gain_sequence(Wmm, dev, fit_idx, T[fit_idx, -1], "var", mean_gain, chunk=args.chunk)
    P3 = run_all(Wmm, dev, n, chunk=args.chunk, mean_gain=mean_gain, var_gain=vg)
    record("C3", P3, {"var_gains": [float(x) for x in vg]})

    print("\nC4  per-layer OFF-DIAGONAL gain (EXPLORATORY, unregistered) ...", flush=True)
    og = fit_gain_sequence(Wmm, dev, fit_idx, T[fit_idx, -1], "off", mean_gain, chunk=args.chunk)
    P4 = run_all(Wmm, dev, n, chunk=args.chunk, mean_gain=mean_gain, offdiag_gain=og)
    record("C4", P4, {"offdiag_gains": [float(x) for x in og]})

    c0 = float(prev["arms"]["C0"]["xmse_hold"])
    c0pm = prev["arms"]["C0"]["per_mlp_xmse_hold"]
    print(f"\nheld-out, versus the UNCALIBRATED control C0 = {c0:.4e}:")
    for k in ("C2", "C3", "C4"):
        r = res["arms"][k]
        pt = paired_t(c0pm, r["per_mlp_xmse_hold"])
        r["paired_vs_C0"] = pt
        print(f"  {k}: {c0/r['xmse_hold']:.3f}x   paired diff {pt['mean_diff']:.3e} "
              f"95% [{pt['lo']:.3e}, {pt['hi']:.3e}]")
    print(f"\nGT noise floor mean {float(z['gt_noise'][:n].mean()):.4e}")
    print(f"total {time.time()-t0:.0f}s")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
