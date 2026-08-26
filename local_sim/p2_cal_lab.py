"""P2-CAL: in-loop per-layer calibration of the K2 state at 1024 x 16.

Contract: `experiments/LOG.md`, entry dated 2026-08-26, committed BEFORE this
file produced any number.

## What this is

A NumPy mirror of `local_sim/p2_arms/arm_k2p1.py` -- written in the same
operation order, so it can be checked bitwise-close against the `fnp` arm -- with
one hook added: the propagated post-ReLU state may be multiplied by a per-layer
scalar *inside* the loop, before it enters the next layer's matmul.

The mirror exists ONLY because fitting needs to run the closure a few hundred
times and the instrumented `fnp` path is far too slow for that. **A mirror that
has not been verified against the instrumented arm produces fiction**, so
`--verify` is not optional decoration: it is the first thing this module does,
and every fitted number in the output artifact carries the verification residual
next to it.

Nothing here is grader-legal and nothing here may ship. `estimator.py` is not
touched. No seed is claimed: the networks come from `truth_P01.npz`, whose seeds
P2-CP1 already spent.

## Why in-loop rather than post-hoc

P2-CP3R measured that 94.61% of K2's error is accumulated mean/variance-state
drift rather than marginal readout error. A drift that compounds through depth is
removable at the layer where it is generated and is NOT removable after 16 layers
of compounding. The `P2` arm here is the post-hoc control that tests exactly
that claim -- it has the same parameter count as `C2` and, if the mechanism story
is right, must lose badly.

## Truth

Two independent 2**22-sample replicas per network. Every MSE is reported with the
GT noise floor beside it, and the headline number is the two-replica cross
product `mean_j (p_j - y1_j)(p_j - y2_j)`, whose expectation is the unbiased MSE
because `E[e1 e2] = 0`.

Run:
    uv run python local_sim/p2_cal_lab.py --verify
    uv run python local_sim/p2_cal_lab.py --out experiments/2026-08-26_p2_cal/cal.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)

TRUTH = os.path.join(_ROOT, "experiments", "2026-08-21_p2_cp1", "truth_P01.npz")

_INV_SQRT_2PI = 0.3989422804014327


# --------------------------------------------------------------------------
# The mirror
# --------------------------------------------------------------------------

def closure(weights, *, mean_gain=None, var_gain=None):
    """K2p1 closure in NumPy, mirroring `arm_k2p1._closure` operation for operation.

    `mean_gain[k]` multiplies the post-ReLU MEAN state produced by layer k before
    it propagates; `var_gain[k]` multiplies the post-ReLU VARIANCE state (the
    covariance diagonal) the same way. Both default to 1.0, which reproduces the
    control exactly.

    Returns `(depth, width)` post-ReLU means, layer 0 first.
    """
    width = weights[0].shape[0]
    depth = len(weights)
    mg = np.ones(depth) if mean_gain is None else np.asarray(mean_gain, dtype=np.float64)
    vg = np.ones(depth) if var_gain is None else np.asarray(var_gain, dtype=np.float64)

    mu = np.zeros(width, dtype=np.float64)
    cov = np.eye(width, dtype=np.float64)
    off = np.ones((width, width), dtype=np.float64) - np.eye(width, dtype=np.float64)
    rows = []
    for k in range(depth):
        wk = weights[k]
        mu_pre = wk.T @ mu
        cov_pre = wk.T @ cov @ wk
        var_pre = np.maximum(np.diagonal(cov_pre), 1e-12)
        sig = np.sqrt(var_pre)
        alpha = mu_pre / sig
        Phi = norm.cdf(alpha)
        phi = norm.pdf(alpha)

        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var = np.maximum(ez2 - mu * mu, 0.0)

        acc = np.outer(Phi, Phi) * cov_pre * off
        cov = acc + np.diag(var)

        # --- the intervention, and the ONLY difference from the control ------
        # Applied AFTER the row is formed for the honest reason that the scored
        # quantity is the post-ReLU mean itself; the gain must therefore correct
        # both what is reported and what propagates, or it is two different
        # models. `rows.append(mu)` below records the corrected mean.
        if mg[k] != 1.0:
            mu = mu * mg[k]
            cov = cov * (mg[k] * mg[k])
        if vg[k] != 1.0:
            d = np.diagonal(cov).copy() * vg[k]
            cov = cov * off + np.diag(d)

        rows.append(mu)
    return np.stack(rows, axis=0)


def predict(weights, *, mean_gain=None, var_gain=None, post_gain=None):
    """Full arm prediction: exact layer 1, then the closure, then optional post-hoc gains."""
    rows = closure(weights, mean_gain=mean_gain, var_gain=var_gain)
    var0 = (weights[0] * weights[0]).sum(axis=0, dtype=np.float64)
    rows = rows.copy()
    rows[0] = np.sqrt(var0) * _INV_SQRT_2PI
    if post_gain is not None:
        rows = rows * np.asarray(post_gain, dtype=np.float64)[:, None]
    return rows


# --------------------------------------------------------------------------
# Verification against the instrumented arm -- REQUIRED before any fit
# --------------------------------------------------------------------------

def verify(seeds, width=1024, depth=16, tol=1e-10):
    """Run the real `fnp` arm and this mirror on the same networks and compare.

    Returns a list of per-network records. The tolerance is on the maximum
    RELATIVE difference over the final layer, which is the scored row.
    """
    from mint_provenance import canonical_mint

    # The arm imports the REAL flopscope/whestbench. `local_sim/` holds shims by
    # those exact names, so it must stay APPENDED and never inserted -- otherwise
    # every "counted FLOP" below becomes fiction (CLAUDE.md, tooling notes).
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE] + [_HERE]
    import whestbench as _wb
    import flopscope as flops
    from p2_arms import arm_k2p1  # noqa: E402

    if "site-packages" not in os.path.abspath(_wb.__file__):
        raise SystemExit(
            "REFUSING to verify against the local_sim shim -- "
            f"whestbench resolved to {_wb.__file__}"
        )

    out = []
    est = arm_k2p1.Estimator()
    est.setup(None)
    for s in seeds:
        w, est_seed, _ = canonical_mint(width, depth, int(s))
        mlp = _wb.MLP(width=width, depth=depth, weights=[np.asarray(x) for x in w], seed=est_seed)
        with flops.BudgetContext(flop_budget=2 ** 41, quiet=True) as ctx:
            got = np.asarray(est.predict(mlp, 2 ** 41))
        mine = predict(w)
        # A PER-ELEMENT relative error is the wrong metric here and the first
        # draft of this check used it. Post-ReLU means at depth 16 span six
        # orders of magnitude within one layer; an entry whose true value is
        # 6e-06 against a layer RMS near 1 shows a "relative error" of order 1
        # for a difference that is irrelevant to every quantity we score. The
        # scored quantity is a MEAN SQUARE over the layer, so the tolerance
        # belongs on the mean square too.
        rel_final = float(
            np.mean((got[-1] - mine[-1]) ** 2) / np.mean(got[-1] ** 2)
        )
        rel_all = float(np.mean((got - mine) ** 2) / np.mean(got ** 2))
        max_abs = float(np.max(np.abs(got[-1] - mine[-1])))
        out.append({
            "seed": int(s),
            "rel_ms_final_layer": rel_final,
            "rel_ms_all_layers": rel_all,
            "max_abs_diff_final_layer": max_abs,
            "counted_flops": int(ctx.flops_used),
            "pass": rel_final <= tol,
        })
    return out


# --------------------------------------------------------------------------
# Truth handling
# --------------------------------------------------------------------------

def load_truth(path=TRUTH):
    z = np.load(path, allow_pickle=True)
    return {
        "seeds": z["seeds"],
        "y1": z["means_r1"].astype(np.float64),
        "y2": z["means_r2"].astype(np.float64),
        "gt_noise": z["gt_noise"],
        "n_samples": int(z["n_samples"]),
    }


def xmse(pred_final, y1_final, y2_final):
    """Two-replica cross-product MSE. Unbiased: E[(p-y1)(p-y2)] = MSE(p)."""
    return float(np.mean((pred_final - y1_final) * (pred_final - y2_final)))


def plain_mse(pred_final, y_final):
    return float(np.mean((pred_final - y_final) ** 2))


# --------------------------------------------------------------------------
# Fits.  Every fit is CLOSED FORM or a 1-D scan -- no optimiser, no seed.
# --------------------------------------------------------------------------

def fit_post_gains(preds, ys):
    """Per-layer post-hoc scalar: the least-squares gain g_l = <p,y>/<p,p> per layer.

    `preds` is (n, depth, width), `ys` is (n, depth, width).
    """
    num = np.einsum("ndw,ndw->d", preds, ys)
    den = np.einsum("ndw,ndw->d", preds, preds)
    return num / np.maximum(den, 1e-300)


def scan_global_gain(weights_list, ys, lo=0.95, hi=1.05, n=41, depth=16):
    """1-D scan of one global in-loop gain applied at every layer."""
    grid = np.linspace(lo, hi, n)
    best, best_g, curve = np.inf, 1.0, []
    for g in grid:
        mg = np.full(depth, g)
        tot, cnt = 0.0, 0
        for w, y in zip(weights_list, ys):
            p = predict(w, mean_gain=mg)
            tot += np.mean((p[-1] - y[-1]) ** 2)
            cnt += 1
        v = tot / cnt
        curve.append([float(g), float(v)])
        if v < best:
            best, best_g = v, float(g)
    return best_g, curve


def fit_layer_gains_coordinate(weights_list, ys, depth=16, rounds=2,
                               lo=0.95, hi=1.05, n=21, init=None):
    """Coordinate scan over per-layer in-loop gains.

    Deliberately a scan and not an optimiser: the objective runs the whole
    closure, gradients are not available, and a scan is reproducible with no
    hidden state. `rounds` passes over the layers, refining a shrinking bracket.
    """
    g = np.ones(depth) if init is None else np.array(init, dtype=np.float64)
    span = (hi - lo) / 2.0
    for r in range(rounds):
        for k in range(depth):
            grid = np.linspace(g[k] - span, g[k] + span, n)
            best, bv = g[k], np.inf
            for cand in grid:
                gg = g.copy()
                gg[k] = cand
                tot = 0.0
                for w, y in zip(weights_list, ys):
                    p = predict(w, mean_gain=gg)
                    tot += np.mean((p[-1] - y[-1]) ** 2)
                if tot < bv:
                    bv, best = tot, float(cand)
            g[k] = best
        span /= 4.0
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="run the fnp arm and compare; always runs before a fit")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mint_provenance import canonical_mint

    t = load_truth()
    seeds = t["seeds"][: args.n]
    print(f"P2-CAL  1024x16   {len(seeds)} networks from truth_P01.npz "
          f"(N={t['n_samples']:,} per replica, two replicas)")

    ver = verify(seeds[:2])
    print("\nMIRROR VERIFICATION against the instrumented fnp arm:")
    for r in ver:
        print(f"  seed {r['seed']:>12}  rel mean-square (final layer) {r['rel_ms_final_layer']:.3e}  "
              f"all layers {r['rel_ms_all_layers']:.3e}  max|diff| {r['max_abs_diff_final_layer']:.3e}  "
              f"{'PASS' if r['pass'] else 'FAIL'}  counted F {r['counted_flops']:,}")
    if not all(r["pass"] for r in ver):
        raise SystemExit("mirror does not reproduce the instrumented arm -- refusing to fit")
    if args.verify:
        return

    print("\nbuilding weights and control predictions ...")
    t0 = time.time()
    W, P0 = [], []
    for i, s in enumerate(seeds):
        w, _, _ = canonical_mint(1024, 16, int(s))
        W.append(w)
        P0.append(predict(w))
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(seeds)}  {time.time()-t0:.0f}s")
    P0 = np.stack(P0)
    y1, y2 = t["y1"][: args.n], t["y2"][: args.n]

    nfit = int(args.fit_frac * len(seeds))
    fit, hold = slice(0, nfit), slice(nfit, len(seeds))
    print(f"\nsplit: fit on {nfit}, held out {len(seeds)-nfit}")

    res = {"n": len(seeds), "n_fit": nfit, "verification": ver,
           "gt_noise_mean": float(np.mean(t["gt_noise"])),
           "n_samples_per_replica": t["n_samples"], "arms": {}}

    def record(name, preds, extra=None):
        r = {
            "xmse_fit": xmse(preds[fit, -1], y1[fit, -1], y2[fit, -1]),
            "xmse_hold": xmse(preds[hold, -1], y1[hold, -1], y2[hold, -1]),
            "mse_r1_hold": plain_mse(preds[hold, -1], y1[hold, -1]),
            "mse_r2_hold": plain_mse(preds[hold, -1], y2[hold, -1]),
            "per_mlp_xmse_hold": [
                float(np.mean((preds[j, -1] - y1[j, -1]) * (preds[j, -1] - y2[j, -1])))
                for j in range(nfit, len(seeds))
            ],
        }
        if extra:
            r.update(extra)
        res["arms"][name] = r
        print(f"  {name:<4}  fit {r['xmse_fit']:.4e}   held-out {r['xmse_hold']:.4e}   "
              f"(r1 {r['mse_r1_hold']:.4e}, r2 {r['mse_r2_hold']:.4e})")
        return r

    print("\narms (cross-product MSE on the FINAL layer):")
    record("C0", P0)

    g1, curve = scan_global_gain([W[i] for i in range(nfit)], y1[fit], depth=16)
    P1 = np.stack([predict(w, mean_gain=np.full(16, g1)) for w in W])
    record("C1", P1, {"global_gain": g1, "scan_curve": curve})

    gl = fit_layer_gains_coordinate([W[i] for i in range(nfit)], y1[fit],
                                    rounds=args.rounds, init=np.full(16, g1))
    P2c = np.stack([predict(w, mean_gain=gl) for w in W])
    record("C2", P2c, {"layer_gains": [float(x) for x in gl]})

    pg = fit_post_gains(P0[fit], y1[fit])
    Pph = P0 * pg[None, :, None]
    record("P2", Pph, {"post_gains": [float(x) for x in pg]})

    c0h = res["arms"]["C0"]["xmse_hold"]
    print("\nheld-out improvement over C0:")
    for k in ("C1", "C2", "P2"):
        print(f"  {k}: {c0h / res['arms'][k]['xmse_hold']:.3f}x")
    print(f"\nGT noise floor (mean over networks): {res['gt_noise_mean']:.4e}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
