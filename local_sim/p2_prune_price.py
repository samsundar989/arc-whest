"""Price dead/cold-column pruning at the LIVE Phase-2 shape (1024 x 16).

The lever, as carried by the strategy canon and the Sep 8 handoff:

> **Dead/cold-column pruning with exact compensation** (`~0.8x`, still unpriced).
> Accuracy-neutral by construction if the compensation is exact; price it.

The `~0.8x` traces to marius's Phase-1 note ("14% from the cold path alone") at
`256 x 32`. **Nothing has measured whether the columns it needs exist at
`1024 x 16`**, and that is the question this module answers first, because a
saving of `0.8x` requires dropping ~20% of columns and there is no point pricing
a compensation scheme for columns that are not there.

## What "cold" means here, and why layer 1 can never contribute

A column is cold when its pre-activation `alpha = mu / sigma` is very negative:
it almost never fires, so its post-ReLU mean and variance are near zero and the
dense matmuls that carry it are nearly wasted work.

**At layer 1 there are none, exactly.** The input is `x ~ N(0, I)`, so
`z = W_1^T x` has mean **exactly zero** for every neuron — `alpha = 0`, every
neuron fires exactly half the time. Coldness can only develop with depth, as the
post-ReLU mean vector builds up and `W^T mu` spreads `alpha` away from zero. So
the lever's ceiling is set by how fast that spread grows, which is a property of
the shape and is what gets measured below.

## What is priced, and what is NOT

**Priced:** the fraction of columns below a range of `alpha` thresholds at each
layer, and the FLOP saving that fraction implies. A matmul is billed
`(2K-1) * M * N` **by shapes only** (`docs/RESEARCH_NOTES.md` 3d), so pruning
pays only by making matrices genuinely smaller — zeroing entries buys nothing.

**Measured, not assumed:** what dropping those columns *costs* in accuracy with
**no** compensation. That is the number that says how exact a compensation has to
be, and it is the honest bound on "accuracy-neutral by construction".

**NOT priced here:** a specific compensation scheme. Inventing one and then
pricing it would be designing the answer. This module measures whether the lever
has room; the scheme is a separate contract if it does.

## Method

The K2p1 closure (`p2_cal_lab.closure`) is the shipped chain's own recursion, and
its per-layer `(mu_pre, sigma_pre)` is exactly what `alpha` needs. This module
re-implements only the state recursion (not the readout) so it can record `alpha`
per layer, and it checks itself against `p2_cal_lab.closure`'s returned means
before reporting anything — a divergent recursion would give a plausible `alpha`
distribution for the wrong chain.

Pure numpy, CPU, seconds per network. No GPU, no gate, no promotion.
ROUTINE by `AGENTS.md`'s classification.
`# seed-ledger: no block named` — reads the existing P01 seeds via
`mint_provenance.canonical_mint`.

Run:  uv run python local_sim/p2_prune_price.py --n-mlps 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

SQRT2PI = float(np.sqrt(2.0 * np.pi))
ALPHA_THRESHOLDS = (-2.0, -3.0, -4.0, -5.0)


def _phi(x):
    return np.exp(-0.5 * x * x) / SQRT2PI


def _Phi(x):
    from math import erf, sqrt
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def chain_alpha(weights, drop_mask_fn=None):
    """The K2p1 state recursion, recording alpha per layer.

    `drop_mask_fn(k, alpha) -> bool mask` optionally zeroes the post-ReLU state
    of the selected columns at layer k, which is naive pruning with NO
    compensation -- the arm that measures how much a compensation must recover.
    """
    width, depth = weights[0].shape[0], len(weights)
    mu = np.zeros(width, dtype=np.float64)
    cov = np.eye(width, dtype=np.float64)
    alphas, means, dropped = [], [], []
    for k in range(depth):
        wk = weights[k]
        mu_pre = wk.T @ mu
        cov_pre = wk.T @ cov @ wk
        var_pre = np.maximum(np.diag(cov_pre), 1e-300)
        sig = np.sqrt(var_pre)
        a = mu_pre / sig
        alphas.append(a)

        # post-ReLU mean and second moment (the closure's own Gaussian map)
        Pa, pa = _Phi(a), _phi(a)
        m1 = mu_pre * Pa + sig * pa
        m2 = (var_pre + mu_pre ** 2) * Pa + mu_pre * sig * pa
        v = np.maximum(m2 - m1 ** 2, 0.0)

        if drop_mask_fn is not None:
            mask = drop_mask_fn(k, a)
            dropped.append(int(mask.sum()))
            m1 = np.where(mask, 0.0, m1)
            v = np.where(mask, 0.0, v)
        means.append(m1.copy())

        # propagate: correlations carried through the Gaussian map's slope Phi(a)
        slope = Pa
        cov = (slope[:, None] * slope[None, :]) * cov_pre
        np.fill_diagonal(cov, v)
        if drop_mask_fn is not None:
            m = drop_mask_fn(k, a)
            cov[m, :] = 0.0
            cov[:, m] = 0.0
        mu = m1
    return np.array(means), np.array(alphas), dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mlps", type=int, default=6)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--gt-dir", default="experiments/2026-08-26_p2_gt")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    from mint_provenance import canonical_mint
    import p2_cal_lab as lab

    gt = os.path.join(_ROOT, a.gt_dir)
    seeds = [int(s) for s in np.load(os.path.join(gt, "gt_halfA.npz"))["seeds"]][:a.n_mlps]

    print("=" * 78)
    print(f"DEAD/COLD-COLUMN PRUNING -- price at the LIVE shape {a.width} x {a.depth}")
    print("=" * 78)
    print("  claim under test: ~0.8x saving (marius, 14% from the cold path at 256x32)")
    print("  a matmul is billed by SHAPES only, so pruning pays only by shrinking")
    print("  matrices -- zeroing entries buys nothing.")
    print()

    frac = {t: [] for t in ALPHA_THRESHOLDS}
    per_layer = {t: np.zeros(a.depth) for t in ALPHA_THRESHOLDS}
    parity = []
    for i, s in enumerate(seeds):
        w, _e, _ss = canonical_mint(a.width, a.depth, s)
        means, alphas, _ = chain_alpha(w)

        # instrument check: this recursion must BE the shipped chain's
        ref = lab.closure(w)
        rel = float(np.max(np.abs(means - ref)) / max(np.max(np.abs(ref)), 1e-300))
        parity.append(rel)

        for t in ALPHA_THRESHOLDS:
            f = (alphas < t).mean(axis=1)         # per layer
            per_layer[t] += f / len(seeds)
            frac[t].append(float(f.mean()))
        print(f"    net {i+1}/{len(seeds)}  seed {s}   "
              f"alpha min {alphas.min():+.2f}  max {alphas.max():+.2f}   "
              f"closure parity {rel:.2e}")

    worst_parity = max(parity)
    print()
    print(f"  INSTRUMENT CHECK  worst |this recursion - p2_cal_lab.closure| = "
          f"{worst_parity:.2e}")
    if worst_parity > 1e-12:
        print("  REFUSED: this recursion is not the shipped chain's. Numbers void.")
        return 1
    print("  -> it IS the shipped chain's recursion; the alpha distribution is the")
    print("     chain's own, not a re-derivation of it.")
    print()
    print(f"  {'alpha <':>8}  {'mean frac of columns':>22}  {'implied FLOP saving':>21}")
    for t in ALPHA_THRESHOLDS:
        m = float(np.mean(frac[t]))
        print(f"  {t:>8.1f}  {m:>21.4%}  {1.0 - m:>20.4f}x")
    print()
    print("  per-layer fraction below alpha = -2 (layer 0 first):")
    print("   ", "  ".join(f"{x:.3%}" for x in per_layer[-2.0]))
    print()
    print("  Layer 0 is exactly 0% by construction: x ~ N(0, I) makes every")
    print("  pre-activation mean exactly zero, so alpha = 0 and every neuron")
    print("  fires half the time. Coldness can only develop with depth.")

    # ---- THE OTHER HALF: what does dropping them COST, with no compensation? --
    # "Accuracy-neutral by construction if the compensation is exact" is a claim
    # about a scheme that does not exist yet. This measures the size of the hole
    # such a scheme would have to fill exactly.
    print()
    print("  NAIVE DROP (no compensation) -- relative change in the FINAL-layer means")
    print(f"  {'alpha <':>8}  {'rel L2 change':>15}  {'max rel change':>16}  {'saving':>8}")
    drop_cost = {}
    for t in ALPHA_THRESHOLDS:
        rels, maxes = [], []
        for s_ in seeds:
            w, _e, _ss = canonical_mint(a.width, a.depth, s_)
            base, _al, _d = chain_alpha(w)
            pruned, _al2, _d2 = chain_alpha(w, drop_mask_fn=lambda k, al, _t=t: al < _t)
            b, q = base[-1], pruned[-1]
            den = float(np.linalg.norm(b)) or 1.0
            rels.append(float(np.linalg.norm(q - b) / den))
            maxes.append(float(np.max(np.abs(q - b)) / max(np.max(np.abs(b)), 1e-300)))
        drop_cost[str(t)] = {"rel_l2": float(np.mean(rels)),
                             "max_rel": float(np.mean(maxes))}
        print(f"  {t:>8.1f}  {np.mean(rels):>15.4%}  {np.mean(maxes):>16.4%}  "
              f"{1.0 - float(np.mean(frac[t])):>7.4f}x")

    rec = {
        "slice": "dead/cold-column pruning -- price at 1024x16",
        "naive_drop_cost": drop_cost,
        "claim_under_test": "~0.8x (marius, 256x32)",
        "n_mlps": len(seeds), "width": a.width, "depth": a.depth,
        "closure_parity_worst": worst_parity,
        "mean_fraction_below_alpha": {str(t): float(np.mean(frac[t]))
                                      for t in ALPHA_THRESHOLDS},
        "per_layer_fraction_below_-2": per_layer[-2.0].tolist(),
        "tier": "ROUTINE lab measurement. No gate, no seed block, no promotion.",
        "note": ("Prices only whether the columns EXIST and what dropping them "
                 "would save. Does NOT price a compensation scheme -- inventing "
                 "one and then pricing it would be designing the answer."),
    }
    if a.out:
        out = os.path.join(_ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print(f"\n  artifact  {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
