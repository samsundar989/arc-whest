"""WP18 / Arm L, step B2: the `E3` map. Ridge, leave-networks-out, `G-L-FIT-E3`.

Contract: `experiments/LOG.md`, 2026-09-03, "CONTRACT WP18" § B, committed at
`49e2e42`. Gate `G-L-FIT-E3`: held-out `R^2` on `E3` **`>= 0.90`**,
layer-averaged over pre-layers `5..15` (the `l + 1` of gate layers `4..14`),
leave-networks-out on `P05`, with the per-layer minimum reported beside it.

**Every fitting choice below was written before any label existed** -- this file
is committed alongside the label job, not after reading its output. That is the
whole point, and the choices are listed here so a reader can check they were not
adjusted to the answer.

THE FIVE CHOICES, DECLARED
--------------------------
1. **`R^2` is UNCENTERED, against zero**: `1 - ||E3 - E3_hat||^2 / ||E3||^2`.
   That is exactly `gm_depth.json`'s `explained`, and it is the right convention
   here because a deployed map must produce the *value*, not the deviation from
   a mean it would have to be told. A centered `R^2` is reported beside it for
   reference and scores no gate.
2. **ONE coefficient vector across all pre-layers**, not one per layer. `CONTRACT
   WP15` caps the basis at 50 coefficients total; eleven per-layer vectors over
   25 features would be 275. The features carry the layer's own scale (they are
   built from its actual state), so a single fixed vector is the deployable
   object -- and it makes the per-layer table a real test rather than eleven
   independent fits.
3. **Per-block weighting `1 / ||E3||^2`.** Pooled least squares would be
   dominated by whichever pre-layer has the largest `||E3||`, and the gate is
   layer-averaged. Weighting each `(net, pre-layer)` block by the inverse of its
   own energy makes the pooled objective the mean of per-block `1 - explained`,
   which is the quantity the gate is written on.
4. **Column scaling by training-fold RMS**, never by held-out data. Absorbed
   into the coefficients at deploy time, so it costs no operations and adds no
   constants to the 50.
5. **The ridge penalty is chosen by an INNER leave-networks-out split on the
   training folds only.** The outer held-out fold is never consulted for it.
   This is the discipline WP13's selector lane lost first.

THE NULL DIFFERS IN CODE PATH
-----------------------------
`--null permute-coefficients` fits normally and then permutes the coefficient
vector across basis terms before prediction. Same arithmetic, same cost, same
magnitudes, wrong assignment. `eps = 0` is a determinism check and is not a
null; `CONTRACT WP15` is explicit and both Skeptics were.

WHAT IS NOT HERE
----------------
The `E21` leg. Its target is `n x n` and a dense store is `4.6 GB`, so the fit
consumes a Gram/moment reduction (`<T_t, T_s>`, `<T_t, E21>`, `||E21||^2`) that
`p2_wp18_labels.py` computes in a SECOND pass once the `E21` basis is declared.
Declaring the basis and then computing its moments cannot be reversed into a
basis chosen after seeing the target, which is why the order is that way round.

Run (CPU, light -- the labels are already reduced to n-vectors):
    uv run python local_sim/p2_wp18_fit.py --labels experiments/2026-09-03_wp18_armL/labels_P05.npz
    uv run python local_sim/p2_wp18_fit.py --labels <npz> --null permute-coefficients
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

#: `G-L-FIT-E3`, fixed in `CONTRACT WP18` § B before any label existed.
G_L_FIT_E3 = 0.90

#: The ridge ladder the inner split chooses from. A ladder, not a continuous
#: search, so the choice is reproducible and reportable.
LAMBDAS = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)

#: Outer folds for leave-networks-out. 8 folds of 16 networks at `n_nets = 128`.
N_FOLDS = 8


def _blocks(FEAT, E3):
    """Flatten `(nets, layers, n, feat)` into per-`(net, layer)` blocks.

    Returns a list of `(net_index, layer_index, X, y)` with `X` shaped
    `(n, n_feat)` and `y` shaped `(n,)`. Keeping the block structure is what
    makes the per-block weighting and the per-layer report possible.
    """
    out = []
    for i in range(FEAT.shape[0]):
        for j in range(FEAT.shape[1]):
            out.append((i, j, FEAT[i, j], E3[i, j]))
    return out


def _weighted_normal_equations(blocks, idx, scale):
    """`(X^T W X, X^T W y, sum_w ||y||^2)` over the blocks named by `idx`.

    `W` is `1 / ||y||^2` per block -- declared choice 3. A block with
    `||y|| == 0` carries no information about the map and is dropped rather than
    given infinite weight; the count of dropped blocks is returned so it can
    never be silent.
    """
    n_feat = blocks[0][2].shape[1]
    G = np.zeros((n_feat, n_feat), dtype=np.float64)
    b = np.zeros(n_feat, dtype=np.float64)
    sst = 0.0
    dropped = 0
    for k in idx:
        _i, _j, X, y = blocks[k]
        e = float(y @ y)
        if not (e > 0.0) or not np.isfinite(e):
            dropped += 1
            continue
        w = 1.0 / e
        Xs = X / scale
        G += w * (Xs.T @ Xs)
        b += w * (Xs.T @ y)
        sst += w * e
    return G, b, sst, dropped


def _solve(G, b, lam):
    n = G.shape[0]
    tr = float(np.trace(G)) / max(n, 1)
    return np.linalg.solve(G + lam * tr * np.eye(n), b)


def _score(blocks, idx, coef, scale):
    """Per-block uncentered and centered `R^2`. No weighting: a score, not a fit."""
    rows = []
    for k in idx:
        i, j, X, y = blocks[k]
        yhat = (X / scale) @ coef
        e = float(y @ y)
        r = y - yhat
        sse = float(r @ r)
        yc = y - y.mean()
        sst_c = float(yc @ yc)
        rows.append({
            "net": int(i), "layer_index": int(j),
            "r2_uncentered": 1.0 - sse / max(e, 1e-300),
            "r2_centered": 1.0 - sse / max(sst_c, 1e-300),
            "norm_y": float(np.sqrt(e)),
        })
    return rows


def cmd_fit(a) -> int:
    z = np.load(a.labels, allow_pickle=False)
    FEAT, E3 = z["FEAT"], z["E3"]
    names = [str(x) for x in z["feature_names"]]
    pre = [int(x) for x in z["pre_layers"]]
    n_nets, n_lay, n_neur, n_feat = FEAT.shape
    ep = bool(z["endpoint_pass"][0]) if "endpoint_pass" in z else None
    print(f"  labels    {os.path.relpath(a.labels, _ROOT)}")
    print(f"  shape     {n_nets} nets x {n_lay} pre-layers {pre[0]}..{pre[-1]} "
          f"x {n_neur} neurons x {n_feat} features")
    print(f"  B1 endpoint check in the label file: "
          f"{'PASS' if ep else ('FAIL' if ep is False else 'absent')}")
    if ep is False:
        print("  REFUSED: the labeller's own endpoint check FAILED, so these "
              "labels are not E3. CONTRACT WP18 B1 stops the slice here.")
        return 1

    blocks = _blocks(FEAT, E3)
    all_idx = list(range(len(blocks)))
    # Column scale from the WHOLE set is forbidden -- choice 4. It is computed
    # per outer fold from the training blocks only, inside the loop.
    folds = [[k for k in all_idx if (blocks[k][0] % N_FOLDS) == f]
             for f in range(N_FOLDS)]

    held, chosen = [], []
    for f in range(N_FOLDS):
        te = folds[f]
        tr = [k for k in all_idx if k not in set(te)]
        Xtr = np.concatenate([blocks[k][2] for k in tr], axis=0)
        scale = np.sqrt((Xtr * Xtr).mean(axis=0))
        scale = np.where(scale > 0, scale, 1.0)

        # --- inner leave-networks-out on the TRAINING folds only, for lambda
        inner_nets = sorted({blocks[k][0] for k in tr})
        inner_hold = set(inner_nets[::N_FOLDS])
        itr = [k for k in tr if blocks[k][0] not in inner_hold]
        ite = [k for k in tr if blocks[k][0] in inner_hold]
        Gi, bi, _s, _d = _weighted_normal_equations(blocks, itr, scale)
        best, best_lam = -np.inf, LAMBDAS[0]
        for lam in LAMBDAS:
            try:
                c = _solve(Gi, bi, lam)
            except np.linalg.LinAlgError:
                continue
            m = float(np.mean([r["r2_uncentered"]
                               for r in _score(blocks, ite, c, scale)]))
            if m > best:
                best, best_lam = m, lam
        chosen.append({"fold": f, "lambda": best_lam, "inner_r2": best})

        G, b, _s, dropped = _weighted_normal_equations(blocks, tr, scale)
        coef = _solve(G, b, best_lam)
        if a.null == "permute-coefficients":
            # CODE-PATH-DISTINCT NULL: same arithmetic, same magnitudes, wrong
            # assignment. Seeded off the fold so it is reproducible.
            rng = np.random.default_rng(20260903 + f)
            coef = coef[rng.permutation(coef.size)]
        rows = _score(blocks, te, coef, scale)
        for r in rows:
            r["fold"] = f
        held.extend(rows)
        print(f"    fold {f}  lambda {best_lam:.0e}  dropped {dropped}  "
              f"held-out mean R^2 (uncentered) "
              f"{np.mean([r['r2_uncentered'] for r in rows]):+.4f}")

    # ---- the gate: layer-averaged held-out R^2, per-layer minimum beside it --
    per_layer = []
    for j, p in enumerate(pre):
        v = [r["r2_uncentered"] for r in held if r["layer_index"] == j]
        vc = [r["r2_centered"] for r in held if r["layer_index"] == j]
        per_layer.append({"pre_layer": p, "n": len(v),
                          "r2_uncentered_mean": float(np.mean(v)),
                          "r2_uncentered_min": float(np.min(v)),
                          "r2_centered_mean": float(np.mean(vc))})
    lay_avg = float(np.mean([r["r2_uncentered_mean"] for r in per_layer]))
    lay_min = float(np.min([r["r2_uncentered_mean"] for r in per_layer]))
    passed = bool(lay_avg >= G_L_FIT_E3)

    print()
    print("  held-out R^2 on E3, leave-networks-out, uncentered")
    print(f"    {'pre-layer':>10}{'mean':>10}{'worst net':>12}{'centered':>11}")
    for r in per_layer:
        print(f"    {r['pre_layer']:>10}{r['r2_uncentered_mean']:>10.4f}"
              f"{r['r2_uncentered_min']:>12.4f}{r['r2_centered_mean']:>11.4f}")
    print()
    print(f"  G-L-FIT-E3: layer-averaged {lay_avg:.4f} vs {G_L_FIT_E3:.2f}"
          f"   {'PASS' if passed else 'FAIL'}"
          + ("   [NULL RUN -- this is not a result]" if a.null else ""))
    print(f"  worst pre-layer mean {lay_min:.4f}")

    # ---- the coefficients, by name, on a full-data refit (REPORT ONLY) ------
    Xall = np.concatenate([blocks[k][2] for k in all_idx], axis=0)
    sc = np.sqrt((Xall * Xall).mean(axis=0))
    sc = np.where(sc > 0, sc, 1.0)
    G, b, _s, _d = _weighted_normal_equations(blocks, all_idx, sc)
    lam_mode = max(set(c["lambda"] for c in chosen),
                   key=lambda L: sum(1 for c in chosen if c["lambda"] == L))
    full = _solve(G, b, lam_mode)
    order = np.argsort(-np.abs(full))
    print()
    print(f"  full-data refit at the modal lambda {lam_mode:.0e} "
          f"-- REPORT ONLY, scores no gate. Coefficients by |size|:")
    for k in order[: min(10, full.size)]:
        print(f"    {names[k]:<16}{full[k]:+.6e}")

    if a.out:
        _dump(a.out, {
            "check": "WP18 Arm L step B2: the E3 map, ridge, leave-networks-out",
            "contract": "experiments/LOG.md 2026-09-03 CONTRACT WP18 § B, "
                        "HEAD 49e2e42",
            "labels": os.path.relpath(a.labels, _ROOT),
            "gate": "G-L-FIT-E3", "threshold": G_L_FIT_E3,
            "null": a.null or "none",
            "n_nets": int(n_nets), "pre_layers": pre,
            "feature_names": names, "n_folds": N_FOLDS,
            "lambda_ladder": list(LAMBDAS), "fold_lambdas": chosen,
            "per_layer": per_layer,
            "layer_averaged_r2": lay_avg, "worst_layer_r2": lay_min,
            "pass": passed,
            "full_refit_lambda": lam_mode,
            "full_refit_coefficients": {names[i]: float(full[i])
                                        for i in range(full.size)},
            "caveat": "DESIGN tier on P05, where CONTRACT WP18 permits tuning. "
                      "G-L-FIT is a FIT gate, not an outcome gate: the outcome "
                      "gate is G-L-E2E, in flight, on P06[..064..127].",
        })
    return 0 if passed else 1


def _dump(path, doc):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"  wrote {os.path.relpath(path, _ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--null", choices=("permute-coefficients",), default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    print("=" * 78)
    print("WP18 / Arm L  --  step B2: the E3 map, and G-L-FIT-E3")
    print("=" * 78)
    return cmd_fit(a)


if __name__ == "__main__":
    raise SystemExit(main())
