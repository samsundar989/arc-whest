"""Price the in-loop learned corrector BEFORE building it: the ridge's real R^2.

    python tools/corr_ceiling.py experiments/<dir>/fit100_features.npz \
        [--folds 5] [--out experiments/<dir>/ceiling.json]

HANDOFF 2026-09-24 Step 2 specifies a two-day build: a teacher-forced, in-loop,
per-layer corrector with ~50 features, ridge and MLP variants, trained on FIT
plus 300 HOLDOUT nets. Its kill criteria are "ridge < 1% on DEV" and "MLP < 3%".

Both of those are, to first order, a statement about ONE number: the fraction of
the per-layer residual's variance that a function of the per-neuron features can
explain. This measures that number directly from `tools/anatomy.py --dump`, in
minutes rather than days, so the build is entered with its own ceiling known.

WHY R^2 IS (APPROXIMATELY) THE SCORE GAIN
-----------------------------------------
The corrector is applied in the loop, so it shrinks each layer's *increment* to
the accumulated error. If the increments are roughly independent and the
corrector removes a fraction rho of each one's variance, the accumulated error at
layer 16 falls by ~rho as well -- NOT by rho x 0.206. So held-out R^2 on the
one-step residual is the right quantity to compare against the 1% and 3% gates,
and it is an OPTIMISTIC reading of them, because:

  * it is measured on the residual the corrector sees under teacher forcing,
    with no compounding of the corrector's own errors;
  * the real corrector must also not damage the covariance / kappa3 channels,
    which it feeds;
  * it ignores the FLOP cost of the corrector, which the score pays for.

So if this number is below the gate, the lane is closed; if it is above, the
lane is open but not yet won.

THE SPLIT IS BY NET, NOT BY NEURON
----------------------------------
1024 neurons of one MLP share that MLP's weights, so a neuron-level split leaks
and reports a hugely inflated R^2. Every fold here holds out whole nets.

FEATURES
--------
The raw per-neuron channels the chain actually computes (from `--dump`), then
squares and pairwise products, which is the expansion the handoff asks for. Two
features in the handoff's list are not here and cannot be: the transported
previous correction is recursive (it does not exist until a corrector does), and
the young/old tier shares of D3 need further chain surgery. Both are noted in
the output so the number is read for what it is -- a ceiling on the ~9-channel
expansion, not on every feature ever proposed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def expand(X: np.ndarray, names: list[str], top: int = 6):
    """[raw | squares | pairwise products of the first `top`] + intercept."""
    cols, out_names = [X], list(names)
    cols.append(X ** 2)
    out_names += [f"{n}^2" for n in names]
    for i in range(min(top, X.shape[1])):
        for j in range(i + 1, min(top, X.shape[1])):
            cols.append((X[:, i] * X[:, j])[:, None])
            out_names.append(f"{names[i]}*{names[j]}")
    return np.concatenate(cols, axis=1), out_names


def ridge_fit(Xtr, ytr, lam):
    """Closed-form ridge on standardized, intercept-free columns."""
    n, p = Xtr.shape
    A = Xtr.T @ Xtr + lam * n * np.eye(p)
    return np.linalg.solve(A, Xtr.T @ ytr)


def r2(y, pred):
    ss = float((y ** 2).sum())
    return float(1.0 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def run_layer(X, y, netid, folds, lams):
    """Held-out R^2 by net-wise K-fold, lambda chosen inside each training fold."""
    nets = np.unique(netid)
    rng = np.random.default_rng(0)
    order = rng.permutation(nets)
    chunks = np.array_split(order, folds)

    oof = np.zeros_like(y)
    for k in range(folds):
        te = np.isin(netid, chunks[k])
        tr = ~te
        Xtr, ytr = X[tr], y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Ztr = (Xtr - mu) / sd
        ym = ytr.mean()
        # inner split for lambda: hold out the first fifth of the TRAINING nets
        trn = np.unique(netid[tr])
        inner = np.isin(netid[tr], trn[: max(1, len(trn) // 5)])
        best, best_r2 = lams[0], -np.inf
        for lam in lams:
            b = ridge_fit(Ztr[~inner], ytr[~inner] - ym, lam)
            s = r2(ytr[inner] - ym, Ztr[inner] @ b)
            if s > best_r2:
                best, best_r2 = lam, s
        b = ridge_fit(Ztr, ytr - ym, best)
        oof[te] = ((X[te] - mu) / sd) @ b + ym
    return r2(y, oof), oof


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--target", default="onestep", choices=("onestep", "accumulated", "both"))
    a = ap.parse_args(argv)

    d = np.load(a.npz, allow_pickle=True)
    X_all = d["X"]                     # (depth, rows, F)
    names = [str(x) for x in d["features"]]
    netid = d["netid"]
    depth = X_all.shape[0]
    print(f"[ceiling] {a.npz}: depth {depth}, rows {X_all.shape[1]}, "
          f"features {names}")
    print(f"[ceiling] nets {len(np.unique(netid))}, folds {a.folds} (split BY NET)\n")

    lams = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    targets = (("onestep", "accumulated") if a.target == "both" else (a.target,))
    res = {"npz": str(a.npz), "features": names, "folds": a.folds,
           "nets": int(len(np.unique(netid))), "layers": {}}

    for tgt in targets:
        Y = d[tgt]
        print(f"=== target: {tgt} residual ===")
        print(f"{'layer':>5} {'rows':>8} {'MSE':>12} {'R2 heldout':>12} "
              f"{'R2 in-sample':>13} {'MSE after':>12}")
        for l in range(depth):
            if tgt == "onestep" and l == 0:
                continue
            y = Y[l].astype(np.float64)
            Xr = X_all[l].astype(np.float64)
            ok = np.isfinite(Xr).all(axis=1) & np.isfinite(y)
            if ok.sum() < 1000:
                # d21e is NaN at the last layer (D21 is not computed there);
                # drop that column rather than the rows.
                keep = np.isfinite(Xr).all(axis=0)
                Xr, nm = Xr[:, keep], [n for n, k in zip(names, keep) if k]
                ok = np.isfinite(Xr).all(axis=1) & np.isfinite(y)
            else:
                nm = names
            Xr, y, nid = Xr[ok], y[ok], netid[ok]
            Z, _ = expand(Xr, nm)
            mse = float((y ** 2).mean())
            rh, oof = run_layer(Z, y, nid, a.folds, lams)
            # in-sample, for the gap
            mu, sd = Z.mean(0), Z.std(0)
            sd[sd == 0] = 1.0
            b = ridge_fit((Z - mu) / sd, y - y.mean(), 1e-4)
            ri = r2(y, ((Z - mu) / sd) @ b + y.mean())
            after = float(((y - oof) ** 2).mean())
            print(f"{l + 1:>5} {len(y):>8} {mse:>12.4e} {rh:>12.5f} "
                  f"{ri:>13.5f} {after:>12.4e}")
            res["layers"].setdefault(tgt, {})[l + 1] = {
                "rows": int(len(y)), "mse": mse, "r2_heldout": rh,
                "r2_insample": ri, "mse_after": after,
                "n_expanded_features": int(Z.shape[1])}
        print()

    if a.out:
        op = Path(a.out)
        if not op.is_absolute():
            op = ROOT / op
        op.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
        print(f"[ceiling] -> {op.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
