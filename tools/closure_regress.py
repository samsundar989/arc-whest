"""Step 3.2 (HANDOFF 09-27): is the closure's exact-input residual predictable from its inputs?

    .venv-gpu/Scripts/python.exe tools/closure_regress.py --bench experiments/<dir>/bench

Reads `tools/closure_bench.py`'s per-net npz files. FIT nets 0..59 fit, 60..99 read (R^2 on
held-out NETS, pooled over layers, and per layer).

  univariate  targets rK1, rK2, rK3, rK4 (normalised by sigma^p)
              features (alpha, gamma3, gamma4, l/16)            [+ k5n, k6n: diagnostic only]
  bivariate   targets rK11, rK21, rK12
              features (rho, alpha_i, alpha_j, gamma3_i, gamma3_j, d21, d12, d22, q31, q13, l/16)

Models: ridge on the standardised features and all their pairwise products (degree 2), then a
16-hidden tanh MLP (Adam, full batch on a 400k-row subsample). The pre kappa5/kappa6 arm answers
"truncation (unobservable at inference) vs state-predictable": the R^2 it ADDS is the share of
the residual that needs moments the chain does not carry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

UNI_T = ("rK1", "rK2", "rK3", "rK4")
UNI_F = ("alpha", "gamma3", "gamma4")
DIAG_F = ("k5n", "k6n")
BIV_T = ("rK11", "rK21", "rK12")
BIV_F = ("rho", "alpha_i", "alpha_j", "gamma3_i", "gamma3_j", "d21", "d12", "d22", "q31", "q13")


def gather(bench, nets, feats, targets, layers, cap=None, seed=0):
    X, Y, L = [], [], []
    rng = np.random.default_rng(seed)
    for g in nets:
        p = bench / f"net_{g:03d}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        for l in layers:
            if f"{targets[0]}@{l}" not in z.files:
                continue
            f = np.stack([z[f"{k}@{l}"] for k in feats] + [np.full_like(z[f"{feats[0]}@{l}"], l / 16.0)], 1)
            y = np.stack([z[f"{k}@{l}"] for k in targets], 1)
            if cap and len(f) > cap:
                idx = rng.choice(len(f), cap, replace=False)
                f, y = f[idx], y[idx]
            X.append(f)
            Y.append(y)
            L.append(np.full(len(f), l))
    X, Y, L = np.concatenate(X), np.concatenate(Y), np.concatenate(L)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    return X[ok].astype(np.float64), Y[ok].astype(np.float64), L[ok]


def poly2(X):
    n = X.shape[1]
    cols = [X] + [X[:, i:i + 1] * X[:, j:j + 1] for i in range(n) for j in range(i, n)]
    return np.concatenate(cols, 1)


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def ridge(Xtr, Ytr, Xte, lam=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = poly2((Xtr - mu) / sd)
    B = poly2((Xte - mu) / sd)
    am, asd = A.mean(0), A.std(0) + 1e-12
    A, B = (A - am) / asd, (B - am) / asd
    ym = Ytr.mean(0)
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    return B @ W + ym


def mlp(Xtr, Ytr, Xte, hidden=16, steps=3000, cap=400_000, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if len(Xtr) > cap:
        i = rng.choice(len(Xtr), cap, replace=False)
        Xtr, Ytr = Xtr[i], Ytr[i]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-30
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((Ytr - ym) / ys, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                              torch.nn.Linear(hidden, yt.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        p = net(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).numpy()
    return p * ys + ym


def block(bench, name, feats, targets, layers, cap):
    tr = range(0, 60)
    te = range(60, 100)
    Xtr, Ytr, _ = gather(bench, tr, feats, targets, layers, cap)
    Xte, Yte, Lte = gather(bench, te, feats, targets, layers, cap)
    out = {"n_train": len(Xtr), "n_test": len(Xte)}
    for mname, fn in (("ridge", ridge), ("mlp16", mlp)):
        P = fn(Xtr, Ytr, Xte)
        out[mname] = {t: float(v) for t, v in zip(targets, r2(Yte, P))}
        out[mname + "_by_layer"] = {int(l): {t: float(v) for t, v in zip(targets, r2(Yte[Lte == l], P[Lte == l]))}
                                    for l in sorted(set(Lte.tolist()))}
    # the systematic part: mean residual per layer (a pure bias, predictable by a constant)
    out["bias_share"] = {t: float(np.mean([Yte[Lte == l, k].mean() ** 2 for l in set(Lte.tolist())])
                                  / np.mean(Yte[:, k] ** 2)) for k, t in enumerate(targets)}
    print(name, json.dumps({k: v for k, v in out.items() if not k.endswith("_by_layer")}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    bench = Path(a.bench)
    res = {
        "uni": block(bench, "uni", UNI_F, UNI_T, range(1, 15), None),
        "uni+k56": block(bench, "uni+k56", UNI_F + DIAG_F, UNI_T, range(1, 15), None),
        "uni_K1_last": block(bench, "uni_K1_last", UNI_F, ("rK1",), (15,), None),
        "biv": block(bench, "biv", BIV_F, BIV_T, range(1, 15), 2048),
    }
    Path(a.out or bench.parent / "regress.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
