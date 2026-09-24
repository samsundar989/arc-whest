"""The MLP half of the corrector's ceiling: is there higher-order structure?

    python tools/corr_mlp_ceiling.py experiments/<dir>/fit100_features.npz \
        [--hidden 32] [--steps 3000] [--folds 5] [--expand] [--out ...]

`tools/corr_ceiling.py` measured what a per-layer RIDGE on the raw channels plus
squares and pairwise products can explain: held-out R^2 ~ 0.022 at layer 16, and
an in-sample/held-out gap of ~0.0005. That gap is the important part -- it says
the ridge is not data-limited, so anything a more flexible model adds must come
from HIGHER-ORDER structure in the same channels, not from more capacity to fit
the same signal.

This trains the handoff's Model 2 (2 hidden layers x 32, Adam) on the same rows
and the same net-wise folds, so the two numbers are directly comparable. Two
variants:

  * per-layer     -- the strict ceiling. If a model with its own parameters for
                     this layer cannot beat the ridge, nothing shared will.
  * shared+embed  -- the handoff's spelling: one model across all layers with a
                     learned 8-dim layer embedding, which is a REGULARISER here,
                     not extra capacity (each layer already has 102,400 rows for
                     45 features).

Targets are standardized per layer before training, because the one-step residual
spans 20x in scale from layer 2 to layer 16 and an unnormalized shared model
would simply ignore the early layers. R^2 is reported back in the original units.

cupy on the 5090. The footprint is a few MB of activations -- 102,400 x 9 inputs
and 32 hidden units -- so this runs directly rather than through workerq.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def expand_feats(X, names, top=6):
    cols, nm = [X], list(names)
    cols.append(X ** 2)
    nm += [f"{n}^2" for n in names]
    for i in range(min(top, X.shape[1])):
        for j in range(i + 1, min(top, X.shape[1])):
            cols.append((X[:, i] * X[:, j])[:, None])
            nm.append(f"{names[i]}*{names[j]}")
    return np.concatenate(cols, axis=1), nm


class MLP:
    """2 hidden layers, tanh, linear head. Adam. Deterministic from `seed`."""

    def __init__(self, xp, fin, hidden, seed, extra=0):
        g = xp.random.default_rng(seed)
        def he(a, b):
            return (g.standard_normal((a, b), dtype=xp.float32)
                    * xp.float32(np.sqrt(2.0 / a)))
        self.xp = xp
        self.W1, self.b1 = he(fin + extra, hidden), xp.zeros(hidden, dtype=xp.float32)
        self.W2, self.b2 = he(hidden, hidden), xp.zeros(hidden, dtype=xp.float32)
        self.W3, self.b3 = he(hidden, 1), xp.zeros(1, dtype=xp.float32)
        self.ps = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [xp.zeros_like(p) for p in self.ps]
        self.v = [xp.zeros_like(p) for p in self.ps]
        self.t = 0

    def forward(self, X):
        xp = self.xp
        h1 = xp.tanh(X @ self.W1 + self.b1)
        h2 = xp.tanh(h1 @ self.W2 + self.b2)
        return (h2 @ self.W3 + self.b3)[:, 0], (h1, h2)

    def step(self, X, y, lr, wd):
        xp = self.xp
        n = X.shape[0]
        pred, (h1, h2) = self.forward(X)
        d = (pred - y)[:, None] * xp.float32(2.0 / n)
        gW3 = h2.T @ d
        gb3 = d.sum(axis=0)
        d2 = (d @ self.W3.T) * (1 - h2 * h2)
        gW2 = h1.T @ d2
        gb2 = d2.sum(axis=0)
        d1 = (d2 @ self.W2.T) * (1 - h1 * h1)
        gW1 = X.T @ d1
        gb1 = d1.sum(axis=0)
        gs = [gW1, gb1, gW2, gb2, gW3, gb3]
        self.t += 1
        b1c = 1 - 0.9 ** self.t
        b2c = 1 - 0.999 ** self.t
        for i, (p, g) in enumerate(zip(self.ps, gs)):
            if p.ndim == 2:
                g = g + xp.float32(wd) * p
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            p -= xp.float32(lr) * (self.m[i] / b1c) / (xp.sqrt(self.v[i] / b2c) + 1e-8)
        return float(((pred - y) ** 2).mean())


def r2_np(y, pred):
    ss = float((y ** 2).sum())
    return float(1.0 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--expand", action="store_true",
                    help="feed the ridge's squares/products too (a fairer ceiling)")
    ap.add_argument("--shared", action="store_true",
                    help="one model for all layers with an 8-dim layer embedding")
    ap.add_argument("--target", default="onestep")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    import cupy as xp
    xp.get_default_memory_pool().set_limit(size=6 * 2 ** 30)

    d = np.load(a.npz, allow_pickle=True)
    X_all, names = d["X"], [str(x) for x in d["features"]]
    netid, Y = d["netid"], d[a.target]
    depth = X_all.shape[0]
    nets = np.unique(netid)
    rng = np.random.default_rng(0)
    chunks = np.array_split(rng.permutation(nets), a.folds)

    layers = list(range(1, depth)) if a.target == "onestep" else list(range(depth))
    res = {"npz": str(a.npz), "hidden": a.hidden, "steps": a.steps,
           "folds": a.folds, "expand": bool(a.expand), "shared": bool(a.shared),
           "target": a.target, "layers": {}}

    # ---- prepare per-layer matrices once (drop all-NaN columns, e.g. d21e@16)
    prepped = {}
    for l in layers:
        Xr = X_all[l].astype(np.float64)
        y = Y[l].astype(np.float64)
        keep = np.isfinite(Xr).all(axis=0)
        Xr, nm = Xr[:, keep], [n for n, k in zip(names, keep) if k]
        ok = np.isfinite(Xr).all(axis=1) & np.isfinite(y)
        Xr, y, nid = Xr[ok], y[ok], netid[ok]
        if a.expand:
            Xr, nm = expand_feats(Xr, nm)
        prepped[l] = (Xr, y, nid, nm)

    print(f"[mlp] {a.npz}  depth {depth}  hidden {a.hidden}  steps {a.steps}  "
          f"expand={a.expand}  shared={a.shared}")
    print(f"[mlp] {len(nets)} nets, {a.folds} folds, split BY NET\n")
    print(f"{'layer':>5} {'feat':>5} {'MSE':>12} {'R2 heldout':>12} {'R2 train':>10}")

    emb_dim = 8
    for l in layers:
        Xr, y, nid, nm = prepped[l]
        oof = np.zeros_like(y)
        tr_r2 = []
        for k in range(a.folds):
            te = np.isin(nid, chunks[k])
            tr = ~te
            mu, sd = Xr[tr].mean(0), Xr[tr].std(0)
            sd[sd == 0] = 1.0
            ym, ys = y[tr].mean(), y[tr].std()
            ys = ys if ys > 0 else 1.0
            Ztr = xp.asarray(((Xr[tr] - mu) / sd).astype(np.float32))
            Zte = xp.asarray(((Xr[te] - mu) / sd).astype(np.float32))
            ytr = xp.asarray(((y[tr] - ym) / ys).astype(np.float32))
            net = MLP(xp, Ztr.shape[1], a.hidden, seed=1234 + k)
            for s in range(a.steps):
                lr = a.lr * (0.5 ** (s / max(1, a.steps // 3)))
                net.step(Ztr, ytr, lr, a.wd)
            ptr, _ = net.forward(Ztr)
            pte, _ = net.forward(Zte)
            oof[te] = xp.asnumpy(pte).astype(np.float64) * ys + ym
            tr_r2.append(r2_np(y[tr], xp.asnumpy(ptr).astype(np.float64) * ys + ym))
            del Ztr, Zte, ytr, net
            xp.get_default_memory_pool().free_all_blocks()
        rh = r2_np(y, oof)
        mse = float((y ** 2).mean())
        print(f"{l + 1:>5} {Xr.shape[1]:>5} {mse:>12.4e} {rh:>12.5f} "
              f"{float(np.mean(tr_r2)):>10.5f}")
        res["layers"][l + 1] = {"mse": mse, "r2_heldout": rh,
                                "r2_train": float(np.mean(tr_r2)),
                                "features": Xr.shape[1]}

    if a.out:
        op = Path(a.out)
        if not op.is_absolute():
            op = ROOT / op
        op.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
        print(f"\n[mlp] -> {op.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
