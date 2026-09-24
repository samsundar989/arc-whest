"""Fit the v57 in-loop mean corrector and emit the weights `build_v57.py` injects.

    python tools/fit_corr.py experiments/2026-09-24_corr/train_fit100.npz \
        experiments/2026-09-24_corr/train_hold300.npz \
        --out experiments/2026-09-24_corr/params.npz [--val-nets 80]

HANDOFF 2026-09-24 Step 2. One `33 -> 32 -> 32 -> 1` tanh MLP per layer, Adam,
trained to predict `mu_l(label) - mu_l(chain)` from the layer's own state.

TARGET
------
The dump's `target` array, which is `label - the CHAIN's own mu_out`. NOT
`-accumulated`: `accumulated` is built from the returned prediction array, whose
row 0 `predict` overwrites with an exact closed form and whose rows are clamped
at zero, so at layer 1 it is a different quantity. The corrector adds to
`mu_out`, so `mu_out` is what the label must be compared against.

Fitting on the ACCUMULATED residual (this target) rather than the one-step
residual is deliberate and matches the handoff: at inference the chain's input at
layer l is its own imperfect mean, so `label - chain` is the quantity that
actually exists to be corrected. Measured on FIT 100, the two targets explain
almost the same ABSOLUTE variance per layer (ratio 0.62-1.01), so the choice
moves the predicted end-to-end gain only from 7.32% to 6.16% -- the bracket, not
a fork in the road. See RESULTS.md "Step 2 priced".

WHY hidden = 32
---------------
Measured, not assumed: at hidden 64 the net-wise held-out R^2 is UNIFORMLY worse
by 0.005-0.011 with a wider train/held-out gap, i.e. 32 is already at the
capacity limit for 100 training nets. That is also why this trains on 400 nets
(FIT 100 + the pinned HOLDOUT-300, `data/panels/holdout_train_split.json`): the
MLP -- unlike the ridge, whose gap is ~0.0005 -- IS data-limited.

SPLIT
-----
Validation nets are held out WHOLE. 1024 neurons of one MLP share that MLP's
weights, so a neuron-level split leaks and reports an R^2 that will not survive
first contact with DEV. The validation set picks the early-stopping step and is
never trained on.

THE TARGET SCALE IS FOLDED INTO W3/b3
-------------------------------------
Training standardises the target per layer. Rather than make the estimator undo
that at inference (two more billed operations per layer), `W3` and `b3` are
rescaled on export so the head emits the correction in the chain's own units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

#: MUST match `_CORR_PI`/`_CORR_PJ` in the generated estimator and
#: `tools/corr_ceiling.py` expand(): pairwise products of the first six raw
#: channels, i < j.
PI = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
PJ = [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]

FEATURES = ("mu_pre", "sigma", "alpha", "d3n", "d4n", "wnorm", "d21e",
            "mu_out", "dmu")


def expand(raw: np.ndarray) -> np.ndarray:
    """(rows, 9) -> (rows, 33). Identical to the estimator's `_corr_delta`."""
    return np.concatenate([raw, raw * raw, raw[:, PI] * raw[:, PJ]], axis=1)


class MLP:
    def __init__(self, xp, fin, hidden, seed):
        g = xp.random.default_rng(seed)
        def he(a, b):
            return (g.standard_normal((a, b), dtype=xp.float32)
                    * xp.float32(np.sqrt(2.0 / a)))
        self.xp = xp
        self.W1, self.b1 = he(fin, hidden), xp.zeros(hidden, dtype=xp.float32)
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
        gW3, gb3 = h2.T @ d, d.sum(axis=0)
        d2 = (d @ self.W3.T) * (1 - h2 * h2)
        gW2, gb2 = h1.T @ d2, d2.sum(axis=0)
        d1 = (d2 @ self.W2.T) * (1 - h1 * h1)
        gW1, gb1 = X.T @ d1, d1.sum(axis=0)
        self.t += 1
        b1c, b2c = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        for i, (p, g) in enumerate(zip(self.ps, [gW1, gb1, gW2, gb2, gW3, gb3])):
            if p.ndim == 2:
                g = g + xp.float32(wd) * p
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            p -= xp.float32(lr) * (self.m[i] / b1c) / (xp.sqrt(self.v[i] / b2c) + 1e-8)

    def snapshot(self):
        return [p.copy() for p in self.ps]

    def restore(self, snap):
        for p, s in zip(self.ps, snap):
            p[...] = s


def r2(y, pred):
    ss = float((y ** 2).sum())
    return float(1.0 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def load_dumps(paths):
    Xs, Ts, Ns, off = [], [], [], 0
    feats = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        nm = [str(x) for x in d["features"]]
        if feats is None:
            feats = nm
        elif nm != feats:
            raise SystemExit(f"{p}: feature order differs from the first dump")
        if "target" not in d:
            raise SystemExit(f"{p}: no 'target' array -- re-dump with the "
                             "current tools/anatomy.py")
        Xs.append(d["X"])
        Ts.append(d["target"])
        Ns.append(d["netid"].astype(np.int64) + off)
        off += int(d["netid"].max()) + 1
    return (np.concatenate(Xs, axis=1), np.concatenate(Ts, axis=1),
            np.concatenate(Ns), feats, off)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--report", default="")
    a = ap.parse_args(argv)

    import cupy as xp
    xp.get_default_memory_pool().set_limit(size=10 * 2 ** 30)

    X_all, T_all, netid, feats, nnets = load_dumps(a.dumps)
    if feats != list(FEATURES):
        raise SystemExit(f"feature order {feats} != {list(FEATURES)}")
    depth = X_all.shape[0]
    rng = np.random.default_rng(0)
    val = set(rng.permutation(nnets)[: int(round(a.val_frac * nnets))].tolist())
    is_val = np.isin(netid, list(val))
    print(f"[fit_corr] {depth} layers, {X_all.shape[1]:,} rows, {nnets} nets "
          f"({nnets - len(val)} train / {len(val)} val), hidden {a.hidden}")
    print(f"{'layer':>5} {'MSE':>12} {'R2 val':>9} {'R2 train':>9} {'step':>6}")

    H = a.hidden
    P = {"fmu": np.zeros((depth, 33), np.float32),
         "fisd": np.zeros((depth, 33), np.float32),
         "W1": np.zeros((depth, 33, H), np.float32),
         "b1": np.zeros((depth, H), np.float32),
         "W2": np.zeros((depth, H, H), np.float32),
         "b2": np.zeros((depth, H), np.float32),
         "W3": np.zeros((depth, H, 1), np.float32),
         "b3": np.zeros((depth, 1), np.float32)}
    rep = {"dumps": [str(p) for p in a.dumps], "hidden": H, "steps": a.steps,
           "nets": nnets, "val_nets": len(val), "layers": {}}

    for l in range(depth):
        raw = X_all[l].astype(np.float64)
        # d21e is NaN at the LAST layer (the chain never forms D21 there). The
        # estimator feeds a constant zero at that layer, so train on zero too --
        # matching, not approximating.
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(T_all[l].astype(np.float64), nan=0.0)
        Z = expand(raw)
        tr, va = ~is_val, is_val
        fmu, fsd = Z[tr].mean(0), Z[tr].std(0)
        fsd[fsd == 0] = 1.0
        ym, ys = y[tr].mean(), y[tr].std()
        ys = ys if ys > 0 else 1.0

        Ztr = xp.asarray(((Z[tr] - fmu) / fsd).astype(np.float32))
        Zva = xp.asarray(((Z[va] - fmu) / fsd).astype(np.float32))
        ytr = xp.asarray(((y[tr] - ym) / ys).astype(np.float32))
        yva_h = y[va]

        net = MLP(xp, 33, H, seed=20260924 + l)
        best, best_snap, best_step = -np.inf, net.snapshot(), 0
        for s in range(a.steps):
            lr = a.lr * (0.5 ** (s / max(1, a.steps // 3)))
            net.step(Ztr, ytr, lr, a.wd)
            if (s + 1) % a.eval_every == 0:
                pv, _ = net.forward(Zva)
                rv = r2(yva_h, xp.asnumpy(pv).astype(np.float64) * ys + ym)
                if rv > best:
                    best, best_snap, best_step = rv, net.snapshot(), s + 1
        net.restore(best_snap)
        pt, _ = net.forward(Ztr)
        rt = r2(y[tr], xp.asnumpy(pt).astype(np.float64) * ys + ym)

        # Fold the target scale into the head so the estimator needs no
        # de-standardisation step.
        P["fmu"][l] = fmu.astype(np.float32)
        P["fisd"][l] = (1.0 / fsd).astype(np.float32)
        P["W1"][l] = xp.asnumpy(net.W1)
        P["b1"][l] = xp.asnumpy(net.b1)
        P["W2"][l] = xp.asnumpy(net.W2)
        P["b2"][l] = xp.asnumpy(net.b2)
        P["W3"][l] = (xp.asnumpy(net.W3) * ys).astype(np.float32)
        P["b3"][l] = (xp.asnumpy(net.b3) * ys + ym).astype(np.float32)

        mse = float((y ** 2).mean())
        print(f"{l + 1:>5} {mse:>12.4e} {best:>9.5f} {rt:>9.5f} {best_step:>6}")
        rep["layers"][l + 1] = {"mse": mse, "r2_val": best, "r2_train": rt,
                                "best_step": best_step, "y_std": float(ys),
                                "y_mean": float(ym)}
        del Ztr, Zva, ytr, net, best_snap
        xp.get_default_memory_pool().free_all_blocks()

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **P)
    print(f"\n[fit_corr] params -> {out.relative_to(ROOT)}")
    if a.report:
        rp = Path(a.report)
        if not rp.is_absolute():
            rp = ROOT / rp
        rp.write_text(json.dumps(rep, indent=1, default=float) + "\n", encoding="utf-8")
        print(f"[fit_corr] report -> {rp.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
