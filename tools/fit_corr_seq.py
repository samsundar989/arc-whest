"""SEQUENTIAL teacher-forced fit of the v57 corrector — the procedure Step 2 specifies.

    python tools/fit_corr_seq.py --panel fit --n 100 \
        --out experiments/2026-09-24_corr/params_seq.npz

WHY THE SINGLE-PASS FIT FAILED
------------------------------
`tools/fit_corr.py` fitted all sixteen correctors on features taken from the
UNCORRECTED chain, in one pass. In the loop that is the wrong distribution: by the
time layer l runs, layers 0..l-1 have already been corrected, so layer l's
features have shifted and its prediction is the right MAGNITUDE in the wrong
direction. Measured end to end on DEV 100: **+0.58% raw, 46/100 wins**, against a
teacher-forced ceiling of 6.16-7.32%. Exposure bias ate all of it.

HANDOFF 09-24 Step 2 specifies the fix in its own words: "for l = 1..16, run the
chain with the corrector active for layers < l (their fitted weights frozen),
collect features at layer l and target t_l = mu_l(label) - mu_l(chain), fit
corrector_l, continue." That is what this does. Each corrector is fitted against
the chain it will actually run inside, so the mismatch is removed by construction.

The price is one sweep of the panel per layer instead of one in total. Each sweep
only has to reach layer l, so the total is sum(l) = 136 layer-passes rather than
16 x 16 = 256, about 8.5 full-pass equivalents.

WHAT REMAINS TRUE EVEN IF THIS WORKS
------------------------------------
Sequential fitting removes the distribution mismatch; it does NOT remove the
overlap noted in RESULTS.md 2a. Each corrector now sees the residual that is LEFT
after the earlier ones, so the total removed is the union of the predictable
signal, not the sum. The gain should therefore land BELOW the 6.16% framing-B
projection even when everything works. The number that decides anything is the
end-to-end DEV run, not the per-layer R^2 printed here.

FEATURES COME FROM THE ESTIMATOR ITSELF
---------------------------------------
The hook calls the generated file's own `_corr_raw`, the single implementation
`_corr_delta` also uses. Nothing here re-derives a feature, so the rows fitted on
cannot drift from the rows served -- which is the failure mode that would be
indistinguishable from the exposure bias above.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _rel(p) -> str:
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


sys.path.insert(0, str(ROOT / "local_sim"))
BACKEND = os.environ.get("WHEST_BACKEND", "cupy").lower()
if BACKEND == "cupy":
    sys.path.insert(0, str(ROOT / "tools" / "gpu_shim"))
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "tools"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np  # noqa: E402

BUDGET = 2 ** 41
PI = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
PJ = [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]


class _MLP:
    __slots__ = ("weights", "width", "depth", "seed")

    def __init__(self, weights, width, depth, seed):
        self.weights, self.width, self.depth, self.seed = weights, width, depth, seed


class _Ctx:
    def __init__(self, width, depth, budget, seed):
        self.width, self.depth = width, depth
        self.flop_budget, self.seed = budget, seed
        self.api_version, self.scratch_dir, self.submission_dir = "1.0", None, None


class MLPFit:
    """2x32 tanh, Adam, on the device. Same model as the ceiling measurement."""

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", default="estimator_k3_v57_corr.py")
    ap.add_argument("--panel", default="fit")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=125)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    a = ap.parse_args(argv)

    xp = None
    if BACKEND == "cupy":
        import cupy as xp
        xp.get_default_memory_pool().set_limit(
            size=int(float(os.environ.get("PROBE_GPU_LIMIT_GIB", "10")) * 2 ** 30))

    import panels
    import real_data as rd
    import importlib.util
    cand = ROOT / a.candidate
    spec = importlib.util.spec_from_file_location("_seq", cand)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_seq"] = mod
    spec.loader.exec_module(mod)
    fnp = sys.modules["flopscope.numpy"] if "flopscope.numpy" in sys.modules else None

    rows = panels.nets(a.panel)[: a.n]

    # DO NOT pre-load the panel onto the device. One net's weights are
    # 16 x 1024 x 1024 float64 = 134 MB, so caching 100 of them is 13.4 GB of
    # VRAM (and the same again in host RAM as float64) -- it blew a 10 GiB pool
    # on the first attempt. `real_data.get` reads from the on-disk HF cache and
    # is cheap next to a chain sweep, which is why `tools/anatomy.py` and
    # `tools/probe.py` both load per net. Same here.
    def load(i):
        split, idx, seed, _nm = rows[i]
        w, tr, _ = rd.get(idx, split, dtype=np.float32)
        dp, wd = w.shape[0], w.shape[1]
        wf = []
        for x in w:
            arr = np.asarray(x, dtype=np.float64)
            arr.setflags(write=False)
            wf.append(xp.asarray(arr) if xp is not None else arr)
        return wf, wd, dp, int(seed), np.asarray(tr, dtype=np.float64)

    _w0, width, depth, _s0, _t0 = load(0)
    del _w0, _t0
    if xp is not None:
        xp.get_default_memory_pool().free_all_blocks()
    print(f"[seq] {len(rows)} nets from {a.panel}, depth {depth}, width {width}",
          flush=True)

    H = a.hidden
    P = {"fmu": np.zeros((depth, 33), np.float32),
         "fisd": np.ones((depth, 33), np.float32),
         "W1": np.zeros((depth, 33, H), np.float32),
         "b1": np.zeros((depth, H), np.float32),
         "W2": np.zeros((depth, H, H), np.float32),
         "b2": np.zeros((depth, H), np.float32),
         "W3": np.zeros((depth, H, 1), np.float32),
         "b3": np.zeros((depth, 1), np.float32)}

    # Serve the LIVE numpy params to the estimator, so a newly fitted layer is
    # active on the next sweep without regenerating the file. All-zero pages are
    # exactly inert, so layers not yet fitted contribute nothing.
    def corr_arrays(dtype):
        return {k: xp.asarray(v) if xp is not None else v for k, v in P.items()}
    mod._corr_arrays = corr_arrays
    mod._CORR_ON = True

    rng = np.random.default_rng(0)
    nval = int(round(a.val_frac * len(rows)))
    val_ids = set(rng.permutation(len(rows))[:nval].tolist())
    print(f"[seq] {len(rows) - nval} train / {nval} val nets (whole nets held out)")
    print(f"{'layer':>5} {'rows':>8} {'MSE':>12} {'R2 val':>9} {'R2 train':>9} {'step':>6}")

    est = mod.Estimator()
    rep = {"panel": a.panel, "n": len(rows), "hidden": H, "steps": a.steps,
           "val_nets": nval, "layers": {}}

    for l in range(depth):
        Xs, ys, ids = [], [], []

        def hook(li, mu_pre, sigma, D3, D4, W, D21, mu_prev, mu_out, n, dtype,
                 _l=l):
            if li != _l:
                return
            raw = mod._corr_raw(mu_pre, sigma, D3, D4, W, D21, mu_prev, mu_out,
                                dtype)
            Xs.append(np.asarray(xp.asnumpy(raw) if xp is not None else raw,
                                 dtype=np.float64))
            ys.append(np.asarray(xp.asnumpy(mu_out) if xp is not None else mu_out,
                                 dtype=np.float64))

        mod._CORR_HOOK = hook
        for i in range(len(rows)):
            wf, wdt, dp, seed, truth = load(i)
            est.setup(_Ctx(wdt, dp, BUDGET, 0))
            before = len(Xs)
            est.predict(_MLP(wf, wdt, dp, seed), BUDGET)
            if len(Xs) == before:
                raise SystemExit(f"layer {l}: hook never fired on net {i}")
            # target = label - the CHAIN's uncorrected mu_out at this layer
            ys[-1] = truth[l] - ys[-1]
            ids.append(np.full(len(ys[-1]), i, dtype=np.int64))
            del wf
            if xp is not None:
                xp.get_default_memory_pool().free_all_blocks()
        mod._CORR_HOOK = None

        raw = np.nan_to_num(np.concatenate(Xs, axis=0), nan=0.0,
                            posinf=0.0, neginf=0.0)
        y = np.nan_to_num(np.concatenate(ys), nan=0.0)
        nid = np.concatenate(ids)
        Z = np.concatenate([raw, raw * raw, raw[:, PI] * raw[:, PJ]], axis=1)

        va = np.isin(nid, list(val_ids))
        tr_m = ~va
        fmu, fsd = Z[tr_m].mean(0), Z[tr_m].std(0)
        fsd[fsd == 0] = 1.0
        ym, ysd = y[tr_m].mean(), y[tr_m].std()
        ysd = ysd if ysd > 0 else 1.0
        Ztr = xp.asarray(((Z[tr_m] - fmu) / fsd).astype(np.float32))
        Zva = xp.asarray(((Z[va] - fmu) / fsd).astype(np.float32))
        ytr = xp.asarray(((y[tr_m] - ym) / ysd).astype(np.float32))

        net = MLPFit(xp, 33, H, seed=20260924 + l)
        best, best_snap, best_step = -np.inf, net.snapshot(), 0
        for s in range(a.steps):
            lr = a.lr * (0.5 ** (s / max(1, a.steps // 3)))
            net.step(Ztr, ytr, lr, a.wd)
            if (s + 1) % a.eval_every == 0:
                pv, _ = net.forward(Zva)
                rv = r2(y[va], xp.asnumpy(pv).astype(np.float64) * ysd + ym)
                if rv > best:
                    best, best_snap, best_step = rv, net.snapshot(), s + 1
        net.restore(best_snap)
        pt, _ = net.forward(Ztr)
        rt = r2(y[tr_m], xp.asnumpy(pt).astype(np.float64) * ysd + ym)

        # A layer whose corrector does not generalise is left at ZERO, which is
        # exactly inert. Shipping a negative-R^2 page would make the chain worse
        # for the sake of having fitted something.
        if best <= 0.0:
            print(f"{l + 1:>5} {len(y):>8} {float((y ** 2).mean()):>12.4e} "
                  f"{best:>9.5f} {rt:>9.5f} {'ZEROED':>6}")
            rep["layers"][l + 1] = {"r2_val": best, "zeroed": True}
        else:
            P["fmu"][l] = fmu.astype(np.float32)
            P["fisd"][l] = (1.0 / fsd).astype(np.float32)
            P["W1"][l] = xp.asnumpy(net.W1)
            P["b1"][l] = xp.asnumpy(net.b1)
            P["W2"][l] = xp.asnumpy(net.W2)
            P["b2"][l] = xp.asnumpy(net.b2)
            P["W3"][l] = (xp.asnumpy(net.W3) * ysd).astype(np.float32)
            P["b3"][l] = (xp.asnumpy(net.b3) * ysd + ym).astype(np.float32)
            print(f"{l + 1:>5} {len(y):>8} {float((y ** 2).mean()):>12.4e} "
                  f"{best:>9.5f} {rt:>9.5f} {best_step:>6}")
            rep["layers"][l + 1] = {"r2_val": best, "r2_train": rt,
                                    "best_step": best_step, "zeroed": False,
                                    "y_std": float(ysd), "y_mean": float(ym)}
        del Ztr, Zva, ytr, net, best_snap, Z, raw, y
        if xp is not None:
            xp.get_default_memory_pool().free_all_blocks()

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **P)
    print(f"\n[seq] params -> {_rel(out)}")
    if a.report:
        rp = Path(a.report)
        if not rp.is_absolute():
            rp = ROOT / rp
        rp.write_text(json.dumps(rep, indent=1, default=float) + "\n",
                      encoding="utf-8")
        print(f"[seq] report -> {_rel(rp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
