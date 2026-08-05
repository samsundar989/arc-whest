"""
Fresh-re-run robustness (GPU; runs in .venv-gpu, NEVER imported by an estimator).

Prizes are decided by re-running one designated submission on MLPs it has never
seen. That makes the *distribution* of per-MLP score the thing to engineer, not
its mean: a method whose worst 1% is catastrophic loses a fresh re-run even with a
good average, and a single failed MLP scores as zeros.

So: mint fresh He MLPs, label them with high-sample Monte Carlo, run the shipped
v2.4 pipeline on each, and report the whole distribution — median, tail
quantiles, worst case, and the ratio between them.

`--stress` additionally builds deliberately atypical networks (inflated row norms,
collapsed or amplified layers, extreme seeds) and checks that dead-neuron pruning
at tau=-2.5 and the closed-form patch stay well behaved on them.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/robustness.py --mlps 1000 --truth 4194304
    .venv-gpu\\Scripts\\python.exe local_sim/robustness.py --stress
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from track_t_mint import alignment_basis, closure, draw_points, sobol_base

BUDGET = 2.72e11
PRUNE_TAU = -2.5
WIDTH, DEPTH = 256, 32
# per-MLP fixed FLOPs measured for v2.4: closure 3.27e9, linearised-Jacobian chain
# ~1.04e9, eigh 1.5e8, fold/slice ~5e7, plus grader residual wall 9.84ms * 1e11.
FIXED_FLOPS = 4.51e9 + 9.84e8


def mint(seed, dev, width=WIDTH, depth=DEPTH, kind="plain"):
    """A fresh He-init MLP, optionally deformed to be atypical."""
    g = torch.Generator(device=dev).manual_seed(int(seed) & 0x7FFFFFFF)
    w = torch.randn(depth, width, width, generator=g, device=dev, dtype=torch.float32)
    w = w * math.sqrt(2.0 / width)
    if kind == "plain":
        return w
    r = torch.Generator(device=dev).manual_seed((int(seed) ^ 0x5EED) & 0x7FFFFFFF)
    if kind == "fat_rows":                     # 10% of rows with 3x the norm
        li = int(torch.randint(0, depth, (1,), generator=r, device=dev))
        m = (torch.rand(width, generator=r, device=dev) < 0.10).float() * 2.0 + 1.0
        w = w.clone()
        w[li] = w[li] * m[:, None]
    elif kind == "collapsed_layer":            # one layer scaled down: near-dead
        li = int(torch.randint(1, depth - 1, (1,), generator=r, device=dev))
        w = w.clone()
        w[li] = w[li] * 0.25
    elif kind == "amplified_layer":            # one layer scaled up
        li = int(torch.randint(1, depth - 1, (1,), generator=r, device=dev))
        w = w.clone()
        w[li] = w[li] * 4.0
    elif kind == "biased_layer":               # shift a layer's rows positive
        li = int(torch.randint(1, depth - 1, (1,), generator=r, device=dev))
        w = w.clone()
        w[li] = w[li] + 0.5 * math.sqrt(2.0 / width)
    return w


def truth_final(w, n_samples, dev, chunk=131072, seed=12345):
    """High-sample MC of the final-layer post-ReLU means (f64 accumulation)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    acc = torch.zeros(w.shape[-1], dtype=torch.float64, device=dev)
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        h = torch.randn(b, w.shape[-1], generator=g, device=dev, dtype=torch.float32)
        for li in range(w.shape[0]):
            h = torch.clamp(h @ w[li], min=0.0)
        acc += h.sum(dim=0, dtype=torch.float64)
        done += b
    return acc / n_samples


def run_v24(w, seed, base_pts, dev, n):
    """The shipped v2.4 pipeline. Returns (prediction, per-sample cost, diagnostics)."""
    depth, width = w.shape[0], w.shape[-1]
    w64 = w.to(torch.float64)
    alph, mus, phis, _, _ = closure(w64, dev)
    Q = alignment_basis(w64, phis)

    alive, dead, degenerate = [], [], 0
    for li in range(depth):
        if li == depth - 1:
            alive.append(torch.arange(width, device=dev))
            dead.append(None)
            continue
        keep = alph[li] > PRUNE_TAU
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            degenerate += 1
            idx = torch.argmax(alph[li]).reshape(1)
            keep = torch.zeros(width, dtype=torch.bool, device=dev)
            keep[idx] = True
        didx = torch.nonzero(~keep, as_tuple=False).squeeze(1)
        alive.append(idx)
        dead.append(didx if didx.numel() else None)

    sliced, offsets, cost, prev = [], [], 0.0, None
    for li in range(depth):
        wl = w64[li]
        if li == 0:
            wl = Q.T @ wl
        wf = wl.to(torch.float32)
        cur = alive[li]
        b = wf if prev is None else wf.index_select(0, prev)
        sliced.append(b.index_select(1, cur))
        if li > 0 and dead[li - 1] is not None:
            wd = wf.index_select(0, dead[li - 1]).index_select(1, cur)
            offsets.append(mus[li - 1].index_select(0, dead[li - 1]).to(torch.float32) @ wd)
        else:
            offsets.append(None)
        kk = width if prev is None else int(prev.numel())
        cost += (2 * kk - 1) * int(cur.numel()) + 1.5 * int(cur.numel())
        prev = cur

    gen = np.random.default_rng(int(seed))
    shift = torch.from_numpy(gen.integers(0, 1 << 30, size=width, dtype=np.uint32).astype(np.int64)).to(dev)
    h = draw_points(base_pts, shift, dev)
    for li in range(depth):
        z = h @ sliced[li]
        if offsets[li] is not None:
            z = z + offsets[li]
        h = torch.clamp(z, min=0.0)
    pred = h.mean(dim=0, dtype=torch.float64)
    diag = {
        "alive_min": min(int(a.numel()) for a in alive),
        "alive_mean": float(np.mean([int(a.numel()) for a in alive])),
        "degenerate_layers": degenerate,
        "max_offset": max([float(o.abs().max()) for o in offsets if o is not None] or [0.0]),
        "max_closure_mu": float(mus.abs().max()),
        "alpha_min": float(alph.min()),
        "alpha_max": float(alph.max()),
    }
    return pred, cost, diag


def quantiles(x, qs=(0.5, 0.9, 0.99, 1.0)):
    s = np.sort(x)
    return {q: float(s[min(len(s) - 1, int(math.ceil(q * len(s)) - 1))]) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=1000)
    ap.add_argument("--truth", type=int, default=4194304)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="", help="npz of weights+truth, so the SHIPPED "
                                               "estimator can be run on the same fresh MLPs")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    base_pts = sobol_base(args.n, WIDTH, dev)
    kinds = (["plain"] if not args.stress
             else ["plain", "fat_rows", "collapsed_layer", "amplified_layer", "biased_layer"])
    per_kind = args.mlps if not args.stress else max(20, args.mlps // len(kinds))
    print(f"device={dev}  kinds={kinds}  {per_kind} MLPs each  truth={args.truth:,} samples  "
          f"N={args.n}\n", flush=True)

    summary = {}
    dump_w, dump_t, dump_s = [], [], []
    t0 = time.perf_counter()
    for kind in kinds:
        mses, adjs, cbs, bad = [], [], [], []
        agg = {"alive_min": 10**9, "degenerate_layers": 0, "max_offset": 0.0,
               "alpha_min": 1e9, "alpha_max": -1e9, "nonfinite": 0}
        for j in range(per_kind):
            seed = args.seed0 + 1000 * kinds.index(kind) + j
            w = mint(seed, dev, kind=kind)
            t = truth_final(w, args.truth, dev, seed=seed ^ 0xABCD)
            pred, cps, diag = run_v24(w, seed, base_pts, dev, args.n)
            if args.dump and len(dump_w) < 250:
                dump_w.append(w.cpu().numpy())
                dump_t.append(t.cpu().numpy())
                dump_s.append(seed)
            if not bool(torch.isfinite(pred).all()):
                agg["nonfinite"] += 1
                bad.append((seed, "non-finite prediction"))
                continue
            mse = float(((pred - t) ** 2).mean())
            C = args.n * cps + FIXED_FLOPS
            cb = C / BUDGET
            mses.append(mse)
            cbs.append(cb)
            adjs.append(mse * max(0.1, cb))
            agg["alive_min"] = min(agg["alive_min"], diag["alive_min"])
            agg["degenerate_layers"] += diag["degenerate_layers"]
            agg["max_offset"] = max(agg["max_offset"], diag["max_offset"])
            agg["alpha_min"] = min(agg["alpha_min"], diag["alpha_min"])
            agg["alpha_max"] = max(agg["alpha_max"], diag["alpha_max"])
            if (j + 1) % 100 == 0:
                print(f"  {kind} {j + 1}/{per_kind}  {time.perf_counter() - t0:.0f}s", flush=True)
        mses, adjs = np.array(mses), np.array(adjs)
        summary[kind] = dict(mse=mses, adj=adjs, cb=np.array(cbs), agg=agg, bad=bad)

    print("\n" + "=" * 96)
    print("PER-MLP SCORE DISTRIBUTION ON FRESH MLPs  (v2.4 pipeline)")
    print("=" * 96)
    print(f"{'set':<18} {'n':>5} {'mean':>11} {'median':>11} {'p90':>11} {'p99':>11} "
          f"{'max':>11} {'max/med':>8}")
    print("-" * 96)
    for kind, s in summary.items():
        a = s["adj"]
        if not len(a):
            print(f"{kind:<18} {'0':>5}  ALL FAILED")
            continue
        q = quantiles(a)
        print(f"{kind:<18} {len(a):>5} {a.mean():>11.4e} {q[0.5]:>11.4e} {q[0.9]:>11.4e} "
              f"{q[0.99]:>11.4e} {q[1.0]:>11.4e} {q[1.0] / q[0.5]:>7.2f}x")

    print(f"\n{'set':<18} {'MSE mean':>11} {'MSE median':>11} {'MSE max':>11} {'C/B mean':>9} "
          f"{'worst 1% mean adj':>18}")
    print("-" * 96)
    for kind, s in summary.items():
        if not len(s["adj"]):
            continue
        a, m = s["adj"], s["mse"]
        w1 = np.sort(a)[int(0.99 * len(a)):]
        print(f"{kind:<18} {m.mean():>11.4e} {np.median(m):>11.4e} {m.max():>11.4e} "
              f"{s['cb'].mean():>9.4f} {w1.mean():>18.4e}")

    print("\n" + "=" * 96)
    print("SAFETY CHECKS")
    print("=" * 96)
    print(f"{'set':<18} {'non-finite':>11} {'min alive':>10} {'degenerate layers':>18} "
          f"{'max patch offset':>17} {'alpha range':>22}")
    print("-" * 96)
    ok = True
    for kind, s in summary.items():
        g = s["agg"]
        ok &= (g["nonfinite"] == 0)
        arange = "[%.1f, %.1f]" % (g["alpha_min"], g["alpha_max"])
        print(f"{kind:<18} {g['nonfinite']:>11} {g['alive_min']:>10} {g['degenerate_layers']:>18} "
              f"{g['max_offset']:>17.3e} {arange:>22}")
        for seed, why in s["bad"]:
            print(f"    !! seed {seed}: {why}")
    print(f"\nno non-finite predictions anywhere: {ok}")

    if args.out:
        np.savez(args.out, **{f"{k}_adj": v["adj"] for k, v in summary.items()},
                 **{f"{k}_mse": v["mse"] for k, v in summary.items()})
        print(f"saved {args.out}")
    if args.dump and dump_w:
        np.savez(args.dump, weights=np.stack(dump_w).astype(np.float32),
                 truth=np.stack(dump_t).astype(np.float64),
                 seeds=np.asarray(dump_s, dtype=np.int64))
        print(f"saved {args.dump} ({len(dump_w)} fresh MLPs with truth, for the shipped-file run)")


if __name__ == "__main__":
    main()
