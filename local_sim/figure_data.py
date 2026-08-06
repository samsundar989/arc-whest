"""
Compute the datasets behind the write-up's three headline figures (GPU; never
imported by an estimator).

  1. effective-rank collapse — participation ratio of the activation
     fluctuation covariance at every layer, which is the empirical fact the
     NNGP correlation-collapse section has to explain.
  2. error-vs-N — final-layer MSE against sample count, with the per-sample
     cost, so the flat-adjusted law (adjusted = V*c/B, independent of N) can be
     drawn rather than asserted.
  3. alpha by layer — the full distribution of the closure firing index at each
     depth, showing bimodality emerging with depth.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/figure_data.py --data <mini100.npz> --out <npz>
"""

import argparse
import time

import numpy as np
import torch

from track_t_mint import alignment_basis, closure, draw_points, sobol_base

PRUNE_TAU = -2.5
BUDGET = 2.72e11


def build(w64, alph, mus, phis, dev, depth, width):
    Q = alignment_basis(w64, phis)
    alive, dead = [], []
    for li in range(depth):
        if li == depth - 1:
            alive.append(torch.arange(width, device=dev))
            dead.append(None)
            continue
        keep = alph[li] > PRUNE_TAU
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            idx = torch.argmax(alph[li]).reshape(1)
            keep = torch.zeros(width, dtype=torch.bool, device=dev)
            keep[idx] = True
        alive.append(idx)
        didx = torch.nonzero(~keep, as_tuple=False).squeeze(1)
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
    return sliced, offsets, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlps", type=int, default=100)
    ap.add_argument("--ns", default="2048,4096,8192,16384,32768")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    d = np.load(args.data, allow_pickle=True)
    W_all, T_all, seeds = d["weights"], d["truth"], d["seeds"]
    M = min(args.mlps, len(W_all))
    depth, width = W_all.shape[1], W_all.shape[2]
    ns = [int(v) for v in args.ns.split(",")]
    nmax = max(ns)
    base_pts = {n: sobol_base(n, width, dev) for n in ns}
    print(f"device={dev} {M} MLPs depth={depth} width={width} ns={ns}", flush=True)

    eff_rank = np.zeros((M, depth))
    alphas = np.zeros((M, depth, width))
    mse_n = np.zeros((M, len(ns)))
    cost_n = np.zeros(M)
    t0 = time.perf_counter()
    for i in range(M):
        w32 = torch.from_numpy(W_all[i]).to(dev)
        w64 = w32.to(torch.float64)
        alph, mus, phis, _, _ = closure(w64, dev)
        alphas[i] = alph.cpu().numpy()
        t = torch.from_numpy(T_all[i][-1]).to(dev)
        sliced, offsets, cost = build(w64, alph, mus, phis, dev, depth, width)
        cost_n[i] = cost

        # --- 1. effective rank at every layer, from the unpruned forward -----
        gen = np.random.default_rng(int(seeds[i]))
        shift = torch.from_numpy(gen.integers(0, 1 << 30, size=width,
                                              dtype=np.uint32).astype(np.int64)).to(dev)
        h = draw_points(base_pts[16384], shift, dev)
        for li in range(depth):
            h = torch.clamp(h @ w32[li], min=0.0)
            H = h.to(torch.float64)
            Dv = H - H.mean(dim=0)
            ev = torch.linalg.eigvalsh((Dv.T @ Dv) / Dv.shape[0])
            eff_rank[i, li] = float(ev.sum() ** 2 / (ev * ev).sum())

        # --- 2. error vs N, through the shipped pipeline ---------------------
        for j, n in enumerate(ns):
            g2 = np.random.default_rng(int(seeds[i]))
            sh = torch.from_numpy(g2.integers(0, 1 << 30, size=width,
                                              dtype=np.uint32).astype(np.int64)).to(dev)
            hh = draw_points(base_pts[n], sh, dev)
            for li in range(depth):
                z = hh @ sliced[li]
                if offsets[li] is not None:
                    z = z + offsets[li]
                hh = torch.clamp(z, min=0.0)
            mse_n[i, j] = float(((hh.mean(dim=0, dtype=torch.float64) - t) ** 2).mean())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{M}  {time.perf_counter() - t0:.0f}s", flush=True)

    np.savez_compressed(args.out, eff_rank=eff_rank, alphas=alphas, mse_n=mse_n,
                        ns=np.asarray(ns), cost=cost_n, budget=BUDGET)
    print(f"\nsaved {args.out}")
    print(f"effective rank: L1 {eff_rank[:, 0].mean():.1f}  L8 {eff_rank[:, 7].mean():.1f}  "
          f"L16 {eff_rank[:, 15].mean():.1f}  L24 {eff_rank[:, 23].mean():.1f}  "
          f"L32 {eff_rank[:, 31].mean():.1f}")
    m = mse_n.mean(axis=0)
    c = cost_n.mean()
    print(f"{'N':>8} {'MSE':>12} {'C/B':>8} {'adjusted':>12}")
    for j, n in enumerate(ns):
        cb = n * c / BUDGET
        print(f"{n:>8} {m[j]:>12.4e} {cb:>8.4f} {m[j] * max(0.1, cb):>12.4e}")
    a = np.log(np.asarray(ns, dtype=float))
    print(f"fitted alpha (MSE ~ N^-alpha): {-np.polyfit(a, np.log(m), 1)[0]:.3f}")


if __name__ == "__main__":
    main()
