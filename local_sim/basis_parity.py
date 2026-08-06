"""
Does truncating the alignment basis cost anything? (GPU; never imported by an estimator.)

v2.4 rotates by the full eigenvector matrix of J J^T; v2.4.1 keeps only the
leading `rank` directions and completes the basis by a deterministic QR. Both are
valid orthogonal rotations of an isotropic Gaussian, so both are unbiased — the
question is whether they have the same VARIANCE.

A single-realisation A/B cannot answer that: swapping the basis changes the
realised point set, and two independent realisations of this estimator differ by
~18% in mean MSE over 100 MLPs. So each basis is run over R independent digital
shifts per MLP and the comparison is made on the per-MLP MEAN over draws, which
removes the realisation noise instead of measuring it.

Also reports the stability that motivated the change: how far each basis moves
when the weights are perturbed by a relative 1e-15.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/basis_parity.py --data <mini100.npz> --draws 32
"""

import argparse
import time

import numpy as np
import torch

from track_t_mint import closure, draw_points, sobol_base

PRUNE_TAU = -2.5


def basis_full(w64, phis, depth):
    J = w64[0]
    for k in range(1, depth):
        J = (J * phis[k - 1].unsqueeze(0)) @ w64[k]
    _, vecs = torch.linalg.eigh(J @ J.T)
    return torch.flip(vecs, dims=[1])


def basis_trunc(w64, phis, depth, rank):
    J = w64[0]
    for k in range(1, depth):
        J = (J * phis[k - 1].unsqueeze(0)) @ w64[k]
    _, vecs = torch.linalg.eigh(J @ J.T)
    width = vecs.shape[0]
    k = min(rank, width)
    top = torch.flip(vecs[:, width - k :], dims=[1])
    hi = top.max(dim=0).values
    lo = top.min(dim=0).values
    top = top * torch.where(hi >= -lo, 1.0, -1.0)
    q, r = torch.linalg.qr(top, mode="complete")
    d = torch.diagonal(r)[:k]
    sgn = torch.where(d < 0.0, -torch.ones_like(d), torch.ones_like(d))
    fix = torch.cat([sgn, torch.ones(width - k, dtype=q.dtype, device=q.device)])
    return q * fix


def run_with_basis(w32, w64, alph, mus, Q, base_pts, dev, n, seeds_rng, depth, width):
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
        didx = torch.nonzero(~keep, as_tuple=False).squeeze(1)
        alive.append(idx)
        dead.append(didx if didx.numel() else None)

    sliced, offsets, prev = [], [], None
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
        prev = cur

    outs = []
    for shift_np in seeds_rng:
        shift = torch.from_numpy(shift_np.astype(np.int64)).to(dev)
        h = draw_points(base_pts, shift, dev)
        for li in range(depth):
            z = h @ sliced[li]
            if offsets[li] is not None:
                z = z + offsets[li]
            h = torch.clamp(z, min=0.0)
        outs.append(h.mean(dim=0, dtype=torch.float64))
    return torch.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mlps", type=int, default=100)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--ranks", default="16,32,64")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    d = np.load(args.data, allow_pickle=True)
    W_all, T_all, seeds = d["weights"], d["truth"], d["seeds"]
    M = min(args.mlps, len(W_all))
    depth, width = W_all.shape[1], W_all.shape[2]
    ranks = [int(v) for v in args.ranks.split(",")]
    base_pts = sobol_base(args.n, width, dev)
    print(f"device={dev} {M} MLPs x {args.draws} draws, ranks={ranks}\n", flush=True)

    res = {("full", 0): [], **{("trunc", r): [] for r in ranks}}
    stab = {("full", 0): [], **{("trunc", r): [] for r in ranks}}
    t0 = time.perf_counter()
    for i in range(M):
        w32 = torch.from_numpy(W_all[i]).to(dev)
        w64 = w32.to(torch.float64)
        alph, mus, phis, _, _ = closure(w64, dev)
        t = torch.from_numpy(T_all[i][-1]).to(dev)
        gen = np.random.default_rng(int(seeds[i]))
        shifts = [gen.integers(0, 1 << 30, size=width, dtype=np.uint32) for _ in range(args.draws)]

        # perturbed copy, for the stability column
        g = torch.Generator(device=dev).manual_seed(i)
        wp64 = w64 * (1.0 + 1e-15 * torch.randn(w64.shape, generator=g, device=dev, dtype=torch.float64))
        alph_p, mus_p, phis_p, _, _ = closure(wp64, dev)

        for key in res:
            kind, rank = key
            if kind == "full":
                Q = basis_full(w64, phis, depth)
                Qp = basis_full(wp64, phis_p, depth)
            else:
                Q = basis_trunc(w64, phis, depth, rank)
                Qp = basis_trunc(wp64, phis_p, depth, rank)
            s = torch.sign((Q * Qp).sum(dim=0))
            s = torch.where(s == 0, torch.ones_like(s), s)
            stab[key].append(float((Q * s - Qp).abs().max()))
            preds = run_with_basis(w32, w64, alph, mus, Q, base_pts, dev, args.n, shifts, depth, width)
            res[key].append(float(((preds - t) ** 2).mean()))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{M}  {time.perf_counter() - t0:.0f}s", flush=True)

    print(f"\n{'basis':<18} {'mean MSE over draws':>20} {'vs full':>9} "
          f"{'max basis move @1e-15':>23}")
    print("-" * 76)
    base = float(np.mean(res[("full", 0)]))
    for key in res:
        kind, rank = key
        m = float(np.mean(res[key]))
        label = "full eigh (v2.4)" if kind == "full" else f"truncated rank {rank}"
        print(f"{label:<18} {m:>20.4e} {base / m:>8.4f}x {float(np.mean(stab[key])):>23.2e}")
    print(f"\n(each entry is the mean over {args.draws} independent digital shifts per MLP, "
          f"so realisation noise is down {args.draws}x)")


if __name__ == "__main__":
    main()
