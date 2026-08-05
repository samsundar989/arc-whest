"""
Track P, stage 1 (GPU; runs in .venv-gpu, NEVER imported by an estimator).

The idea being tested
---------------------
flopscope bills a matmul by shapes, so an (N,256)@(256,256) layer costs
(2*256-1)*N*256. If the sample-to-sample FLUCTUATION of the activation vector
lived in a k-dimensional subspace with k << 256, we could propagate the deviation
in that subspace — (N,k)@(k,256) — and carry the mean analytically, cutting the
per-sample cost by roughly 256/k. Dropping directions also drops variance, which
biases E[ReLU] downward (Jensen); the plan is to add the dropped variance back
through the closure formula.

Stage 1 asks only the question that decides whether any of that is worth
building:

  * how much of the deviation energy is in the top {32, 64, 128} directions at
    layers 8 / 16 / 24 / 32?
  * does a PRECOMPUTABLE basis (eigenvectors of the Gaussian closure's own
    propagated covariance — free, the estimator already computes the chain) find
    the same directions as the data PCA, which we could not afford online?
  * what does projecting at layer l and continuing to layer 32 actually cost in
    final-layer MSE?

Gate to proceed to a prototype: >= 95% energy in the top 64 for layers >= 16.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/track_p_spectrum.py --data <mini100.npz> \\
        --mlps 40 --n 16384
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from track_t_mint import closure, draw_points, sobol_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mlps", type=int, default=40)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--layers", default="8,16,24,32", help="1-based")
    ap.add_argument("--ks", default="32,64,128")
    ap.add_argument("--out", default="")
    ap.add_argument("--jensen", action="store_true",
                    help="also run the ORACLE Jensen correction (upper bound on any implementation)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    d = np.load(args.data, allow_pickle=True)
    W_all, T_all, seeds = d["weights"], d["truth"], d["seeds"]
    M = min(args.mlps, len(W_all))
    depth, width = W_all.shape[1], W_all.shape[2]
    layers = [int(v) - 1 for v in args.layers.split(",")]
    ks = [int(v) for v in args.ks.split(",")]
    base = sobol_base(args.n, width, dev)
    print(f"device={dev} {M} MLPs N={args.n} layers(1-based)={args.layers} ks={args.ks}\n", flush=True)

    L, K = len(layers), len(ks)
    en_pca = np.zeros((M, L, K))      # energy captured by the data PCA top-k
    en_cls = np.zeros((M, L, K))      # ... by the closure covariance's top-k
    en_grm = np.zeros((M, L, K))      # ... by eigenvectors of W_l W_l^T (cheapest basis)
    prj_mse = np.zeros((M, L, K))     # final-layer MSE after projecting at l and continuing
    prj_var = np.zeros((M, L, K))     # fraction of pre-activation variance dropped at l+1
    eff_rank = np.zeros((M, L))
    base_mse = np.zeros(M)
    jen_mse = np.zeros((M, L, K))     # ... with the ORACLE Jensen correction applied

    def relu_smooth(z, delta):
        """E[ReLU(z + eps)] for eps ~ N(0, delta): the exact conditional mean of
        the dropped component, i.e. the best any Jensen correction can do."""
        sd = torch.sqrt(torch.clamp(delta, min=0.0))
        safe = sd > 1e-12
        a = z / torch.where(safe, sd, torch.ones_like(sd))
        Phi = 0.5 * torch.erfc(-a * (1.0 / math.sqrt(2.0)))
        phi = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * a * a)
        return torch.where(safe, z * Phi + sd * phi, torch.clamp(z, min=0.0))

    t0 = time.perf_counter()
    for i in range(M):
        w32 = torch.from_numpy(W_all[i]).to(dev)
        w64 = w32.to(torch.float64)
        _, _, phis, _, _ = closure(w64, dev)
        # closure covariance chain, kept per layer (f64)
        covs = {}
        mu = torch.zeros(width, dtype=torch.float64, device=dev)
        cov = torch.eye(width, dtype=torch.float64, device=dev)
        off = 1.0 - torch.eye(width, dtype=torch.float64, device=dev)
        for li in range(depth):
            wl = w64[li]
            mu_pre = wl.T @ mu
            cov_pre = wl.T @ cov @ wl
            var_pre = torch.clamp(torch.diagonal(cov_pre), min=1e-12)
            sig = torch.sqrt(var_pre)
            a = mu_pre / sig
            Phi = 0.5 * torch.erfc(-a * (1.0 / math.sqrt(2.0)))
            phi = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * a * a)
            mu = mu_pre * Phi + sig * phi
            ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
            var = torch.clamp(ez2 - mu * mu, min=0.0)
            cov = torch.outer(Phi, Phi) * cov_pre * off + torch.diag(var)
            cov = 0.5 * (cov + cov.T)
            if li in layers:
                covs[li] = cov.clone()

        gen = np.random.default_rng(int(seeds[i]))
        shift = torch.from_numpy(gen.integers(0, 1 << 30, size=width, dtype=np.uint32).astype(np.int64)).to(dev)
        x = draw_points(base, shift, dev)

        acts = []
        full_var = []  # per-layer pre-activation variance of the UNPROJECTED run
        h = x
        for li in range(depth):
            z = h @ w32[li]
            full_var.append(z.to(torch.float64).var(dim=0))
            h = torch.clamp(z, min=0.0)
            acts.append(h if li in layers else None)
        t = torch.from_numpy(T_all[i][-1]).to(dev)
        base_mse[i] = float(((acts[-1].mean(dim=0).to(torch.float64) - t) ** 2).mean())

        for a_i, li in enumerate(layers):
            H = acts[li].to(torch.float64)
            m = H.mean(dim=0)
            Dv = H - m
            S = (Dv.T @ Dv) / Dv.shape[0]
            ev, V = torch.linalg.eigh(S)
            ev = torch.flip(ev, dims=[0])
            V = torch.flip(V, dims=[1])
            tot = float(ev.sum())
            eff_rank[i, a_i] = float(ev.sum() ** 2 / (ev * ev).sum())  # participation ratio

            _, Vc = torch.linalg.eigh(covs[li])
            Vc = torch.flip(Vc, dims=[1])
            wl = w64[li]
            _, Vg = torch.linalg.eigh(wl @ wl.T)
            Vg = torch.flip(Vg, dims=[1])

            for k_i, k in enumerate(ks):
                en_pca[i, a_i, k_i] = float(ev[:k].sum()) / tot
                for name, Vb, store in (("cls", Vc, en_cls), ("grm", Vg, en_grm)):
                    P = Vb[:, :k]
                    store[i, a_i, k_i] = float(torch.trace(P.T @ S @ P)) / tot

                # project the deviation onto the data top-k and continue forward
                P = V[:, :k]
                Hp = (m + (Dv @ P) @ P.T).to(torch.float32)
                zn = Hp @ w32[li + 1] if li + 1 < depth else None
                if zn is not None:
                    v_full = (acts[li].to(torch.float64) @ w64[li + 1]).var(dim=0)
                    prj_var[i, a_i, k_i] = float(1.0 - (zn.to(torch.float64).var(dim=0) / v_full).mean())
                    hh = torch.clamp(zn, min=0.0)
                    for lj in range(li + 2, depth):
                        hh = torch.clamp(hh @ w32[lj], min=0.0)
                    prj_mse[i, a_i, k_i] = float(((hh.mean(dim=0).to(torch.float64) - t) ** 2).mean())
                else:
                    prj_mse[i, a_i, k_i] = float(((Hp.mean(dim=0).to(torch.float64) - t) ** 2).mean())
                    prj_var[i, a_i, k_i] = 0.0

                # --- ORACLE Jensen correction: at every layer downstream, add
                # back the variance the projection dropped, using the exact
                # conditional mean E[ReLU(z + eps)]. Uses the unprojected run's
                # variances, so it is an upper bound on any real implementation.
                if args.jensen and li + 1 < depth:
                    hh = (m + (Dv @ P) @ P.T).to(torch.float32)
                    for lj in range(li + 1, depth):
                        zz = hh @ w32[lj]
                        d64 = zz.to(torch.float64)
                        delta = torch.clamp(full_var[lj] - d64.var(dim=0), min=0.0)
                        if lj == depth - 1:
                            hh = relu_smooth(d64, delta)
                        else:
                            hh = relu_smooth(d64, delta).to(torch.float32)
                    jen_mse[i, a_i, k_i] = float(((hh.mean(dim=0) - t) ** 2).mean())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{M}  {time.perf_counter() - t0:.0f}s", flush=True)

    print(f"\n{M} MLPs, N={args.n}. Baseline (full-width sampling) final-layer MSE = {base_mse.mean():.4e}\n")
    print("DEVIATION-ENERGY SPECTRUM  (fraction of sample-fluctuation energy in the top k)")
    print(f"{'layer':>6} {'eff.rank':>9} " + " ".join(f"{'PCA k=' + str(k):>11}" for k in ks)
          + " " + " ".join(f"{'closure ' + str(k):>12}" for k in ks)
          + " " + " ".join(f"{'W W^T ' + str(k):>11}" for k in ks))
    print("-" * 130)
    for a_i, li in enumerate(layers):
        print(
            f"{li + 1:>6} {eff_rank[:, a_i].mean():>9.1f} "
            + " ".join(f"{en_pca[:, a_i, k_i].mean():>11.4f}" for k_i in range(len(ks)))
            + " " + " ".join(f"{en_cls[:, a_i, k_i].mean():>12.4f}" for k_i in range(len(ks)))
            + " " + " ".join(f"{en_grm[:, a_i, k_i].mean():>11.4f}" for k_i in range(len(ks)))
        )

    print("\nPROJECT-AND-CONTINUE DAMAGE  (final-layer MSE after truncating at that layer)")
    print(f"{'layer':>6} " + " ".join(f"{'k=' + str(k):>14}" for k in ks) + "   (x = vs full-width baseline)")
    print("-" * 70)
    for a_i, li in enumerate(layers):
        cells = []
        for k_i in range(len(ks)):
            m = prj_mse[:, a_i, k_i].mean()
            cells.append(f"{m:.3e} {base_mse.mean() / m:>5.2f}x")
        print(f"{li + 1:>6} " + " ".join(f"{c:>14}" for c in cells))

    if args.jensen:
        print("\nWITH THE ORACLE JENSEN CORRECTION  (exact E[ReLU(z+eps)] at every downstream layer)")
        print(f"{'layer':>6} " + " ".join(f"{'k=' + str(k):>14}" for k in ks) + "   (x = vs full-width baseline)")
        print("-" * 70)
        for a_i, li in enumerate(layers):
            cells = []
            for k_i in range(len(ks)):
                m = jen_mse[:, a_i, k_i].mean()
                cells.append(f"{m:.3e} {base_mse.mean() / m:>5.2f}x" if m else f"{'--':>14}")
            print(f"{li + 1:>6} " + " ".join(f"{c:>14}" for c in cells))

    print("\nVARIANCE DROPPED at the next layer's pre-activation (what a Jensen correction must restore)")
    print(f"{'layer':>6} " + " ".join(f"{'k=' + str(k):>10}" for k in ks))
    print("-" * 44)
    for a_i, li in enumerate(layers):
        print(f"{li + 1:>6} " + " ".join(f"{prj_var[:, a_i, k_i].mean():>10.4f}" for k_i in range(len(ks))))

    ok = all(en_pca[:, a_i, ks.index(64)].mean() >= 0.95 for a_i, li in enumerate(layers) if li + 1 >= 16)
    print(f"\nSTAGE-1 GATE (>=95% energy in top 64 for layers >= 16): {'PASS' if ok else 'FAIL'}")
    if args.out:
        np.savez(args.out, en_pca=en_pca, en_cls=en_cls, en_grm=en_grm, prj_mse=prj_mse,
                 prj_var=prj_var, eff_rank=eff_rank, base_mse=base_mse, jen_mse=jen_mse,
                 layers=layers, ks=ks)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
