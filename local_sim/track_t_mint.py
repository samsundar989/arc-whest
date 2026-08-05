"""
Track T, data collection (GPU; runs in .venv-gpu, NEVER imported by an estimator).

What Track T is asking
----------------------
v2.4's final-layer estimate is the plain sample mean `mean(ReLU(z))`, where z is
the layer-32 pre-activation of the pruned, rotated pipeline. The alternative is
to *close* the last step analytically: treat z as Gaussian per neuron and use the
exact rectified-Gaussian mean

    a_i = mu_i * Phi(mu_i/sigma_i) + sigma_i * phi(mu_i/sigma_i)

with (mu, sigma^2) estimated from the same samples. This is the k=1 analytic tail
that `b2_tail_probe.py` measured at -67%, and thread 18097's finding — that the
same closure *from exact moments* beats sampling — says the whole gap is
moment-estimation noise. Track T tests whether that noise can be shrunk away.

An identity worth stating, because it collapses the design space:

    diag(W^T Sigma_hat W) = mean_s(z^2) - mean_s(z)^2        (exactly)

where Sigma_hat is the sample covariance of the layer-31 activations. Only the
DIAGONAL of the propagated covariance enters the final-layer means, so there is
no 256x256 covariance to estimate and no Marchenko-Pastur regime to fix — the
"covariance noise" is 256 independent scalar second moments. Linear shrinkage of
Sigma toward a structured target therefore reduces exactly to a scalar blend of
two per-neuron variance estimates, which is what we fit.

What this script produces
-------------------------
For each MLP, R independently digitally-shifted Sobol draws at N=16384 through
the exact v2.4 pipeline (closure -> prune at tau -> eig(JJ^T) rotation folded into
W1), storing per draw and per output neuron:

    m1 = mean(z)      m2 = mean(z^2)      s = mean(ReLU(z))      [the v2.4 output]

Draw 0 uses the shipped shift, so its `s` reproduces the graded estimator exactly.
Pooling all R draws gives a reference at R*N samples, which serves as the
"exact moments" arm; the spread across draws gives the bias/variance split.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/track_t_mint.py --data <mini100.npz> \\
        --out <track_t_mini.npz> --draws 256
"""

import argparse
import math
import os
import time

import numpy as np
import torch

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
SOBOL_BITS = 30
PRUNE_TAU = -2.5


def ndtr_t(x):
    return 0.5 * torch.erfc(-x * (1.0 / math.sqrt(2.0)))


def npdf_t(x):
    return INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def closure(w, dev):
    """Gaussian closure in f64 on GPU. w: (D,n,n) f64 tensor.

    Returns per-layer alpha, post-ReLU mean mu, firing prob Phi, plus the FINAL
    layer's closure pre-activation mean and variance (the shrinkage target)."""
    d, n, _ = w.shape
    mu = torch.zeros(n, dtype=torch.float64, device=dev)
    cov = torch.eye(n, dtype=torch.float64, device=dev)
    off = 1.0 - torch.eye(n, dtype=torch.float64, device=dev)
    alphas, mus, phis = [], [], []
    mu_pre_last = var_pre_last = None
    for li in range(d):
        wl = w[li]
        mu_pre = wl.T @ mu
        cov_pre = wl.T @ cov @ wl
        var_pre = torch.clamp(torch.diagonal(cov_pre), min=1e-12)
        sig = torch.sqrt(var_pre)
        a = mu_pre / sig
        Phi = ndtr_t(a)
        phi = npdf_t(a)
        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var = torch.clamp(ez2 - mu * mu, min=0.0)
        cov = torch.outer(Phi, Phi) * cov_pre * off + torch.diag(var)
        cov = 0.5 * (cov + cov.T)
        alphas.append(a)
        mus.append(mu)
        phis.append(Phi)
        if li == d - 1:
            mu_pre_last, var_pre_last = mu_pre, var_pre
    return (torch.stack(alphas), torch.stack(mus), torch.stack(phis),
            mu_pre_last, var_pre_last)


def alignment_basis(w, phis):
    """Leading eigenvectors of J J^T, J = W1 D1 W2 D2 ... (v2.4's rotation)."""
    d = w.shape[0]
    J = w[0]
    for k in range(1, d):
        J = (J * phis[k - 1].unsqueeze(0)) @ w[k]
    _, vecs = torch.linalg.eigh(J @ J.T)
    return torch.flip(vecs, dims=[1])


def sobol_base(n, dim, dev):
    """Unshifted Sobol integers (n, dim) uint32, via scipy's direction numbers."""
    from scipy.stats import _sobol

    v = np.zeros((dim, SOBOL_BITS), dtype=np.uint32)
    _sobol._initialize_v(v, dim, SOBOL_BITS)
    vt = np.ascontiguousarray(v.T)  # (bits, dim)
    idx = np.arange(n, dtype=np.uint32)
    idx = (idx ^ (idx >> np.uint32(1)))[:, None]
    pts = np.zeros((n, dim), dtype=np.uint32)
    for b in range(SOBOL_BITS):
        pts = np.bitwise_xor(pts, ((idx >> np.uint32(b)) & np.uint32(1)) * vt[b])
    return torch.from_numpy(pts.astype(np.int64)).to(dev)


def draw_points(base, shift, dev):
    """XOR digital shift -> uniforms -> standard normals, matching the estimator
    (f64 quantile transform, then cast to f32 for the forward pass)."""
    u = (torch.bitwise_xor(base, shift).to(torch.float64) + 0.5) * (1.0 / float(1 << SOBOL_BITS))
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return z.to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz from export_split.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=256)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--tau", type=float, default=PRUNE_TAU)
    ap.add_argument("--mlps", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False  # f32 must mean f32
    torch.backends.cudnn.allow_tf32 = False

    d = np.load(args.data, allow_pickle=True)
    W_all, T_all, seeds = d["weights"], d["truth"], d["seeds"]
    n_mlps = len(W_all) if args.mlps == 0 else min(args.mlps, len(W_all))
    depth, width = W_all.shape[1], W_all.shape[2]
    print(f"device={dev}  {n_mlps} MLPs  depth={depth} width={width}  "
          f"draws={args.draws} N={args.n} tau={args.tau}", flush=True)

    base = sobol_base(args.n, width, dev)
    R = args.draws

    out = {k: np.zeros((n_mlps, R, width), dtype=np.float64) for k in ("m1", "m2", "s")}
    aux = {
        "mu_c": np.zeros((n_mlps, width)),      # closure pre-activation mean, final layer
        "v_c": np.zeros((n_mlps, width)),       # closure pre-activation variance (shrink target)
        "p_c": np.zeros((n_mlps, width)),       # closure post-ReLU mean (the v0 analytic prediction)
        "truth": np.zeros((n_mlps, width)),
        "n_alive": np.zeros((n_mlps, depth), dtype=np.int64),
        "cost": np.zeros(n_mlps),               # v2.4 per-sample matmul cost (flopscope pricing)
    }

    t0 = time.perf_counter()
    for i in range(n_mlps):
        w32 = torch.from_numpy(W_all[i]).to(dev)
        w64 = w32.to(torch.float64)
        alphas, mus, phis, mu_pre_last, var_pre_last = closure(w64, dev)
        Q = alignment_basis(w64, phis)

        # --- v2.4 pruning: alive sets from the closure alpha, last layer full ---
        alive, dead = [], []
        for li in range(depth):
            if li == depth - 1:
                alive.append(torch.arange(width, device=dev))
                dead.append(None)
                continue
            keep = alphas[li] > args.tau
            idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                idx = torch.argmax(alphas[li]).reshape(1)
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
            k = width if prev is None else int(prev.numel())
            cost += (2 * k - 1) * int(cur.numel())
            prev = cur
        aux["cost"][i] = cost
        aux["n_alive"][i] = np.array([int(a.numel()) for a in alive])

        gen = np.random.default_rng(int(seeds[i]))
        for r in range(R):
            shift_np = gen.integers(0, 1 << SOBOL_BITS, size=width, dtype=np.uint32)
            shift = torch.from_numpy(shift_np.astype(np.int64)).to(dev)
            h = draw_points(base, shift, dev)
            for li in range(depth):
                z = h @ sliced[li]
                if offsets[li] is not None:
                    z = z + offsets[li]
                if li == depth - 1:
                    zf = z.to(torch.float64)
                    out["m1"][i, r] = zf.mean(dim=0).cpu().numpy()
                    out["m2"][i, r] = (zf * zf).mean(dim=0).cpu().numpy()
                    out["s"][i, r] = torch.clamp(zf, min=0.0).mean(dim=0).cpu().numpy()
                else:
                    h = torch.clamp(z, min=0.0)

        aux["mu_c"][i] = mu_pre_last.cpu().numpy()
        aux["v_c"][i] = var_pre_last.cpu().numpy()
        aux["p_c"][i] = mus[-1].cpu().numpy()
        aux["truth"][i] = T_all[i][-1]
        if (i + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{n_mlps}  {el:.0f}s  ({el / (i + 1):.2f}s/MLP)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, draws=R, n=args.n, tau=args.tau, **out, **aux)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB) in "
          f"{time.perf_counter() - t0:.0f}s")

    # immediate sanity: draw 0 is the shipped estimator, so its MSE must match
    s0 = out["s"][:, 0, :]
    print(f"draw-0 (shipped shift) final-layer MSE = {np.mean((s0 - aux['truth']) ** 2):.4e}"
          f"   [v2.4 on the same split measured 1.6079e-6 for mini]")


if __name__ == "__main__":
    main()
