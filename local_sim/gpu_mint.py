"""
B3 — GPU data mint + learnability probe (runs in .venv-gpu, NEVER imported by
any estimator file).

Mints He-init 256x32 MLPs on the GPU, computes high-sample Monte Carlo labels for
the final-layer means, runs the same Gaussian-closure covariance propagation the
shipped estimator uses, and stores the per-neuron features + the calibrated
covprop residual b_i = truth_i - p_i.

The question B3 answers: is that residual LEARNABLE?
  R^2 > 0.5  -> a learned error model can replace B1's static one; Phase C alive
  R^2 < 0.15 -> drop Phase C; the GPU pivots to B2 calibration data

Stage 0 (--parity) verifies the GPU forward pass agrees with the CPU/flopscope
path on real dataset MLPs before we trust 50k synthetic ones.

Run with the GPU interpreter:
    .venv-gpu\\Scripts\\python.exe local_sim/gpu_mint.py --parity
    .venv-gpu\\Scripts\\python.exe local_sim/gpu_mint.py --mint 50000 --samples 65536
"""

import argparse
import math
import os
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT = os.path.join(_ROOT, "experiments", "fusion_data")

WIDTH, DEPTH = 256, 32
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

# v0's per-layer calibration, mirrored from estimator_v0_archive.py
CAL = [
    1.0, 0.998414, 0.997486, 0.996835, 0.996253, 0.995831, 0.995492,
    0.994881, 0.994833, 0.994597, 0.994315, 0.994229, 0.993970, 0.993850,
    0.993429, 0.993266, 0.993268, 0.993313, 0.992924, 0.992802, 0.992655,
    0.992586, 0.992734, 0.992925, 0.992730, 0.992784, 0.992899, 0.993107,
    0.993026, 0.993091, 0.993310, 0.993142,
]


def ndtr(x):
    return 0.5 * torch.erfc(-x * (1.0 / math.sqrt(2.0)))


def npdf(x):
    return INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def mc_final_means(weights, n_samples, gen, batch=16384, dtype=torch.float32):
    """Monte Carlo per-neuron post-ReLU means for every layer. weights: (D,n,n)."""
    dev = weights.device
    n = weights.shape[-1]
    acc = torch.zeros(weights.shape[0], n, device=dev, dtype=torch.float64)
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        h = torch.randn(b, n, device=dev, dtype=dtype, generator=gen)
        for li in range(weights.shape[0]):
            h = torch.relu(h @ weights[li])
            acc[li] += h.sum(dim=0, dtype=torch.float64)
        done += b
    return acc / float(n_samples)


def covprop_gpu(weights, calibrate=True):
    """Gaussian closure, mirroring estimator_v0_archive.py. weights: (D,n,n) f64."""
    dev = weights.device
    n = weights.shape[-1]
    depth = weights.shape[0]
    mu = torch.zeros(n, device=dev, dtype=torch.float64)
    cov = torch.eye(n, device=dev, dtype=torch.float64)
    off = 1.0 - torch.eye(n, device=dev, dtype=torch.float64)
    rows, diag = [], {}
    for li in range(depth):
        w = weights[li]
        mu_pre = w.T @ mu
        cov_pre = w.T @ cov @ w
        var_pre = torch.clamp(torch.diagonal(cov_pre), min=1e-12)
        sig = torch.sqrt(var_pre)
        alpha = mu_pre / sig
        Phi, phi = ndtr(alpha), npdf(alpha)
        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var = torch.clamp(ez2 - mu * mu, min=0.0)
        cov = torch.outer(Phi, Phi) * cov_pre * off + torch.diag(var)
        cov = 0.5 * (cov + cov.T)
        rows.append(mu.clone())
        if li == depth - 1:
            diag = dict(mu_pre=mu_pre, sig=sig, alpha=alpha, Phi=Phi, var=var)
    pred = torch.stack(rows)
    if calibrate:
        cal = torch.tensor(CAL[:depth], device=dev, dtype=torch.float64)
        pred = pred * cal[:, None]
    diag["rows"] = pred
    diag["p"] = pred[-1]
    return diag


def he_mlp(seed, width=WIDTH, depth=DEPTH, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(int(seed))
    scale = math.sqrt(2.0 / width)
    return torch.randn(depth, width, width, device=dev, dtype=torch.float32, generator=g) * scale


# ------------------------------------------------------------------ parity ---
def parity(n_mlps=5, n_samples=262144):
    """GPU forward pass vs the CPU path on REAL dataset MLPs.

    Reads MLPs exported from the main env (the GPU venv deliberately has no
    whestbench/datasets — it stays a torch-only sandbox).
    """
    import sys

    sys.path.insert(0, _HERE)
    blob = np.load(os.path.join(OUT, "parity_mlps.npz"))
    W_all, T_all = blob["W"], blob["T"]
    n_mlps = min(n_mlps, len(W_all))

    dev = "cuda"
    print(f"parity check on {n_mlps} real dataset MLPs, {n_samples} samples\n")
    print(f"{'mlp':>4} {'max|GPU-CPU| means':>20} {'GPU vs 1e9 truth MSE':>22} {'covprop max diff':>18}")
    print("-" * 70)
    ok = True
    for i in range(n_mlps):
        w_np, truth = W_all[i].astype(np.float64), T_all[i].astype(np.float64)
        w = torch.tensor(w_np, device=dev, dtype=torch.float32)
        gen = torch.Generator(device=dev).manual_seed(1234 + i)
        gpu_means = mc_final_means(w, n_samples, gen).cpu().numpy()

        # CPU reference forward pass, same sample count, numpy RNG
        rng = np.random.default_rng(1234 + i)
        acc = np.zeros((DEPTH, WIDTH))
        done = 0
        while done < n_samples:
            b = min(16384, n_samples - done)
            h = rng.standard_normal((b, WIDTH), dtype=np.float32)
            for li in range(DEPTH):
                h = np.maximum(h @ w_np[li].astype(np.float32), np.float32(0))
                acc[li] += h.sum(axis=0, dtype=np.float64)
            done += b
        cpu_means = acc / n_samples

        d_means = float(np.abs(gpu_means[-1] - cpu_means[-1]).max())
        mse_truth = float(((gpu_means[-1] - truth[-1]) ** 2).mean())

        from covprop_features import covprop

        cp_cpu = covprop(w_np)["p"]
        cp_gpu = covprop_gpu(torch.tensor(w_np, device=dev, dtype=torch.float64))["p"].cpu().numpy()
        d_cov = float(np.abs(cp_cpu - cp_gpu).max())

        flag = "" if d_means < 5e-3 else "  <-- TOO BIG"
        ok &= d_means < 5e-3 and d_cov < 1e-9
        print(f"{i:>4} {d_means:>20.3e} {mse_truth:>22.3e} {d_cov:>18.3e}{flag}")
    print(f"\nparity {'PASSED' if ok else 'FAILED'} (means agree to ~1e-3 as required; covprop to 1e-9)")
    return ok


# -------------------------------------------------------------------- mint ---
def mint(n_mlps, n_samples, seed0=7_000_000, chunk_report=250):
    dev = "cuda"
    feats, targs = [], []
    t0 = time.perf_counter()
    for k in range(n_mlps):
        w32 = he_mlp(seed0 + k, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(seed0 + k + 500_000_000)
        truth = mc_final_means(w32, n_samples, gen)[-1]  # (n,)

        d = covprop_gpu(w32.double())
        p = d["p"]
        b = truth - p  # the residual we want to learn

        wL = w32[-1].double()
        wLm1 = w32[-2].double()
        f = torch.stack(
            [
                d["mu_pre"], d["sig"], d["alpha"], d["Phi"], d["var"], p,
                wL.pow(2).sum(0).sqrt(),          # incoming column norm, last layer
                wL.pow(2).sum(1).sqrt(),          # outgoing row norm, last layer
                wLm1.pow(2).sum(0).sqrt(),        # column norm, penultimate
                d["rows"][-2],                    # covprop mean, penultimate layer
            ],
            dim=1,
        )
        feats.append(f.float().cpu())
        targs.append(b.float().cpu())
        if (k + 1) % chunk_report == 0:
            el = time.perf_counter() - t0
            print(f"  {k + 1}/{n_mlps}  {el:7.1f}s  eta {el / (k + 1) * (n_mlps - k - 1):7.1f}s", flush=True)

    X = torch.stack(feats).numpy()
    Y = torch.stack(targs).numpy()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"gpu_mint_{n_mlps}x{n_samples}.npz")
    np.savez_compressed(path, X=X, Y=Y, n_samples=n_samples)
    print(f"\nsaved {path}  X{X.shape} Y{Y.shape}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--mint", type=int, default=0)
    ap.add_argument("--samples", type=int, default=65536)
    a = ap.parse_args()
    if a.parity:
        parity()
    if a.mint:
        mint(a.mint, a.samples)
