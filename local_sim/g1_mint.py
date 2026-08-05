"""
Track G1 — data mint for the amortized common-mode regressor.

48% of v1.1's final-layer error is a single per-MLP scalar: the whole layer
scaled by 1+g. Removing it exactly is worth 1.9x. B1 failed to estimate g because
its only reference (covariance propagation) has a scale error 3x larger than the
one we are trying to measure. G1 asks a different question: can a model trained
OFFLINE over many (MLP, sample-set) pairs predict g from cheap observables of the
sampling run itself?

Per (MLP, draw) we record:
  observables (all computable online, cheap):
    - per-layer empirical gain trajectory  mean||h_l||^2 / mean||h_{l-1}||^2
    - per-layer mean activation, and its ratio to the previous layer
    - a few weight summaries (per-layer Frobenius norm, spectral-ish proxy)
  targets (need truth, so offline only):
    - the realized final-layer error projected on the top error PCs; PC1 is the
      common mode, which is what we want to predict.

K independent draws per MLP is what makes g identifiable: the MLP is fixed, so
everything that varies across draws is the sample-set fluctuation we are after.

Run: .venv-gpu\\Scripts\\python.exe local_sim/g1_mint.py --mlps 4000 --draws 8
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
BITS = 30


def he_mlp(seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(int(seed))
    return torch.randn(DEPTH, WIDTH, WIDTH, device=dev, dtype=torch.float32, generator=g) * math.sqrt(2.0 / WIDTH)


def sobol_dirnum(dev):
    """Direction numbers, read out of the shipped estimator (single source of truth)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("v11_g1", os.path.join(_ROOT, "estimator_v1_1.py"))
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["v11_g1"] = mod
    spec.loader.exec_module(mod)
    d = mod._direction_numbers()
    return torch.tensor(np.array([d[b][:WIDTH] for b in range(BITS)], dtype=np.int64), device=dev)


def sobol_normals(n, vt, shift, dev):
    idx = torch.arange(n, device=dev, dtype=torch.int64)
    idx = idx ^ (idx >> 1)
    pts = torch.zeros(n, WIDTH, device=dev, dtype=torch.int64)
    for b in range(BITS):
        bit = (idx >> b) & 1
        pts = torch.bitwise_xor(pts, bit[:, None] * vt[b][None, :])
    pts = torch.bitwise_xor(pts, shift[None, :])
    u = (pts.double() + 0.5) / float(1 << BITS)
    u = u.clamp(1e-12, 1 - 1e-12)
    return (torch.erfinv(2 * u - 1) * math.sqrt(2.0)).float()


def truth_means(w, n_samples, gen, batch=32768):
    acc = torch.zeros(WIDTH, device=w.device, dtype=torch.float64)
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        h = torch.randn(b, WIDTH, device=w.device, dtype=torch.float32, generator=gen)
        for li in range(DEPTH):
            h = torch.relu(h @ w[li])
        acc += h.sum(0, dtype=torch.float64)
        done += b
    return acc / n_samples


def draw_observables(w, x):
    """One sampling run: returns final-layer estimate and the online observables."""
    h = x
    prev_sq = (h.double() ** 2).mean().item()
    gains, means = [], []
    for li in range(DEPTH):
        h = torch.relu(h @ w[li])
        sq = (h.double() ** 2).mean().item()
        gains.append(sq / max(prev_sq, 1e-30))
        means.append(h.mean().item())
        prev_sq = sq
    est = h.mean(0, dtype=torch.float64)
    return est, np.array(gains + means, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=4000)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--truth-samples", type=int, default=1 << 21)
    ap.add_argument("--seed0", type=int, default=31_000_000)
    a = ap.parse_args()

    dev = "cuda"
    vt = sobol_dirnum(dev)
    OBS, ERR, TRU = [], [], []
    t0 = time.perf_counter()
    for k in range(a.mlps):
        w = he_mlp(a.seed0 + k, dev)
        gen = torch.Generator(device=dev).manual_seed(a.seed0 + k + 900_000_000)
        t = truth_means(w, a.truth_samples, gen)

        obs_k, err_k = [], []
        for d in range(a.draws):
            g2 = torch.Generator(device=dev).manual_seed(a.seed0 + k * 1000 + d)
            shift = torch.randint(0, 1 << BITS, (WIDTH,), device=dev, dtype=torch.int64, generator=g2)
            x = sobol_normals(a.n, vt, shift, dev)
            est, obs = draw_observables(w, x)
            obs_k.append(obs)
            err_k.append((est - t).float().cpu().numpy())
        OBS.append(np.stack(obs_k))
        ERR.append(np.stack(err_k))
        TRU.append(t.float().cpu().numpy())
        if (k + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {k + 1}/{a.mlps}  {el:7.1f}s  eta {el / (k + 1) * (a.mlps - k - 1):7.1f}s", flush=True)

    OBS = np.stack(OBS)   # (M, K, F)
    ERR = np.stack(ERR)   # (M, K, 256)
    TRU = np.stack(TRU)   # (M, 256)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"g1_{a.mlps}x{a.draws}_N{a.n}.npz")
    np.savez_compressed(path, OBS=OBS, ERR=ERR, TRU=TRU, n=a.n, truth_samples=a.truth_samples)
    print(f"\nsaved {path}  OBS{OBS.shape} ERR{ERR.shape}  ({os.path.getsize(path) / 1e6:.1f} MB)")

    g = (ERR * TRU[:, None, :]).sum(-1) / (TRU**2).sum(-1)[:, None]
    print(f"realized scale mode g: sd {g.std():.5f}")
    print(f"share of error variance in the scale mode: "
          f"{1 - ((ERR - g[..., None] * TRU[:, None, :]) ** 2).mean() / (ERR**2).mean():.3f}")


if __name__ == "__main__":
    main()
