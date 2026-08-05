"""
Track G2 — optimise the base point set directly against the task loss.

Every point-set variant we have tried so far was chosen on theory (Sobol, Owen
scrambling, Korobov lattices, radial stratification, antithetic pairing) and they
all land within about +/-15% of plain digitally-shifted Sobol. G2 asks the
different question: if we optimise the points themselves against the actual
final-layer error, averaged over minted He MLPs, how much is there to win?

Parameters: the base point matrix in normal space, initialised from the shipped
Sobol set. Loss: mean final-layer squared error against GPU truth over a fresh
batch of minted MLPs each step. Regularisation, so the set stays a valid sampler:
  * exact per-coordinate mean 0 and variance 1 re-imposed after every step
  * an L2 penalty on drift from the Sobol initialisation
Held-out MLPs (never trained on) are evaluated periodically; the gap between
train-batch and held-out loss is the ensemble-overfit warning the plan asks for.

Run: .venv-gpu\\Scripts\\python.exe local_sim/g2_optimize.py --steps 300
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
WIDTH, DEPTH, BITS = 256, 32, 30


def he_batch(seeds, dev):
    ws = []
    for s in seeds:
        g = torch.Generator(device=dev).manual_seed(int(s))
        ws.append(torch.randn(DEPTH, WIDTH, WIDTH, device=dev, dtype=torch.float32, generator=g)
                  * math.sqrt(2.0 / WIDTH))
    return ws


def sobol_init(n, dev):
    import sys

    sys.path.insert(0, _HERE)
    from make_sobol_blob import direction_numbers, sobol_uint

    vt = direction_numbers()
    pts = sobol_uint(n, vt).astype(np.float64)
    u = (pts + 0.5) / float(1 << BITS)
    from scipy.special import ndtri

    return torch.tensor(ndtri(u), device=dev, dtype=torch.float32)


def truth_of(w, n_samples, gen, batch=32768):
    acc = torch.zeros(WIDTH, device=w.device, dtype=torch.float64)
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        h = torch.randn(b, WIDTH, device=w.device, dtype=torch.float32, generator=gen)
        for li in range(DEPTH):
            h = torch.relu(h @ w[li])
        acc += h.sum(0, dtype=torch.float64)
        done += b
    return (acc / n_samples).float()


def est_of(w, x):
    h = x
    for li in range(DEPTH):
        h = torch.relu(h @ w[li])
    return h.mean(0)


def normalise_(X):
    """DISABLED -- measured to be actively harmful.

    Forcing exact per-coordinate mean 0 / variance 1 was the plan's suggested
    regulariser, and it alone (with no gradient steps at all) costs 2.35x:
    9.156e-6 -> 2.152e-5 on held-out MLPs. The Sobol set's sample moments are
    mean -0.00149, std 1.00278; standardising them applies a ~0.3% rescale to
    every point, and a systematic 0.3% scale error on activations of ~0.45 is
    far larger than the ~1e-6 MSE we are chasing. A QMC set is unbiased because
    it is the correct quantile transform of an equidistributed net, NOT because
    its sample moments are exactly (0, 1) -- moment-matching trades a negligible
    variance reduction for a first-order bias. Drift regularisation only."""
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096, help="point-set size (smaller = faster to train)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8, help="minted MLPs per step")
    ap.add_argument("--held", type=int, default=24)
    ap.add_argument("--truth-samples", type=int, default=1 << 20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--drift", type=float, default=1e-3)
    a = ap.parse_args()

    dev = "cuda"
    X0 = sobol_init(a.n, dev)
    X = X0.clone().requires_grad_(True)
    opt = torch.optim.Adam([X], lr=a.lr)

    # fixed held-out set, never trained on
    held_seeds = [90_000_000 + i for i in range(a.held)]
    held_w = he_batch(held_seeds, dev)
    held_t = [truth_of(w, a.truth_samples, torch.Generator(device=dev).manual_seed(s + 7)) for w, s in zip(held_w, held_seeds)]

    def held_loss(pts):
        with torch.no_grad():
            return float(np.mean([((est_of(w, pts) - t) ** 2).mean().item() for w, t in zip(held_w, held_t)]))

    base = held_loss(X0)
    print(f"n={a.n}  held-out MLPs={a.held}\nSobol baseline held-out MSE {base:.4e}\n")
    print(f"{'step':>6} {'train batch':>13} {'held-out':>13} {'vs Sobol':>9} {'drift':>8}")
    print("-" * 56)

    t0 = time.perf_counter()
    for step in range(a.steps + 1):
        seeds = [50_000_000 + step * a.batch + i for i in range(a.batch)]
        ws = he_batch(seeds, dev)
        ts = [truth_of(w, 1 << 18, torch.Generator(device=dev).manual_seed(s + 13)) for w, s in zip(ws, seeds)]
        loss = sum(((est_of(w, X) - t) ** 2).mean() for w, t in zip(ws, ts)) / len(ws)
        loss = loss + a.drift * ((X - X0) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        normalise_(X.data)
        if step % 25 == 0:
            h = held_loss(X.detach())
            drift = float(((X.detach() - X0) ** 2).mean().sqrt())
            print(f"{step:>6} {loss.item():>13.4e} {h:>13.4e} {base / h:>8.3f}x {drift:>8.4f}", flush=True)

    h = held_loss(X.detach())
    print(f"\nfinal held-out {h:.4e}   vs Sobol {base / h:.3f}x   ({time.perf_counter() - t0:.0f}s)")
    print(f"ship gate >=1.3x: {'PASS' if base / h >= 1.3 else 'FAIL'}")
    np.savez_compressed(os.path.join(OUT, f"g2_points_{a.n}.npz"), X=X.detach().cpu().numpy())


if __name__ == "__main__":
    main()
