"""
B3 — is the calibrated-covprop residual LEARNABLE?

Trains a small per-neuron regressor on GPU-minted data to predict
b_i = truth_i - p_i from covprop's own diagnostics plus weight-matrix statistics,
and reports R^2 on HELD-OUT MLPs (split by MLP, never by neuron — splitting
neurons would leak per-MLP structure and inflate R^2).

It also answers the question that actually matters, which is not R^2 itself but
what R^2 BUYS. If a learned model shrinks covprop's error variance to
(1-R^2)*Var(b), the best achievable precision-weighted fusion with the sampling
estimator becomes

    MSE_fused = Vs * Vc' / (Vs + Vc'),   Vc' = (1-R^2) * Var(b)

so the table at the end converts R^2 into the final-layer MSE we would actually
score. Note Var(b) is ~18x the sampling variance, so even R^2 = 0.5 moves the
fused score only ~10%: the useful threshold is far higher than it looks.

Run: .venv-gpu\\Scripts\\python.exe local_sim/gpu_train_probe.py --data <npz>
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA = os.path.join(_ROOT, "experiments", "fusion_data")

FEATURE_NAMES = [
    "mu_pre", "sig", "alpha", "Phi", "var", "p",
    "wL_colnorm", "wL_rownorm", "wLm1_colnorm", "p_prev",
]


def build_features(X):
    """(m, n, f) raw -> (m, n, f') with per-MLP global context appended."""
    m, n, f = X.shape
    glob = np.concatenate(
        [X.mean(axis=1), X.std(axis=1), np.percentile(X, 90, axis=1)], axis=1
    )  # (m, 3f)
    glob_b = np.broadcast_to(glob[:, None, :], (m, n, glob.shape[1]))
    return np.concatenate([X, glob_b], axis=2).astype(np.float32)


class Net(nn.Module):
    def __init__(self, d_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--sampling-var", type=float, default=1.4893e-06,
                    help="v1.1 held-out final-layer MSE, for the fusion table")
    a = ap.parse_args()

    path = a.data or sorted(glob.glob(os.path.join(DATA, "gpu_mint_*.npz")))[-1]
    blob = np.load(path)
    X, Y = blob["X"], blob["Y"]
    n_samples = int(blob["n_samples"]) if "n_samples" in blob else 0
    m, n, _ = X.shape
    print(f"data {os.path.basename(path)}: {m} MLPs x {n} neurons, MC labels at {n_samples} samples")

    Xf = build_features(X)
    d_in = Xf.shape[2]
    nh = int(m * (1 - a.holdout_frac))
    print(f"features {d_in}  |  train MLPs {nh}, held-out MLPs {m - nh}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xtr = torch.tensor(Xf[:nh].reshape(-1, d_in), device=dev)
    ytr = torch.tensor(Y[:nh].reshape(-1), device=dev)
    xte = torch.tensor(Xf[nh:].reshape(-1, d_in), device=dev)
    yte = torch.tensor(Y[nh:].reshape(-1), device=dev)

    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-8)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    ysc = ytr.std()
    print(f"target sd (covprop residual) = {ysc.item():.4e}   Var = {ysc.item() ** 2:.4e}")

    # ---- baseline: best constant, and a linear model ----------------------
    var_te = yte.var().item()
    Xd = torch.cat([xtr, torch.ones(len(xtr), 1, device=dev)], 1)
    beta = torch.linalg.lstsq(Xd, ytr.unsqueeze(1)).solution
    lin_pred = torch.cat([xte, torch.ones(len(xte), 1, device=dev)], 1) @ beta
    r2_lin = 1 - ((yte - lin_pred.squeeze(1)) ** 2).mean().item() / var_te
    print(f"\nlinear model      held-out R^2 = {r2_lin:.4f}")

    # ---- small MLP --------------------------------------------------------
    model = Net(d_in).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    bs = 1 << 16
    ntr = len(xtr)
    best = -1e9
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(ntr, device=dev)
        for i in range(0, ntr, bs):
            idx = perm[i : i + bs]
            loss = ((model(xtr[idx]) - ytr[idx]) ** 2).mean() / (ysc**2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pr = torch.cat([model(xte[i : i + bs]) for i in range(0, len(xte), bs)])
            r2 = 1 - ((yte - pr) ** 2).mean().item() / var_te
        best = max(best, r2)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:>3}  held-out R^2 = {r2:.4f}")
    print(f"\nMLP regressor     held-out R^2 = {best:.4f}")

    # ---- what does that R^2 buy? -----------------------------------------
    varb = yte.var().item()
    vs = a.sampling_var
    print(f"\ncovprop residual variance Var(b) = {varb:.4e}")
    print(f"sampling (v1.1) variance   Vs    = {vs:.4e}   ratio {varb / vs:.1f}x\n")
    print(f"  {'R^2':>6} {'Vc after correction':>21} {'best fused MSE':>16} {'vs v1.1':>9}")
    print("-" * 58)
    for r2 in (0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, best):
        vc = (1 - r2) * varb
        fused = vs * vc / (vs + vc) if vc > 0 else 0.0
        tag = "  <-- measured" if abs(r2 - best) < 1e-9 else ""
        print(f"  {r2:>6.3f} {vc:>21.4e} {fused:>16.4e} {100 * (1 - fused / vs):>+8.1f}%{tag}")

    verdict = "Phase C ALIVE" if best > 0.5 else ("inconclusive" if best > 0.15 else "DROP Phase C")
    print(f"\nverdict at the mission's thresholds: {verdict}  (R^2={best:.3f})")


if __name__ == "__main__":
    main()
