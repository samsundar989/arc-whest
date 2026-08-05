"""
Track G1 — train the amortized common-mode regressor, honestly.

Goal: predict the per-MLP scale fluctuation g (the common mode carrying 48% of
v1.1's error) from observables the estimator can compute online, so it can be
divided out. Removing g exactly would be worth 1.9x.

Three guards make this safe to ship, because a corrector that is wrong is worse
than no corrector:
  1. CROSS-FITTING. Predictions for an MLP always come from a model that never
     saw that MLP (K-fold over MLPs, not over draws).
  2. JAMES-STEIN SHRINKAGE. The prediction is multiplied by the out-of-fold R^2,
     so a useless model shrinks to zero and adds nothing. With shrinkage s and
     true correlation rho, the variance removed is s*(2*rho*sd_g*sd_pred - s*var_pred),
     which is never negative at the optimal s = rho*sd_g/sd_pred.
  3. The reported number is the MSE reduction actually realised on held-out
     (MLP, draw) pairs, not R^2.

Run: .venv-gpu\\Scripts\\python.exe local_sim/g1_train.py --data <npz>
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


class Head(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()

    path = a.data or sorted(glob.glob(os.path.join(DATA, "g1_*.npz")))[-1]
    blob = np.load(path)
    OBS, ERR, TRU = blob["OBS"], blob["ERR"], blob["TRU"]
    M, K, F = OBS.shape
    print(f"{os.path.basename(path)}: {M} MLPs x {K} draws, {F} observables")

    # target: realised scale mode of each draw
    g = (ERR * TRU[:, None, :]).sum(-1) / (TRU**2).sum(-1)[:, None]  # (M, K)
    err_var = float((ERR**2).mean())
    mode_share = 1 - float(((ERR - g[..., None] * TRU[:, None, :]) ** 2).mean()) / err_var
    print(f"error variance {err_var:.4e}   scale-mode share {100 * mode_share:.1f}%   sd(g) {g.std():.5f}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(OBS.reshape(M * K, F), dtype=torch.float32, device=dev)
    Y = torch.tensor(g.reshape(M * K), dtype=torch.float32, device=dev)
    mlp_id = torch.arange(M, device=dev).repeat_interleave(K)

    # standardise using all data (features are observables, not targets)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-8)
    X = (X - mu) / sd

    oof = torch.zeros_like(Y)
    for f in range(a.folds):
        te = (mlp_id % a.folds) == f
        tr = ~te
        model = Head(F).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        xt, yt = X[tr], Y[tr]
        for _ in range(a.epochs):
            model.train()
            perm = torch.randperm(len(xt), device=dev)
            for i in range(0, len(xt), 8192):
                idx = perm[i : i + 8192]
                loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            oof[te] = model(X[te])
        print(f"  fold {f}: held-out corr {torch.corrcoef(torch.stack([oof[te], Y[te]]))[0, 1].item():+.4f}")

    r2 = 1 - ((Y - oof) ** 2).mean().item() / Y.var().item()
    rho = torch.corrcoef(torch.stack([oof, Y]))[0, 1].item()
    print(f"\nout-of-fold R^2 = {r2:.4f}   corr = {rho:+.4f}")

    # James-Stein style shrinkage: scale the prediction so it cannot add variance
    s_opt = float(rho * Y.std().item() / max(oof.std().item(), 1e-12)) if rho > 0 else 0.0
    shrink = max(0.0, min(s_opt, 1.0))
    print(f"shrinkage factor = {shrink:.4f}")

    ghat = (shrink * oof).reshape(M, K).cpu().numpy()
    corrected = ERR - ghat[..., None] * TRU[:, None, :]
    new_var = float((corrected**2).mean())
    print(f"\nfinal-layer error variance:")
    print(f"  before            {err_var:.4e}")
    print(f"  after correction  {new_var:.4e}   ({err_var / new_var:.3f}x)")
    print(f"  oracle (g known)  {float(((ERR - g[..., None] * TRU[:, None, :]) ** 2).mean()):.4e}   "
          f"({err_var / float(((ERR - g[..., None] * TRU[:, None, :]) ** 2).mean()):.3f}x ceiling)")

    gain = err_var / new_var
    print(f"\nship gate is >=1.3x on held-out minted MLPs: {'PASS' if gain >= 1.3 else 'FAIL'} ({gain:.3f}x)")


if __name__ == "__main__":
    main()
