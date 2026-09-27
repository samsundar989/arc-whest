"""Step 3.3a (HANDOFF 09-27): offline fit of the closure-correction nets on EXACT-input pairs.

    .venv-gpu/Scripts/python.exe tools/closure_fit.py --bench experiments/<dir>/bench --out <npz>

Targets are the corrections the closure MISSED, `(exact - closure)/sigma^p`, from
`tools/closure_bench.py` (FIT nets; train 0..79, report R^2 on 80..99; every FIT net is inside
the twin's TRAIN rows, HOLDOUT-100 is untouched).

  uni  per neuron:        (alpha, gamma3, gamma4, l/16) -> 16 -> 16 -> (dK1, dK2, dK3, dK4)
                          dK2 is normalised by var, dKp by sigma^p; layer 15 trains dK1 only
  biv  per ordered pair:  (rho, alpha_i, alpha_j, gamma3_i, gamma3_j, d21, d12, d22, l/16)
                          -> 16 -> (dK11, dK21), normalised by sigma_i sigma_j / sigma_i^2 sigma_j

q31/q13 (in the regression) are dropped: the chain forms the normalised Q31 inside
`_closure_apply`, after the point where the correction is applied. The output npz is what
`tools/build_clos.py` embeds; this is an INITIALISER -- the handoff forbids shipping it
untuned (the twin fine-tunes it end to end).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

UNI_F = ("alpha", "gamma3", "gamma4")
UNI_T = ("rK1", "rK2", "rK3", "rK4")
BIV_F = ("rho", "alpha_i", "alpha_j", "gamma3_i", "gamma3_j", "d21", "d12", "d22")
BIV_T = ("rK11", "rK21")


def load(bench, nets, feats, targets, layers, biv):
    X, Y, M = [], [], []
    for g in nets:
        p = bench / f"net_{g:03d}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        for l in layers:
            if f"{feats[0]}@{l}" not in z.files:
                continue
            f = [z[f"{k}@{l}"] for k in feats]
            n = len(f[0])
            f.append(np.full(n, l / 16.0, np.float32))
            ys, ms = [], []
            for t in targets:
                k = f"{t}@{l}"
                if k in z.files:
                    ys.append(-z[k])                 # the correction = minus the residual
                    ms.append(np.ones(n, np.float32))
                else:                                # layer 15: only rK1 exists
                    ys.append(np.zeros(n, np.float32))
                    ms.append(np.zeros(n, np.float32))
            X.append(np.stack(f, 1))
            Y.append(np.stack(ys, 1))
            M.append(np.stack(ms, 1))
    X, Y, M = (np.concatenate(a).astype(np.float64) for a in (X, Y, M))
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    return X[ok], Y[ok], M[ok]


def fit(X, Y, M, Xv, Yv, Mv, hidden, deep, steps, dev, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    ys = np.sqrt((Y ** 2 * M).sum(0) / np.maximum(M.sum(0), 1)) + 1e-30   # output scale
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    xt, yt, mt = t((X - mu) / sd), t(Y / ys), t(M)
    xv, yv, mv = t((Xv - mu) / sd), t(Yv / ys), t(Mv)
    layers = [torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh()]
    if deep:
        layers += [torch.nn.Linear(hidden, hidden), torch.nn.Tanh()]
    layers += [torch.nn.Linear(hidden, Y.shape[1])]
    net = torch.nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for s in range(steps):
        idx = torch.randint(0, len(xt), (65536,), generator=g).to(dev)
        loss = (((net(xt[idx]) - yt[idx]) ** 2) * mt[idx]).sum() / mt[idx].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        p = net(xv)
        r2 = 1 - (((p - yv) ** 2) * mv).sum(0) / (((yv - (yv * mv).sum(0) / mv.sum(0)) ** 2) * mv).sum(0)
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    W = {f"W{i + 1}": m.weight.detach().cpu().numpy().T.astype(np.float32) for i, m in enumerate(lin)}
    W.update({f"b{i + 1}": m.bias.detach().cpu().numpy().astype(np.float32) for i, m in enumerate(lin)})
    W.update({"fmu": mu.astype(np.float32), "fisd": (1.0 / sd).astype(np.float32),
              "osd": ys.astype(np.float32)})
    return W, r2.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    bench = Path(a.bench)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = range(0, 80), range(80, 100)
    Xu, Yu, Mu = load(bench, tr, UNI_F, UNI_T, range(1, 16), False)
    Xuv, Yuv, Muv = load(bench, va, UNI_F, UNI_T, range(1, 16), False)
    Wu, r2u = fit(Xu, Yu, Mu, Xuv, Yuv, Muv, 16, True, a.steps, dev)
    print("uni R2 (FIT 80-99):", dict(zip(UNI_T, np.round(r2u, 3))), flush=True)
    Xb, Yb, Mb = load(bench, tr, BIV_F, BIV_T, range(1, 15), True)
    Xbv, Ybv, Mbv = load(bench, va, BIV_F, BIV_T, range(1, 15), True)
    Wb, r2b = fit(Xb, Yb, Mb, Xbv, Ybv, Mbv, 16, False, a.steps, dev)
    print("biv R2 (FIT 80-99):", dict(zip(BIV_T, np.round(r2b, 3))), flush=True)
    out = {f"u{k}": v for k, v in Wu.items()}
    out.update({f"b{k}": v for k, v in Wb.items()})
    np.savez(a.out, **out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"uni_r2": dict(zip(UNI_T, map(float, r2u))), "biv_r2": dict(zip(BIV_T, map(float, r2b)))}))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
