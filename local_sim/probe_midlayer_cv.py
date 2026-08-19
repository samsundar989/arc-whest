"""STAGE-0 PROBE P0.1 -- kill or license the mid-layer control variate, by identity.

The lane reduces to one measurable question.  Our closure already produces a
mean vector at every layer (rows 1..31 of the estimator's own output).  Using it
as the analytic centre of a control variate on the layer-c activations gives

    mu_hat = mean(a_D) - (mean(a_c) - m_c^closure) @ M_{c->D}

with M_{c->D} = prod_{l=c+1..D} W_l diag(Phi_l), the closure's own linearised
error-transport operator -- free, since _closure already computes Phi.

Its deployable gain is bounded by the analytic centre's fidelity:

    G_c = 1 / [ (1 - rho^2) + rho^2 r_c^2 ],   r_c = |delta_c| / SE_c

where delta_c = m_c^closure - E[a_c] is the closure's error at layer c and SE_c
is OUR sampling error on that same mean at N=16384.  If r_c > 1 for every c the
lane is closed by identity: no ridge, no cross-fit, no clipping can lift G above
1, because the correction is less accurate than the thing it corrects.

Rather than fit anything, this probe DEPLOYS the correction and measures the
paired MSE ratio directly -- exactly what would ship -- and reports r_c
alongside so the result has a mechanism attached.

DESIGN TIER.  No gate, no decision block, estimator.py untouched.
Seeds: D16 unused tail.  Run:
    .venv-gpu\\Scripts\\python.exe local_sim/probe_midlayer_cv.py
"""

import os
import sys
import time
import json

import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mint_provenance import canonical_mint  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"
BITS, DIM, DEPTH, N = 30, 256, 32, 16384
N_MLPS, N_SHIFTS, N_GT = 20, 16, 1 << 23
SEED0 = 30_806_016_400
LAYERS = (1, 2, 4, 6, 8, 12, 16, 24)

_dn = np.load(os.path.join(_HERE, "sobol_dirnum.npz"))
DIRNUM = _dn[_dn.files[0]]


def base_points(n, w):
    idx = np.arange(n, dtype=np.uint32)
    idx = idx ^ (idx >> np.uint32(1))
    idx = idx.reshape((n, 1))
    pts = np.zeros((n, w), dtype=np.uint32)
    for k in range(BITS):
        pts ^= ((idx >> np.uint32(k)) & np.uint32(1)) * DIRNUM[k][:w].astype(np.uint32)
    return pts


PTS = base_points(N, DIM)


def sobol(seed):
    rng = np.random.default_rng(seed)
    sh = rng.integers(0, 1 << BITS, size=DIM, dtype=np.uint32)
    u = ((PTS ^ sh).astype(np.float64) + 0.5) * (1.0 / float(1 << BITS))
    return norm.ppf(u).astype(np.float32)


def closure(w64):
    """estimator.py:_closure -- post-ReLU means and the firing gates."""
    mu = np.zeros(DIM)
    cov = np.eye(DIM)
    off = np.ones((DIM, DIM)) - np.eye(DIM)
    mus, phis = [], []
    for w in w64:
        mp = w.T @ mu
        cp = w.T @ cov @ w
        vp = np.maximum(np.diag(cp), 1e-12)
        sg = np.sqrt(vp)
        a = mp / sg
        Phi = ndtr(a)
        phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
        mu = mp * Phi + sg * phi
        ez2 = (mp * mp + vp) * Phi + mp * sg * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
        cov = np.outer(Phi, Phi) * cp * off + np.diag(var)
        mus.append(mu.copy())
        phis.append(Phi.copy())
    return mus, phis


def rotation(w64, phis):
    """estimator.py:_alignment_basis -- the shipped rank-32 reproducible basis."""
    J = w64[0]
    for k in range(1, DEPTH):
        J = (J * phis[k - 1]) @ w64[k]
    _, vecs = np.linalg.eigh(J @ J.T)
    top = vecs[:, DIM - 32:][:, ::-1]
    hi, lo = top.max(axis=0), top.min(axis=0)
    top = top * np.where(hi >= -lo, 1.0, -1.0)
    q, r = np.linalg.qr(top, mode="complete")
    d = np.diag(r)
    return q * np.concatenate([np.where(d < 0, -1.0, 1.0), np.ones(DIM - 32)])


def transport(w64, phis, c):
    """M_{c->D} = prod_{l=c+1..D-1} W_l diag(Phi_l): the closure's own linearised
    error-transport operator from post-ReLU layer c to the scored layer."""
    M = np.eye(DIM)
    for l in range(c + 1, DEPTH):
        M = (M @ w64[l]) * phis[l]
    return M


def forward_all(X, W, want):
    """Return the sample-mean post-ReLU activation at each layer in `want`."""
    out = {}
    h = X
    for li in range(DEPTH):
        h = torch.clamp(h @ W[li], min=0.0)
        if li in want:
            out[li] = h.mean(dim=0, dtype=torch.float64)
    return out


def truth(W, want):
    acc = {li: torch.zeros(DIM, dtype=torch.float64, device=DEV) for li in want}
    B = 1 << 17
    g = torch.Generator(device=DEV)
    g.manual_seed(12345)
    for _ in range(N_GT // B):
        h = torch.randn(B, DIM, generator=g, device=DEV, dtype=torch.float32)
        for li in range(DEPTH):
            h = torch.clamp(h @ W[li], min=0.0)
            if li in acc:
                acc[li] += h.sum(dim=0, dtype=torch.float64)
    return {li: a / N_GT for li, a in acc.items()}


def main():
    want = set(LAYERS) | {DEPTH - 1}
    plain, corr = [], {c: [] for c in LAYERS}
    rc, dl, se = {c: [] for c in LAYERS}, {c: [] for c in LAYERS}, {c: [] for c in LAYERS}
    t0 = time.time()

    for mi in range(N_MLPS):
        Wn, _, _ = canonical_mint(DIM, DEPTH, SEED0 + mi)
        w64 = [x.astype(np.float64) for x in Wn]
        mus, phis = closure(w64)
        Bs = rotation(w64, phis)

        # shipped carrier: rotation folded into W1 (pruning omitted -- it changes
        # cost, not the CV's variance question, and keeps the arms exactly paired)
        Wr = [np.ascontiguousarray((Bs.T @ w64[0]).astype(np.float32))] + \
             [np.ascontiguousarray(x) for x in Wn[1:]]
        W = [torch.from_numpy(x).to(DEV) for x in Wr]
        t = truth(W, want)

        M = {c: torch.from_numpy(transport(w64, phis, c)).to(DEV) for c in LAYERS}
        mc = {c: torch.from_numpy(mus[c]).to(DEV) for c in LAYERS}

        for c in LAYERS:
            dl[c].append(float(torch.sqrt(((mc[c] - t[c]) ** 2).mean())))

        for si in range(N_SHIFTS):
            X = torch.from_numpy(sobol(900_000 + si)).to(DEV)
            a = forward_all(X, W, want)
            e_fin = a[DEPTH - 1] - t[DEPTH - 1]
            plain.append(float((e_fin ** 2).mean()))
            for c in LAYERS:
                se[c].append(float(torch.sqrt(((a[c] - t[c]) ** 2).mean())))
                fix = (a[c] - mc[c]) @ M[c]
                corr[c].append(float(((e_fin - fix) ** 2).mean()))
        print(f"  mlp {mi+1}/{N_MLPS}  {time.time()-t0:.0f}s", flush=True)

    P = np.array(plain).reshape(N_MLPS, N_SHIFTS)
    rng = np.random.default_rng(0)
    print(f"\n{N_MLPS} MLPs x {N_SHIFTS} shifts, N={N}, GT 2^23, shipped rotation on")
    print(f"plain mean MSE = {P.mean():.4e}\n")
    print(f"{'layer c':>8}{'delta_c':>12}{'SE_c':>12}{'r_c':>9}"
          f"{'deployed G_c':>15}{'95% CI':>18}{'wins':>10}")
    rows = {}
    for c in LAYERS:
        C = np.array(corr[c]).reshape(N_MLPS, N_SHIFTS)
        r = P / C
        per = np.exp(np.log(r).mean(axis=1))
        bs = [np.exp(np.log(rng.choice(per, N_MLPS)).mean()) for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        d_, s_ = float(np.mean(dl[c])), float(np.mean(se[c]))
        g = float(np.exp(np.log(r).mean()))
        rows[c] = dict(delta=d_, se=s_, r=d_ / s_, G=g, lo=lo, hi=hi,
                       wins=int((r > 1).sum()))
        print(f"{c:>8}{d_:>12.3e}{s_:>12.3e}{d_/s_:>9.2f}"
              f"{g:>14.4f}x  [{lo:.3f},{hi:.3f}]{rows[c]['wins']:>7}/{r.size}")

    best = max(rows, key=lambda c: rows[c]["G"])
    print(f"\nbest layer c={best}: G={rows[best]['G']:.4f}x "
          f"[{rows[best]['lo']:.3f},{rows[best]['hi']:.3f}], r_c={rows[best]['r']:.2f}")
    print("VERDICT:", "LICENSE -- some c has r_c < 1 and G_c CI excludes 1.0"
          if (min(v["r"] for v in rows.values()) < 1.0 and rows[best]["lo"] > 1.0)
          else "CLOSED BY IDENTITY -- the analytic centre is less accurate than "
               "the sampling estimate it would correct")
    json.dump({str(c): rows[c] for c in LAYERS},
              open(os.path.join(_ROOT, "experiments",
                                "2026-08-19_carrier_probes", "midlayer_cv.json"), "w"),
              indent=1)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
