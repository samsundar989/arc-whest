"""Step 3.1 (HANDOFF 09-27): the closure on EXACT inputs vs exact post-ReLU moments.

    .venv-gpu/Scripts/python.exe tools/closure_bench.py --nets 0-99 --out experiments/<dir>

Runs the SHIPPED chain in the twin (torch shim, candidate v62_s1 + the S3 parameter set =
v67/v68 accuracy) with EVERY pre-activation object the closure reads -- mu, var, C11, D3,
D21, dG, D22, Q31 -- replaced by its exact label (keenanpepper full1000, N = 1e9) at every
layer, and taps the closure's output dict K. Each layer's K is then exactly
closure(exact inputs): the estimator's own `_wick`, term table, `_closure_apply`, per-term
gains and `pK -> K` code path, nothing re-implemented.

Compared per layer with the exact post-activation moments:
    K1  = E[a]                      full1000 `mean`
    K2  = Var a                     full1000 post_m2 - mean^2
    K11 = Cov(a_i, a_j), i != j     full1000 M11 - a a^T
    K3  = kappa3(a)                 full1000 post_m3 ...
    K4  = kappa4(a)                 full1000 post_m4 ...
    K21 = kappa(a_i, a_i, a_j)      bench-supp layer_LL post_K21 (N = 1e8)
    K22 = kappa(a_i,a_i,a_j,a_j)    bench-supp layer_LL post_K22 (N = 1e8)

Per layer: relative Frobenius residual ||K - K*|| / ||K*|| (pairwise objects off-diagonal).
Saved for the Step 3.2 regression, per net: per-neuron normalised inputs (alpha, gamma3,
gamma4, and the diagnostic-only exact pre kappa5/sigma^5, kappa6/sigma^6 from bench-supp
`pre_s`) with normalised K1/K2/K3/K4 residuals; and 8,192 fixed random off-diagonal pairs
per layer with (rho, d21, d12, d22, q31, q13, alpha_i, alpha_j, gamma3_i, gamma3_j) and the
normalised K11/K21/K12 residuals.

NOISE. K1..K4/K11 labels are N = 1e9 (floor ~1e-4 relative or below); K21/K22 are N = 1e8.
Their floor is measured on bench nets 0..7 from the independent `rep1/` re-bake when it is
present locally (`--rep1`), else quoted as not measured.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "twin"))
import twin as T  # noqa: E402  (puts the torch shim on the path, this process only)
import torch  # noqa: E402

BENCH = ROOT / "cache" / "hf" / "whest-p2-bakev2-bench-supp"
FULL = ROOT / "cache" / "hf" / "arc-whestbench-p2-full1000-N1e9" / "moments"
NPAIR = 8192
ALL = ("mu", "var", "C11", "D3", "D21", "dG", "D22", "Q31n")


class Tap:
    """`_A2`: exact inputs in, K recorded out."""

    def __init__(self, gidx):
        self.sub = T.Subst(gidx, ALL)
        self.K = {}

    def __call__(self, l, name, value, *aux):
        if name == "K":
            self.K[l] = {k: v.detach() for k, v in value.items()}
            return value
        if l < 1:                                   # layer 0 is exact Gaussian anyway
            return value if name != "mu" else self.sub.t["mu"][0].to(value.dtype)
        return self.sub(l, name, value, *aux)


def load_post(g):
    """One read per net: the post marginals and the dense post M11, (16, ...)."""
    z = np.load(FULL / f"mlp_{g:05d}.npz")
    return {k: np.asarray(z[k]) for k in ("mean", "post_m2", "post_m3", "post_m4", "M11")}


def post_truth(z, g, l, dev):
    a = np.asarray(z["mean"][l], np.float64)
    m2, m3, m4 = (np.asarray(z[k][l], np.float64) for k in ("post_m2", "post_m3", "post_m4"))
    c2 = m2 - a * a
    k3 = m3 - 3 * a * m2 + 2 * a ** 3
    c4 = m4 - 4 * a * m3 + 6 * a * a * m2 - 3 * a ** 4
    M11 = torch.from_numpy(np.asarray(z["M11"][l])).to(dev, torch.float64)
    at = torch.from_numpy(a).to(dev)
    K11 = M11 - at[:, None] * at[None, :]
    b = np.load(BENCH / "s00" / f"net_{g:05d}" / f"layer_{l:02d}.npz")
    t = {"K1": at, "K2": torch.from_numpy(c2).to(dev), "K3": torch.from_numpy(k3).to(dev),
         "K4": torch.from_numpy(c4 - 3 * c2 * c2).to(dev), "K11": K11,
         "K21": torch.from_numpy(np.asarray(b["post_K21"])).to(dev, torch.float64),
         "K22": torch.from_numpy(np.asarray(b["post_K22"])).to(dev, torch.float64)}
    return t


def pre_k56(g):
    """Exact pre kappa5, kappa6 (16, n) from bench-supp marg `pre_s` (shifted raw moments)."""
    m = np.load(BENCH / "s00" / f"net_{g:05d}" / "marg.npz")
    s = np.asarray(m["pre_s"], np.float64)            # s[p-1] = E[(z - c)^p]
    e1 = s[0]
    mu = [None, 0.0]
    from math import comb
    for p in range(2, 7):                             # central moments from shifted raw
        mu.append(sum(comb(p, k) * s[k - 1] * (-e1) ** (p - k) for k in range(1, p + 1))
                  + (-e1) ** p)
    m2, m3, m4, m5, m6 = mu[2], mu[3], mu[4], mu[5], mu[6]
    k5 = m5 - 10 * m3 * m2
    k6 = m6 - 15 * m4 * m2 - 10 * m3 * m3 + 30 * m2 ** 3
    return k5, k6


def relfro(x, y, off=None):
    d = x - y
    if off is not None:
        d, y = d * off, y * off
    return float(d.norm() / y.norm())


def run(a):
    tw = T.Twin(Path(a.cand))
    tw.install(T.load_params(a.params))
    data = T.Data()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.nets.split("-"))
    n = tw.width
    dev = T.DEV
    off = (1.0 - torch.eye(n, dtype=torch.float64, device=dev))
    g = torch.Generator(device="cpu").manual_seed(20260927)
    pi = torch.randint(0, n, (NPAIR,), generator=g)
    pj = (pi + torch.randint(1, n, (NPAIR,), generator=g)) % n
    pi, pj = pi.to(dev), pj.to(dev)
    table = {}
    for gi in range(lo, hi + 1):
        pnet = out / f"net_{gi:03d}.npz"
        if pnet.exists():
            continue
        t0 = time.time()
        w, _ = data.net("full", gi)
        tap = Tap(gi)
        tw.mod._A2 = tap
        with torch.no_grad():
            tw.forward(w)
        tw.mod._A2 = None
        lab = tap.sub.t
        k5, k6 = pre_k56(gi)
        zpost = load_post(gi)
        rows, uni, biv = {}, {}, {}
        # LAYER 0 IS THE NOISE FLOOR: its input is exactly Gaussian and the chain is exact
        # there, so K(layer 0) vs the labels is pure label noise (N = 1e9 marginals / M11,
        # N = 1e8 bench K21/K22), in the same relative units as every other row.
        for l in range(0, tw.depth):
            K = tap.K[l]
            tr = post_truth(zpost, gi, l, dev)
            last = (l == tw.depth - 1)
            r = {"K1": relfro(K[(1,)].double(), tr["K1"])}
            var = lab["var"][l].double()
            sig = var.clamp_min(1e-30).sqrt()
            mu = lab["mu"][l].double()
            alpha = mu / sig
            g3 = lab["D3"][l].double() / sig ** 3
            g4 = 2.0 * lab["dG"][l].double() / sig ** 4
            u = {"alpha": alpha, "gamma3": g3, "gamma4": g4,
                 "k5n": torch.from_numpy(k5[l]).to(dev) / sig ** 5,
                 "k6n": torch.from_numpy(k6[l]).to(dev) / sig ** 6,
                 "rK1": (K[(1,)].double() - tr["K1"]) / sig}
            if not last:
                r.update({"K2": relfro(K[(2,)].double(), tr["K2"]),
                          "K3": relfro(K[(3,)].double(), tr["K3"]),
                          "K4": relfro(K[(4,)].double(), tr["K4"]),
                          "K11": relfro(K[(1, 1)].double(), tr["K11"], off),
                          "K21": relfro(K[(2, 1)].double(), tr["K21"], off),
                          "K21T": relfro(K[(2, 1)].double().T, tr["K21"], off),
                          "K22": relfro(K[(2, 2)].double(), tr["K22"], off)})
                # the covariance residual AS THE NEXT LAYER RECEIVES IT: W^T dC W vs the exact
                # pre-activation var / C_off at l+1 -- directly comparable with Anatomy 2's
                # per-layer state-error increments (var ~1.3e-4, C_off ~6e-4 per layer)
                Wn = w[l + 1].double()
                dC = torch.diag(K[(2,)].double() - tr["K2"]) + (K[(1, 1)].double() - tr["K11"]) * off
                dCp = Wn.T @ dC @ Wn
                vn = lab["var"][l + 1].double()
                Cn = lab["C11"][l + 1].double()
                r["var_next"] = float(torch.diagonal(dCp).norm() / vn.norm())
                r["Coff_next"] = float((dCp * off).norm() / Cn.norm())
                u.update({"rK2": (K[(2,)].double() - tr["K2"]) / var,
                          "rK3": (K[(3,)].double() - tr["K3"]) / sig ** 3,
                          "rK4": (K[(4,)].double() - tr["K4"]) / sig ** 4})
                si, sj = sig[pi], sig[pj]
                C = lab["C11"][l].double()
                D21 = lab["D21"][l].double()
                D22 = lab["D22"][l].double()
                Q31 = lab["Q31"][l].double()
                biv = {k: v for k, v in biv.items()}
                bl = {"rho": C[pi, pj] / (si * sj),
                      "d21": D21[pi, pj] / (si * si * sj), "d12": D21[pj, pi] / (si * sj * sj),
                      "d22": D22[pi, pj] / (si * si * sj * sj),
                      "q31": Q31[pi, pj] / (si ** 3 * sj), "q13": Q31[pj, pi] / (si * sj ** 3),
                      "alpha_i": alpha[pi], "alpha_j": alpha[pj],
                      "gamma3_i": g3[pi], "gamma3_j": g3[pj],
                      "rK11": (K[(1, 1)].double()[pi, pj] - tr["K11"][pi, pj]) / (si * sj),
                      "rK21": (K[(2, 1)].double()[pi, pj] - tr["K21"][pi, pj]) / (si * si * sj),
                      "rK12": (K[(2, 1)].double()[pj, pi] - tr["K21"][pj, pi]) / (si * sj * sj)}
                for k, v in bl.items():
                    biv[f"{k}@{l}"] = v.float().cpu().numpy()
            for k, v in u.items():
                uni[f"{k}@{l}"] = v.float().cpu().numpy()
            rows[l] = r
            del tr
        np.savez(pnet, **uni, **biv, pi=pi.cpu().numpy(), pj=pj.cpu().numpy())
        table[gi] = rows
        (out / f"rel_{gi:03d}.json").write_text(json.dumps(rows))
        print(f"net {gi}: {time.time() - t0:.1f}s  L15 K1 {rows[15]['K1']:.2e}  "
              f"L8 K2 {rows[8]['K2']:.2e} K11 {rows[8]['K11']:.2e} K3 {rows[8]['K3']:.2e} "
              f"K21 {rows[8]['K21']:.2e} (T {rows[8]['K21T']:.2e}) | next var {rows[8]['var_next']:.2e} "
              f"Coff {rows[8]['Coff_next']:.2e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default=str(ROOT / "estimator_k3_v62_s1.py"))
    ap.add_argument("--params", default=str(ROOT / "experiments" / "2026-09-25_twin" / "S3"
                                            / "params_best.npz"))
    ap.add_argument("--nets", default="0-99")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
