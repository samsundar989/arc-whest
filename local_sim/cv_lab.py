"""
How much final-layer sampling variance can an EXACT-mean control variate remove?

Layer 1 is the only layer whose statistics we know in closed form: for x ~ N(0,I),
z1 = W1^T x is exactly Gaussian with covariance S = W1^T W1, so with s_i = sqrt(S_ii)

    E[ReLU(z1_i)]   = s_i / sqrt(2*pi)
    E[ReLU(z1_i)^2] = s_i^2 / 2

Any statistic of layer 1 is therefore a control variate with an EXACTLY known mean —
no bias introduced. This script measures the ceiling: how much of the final layer's
per-sample variance is linearly explainable by layer-1 statistics (and, for
reference, by ||x||^2 and by deeper layers whose means we do NOT know exactly).

Everything here is a per-sample covariance computation estimated from one large
sample set, so the numbers are population quantities, not R-replication noise.

Run:  uv run python local_sim/cv_lab.py [--samples 131072] [--mlps 3]
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

import real_data as rd

F32 = np.float32
CHUNK = 16384


def collect(w, n, seed, probe_layers=(0, 1, 2)):
    """Return per-sample final activations plus probe-layer activations and ||x||^2."""
    d = w[0].shape[1]
    fin = np.empty((n, d), dtype=F32)
    probes = {li: np.empty((n, d), dtype=F32) for li in probe_layers}
    r2 = np.empty(n, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for s in range(0, n, CHUNK):
        b = min(CHUNK, n - s)
        x = rng.standard_normal((b, d), dtype=F32)
        r2[s : s + b] = (x.astype(np.float64) ** 2).sum(axis=1)
        h = x
        for li, wi in enumerate(w):
            h = np.maximum(h @ wi, F32(0))
            if li in probes:
                probes[li][s : s + b] = h
        fin[s : s + b] = h
    return fin, probes, r2


def residual_var(y, c):
    """Per-column variance of y after regressing out control matrix c (both
    (n, ...) per-sample). Returns (var_before, var_after) averaged over columns.

    c=None means no control (variance is unchanged). The solve is done with
    lstsq so a rank-deficient control block degrades gracefully instead of
    raising.
    """
    y = y.astype(np.float64)
    yc = y - y.mean(axis=0)
    var_before = float(yc.var(axis=0).mean())
    if c is None:
        return var_before, var_before
    c = c.astype(np.float64)
    cc = c - c.mean(axis=0)
    Scc = cc.T @ cc / len(cc)
    Scy = cc.T @ yc / len(cc)
    ridge = 1e-9 * max(np.trace(Scc) / max(Scc.shape[0], 1), 1e-300)
    beta = np.linalg.lstsq(Scc + ridge * np.eye(Scc.shape[0]), Scy, rcond=None)[0]
    resid = yc - cc @ beta
    return var_before, float(resid.var(axis=0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=131072)
    ap.add_argument("--mlps", type=int, default=3)
    ap.add_argument("--n-eff", type=int, default=8192, help="N to quote MSE at")
    args = ap.parse_args()

    d = 256
    cost = 32 * 2 * d * d + 32 * d
    cb = args.n_eff * cost / 2.72e11

    print(f"per-sample variance decomposition, quoted as MSE at N={args.n_eff} (C/B={cb:.3f})\n")
    agg = {}
    for mi in range(args.mlps):
        w, truth, meta = rd.get(mi)
        fin, probes, r2 = collect(w, args.samples, seed=4242 + mi)
        h1 = probes[0]

        # exact-mean layer-1 scalars
        S = (w[0].astype(np.float64).T @ w[0].astype(np.float64))
        s = np.sqrt(np.maximum(np.diag(S), 0.0))
        exact_mean_h1 = s / np.sqrt(2 * np.pi)
        emp_mean_h1 = h1.mean(axis=0, dtype=np.float64)
        # express the closed-form check in units of the MC standard error of the
        # empirical mean — anything near 1 means the closed form is exact and we
        # are only seeing sampling noise.
        se = h1.astype(np.float64).std(axis=0) / np.sqrt(args.samples)
        rel = float(np.abs((emp_mean_h1 - exact_mean_h1) / se).max())
        s1 = h1.sum(axis=1, dtype=np.float64)  # exact mean known
        s2 = (h1.astype(np.float64) ** 2).sum(axis=1)  # exact mean known

        controls = {
            "none (plain MC)": None,
            "||x||^2": r2[:, None],
            "sum(h1)": s1[:, None],
            "sum(h1), sum(h1^2)": np.stack([s1, s2], axis=1),
            "sum(h1),sum(h1^2),||x||^2": np.stack([s1, s2, r2], axis=1),
            "full h1 vector (256)": h1,
            "[ceiling] full h2 (256)": probes[1],
            "[ceiling] full h3 (256)": probes[2],
        }
        print(
            f"MLP {mi} ({meta['mlp_name']})  layer-1 closed form vs MC: "
            f"worst neuron {rel:.2f} sigma  (~3 = pure sampling noise, so exact)"
        )
        print(f"    {'control variate':<28} {'resid var':>11} {'reduction':>10} {'MSE@N':>11} {'adjusted':>11}")
        for name, c in controls.items():
            v0, v = residual_var(fin, c)
            mse = v / args.n_eff
            agg.setdefault(name, []).append((v0, v))
            print(
                f"    {name:<28} {v:>11.5f} {v0 / v:>9.2f}x {mse:>11.3e} {mse * max(0.1, cb):>11.3e}"
            )
        print()

    print("=" * 78)
    print(f"{'control variate':<28} {'mean reduction':>15} {'mean adjusted @N':>18}")
    print("-" * 78)
    for name, vals in agg.items():
        v0 = float(np.mean([a for a, _ in vals]))
        v = float(np.mean([b for _, b in vals]))
        print(f"{name:<28} {v0 / v:>14.2f}x {v / args.n_eff * max(0.1, cb):>18.3e}")
    print("\n[ceiling] rows use layers whose exact means we do NOT know — upper bounds only.")


if __name__ == "__main__":
    main()
