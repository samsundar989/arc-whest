"""P2-CAL, GPU executor: fit and evaluate in-loop per-layer K2 state calibration.

Contract: `experiments/LOG.md`, entry dated 2026-08-26, committed BEFORE this
file produced any number. DESIGN tier, no seed claim, no `estimator.py` edit,
no promotion.

## The verification chain, which is the only reason any number here is believable

    instrumented fnp arm  (local_sim/p2_arms/arm_k2p1.py, the shipped spelling)
        |  verified to 4e-30 relative mean square on the scored row
    NumPy mirror          (local_sim/p2_cal_lab.py)
        |  verified here, same metric, printed before any fit runs
    batched torch/CUDA float64 closure   (this file)

If the third link is not printed as PASS, this module refuses to fit. A fast
executor that has not been chained back to the instrumented arm is a random
number generator with good manners.

## Why the fit is a single sequential sweep and not a search

The truth file carries the true post-ReLU mean for **every** layer, not just the
scored one. So the per-layer gain has a closed form: at layer k, with the
upstream state already corrected, the least-squares scalar that puts the
predicted mean row onto the true mean row is

    g_k = sum_nets <p_k, t_k> / sum_nets <p_k, p_k>

Sweeping k = 0..15 and applying each gain before propagating gives all 16
parameters in ONE forward pass. A coordinate search over the same 16 parameters
would cost several hundred passes to land in the same place, and would invite the
suspicion that the answer came from the search rather than from the data.

`g_0` is **pinned to 1.0** and this is a mechanism decision, not a convenience:
the layer-1 post-ReLU mean is EXACT in closed form (`||W_1[:,j]|| / sqrt(2 pi)`,
because `z_1 = x W_1` is exactly Gaussian with zero mean). Scaling an exact
quantity can only hurt it. Layer 0's *covariance* is not exact -- its
off-diagonal is a Price-order-1 truncation -- which is what the separate
off-diagonal arm exists to probe.

## Memory

This host had ~1 GB of free RAM when the module was written, so weights are
staged through a float32 disk memmap and streamed to the GPU one layer at a time
in float64. Nothing large is ever held in system RAM.

Run (GPU venv):
    .venv-gpu\\Scripts\\python.exe local_sim/p2_cal_gpu.py --verify
    .venv-gpu\\Scripts\\python.exe local_sim/p2_cal_gpu.py \\
        --out experiments/2026-08-26_p2_cal/cal.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.append(_p)

from mint_provenance import canonical_mint  # noqa: E402

TRUTH = os.path.join(_ROOT, "experiments", "2026-08-21_p2_cp1", "truth_P01.npz")
CACHE = os.path.join(_ROOT, "experiments", "2026-08-26_p2_cal", "weights_P01.f32")

_INV_SQRT_2PI = 0.3989422804014327
_SQRT_HALF = 0.7071067811865476


# --------------------------------------------------------------------------
# Weight staging
# --------------------------------------------------------------------------

def stage_weights(seeds, width=1024, depth=16, path=CACHE, force=False):
    """Mint every network once into a float32 memmap of shape (n, depth, w, w)."""
    n = len(seeds)
    shape = (n, depth, width, width)
    if os.path.exists(path) and not force:
        want = n * depth * width * width * 4
        if os.path.getsize(path) == want:
            return np.memmap(path, dtype=np.float32, mode="r", shape=shape)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
    t0 = time.time()
    for i, s in enumerate(seeds):
        w, _, _ = canonical_mint(width, depth, int(s))
        for k in range(depth):
            mm[i, k] = w[k]
        if (i + 1) % 16 == 0:
            print(f"  minted {i+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    mm.flush()
    del mm
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape)


# --------------------------------------------------------------------------
# The batched closure
# --------------------------------------------------------------------------

def _ncdf(x):
    return 0.5 * torch.erfc(-x * _SQRT_HALF)


def _npdf(x):
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def closure_batch(Wmm, idx, dev, *, mean_gain=None, var_gain=None, offdiag_gain=None,
                  depth=16, width=1024, hook=None):
    """K2p1 closure over a batch of networks, float64 on `dev`.

    `Wmm[idx]` are the networks. Gains default to 1.0. `hook(k, mu, cov)` is
    called after layer k has produced its state and BEFORE the gains for layer k
    are applied; the sequential fitter uses it to read the uncorrected row and
    return the gain to apply.

    Returns `(b, depth, width)` post-ReLU means with layer 0 replaced by its
    exact closed form, matching the shipped estimator.
    """
    b = len(idx)
    one = torch.ones(depth, dtype=torch.float64, device=dev)
    mg = one if mean_gain is None else torch.as_tensor(mean_gain, dtype=torch.float64, device=dev)
    vg = one if var_gain is None else torch.as_tensor(var_gain, dtype=torch.float64, device=dev)
    og = one if offdiag_gain is None else torch.as_tensor(offdiag_gain, dtype=torch.float64, device=dev)

    mu = torch.zeros(b, width, dtype=torch.float64, device=dev)
    cov = torch.eye(width, dtype=torch.float64, device=dev).expand(b, width, width).clone()
    eye = torch.eye(width, dtype=torch.float64, device=dev)
    off = 1.0 - eye
    rows = []
    for k in range(depth):
        wk = torch.from_numpy(np.ascontiguousarray(Wmm[idx, k])).to(dev).double()
        wkT = wk.transpose(1, 2)
        mu_pre = torch.bmm(wkT, mu.unsqueeze(2)).squeeze(2)
        cov_pre = torch.bmm(torch.bmm(wkT, cov), wk)
        var_pre = torch.clamp(torch.diagonal(cov_pre, dim1=1, dim2=2), min=1e-12)
        sig = torch.sqrt(var_pre)
        alpha = mu_pre / sig
        Phi = _ncdf(alpha)
        phi = _npdf(alpha)

        mu = mu_pre * Phi + sig * phi
        ez2 = (mu_pre * mu_pre + var_pre) * Phi + mu_pre * sig * phi
        var = torch.clamp(ez2 - mu * mu, min=0.0)

        acc = (Phi.unsqueeze(2) * Phi.unsqueeze(1)) * cov_pre * off
        cov = acc + torch.diag_embed(var)
        del wk, wkT, cov_pre, acc

        gk = mg[k]
        if hook is not None:
            gk = hook(k, mu)
        if float(gk) != 1.0:
            mu = mu * gk
            cov = cov * (gk * gk)
        if float(vg[k]) != 1.0:
            d = torch.diagonal(cov, dim1=1, dim2=2) * vg[k]
            cov = cov * off + torch.diag_embed(d)
        if float(og[k]) != 1.0:
            d = torch.diagonal(cov, dim1=1, dim2=2).clone()
            cov = cov * off * og[k] + torch.diag_embed(d)
        rows.append(mu)

    out = torch.stack(rows, dim=1)
    w0 = torch.from_numpy(np.ascontiguousarray(Wmm[idx, 0])).to(dev).double()
    out[:, 0, :] = torch.sqrt((w0 * w0).sum(dim=1)) * _INV_SQRT_2PI
    return out


def run_all(Wmm, dev, n, chunk=16, **kw):
    outs = []
    for a in range(0, n, chunk):
        idx = np.arange(a, min(a + chunk, n))
        outs.append(closure_batch(Wmm, idx, dev, **kw).cpu())
        torch.cuda.empty_cache()
    return torch.cat(outs, 0)


# --------------------------------------------------------------------------
# Sequential closed-form fit
# --------------------------------------------------------------------------

def fit_sequential_gains(Wmm, dev, fit_idx, T, depth=16, chunk=16, pin_first=True):
    """One forward sweep over the fit set, solving for g_k layer by layer.

    Chunking forces the sweep to be done per chunk with a shared gain vector, so
    the gain for layer k is solved on the FIRST chunk and then applied to the
    rest. With the chunk drawn from a common random-number ordering this is a
    valid estimator, but it uses less data than the whole fit set. To avoid that
    the sweep is run once per layer over ALL fit chunks: `depth` passes, each
    reusing the gains already fixed for layers < k.
    """
    g = np.ones(depth)
    Tf = T[fit_idx]
    for k in range(depth):
        if pin_first and k == 0:
            continue
        num = 0.0
        den = 0.0
        for a in range(0, len(fit_idx), chunk):
            idx = fit_idx[a:a + chunk]
            got = closure_batch(Wmm, idx, dev, mean_gain=g)[:, k, :]
            t = torch.as_tensor(Tf[a:a + chunk, k, :], dtype=torch.float64, device=dev)
            num += float((got * t).sum())
            den += float((got * got).sum())
            torch.cuda.empty_cache()
        g[k] = num / den
        print(f"  layer {k:>2}  g = {g[k]:.6f}", flush=True)
    return g


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def xmse_final(P, y1, y2, idx):
    p = P[idx, -1, :].numpy().astype(np.float64)
    return float(np.mean((p - y1[idx, -1, :]) * (p - y2[idx, -1, :])))


def per_mlp_xmse_final(P, y1, y2, idx):
    p = P[idx, -1, :].numpy().astype(np.float64)
    return [float(np.mean((p[j] - y1[i, -1, :]) * (p[j] - y2[i, -1, :])))
            for j, i in enumerate(idx)]


def paired_t(a, b):
    """Paired arithmetic difference a-b with a studentised-t 95% interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    # normal quantile is adequate at n>=30; use it and say so
    return {"mean_diff": m, "se": se, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "n": n}


# --------------------------------------------------------------------------

def verify_against_numpy(Wmm, dev, seeds, tol=1e-20):
    """Chain link 2->3: torch batched closure vs the verified NumPy mirror."""
    import p2_cal_lab as lab
    out = []
    for i in (0, 1):
        w = [np.asarray(Wmm[i, k]) for k in range(Wmm.shape[1])]
        mine = lab.predict(w)
        got = closure_batch(Wmm, np.array([i]), dev)[0].cpu().numpy()
        rel = float(np.mean((got[-1] - mine[-1]) ** 2) / np.mean(mine[-1] ** 2))
        rel_all = float(np.mean((got - mine) ** 2) / np.mean(mine ** 2))
        out.append({"net": i, "seed": int(seeds[i]), "rel_ms_final_layer": rel,
                    "rel_ms_all_layers": rel_all, "pass": rel <= tol})
        print(f"  net {i} seed {int(seeds[i])}  rel mean-square final {rel:.3e}  "
              f"all {rel_all:.3e}  {'PASS' if rel <= tol else 'FAIL'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(TRUTH, allow_pickle=True)
    seeds = z["seeds"][: args.n]
    y1 = z["means_r1"][: args.n].astype(np.float64)
    y2 = z["means_r2"][: args.n].astype(np.float64)
    gt_noise = z["gt_noise"][: args.n]
    T = 0.5 * (y1 + y2)  # the fit target; evaluation never uses this
    n = len(seeds)
    print(f"P2-CAL/GPU  1024x16  {n} networks  device={dev}  "
          f"N={int(z['n_samples']):,}/replica", flush=True)

    print("\nstaging weights (float32 memmap; system RAM is not used) ...", flush=True)
    Wmm = stage_weights(seeds)

    print("\nCHAIN LINK 2->3  torch batched closure vs verified NumPy mirror:", flush=True)
    ver = verify_against_numpy(Wmm, dev, seeds)
    if not all(v["pass"] for v in ver):
        raise SystemExit("torch closure does not reproduce the NumPy mirror -- refusing to fit")
    if args.verify:
        return

    nfit = int(args.fit_frac * n)
    fit_idx = np.arange(0, nfit)
    hold_idx = np.arange(nfit, n)
    print(f"\nsplit: fit {nfit}, held out {n-nfit}", flush=True)

    res = {"n": n, "n_fit": nfit, "device": dev, "verification_chain": ver,
           "gt_noise_mean": float(gt_noise.mean()),
           "gt_noise_max": float(gt_noise.max()),
           "n_samples_per_replica": int(z["n_samples"]), "arms": {}}

    def record(name, P, extra=None):
        r = {"xmse_fit": xmse_final(P, y1, y2, fit_idx),
             "xmse_hold": xmse_final(P, y1, y2, hold_idx),
             "mse_r1_hold": float(np.mean((P[hold_idx, -1].numpy() - y1[hold_idx, -1]) ** 2)),
             "mse_r2_hold": float(np.mean((P[hold_idx, -1].numpy() - y2[hold_idx, -1]) ** 2)),
             "per_mlp_xmse_hold": per_mlp_xmse_final(P, y1, y2, hold_idx)}
        if extra:
            r.update(extra)
        res["arms"][name] = r
        print(f"  {name:<4} fit {r['xmse_fit']:.4e}  held-out {r['xmse_hold']:.4e}  "
              f"(r1 {r['mse_r1_hold']:.4e} r2 {r['mse_r2_hold']:.4e})", flush=True)
        return r

    t0 = time.time()
    print("\nC0 control ...", flush=True)
    P0 = run_all(Wmm, dev, n, chunk=args.chunk)
    record("C0", P0)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

    print("\nC2 sequential per-layer in-loop gain fit (g_0 pinned to 1) ...", flush=True)
    g = fit_sequential_gains(Wmm, dev, fit_idx, T, chunk=args.chunk)
    P2c = run_all(Wmm, dev, n, chunk=args.chunk, mean_gain=g)
    record("C2", P2c, {"layer_gains": [float(x) for x in g]})

    print("\nC1 single global in-loop gain (geometric mean of the C2 gains, "
          "then a 1-D refinement scan) ...", flush=True)
    g0 = float(np.exp(np.mean(np.log(g[1:]))))
    best, bestv, curve = g0, np.inf, []
    for cand in np.linspace(g0 - 0.004, g0 + 0.004, 9):
        gg = np.ones(16)
        gg[1:] = cand
        Pc = run_all(Wmm, dev, nfit, chunk=args.chunk, mean_gain=gg)
        v = float(np.mean((Pc[fit_idx, -1].numpy() - T[fit_idx, -1]) ** 2))
        curve.append([float(cand), v])
        if v < bestv:
            bestv, best = v, float(cand)
        print(f"    g={cand:.6f}  fit-set MSE {v:.4e}", flush=True)
    gg = np.ones(16)
    gg[1:] = best
    P1 = run_all(Wmm, dev, n, chunk=args.chunk, mean_gain=gg)
    record("C1", P1, {"global_gain": best, "scan_curve": curve})

    print("\nP2 post-hoc per-layer output gain (the mechanism control) ...", flush=True)
    pf = P0[fit_idx].numpy()
    num = np.einsum("ndw,ndw->d", pf, T[fit_idx])
    den = np.einsum("ndw,ndw->d", pf, pf)
    pg = num / den
    Pph = P0 * torch.as_tensor(pg, dtype=torch.float64)[None, :, None]
    record("P2", Pph, {"post_gains": [float(x) for x in pg]})

    c0 = res["arms"]["C0"]
    print("\nheld-out improvement over C0 (cross-product MSE on the scored row):")
    for k in ("C1", "C2", "P2"):
        r = res["arms"][k]
        pt = paired_t(c0["per_mlp_xmse_hold"], r["per_mlp_xmse_hold"])
        r["paired_vs_C0"] = pt
        print(f"  {k}: {c0['xmse_hold']/r['xmse_hold']:.3f}x   paired diff "
              f"{pt['mean_diff']:.3e}  95% [{pt['lo']:.3e}, {pt['hi']:.3e}]")
    print(f"\nGT noise floor: mean {res['gt_noise_mean']:.4e}  max {res['gt_noise_max']:.4e}")
    print(f"total {time.time()-t0:.0f}s")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
