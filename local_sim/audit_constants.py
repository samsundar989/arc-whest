"""
CONSTANTS AUDIT — paying off the overfitting debt.

Every frozen number in `estimator.py` was chosen at some point, on some data,
against some objective. Most were chosen on the public mini split, which we have
since decided is the wrong instrument: it steered a month of decisions, so its
optimum is partly a memory of our own search. The private re-run does not care
what was optimal on the mini split; it cares what is optimal on networks nobody
has seen.

So: re-measure each constant on **freshly minted, never-reused seeds**, and ask
one question — *is the frozen value inside the measurement error of the fresh
optimum?* Not "is there a better value" (there always is, by noise), but "is the
difference real?"

Two design choices make that question answerable at the ~1% level, which a
single 100-MLP run cannot reach (two realisations of the same estimator differ
~18% in mean MSE):

  PAIRED, COMMON RANDOM NUMBERS. Every configuration sees the same MLPs and the
  same digital shifts. The comparison is then on per-MLP *differences*, whose
  variance is a small fraction of the variance of either arm.

  MULTI-SHIFT AVERAGING. Each MLP is scored as the mean over R independent
  digital shifts, which removes the point-set realisation noise instead of
  measuring it.

The reported error bar is the standard error of the paired difference across
MLPs. A constant "drifts" only if the fresh optimum beats the frozen value by
more than ~2 of those.

Seed blocks (never reused; see seed_blocks.py):
    D01 prune tau      D02 rotation rank    D03 C/B operating point
    D04 patch bias     D05 v0 calibration

Run (GPU venv):
    .venv-gpu\\Scripts\\python.exe local_sim/audit_constants.py --which all --mlps 200 --draws 32
"""

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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mint_provenance import canonical_mint, seed_streams  # noqa: E402
from track_t_mint import closure, draw_points, sobol_base  # noqa: E402

WIDTH, DEPTH = 256, 32
BUDGET = 2.72e11
FIXED_FLOPS = 4.51e9 + 9.84e8

# frozen values under audit
FROZEN = {"tau": -2.5, "rank": 32, "cb_target": 0.30, "n_points": 16384}

BLOCKS = {"tau": 30_806_001_000, "rank": 30_806_002_000, "cb": 30_806_003_000,
          "patch": 30_806_004_000, "cal": 30_806_005_000}


# ---------------------------------------------------------------- pipeline ---
def basis_trunc(w64, phis, rank):
    """v2.4.1's rotation: top-`rank` eigenvectors of JJ^T, signs pinned, completed by QR."""
    J = w64[0]
    for k in range(1, w64.shape[0]):
        J = (J * phis[k - 1].unsqueeze(0)) @ w64[k]
    _, vecs = torch.linalg.eigh(J @ J.T)
    width = vecs.shape[0]
    k = min(int(rank), width)
    top = torch.flip(vecs[:, width - k:], dims=[1])
    hi, lo = top.max(dim=0).values, top.min(dim=0).values
    top = top * torch.where(hi >= -lo, 1.0, -1.0)
    q, r = torch.linalg.qr(top, mode="complete")
    d = torch.diagonal(r)[:k]
    sgn = torch.where(d < 0.0, -torch.ones_like(d), torch.ones_like(d))
    return q * torch.cat([sgn, torch.ones(width - k, dtype=q.dtype, device=q.device)])


def build(w64, alph, mus, Q, tau, dev):
    """Prune at `tau`, fold Q into W1, slice. Returns (sliced, offsets, cost/sample)."""
    depth, width = w64.shape[0], w64.shape[-1]
    alive, dead = [], []
    for li in range(depth):
        if li == depth - 1:
            alive.append(torch.arange(width, device=dev))
            dead.append(None)
            continue
        keep = alph[li] > tau
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            idx = torch.argmax(alph[li]).reshape(1)
            keep = torch.zeros(width, dtype=torch.bool, device=dev)
            keep[idx] = True
        didx = torch.nonzero(~keep, as_tuple=False).squeeze(1)
        alive.append(idx)
        dead.append(didx if didx.numel() else None)

    sliced, offsets, cost, prev = [], [], 0.0, None
    for li in range(depth):
        wl = Q.T @ w64[li] if li == 0 else w64[li]
        wf = wl.to(torch.float32)
        cur = alive[li]
        b = wf if prev is None else wf.index_select(0, prev)
        sliced.append(b.index_select(1, cur))
        if li > 0 and dead[li - 1] is not None:
            wd = wf.index_select(0, dead[li - 1]).index_select(1, cur)
            offsets.append(mus[li - 1].index_select(0, dead[li - 1]).to(torch.float32) @ wd)
        else:
            offsets.append(None)
        kk = width if prev is None else int(prev.numel())
        cost += (2 * kk - 1) * int(cur.numel()) + 1.5 * int(cur.numel())
        prev = cur
    return sliced, offsets, cost, alive, dead


def forward(h, sliced, offsets):
    for li in range(len(sliced)):
        z = h @ sliced[li]
        if offsets[li] is not None:
            z = z + offsets[li]
        h = torch.clamp(z, min=0.0)
    return h


def predict_multishift(sliced, offsets, base_pts, shifts, dev, batch=8):
    """Mean prediction per digital shift. Returns (R, width)."""
    outs = []
    for i in range(0, len(shifts), batch):
        chunk = shifts[i:i + batch]
        hs = torch.cat([draw_points(base_pts, s, dev) for s in chunk], dim=0)
        out = forward(hs, sliced, offsets)
        n = base_pts.shape[0]
        for j in range(len(chunk)):
            outs.append(out[j * n:(j + 1) * n].mean(dim=0, dtype=torch.float64))
        del hs, out
    return torch.stack(outs)


def truth_final(w, n_samples, dev, gt_seed, chunk=262144):
    """Final-layer truth and its own MC noise.

    Returns (mean, gt_noise_mse) where gt_noise_mse = mean_i Var_i / N — the MSE
    this label contributes purely by being a finite sample. Reported with every
    measurement so an "improvement" smaller than it cannot be mistaken for one.
    (It is common to every arm of a paired sweep and cancels in the difference,
    but it sets the floor on any ABSOLUTE claim.)
    """
    g = torch.Generator(device=dev).manual_seed(gt_seed)
    acc = torch.zeros(w.shape[-1], dtype=torch.float64, device=dev)
    acc_sq = torch.zeros(w.shape[-1], dtype=torch.float64, device=dev)
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        h = torch.randn(b, w.shape[-1], generator=g, device=dev, dtype=torch.float32)
        for li in range(w.shape[0]):
            h = torch.clamp(h @ w[li], min=0.0)
        acc += h.sum(dim=0, dtype=torch.float64)
        acc_sq += (h.double() ** 2).sum(dim=0)
        done += b
    m = acc / n_samples
    var = torch.clamp(acc_sq / n_samples - m * m, min=0.0)
    return m, float(var.mean()) / n_samples


def shifts_for(est_seed, width, draws, dev):
    """`draws` independent digital shifts, the first being the SHIPPED one."""
    g = np.random.default_rng(int(est_seed))
    out = []
    for _ in range(draws):
        s = g.integers(0, 1 << 30, size=width, dtype=np.uint32).astype(np.int64)
        out.append(torch.from_numpy(s).to(dev))
    return out


# ------------------------------------------------------------------ sweeps ---
def _paired_report(name, values, frozen, per_mlp_adj, gt_noise, per_mlp_mse=None, extra=None):
    """per_mlp_adj: dict value -> (M,) adjusted score per MLP (shift-averaged)."""
    print("\n" + "=" * 100)
    print(f"CONSTANT: {name}     frozen = {frozen}")
    print("=" * 100)
    base = per_mlp_adj[frozen]
    M = len(base)
    print(f"{'value':>12} {'mean adjusted':>15} {'vs frozen':>11} {'paired diff':>14} "
          f"{'SE':>11} {'sigma':>7}  verdict")
    print("-" * 100)
    best, best_v = None, None
    for v in values:
        a = per_mlp_adj[v]
        d = a - base
        se = float(d.std(ddof=1) / math.sqrt(M)) if M > 1 else float("nan")
        md = float(d.mean())
        sig = md / se if se > 0 else 0.0
        ratio = float(base.mean() / a.mean())
        tag = "FROZEN" if v == frozen else (
            "better (real)" if md < 0 and abs(sig) > 2 else
            "better (noise)" if md < 0 else
            "worse (real)" if abs(sig) > 2 else "worse (noise)")
        print(f"{str(v):>12} {a.mean():>15.5e} {ratio:>10.4f}x {md:>14.3e} {se:>11.2e} "
              f"{sig:>7.1f}  {tag}")
        if best is None or a.mean() < best:
            best, best_v = a.mean(), v
    print("-" * 100)
    drift = best_v != frozen
    best_sigma = 0.0
    real = False
    if drift:
        d = per_mlp_adj[best_v] - base
        se = float(d.std(ddof=1) / math.sqrt(M))
        best_sigma = abs(float(d.mean()) / se) if se > 0 else 0.0
        real = best_sigma > 2
        print(f"fresh-mint optimum: {best_v}  ({float(base.mean() / best):.4f}x better, "
              f"{best_sigma:.1f} sigma) -> {'REFIT REQUIRED' if real else 'within noise, KEEP FROZEN'}")
    else:
        print(f"fresh-mint optimum: {best_v} == frozen value -> KEEP")
    mse_level = float(per_mlp_mse[frozen].mean()) if per_mlp_mse else None
    if mse_level:
        print(f"GT noise {gt_noise:.2e} MSE = {gt_noise / mse_level * 100:.2f}% of the frozen "
              f"arm's raw MSE {mse_level:.3e}. Common to every arm, so it cancels in the")
        print("paired column; it is the floor on any ABSOLUTE claim, not on these comparisons.")
    else:
        print(f"GT noise {gt_noise:.2e} MSE (common to every arm; cancels in the paired column)")
    if extra:
        print(extra)
    return {"constant": name, "frozen": frozen, "fresh_optimum": best_v,
            "means": {str(v): float(per_mlp_adj[v].mean()) for v in values},
            "optimum_differs": bool(drift),
            "paired_sigma_vs_frozen": float(best_sigma),
            "statistically_resolved_drift": bool(drift and real),
            "statistical_action": ("refit-required" if drift and real else "keep-frozen")}


def sweep(name, block0, values, mlps, draws, n_points, gt, dev, vary):
    """vary: 'tau' | 'rank' | 'n'. Everything else is held at the frozen value."""
    base_cache = {}
    per_mlp = {v: [] for v in values}
    per_mse = {v: [] for v in values}
    noise_acc = []
    t0 = time.perf_counter()
    for i in range(mlps):
        s = block0 + i
        w_np, est_seed, sample_ss = canonical_mint(WIDTH, DEPTH, s)
        w32 = torch.from_numpy(np.stack(w_np)).to(dev)
        w64 = w32.double()
        gt_seed = int(sample_ss.generate_state(1)[0]) & 0x7FFFFFFF
        t, gt_noise = truth_final(w32, gt, dev, gt_seed)
        noise_acc.append(gt_noise)

        alph, mus, phis, _, _ = closure(w64, dev)
        sh = shifts_for(est_seed, WIDTH, draws, dev)

        for v in values:
            tau = v if vary == "tau" else FROZEN["tau"]
            rank = v if vary == "rank" else FROZEN["rank"]
            npt = v if vary == "n" else n_points
            key = rank
            if key not in base_cache or base_cache[key][0] is not w64:
                base_cache[key] = (w64, basis_trunc(w64, phis, rank))
            Q = base_cache[key][1]
            sliced, offsets, cps, _alive, _dead = build(w64, alph, mus, Q, tau, dev)
            if npt not in _BASE:
                _BASE[npt] = sobol_base(npt, WIDTH, dev)
            bp = _BASE[npt]
            preds = predict_multishift(sliced, offsets, bp, sh, dev)
            mse = ((preds - t.unsqueeze(0)) ** 2).mean(dim=1).mean()  # avg over shifts
            cb = (npt * cps + FIXED_FLOPS) / BUDGET
            per_mlp[v].append(float(mse) * max(0.1, cb))
            per_mse[v].append(float(mse))
        base_cache.clear()
        if (i + 1) % 25 == 0:
            print(f"  {name}: {i + 1}/{mlps}  {time.perf_counter() - t0:.0f}s", flush=True)
    return ({v: np.asarray(per_mlp[v]) for v in values},
            {v: np.asarray(per_mse[v]) for v in values},
            float(np.mean(noise_acc)))


_BASE = {}


def truth_all_layers(w, n_samples, dev, gt_seed, chunk=262144):
    """(depth, width) post-ReLU means and the PER-NEURON GT-noise variance.

    Per-neuron, not layer-averaged, because the patch audit compares the closure
    against truth on *pruned* neurons — which are nearly dead, so their activation
    variance is orders below the layer mean. Quoting the layer-average noise there
    would overstate the floor by ~100x and hide a real result under a fake one.
    """
    g = torch.Generator(device=dev).manual_seed(gt_seed)
    D, W = w.shape[0], w.shape[-1]
    acc = torch.zeros((D, W), dtype=torch.float64, device=dev)
    acc_sq = torch.zeros((D, W), dtype=torch.float64, device=dev)
    done = 0
    while done < n_samples:
        b = min(chunk, n_samples - done)
        h = torch.randn(b, W, generator=g, device=dev, dtype=torch.float32)
        for li in range(D):
            h = torch.clamp(h @ w[li], min=0.0)
            hd = h.double()
            acc[li] += hd.sum(dim=0)
            acc_sq[li] += (hd * hd).sum(dim=0)
        done += b
    m = acc / n_samples
    var = torch.clamp(acc_sq / n_samples - m * m, min=0.0)
    return m, var / n_samples  # (D,W) means, (D,W) per-neuron noise variance


def audit_patch_bias(mlps, gt, dev, taus=(-3.5, -3.0, -2.5, -2.0, -1.5)):
    """How wrong is the closed-form mean we substitute for a PRUNED neuron?

    Pruning drops a neuron from the sampled forward pass and folds the closure's
    rectified-Gaussian mean into the next layer as a constant offset. The bias
    that introduces is exactly the closure's error ON THE PRUNED SET — which is
    measurable directly, needs no sampling arm, and is the quantity the frozen
    3.8e-9 bound refers to.
    """
    print("\n" + "=" * 100)
    print("CONSTANT: pruned-neuron analytic patch bias     frozen bound = 3.8e-9")
    print("=" * 100)
    per_tau = {t: [] for t in taus}
    worst = {t: 0.0 for t in taus}
    frac = {t: [] for t in taus}
    pruned_noise = {t: [] for t in taus}
    for i in range(mlps):
        s = BLOCKS["patch"] + i
        w_np, _e, sample_ss = canonical_mint(WIDTH, DEPTH, s)
        w32 = torch.from_numpy(np.stack(w_np)).to(dev)
        gt_seed = int(sample_ss.generate_state(1)[0]) & 0x7FFFFFFF
        t_all, nvar = truth_all_layers(w32, gt, dev, gt_seed)
        alph, mus, _phis, _a, _b = closure(w32.double(), dev)
        for tau in taus:
            errs, nz_acc, n_pruned, n_tot = [], [], 0, 0
            for li in range(DEPTH - 1):
                dead = alph[li] <= tau
                n_tot += WIDTH
                k = int(dead.sum())
                if k == 0:
                    continue
                n_pruned += k
                d = (mus[li][dead] - t_all[li][dead])
                errs.append(d)
                nz_acc.append(nvar[li][dead])
                worst[tau] = max(worst[tau], float(d.abs().max()))
            if errs:
                e = torch.cat(errs)
                per_tau[tau].append(float((e ** 2).mean()))
                pruned_noise[tau].append(float(torch.cat(nz_acc).mean()))
            else:
                per_tau[tau].append(0.0)
                pruned_noise[tau].append(0.0)
            frac[tau].append(n_pruned / max(n_tot, 1))
    print(f"{'tau':>8} {'pruned share':>14} {'mean sq err on pruned':>23} "
          f"{'GT noise on pruned':>20} {'signal/noise':>13} {'max |err|':>11}")
    print("-" * 100)
    for tau in taus:
        v = float(np.mean(per_tau[tau]))
        nz = float(np.mean(pruned_noise[tau]))
        f = float(np.mean(frac[tau]))
        snr = v / nz if nz > 0 else float("inf")
        mark = "  <== SHIPPED" if tau == -2.5 else ""
        print(f"{tau:>8} {f * 100:>13.2f}% {v:>23.3e} {nz:>20.3e} {snr:>13.1f}x "
              f"{worst[tau]:>11.3e}{mark}")
    print("-" * 100)
    v25 = float(np.mean(per_tau[-2.5]))
    nz25 = float(np.mean(pruned_noise[-2.5]))
    print(f"At the shipped tau = -2.5 the substituted closed-form mean is wrong by "
          f"{v25:.3e} in mean")
    print(f"square on the pruned set, against the recorded bound 3.8e-9 -> "
          f"{'HOLDS' if v25 <= 3.8e-9 else 'EXCEEDED'}.")
    print(f"GT noise ON THE PRUNED SET is {nz25:.2e} "
          f"({'measurement is real signal' if v25 > 5 * nz25 else 'NOISE-DOMINATED — rerun with more GT samples'}).")
    print("Note this is the error on the SUBSTITUTED VALUES, not its contribution to the")
    print("scored layer: the pruned neuron's mean enters downstream only through a")
    print("constant offset, which the next matmul contracts.")
    return {"constant": "patch bias", "frozen": 3.8e-9, "measured_at_tau_2.5": v25,
            "gt_noise_on_pruned": nz25, "holds": bool(v25 <= 3.8e-9)}


def audit_calibration(mlps, gt, dev):
    """Refit v0's per-layer calibration on fresh mints and compare to the frozen table.

    Only the analytic fallback path uses this (budget below the sampling floor),
    so it cannot move the graded score — but a fallback nobody checks is exactly
    where a stale constant survives, and the fallback is what runs if Phase 2
    ships a budget we did not expect.
    """
    print("\n" + "=" * 100)
    print("CONSTANT: v0 per-layer calibration table (_CAL_LAYER_SCALE, fallback path only)")
    print("=" * 100)
    num = np.zeros(DEPTH)
    den = np.zeros(DEPTH)
    noise = []
    for i in range(mlps):
        s = BLOCKS["cal"] + i
        w_np, _e, sample_ss = canonical_mint(WIDTH, DEPTH, s)
        w32 = torch.from_numpy(np.stack(w_np)).to(dev)
        gt_seed = int(sample_ss.generate_state(1)[0]) & 0x7FFFFFFF
        t_all, nvar = truth_all_layers(w32, gt, dev, gt_seed)
        noise.append(float(nvar[-1].mean()))
        _alph, mus, _phis, _a, _b = closure(w32.double(), dev)
        p = mus.cpu().numpy()
        t = t_all.cpu().numpy()
        num += (p * t).sum(axis=1)
        den += (p * p).sum(axis=1)
    fresh = num / np.maximum(den, 1e-300)

    # The v0 table lives in experiments/calibration.json, keyed by depth.
    frozen = None
    try:
        import json as _j
        cal = _j.load(open(os.path.join(_ROOT, "experiments", "calibration.json")))
        frozen = np.asarray(cal["depths"][str(DEPTH)]["c_layers"], dtype=np.float64)
    except Exception as ex:
        print(f"  (could not load experiments/calibration.json: {ex})")

    # Is it actually IN the shipped file? v0 had _CAL_LAYER_SCALE; the sampling
    # estimators dropped it, so the analytic fallback may be running uncalibrated.
    shipped_has = None
    try:
        src = open(os.path.join(_ROOT, "estimator.py"), encoding="utf-8").read()
        shipped_has = "_CAL_LAYER_SCALE" in src
    except Exception:
        pass
    if shipped_has is False:
        print("  !! `_CAL_LAYER_SCALE` is NOT PRESENT in the shipped estimator.py.")
        print("     `_analytic_only` returns the RAW closure means, uncalibrated. That")
        print("     path only runs when the budget is below the sampling floor (~1e10),")
        print("     so it cannot affect the Phase-1 score — but it is what would run if")
        print("     Phase 2 shipped a much smaller budget, and v0 measured calibration")
        print("     as a ~3x win on exactly this estimator. Flagged, not changed:")
        print("     estimator.py is a designation candidate.")

    print(f"{'layer':>6} {'frozen':>12} {'fresh-mint refit':>18} {'ratio':>10}")
    print("-" * 52)
    show = [0, 1, 3, 7, 15, 23, 30, 31]
    worst = 0.0
    for li in show:
        if li >= DEPTH:
            continue
        fz = float(frozen[li]) if frozen is not None and li < len(frozen) else float("nan")
        r = fresh[li] / fz if fz == fz and fz else float("nan")
        if r == r:
            worst = max(worst, abs(r - 1.0))
        print(f"{li + 1:>6} {fz:>12.5f} {fresh[li]:>18.5f} {r:>10.5f}")
    print("-" * 52)
    print(f"max |fresh/frozen - 1| over the shown layers: {worst:.4f}")
    print(f"GT noise {float(np.mean(noise)):.2e} MSE")
    print("Verdict: " + ("KEEP — within 1%" if worst < 0.01 else
                         "drifted >1%; refit if the fallback ever matters"))
    return {"constant": "v0 calibration", "max_rel_drift": float(worst),
            "fresh": fresh.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "tau", "rank", "cb", "patch", "cal", "cheap"])
    ap.add_argument("--mlps", type=int, default=200)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--gt", type=int, default=8_388_608)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"device {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}   "
          f"{a.mlps} fresh MLPs x {a.draws} digital shifts   GT {a.gt:,} samples/MLP\n", flush=True)

    _BASE[FROZEN["n_points"]] = sobol_base(FROZEN["n_points"], WIDTH, dev)
    out = []

    if a.which in ("all", "tau"):
        vals = [-3.5, -3.0, -2.5, -2.0, -1.5]
        pm, pe, nz = sweep("tau", BLOCKS["tau"], vals, a.mlps, a.draws,
                           FROZEN["n_points"], a.gt, dev, "tau")
        out.append(_paired_report("prune tau", vals, -2.5, pm, nz, pe))

    if a.which in ("all", "rank"):
        vals = [8, 16, 32, 64, 256]
        pm, pe, nz = sweep("rank", BLOCKS["rank"], vals, a.mlps, a.draws,
                           FROZEN["n_points"], a.gt, dev, "rank")
        out.append(_paired_report("rotation rank", vals, 32, pm, nz, pe))

    if a.which in ("all", "cb"):
        vals = [4096, 8192, 16384, 32768, 65536]
        for v in vals:
            _BASE[v] = sobol_base(v, WIDTH, dev)
        pm, pe, nz = sweep("cb", BLOCKS["cb"], vals, a.mlps, a.draws,
                           FROZEN["n_points"], a.gt, dev, "n")
        note = ("Theory (RESEARCH_NOTES 3d): above the 0.1 multiplier floor\n"
                "adjusted = V*c/B is INDEPENDENT of N, so this column should be flat\n"
                "once C/B >= 0.1 and rising only where the floor binds. A slope above\n"
                "the floor would mean the cost model and the measured cost disagree.")
        out.append(_paired_report("sample count N (C/B operating point)", vals,
                                  16384, pm, nz, pe, extra=note))

    if a.which in ("all", "cheap", "patch"):
        out.append(audit_patch_bias(max(20, a.mlps // 4), a.gt, dev))

    if a.which in ("all", "cheap", "cal"):
        out.append(audit_calibration(max(20, a.mlps // 4), a.gt, dev))

    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
