"""
VARIANCE-OR-BIAS: does the shipped estimator have a bias floor at all?

The score is MSE * max(0.1, C/B), so the only way to buy accuracy at fixed compute
is to remove variance -- unless the residual has a bias floor, in which case no
amount of sampling helps and the closure/calibration core is the target instead.

Model, FROZEN in experiments/LOG.md 2026-08-11 before any D16 seed was drawn:

    MSE_measured(N) = V/N + B^2 + sigma_gt^2

The third term is the killer and is why the previous inference was unsafe. Ground
truth is itself a Monte Carlo average, so the labels carry noise eta with variance
sigma_gt^2 per neuron; since eta is independent of the estimator,

    E[(pred - truth - eta)^2] = E[(pred - truth)^2] + sigma_gt^2

i.e. GT noise is *perfectly confounded* with a bias floor. Varying N alone cannot
separate them. So sigma_gt^2 is COMPUTED here (not fitted) in the same GPU pass that
builds the truth -- the method vault.py:114-120 uses -- and subtracted:

    B^2 = (MSE(N2)*N2 - MSE(N1)*N1) / (N2 - N1)  -  sigma_gt^2

Paired common random numbers throughout: the same MLPs and the same digital shifts
at both sample counts, so the two arms differ only in N.

Run (GPU venv):
    .venv-gpu\\Scripts\\python.exe local_sim/bias_floor_lab.py --mlps 200 --shifts 8
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mint_provenance import canonical_mint
import seed_blocks

WIDTH, DEPTH = 256, 32
BLOCK = "D16"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "experiments", "bias_floor_D16.json")


def _truth_gpu(w_np, sample_ss, n_gt, chunk, dev, torch):
    """Final-layer truth AND the per-neuron sample variance behind sigma_gt^2.

    Mirrors vault.py:102-120 exactly so the GT-noise figure is comparable to the
    vault's audited 1.575e-9.
    """
    w = torch.from_numpy(np.stack(w_np)).to(dev)
    gt_seed = int(sample_ss.generate_state(1)[0]) & 0x7FFFFFFF
    g = torch.Generator(device=dev).manual_seed(gt_seed)

    acc = torch.zeros((DEPTH, WIDTH), dtype=torch.float64, device=dev)
    acc_sq = torch.zeros(WIDTH, dtype=torch.float64, device=dev)
    done = 0
    while done < n_gt:
        b = min(chunk, n_gt - done)
        h = torch.randn(b, WIDTH, generator=g, device=dev, dtype=torch.float32)
        for li in range(DEPTH):
            h = torch.clamp(h @ w[li], min=0.0)
            acc[li] += h.sum(dim=0, dtype=torch.float64)
        acc_sq += (h.double() ** 2).sum(dim=0)
        done += b

    m = (acc / n_gt).cpu().numpy()
    v = np.maximum((acc_sq / n_gt).cpu().numpy() - m[-1] ** 2, 0.0)
    return m[-1], float(v.mean() / n_gt)          # final-layer truth, sigma_gt^2


def _run_estimator(mlp, n_target):
    """Shipped estimator at a chosen N, via the budget ladder (never by editing it)."""
    import estimator as est_mod
    est = est_mod.Estimator()
    est.setup_called = False
    # _log2_samples maps budget -> N; find the budget that lands on n_target.
    lo, hi = 1, 1 << 62
    want = int(np.log2(n_target))
    for _ in range(200):
        mid = (lo + hi) // 2
        got = est._log2_samples(mlp.width, len(mlp.weights), mid)
        if got < want:
            lo = mid + 1
        else:
            hi = mid
    budget = hi
    got = est._log2_samples(mlp.width, len(mlp.weights), budget)
    if got != want:
        raise SystemExit(f"could not reach N={n_target} (got 2^{got}); ladder mismatch")
    return np.asarray(est.predict(mlp, budget), dtype=np.float64)[-1], budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=200)
    ap.add_argument("--shifts", type=int, default=8)
    ap.add_argument("--n-gt", type=int, default=1 << 24)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--n1", type=int, default=16384)
    ap.add_argument("--n2", type=int, default=65536)
    a = ap.parse_args()

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = seed_blocks.seeds(BLOCK)[:a.mlps]
    print(f"block {BLOCK}: {seeds[0]:,} .. {seeds[-1]:,}   {a.mlps} MLPs x {a.shifts} shifts")
    print(f"N1={a.n1:,}  N2={a.n2:,}   GT {a.n_gt:,} samples/MLP   device {dev}\n", flush=True)

    import harness

    mse1, mse2, sig2 = [], [], []
    t0 = time.perf_counter()
    for i, s in enumerate(seeds):
        w_np, _est_seed, sample_ss = canonical_mint(WIDTH, DEPTH, s)
        truth_final, sg2 = _truth_gpu(w_np, sample_ss, a.n_gt, a.chunk, dev, torch)

        # Same MLP, same shift sequence, only N differs -> paired CRN.
        e1, e2 = [], []
        for k in range(a.shifts):
            mlp = harness.MLP(weights=[np.asarray(x) for x in w_np],
                              width=WIDTH, depth=DEPTH, seed=int(s) * 1000 + k)
            p1, _ = _run_estimator(mlp, a.n1)
            p2, _ = _run_estimator(mlp, a.n2)
            e1.append(float(np.mean((p1 - truth_final) ** 2)))
            e2.append(float(np.mean((p2 - truth_final) ** 2)))

        mse1.append(float(np.mean(e1)))
        mse2.append(float(np.mean(e2)))
        sig2.append(sg2)
        if (i + 1) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1:>4}/{a.mlps}  {el:7.1f}s  eta {el/(i+1)*(a.mlps-i-1):7.1f}s  "
                  f"MSE1 {np.mean(mse1):.4e}  MSE2 {np.mean(mse2):.4e}  "
                  f"gt-noise {np.mean(sig2):.3e}", flush=True)

    mse1 = np.array(mse1); mse2 = np.array(mse2); sig2 = np.array(sig2)

    def stats(m1, m2, sg):
        b2_raw = (m2.mean() * a.n2 - m1.mean() * a.n1) / (a.n2 - a.n1)
        b2 = b2_raw - sg.mean()
        v = m1.mean() * a.n1 - (b2_raw) * a.n1
        alpha = float(np.log(m1.mean() / m2.mean()) / np.log(a.n2 / a.n1))
        return b2_raw, b2, v, alpha

    b2_raw, b2, V, alpha = stats(mse1, mse2, sig2)

    rng = np.random.default_rng(12345)
    boot = []
    for _ in range(20000):
        idx = rng.integers(0, len(mse1), len(mse1))
        boot.append(stats(mse1[idx], mse2[idx], sig2[idx])[1])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    alpha_b = []
    for _ in range(20000):
        idx = rng.integers(0, len(mse1), len(mse1))
        alpha_b.append(stats(mse1[idx], mse2[idx], sig2[idx])[3])

    res = {
        "block": BLOCK, "n_mlps": int(a.mlps), "n_shifts": int(a.shifts),
        "n_gt": int(a.n_gt), "N1": int(a.n1), "N2": int(a.n2),
        "mse_N1_mean": float(mse1.mean()), "mse_N2_mean": float(mse2.mean()),
        "mse_N1_median": float(np.median(mse1)), "mse_N2_median": float(np.median(mse2)),
        "mseN_N1": float(mse1.mean() * a.n1), "mseN_N2": float(mse2.mean() * a.n2),
        "sigma_gt2_mean": float(sig2.mean()),
        "B2_before_gt_subtraction": float(b2_raw),
        "B2": float(b2), "B2_ci95": [lo, hi],
        "B2_share_of_mse_N2": float(b2 / mse2.mean()),
        "alpha": alpha, "alpha_ci95": [float(np.percentile(alpha_b, 2.5)),
                                       float(np.percentile(alpha_b, 97.5))],
        "runtime_s": time.perf_counter() - t0,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({**res, "per_mlp": {"mse_N1": mse1.tolist(), "mse_N2": mse2.tolist(),
                                      "sigma_gt2": sig2.tolist(), "seeds": seeds}}, f, indent=2)

    print("\n" + "=" * 72)
    print(f"MSE(N1={a.n1:,})           {res['mse_N1_mean']:.6e}")
    print(f"MSE(N2={a.n2:,})           {res['mse_N2_mean']:.6e}")
    print(f"MSE x N   at N1 / N2       {res['mseN_N1']:.5f} / {res['mseN_N2']:.5f}")
    print(f"alpha (MSE ~ N^-alpha)     {alpha:.4f}  95% CI {res['alpha_ci95']}")
    print(f"sigma_gt^2 (label noise)   {res['sigma_gt2_mean']:.4e}")
    print(f"B^2 BEFORE subtracting it  {b2_raw:.4e}   <-- what the old inference saw")
    print(f"B^2 AFTER  subtracting it  {b2:.4e}   95% CI [{lo:.3e}, {hi:.3e}]")
    print(f"B^2 as share of MSE(N2)    {res['B2_share_of_mse_N2']*100:.3f}%")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
