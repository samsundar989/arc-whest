"""
Sprint 3 (GPU; runs in .venv-gpu, NEVER imported by an estimator).

Structure-aware FLOP allocation in the last k layers
---------------------------------------------------
Track T proved that for a confidently-on neuron the rectified-Gaussian formula
reduces algebraically to the sample mean (Phi=1, phi=0, rho=1.000 at alpha>=3).
That is a statement about the SCORED layer. Sprint 3 asks whether the same
reduction can be spent as compute in the last k layers, where it says something
stronger: if a neuron is confidently on, ReLU is the identity there, so

  * at the scored layer it needs no per-sample values at all — its mean is
    w^T (incoming sample mean), plus an exact rectification correction;
  * at a feed-forward layer it is LINEAR, so its path composes with the next
    layer's weights into a precomputed matrix and costs nothing per sample.

Only KINK neurons need a per-sample ReLU. Under shape-based billing that pays
only if we physically materialise the narrow matmul, which is what the cost model
below prices.

Two ways to treat an on-neuron feeding forward, and the difference is the whole
correctness question:

  linearise   h = z            keeps the fluctuation; exact wherever z > 0
  mean-replace h = E[z]        drops the fluctuation entirely

The second is the literal "use the mean" reading, and it is the same move that
killed Track P: removing variance biases every downstream ReLU (Jensen). Both are
measured here rather than argued about.

Accuracy is simulated by running the full-width forward with the modified
activation rule, which is numerically identical (up to f32 rounding) to the
composed-matrix implementation; cost is priced analytically from the shapes the
composed implementation would materialise.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/sprint3_probe.py --data <mini100.npz> \\
        --ks 1,2,3,4,6,8,12 --alphas 1.0,1.5,2.0,2.5,3.0,4.0
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from track_t_mint import alignment_basis, closure, draw_points, sobol_base

BUDGET = 2.72e11
PRUNE_TAU = -2.5
KAPPA = 1.363  # measured mean(v_reference)/mean(v_closure) at the final layer


def ndtr_t(x):
    return 0.5 * torch.erfc(-x * (1.0 / math.sqrt(2.0)))


def npdf_t(x):
    return (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * x * x)


def _min_cost(nk, av, src0):
    """Cheapest per-sample cost for the modified block, in FLOPs per sample.

    At each modified layer we may either
      * keep the COMPOSED representation — compute only the kink outputs from
        [state | all kink activations so far], cost (2*src-1)*nk, src += nk; or
      * RE-MATERIALISE all alive neurons, cost (2*src-1)*alive, resetting
        src = alive.
    Composing is cheap per layer but grows the source width, so on layers with
    few on-neurons it eventually costs more than the dense form. The choice is a
    cost decision only — accuracy is identical either way — so we take the exact
    optimum by DP over the last re-materialised layer. The final layer is always
    composed (only its kink outputs are ever needed per sample).
    """
    k = len(nk)
    INF = float("inf")

    def seg(p_src, lo, hi, dense_at_hi):
        """positions lo..hi-1 composed from source p_src; optionally dense at hi."""
        c, s = 0.0, p_src
        for m in range(lo, hi):
            c += (2 * s - 1) * nk[m] + 0.5 * nk[m]
            s += nk[m]
        if dense_at_hi:
            c += (2 * s - 1) * av[hi] + 0.5 * av[hi]
            s = av[hi]
        return c, s

    # D[j] = (cost, src) with position j re-materialised densely
    D = {-1: (0.0, src0)}
    for j in range(k - 1):  # the last position is never dense
        best = INF
        for p, (cp, sp) in list(D.items()):
            if p >= j:
                continue
            c, s = seg(sp, p + 1, j, True)
            if cp + c < best:
                best, bsrc = cp + c, s
        if best < INF:
            D[j] = (best, bsrc)
    out = INF
    for p, (cp, sp) in D.items():
        c, _ = seg(sp, p + 1, k, False)
        out = min(out, cp + c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--mlps", type=int, default=100)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--ks", default="1,2,3,4,6,8,12")
    ap.add_argument("--alphas", default="1.0,1.5,2.0,2.5,3.0,4.0")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    d = np.load(args.data, allow_pickle=True)
    W_all, T_all, seeds = d["weights"], d["truth"], d["seeds"]
    M = min(args.mlps, len(W_all))
    depth, width = W_all.shape[1], W_all.shape[2]
    ks = [int(v) for v in args.ks.split(",")]
    alphas_hi = [float(v) for v in args.alphas.split(",")]
    kmax = max(ks)
    base_pts = sobol_base(args.n, width, dev)
    N = args.n
    print(f"device={dev} {M} MLPs N={N} ks={ks} alpha_hi={alphas_hi}\n", flush=True)

    # results[(k, a_hi, variant, final_mode)] -> list of per-MLP mse
    res = {}
    cost_s3 = {}
    base_mse, base_cost = [], []
    frac_on = np.zeros((M, depth, len(alphas_hi)))
    t0 = time.perf_counter()

    for i in range(M):
        w32 = torch.from_numpy(W_all[i]).to(dev)
        w64 = w32.to(torch.float64)
        alph, mus, phis, mu_pre_last, var_pre_last = closure(w64, dev)
        Q = alignment_basis(w64, phis)
        t = torch.from_numpy(T_all[i][-1]).to(dev)

        # ---- v2.4 pruning + rotation + slicing (identical to the shipped file)
        alive, dead = [], []
        for li in range(depth):
            if li == depth - 1:
                alive.append(torch.arange(width, device=dev))
                dead.append(None)
                continue
            keep = alph[li] > PRUNE_TAU
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
            wl = w64[li]
            if li == 0:
                wl = Q.T @ wl
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
            cost += (2 * kk - 1) * int(cur.numel()) + 0.5 * int(cur.numel())
            prev = cur
        base_cost.append(cost)

        # alive-relative on/kink masks per layer
        on_mask, n_alive = [], []
        for li in range(depth):
            a_l = alph[li].index_select(0, alive[li])
            on_mask.append(a_l)  # store alphas; threshold later
            n_alive.append(int(alive[li].numel()))

        # ---- baseline forward, storing the states we may branch from ---------
        gen = np.random.default_rng(int(seeds[i]))
        shift = torch.from_numpy(gen.integers(0, 1 << 30, size=width, dtype=np.uint32).astype(np.int64)).to(dev)
        h = draw_points(base_pts, shift, dev)
        states = {}
        for li in range(depth):
            z = h @ sliced[li]
            if offsets[li] is not None:
                z = z + offsets[li]
            if li == depth - 1:
                base_mse.append(float(((torch.clamp(z, min=0.0).mean(dim=0).to(torch.float64) - t) ** 2).mean()))
            else:
                h = torch.clamp(z, min=0.0)
                if li >= depth - 1 - kmax:
                    states[li] = h  # state AFTER layer li, i.e. input to layer li+1

        sig_c = torch.sqrt(torch.clamp(var_pre_last, min=1e-30))
        sig_k = sig_c * math.sqrt(KAPPA)
        a_last_full = alph[depth - 1]
        mu_closure_last = mus[depth - 1]

        # ---- the modified continuations -------------------------------------
        for k in ks:
            start = depth - k                     # first modified layer index
            h0 = states[start - 1]
            for a_hi in alphas_hi:
                for variant in ("linearise", "mean-replace"):
                    hh = h0
                    for lj in range(start, depth):
                        z = hh @ sliced[lj]
                        if offsets[lj] is not None:
                            z = z + offsets[lj]
                        if lj == depth - 1:
                            break
                        on = (on_mask[lj] > a_hi)
                        if variant == "linearise":
                            hh = torch.where(on, z, torch.clamp(z, min=0.0))
                        else:
                            hh = torch.where(on, z.mean(dim=0, keepdim=True).expand_as(z),
                                             torch.clamp(z, min=0.0))
                    # final layer: kink sampled, on analytic, dead analytic
                    on_f = a_last_full > a_hi
                    dead_f = a_last_full <= PRUNE_TAU
                    kink_f = (~on_f) & (~dead_f)
                    zm = z.mean(dim=0).to(torch.float64)
                    samp = torch.clamp(z, min=0.0).mean(dim=0).to(torch.float64)
                    for final_mode, sg in (("plain", None), ("corr", sig_c), ("corr_kappa", sig_k)):
                        if sg is None:
                            on_val = zm
                        else:
                            aa = zm / sg
                            on_val = zm + sg * (npdf_t(aa) - aa * ndtr_t(-aa))
                        out = torch.where(kink_f, samp, torch.where(on_f, on_val, mu_closure_last))
                        key = (k, a_hi, variant, final_mode)
                        res.setdefault(key, []).append(float(((out - t) ** 2).mean()))

                    # ---- cost of the composed implementation ----------------
                    key = (k, a_hi, variant)
                    if key not in cost_s3:
                        cost_s3[key] = []
                    c = 0.0
                    for li in range(start):
                        kk = width if li == 0 else n_alive[li - 1]
                        c += (2 * kk - 1) * n_alive[li] + 0.5 * n_alive[li]
                    nk = [int((on_mask[lj] <= a_hi).sum()) for lj in range(start, depth - 1)]
                    nk.append(int(kink_f.sum()))
                    av = [n_alive[lj] for lj in range(start, depth)]
                    c += _min_cost(nk, av, n_alive[start - 1])
                    cost_s3[key].append(c)

        for li in range(depth):
            for ai, a_hi in enumerate(alphas_hi):
                frac_on[i, li, ai] = float((on_mask[li] > a_hi).float().mean())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{M}  {time.perf_counter() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ report
    PPF_SOBOL = 171 * width + 30 * 3 * width + 1.0 * width * depth
    bm, bc = float(np.mean(base_mse)), float(np.mean(base_cost)) + PPF_SOBOL
    b_adj = bm * N * bc / BUDGET
    print(f"\n{M} MLPs, N={N}")
    print(f"v2.4 baseline: MSE {bm:.4e}  cost/sample {bc:,.0f}  C/B {N * bc / BUDGET:.4f}  "
          f"adjusted {b_adj:.4e}")
    print("\nON-FRACTION BY LAYER (share of the ALIVE neurons with closure alpha above the threshold)")
    print(f"{'layer':>6} " + " ".join(f"{'a>' + str(a):>8}" for a in alphas_hi))
    print("-" * (7 + 9 * len(alphas_hi)))
    for li in (0, 3, 7, 11, 15, 19, 23, 27, 29, 30, 31):
        print(f"{li + 1:>6} " + " ".join(f"{frac_on[:, li, ai].mean():>8.3f}" for ai in range(len(alphas_hi))))

    print("\n" + "=" * 100)
    print("LINEARISE on-neurons (final-layer on-value = mean + exact rectification correction, kappa-scaled sigma)")
    print("=" * 100)
    print(f"{'k':>4} " + " ".join(f"{'a>' + str(a):>17}" for a in alphas_hi))
    print(f"{'':>4} " + " ".join(f"{'MSE   cost  V*c':>17}" for _ in alphas_hi))
    print("-" * 100)
    best = (0.0, None)
    for k in ks:
        cells = []
        for a_hi in alphas_hi:
            m = float(np.mean(res[(k, a_hi, "linearise", "corr_kappa")]))
            c = float(np.mean(cost_s3[(k, a_hi, "linearise")])) + PPF_SOBOL
            adj = m * N * c / BUDGET
            g = b_adj / adj
            if g > best[0]:
                best = (g, (k, a_hi, "linearise", "corr_kappa", m, c, adj))
            cells.append(f"{m:.2e} {bc / c:>4.2f} {g:>5.3f}x")
        print(f"{k:>4} " + " ".join(f"{c:>17}" for c in cells))

    print("\n" + "=" * 100)
    print("MEAN-REPLACE on-neurons  (the literal 'use the mean' reading — correctness check)")
    print("=" * 100)
    print(f"{'k':>4} " + " ".join(f"{'a>' + str(a):>17}" for a in alphas_hi))
    print("-" * 100)
    for k in ks:
        cells = []
        for a_hi in alphas_hi:
            m = float(np.mean(res[(k, a_hi, "mean-replace", "corr_kappa")]))
            c = float(np.mean(cost_s3[(k, a_hi, "mean-replace")])) + PPF_SOBOL
            adj = m * N * c / BUDGET
            cells.append(f"{m:.2e} {bc / c:>4.2f} {b_adj / adj:>5.3f}x")
        print(f"{k:>4} " + " ".join(f"{c:>17}" for c in cells))

    print("\n" + "=" * 78)
    print("FINAL-LAYER on-neuron treatment (k=1, isolates the scored-layer reduction)")
    print("=" * 78)
    print(f"{'alpha_hi':>9} {'E[z] only':>13} {'+corr(sigma_c)':>16} {'+corr(kappa)':>14}")
    print("-" * 78)
    for a_hi in alphas_hi:
        row = [float(np.mean(res[(1, a_hi, "linearise", md)])) for md in ("plain", "corr", "corr_kappa")]
        print(f"{a_hi:>9} {row[0]:>13.4e} {row[1]:>16.4e} {row[2]:>14.4e}")

    g, cfg = best
    print(f"\nBEST: k={cfg[0]} alpha_hi={cfg[1]} -> MSE {cfg[4]:.4e}, cost {cfg[5]:,.0f} "
          f"({bc / cfg[5]:.3f}x cheaper), adjusted {cfg[6]:.4e}  =>  {g:.4f}x over v2.4")
    print(f"GATE (>=1.20x and adjusted <=2.27e-7): "
          f"{'PASS' if (g >= 1.20 and cfg[6] <= 2.27e-7) else 'FAIL'}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"base_mse": bm, "base_cost": bc, "base_adj": b_adj, "N": N, "mlps": M,
                       "rows": [{"k": k, "alpha_hi": a, "variant": v, "final_mode": f,
                                 "mse": float(np.mean(res[(k, a, v, f)])),
                                 "cost": float(np.mean(cost_s3[(k, a, v)])) + PPF_SOBOL}
                                for (k, a, v, f) in res]}, fh, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
