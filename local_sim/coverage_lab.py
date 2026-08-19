"""
COVERAGE LAB — does the promotion interval actually cover 95%?

Stage 0. The Skeptic's discriminating follow-up on the P0.1 slice (finding D5):
`competition_score.paired_compare` proposes a paired-bootstrap interval on the
arithmetic score difference as the primary promotion statistic, and **no coverage
study existed for it**. A gate whose interval undercovers promotes candidates that
are not better; one that overcovers kills candidates that are.

The competition's per-MLP score is right-skewed (measured on the 50 graded rows of
327743: worst/mean 3.59x, top-1 tail share 7.2%). Under skew the four standard
intervals do NOT agree, and which one is right is an empirical question:

    t            symmetric, assumes normality of the mean. Fastest.
    percentile   bootstrap quantiles. Biased OPPOSITE the skew.
    basic        reverse percentile. Biased WITH the skew.
    BCa          bias-corrected + accelerated. The textbook remedy.

TWO PHASES, because the expensive half is ground truth.

  --phase truth : GPU. Mint M fresh networks from D18 (claimed through
                  `seed_ledger`), compute final-layer truth and the per-neuron
                  sample variance behind sigma_gt^2. Mirrors `bias_floor_lab` and
                  `vault.py` so the noise figure is comparable to both.
  --phase arms  : CPU, main venv. Run the SHIPPED estimator under flopscope for
                  each arm, giving real per-MLP MSE, real counted FLOPs and real
                  residual wall time -> a real per-MLP competition score.
  --phase cover : pure numpy. The coverage simulation, plus a parametric skew
                  sweep that maps where each interval breaks.

WHY GT NOISE IS NOT FATAL HERE. Ground truth is itself a Monte-Carlo average, so
each arm's MSE is inflated by the same sigma_gt^2 for the same network. In a
PAIRED difference that term cancels in expectation, which is why this study can
afford 2^22 samples where `bias_floor_lab` needed 2^24. The residual correlated
noise is reported, not assumed away.

ARMS. All share the network and differ only as stated, with common random numbers
inside an MLP and independent streams across MLPs:

    base    shipped estimator, N = 16,384, shift stream 0   (the control)
    null    shipped estimator, N = 16,384, shift stream 1   (TRUE effect zero)
    small   shipped estimator, N = 32,768                   (one rung up the ladder)
    large   shipped estimator, N = 65,536                   (an obvious candidate)

`base` vs `null` is the important one: the truth is known to be zero by
construction, so a 95% interval that excludes zero more than 5% of the time is
manufacturing promotions.

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/coverage_lab.py --phase truth --mlps 192
    uv run python local_sim/coverage_lab.py --phase arms
    uv run python local_sim/coverage_lab.py --phase cover
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# `local_sim/` contains numpy SHIMS named `flopscope` and `whestbench` so the
# torch-only GPU venv can import them. In the main venv those shims must NOT
# shadow the real packages — the arms phase measures counted FLOPs and would
# silently measure the shim instead. Same convention as every other lab here:
# drop this directory from the front of the path, then append it.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

WIDTH, DEPTH = 256, 32
BLOCK = "D18"
OUTDIR = os.path.join(_ROOT, "experiments", "2026-08-19_coverage")
TRUTH_NPZ = os.path.join(OUTDIR, "truth_D18.npz")
ARMS_JSON = os.path.join(OUTDIR, "arms_D18.json")
COVER_JSON = os.path.join(OUTDIR, "coverage.json")

ARMS = {"base": 16384, "null": 16384, "small": 32768, "large": 65536}
#: Shift-stream offset per arm. `base` and `null` share N and differ ONLY here,
#: which is what makes their true mean difference zero by construction.
ARM_STREAM = {"base": 0, "null": 1, "small": 0, "large": 0}


# --------------------------------------------------------------------------- #
# phase 1 — ground truth (GPU venv)
# --------------------------------------------------------------------------- #


def _truth_gpu(w_np, sample_ss, n_gt, chunk, dev, torch):
    """Final-layer truth and sigma_gt^2. Mirrors bias_floor_lab / vault.py."""
    w = torch.from_numpy(np.stack(w_np)).to(dev)
    gt_seed = int(sample_ss.generate_state(1)[0]) & 0x7FFFFFFF
    g = torch.Generator(device=dev).manual_seed(gt_seed)
    acc = torch.zeros(WIDTH, dtype=torch.float64, device=dev)
    acc_sq = torch.zeros(WIDTH, dtype=torch.float64, device=dev)
    done = 0
    while done < n_gt:
        b = min(chunk, n_gt - done)
        h = torch.randn(b, WIDTH, generator=g, device=dev, dtype=torch.float32)
        for li in range(DEPTH):
            h = torch.clamp(h @ w[li], min=0.0)
        acc += h.sum(dim=0, dtype=torch.float64)
        acc_sq += (h.double() ** 2).sum(dim=0)
        done += b
    m = (acc / n_gt).cpu().numpy()
    v = np.maximum((acc_sq / n_gt).cpu().numpy() - m ** 2, 0.0)
    return m, float(v.mean() / n_gt)


def phase_truth(a):
    import torch

    from mint_provenance import canonical_mint
    import seed_ledger

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTDIR, exist_ok=True)

    global TRUTH_NPZ
    if a.smoke:
        # Smoke mode NEVER claims a block: it reuses the tail of the legacy L3
        # range, mints nothing that any result depends on, and writes to a
        # throwaway file. Its only job is to prove the plumbing before the real
        # run spends real seeds.
        TRUTH_NPZ = os.path.join(OUTDIR, "truth_SMOKE.npz")
        seeds = [900_000 + 5_500 + i for i in range(a.mlps)]
        print(f"SMOKE: legacy L3 tail, no block claimed, -> {TRUTH_NPZ}")
    else:
        if os.path.exists(TRUTH_NPZ):
            raise SystemExit(
                f"{TRUTH_NPZ} already exists. Delete it deliberately to re-mint — "
                "re-running would claim a SECOND range from D18."
            )
        seeds = seed_ledger.claim(
            BLOCK, a.mlps, "local_sim/coverage_lab.py", ts=a.date,
            artifact="experiments/2026-08-19_coverage/truth_D18.npz",
            note=f"interval-coverage study: {a.mlps} fresh networks, GT 2^"
                 f"{int(np.log2(a.n_gt))}, arms {sorted(ARMS)}",
        )
    print(f"claimed {BLOCK}[{seeds[0]:,}..{seeds[-1]:,}]  {len(seeds)} networks")
    print(f"GT {a.n_gt:,} samples/MLP   device {dev}\n", flush=True)

    truth = np.zeros((len(seeds), WIDTH), dtype=np.float64)
    sig2 = np.zeros(len(seeds), dtype=np.float64)
    t0 = time.perf_counter()
    for i, s in enumerate(seeds):
        w_np, _est, sample_ss = canonical_mint(WIDTH, DEPTH, s)
        truth[i], sig2[i] = _truth_gpu(w_np, sample_ss, a.n_gt, a.chunk, dev, torch)
        if (i + 1) % 16 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1:>4}/{len(seeds)}  {el:7.1f}s  "
                  f"eta {el/(i+1)*(len(seeds)-i-1):7.1f}s  "
                  f"gt-noise {sig2[:i+1].mean():.3e}", flush=True)
    np.savez_compressed(TRUTH_NPZ, seeds=np.array(seeds, dtype=np.int64),
                        truth=truth, sigma_gt2=sig2, n_gt=a.n_gt)
    print(f"\nsigma_gt^2 = {sig2.mean():.4e}   saved {TRUTH_NPZ}")


# --------------------------------------------------------------------------- #
# phase 2 — the arms (CPU, main venv, real flopscope accounting)
# --------------------------------------------------------------------------- #


def _budget_for_n(est, width, depth, n_target):
    """The budget whose ladder rung is exactly n_target. Never edit the ladder.

    The ladder emits powers of two only, so a non-power-of-two target would
    silently round down to the rung below and produce an arm identical to another.
    """
    if n_target & (n_target - 1):
        raise SystemExit(f"N={n_target:,} is not a power of two; the ladder cannot hit it")
    want = int(np.log2(n_target))
    lo, hi = 1, 1 << 62
    for _ in range(200):
        mid = (lo + hi) // 2
        if est._log2_samples(width, depth, mid) < want:
            lo = mid + 1
        else:
            hi = mid
    if est._log2_samples(width, depth, hi) != want:
        raise SystemExit(f"ladder cannot reach N={n_target:,}")
    return hi


def phase_arms(a):
    global TRUTH_NPZ, ARMS_JSON
    if a.smoke:
        TRUTH_NPZ = os.path.join(OUTDIR, "truth_SMOKE.npz")
        ARMS_JSON = os.path.join(OUTDIR, "arms_SMOKE.json")
    import flopscope as flops

    flops.configure(symmetry_warnings=False)
    import estimator as est_mod
    import harness
    from mint_provenance import canonical_mint

    d = np.load(TRUTH_NPZ)
    seeds, truth, sig2 = d["seeds"], d["truth"], d["sigma_gt2"]
    m = len(seeds) if a.mlps <= 0 else min(a.mlps, len(seeds))
    print(f"{m} networks, GT 2^{int(np.log2(int(d['n_gt'])))}, "
          f"sigma_gt^2 = {sig2[:m].mean():.4e}\n", flush=True)

    est = est_mod.Estimator()
    budgets = {k: _budget_for_n(est, WIDTH, DEPTH, n) for k, n in ARMS.items()}
    for k, n in ARMS.items():
        print(f"  arm {k:<6} N = {n:>6,}  budget = {budgets[k]:,}  "
              f"shift stream {ARM_STREAM[k]}")
    print(flush=True)

    out = {k: {"mse": [], "flops": [], "resid": []} for k in ARMS}
    t0 = time.perf_counter()
    for i in range(m):
        s = int(seeds[i])
        w_np, _est_seed, _ss = canonical_mint(WIDTH, DEPTH, s)
        for k in ARMS:
            # Same network for every arm; the estimator seed carries the arm's
            # shift stream, so base/null are CRN-paired and differ only in stream.
            mlp = harness.MLP(weights=[np.asarray(x) for x in w_np],
                              width=WIDTH, depth=DEPTH,
                              seed=s * 1000 + ARM_STREAM[k])
            e = est_mod.Estimator()
            ctx = flops.BudgetContext(flop_budget=budgets[k], quiet=True)
            with ctx:
                pred = e.predict(mlp, budgets[k])
            p = np.asarray(pred, dtype=np.float64)[-1]
            out[k]["mse"].append(float(np.mean((p - truth[i]) ** 2)))
            out[k]["flops"].append(float(ctx.flops_used))
            out[k]["resid"].append(float(ctx.residual_wall_time_s or 0.0))
        if (i + 1) % 16 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1:>4}/{m}  {el:7.1f}s  eta {el/(i+1)*(m-i-1):7.1f}s  "
                  f"base MSE {np.mean(out['base']['mse']):.4e}", flush=True)

    res = {
        "block": BLOCK,
        "seeds": [int(x) for x in seeds[:m]],
        "n_gt": int(d["n_gt"]),
        "sigma_gt2_mean": float(sig2[:m].mean()),
        "arms": {k: {"n_points": ARMS[k], "stream": ARM_STREAM[k],
                     "budget": int(budgets[k])} for k in ARMS},
        "per_mlp": out,
        "runtime_s": time.perf_counter() - t0,
    }
    with open(ARMS_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nsaved {ARMS_JSON}")
    for k in ARMS:
        print(f"  {k:<6} MSE {np.mean(out[k]['mse']):.5e}   "
              f"counted {np.mean(out[k]['flops']):.4e}   "
              f"resid {np.median(out[k]['resid'])*1e3:.2f} ms")


# --------------------------------------------------------------------------- #
# phase 3 — the coverage simulation
# --------------------------------------------------------------------------- #


def _intervals(dmat, rng, resamples, methods):
    """CIs for the mean of every row of `dmat` (R x n), by each method."""
    from competition_score import _norm_cdf, _norm_ppf, _t975

    R, n = dmat.shape
    theta = dmat.mean(axis=1)
    out = {}
    if "t" in methods:
        se = dmat.std(axis=1, ddof=1) / np.sqrt(n)
        q = _t975(n - 1)
        out["t"] = (theta - q * se, theta + q * se)
    need_boot = {"percentile", "basic", "bca"} & set(methods)
    if not need_boot:
        return out
    idx = rng.integers(0, n, size=(resamples, n))
    boot = dmat[:, idx].mean(axis=2)                     # (R, resamples)
    lo_p = np.percentile(boot, 2.5, axis=1)
    hi_p = np.percentile(boot, 97.5, axis=1)
    if "percentile" in methods:
        out["percentile"] = (lo_p, hi_p)
    if "basic" in methods:
        out["basic"] = (2 * theta - hi_p, 2 * theta - lo_p)
    if "bca" in methods:
        prop = np.clip((boot < theta[:, None]).mean(axis=1),
                       1.0 / (2 * resamples), 1.0 - 1.0 / (2 * resamples))
        z0 = np.array([_norm_ppf(float(p)) for p in prop])
        tot = dmat.sum(axis=1, keepdims=True)
        jack = (tot - dmat) / (n - 1)
        jbar = jack.mean(axis=1, keepdims=True)
        num = ((jbar - jack) ** 3).sum(axis=1)
        den = 6.0 * (((jbar - jack) ** 2).sum(axis=1) ** 1.5)
        acc = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        bl, bh = [], []
        for z, store in ((_norm_ppf(0.025), bl), (_norm_ppf(0.975), bh)):
            denom = 1.0 - acc * (z0 + z)
            adj = np.where(denom != 0, z0 + (z0 + z) / np.where(denom != 0, denom, 1.0),
                           z0 + z)
            pct = np.array([100.0 * _norm_cdf(float(x)) for x in adj])
            store.append(np.array([np.percentile(boot[r], pct[r]) for r in range(R)]))
        lo, hi = bl[0], bh[0]
        out["bca"] = (np.minimum(lo, hi), np.maximum(lo, hi))
    return out


def _coverage(d_pop, truth, n, reps, resamples, seed, methods, chunk=200):
    """Actual coverage of each interval method for mean(d) at sample size n.

    Chunked over the outer replicates: the bootstrap tensor is
    (reps x resamples x n) doubles, which at 4000 x 2000 x 50 is 3.2 GB. Chunking
    also gives each chunk its OWN bootstrap indices, so the R replicates are not
    all sharing one index set.
    """
    rng = np.random.default_rng(seed)
    M = d_pop.size
    acc = {m: {"cov": 0, "lo": 0, "hi": 0, "w": 0.0, "excl": 0} for m in methods}
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        dmat = d_pop[rng.integers(0, M, size=(r, n))]
        cis = _intervals(dmat, rng, resamples, methods)
        for name, (lo, hi) in cis.items():
            acc[name]["cov"] += int(np.sum((lo <= truth) & (truth <= hi)))
            acc[name]["lo"] += int(np.sum(truth < lo))    # interval entirely ABOVE truth
            acc[name]["hi"] += int(np.sum(truth > hi))    # interval entirely BELOW truth
            acc[name]["w"] += float(np.sum(hi - lo))
            acc[name]["excl"] += int(np.sum((lo > 0) | (hi < 0)))
        done += r
    return {
        m: {
            "coverage": acc[m]["cov"] / reps,
            "miss_low": acc[m]["lo"] / reps,
            "miss_high": acc[m]["hi"] / reps,
            "mean_width": acc[m]["w"] / reps,
            "excludes_zero": acc[m]["excl"] / reps,
            # Monte-Carlo error on the coverage estimate itself, so nobody reads a
            # 94.6% as different from 95.0% when it is not.
            "mc_se": float(np.sqrt(max(acc[m]["cov"] / reps * (1 - acc[m]["cov"] / reps), 0)
                                   / reps)),
        }
        for m in methods
    }


def phase_cover(a):
    global ARMS_JSON, COVER_JSON
    if a.smoke:
        ARMS_JSON = os.path.join(OUTDIR, "arms_SMOKE.json")
        COVER_JSON = os.path.join(OUTDIR, "coverage_SMOKE.json")
    from competition_score import (PHASE1_FLOP_BUDGET, MlpRow,  # noqa: F401
                                   score_suite)

    with open(ARMS_JSON, encoding="utf-8") as fh:
        A = json.load(fh)
    per = A["per_mlp"]
    B = PHASE1_FLOP_BUDGET
    methods = ["t", "percentile", "basic", "bca"]

    def scores(arm):
        rows = [
            MlpRow(mse=per[arm]["mse"][i], flops_used=per[arm]["flops"][i],
                   residual_wall_time_s=0.0, budget=B)
            for i in range(len(per[arm]["mse"]))
        ]
        return score_suite(rows).scores

    print("=" * 100)
    print("COVERAGE OF THE PROMOTION INTERVAL — real fresh networks, block D18")
    print("=" * 100)
    print(f"\n{len(per['base']['mse'])} networks, GT 2^{int(np.log2(A['n_gt']))}, "
          f"sigma_gt^2 = {A['sigma_gt2_mean']:.4e}")
    print("Residual wall time is EXCLUDED from C here: this machine's residual is E1 "
          "(docs/CURRENT_STATE.md)\nand injecting it would make the study about this "
          "laptop. C = counted FLOPs only.\n")

    out = {"pairs": {}, "parametric": {}, "meta": {
        "n_networks": len(per["base"]["mse"]), "n": a.n, "reps": a.reps,
        "resamples": a.resamples, "seed": a.seed, "budget": B,
        "n_gt": A["n_gt"], "sigma_gt2_mean": A["sigma_gt2_mean"],
    }}

    for cand in ("null", "small", "large"):
        sc, sk = scores("base"), scores(cand)
        d_pop = sc - sk
        truth = float(d_pop.mean())
        sd = float(d_pop.std(ddof=1))
        skew = float(np.mean(((d_pop - d_pop.mean()) / sd) ** 3)) if sd > 0 else 0.0
        print(f"\n--- base vs {cand} " + "-" * 74)
        print(f"  population: M = {d_pop.size}   mean difference {truth:+.5e}   "
              f"sd {sd:.3e}   skew {skew:+.3f}")
        print(f"  base score {sc.mean():.5e}   {cand} score {sk.mean():.5e}   "
              f"relative {truth/sc.mean()*100:+.3f}%")
        r = _coverage(d_pop, truth, a.n, a.reps, a.resamples, a.seed, methods)
        out["pairs"][cand] = {"truth": truth, "sd": sd, "skew": skew,
                              "mean_base": float(sc.mean()),
                              "mean_cand": float(sk.mean()), "methods": r}
        print(f"  {'method':<12}{'coverage':>16}{'miss lo':>9}{'miss hi':>9}"
              f"{'width':>12}{'excl. 0':>9}")
        for mth in methods:
            v = r[mth]
            lo95 = v["coverage"] - 1.96 * v["mc_se"]
            hi95 = v["coverage"] + 1.96 * v["mc_se"]
            flag = "  <-- UNDERCOVERS" if hi95 < 0.95 else (
                   "  <-- overcovers" if lo95 > 0.95 else "")
            print(f"  {mth:<12}{v['coverage']*100:>9.1f}% +-{v['mc_se']*100:>3.1f}"
                  f"{v['miss_low']*100:>8.1f}%"
                  f"{v['miss_high']*100:>8.1f}%{v['mean_width']:>12.3e}"
                  f"{v['excludes_zero']*100:>8.1f}%{flag}")

    # FALSE-PROMOTION RATE. Prediction B4 tried to read this off the `base` vs
    # `null` arm's "excludes zero" column, which was mis-specified: that arm's
    # POPULATION mean difference is not exactly zero (it is -4.10e-08, about 1.7
    # standard errors from zero at M = 192), so its exclusion rate mixes power
    # with false positives. The clean measurement recentres each population to a
    # true mean of exactly zero and asks how often a 95% interval still excludes
    # it. That is the rate at which a gate promotes a candidate that is not
    # better, and it is the number a promotion rule actually has to defend.
    print("\n\n--- FALSE PROMOTION: population recentred to a TRUE mean of exactly 0 " + "-" * 12)
    print(f"  {'population':<16}{'skew':>7}" + "".join(f"{m:>13}" for m in methods)
          + "     (nominal 5.0%)")
    for cand in ("null", "small", "large"):
        sc, sk = scores("base"), scores(cand)
        d0 = sc - sk
        d0 = d0 - d0.mean()
        skew = float(np.mean((d0 / d0.std(ddof=1)) ** 3))
        r = _coverage(d0, 0.0, a.n, a.reps, a.resamples, a.seed, methods)
        out["false_promotion"] = out.get("false_promotion", {})
        out["false_promotion"][cand] = {"skew": skew, "methods": r}
        print(f"  base vs {cand:<8}{skew:>7.3f}"
              + "".join(f"{r[m]['excludes_zero'] * 100:>12.1f}%" for m in methods))

    # Parametric sweep: where does each method break as skew rises? The paired
    # difference is modelled as a centred lognormal, whose skew is set by sigma.
    print("\n\n--- parametric sweep: coverage vs skew of the paired difference " + "-" * 22)
    print(f"  {'sigma':<8}{'skew':>8}" + "".join(f"{m:>13}" for m in methods))
    rng0 = np.random.default_rng(a.seed + 99)
    for sigma in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        pop = np.exp(rng0.normal(0.0, sigma, 200_000))
        pop = pop - pop.mean()
        sk_ = float(np.mean((pop / pop.std()) ** 3))
        r = _coverage(pop, 0.0, a.n, a.reps, a.resamples, a.seed, methods)
        out["parametric"][f"{sigma}"] = {"skew": sk_, "methods": r}
        print(f"  {sigma:<8.2f}{sk_:>8.2f}"
              + "".join(f"{r[m]['coverage']*100:>12.1f}%" for m in methods))

    os.makedirs(OUTDIR, exist_ok=True)
    with open(COVER_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nsaved {COVER_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["truth", "arms", "cover"])
    ap.add_argument("--mlps", type=int, default=192)
    ap.add_argument("--n-gt", dest="n_gt", type=int, default=1 << 22)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--date", default="2026-08-19")
    ap.add_argument("--n", type=int, default=50, help="subsample size to study")
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing check: claims NO block, writes throwaway files")
    a = ap.parse_args()
    {"truth": phase_truth, "arms": phase_arms, "cover": phase_cover}[a.phase](a)


if __name__ == "__main__":
    main()
