"""
OA CARRIER, END TO END — what is the strength-5 orthogonal array actually worth?

Stage 1 / item 5 part 3. Contract and predictions T1-T5: `experiments/LOG.md`,
2026-08-19 (night).

WHAT THIS ANSWERS
-----------------
Part 2 measured the dual-BCH OA's *moment* exactness: 17x better than the shipped
Sobol carrier at degree 4, machine-precision at degrees 1, 2, 3 and 5. That is a
property of the POINT SET. What it is worth is a property of the NETWORK
RESPONSE, and only a paired end-to-end run answers it.

THE THREE ARMS, and why the middle one is the important one
------------------------------------------------------------
    A  sobol   N =  16,384   the shipped operating point — the incumbent
    B  sobol   N = 131,072   MATCHED POINT COUNT — isolates "more points"
    C  bch OA  N = 131,072   MATCHED POINT COUNT — isolates "the design"

The arms are matched at 131,072 because the array only works WHOLE: halving it
returns the linear code, which has a constant column and a centroid of norm 1.0,
and a RANDOM half is worse than Sobol at k=2 and k=4. See the note on ARMS.

`C vs B` is the design's own contribution. `C vs A` is the deployable comparison.
Without B, any gain could simply be 4x the samples — and `docs/CURRENT_STATE.md`'s
standing instruction from the carrier audit is to **read every ladder for what its
baseline arm contains**. B is that baseline.

Everything except the point set is the SHIPPED pipeline, called through the
shipped file's own methods (`_closure`, `_alignment_basis`, `_alive_sets`,
`_slice`) rather than reimplemented — so a difference between arms cannot be a
difference in the surrounding engineering. The audit's governing finding is
SUBSTITUTION: a device must be tested against the stack it would join, never bare.

THE ROTATION IS FOLDED INTO W1, NOT APPLIED TO THE POINTS
----------------------------------------------------------
The OA must be randomly rotated to be unbiased and to stop being axis-aligned.
Rotating 131,072 x 256 points costs ~1.7e10 FLOPs, a third of the shipped spend.
`h @ (R^T S) == (h R^T) @ S`, so rotating the first-layer weight instead costs one
256x256 matmul, ~3.3e7 FLOPs. Same estimator, ~260x cheaper.

UNBIASEDNESS. For Haar `R`, `R u_i` is uniform on the sphere, so
`E_R[(1/n) sum f(R u_i)] = E_u[f(u)]`; positive homogeneity then gives
`E[f(x)] = E||x|| * E_u[f(u)]`. The estimator is unbiased, which this repository
has learned the hard way to check first (the 2.35x moment-matching regression).

Run:
    .venv-gpu\\Scripts\\python.exe local_sim/probe_oa_carrier.py --phase truth --mlps 192
    uv run python local_sim/probe_oa_carrier.py --phase arms
    uv run python local_sim/probe_oa_carrier.py --phase score
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# APPEND: local_sim holds numpy shims named flopscope/whestbench for the GPU venv,
# and the arms phase MEASURES counted FLOPs with the real package.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

WIDTH, DEPTH = 256, 32
BLOCK = "D18"
BUDGET = 272_000_000_000
#: The MEASUREMENT context gets a deliberately huge budget. Arm B and arm C run at
#: N = 65,536, ~4x the shipped spend, so a context capped at the real B would raise
#: BudgetExhaustedError mid-probe and we would measure nothing. C/B is computed
#: against the REAL budget afterwards — which is the number that decides
#: affordability, and it is allowed to exceed 1.0 here so that we can SEE it.
MEASURE_BUDGET = 10_000_000_000_000
OUTDIR = os.path.join(_ROOT, "experiments", "2026-08-19_carriers")
TRUTH_NPZ = os.path.join(OUTDIR, "oa_truth_D18.npz")
ARMS_JSON = os.path.join(OUTDIR, "oa_arms.json")
SCORE_JSON = os.path.join(OUTDIR, "oa_score.json")

# The array only works WHOLE. Taking one row per antipodal pair returns the linear
# code, which has a constant column and a centroid of norm 1.0 (measured 41x worse
# than Sobol end to end). Puncturing + parity is constant too. And a RANDOM half is
# no design at all: measured k=2 4.06e-03 / k=4 9.31e-03 against the full array's
# 8.54e-15 / 7.55e-04 — worse than Sobol at both. So the arms are matched at the
# array's own size, 2^17, and the cost of that is the finding.
ARMS = (("A sobol N=16384", "sobol", 1 << 14),
        ("B sobol N=131072", "sobol", 1 << 17),
        ("C bchOA N=131072", "bch_oa", 1 << 17))


# --------------------------------------------------------------------------- #
# phase 1 — ground truth (GPU venv)
# --------------------------------------------------------------------------- #


def phase_truth(a):
    import torch

    from mint_provenance import canonical_mint
    import seed_ledger

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTDIR, exist_ok=True)
    if os.path.exists(TRUTH_NPZ):
        raise SystemExit(f"{TRUTH_NPZ} exists — delete deliberately; a re-run would "
                         "claim a SECOND range from D18.")
    seeds = seed_ledger.claim(
        BLOCK, a.mlps, "local_sim/probe_oa_carrier.py", ts=a.date,
        artifact="experiments/2026-08-19_carriers/oa_truth_D18.npz",
        note=f"OA carrier end-to-end: {a.mlps} fresh networks, GT 2^"
             f"{int(np.log2(a.n_gt))}, arms sobol@16384 / sobol@65536 / bchOA@65536",
    )
    print(f"claimed {BLOCK}[{seeds[0]:,}..{seeds[-1]:,}]   GT {a.n_gt:,}/MLP   {dev}\n",
          flush=True)

    truth = np.zeros((len(seeds), WIDTH), dtype=np.float64)
    sig2 = np.zeros(len(seeds), dtype=np.float64)
    t0 = time.perf_counter()
    for i, s in enumerate(seeds):
        w_np, _e, ss = canonical_mint(WIDTH, DEPTH, s)
        w = torch.from_numpy(np.stack(w_np)).to(dev)
        g = torch.Generator(device=dev).manual_seed(int(ss.generate_state(1)[0]) & 0x7FFFFFFF)
        acc = torch.zeros(WIDTH, dtype=torch.float64, device=dev)
        accsq = torch.zeros(WIDTH, dtype=torch.float64, device=dev)
        done = 0
        while done < a.n_gt:
            b = min(a.chunk, a.n_gt - done)
            h = torch.randn(b, WIDTH, generator=g, device=dev, dtype=torch.float32)
            for li in range(DEPTH):
                h = torch.clamp(h @ w[li], min=0.0)
            acc += h.sum(dim=0, dtype=torch.float64)
            accsq += (h.double() ** 2).sum(dim=0)
            done += b
        m = (acc / a.n_gt).cpu().numpy()
        v = np.maximum((accsq / a.n_gt).cpu().numpy() - m ** 2, 0.0)
        truth[i], sig2[i] = m, float(v.mean() / a.n_gt)
        if (i + 1) % 16 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1:>4}/{len(seeds)}  {el:7.1f}s  "
                  f"eta {el/(i+1)*(len(seeds)-i-1):7.1f}s  gt-noise {sig2[:i+1].mean():.3e}",
                  flush=True)
    np.savez_compressed(TRUTH_NPZ, seeds=np.array(seeds, dtype=np.int64),
                        truth=truth, sigma_gt2=sig2, n_gt=a.n_gt)
    print(f"\nsigma_gt^2 = {sig2.mean():.4e}   saved {TRUTH_NPZ}")


# --------------------------------------------------------------------------- #
# phase 2 — the arms (CPU, real flopscope)
# --------------------------------------------------------------------------- #


def phase_arms(a):
    import flopscope as flops
    import flopscope.numpy as fnp

    flops.configure(symmetry_warnings=False)
    import estimator as E
    from carriers import bch_oa_signs, exact_radius, sobol, to_sphere
    from mint_provenance import canonical_mint

    d = np.load(TRUTH_NPZ)
    seeds, truth, sig2 = d["seeds"], d["truth"], d["sigma_gt2"]
    m = len(seeds) if a.mlps <= 0 else min(a.mlps, len(seeds))
    print(f"{m} networks, GT 2^{int(np.log2(int(d['n_gt'])))}, "
          f"sigma_gt^2 = {sig2[:m].mean():.4e}\n", flush=True)

    # The OA table is a FIXED constant, so a deployed estimator would build it in
    # setup(), which is off-budget. Built once here for the same reason.
    t0 = time.perf_counter()
    oa_full = bch_oa_signs()                       # (131072, 256) int8, +-1
    print(f"OA table {oa_full.shape} in {time.perf_counter()-t0:.1f}s — used WHOLE. "
          f"centroid norm {np.linalg.norm(oa_full.mean(0)):.3e}\n", flush=True)
    oa_unit = exact_radius(to_sphere(oa_full.astype(np.float64)), WIDTH).astype(np.float32)

    out = {lbl: {"mse": [], "flops": []} for lbl, _, _ in ARMS}
    t0 = time.perf_counter()
    for i in range(m):
        s = int(seeds[i])
        w_np, est_seed, _ = canonical_mint(WIDTH, DEPTH, s)
        for lbl, kind, n in ARMS:
            ctx = flops.BudgetContext(flop_budget=MEASURE_BUDGET, quiet=True)
            with ctx:
                est = E.Estimator()
                w64 = [fnp.asarray(x).astype(fnp.float64) for x in w_np]
                alphas, mus, phis = est._closure(w64, WIDTH, DEPTH)
                basis = est._alignment_basis(w64, phis, WIDTH, DEPTH)
                trim = False          # dense chain in every arm: the carrier is
                                      # the only thing that differs between them
                alive, dead = est._alive_sets(alphas, WIDTH, DEPTH, trim)
                sliced, offsets = est._slice(w64, basis, mus, alive, dead, DEPTH)

                if kind == "sobol":
                    h = fnp.asarray(sobol(n, WIDTH, est_seed).astype(np.float32))
                else:
                    if n != oa_unit.shape[0]:
                        raise SystemExit(f"OA arm needs n = {oa_unit.shape[0]:,}")
                    # Fold the random rotation into the FIRST-LAYER WEIGHT, not the
                    # points: h @ (R^T S) == (h R^T) @ S, and S is 256 x alive.
                    rng = np.random.default_rng(est_seed)
                    q, r = np.linalg.qr(rng.standard_normal((WIDTH, WIDTH)))
                    q = (q * np.sign(np.diag(r))).astype(np.float32)
                    h = fnp.asarray(oa_unit)
                    sliced = [fnp.asarray(q.T) @ sliced[0]] + list(sliced[1:])

                for li in range(DEPTH):
                    z = h @ sliced[li]
                    if offsets[li] is not None:
                        z = z + offsets[li]
                    h = fnp.maximum(z, fnp.float32(0.0))
                final = np.asarray(h.mean(axis=0, dtype=fnp.float64), dtype=np.float64)

            out[lbl]["mse"].append(float(np.mean((final - truth[i]) ** 2)))
            out[lbl]["flops"].append(float(ctx.flops_used))
        if (i + 1) % 8 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1:>4}/{m}  {el:7.1f}s  eta {el/(i+1)*(m-i-1):7.1f}s   "
                  + "   ".join(f"{lbl.split()[0]} {np.mean(out[lbl]['mse']):.4e}"
                               for lbl, _, _ in ARMS), flush=True)

    res = {"block": BLOCK, "seeds": [int(x) for x in seeds[:m]],
           "n_gt": int(d["n_gt"]), "sigma_gt2": [float(x) for x in sig2[:m]],
           "sigma_gt2_mean": float(sig2[:m].mean()),
           "arms": {lbl: {"kind": k, "n": n} for lbl, k, n in ARMS},
           "per_mlp": out, "runtime_s": time.perf_counter() - t0}
    with open(ARMS_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nsaved {ARMS_JSON}")
    for lbl, _, n in ARMS:
        print(f"  {lbl:<18} MSE {np.mean(out[lbl]['mse']):.5e}   "
              f"counted {np.mean(out[lbl]['flops']):.4e}   "
              f"C/B {np.mean(out[lbl]['flops'])/BUDGET:.4f}")


# --------------------------------------------------------------------------- #
# phase 3 — the paired comparison
# --------------------------------------------------------------------------- #


def phase_score(a):
    from competition_score import (PHASE1_FLOP_BUDGET, MlpRow, format_paired,
                                   paired_compare, score_suite)

    with open(ARMS_JSON, encoding="utf-8") as fh:
        A = json.load(fh)
    per = A["per_mlp"]
    sg2 = np.asarray(A["sigma_gt2"])
    labels = [lbl for lbl, _, _ in ARMS]

    # Arms B and C run at N = 131,072 and land at C/B ~ 1.35 — OVER the budget.
    # `score_suite` refuses those by default, which is the guard working: on the
    # real grader C > B is a FAILURE (multiplier 1.0 AND zeroed predictions). Both
    # readings are reported below: "ignore" is the counterfactual value of the
    # design if it were affordable, "fail" is what the grader would actually do.
    MODE = "ignore"

    def rows(lbl, gt_sub):
        mse = np.asarray(per[lbl]["mse"])
        if gt_sub:
            mse = np.maximum(mse - sg2, 0.0)
        fl = np.asarray(per[lbl]["flops"])
        return [MlpRow(mse=float(mm), flops_used=float(ff), residual_wall_time_s=0.0,
                       budget=PHASE1_FLOP_BUDGET) for mm, ff in zip(mse, fl)]

    print("=" * 100)
    print("OA CARRIER, END TO END — paired on fresh D18 networks")
    print("=" * 100)
    print(f"\n{len(sg2)} networks, GT 2^{int(np.log2(A['n_gt']))}, "
          f"sigma_gt^2 = {A['sigma_gt2_mean']:.4e}")
    print("C = counted FLOPs only; this machine's residual is E1 "
          "(docs/CURRENT_STATE.md).\n")

    print(f"  {'arm':<18}{'raw MSE':>13}{'GT-subtracted':>15}{'counted/MLP':>14}"
          f"{'C/B':>8}{'adjusted':>13}")
    print("  " + "-" * 82)
    summ = {}
    for lbl in labels:
        s_raw = score_suite(rows(lbl, False), on_over_budget=MODE)
        s_sub = score_suite(rows(lbl, True), on_over_budget=MODE)
        summ[lbl] = {"mse_raw": s_raw.mean_mse, "mse_sub": s_sub.mean_mse,
                     "counted": float(np.mean(per[lbl]["flops"])),
                     "cb": s_raw.mean_multiplier,
                     "adjusted_raw": s_raw.objective, "adjusted_sub": s_sub.objective}
        print(f"  {lbl:<18}{s_raw.mean_mse:>13.5e}{s_sub.mean_mse:>15.5e}"
              f"{summ[lbl]['counted']:>14.4e}{summ[lbl]['cb']:>8.4f}"
              f"{s_raw.objective:>13.5e}")

    out = {"summary": summ, "pairs": {}}
    pairs = [(labels[1], labels[2], "the DESIGN's own contribution, matched cost"),
             (labels[0], labels[2], "vs the shipped operating point"),
             (labels[0], labels[1], "the flat-adjusted law, as a check")]
    for ctrl, cand, why in pairs:
        for gt_sub in (False, True):
            sc = score_suite(rows(ctrl, gt_sub), on_over_budget=MODE)
            sk = score_suite(rows(cand, gt_sub), on_over_budget=MODE)
            r = paired_compare(sc.scores, sk.scores)
            mse_gain = sc.mean_mse / sk.mean_mse
            tag = "GT-subtracted" if gt_sub else "raw"
            key = f"{ctrl} -> {cand} [{tag}]"
            out["pairs"][key] = {
                "why": why, "mse_gain": mse_gain,
                "arithmetic_gain": r.arithmetic_gain,
                "relative_improvement": r.relative_improvement,
                "t_ci_difference": list(r.t_ci_difference),
                "boot_ci_relative": list(r.boot_ci_relative),
                "geometric_gain": r.geometric_gain,
                "n_candidate_better": r.n_candidate_better,
            }
            if not gt_sub:
                print(f"\n--- {ctrl}  ->  {cand}   ({why}) " + "-" * 8)
                print(format_paired(r, control=ctrl, candidate=cand))
            print(f"  [{tag}] raw-MSE gain {mse_gain:.4f}x   "
                  f"adjusted gain {r.arithmetic_gain:.4f}x   "
                  f"better on {r.n_candidate_better}/{r.n}")

    # What the grader would ACTUALLY do with an over-budget arm.
    print("\n--- what the grader would actually do (C > B is a FAILURE) " + "-" * 20)
    from competition_score import over_budget_rows
    for lbl in labels:
        rr = rows(lbl, False)
        bad = over_budget_rows(rr)
        sf = score_suite(rr, on_over_budget="fail")
        print(f"  {lbl:<18}{len(bad):>4}/{len(rr)} MLPs over budget   "
              f"objective if failed {sf.objective:.5e}   "
              f"({'DEPLOYABLE' if not bad else 'NOT DEPLOYABLE'})")
    out["over_budget"] = {lbl: len(over_budget_rows(rows(lbl, False))) for lbl in labels}

    os.makedirs(OUTDIR, exist_ok=True)
    with open(SCORE_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nsaved {SCORE_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["truth", "arms", "score"])
    ap.add_argument("--mlps", type=int, default=192)
    ap.add_argument("--n-gt", dest="n_gt", type=int, default=1 << 24)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--date", default="2026-08-19")
    a = ap.parse_args()
    {"truth": phase_truth, "arms": phase_arms, "score": phase_score}[a.phase](a)


if __name__ == "__main__":
    main()
