"""P-A: how much information does ARC's factorized K3 actually carry at 1024 x 16?

Contract: `experiments/LOG.md`, entry "CONTRACT (Rule 1): P-A". Samuel's
prediction was fixed in the P2-GT contract (`ca18439`) **before** any kprop run
existed in this tree, which is what makes this a test rather than a description.

Reads the P2-GT asset (`gt_halfA.npz` / `gt_halfB.npz`) and never averages the
halves -- see the estimator note below.

## The estimand, and why it is not an ordinary MSE

A single-replica MSE is biased upward by exactly the label's own variance:
`E[(p - y)^2] = MSE(p) + Var(y)`. At the accuracy we are now trying to resolve
(`~3e-8`) the old label's noise was `2.1e-8`, i.e. the bias was 70% of the
signal. The two-half cross estimator

    xmse(p) = mean_j (p_j - A_j)(p_j - B_j)

has expectation `MSE(p)` **exactly**, because the halves' errors are independent
by construction. Every number this module reports is that estimator. The plain
single-half MSEs are reported **beside** it, not instead of it, so the size of
the correction is visible rather than taken on trust.

## Arms

| arm | what | why it is here |
|---|---|---|
| `k2` | `p2_cal_lab.closure` -- the NumPy mirror of the shipped K2p1 arm | the control. It is what we ship today. |
| `k2cal` | the same, with v2.7's sixteen output-row constants | the control that is actually on the board |
| `k3` | upstream `mlp_kprop`, `k_max=3, factor=True`, UNMODIFIED | **the arm under test** |

`k3` is run through `local_sim/p2_kprop_bridge.py`, which re-verifies on every
call that the upstream checkout is rev `93d091a4` plus exactly the recorded
patch, and refuses otherwise.

## dtype

The default is **float32**, measured at 89 s and 22.3 GiB peak per network
against float64's 193 s and 44.6 GiB, with the final-row mean agreeing to six
decimals on the probe network. `--dtype-delta N` additionally runs the first `N`
networks in float64 and reports the paired difference, which is Samuel's
open item (iii) and the evidence the f32 port default needs. **The dtype delta
is measured, never assumed** -- `docs/GOTCHAS.md` records flopscope pricing
counterexamples in both directions and the same caution applies to accuracy.

## What this module does NOT do

No cost is claimed. Upstream's FLOP model is not flopscope pricing
(`p2_kprop_bridge.py --flops` prints that caveat on every run), and P-A is an
**information** question only: can the method carry the accuracy at all, at any
price. P-B is the price question and it is a separate contract.

The asset is **P01, a DESIGN block.** A DESIGN measurement orients; it does not
close a lane. If P-A's falsifier fires, the consequence is a change of research
priority, which is a planning decision -- not a lane kill, which would need
CONFIRM.

Run (GPU venv):
    .venv-gpu\\Scripts\\python.exe -u local_sim/p2_pa_kprop.py \\
        --gt-dir experiments/2026-08-26_p2_gt --out experiments/2026-08-26_p2_gt/pa_k3.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np  # noqa: E402

# Samuel's pre-registered P-A prediction, fixed in ca18439 before any data.
PA_BAND = (1.5e-8, 8.0e-8)
PA_FALSIFIER = 1.5e-7


def xmse(pred, A, B):
    """Unbiased two-half cross estimator. E[(p-A)(p-B)] = MSE(p)."""
    return float(np.mean((pred - A) * (pred - B)))


def studentised_t(d, alpha=0.05):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    try:
        from scipy import stats
        crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    except Exception:
        crit = 1.96 if n > 60 else 2.064
    return m, se, (m - crit * se, m + crit * se)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="experiments/2026-08-26_p2_gt")
    ap.add_argument("--out", default="")
    ap.add_argument("--dtype-delta", type=int, default=3,
                    help="also run this many networks in float64 and report the "
                         "paired f32-vs-f64 accuracy difference")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch

    import p2_cal_lab as lab
    import p2_kprop_bridge as bridge
    from mint_provenance import canonical_mint

    gt = os.path.join(_ROOT, a.gt_dir)
    man = json.load(open(os.path.join(gt, "gt_manifest.json"), encoding="utf-8"))
    if not man.get("all_pass"):
        raise SystemExit("REFUSED: the P2-GT asset did not pass its own fitness "
                         "checks. An arm measured against a void label is not a "
                         "measurement.")
    zA = np.load(os.path.join(gt, "gt_halfA.npz"))
    zB = np.load(os.path.join(gt, "gt_halfB.npz"))
    seeds = [int(s) for s in zA["seeds"]]
    assert seeds == [int(s) for s in zB["seeds"]], "halves disagree about seeds"
    A_all, B_all = zA["means"], zB["means"]
    if a.limit:
        seeds, A_all, B_all = seeds[:a.limit], A_all[:a.limit], B_all[:a.limit]

    print("=" * 78)
    print("P-A  --  factorized K3 information content at 1024 x 16")
    print("=" * 78)
    print(f"  asset            {a.gt_dir}  ({len(seeds)} networks, "
          f"{man['samples_per_half']:,} samples per half)")
    print(f"  label noise/half {man['gt_noise_per_half_mean']:.4e}")
    print(f"  cross-est SE     {man['cross_estimator_se_at_target_mse']['max_over_mlps']:.3e} "
          f"at MSE 3e-8")
    print(f"  prediction       mean true final-layer MSE in "
          f"[{PA_BAND[0]:.1e}, {PA_BAND[1]:.1e}]   falsifier > {PA_FALSIFIER:.1e}")
    print()

    _ = bridge._src()
    v = bridge.verify_upstream(bridge._src())
    if not v["pass"]:
        raise SystemExit("REFUSED: upstream checkout is not the pinned rev + patch.")
    print(f"  upstream         {v['rev'][:12]} + recorded patch, verified")
    print()

    rows = []
    g = np.asarray(_v27_gains(), dtype=np.float64)
    t0 = time.perf_counter()
    for i, s in enumerate(seeds):
        w, _e, _ss = canonical_mint(man["width"], man["depth"], s)
        Af, Bf = A_all[i, -1, :], B_all[i, -1, :]
        k2 = lab.closure(w)
        k2p = lab.predict(w)
        k2cal = k2p * g[:, None]
        t1 = time.perf_counter()
        k3 = bridge.kprop_means(w, k_max=3, factor=True, dtype=torch.float32)
        dt = time.perf_counter() - t1
        row = {
            "seed": s,
            "xmse_k2": xmse(k2[-1], Af, Bf),
            "xmse_k2cal": xmse(k2cal[-1], Af, Bf),
            "xmse_k3": xmse(k3[-1], Af, Bf),
            "plain_mse_k3_halfA": float(np.mean((k3[-1] - Af) ** 2)),
            "k3_seconds": dt,
            "k3_finite": bool(np.isfinite(k3).all()),
        }
        if a.dtype_delta and i < a.dtype_delta:
            k3d = bridge.kprop_means(w, k_max=3, factor=True, dtype=torch.float64)
            row["xmse_k3_f64"] = xmse(k3d[-1], Af, Bf)
            row["f32_minus_f64"] = row["xmse_k3"] - row["xmse_k3_f64"]
        rows.append(row)
        el = time.perf_counter() - t0
        print(f"    {i + 1:>3}/{len(seeds)}  seed {s}   "
              f"k2 {row['xmse_k2']:.4e}   k2cal {row['xmse_k2cal']:.4e}   "
              f"k3 {row['xmse_k3']:.4e}   ({dt:.0f}s, eta {el / (i + 1) * (len(seeds) - i - 1) / 60:.0f}m)")

    x3 = np.array([r["xmse_k3"] for r in rows])
    x2 = np.array([r["xmse_k2"] for r in rows])
    x2c = np.array([r["xmse_k2cal"] for r in rows])
    m3, se3, ci3 = studentised_t(x3)
    d, sed, cid = studentised_t(x2c - x3)

    in_band = PA_BAND[0] <= m3 <= PA_BAND[1]
    falsified = m3 > PA_FALSIFIER
    print()
    print(f"  K2      mean xmse {x2.mean():.4e}   median {np.median(x2):.4e}")
    print(f"  K2+cal  mean xmse {x2c.mean():.4e}   median {np.median(x2c):.4e}")
    print(f"  K3      mean xmse {m3:.4e}   median {np.median(x3):.4e}   "
          f"se {se3:.3e}   95% [{ci3[0]:.4e}, {ci3[1]:.4e}]")
    print(f"  paired K2+cal - K3: {d:.4e}   95% [{cid[0]:.4e}, {cid[1]:.4e}]   "
          f"K3 better on {int((x2c > x3).sum())}/{len(x3)}")
    print()
    print(f"  PREDICTION P-A: mean in [{PA_BAND[0]:.1e}, {PA_BAND[1]:.1e}] -> "
          f"{'HELD' if in_band else 'MISSED'}  (measured {m3:.4e})")
    print(f"  FALSIFIER     : mean > {PA_FALSIFIER:.1e} -> "
          f"{'FIRED -- pivot primary to A0 + readout + corrector' if falsified else 'not fired'}")
    dd = [r["f32_minus_f64"] for r in rows if "f32_minus_f64" in r]
    if dd:
        print(f"  f32 - f64 xmse over {len(dd)} networks: mean {np.mean(dd):+.3e} "
              f"(vs K3 xmse {m3:.3e}, i.e. {abs(np.mean(dd)) / m3 * 100:.4f}% of it)")

    doc = {
        "slice": "P-A", "asset": a.gt_dir, "n_mlps": len(seeds),
        "prediction_band": list(PA_BAND), "falsifier": PA_FALSIFIER,
        "upstream": v, "label_noise_per_half": man["gt_noise_per_half_mean"],
        "k3_mean_xmse": m3, "k3_se": se3, "k3_ci95": list(ci3),
        "k3_median_xmse": float(np.median(x3)),
        "k2_mean_xmse": float(x2.mean()), "k2cal_mean_xmse": float(x2c.mean()),
        "paired_k2cal_minus_k3": d, "paired_ci95": list(cid),
        "prediction_held": bool(in_band), "falsifier_fired": bool(falsified),
        "f32_minus_f64_mean": float(np.mean(dd)) if dd else None,
        "rows": rows,
        "tier": "DESIGN (P01). Orients priority; does NOT close a lane.",
        "cost_note": "No cost claim. Upstream's FLOP model is not flopscope pricing.",
    }
    if a.out:
        p = os.path.abspath(os.path.join(_ROOT, a.out))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(doc, open(p, "w", encoding="utf-8"), indent=2)
        print(f"  wrote {os.path.relpath(p, _ROOT)}")
    print("=" * 78)
    return 0


def _v27_gains():
    """The sixteen shipped v2.7 constants, read from the file rather than retyped."""
    import re
    src = open(os.path.join(_ROOT, "estimator_v2_7.py"), encoding="utf-8").read()
    m = re.search(r"_CAL_G\s*=\s*\(([^)]*)\)", src, re.S)
    if not m:
        raise SystemExit("could not read _CAL_G from estimator_v2_7.py")
    return [float(x) for x in re.findall(r"[-+0-9.eE]+", m.group(1))]


if __name__ == "__main__":
    raise SystemExit(main())
