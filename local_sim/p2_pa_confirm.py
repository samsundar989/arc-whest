"""P-A-CONFIRM: read K3 against ORGANISER networks and ORGANISER labels.

Contract: `experiments/LOG.md`, "CONTRACT (Rule 1): P-A-CONFIRM", committed
before any organiser label was loaded. The band and the falsifier are the
independent Skeptic's, adopted verbatim.

## The estimand is different here, and it is quoted honestly

P-A used the unbiased two-half cross estimator because we built both halves. The
organisers ship **one** label, so the estimand here is the ordinary MSE, which is
biased UPWARD by exactly their label's variance. Theirs is `n_samples = 1e9` by
the same fp32-forward / fp64-accumulate convention we used
(`whestbench/_simulation_torch.py:96-101`), so that bias is
`avg_variance / 1e9 ~ 7.5e-11` — **0.19% of the quantity being measured.** It is
reported, not corrected.

## READ-ONLY, and this is the whole reason a CONFIRM split may be touched

Nothing here is fitted, chosen, ranked or thresholded on this split. The number
under test was fixed in `698588b`; the band was fixed by an independent reviewer
before any organiser label was read. This module verifies; it does not select.

Run (GPU venv):
    .venv-gpu\\Scripts\\python.exe -u local_sim/p2_pa_confirm.py \\
        --labels experiments/2026-08-27_pa_confirm/mini_labels.npz \\
        --out experiments/2026-08-27_pa_confirm/pa_confirm.json
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

# The Skeptic's band and falsifier, and the P-A number they are testing.
BAND = (3.0e-08, 5.5e-08)
FALSIFIER = 8.0e-08
PA_DESIGN = 4.0047e-08          # P-A, 24 DESIGN mints, cross estimator
AGREE_PCT = 15.0                # my prediction 3
CANARY_GRADED_MSE = 3.933e-06   # submission 328662, organiser networks
CANARY_TOL_PCT = 10.0           # my prediction 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch

    import p2_cal_lab as lab
    import p2_kprop_bridge as bridge
    from mint_provenance import canonical_mint

    z = np.load(os.path.join(_ROOT, a.labels), allow_pickle=False)
    rev = str(z["revision"]) if "revision" in z else "(absent)"
    W, D = int(z["width"]), int(z["depth"])
    if rev != "v2-phase2" or (W, D) != (1024, 16):
        raise SystemExit(
            f"REFUSED: labels are revision {rev!r} at {W}x{D}. The live Phase-2 "
            "round is v2-phase2 at 1024x16. real_data.REVISION still defaults to "
            "the Phase-1 revision, which is exactly how a wrong-round label set "
            "gets this far looking normal."
        )
    seeds = [int(s) for s in z["seeds"]]
    labels = z["labels"]
    avgvar = z["avg_variance"]
    if a.limit:
        seeds, labels, avgvar = seeds[:a.limit], labels[:a.limit], avgvar[:a.limit]

    label_noise = float(np.mean(avgvar)) / 1e9
    print("=" * 78)
    print("P-A-CONFIRM  --  K3 vs ORGANISER networks and ORGANISER labels")
    print("=" * 78)
    print(f"  dataset          {rev}  split {str(z['split'])}  {len(seeds)} MLPs at {W}x{D}")
    print(f"  organiser label  n_samples 1e9, mean avg_variance {np.mean(avgvar):.6f}")
    print(f"  -> label noise   {label_noise:.4e}  = {label_noise / PA_DESIGN * 100:.2f}% of "
          f"the P-A value (an UPWARD bias on plain MSE, reported not corrected)")
    print(f"  band             [{BAND[0]:.1e}, {BAND[1]:.1e}]   hard falsifier > {FALSIFIER:.1e}")
    print(f"  testing          P-A's {PA_DESIGN:.4e} from 24 DESIGN mints")
    v = bridge.verify_upstream(bridge._src())
    if not v["pass"]:
        raise SystemExit("REFUSED: upstream is not the pinned rev + recorded patch.")
    print(f"  upstream         {v['rev'][:12]} + recorded patch, verified")
    print()

    rows = []
    t0 = time.perf_counter()
    for i, s in enumerate(seeds):
        w, _e, _ss = canonical_mint(W, D, s)
        y = labels[i]
        k2 = lab.closure(w)
        k3 = bridge.kprop_means(w, k_max=3, factor=True, dtype=torch.float32)
        r = {"seed": s,
             "mse_k2": float(np.mean((k2[-1] - y[-1]) ** 2)),
             "mse_k3": float(np.mean((k3[-1] - y[-1]) ** 2)),
             "mse_k3_all_layers": float(np.mean((k3 - y) ** 2)),
             "finite": bool(np.isfinite(k3).all())}
        rows.append(r)
        if (i + 1) % 10 == 0 or i + 1 == len(seeds):
            el = time.perf_counter() - t0
            m = np.mean([x["mse_k3"] for x in rows])
            print(f"    {i + 1:>4}/{len(seeds)}   running mean k3 {m:.4e}   "
                  f"{el:6.1f}s   eta {el / (i + 1) * (len(seeds) - i - 1):5.0f}s")

    m3 = np.array([r["mse_k3"] for r in rows])
    m2 = np.array([r["mse_k2"] for r in rows])
    n = len(m3)
    mean3, sd3 = float(m3.mean()), float(m3.std(ddof=1))
    se3 = sd3 / np.sqrt(n)
    try:
        from scipy import stats
        crit = float(stats.t.ppf(0.975, n - 1))
    except Exception:
        crit = 1.984
    ci = (mean3 - crit * se3, mean3 + crit * se3)

    in_band = BAND[0] <= mean3 <= BAND[1]
    falsified = mean3 > FALSIFIER
    agree = abs(mean3 - PA_DESIGN) / PA_DESIGN * 100
    canary_dev = abs(float(m2.mean()) - CANARY_GRADED_MSE) / CANARY_GRADED_MSE * 100

    print()
    print(f"  K2  mean {m2.mean():.4e}   median {np.median(m2):.4e}")
    print(f"  K3  mean {mean3:.4e}   median {np.median(m3):.4e}   se {se3:.3e}   "
          f"95% [{ci[0]:.4e}, {ci[1]:.4e}]")
    print(f"      min {m3.min():.4e}   max {m3.max():.4e}   spread {m3.max()/m3.min():.2f}x")
    print()
    print(f"  P1 band [{BAND[0]:.1e}, {BAND[1]:.1e}]           -> {mean3:.4e}   "
          f"{'HELD' if in_band else 'MISSED'}")
    print(f"  P2 hard falsifier > {FALSIFIER:.1e}            -> "
          f"{'FIRED -- D1 IS BACK IN PLAY' if falsified else 'not fired'}")
    print(f"  P3 agrees with P-A within {AGREE_PCT:.0f}%          -> {agree:.2f}%   "
          f"{'HELD' if agree <= AGREE_PCT else 'MISSED'}")
    print(f"  P4 K2 control vs graded canary {CANARY_GRADED_MSE:.3e} -> "
          f"{m2.mean():.4e}, {canary_dev:.2f}%   "
          f"{'HELD' if canary_dev <= CANARY_TOL_PCT else 'MISSED'}")

    doc = {"slice": "P-A-CONFIRM", "revision": rev, "split": str(z["split"]),
           "n_mlps": n, "width": W, "depth": D,
           "estimand": "plain MSE vs the organisers' 1e9-sample label",
           "label_noise_upward_bias": label_noise,
           "band": list(BAND), "falsifier": FALSIFIER,
           "pa_design_value": PA_DESIGN,
           "k3_mean": mean3, "k3_median": float(np.median(m3)), "k3_se": se3,
           "k3_ci95": list(ci), "k3_min": float(m3.min()), "k3_max": float(m3.max()),
           "k2_mean": float(m2.mean()),
           "checks": {"P1_in_band": bool(in_band),
                      "P2_falsifier_fired": bool(falsified),
                      "P3_agreement_pct": agree,
                      "P3_held": bool(agree <= AGREE_PCT),
                      "P4_canary_dev_pct": canary_dev,
                      "P4_held": bool(canary_dev <= CANARY_TOL_PCT)},
           "upstream": v, "rows": rows,
           "tier": "CONFIRM, read-only. Nothing fitted, chosen or ranked here."}
    if a.out:
        p = os.path.abspath(os.path.join(_ROOT, a.out))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(doc, open(p, "w", encoding="utf-8"), indent=2)
        print(f"\n  wrote {os.path.relpath(p, _ROOT)}")
    print("=" * 78)
    return 0 if (in_band and not falsified) else 1


if __name__ == "__main__":
    raise SystemExit(main())
