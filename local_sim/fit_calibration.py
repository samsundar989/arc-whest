"""
Fit per-layer multiplicative calibration factors for the covariance-propagation
estimator, offline, on self-generated He-init MLPs vs Monte Carlo truth.

For each layer l we fit the least-squares scalar
    c_l = sum(pred_l * truth_l) / sum(pred_l * pred_l)
pooled over K MLPs (this is unbiased w.r.t. independent MC noise in truth).

Outputs experiments/calibration.json and prints a summary. Run time on 2 vCPU:
a few minutes (dominated by the Monte Carlo).

NOTE (2026-08-06): mints are now grader-identical (mint_provenance), so a refit
will not reproduce experiments/calibration.json exactly. Refit only if the
shape changes; the table is used solely by the analytic fallback path.
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

os.environ["ARC_WHEST_DISABLE_CAL"] = "1"  # fit on the RAW estimator

import numpy as np  # noqa: E402

from harness import gen_mlp, mc_layer_means, run_estimator  # noqa: E402

WIDTH = 256
DEPTHS = [8, 16, 32]
K = 5                 # MLPs per depth
N_SAMPLES = 200_000   # MC samples per MLP  (per-neuron sd ~ sqrt(0.4/2e5) ~ 1.4e-3)

def main():
    results = {"width": WIDTH, "k_mlps": K, "n_samples": N_SAMPLES, "depths": {}}
    for depth in DEPTHS:
        t0 = time.time()
        preds, truths = [], []
        for k in range(K):
            mlp = gen_mlp(WIDTH, depth, seed=1_000 * depth + k)
            preds.append(run_estimator(mlp))
            truths.append(
                mc_layer_means(mlp, N_SAMPLES, seed=5_000_000 + 1_000 * depth + k)
            )
        P = np.stack(preds)   # (K, depth, n)
        T = np.stack(truths)

        c_layers = (P * T).sum(axis=(0, 2)) / (P * P).sum(axis=(0, 2))
        mse_raw = float(((P[:, -1] - T[:, -1]) ** 2).mean())
        mse_cal = float(((P[:, -1] * c_layers[-1] - T[:, -1]) ** 2).mean())

        results["depths"][str(depth)] = {
            "c_layers": [float(c) for c in c_layers],
            "c_final": float(c_layers[-1]),
            "final_layer_mse_raw": mse_raw,
            "final_layer_mse_scalar_cal": mse_cal,
            "seconds": round(time.time() - t0, 1),
        }
        print(
            f"depth {depth:3d} | c_final {c_layers[-1]:.6f} | "
            f"final-layer MSE raw {mse_raw:.3e} -> cal {mse_cal:.3e} | "
            f"{time.time()-t0:.0f}s",
            flush=True,
        )

    # MC-vs-MC noise floor at depth 32 (measurement noise in this experiment)
    mlp = gen_mlp(WIDTH, 32, seed=777_001)
    a = mc_layer_means(mlp, N_SAMPLES, seed=1)
    b = mc_layer_means(mlp, N_SAMPLES, seed=2)
    floor = float(((a[-1] - b[-1]) ** 2).mean())
    results["mc_vs_mc_final_mse_200k"] = floor
    print(f"MC(200k) vs MC(200k) final-layer MSE: {floor:.3e}  (2*var/N)", flush=True)

    out = os.path.join(os.path.dirname(_HERE), "experiments", "calibration.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
