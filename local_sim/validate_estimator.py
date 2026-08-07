"""
Held-out validation of the FINAL estimator.py (with baked-in calibration)
on fresh MLPs (seeds disjoint from the calibration fit).

Reports final-layer MSE raw vs calibrated, plus the approximate adjusted
leaderboard score (MSE x 0.1, since covariance propagation uses <10% of the
FLOP budget and lands on the multiplier floor).

NOTE (2026-08-06): `gen_mlp` now mints grader-identical networks via
mint_provenance.canonical_mint, so the SAME seeds produce DIFFERENT (correct)
networks than before. The v0 reference numbers in older docs (raw 7.98e-5 ->
calibrated 2.67e-5) were measured on the legacy mint and will not reproduce
bit-for-bit; the level is unchanged. Seeds here are legacy block L1.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

from harness import gen_mlp, mc_layer_means, run_estimator  # noqa: E402

WIDTH = 256
DEPTH = 32
K = 6
N_SAMPLES = 200_000


def main():
    mses_cal, mses_raw = [], []
    for k in range(K):
        mlp = gen_mlp(WIDTH, DEPTH, seed=9_000_000 + k)  # fresh seeds
        truth = mc_layer_means(mlp, N_SAMPLES, seed=8_500_000 + k)

        pred_cal = run_estimator(mlp)

        os.environ["ARC_WHEST_DISABLE_CAL"] = "1"
        pred_raw = run_estimator(mlp)
        del os.environ["ARC_WHEST_DISABLE_CAL"]

        m_cal = float(((pred_cal[-1] - truth[-1]) ** 2).mean())
        m_raw = float(((pred_raw[-1] - truth[-1]) ** 2).mean())
        mses_cal.append(m_cal)
        mses_raw.append(m_raw)
        print(f"mlp {k}: final-layer MSE raw {m_raw:.3e} | calibrated {m_cal:.3e}", flush=True)

    raw, cal = float(np.mean(mses_raw)), float(np.mean(mses_cal))
    print("-" * 60)
    print(f"mean final-layer MSE  raw        : {raw:.3e}")
    print(f"mean final-layer MSE  calibrated : {cal:.3e}")
    print(f"improvement factor               : {raw / cal:.2f}x")
    print(f"approx adjusted score (MSE x 0.1): {cal * 0.1:.3e}")
    print("(MC measurement noise adds ~var/N ≈ 1e-6-ish per-neuron^2 -> "
          f"~{0.4 / N_SAMPLES:.1e} to each reported MSE; true MSEs are slightly lower)")


if __name__ == "__main__":
    main()
