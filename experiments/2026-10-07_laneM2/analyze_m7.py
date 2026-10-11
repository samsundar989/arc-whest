"""M7 table: v77's coherent early-layer error vs MC labels (m7_state.npz + m7_mc.npz).

Relative errors (v77 − truth)/truth, per layer 1-6, mean ± 95 % over nets; the MC standard
error (10 blocks) is reported beside each so a number below it reads as "not resolved".

  mean_mhat    m̂ᵀ(μ_v77 − μ) / ‖μ‖              coherent mean (scale) error   budget 5e-5
  var_mhat     (m̂ᵀC_v77 m̂ − m̂ᵀCm̂) / m̂ᵀCm̂      covariance along the gain      budget 1e-4
  var_u1..u4   same along v77's top-4 eigvecs
  trC          (tr C_v77 − tr C) / tr C          variance level (= mean σ²)     budget 1e-4
  mse_layer    mean_i (μ_v77 − μ)_i² minus the MC floor;  frac_mhat/top4 = share of ‖δμ‖² in those directions

Final-MSE projection uses R2 §3.2's measured transfer (final MSE per η²): covariance
l0 = 1/2/4: 0.128/0.076/0.040 (interpolated at 3, 5, 6 log-linearly), coherent mean
direction gain 0.45-0.64 with E_i[m²] ≈ 0.5-0.65.

    uv run python experiments/2026-10-07_laneM2/analyze_m7.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T_COV = {1: 0.128, 2: 0.076, 3: 0.055, 4: 0.040, 5: 0.026, 6: 0.017}   # R2 §3.2; 3/5/6 interpolated
T_MEAN = {1: 0.21, 2: 0.16, 3: 0.135, 4: 0.115, 5: 0.090, 6: 0.072}    # uniform mean shift, per η²


def ci(x):
    x = np.asarray(x, float)
    return 2.365 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def main() -> int:
    st0 = np.load(HERE / "m7_state.npz")
    parts = [np.load(p) for p in sorted(HERE.glob("m7_mc_*.npz"))]
    rows_ = np.concatenate([p["rows"] for p in parts])
    S1, S2, P1, P2 = (np.concatenate([p[k] for p in parts]) for k in ("S1", "S2", "P1", "P2"))  # (K, B, L, ...)
    st = {k: st0[k][rows_] for k in ("U", "mu", "var", "uCu")}
    K = len(rows_)
    N_block = int(parts[0]["per_block"])
    B = S1.shape[1]
    L = S1.shape[2]
    keys = ["mean_mhat", "var_mhat", "var_u1", "var_u2", "var_u3", "var_u4", "trC", "mse_layer", "frac_mhat", "frac_top4"]
    rows = {k: np.zeros((K, L)) for k in keys}
    se = {k: np.zeros((K, L)) for k in keys}
    for k in range(K):
        for l in range(L):
            mu_b = S1[k, :, l]                       # (B, n)
            var_b = S2[k, :, l] - mu_b ** 2
            mu, var = mu_b.mean(0), S2[k, :, l].mean(0) - mu_b.mean(0) ** 2
            p1b, p2b = P1[k, :, l], P2[k, :, l]       # (B, 5), (B, 5, 5)
            pv_b = np.einsum("baa->ba", p2b) - p1b ** 2
            p1, pv = p1b.mean(0), np.einsum("aa->a", p2b.mean(0)) - p1b.mean(0) ** 2
            U = st["U"][k, l]
            mu_v, var_v, uCu = st["mu"][k, l], st["var"][k, l], st["uCu"][k, l]
            mnorm = np.linalg.norm(mu)
            rows["mean_mhat"][k, l] = (U[:, 0] @ mu_v - p1[0]) / mnorm
            se["mean_mhat"][k, l] = p1b[:, 0].std(ddof=1) / np.sqrt(B) / mnorm
            for j, name in enumerate(["var_mhat", "var_u1", "var_u2", "var_u3", "var_u4"]):
                rows[name][k, l] = (uCu[j] - pv[j]) / pv[j]
                se[name][k, l] = pv_b[:, j].std(ddof=1) / np.sqrt(B) / pv[j]
            rows["trC"][k, l] = (var_v.sum() - var.sum()) / var.sum()
            se["trC"][k, l] = var_b.sum(1).std(ddof=1) / np.sqrt(B) / var.sum()
            d = mu_v - mu
            floor = var.mean() / (N_block * B)
            rows["mse_layer"][k, l] = (d ** 2).mean() - floor
            se["mse_layer"][k, l] = floor
            Q, _ = np.linalg.qr(U)
            rows["frac_mhat"][k, l] = (U[:, 0] @ d) ** 2 / (d @ d)
            rows["frac_top4"][k, l] = ((Q[:, 1:].T @ d) ** 2).sum() / (d @ d)
    lines = ["| layer | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for l in range(L):
        cells = []
        for k_ in keys:
            x = rows[k_][:, l]
            cells.append(f"{x.mean():+.2e} ± {ci(x):.1e} (se {se[k_][:, l].mean():.0e})")
        lines.append(f"| {l + 1} | " + " | ".join(cells) + " |")
    # projected final-MSE cost of the coherent parts (η² × transfer)
    # R2's transfer coefficients are per η² of a relative error of the WHOLE covariance. A
    # relative error η_m of the variance along m̂ is a rank-1 perturbation; in tr C units it is
    # η_m · (m̂ᵀCm̂ / tr C). Compare like with like: η_eq = max(|trC err|, |η_m| · share_m̂).
    share = st["uCu"][:, :, 0] / np.array([[np.trace(np.diag(v)) for v in vv] for vv in st["var"]])
    proj = ["", "| layer | share of tr C along m̂ | var-level η (trC) | m̂ η in trC units | budget (R2) | → final MSE (η_eq²·T) | coherent mean η | budget | → final MSE |",
            "|---|---|---|---|---|---|---|---|---|"]
    BUD_C = {1: 9e-5, 2: 1.1e-4, 3: 1.35e-4, 4: 1.6e-4, 5: 2.0e-4, 6: 2.3e-4}
    BUD_M = {1: 5e-5, 2: 5e-5, 3: 6e-5, 4: 7e-5, 5: 7e-5, 6: 7e-5}
    tot = 0.0
    for l in range(L):
        eq_m = rows["var_mhat"][:, l] * share[:, l]
        eta_c = np.sqrt(np.mean(np.maximum(rows["trC"][:, l] ** 2, eq_m ** 2)))
        eta_m = np.sqrt(np.mean(rows["mean_mhat"][:, l] ** 2))
        fc, fm = T_COV[l + 1] * eta_c ** 2, T_MEAN[l + 1] * eta_m ** 2
        tot += fc + fm
        proj.append(f"| {l + 1} | {share[:, l].mean():.4f} | {np.sqrt(np.mean(rows['trC'][:, l] ** 2)):.2e} | "
                    f"{np.sqrt(np.mean(eq_m ** 2)):.2e} | {BUD_C[l + 1]:.1e} | {fc:.1e} | {eta_m:.2e} | "
                    f"{BUD_M[l + 1]:.0e} | {fm:.1e} |")
    proj.append(f"\nsum over layers 1-6 (upper bound, errors not independent): {tot:.2e} "
                f"vs v77 raw final MSE ≈ 2.2e-8")
    txt = "\n".join(lines + proj)
    print(txt)
    (HERE / "m7_table.md").write_text(txt + "\n", encoding="utf-8")
    (HERE / "m7_summary.json").write_text(json.dumps(
        {k: {"mean": rows[k].mean(0).tolist(), "rms": np.sqrt((rows[k] ** 2).mean(0)).tolist()} for k in keys},
        indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
