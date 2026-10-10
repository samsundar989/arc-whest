# Lane M5 results (lane-m) — RESULTS-style rows

Chain: twin v81d = `v62_gb` + mask (R2.0, Hd2.5, Dz2.5, K = aggr_q50) + dress all + bank star,starall,
params `refit_starall2` (VAL 1.945887e-08 = v81d's recorded VAL, exactly). Panel VAL-100 = full rows
900–999. Raw JSONs in `val/`, `smoke/`. Arms: `tools/twin/k4x_arm.py`.

## 2026-10-10 — unrefit, VAL-100

| arm | VAL raw | Δ vs base | wins/100 | notes |
|---|---:|---:|---:|---|
| base (v81d twin) | 1.945887e-08 | — | — | |
| M5.1 fold ns (exact, τ 2, g★ = sg) | 1.927409e-08 | −0.95 % | 56 | D21 change = g·ω·(r/3)·Q★ to 3e-4 rel. (float32); |ΔH|/|H| ≤ 1.3 %, |Δd21e| ≤ 0.4 % (smoke, 3 nets) |
| M5.1 fold st (zero products; `_qold` sees the fold) | 1.927303e-08 | −0.96 % | 57 | |
| **M5.1 fold rec (zero products; recent lineages only)** | **1.921925e-08** | **−1.23 %** | 62 | the free shippable variant; old-tier fold does not help |
| M5.2 radial all (v′ physical, quenched law) | 2.386682e-07 | +1127 % | 0 | |
| M5.2 radial Q31 only | 2.022047e-08 | +3.91 % | 7 | |
| M5.2 radial D22 only | 2.178374e-07 | +1019 % | 0 | |
| M5.2 radial D4 only | 2.439048e-08 | +25.3 % | 0 | |
| M5.4 step 0: Hd2.5 by threshold (K table R, D only) | 1.944064e-08 | −0.09 % | 57 | reference for Hkh |
| **M5.4 step 0: Hkh2.5** | 1.947732e-08 | +0.09 % (+0.19 % vs Hd2.5) | 44 | **passes** (stop bar +1 %) |
| M5.4 step 0: H2.5 literal | 1.976883e-08 | +1.59 % | 21 | P1 measured +3.41 % on v73 |

Predictions on record (`PREDICTIONS.md`): fold unrefit 0 to +1 % (mine), ±1 % (strategist) → measured −1.2 %,
better than both. Radial unrefit −0.3 to +0.3 % (mine) → measured catastrophic at the physical v′.
Hkh2.5 +0.0 to +0.5 % (mine) → +0.19 %.

## 2026-10-10 — refits (1 epoch TRAIN-900, lr 1e-4, cosine, from `refit_starall2`; best-step VAL-100)

| arm | stage | VAL raw (step) | vs matched control | verdict |
|---|---|---:|---:|---|
| control (v81d twin, no arm) | S2ts | 1.941552e-08 (112) | — | |
| **M5.1 fold rec (τ 2, g★ init = sg)** | S2tsf | **1.891407e-08** (112) | **−2.58 %** | **PASS** (kill: not better than control) |

Learned g★ ≈ sg (moved < 0.5 %). Params: `refit_fold_rec/params_best.npz`.
Interpreter on v81d (`build_fold.py`, rec mode): +983,123,392 FLOPs (+0.254 % F, +0.46 u), +243 ops
(25,741 → 25,984), matmul total unchanged. Marginal rule −Δraw > ΔF/(1+ΔF): 2.58 % ≫ 0.25 %.

## M5.3 regression (twin v81d, FIT nets 0–9 fit / 10–19 test, weighted by how Q31 / D4 enter)

`k4reg/report.md`, `k4reg/summary.json`. Label-noise ceiling ≈ 1.0 (Gaussian approx.), so R² gaps are model error.

- Q31: R²_S0 = 0.36–0.49. Free pieces, single ΔR² at l ≥ 6: **star_all 0.028–0.056** (rises with depth),
  **radial 0.008–0.06** (its α_b·γ_a half carries most); chain D21/D3 ≈ true D21/D3 (representation is not
  the limit). Paid: path (age 1) +0.01–0.02, (2,2)-block ≤ 0.02 → fail the build rule. Greedy best set at
  l 10–14 (star_all, radB, aa, rad, k31_dG, path1) reaches ΔR² 0.18–0.21.
- Fitted radial coefficients vs physics: Q31 0.003–0.006 (≈ ½ v′_phys); **D4 v = 0.0028–0.0045 (≈ v_phys)**;
  D22 (labels only, net 0) ≈ 0.4× physical. The physics is right in the labels.
- D4: R²_S0 is NEGATIVE at l 2–6 (−0.6 to −0.06): a per-layer D4 scale/offset alone gives +0.25–0.86 there
  (the refit tuned D4 as a compensator). Radial D4 ΔR² 0.03–0.21 at every layer. Paid pieces ≤ 0.03.

## M5.4 hub move

- Step 0: Hkh2.5 +0.19 % vs Hd2.5 → pass. Hk2.5 (hubs dropped on always-on columns) +1.37 % → the always-on
  hubs are worth ~1.3 %, and they are nearly all of H2.5-literal's +1.59 %.
- Step 1 (`hubfid_report.md`, FIT 0–4, l 4–14): γ·C̃ alone R² ≈ 0 (hub born on off-diagonal C11, no diagonal);
  γ₁C̃ + γ₂Wᵀdiag(v)W: **age 1 R² 0.95–0.99**, age 2 0.52–0.80, age ≥ 3 ≤ 0.66 falling with age →
  **gate fails ("R² ≥ 0.8 at every age"); Step 2 arm not built.**
- Instead: H2.5 literal (no hub move) refit queued — −6.03 % F at +1.59 % unrefit nets ≈ −4.5 % score.

## M5.2 radial scale sweep (VAL-100 unrefit, v′ = scale × physical)

| slice | 0.1 | 0.3 | 0.5 | 1.0 |
|---|---:|---:|---:|---:|
| Q31 | +0.10 % | +0.50 % | +1.16 % | +3.91 % |
| D4 | +0.02 % | +1.59 % | +7.74 % | +25.3 % |
| D22 | +6.16 % | +66 % | +208 % | +1019 % |

Monotone in scale for every slice: the refit-tuned chain has absorbed whatever the radial terms do.
Refit queued: fold rec + radial (Q31, D4) at 0.1× physical with learned per-layer v′ (S2tsfr), compared
with the fold-rec refit (same start, same recipe). D22 excluded.
