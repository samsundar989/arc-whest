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
