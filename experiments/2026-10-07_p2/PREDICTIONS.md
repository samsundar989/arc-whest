# Lane P2 predictions (written before each run)

## P2.0.3 — v77 on FRESH-A (written ~22:30 UTC while the eval, started 22:22, had produced no result)
- Strategist: ≈ 4.03e-9 (graded 3.9596e-9 × 1.019).
- Worker: 4.02–4.06e-9 floor-free; paired vs v73 ≈ −14 % score (graded −14.3 %).

## P2.1 — dressed legs (twin, VAL-100 free-running; written 2026-10-06 22:25 UTC)
Baseline: v62_gb + mask R2.0/Hd2.5/Dz2.5 (q50 K table), G1bg params (VAL 2.2699e-8 at init, job 2423).

| arm | strategist | worker |
|---|---|---|
| full set, unrefit | (not a verdict) | −1 % to +3 % raw; learned gains were fitted without the families |
| full set, after gentle refit | −4 to −12 % raw [60 % that it is ≥ 4 %] | −1 to −5 % raw [40 % that it is ≥ 4 %] |
| {C²} alone | can be worse than none | worse than full set; sign uncertain |
| {D21, D21ᵀ} alone | can be worse than none | worse than full set |
| {D22} alone | — | smallest effect (|Δ| < 1 %): D22 is the chain's additive dG form, already small |

Why mine is lower: the twin's learned gains (tt, t1, feed) already absorb a share of the
missing families' mean effect, the corrector targets the residual, and our κ4 (2,2) slice is
a crude additive template, so the F3 family is carried with the wrong shape.

Kill (handoff): full set after refit better than −2 % raw on VAL fails ⇒ record and stop.

## P2.2 — gain-mode κ4 (twin, VAL-100; written 2026-10-06 22:33 UTC, before any run)
| arm | strategist | worker |
|---|---|---|
| (a) D22 product template, β 0.7, unrefit | — | −0.5 % to +2 % raw (layers 1–4 obey the law weakly; β 0.7 is too big there) |
| (b) + Q31 template, keep / drop q31 modes, unrefit | — | within ±1 % of (a); drop slightly worse |
| (c) blend γ 0.35, unrefit | — | +0 to +2 % (adds κ4 on top of an already-fitted slice) |
| best arm after refit (β or γ per layer learned) | −2 to −8 % raw | −1 to −3 % raw [35 % that it is ≥ 2 %] |
Kill (handoff): best arm after refit better than −1 % fails ⇒ record and stop.

## P2.1 result → v79_dress on FRESH-A (written 2026-10-07 ~01:00 UTC, eval just launched)
Twin: dressed refit VAL 2.11466e-8 vs refit control 2.26911e-8 = **−6.81 % raw** (unrefit −1.58 %).
- Worker: FRESH-A paired raw −5 to −7 % vs v77 (VAL→fresh shrink ~1 pp), F/B +0.05 %; fresh
  score ≈ 3.75–3.83e-9 → projected public ≈ 3.68–3.76e-9; graded ≈ 3.70e-9 ± 2 %.
- Strategist (P2.3, both levers): graded 3.4–3.8e-9; no figure for dressing alone.

## P2.2 revised (written 2026-10-07 ~01:55 UTC, after the unrefit screen, before the refits)
Unrefit β 0.7 blew up (×7.7): the chain's D22 is 0.31 (L1) → 0.56 (L14) of the β=1 template, with
corr 0.86–0.97, so β 0.7 doubled κ4. Refits start from β_l = chain-matched (0.31 → 0.56), learned:
- Worker: a / b / c after refit vs the refit control (2.26911e-8): −0.3 to −1.5 % each [25 % that
  the best is ≥ 1 %]; the template's shape barely differs from the chain's at depth (corr 0.97).
- Strategist: best arm −2 to −8 %. Kill: < −1 %.

## P2.4.1 — always-on D21 rows (twin, VAL-100, v79-equivalent; written 2026-10-07 before the run)
- Handoff: −4 to −7 % FLOPs at ≤ +1 % raw.
- Worker: Da3.0 +0.2 to +0.8 % raw; Da2.5 +0.5 to +1.5 %; Da2.0 +1.5 to +4 % (the corrector
  delta is lost on those rows, as it was for dead rows before Dz). FLOP saving ≈ 30 % (D21
  readout) × always-on fraction (≈ 8–14 % of rows at τ 2.5 by depth) ≈ −2.5 to −4 %.

## P2.4.1 result (recorded) and P2.4.2 prediction (written before the run)
P2.4.1 always-on D21 rows: Da3.0 +34 %, Da2.5 +87 %, Da2.0 ×3.9; the corrector variants
(full-row diag +34 %, column proxy +38 %) prove the corrector is not the cause. Closed.
P2.4.2 covariance sandwich on the previous layer's R rows (alive block): worker +0.0 to +0.3 %
raw; FLOPs −1.5 to −2.5 % (covariance 9.2 %; compact K_R/n 0.72–1.0). Handoff −2 to −3 %.

## P2.4.2 result (recorded) and v80_hyg prediction (written before its FRESH-A read)
P2.4.2: alive-block covariance C=1 ×13, C=2 (keep dropped variances) ×3.2 — the dead rows'
covariance is not negligible at our accuracy. Closed.
v80_hyg = v79 + layer-0 W^T W (−0.32 %) + one extra Strassen level for products with
m·k·n ≥ 2^29 (P2.4.5): F −1.49 %, ops 25,411. Worker: raw within ±0.1 % of v79 (rounding only),
score −1.4 % ± 0.2 %, projected public ≈ 3.61e-9 on FRESH-A.
