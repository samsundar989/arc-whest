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
