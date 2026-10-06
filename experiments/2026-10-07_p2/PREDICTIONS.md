# Lane P2 predictions (written before each run)

## P2.0.3 — v77 on FRESH-A (written 2026-10-06 22:40 UTC)
- Strategist: ≈ 4.03e-9 (graded 3.9596e-9 × 1.019).
- Worker: 4.02–4.06e-9 floor-free; paired vs v73 ≈ −14 % score (graded −14.3 %).

## P2.1 — dressed legs (twin, VAL-100 free-running; written 2026-10-06 22:45 UTC)
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
