# Lane M3 predictions (recorded 2026-10-08, before any run)

Ratios are final-layer MSE (floor-subtracted) / the same chain with nothing dropped (A = ∞).

## M8 — is κ4 a memory?

| item | strategist | worker |
|---|---|---|
| b+E, A = 5 dropped | ×2–5 [60 %] | ×3 (range 1.8–6) |
| b+E, A = 3 dropped | — (GO bar ≤ 1.3×, kill bar ≥ 3×) | ×6 — I expect NO-GO (≥ 3×) [70 %] |
| b+E, A = 1 dropped | — | ×20+ |
| a, A = 5 dropped | ×10–20 [70 %] | ×8 (range 4–20) |
| a vs b+E | core reduces old κ3's value but does not remove it | a degrades faster than b+E at every A |

Worker reasoning: in b+E the core is fed by K4.3/K4.4 from κ3 factors (old ones included), so it
summarises old κ3 only at trace level; the D21 consumers (covariance, heavy legs) read the
distinct-index content, which the r = 1 core cannot store. Our chain loses ×19 at "old tier
dropped" (A ≈ 5).

## M9 — how much is fourth order worth?

| item | strategist | worker |
|---|---|---|
| K4 SIMPLE / K3 AUG (unfactored), n = 256 | 0.3–0.6 [I] | 0.5 |
| same, extrapolated to n = 1024 | 0.05–0.2 [50 %] | 0.25 (4× headroom) |
| naive per-order extrapolation overstates | yes | yes |

## M10 — which fourth-order pieces carry it?

| item | strategist | worker |
|---|---|---|
| star + path ≥ 60 % of the K4 gain | 40 % | 45 % |
| D4-star-only shared-factor arm ≥ 20 % of the gap | 30 % | 20 % |
