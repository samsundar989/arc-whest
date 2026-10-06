# Lane M — written predictions (recorded 2026-10-06, before any labelled M2 run)

Evidence available at the time: m0 enumeration (factored AUGMENT drops 4 of 91 terms:
T1 triangle, E3 κ3×edge top slice, E21a/E21b its (2,1,1) analogues) and ONE unlabelled
bench net (n = 256, depth 4): rms final-layer shift c − b = 5.9e-5 vs a − b = 3.1e-4,
i.e. the dropped terms move the output ~28× less (in squared size) than the augmentation
does. By family: E3 5.5e-5, E21b 4.2e-5, T1 1.7e-5, E21a 1.6e-5 (rms vs b).

| quantity | strategist | worker (this lane) |
|---|---|---|
| G_drop = (MSE_b − MSE_c)/MSE_b at n = 512 | 5–25 %, P(≥ 10 %) = 50 % | **3 %** (80 % interval 0.5–8 %), P(≥ 10 %) = 10 % |
| dominant family | κ3 × edge | κ3 × edge (E3 alone ≥ 50 % of G_drop; T1 < 25 %) |
| trend n = 256 → 512 | — | flat within ±2 pp (both Θ(n^-3) squared size) |
| G_aug = (MSE_a − MSE_b)/MSE_a | — | 30–60 % |
| d0 vs b | equal | equal to float32 rounding (bench: 3.6e-7 max abs) ✓ |

Decision rule (handoff M3): GO iff G_drop ≥ 10 % at n = 512 and the trend is flat or
growing. Worker expectation: CLOSE → M4.

Caveat recorded in advance: n = 1024 unfactored needs ~54 GiB (dense κ3 n³ temporaries,
0.85 GiB at n = 256 scaling n³), so it is out of reach on the 32 GiB card unless chunked.

## Addendum after the n = 256/512 read (2026-10-06, before the n = 384/768 runs)

Measured G_drop: n = 256 +26.6 % [16.5, 36.6]; n = 512 +15.6 % [7.8, 23.5] (8/8 nets each).
My 3 % prediction was wrong (too low by ~5x); the strategist's 5-25 % range and "κ3 × edge
dominates" were right (T1 ≈ 0; the E family as a whole carries all of it, E3 and E21a alone
HURT, E21b alone gives most). The trend is DOWN (×0.59 per width doubling), which fails the
"flat or growing" half of the GO rule, so a third and fourth width settle it.

Prediction (worker) if the decline is a power law n^-0.77: G_drop(384) ≈ 20 %, G_drop(768) ≈ 11 %,
extrapolated n = 1024 ≈ 9 %. If instead 256 was an outlier and the trend is flat:
G_drop(768) ≈ 15 %. I put 60 % on the power-law reading.

## Outcome of the trend runs (2026-10-06)

G_drop final layer: n = 256 26.6 %, 384 18.6 %, 512 15.6 %, 768 16.8 % (8/8 nets at every width;
all-layer 19.4 / 16.1 / 12.1 / 14.4 %). Flat from n = 384 on; my power-law reading (768 ≈ 11 %)
was WRONG, the flat reading (≈ 15 %) right. E21b alone carries ~80 % at every width (13.5 of
16.8 at n = 768); the full E family reproduces c to 0.2 pp; T1 ≈ 0.

**M3 by the handoff rule: GO** (≥ 10 % at n = 512, trend flat 384 → 768). Caveat for the
transfer: ARC factored AUGMENT at n = 768 sits at final MSE 4.57e-8, which scales (n^-3) to
~1.9e-8 at n = 1024 — the same level as v73 raw 2.15e-8, so the reference is a fair proxy
for our chain's diagram set, but v73's fitted gains/corrector may already absorb part of it.
The prototype on the v73 path is what decides.
