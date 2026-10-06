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
