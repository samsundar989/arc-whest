# Lane M2 — written predictions (recorded 2026-10-06, before any M5/M6/M7 run)

## M5 — ARC-b ablation, n = 1024, 16 organizer DEV nets

Reference points (`m2_mini1024.json`): a (SIMPLE) 3.93e-8, b (AUGMENT) 1.79e-8, so b's gain
over SIMPLE is G = a − b ≈ 2.14e-8. "Loses x %" below means (arm − b)/(a − b).

Arms change one ingredient of factored AUGMENT each; everything else stays AUGMENT (the
raised term budget in the dense mean/cov/(2,1)/(3) slices, the r = 1 κ4 core, 211→111,
K211).

| arm | strategist | worker (this lane) |
|---|---|---|
| 1 no heavy legs (SIMPLE births) | loses 35–60 % | loses 30 % (80 % interval 15–50 %) |
| 2 dressed legs (one X̃ leg) | recovers ≥ 70 % of what (1) loses | recovers 85 % (spurious H⊗H term is ~2–3 % of the net correction) |
| 3a memoryless κ4 core | loses 20–40 % | loses 20 % |
| 3c memoryless + gain-mode core | recovers half of (3a)'s loss | recovers 40 % of (3a)'s loss |
| 3b gain-mode core replaces the whole core | — | loses 25 % (it also drops the one-layer Gaussian-born κ4) |
| 4 no 211→111 | < 10 % | loses 5 % |
| 5 no K211 (pK_111, pK_211 → κ4 core) | < 10 % | loses 8 % |

Validation arm: dressed legs minus the spurious Sym(H ⊗ w2 e_j ⊗ H) births must reproduce b
to float rounding (it is algebraically the same term set).
