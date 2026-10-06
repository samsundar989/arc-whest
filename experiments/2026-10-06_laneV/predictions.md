# Lane V — written predictions (recorded 2026-10-06, before any fresh label exists)

Basis: v73_gb DEV (mini, 100) raw 2.227946e-8, per-net MSE CV 13.6 %, adjusted
4.784e-9; graded public (50) 4.6199e-9, raw 2.1515e-8 → the mini nets are ~3.6 % harder
than the public 50. Both are samples of one distribution, so a fresh 100 should land
between them, weighted toward the larger sample.

| quantity | strategist | worker (this lane) |
|---|---|---|
| v73 FRESH-A score, floor-subtracted | within ±2 % of 4.62e-9 | **4.71e-9** (+2 % vs graded), 80 % interval 4.52–4.90e-9 |
| per-net score CV, v73 | ≈ 14 % | 13 % (11–16 %) |
| v73 − v68_ws paired score difference | reproduces graded −8.81 % within 0.3 pp | −8.8 % ± 0.6 pp (raw part −1.8 % ± 0.5 pp; the rest is the fixed F/B ratio 0.21473/0.23127) |
| N = 1e8 floor at layer 16 (mean over nets) | — | 7.7e-10 (≈ 3.6 % of raw: floor subtraction is NOT optional at N = 1e8) |
| MSE(ours N = 1e8 vs organizer N = 1e9) on 10 mini nets | ≈ 8.5e-10 | 8.4e-10 ± 10 % (sum of the two floors) |
| FRESH-D (if available) vs FRESH-A | — | same within 3 % |

Stop rule for V2 validation: if the 10-net mini MSE minus the predicted floor sum differs from
0 by more than 15 % of the sum, the labeller is wrong — do not label panels until explained.
