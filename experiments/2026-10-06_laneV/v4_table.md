# V4 calibration table (tools/fresh/v4_table.py)

### FRESH_D (100 nets; mean label floor 7.473e-10 subtracted)

| candidate | score (floor-free) | ± se | raw MSE | raw floor-free | per-net CV | ÷ DEV projection | ÷ graded |
|---|---|---|---|---|---|---|---|
| v73_gb | 4.7059e-09 | 7.67e-11 | 2.2663e-08 | 2.1915e-08 | 16.3 % | 1.0210 | 1.0186 |
| v68_ws | 5.1745e-09 | 8.47e-11 | 2.3121e-08 | 2.2374e-08 | 16.4 % | 1.0186 | 1.0213 |

paired v73 − v68 (raw arithmetic, t95): score -9.06 % ± 0.43 pp, raw -2.05 % ± 0.33 pp, v73 wins 100/100; graded -8.81 %
