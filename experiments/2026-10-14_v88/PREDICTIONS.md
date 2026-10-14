# v88_k5 — predictions on record, 2026-10-14 ~00:40 UTC, before gates

`--atoms "sb2+sbtrim+depth8:1.5e6/6/12+k0+k1:2.5_3.0+k2:3.0+k3+k5"` on R1 B params → `estimator_k3_v88_k5.py`.
Lane K real meter (DEV 0–19): F/B mean 0.14550 (CV 2.75 %), ops 23,392–23,769, 16 materializations/net; worst residual
0.2935 s + 0.9 ms read-backs at 12.35 µs/op (materialization clause ≤ 0.30 s ✓). DEV-100 raw −1.62 % vs v87; parity −2.25e-4.

| quantity | worker |
|---|---|
| FRESH-A raw vs v87 | −1.0 to −1.6 % |
| FRESH-A score ratio vs v87 | 1.003–1.010× better |
| graded | 2.40–2.41e-9 (v87 2.4169e-9) |
| wall | ≈ v87 (λ 1.5e6 depth 1.050× sb2 vs v87's 1.070×; mean F +0.9 %) ⇒ ≈ 90 s noisy-implied |
| WSL VmPeak (5 repeats) | 5.5–6.5 GB, PASS |

Stop rule: FRESH-A not better than v87 ⇒ do not submit; WSL FAIL ⇒ add sbfree and retest.
