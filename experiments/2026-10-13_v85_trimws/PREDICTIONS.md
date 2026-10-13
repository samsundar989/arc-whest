# v85_trimws (B twin) — predictions on record, 2026-10-13 ~06:15 UTC, before any gate run

Build `tools/build_v85.py --params experiments/2026-10-11_p6/r8/age6/params_best.npz --atoms sb2+sbtrim --ws` →
`estimator_k3_v85_trimws.py` (`3208b2f3`). = v84_age6 (R8) at Strassen block 40 + Strassen v2 glue + op trim.
Real meter (DEV 3): **F/B 0.16943, 21,361 ops** (r7ws 0.16941 / 20,512). Projected residual worst 21,361 × 12.64 µs = 0.270 s.

Why now: age6's B twin failed the WSL settled-growth check on Oct 12 (+476 MB, VmPeak 6,809 MiB); sb2 cut v84's VmPeak 27 %.

| gate / quantity | prediction (worker) |
|---|---|
| WSL 8 GiB, 5 repeats | PASS; VmPeak 4.9–5.3 GB; settled growth < 150 MB (the Oct 12 failure was allocator steps at 6.8 GB) |
| FRESH-A raw vs r7ws (paired) | −5 to −6.5 % (age6 on A: −5.84 % FRESH-A, −6.71 % graded) |
| FRESH-A score ratio vs r7ws | 1.05–1.07× better |
| projected graded | 2.77–2.80e-9 (B today 2.9808e-9) |
| wall vs r7ws (A/B, DEV 3) | 0.85–1.0× (sb2 −22 % vs age6 +5.5 %) ⇒ ≤ 78 s noisy-implied, B gate ≤ 95 s |

Stop rule: WSL FAIL twice, or FRESH-A not better than r7ws ⇒ B stays r7ws.
