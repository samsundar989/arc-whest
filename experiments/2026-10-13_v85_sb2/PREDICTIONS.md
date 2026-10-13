# v85_sb2 — predictions on record (Lane P7, P7.1), written 2026-10-13 ~04:10 UTC, before submission

Build: `tools/build_v85.py --params experiments/2026-10-11_p6/r8/age6/params_best.npz --atoms sb2` → `estimator_k3_v85_sb2.py`
sha256 `d3359e48…8264` (= Lane S's lane-s 6d7e71e build). v84_age6 + Strassen variant B (pooled `out=` glue, fused last
level), same recursion depths. Bit-identical to v84 by construction (Lane S check 1; reproduced on main, below).

| quantity | strategist (handoff P7.1) | worker (P7) | basis |
|---|---|---|---|
| graded score | 2.713e-9 ± 0.2 % | **2.7150e-9** (2.7123–2.7177e-9) | raw identical to #335562 (1.642152e-8) × F/B 0.165329 |
| graded raw | = #335562 | = 1.642151991e-8 exactly | bit-identical predictions on the same 100 graded nets |
| server wall mean | 62–70 s | 60–66 s | v84 server mean ≈ 82 s × local A/B 0.753 |
| server wall max | — | 64–71 s | v84 max ≈ 86 s × 0.753–0.82 |
| residual max | ≤ 0.27 s | 0.22–0.27 s | 24,576 ops × 9–11 µs/op (v84: 0.240 s at 22,113 ops ⇒ 10.9 µs/op); pooled buffers may lower the per-op rate |

Stop rule: graded score > 2 % worse than 2.7150e-9 (> 2.769e-9), or raw ≠ #335562's raw, stops the line (P7 canary rule).
After grading: record server wall mean/max, residual mean/max and per-op residual, kernel seconds — the per-op residual
becomes the ops gate (projected residual ≤ 0.33 s, Samuel's decision 3).
