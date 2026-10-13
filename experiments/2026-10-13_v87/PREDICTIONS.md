# v87_B — predictions on record, 2026-10-13 ~12:40 UTC, before any gate run

Build `tools/build_v85.py --params experiments/2026-10-13_v87/r1_B_params.npz --atoms
"sb2+sbtrim+depth8:1.2e6/6/12+k0+k1:2.5_3.0+k2:3.0+k3"` → `estimator_k3_v87_B.py` **`6e1d4862`** (= lane-k build).
Params: Lane K R1 (2 epochs from R8, matched ctl), stack B, step 224. Real meter: **F/B 0.144217, 24,371 ops**, 0 materializations.
VAL-100 (Lane K): B 1.741472e-8 vs ctl 1.711219e-8 (+1.77 %); vs R8 1.716023e-8 (+1.48 %). Stack A fallback −0.06 % vs ctl.

| quantity | strategist (P7.3) | worker (P7) | basis |
|---|---|---|---|
| FRESH-A raw vs v86 | — | +1.2 to +1.9 % | VAL +1.48 % vs R8; d12c6 rounding ≈ +0.03 % |
| FRESH-A score ratio vs v86 | — | 1.10–1.12× better | F/B 0.14422 / 0.16370 = 0.881 |
| graded | 2.25–2.40e-9 (F/B 0.136–0.142) | **2.40e-9** (2.38–2.42e-9) | 1.6425e-8 × 1.0148 × 0.14422 |
| server wall median / noisy-implied | — | 75–78 s / 93.6 s | 67.69 × A/B 1.070 (stack B, lane-s #2704) × 1.042 × 1.24 |
| residual max | — | 0.30–0.315 s | 24,371 × 12.2–12.7 µs |
| WSL VmPeak | — | 4.9–5.4 GB, PASS | — |

Stop rules: FRESH-A not better than v86 ⇒ fall back to stack A (r1_A_params, drop +k3); file↔arm parity > 1e-3 ⇒ stop;
graded > 2 % worse than projected ⇒ stop the line.

**13:23 UTC:** WSL ×5 (workerq 2720) **23/23 PASS, VmPeak 5,889 MB, settled +51 MB** (prediction 4.9–5.4 GB ✗ high but PASS;
d12c6's deeper buffers, fix = Lane S sbfree for v88).
