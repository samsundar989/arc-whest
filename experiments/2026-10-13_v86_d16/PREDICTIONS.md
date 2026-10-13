# v86_d16 — predictions on record, 2026-10-13 ~09:00 UTC, before FRESH-A

Build `tools/build_v85.py --params experiments/2026-10-11_p6/r8/age6/params_best.npz --atoms sb2+sbtrim+depth16` →
`estimator_k3_v86_d16.py` (`7453e9c4`, = lane-s 09fd0f5). Real meter: **F/B 0.163696, 23,891 ops** (v85 0.165329 / 24,576).
Lane S gates (lane-s `experiments/2026-10-13_s/S2_handover.md`): WSL ×5 PASS VmPeak 4,954 MB; wall 1.064× sb2 (DEV 3, quiet
queue) ⇒ v85 server median 67.7 s × 1.064 = 72.0 s, noisy-implied worst ×1.24 = 89.3 s (A ≤ 105 ✓); DEV-100 raw +0.044 % ± 0.063.
Residual worst 23,891 × 12.64 µs = 0.302 s (≤ 0.33 ✓).

**depth8 not taken for v86:** wall 1.330× sb2 ⇒ 67.7 × 1.33 × 1.24 = 112 s noisy-implied > 105 s (A gate). Revisit on the K stack.

| quantity | strategist (P7.2) | worker (P7) |
|---|---|---|
| FRESH-A raw vs v85 | +0.1 to +0.5 % | +0.0 to +0.1 % (DEV +0.044 %) |
| FRESH-A score ratio vs v85 | — | 1.008–1.010× better |
| graded | 2.57–2.64e-9 (assumed leaf ≥ 8) | **2.689e-9** (2.684–2.694e-9) |
| server wall mean / max | — | 72–74 s / 79–83 s |
| residual max | — | 0.29–0.31 s |
