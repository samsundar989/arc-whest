# P6 predictions (recorded before each run)

## 2026-10-11 01:35 UTC — v82c_r2 (P6.0d)
R2 = v82b arm, 2 more epochs from R1 on the pod (cosine, lr 2.1e-5 peak). VAL-100 best 1.903403e-08 (step 224)
vs R1 best 1.912189e-08: **−0.46 %** (≥ 0.3 % bar → bake). Build: `build_star --all` on v80_hyg with R2 params,
then `build_v82 --items 0,1,4,5,7` (the same pipeline rebuilds v82c_r1 byte for byte, sha 66f3ee4a). Constants only.
- strategist: none given for R2 specifically; P6 handoff counts "two refit epochs −0.5 %" inside v83's budget.
- worker: FRESH-A raw −0.25 to −0.5 % vs v82c_r1 (R1's VAL→FRESH-A transfer was ~1:1, −0.85 % → −0.82 %),
  wins 75–95/100; same F/B 0.16508, ops 22,375; graded ≈ 3.03e-09.

## v82c_r1ws / v82c_r2ws (P6.0c, B slot)
- strategist: v82c_r1ws ≈ 3.12e-09.
- worker: r1ws ≈ 3.12–3.13e-09 (v82b_ws/v82b ratio 1.0233 applied to 3.0464e-09 → 3.117e-09); r2ws ≈ that × (1 − 0.004).
