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

## 2026-10-11 03:00 UTC — v82c_r2ws (P6.0c A/B slot; replaces v82c_r1ws, which r2 dominates)
- worker: FRESH-A raw ≈ v82c_r2's (block 40 changes float32 rounding only; v82bws/v82b raw ratio 1.0000);
  paired vs v82b_bundlews ≈ 1.012×; F/B 0.16895, ops 20,269; graded ≈ 3.10e-09 (3.1433e-09 × 0.9873).
- strategist: (for r1ws) ≈ 3.12e-09.

## 2026-10-11 ~07:30 UTC — v83_fold = v82c_r2 + fold (R3 fold arm baked)
R3 matched end (step 224): fold 1.851037e-08 vs ctl 1.892640e-08 = −2.20 % (ctl itself −0.56 % vs R2 init).
- strategist: (v83 with H, not applicable) fold+H joint −1.3 %, refit −0.5 %; nothing for fold alone.
- worker: FRESH-A raw −2.3 to −2.8 % vs v82c_r2 (fold −1.03 % unrefit + refit; R3 fold vs R2 init −2.75 % VAL),
  F +0.27 % → score ≈ −2.1 to −2.5 %, graded ≈ 2.96–2.97e-09.

## v83_od16 = v83_fold + per-shape Strassen depth leaf ≥ 16
- strategist: −0.71 % F at 25,167 ops (on fold+Habs); no gate change needed.
- worker: F −0.68 % (0.16441, 24,688 ops); DEV raw +0.03 % vs fold → FRESH-A score ≈ 1.006× vs v83_fold; wall
  1.133× v82c_r2 ≈ 101.9 s noisy-implied (thin vs 105 s).

## 2026-10-11 13:25 UTC — v83_r4od16 = v83_fold + R4 constants + od16 (FRESH-A paired vs graded v83_fold)
- worker: raw −0.3 to −0.45 % (R4 VAL −0.43 %, od16 +0.075 % FRESH-A), F −0.68 % → score ≈ −1.0 %, graded ≈ 2.92e-09.
- strategist: od16 −0.71 % F (no raw prediction); refit polish ≈ −0.5 %.

## 2026-10-11 14:45 UTC — v83_r4ws (B twin of v83_r4, block 40) for the Oct 12 A/B slot
- worker: raw ≈ v83_r4 ≈ −3.3 % vs v82c_r2ws's FRESH-A (fold −2.84 %, R4 −0.4 %), F +0.27 % → ≈ 1.033× vs v82c_r2ws;
  graded ≈ 3.00e-09. Wall A/B 1.0028× v82c_r2ws → ≈ 78 s noisy-implied.
