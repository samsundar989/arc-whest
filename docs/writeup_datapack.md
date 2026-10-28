# WRITEUP DATAPACK — 2026-10-21 (numbers only; every number carries its source path and date)

Conventions: raw = final-layer MSE; F = billed FLOPs (flopscope 0.12.3); "served F" = per-MLP `flops_used` in the graded report (= own bill − 414,389…414,692 offset); negative Δ = better; VAL = nets 900–999 paired; FRESH-A100 = 100 own fresh nets, N = 1e8 labels. Graded-report stats (med/max) are recomputed from the per-MLP telemetry in `experiments/run_reports/*_graded_report.json` (read 2026-10-21). `RR:` = `experiments/run_reports/`, `DES:` = `docs/DESIGNATION.md` (2026-10-21), `L7:` = `experiments/2026-10-21_L7/RESULT.md`, `LED:` = `docs/WRITEUP_LEDGER_2026-10-21.md`, `A:` = `_claude_snap/handoff_2026-10-19/A_architecture.md` (2026-10-19).

## 1. Graded entries (all Phase 2, 1024×16, 100 MLPs, flop_budget 2,199,023,255,552; whestbench 0.16.1 / flopscope 0.12.3 / numpy 2.2.6 / py 3.10.21)

### 1.1 Score table (`RR:*_graded_report.json` `score`, `scoreSecondary`, `aggregates.public`)

| entry | AIcrowd # | created (UTC) | graded score (adjusted) | raw final-layer MSE | all-layers MSE | served F / MLP | multiplier (= mean_compute_utilization) | best / worst MLP adj | failed | smoke |
|---|---|---|---|---|---|---|---|---|---|---|
| **A5s** (nominated A) | **337601** | 2026-10-21 19:02:06 | **1.8751280753233397e-9** | 1.4035382882582325e-8 | 5.541e-9 | 293,789,651,431 | 0.13360006570610494 | 1.525e-9 / 2.401e-9 | 0/100 | PASS 17.2 s, 5 passes |
| **B4** (nominated B) | **337562** | 2026-10-21 16:33:39 | **1.9724068470701697e-9** | 1.4069473870392813e-8 | 5.566e-9 | 308,282,211,977 | 0.14019051922196013 | 1.610e-9 / 2.552e-9 | 0/100 | PASS 14.7 s, 8 passes |
| A4s (fallback A) | 337511 | 2026-10-21 11:21:25 | 1.8800496320794433e-9 | 1.4072220864136398e-8 | 5.565e-9 | 293,789,651,431 | 0.13360006570610494 | 1.535e-9 / 2.428e-9 | 0/100 | PASS 17.5 s |
| A3 (not nominated, residual cliff) | 337495 | 2026-10-21 08:45:21 | 1.8527052525413123e-9 | 1.416773756091061e-8 | 5.616e-9 | 287,564,751,853 | 0.13076930911347517 | 1.507e-9 / 2.367e-9 | 0/100 | PASS 24.6 s |
| A1 (not nominated) | 337456 | 2026-10-21 03:24:18 | 1.8656192469864824e-9 | 1.4338384595191656e-8 | 5.702e-9 | 286,122,895,009 | 0.13011362853330866 | 1.535e-9 / 2.375e-9 | 0/100 | PASS 24.3 s |
| incumbent 8be | 336993 | 2026-10-18 21:16:35 | 1.8962023807611765e-9 | 1.4535553773953325e-8 | 5.800e-9 | 286,868,542,979 | 0.13045270997236003 | 1.569e-9 / 2.421e-9 | 0/100 | PASS 22.2 s |
| A1s (fallback A before A4s) | 337481 | 2026-10-21 06:13:35 | 1.897391135725812e-9 | 1.4341183334209973e-8 | 5.702e-9 | 290,938,839,223 | 0.13230366640664215 | 1.565e-9 / 2.411e-9 | 0/100 | PASS 17.4 s |
| B-young (fallback B) | 337462 | 2026-10-21 03:55:57 | 2.0005580379785525e-9 | 1.4337337237435577e-8 | 5.701e-9 | 306,840,355,133 | 0.1395348386417936 | 1.645e-9 / 2.547e-9 | 0/100 | PASS 14.5 s |
| A1 λ 1e6 twin | 337470 | 2026-10-21 05:14:20 | 1.8708071209909215e-9 | 1.4340883538466187e-8 | 5.702e-9 | 286,868,542,979 | 0.13045270997236003 | 1.540e-9 / 2.383e-9 | 0/100 | PASS 22.6 s |

sampling_mse (label floor of the public panel) = 1.132e-6 in every report. failure_breakdown all zero (budget / time / residual_wall / combined / error) for every row.

### 1.2 Own bill, ops, timing, forecast, FRESH (`DES:`, `L7:` §5, `experiments/2026-10-21_L6/RESULT.md` §3, `experiments/2026-10-21_L2/RESULT.md`, `experiments/2026-10-20_L2/RESULT.md`, `experiments/2026-10-21_L3b/RESULT.md` §A; per-MLP med/max recomputed from `RR:`)

| entry | own 0.12.3 bill F / ops | served F (= bill − offset) | residual med / max (s) | wall med / max (s) | µs/op | forecast (written first) → graded | FRESH-A100 | VAL-100 |
|---|---|---|---|---|---|---|---|---|
| A5s #337601 | 293,790,066,123 / 20,991 (verify #3689, +0) | 293,789,651,431 (−414,692) | 0.2913 / 0.3017 | 71.26 / 78.51 | 13.88 | 1.8715e-9 → 1.8751e-9 (**+0.19 %**, window ±1.5 % = 1.8434–1.8996e-9) | −0.531 % vs A4s [−0.665, −0.397], 77/100 (#3688; raw 1.505352e-8 vs 1.513606e-8) | −0.454 % vs A4 [−0.599, −0.309], 74/100 |
| B4 #337562 | 308,282,626,669 / 19,684 (verify #3680) | 308,282,211,977 | 0.2674 / 0.2819 | 54.85 / 59.60 | 13.59 | 1.9728e-9 → 1.9724e-9 (**−0.02 %**, keep band 1.943–2.002e-9) | not run (VAL22 parity vs A4s +0.040 % [−0.026, +0.106], 8/22, max|Δpred| 1.03e-5) | — |
| A4s #337511 | 293,790,066,123 / 20,991 (verify #3669) | 293,789,651,431 | 0.2900 / 0.3115 | 71.39 / 80.23 | 13.82 | 1.8819e-9 → 1.8800e-9 (**−0.10 %**) | −0.583 % vs A3 [−0.787, −0.379], 72/100 (#3668) | −0.600 % vs A3 [−0.793, −0.407], 73/100 |
| A3 #337495 | 287,565,166,545 / 23,374 (verify #3665; +0.504 % F vs A1) | 287,564,751,853 | 0.3331 / 0.3447 | 78.41 / 85.91 | 14.25 | 1.8447e-9 → 1.8527e-9 (**+0.43 %**) | −1.625 % vs A1 [−1.856, −1.394], 89/100 (#3661) | −1.418 % vs A1 [−1.665, −1.171], 85/100 |
| A1 #337456 | 286,123,309,398 / 22,633 | 286,122,895,009 (−414,389) | 0.3261 / 0.3349 | 77.98 / 83.21 | 14.41 | 1.863e-9 (interval 1.76–1.88) → 1.8656e-9 (**+0.14 %**) | −1.480 % vs 8be, t95 Δ [1.89e-11, 2.76e-11], 87/100 | −1.361 % vs 8be [−1.588, −1.134], 89/100 |
| 8be #336993 | 286,868,957,368 / 22,235 | 286,868,542,979 (−414,389) | 0.3081 / 0.3389 | 73.10 / 81.41 | 15.2 (0.3389/22,235, max) | 1.89564e-9 FRESH-anchored → 1.89620e-9 (**+0.030 %**; `RESULTS.md:95-97`) | FRESH100 score gain 7.646 %, raw 1.783 % vs prior best (`RESULTS.md:7479`) | canonical VAL raw −1.518 % vs b636, 90/100 (`LED:` row 20c) |
| A1s #337481 | 289.5…290.9 G / 20,630 (+1.68 % F vs A1) | 290,938,839,223 | 0.2905 / 0.3023 | 70.22 / 73.99 | 14.08 | 1.8970e-9 → 1.8974e-9 (+0.02 %) | −0.009 % vs A1 t95 [−6.69e-13, +3.86e-13], 50/100 (#3657) | max|Δ| vs A1 7e-6 |
| B-young #337462 | 306,840,769,522 / 18,943 (+7.24 % F vs A1) | 306,840,355,133 | 0.2594 / 0.2760 | 53.49 / 60.63 | 13.69 | 2.0007e-9 → 2.0006e-9 (−0.007 %) | −0.008 % vs A1 [−0.051, +0.035], 49/100 | −0.010 % vs A1 [−0.048, +0.028], 47/100 |
| λ 1e6 twin #337470 | 286,868,957,368 / 22,235 | 286,868,542,979 | 0.3171 / 0.3256 | 75.91 / 79.20 | 14.26 | 1.8706e-9 → 1.8708e-9 (+0.011 %) | — | — |

Ratios (`DES:`, `L7:` §5): A5s = A4s × 0.99738 (−0.262 % score = raw, same served F); A5s = B4 × 0.9507 (B4 scores 4.9 % worse, 1.9724/1.8751 = 1.0519); B4 = B-young × 0.98593; A4s = A1s × 0.9909 = A1 × 1.0077; A3 = A1 × 0.9931; A1 = 8be × 0.9839 (−1.61 %); A1s = A1 × 1.0170; B-young = A1 × 1.07233; twin = A1 × 1.00278.
Packages / sha (`DES:`): A5s `a5s-soupbc368-lam35-180e488b.tar.gz` 2ee01ae9 (estimator 180e488b); B4 `b4-a4s368-4af12b5e.tar.gz` 882541f0 (estimator 4af12b5e); A4s `a4s-s368-lam35-7b0d6800.tar.gz` c0fdf5f2; A3 `a3-s368-lam06-5ddf3290.tar.gz` 924130df; A1 `a1-r3s736-lam06-7d381fc5.tar.gz` 4c5ca48e; 8be `estimator_8be.py` 8be40031 (package bf5805db); A1s e823d62f / 14f94773; B-young 7a146f84 / bfba72f7.
Gates A5s (`L7:` §5, 2026-10-21): 100-fresh-seed sweep 100/100, 293,789,618,663 F every call, 21,634 dispatches, VmPeak 3,651 MiB (`bake/summary.json`: residual p50/p90/max 0.276/0.326/0.385 s pod-local, wall p50/max 43.6/58.4 s); verify #3689 PASS (served-eq residual 0.3200 s, wall 81.9 s); parity VAL 900/901 max|Δ| 6.9e-6 / 6.0e-6.
Gates B4 (`experiments/2026-10-21_L3b/RESULT.md` §A3): native-Linux 100-call memory VmPeak 3,848.2 MB, settled call 10→99 −30.2 MiB, block medians 2170/2163/2159/2141 MiB (B control 3,853.7 MB, +1.0 MiB); client/server RLIMIT rehearsal 100/100, server VmPeak 2,740 MiB; one FLOP count 308,282,179,209 per call.

## 2. Architecture constants of A5s (`experiments/2026-10-21_L7/src/cand_a5s_soupbc368_180e488b.py`, sha256 180e488bfb2e6587…, read 2026-10-21; ≡ `src/cand_a5s_soupbc_180e488b.py`, ≡ `release_a5s_soupbc368/estimator.py`)

| item | value | line |
|---|---|---|
| source length | **4,264 lines** (A1 7d381fc5: 3,863 lines, `LED:` header) | `wc -l` |
| `_SHARED_AGE` | 6 | 96 |
| `_SHARED_RANK` | 288 (comment: b636 256) | 97 |
| `_JOIN_SKIP_TAIL` / `_SHARED_PASSES` / `_RIDE` | 3 / 1 / True | 95, 99, 98 |
| `_P2_RANK` / `_P2_SEPARABLE` | 4 / True | 134, 82 |
| `_HUB_ROUTES` / `_HUB_COMPENSATE` | ('i','ii') / True | 137, 139 |
| `_CLOSURE_SCOPE` / `_CLOSURE_NBIV` / `_OWN_THETA_FAM` | 'all' / 27 / 0.625 | 142, 204, 153 |
| `_OWN_S_LAM` / `_OWN_T_LAM` / `_K4_PERNEURON` | 1.225 / −0.1 / True | 154, 155, 158 |
| `_P3_ON` / `_SVD_JOIN` | False / False (K9: no svd(k=)) | 148, 113 |
| `_SW_LEAF` | 8 | 1932 |
| `_SW_CAP` | 6 | 1933 |
| `_SW_OP_FLOPS` | **3,500,000** (A4s/A5s bake; A1 600,000; 8be 1,000,000; b636 4,200,000) | 1934 |
| `_SW_WALL_FLOPS` | 400,000,000 | 1935 |
| `_SW_PAD` / `_SW_PAD_KMAX` / `_SW_TRIM` / `_SW_KS` / `_SW_KS_TRIM` / `_SW_KS_LTRIM` | True / 0 / True / True / True / 2 | 1986, 1987, 2044, 2200, 2201, 1937 |
| `_TW_LEAF` / `_TW_NLEAF` / `_TW_GLUE` / `_TW_OP` (wall model) | 7.56e-11 / 3.26e-6 / 8.89e-10 / 0.0004 | 1938–1941 |
| `_STRASSEN_DEPTH` / `_STRASSEN_MIN_BLOCK` / `_STRASSEN_BIG_VOL` / `_STRASSEN_BIG_BLOCK` | 5 / 20 / 1,073,741,824 / 16 | 1836, 1921–1923 |
| `_KAS_ON` / `_KAS_SHAPE` | True / (1024, 16) | 2710, 2711 |
| `_KAS_K["R"]` (transport rows, R2.0) | 1024 1024 992 960 928 896 896 864 832 832 800 768 768 768 736 736 | 2728 |
| `_KAS_K["D"]` (D21 columns, Dr2.5) | 1024 1024 1024 992 992 960 928 928 896 864 864 832 832 800 800 800 | 2729 |
| `_KAS_K["H"]` (hub/birth columns, Hd2.5) | 1024 1024 1024 992 992 960 928 928 896 864 864 832 832 800 800 800 | 2730 |
| `_K1_ROWS` (D21 rows, smallest α²) | 1024 1024 1024 992 928 896 832 800 768 736 704 640 608 576 576 544 | 2720 |
| `_K3_A` | = `_K1_ROWS` | 2727 |
| `_K2_O` (transported leg output rows) | 1024 1024 1024 1024 992 992 960 960 928 928 896 896 864 864 832 **704** | 2723 |
| `_CL_K` (live covariance-sandwich rows) | 1024 ×5, 992 ×3, 960 ×2, 928 ×2, 896 ×2, 864 (15 entries) | 2714 |
| `_D21E_RHO` | 1.0 1.0 1.0 1.11982 1.12823 1.235294 1.331374 1.341067 1.431506 1.518142 1.516327 1.601306 1.604 1.663418 1.678725 1.0 | 2716 |
| `_CORR_ON` / `_CORR_NF` | True / 9 raw channels | 3001, 3042 |
| `_CORR_PI` / `_CORR_PJ` (15 pairwise products of channels 0–5) | i<j over 6 channels → 15 | 3046–3047 |
| corrector shape per layer (`_CORR_B64`, 16 layers) | fmu (16,33), fisd (16,33), **W1 (16,33,32), b1 (16,32), W2 (16,32,32), b2 (16,32), W3 (16,32,1), b3 (16,1)** = 33→32→32→1 tanh MLP; **35,888 float32 total** = 34,832 MLP params (W1 16,896 + b1 512 + W2 16,384 + b2 512 + W3 512 + b3 16) + 1,056 fmu/fisd normalisers (`A:` KNOBS: "34.8 k weights") | 3051 |
| `_ELEM_PER_LAYER` / `_BUDGET_FRACTION` | 400 / 0.95 | 2848, 2854 |
| fitted tables (lines differing A5s vs A4 export `src/a4_export_s368.py` 67ed6090) | **35 fitted tables + 1 λ line** (36 lines): `_LAM_GAIN _THETA_GAIN _FEED_GAIN _TT_GAIN _T1_GAIN _CU_GAIN _CB_GAIN _TMET_GAIN _MARG2_GAIN _MARG2_C2_GAIN _CL_C3_GAIN _CL_CQ_GAIN _YP_D4_GAIN _YP_Q31_GAIN _YP_TP_GAIN _YP_TPQ_GAIN _K4T1_CV _D4_LAW_SCALE _K0_VG _K2_VG4 _HUB_A _CORR_B64 _DRESS_DG _K4T1_B1 _K4T1_B2 _K4T1_C0 _K4T1_CD _K4T1_CM _K4M_DS _K4T15_C31 _K4T15_C211 _K4T15_CS _K4T2_C31 _K4T2_C211 _K4T2_CS` (+ `_SW_OP_FLOPS`) | diff |
| code lines differing A5s vs A4s 7b0d6800 | **0** (`L7:` §4) | — |
| B4 vs A4s | 3 planner lines: `_SW_CAP` 6→5, `_SW_PAD` True→False, `_SW_OP_FLOPS` 3.5e6→5e6 (`DES:` B4 row) | — |

## 3. Cost anatomy

### 3.1 Bill by family — 8be #336993, 286,868,957,368 F / 22,235 ops (`A:` §3 census table, from `experiments/2026-10-19_research_lead/cost/analysis_3477/REPORT.txt`, 2026-10-19; A5s = same code + planner λ 3.5e6 ⇒ 293.79 GF / 20,991 ops)

| family | GF | % of 286.87 | ops | b636 GF (305.09) |
|---|---:|---:|---:|---:|
| κ3 transport `Wᵀ[O,R] @ [A_L|M|Z|Y|Q]`, 15 layers | 104.25 | 36.3 | 1,847 | 104.98 |
| κ3 young read-out, Dx pullback (layers 3–14) | ≈ 68.9 (bucket 71.20 incl. glue) | 24.0 | 1,506 | 72.3 |
| κ3 young read-out, newborn product + layer-1 contract | 20.6 + 2.1 | 7.9 | 1,313 | 22.7 |
| **young κ3 window (rows 1–3)** | **193.7** | **67.5** | | |
| exact K2 (covariance) chain `WᵀCW` + layer-0 `WᵀW` + last-layer diag | 29.19 | 10.2 | 3,705 | 31.55 |
| old tier: dense rebuild `Q @ [FA|FM]`, layers 7–15 | 22.10 | 7.7 | 958 | 19.77 |
| old tier: `Xold` read, layers 7–14 | 13.88 | 4.8 | 808 | 12.60 |
| old tier: join (riding band) | 7.35 | 2.6 | 1,140 | 20.59 |
| **old tier (rows 5–7)** | **43.3** | **15.1** | | |
| closure (27-monomial einsum 3.8, stacks, univariate extras, Q31 law) | 5.02 | 1.7 | 1,347 | 5.21 |
| κ3 read-out elementwise (P2 recon 2.55, hub parts 1.12, LA/LP + fold 1.61, D3 0.61) | 5.89 | 2.1 | ≈ 1,630 | 5.95 |
| `_layer` inline remainder (pK terms, S/K assembly, C11, D22, dG, **D4 laws (κ4)**, fills) | 2.77 | 1.0 | 4,482 | 3.0 |
| birth (P2 rank-4 range finder 0.54, gate/dress/append 0.31, Wick + KAS gathers 0.56) | 1.41 | 0.5 | 963 | 1.4 |
| corrector (features + 3 matmuls + hub features) | 0.34 | 0.12 | 801 | 0.34 |
| Wick 1-D coefficients, constants, init, harness | 0.06 | 0.02 | 636 | 0.06 |
| last layer alone (transport 7.28 + old rebuild 2.71 + var 0.84 + D3/D4 0.7) | 11.45 | 4.0 | 708 | 13.05 |
| matmul raw share | 241.996 GF = 84.36 % in 571 calls | | | |
Ops by name (8be, proxy): add 6,416 · multiply 5,990 · subtract 5,118 · reshape 1,141 · take 598 · matmul 594 · concatenate 425 · stack 397 · einsum 346; ≈ 11 k Strassen glue, ≈ 4.5 k `_layer` body, 1.9 k closure (`A:` §3). Exact K2 chain = 13.3 % of the 219.9 GF floor; billed at 0.58× naive at depth 6 (`A:` §3).

### 3.2 X2 ablations on 8be, VAL-100 paired, unrefit, corrector on (`experiments/2026-10-20_x2/RESULT.md:13-22,37-47`, 2026-10-20)

| item | F | ΔF | ops (Δ) | Δraw [95 % CI] | wins | kill line | verdict |
|---|---:|---:|---:|---|---:|---|---|
| control 8be40031 | 286,868,957,368 | — | 22,235 | GPU vs CPU control −0.004 % [−0.041, +0.035] | 50 | | |
| A `_K2_O` → max(R,D,H) (`vA_omax`) | 282,504,610,375 | −1.521 % | 22,278 (+43) | **+6.149 %** [+5.14, +6.82] | 4 | ≤ +0.608 % | KILL |
| B1 no old rebuild at L15 (`vB1_l15noold`) | 284,153,403,562 | −0.946 % | 22,134 (−101) | **+107.1 %** [+102.7, +111.4] | 0 | ≤ +0.378 % | KILL |
| B2 O[15] 704 → 640 (`vB2_o640`) | 285,948,268,658 | −0.321 % | 22,235 (0) | +0.153 % [+0.084, +0.231] | 36 | ≤ +0.128 % | KILL (marginal) |
| C no Xold at L7–9 (`vC_noxold79`) | 283,237,526,017 | −1.266 % | 21,924 (−311) | **+60.87 %** [+57.4, +63.3] | 0 | ≤ +0.506 % | KILL |
| D λ_op 1e6 → 0.6e6 (`vD_lam06`) | 286,123,309,398 | −0.260 % | 22,633 (+398) | +0.006 % [−0.029, +0.038] | 50 | residual ≤ 0.37 s (0.345 s proj.) | **KEEP** |
| E ⟨4,4,4;48⟩ | — | not run | | | | | |
| F K1 −128 rows at L2–5 (`vF_k1m128`) | 283,261,914,327 | −1.257 % | 22,285 (+50) | **+5.678 %** [+4.82, +6.65] | 9 | ≤ +0.503 % | KILL |
| all six composed (`compose.py` 43d789b6) | 271,471,929,471 | −5.367 % | 22,274 (+39) | +183.2 % [+175.9, +189.2] | 0 | | |
Elasticity Δraw/|ΔF|: A 4.0, F 4.5 (unrefit, 8be); cut-A with refit 3.7 (§3.6).

### 3.3 Rank 320 (`_SHARED_RANK` 288 → 320)

| measurement | Δraw | ΔF (real bill) | score-equiv | source | date |
|---|---|---|---|---|---|
| ALL-R vs ALL, L3 gain-group screen, step 0 / 112 (8be) | −1.182 % [−1.440, −0.924], 86/100 / −1.175 % [−1.507, −0.843], 77/100 | +1.86 % (292,214,932,424 F) | +0.65 % | `experiments/2026-10-20_L3/RESULT.md:64,68,86`; `experiments/2026-10-20_L2/RESULT.md:147-148` | 2026-10-20 |
| J step 0 (A1 export + NEW10 + U2b, gains 0, r320) vs A1 | −1.195 % [−1.449, −0.940], 85/100 | +1.864 % (292,214,932,424 vs 286,868,957,368 F; ops 22,235 both) | +0.67 % | `experiments/2026-10-21_L4/RESULT.md:57,104,137` | 2026-10-21 |
| A6 vs A4 (unrefit, historic) | −1.177 %, A6 wins 28/100; A4−A6 score CI [−1.87e-11, −8.31e-12] | +1.864 % | +0.66 % worse | `RESULTS.md:6412-6417` (`LED:` N1) | 2026-10-18 |
| J3 (not run): r320 + fresh epoch on A4 | predicted ≈ −1.7 % raw ⇒ score-equiv ≈ +0.15 %; vs A5s +0.67 % | +1.86 % | NO-GO | `experiments/2026-10-21_L4b/RESULT.md` | 2026-10-21 |

### 3.4 J — joint content arm (A1 export + NEW10 + U2b nested-Dx ages 2–3 + r320, 368 updates, pod) — `experiments/2026-10-21_L4/RESULT.md` §4, §6, §7 (2026-10-21)

| step | mean raw | ratio | paired Δraw vs A1 (1.513612e-8) | wins/100 | 95 % CI | score-equiv (Δraw + 6.779 % F) |
|---|---|---|---|---|---|---|
| 0 | 1.495211e-08 | 0.98784 | −1.195 % | 85 | [−1.449, −0.940] | +5.584 % |
| 92 | 1.506197e-08 | 0.99510 | −0.457 % | 56 | [−1.107, +0.193] | +6.322 % |
| 184 | 1.479854e-08 | 0.97770 | −2.186 % | 79 | [−2.731, −1.641] | +4.593 % |
| 276 | 1.462410e-08 | 0.96617 | −3.366 % | 96 | [−3.798, −2.934] | +3.413 % |
| **368** | **1.454358e-08** | **0.96085** | **−3.901 %** | **99** | **[−4.255, −3.547]** | **+2.878 % (NO-GO)** |
Bills: J start (r320, gains 0) 292,214,932,424 F / 22,235 ops (+1.864 %); J live 306,317,429,884 F / 24,580 ops (**+6.779 %**), residual proj. 0.375 s (×15.24 µs) / 0.364 s (×14.80); + λ 6e5 ≈ 24,978 ops ≈ 0.381 s. Price split: r320 +1.864 %, NEW10 ≈ +0.50 %, U2b ≈ +4.4 %. Additivity: parts r320 −1.195 + NEW10 −1.418 + U2b −1.24 = ≈ −3.85 % vs joint −3.90 %.

### 3.5 Tier reallocation on A4's export (constants only; `experiments/2026-10-21_L4c/RESULT.md` §4, §5, §7, §8, 2026-10-21; bills pod flopscope 0.12.3, nets 900/901 identical; VAL unrefit paired vs A4 recorded VAL 1.483245e-8)

| variant (E = `_SHARED_AGE`, r = `_SHARED_RANK`, τ_hub) | F at λ 3.5e6 | ΔF vs A4s | ops | resid. proj. (13.82 µs/op) | F export (λ 1e6) | Δraw unrefit | wins | 95 % CI | score-equiv |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| (6, 288) = A4s / A4 export | 293,790,066,123 | — | 20,991 | 0.290 s | 288,310,814,515 / 22,976 ops | — | | | |
| (5, 384) | 300,270,222,859 | +2.206 % | 21,453 | 0.296 | 294,055,141,044 | −1.073 % | 74 | [−1.485, −0.661] | **+1.133 %** |
| (4, 512) | 316,405,198,743 | +7.698 % | 22,497 | 0.311 | 310,010,463,138 | −1.889 % | 83 | [−2.318, −1.461] | **+5.809 %** |
| (3, 576) | 325,950,932,824 | +10.947 % | 23,014 | 0.318 | 320,697,865,212 | −0.837 % | 70 | [−1.326, −0.348] | **+10.110 %** |
| (6, 288, τ_hub 3.0) | 296,810,783,076 | +1.028 % | 20,753 | | 291,307,887,467 / 22,738 | +0.011 % | 48 | [−0.153, +0.174] | +1.039 % |
| (4, 512, τ_hub 3.0) | 318,986,730,446 | +8.576 % | 22,274 | | 313,232,318,487 / 23,948 | −1.889 % | 79 | [−2.355, −1.423] | +6.687 % |
Prediction was −6…−20 % F and +1…+4 % raw; keep ≤ −2 % score-equiv: none. τ_hub 3.0 H list `[1024,1024,1024,1024,992,992,960,960,928,928,896,896,864,864,832,832]`.

### 3.6 Cut-and-refit (X2 variant A on A4's export, 368 updates on fresh nets; `experiments/2026-10-21_L3b/RESULT.md` §B2, §B4, §B5, 2026-10-21)

| source | λ | F | ops | ΔF vs A4 same λ |
|---|---|---:|---:|---|
| A4 export 67ed6090 | 1e6 | 288,310,814,515 | 22,976 | — |
| cut 76ee5c86 | 1e6 | 283,946,467,522 | 23,019 (+43) | −1.514 % |
| A4s 7b0d6800 | 3.5e6 | 293,790,066,123 | 20,991 | — |
| cut + λ 3.5e6 fee86bf6 | 3.5e6 | 289,126,365,310 | 21,054 (+63) | **−1.587 %** |

| step | mean raw | Δraw vs A4 | wins/100 | 95 % CI | score-equiv [CI] | export |
|---|---|---|---|---|---|---|
| 0 (cut, unrefit) | 1.575572e-08 | +6.145 % | 5 | [+5.271, +7.019] | +4.460 % [+3.600, +5.320] | 76ee5c86 |
| 92 | 1.626087e-08 | +9.588 % | 5 | [+8.500, +10.675] | +7.848 % | f1923093 |
| 184 | 1.593010e-08 | +7.341 % | 5 | [+6.308, +8.375] | +5.637 % | de292cfb |
| 276 | 1.577493e-08 | +6.292 % | 6 | [+5.389, +7.194] | +4.604 % | f01248e3 |
| **368** | **1.571107e-08** | **+5.832 %** | **6** | **[+4.948, +6.716]** | **+4.152 % [+3.282, +5.022]** | 518e3e9e |
Row elasticity with refit Δraw/|ΔF| = 5.83/1.59 ≈ **3.7** (unrefit 3.9; X2 on 8be 4.0). Pod spend ≈ $1.05.

### 3.7 D21 column question (`experiments/2026-10-21_L4d/RESULT.md`, 2026-10-21): **columns are not full** — every D21 block is formed K1 × D (rows `_K1_ROWS` smallest α², columns `_KAS_K["D"]` top-α, Dr2.5 one-sided); full-n columns only at L1–L3 where D = K1 = 1024. No variant built, $0. Column neuron enters the closure at first order with Φ(α_j) (≈ 1 for decided-on), so a two-sided kink-set column cut is a first-order error, not the KAS floor.

### 3.8 Planner frontier (`experiments/2026-10-21_L1/planner_sweep.json`, RESULT §5; `experiments/2026-10-20_L3/RESULT.md:121-132`, 2026-10-20/21)

| build (cap, pad, leaf, λ_op) | F | ΔF | ops | resid. proj. (s) |
|---|---:|---:|---:|---:|
| A1 chain: 6e5 (A1) / 2e6 / **3e6 (A1s)** / 5e6 / 8e6 | — / — / 289.5…290.9 G / — / — | 0 / +1.17 % / **+1.68 %** / +3.81 % / +9.00 % | 22,633 / 20,952 / **20,630** / 19,328 / 17,293 | |
| 8be chain (6, pad, 8, 1e6) | 286,868,957,368 | — | 22,235 | 0.339 |
| (5, no pad, 8, 1e6 / 2e6 / 3e6 / 4e6) | 297,744,578,986 / 300,103,663,483 / 301,082,756,245 / 303,256,906,679 | +3.79 / +4.61 / +4.96 / +5.71 % | 21,773 / 20,509 / 20,225 / 19,671 | 0.332 / 0.313 / 0.308 / 0.300 |
| **(5, no pad, 8, 5e6) = B-young** | **306,840,769,522** | **+6.96 %** | **18,943** | **0.289** |
| (5, no pad, 8, 6e6 / 8e6) | 313,954,921,534 / 319,279,604,852 | +9.44 / +11.30 % | 17,741 / 17,045 | 0.270 / 0.260 |
| (5, no pad, 40, 1e6 / 4e6) "block-40" | 343.1 G / 344.5 G | +19.6 / +20.1 % | 18,525 / 17,645 | 0.282 / 0.269 |
≥ 10 % central residual headroom needs ≤ 20,889 ops at A1's per-op rate (≤ 21,719 at 8be's) (`experiments/2026-10-21_L1/RESULT.md` §2). λ does not change per-op rate: twin 14.26 vs A1 14.41 µs/op (§4).

## 4. Accuracy ledger (the A chain: 8be → R3 = A1 → A3 → A4 → A5; all VAL-100 paired, final layer, raw)

### 4.1 R3 refit of 8be (fresh Adam, corrector LR ×10, 736 updates, TRAIN900 + d8b2048; `experiments/2026-10-20_L2/RESULT.md:46-66,103`, 2026-10-20/21)

| step | mean raw | ratio | paired Δraw vs 8be (VAL0 1.534575e-8) | wins/100 | 95 % CI |
|---|---|---|---|---|---|
| 0 | 1.534575e-08 | 1.00000 | 0 | — | — |
| 92 | 1.546030e-08 | 1.00746 | +0.751 % | 32 | [+0.431, +1.070] |
| 184 | 1.537656e-08 | 1.00201 | +0.226 % | 45 | [−0.063, +0.515] |
| 276 | 1.534552e-08 | 0.99999 | +0.032 % | 48 | [−0.279, +0.343] |
| 368 | 1.525961e-08 | 0.99439 | −0.552 % | 66 | [−0.956, −0.147] |
| 460 | 1.538739e-08 | 1.00271 | +0.319 % | 46 | [−0.250, +0.888] |
| 552 | 1.514229e-08 | 0.98674 | −1.327 % | 87 | [−1.540, −1.114] |
| 644 | 1.514594e-08 | 0.98698 | −1.296 % | 85 | [−1.526, −1.066] |
| **736** (= A1 export 73321458) | **1.513612e-08** | **0.98634** | **−1.361 %** | **89** | **[−1.588, −1.134]** |
FRESH-A100 s552 vs 8be: 1.547377e-8 vs 1.570643e-8 = −1.481 %, Δ 2.33e-11 t95 [1.95e-11, 2.70e-11], 90/100 (#3599); s736: −1.480 %, 87/100. Prediction was −1.7 % (80 %: −0.5…−3.0). Second stage (gain groups) SKIPPED (null screen, §4.6).

### 4.2 A3 = A1 export + NEW10 (U1 tp2 reader, mean-field ages, var² D4 face: 9 gain groups), 368 updates on TRAIN900 ∪ d8b (`experiments/2026-10-21_L2/RESULT.md:46-70,85-88`, 2026-10-21)

| step | mean raw | ratio | paired Δraw vs A1 (1.513612e-8) | wins/100 | 95 % CI |
|---|---|---|---|---|---|
| 0 | 1.513612e-08 | 1.00000 | 0 (byte-identical to R3 s736) | — | — |
| 92 | 1.532615e-08 | 1.01255 | +1.274 % | 35 | [+0.702, +1.845] |
| 184 | 1.513273e-08 | 0.99978 | +0.001 % | 47 | [−0.477, +0.478] |
| 276 | 1.499746e-08 | 0.99084 | −0.918 % | 70 | [−1.283, −0.554] |
| **368** (export 49179dfc → A3s 5ddf3290) | **1.492264e-08** | **0.98590** | **−1.418 %** | **85** | **[−1.665, −1.171]** |
FRESH-A100 vs A1: 1.522384e-8 vs 1.547390e-8, ratio 0.98384, −1.625 % [−1.856, −1.394], Δ 2.501e-11 t95 [2.137e-11, 2.864e-11], 89/100 (#3661). Prediction −0.5 % (80 %: −0.1…−1.0): outside, better. F +0.504 %, ops +741 (23,374). NEW10 as a 112-update gain group on 8be read +0.032 % (§4.6) — the gain needed 368 updates from A1's export.

### 4.3 A4 = A3 s368 + one epoch on 2,048 fresh corpus nets, d8bext2 (`experiments/2026-10-21_L6/RESULT.md` §1, §2, §3, 2026-10-21)

Corpus facts: `hf://datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k` (HF rev cc3405a9; H200 MC, N = 1e8, raw moments m1..m4) holds **14,048 labelled nets** = d8b idx 0..2047 (16 shards; the TRAIN/d8b2048 set A1/A3 trained on) + 12,000 in `shard_ext00..07` (1,500 each, `n_done = 1500`, `w_seed0 = 770000`; shard_ext*s* holds idx 2048 + 8k + s). FRESH-D owns shard_ext00 rows 0..99. Used: **d8bext2** = ext01/02 rows 0..1023 → 2,048 nets (idx 2049…10234; mean sha 9ce74dc3; 756 MB download); **d8bext3** = ext03/04 rows 0..1471 → 2,944 nets (idx 2051…13820; sha 1e9711b2); **d8bext4** = ext05/06 rows 0..1471 → 2,944 nets (sha 82800023); cut-refit (L3b) used **l3bext** = ext07 rows 0..1499 ∪ ext00 rows 100..647 → 2,048 nets (sha 59bde3da; label check idx 2055). Unused for any post-deadline build: ext00 rows ≥ 648, ext01/02 rows ≥ 1024, ext03–06 rows ≥ 1472 (≈ 3,300 nets; `L7:` §5 says ≈ 4,000 before L3b's draw). Label check vs own N = 1e8 MC (pod RTX 5090): mean |Δm1| 4.25e-5 (d8b net 0 control), 4.30e-5 (2049), 4.36e-5 (2050), 4.64e-5 (2051), 4.55e-5 (2052), 5.0e-5 (2055); max |Δm1| 4.2–5.5e-4; layer-16 MSE 1.19–2.3e-9 ≈ 2× label noise 7.7e-10; gate ≤ 2e-4 PASS ×6.

| step | mean raw | ratio | paired Δraw vs A3 s368 (1.492264e-8) | wins/100 | 95 % CI |
|---|---|---|---|---|---|
| 0 | 1.492264e-08 | 1.00000 | 0 | — | — |
| 92 | 1.533169e-08 | 1.02741 | +2.772 % | 16 | [+2.220, +3.324] |
| 184 | 1.506249e-08 | 1.00937 | +0.966 % | 32 | [+0.598, +1.334] |
| 276 | 1.489058e-08 | 0.99785 | −0.190 % | 58 | [−0.485, +0.104] |
| **368** (export 67ed6090 → A4s 7b0d6800) | **1.483245e-08** | **0.99396** | **−0.600 %** | **73** | **[−0.793, −0.407]** |
FRESH-A100 A4s vs A3: 1.513606e-8 vs 1.522384e-8, ratio 0.994234, −0.583 % [−0.787, −0.379], 72/100 (#3668). Cumulative vs 8be VAL: A4 1.483245 / 1.534575 = −3.345 %.

### 4.4 A5 arms (one fresh epoch each on A4's export; 368 × 8 = 2,944 nets, every update a fresh net; `L7:` §2–§5, 2026-10-21)

| arm | corpus / LR | step | mean raw | paired Δraw vs A4 (1.483245e-8) | wins/100 | 95 % CI | backend |
|---|---|---|---|---|---|---|---|
| a (local 5090, #3674; landed 19:16, after cut-off) | d8bext3, corr ×10 | 92 / 184 / 276 / **368** | 1.505271 / 1.498180 / 1.485219 / **1.476130e-08** | +1.496 / +1.042 / +0.134 / **−0.491 %** | 24 / 30 / 46 / **73** | 368: [−0.667, −0.315] | WSL torch, VAL0 byte-identical |
| b (pod cbets7hsyjiodl) | d8bext4, ×10 | 368 | 1.477498e-08 | −0.379 % | 68 | [−0.557, −0.202] | pod torch 2.13, VAL0 byte-identical |
| c (pod) | d8bext3, ×3 | 368 | 1.477388e-08 | −0.400 % | 67 | [−0.548, −0.252] | same |
| **soup(b,c) = A5s** (mean of 35 tables, sha c24b7f38 → 180e488b) | | 368 | **1.476517e-08** | **−0.454 %** | **74** | **[−0.599, −0.309]** | same |
| soup(a,b,c) 5270942f (diagnostic #3701) | | 368 | 1.475947e-08 | −0.497 % | 76 | [−0.647, −0.347] | local |
| soup(a,b) 534fd278 (#3702) | | 368 | 1.475692e-08 | −0.513 % | 75 | [−0.670, −0.356] | local |
Head-to-head: soup(b,c) vs b −0.072 % [−0.148, +0.003] 55/100; vs c −0.053 % [−0.127, +0.021] 54/100; arm a vs soup(b,c) −0.037 % [−0.140, +0.065] 49/100; a vs c −0.092 % [−0.171, −0.012] 59/100; a vs b −0.109 % [−0.269, +0.052]; soup(a,b,c) vs soup(b,c) −0.044 % [−0.080, −0.008] 58; soup(a,b) vs soup(b,c) −0.060 % [−0.100, −0.020] 63. Duplicate pod run (torch 2.8, superseded): b 92 +1.282 %, 184 +0.649 %; c 92 +1.361 %. Cross-torch f32 floor: A4 VAL0 1.483189e-8 vs 1.483245e-8 (−0.004 %, max|Δpred| 2.1e-6). Prediction arm a −0.5 % (80 %: −0.2…−0.8): on. Fresh-data law: one epoch over ≈ 2,048–2,944 new nets = −0.38…−0.60 % raw (A4, a, b, c); soups add ≤ 0.06 %. Pod spend L7 ≈ $2.07 + sweep $0.42.

### 4.5 X1 transported-state corrector (post-hoc, frozen chain; `experiments/2026-10-20_x1/RESULT.md:9-23,31-34`; `experiments/2026-10-20_L1/RESULT.md` §3–§7, 2026-10-20/21)

| arm | params | Δraw VAL-100 (8be chain 1.53509e-8) | 95 % CI | wins | per-layer Δ L4/L8/L12/L16 | forecast ΔF | score-equiv |
|---|---|---|---|---|---|---|---|
| A seed 0 (state 32, 3×128) | 49,697 | **−0.840 %** | [−1.167, −0.516] | 68 | −7.6 / −4.5 / −2.2 / −0.8 % | +1.29 % (≈ 3.7 GF) | +0.44 % worse |
| A seed 1 | 49,697 | −0.741 % | [−1.069, −0.414] | 67 | −7.6 / −4.4 / −2.0 / −0.7 | +1.29 % | +0.54 % |
| B seed 0 (state 1, 2×64) | 6,530 | −0.329 % | [−0.608, −0.044] | 56 | −4.8 / −3.1 / −1.5 / −0.3 | +0.11 % | −0.22 % |
| B seed 1 | 6,530 | −0.512 % | [−0.798, −0.225] | 64 | −4.7 / −2.9 / −1.5 / −0.5 | +0.11 % | −0.40 % |
| C seed 0 (no learned state) | 37,377 | −0.476 % | [−0.712, −0.242] | 65 | −6.2 / −3.1 / −1.4 / −0.5 | +0.44 % | −0.04 % |
| C seed 1 | 37,377 | −0.446 % | [−0.689, −0.206] | 62 | −6.3 / −3.0 / −1.4 / −0.4 | +0.44 % | −0.01 % |
Pre-registered −4 % (80 %: −1.5…−8); keep ≤ −2 %: STOP. Transport premise: per-layer error explained by Φ(α)∘(Wᵀd) with gain 1.00 ± 0.03, R² 0.45–0.5 at L2–4, **0.74–0.81 at L11–15**.
X1-B as fnp module on 8be (0937b9cb; L1 §3): VAL −0.437 % t95 [−0.755, −0.119], 56/100, ΔF +0.142 % (+407,143,174 F, +735 ops; lean f96c853f +0.1347 %, +612 ops), score-equiv −0.296 %; FRESH-A100 −0.591 % [−0.901, −0.281], 66/100, score-equiv −0.450 %; misses the 60-wins rule. fnp vs torch max|Δ final| 1.2e-7; zero control exact.
X1-B retrained on A1's chain (L1 §7, D1): seeds B0 −0.017 % / B1 −0.002 % hold; VAL −0.015 % t95 [−0.065, +0.035], 50/100, score-equiv +0.120 %; FRESH −0.034 % [−0.094, +0.025], 51/100, +0.101 %; ΔF +0.135 %, 23,245 ops ⇒ residual-eq 0.354 s. **NO-GO** — the R3 refit absorbed what the module found on 8be.

### 4.6 Gain-group screen on exact 8be, 112 updates, corrector ×10, pod (`experiments/2026-10-20_L3/RESULT.md:62-68,94-98`, 2026-10-20)

| arm vs CONTROL | step 28 | 56 | 84 | **112** | wins @112 | 95 % CI @112 | bill | verdict |
|---|---|---|---|---|---|---|---|---|
| ALL = G1 ∪ G2 (+G3 ⊂ G1) | +5.728 % | +0.973 % | +0.289 % | **+0.018 %** | 52 | [−0.220, +0.255] | = 8be | DROP (predicted −1.3 %) |
| ALL-R = ALL + r320 | +3.434 % (step 0 −1.182 %, 86) | −0.041 % | −0.906 % | **−1.175 %** | 77 | [−1.507, −0.843] | +1.86 % F | DROP (+0.65 % score) |
| G1 U1 NEW10 (9 tables, lr 1e-2) | +4.781 % | +1.073 % | +0.234 % | **+0.032 %** | 50 | [−0.191, +0.255] | — | DROP (predicted −0.9 %) |
| G2 ρ-face `_RHO` (lr 1e-6) | +0.260 % | +0.060 % | +0.003 % | **−0.001 %** | 54 | [−0.013, +0.011] | — | DROP; J-table init 7.8× worse at step 0 (1.29e-7 vs 1.65e-8); learned ρ ≤ 0.01× table |
| G3 marg2 + metric + closure | −0.110 % | −0.145 % | −0.198 % | **−0.209 %** | 74 | [−0.274, −0.144] | 288,244,803,047 F (+0.480 %), 22,735 ops (+500) | +0.27 % score ⇒ DROP |
Historic NEW10 numbers (`LED:` N4, N4a): matched refit 3431 −0.300 % 77/100 [0.216, 0.384]; NEW10 lr 0.01 job 3472 −0.916 % 81/100 [0.689, 1.144].

### 4.7 L4 pair-local κ3 slice (D21) error, 8be, 10 TRAIN nets, N = 1e9 labels (`experiments/2026-10-20_L4/RESULT.md:36-50,53-56,69`, 2026-10-20)

| layer l | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| served D21 rel. error | 3.6 % | 34.9 | 38.8 | 48.5 | 55.5 | 54.2 | 59.7 | 63.4 | 63.1 | 64.5 | 67.4 | 68.1 | 68.1 |
| weighted OOS R² ridge13 | 0.088 | 0.157 | 0.190 | 0.216 | 0.232 | 0.225 | 0.239 | 0.242 | 0.226 | 0.234 | 0.230 | 0.228 | 0.210 |
| R² w MLP 3×32 (ceiling) | 0.140 | 0.189 | 0.213 | 0.238 | 0.245 | 0.239 | 0.252 | 0.258 | 0.232 | 0.240 | 0.231 | 0.234 | 0.221 |
| R² w scale only | 0.059 | 0.132 | 0.165 | 0.194 | 0.211 | 0.207 | 0.218 | 0.219 | 0.201 | 0.215 | 0.209 | 0.207 | 0.187 |
| scale g (served D21 too small) | 1.0086 | 1.0183 | 1.0266 | 1.0324 | 1.0356 | 1.0382 | 1.0427 | 1.0449 | 1.0461 | 1.0495 | 1.0495 | 1.0517 | 1.0494 |
| w-error share kk / kd / dd | .48/.45/.07 | .32/.57/.11 | .24/.63/.13 | .18/.68/.14 | .14/.69/.17 | .12/.71/.18 | .11/.72/.18 | .10/.72/.18 | .09/.72/.19 | .07/.74/.19 | .07/.74/.19 | .06/.74/.19 | .07/.74/.19 |
Prediction P(R²w ≥ 0.3 at l ≥ 9) ≈ 0.10: did not fire (max 0.242 / 0.258 at l = 9); kill line (< 0.15 all l ≥ 8) not met either. Scale alone removes ≈ 44 % of weighted D21 error energy at l ≥ 8. Unrefit `_G21` screen (VAL 900–928): **+34.7 % raw** t95 [+30.5, +38.9], 0/29 — closed. Dump checks: recorded == unrecorded bit-identical; `pred[l−1] @ W_l` vs `mu_pre[l]` max|Δ| ≤ 5.7e-6.

### 4.8 L4e — young κ4 tree fidelity on A4's export, FIT 0–9, L6–15, R²w vs dG label (N = 1e9), K1 rows (`experiments/2026-10-21_L4e/RESULT.md:33-44,76-90,102-113`, 2026-10-21)

| layer | transport dG | Tier law | **shipped D4/2** | no young | chain1 (coef 1) | clean1 | exact1 | chain1 @ best gain (g) | exact1 @ gain (g) | coefficient cost |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.338 | 0.664 | **0.779** | 0.614 | 0.646 | 0.645 | 0.634 | 0.807 (0.52) | 0.808 (0.51) | +0.028 |
| 7 | 0.416 | 0.719 | **0.728** | 0.596 | 0.752 | 0.750 | 0.741 | 0.831 (0.63) | 0.831 (0.62) | +0.103 |
| 8 | 0.394 | 0.670 | **0.780** | 0.634 | 0.660 | 0.659 | 0.638 | 0.780 (0.52) | 0.780 (0.50) | +0.000 |
| 9 | 0.388 | 0.625 | **0.693** | 0.528 | 0.771 | 0.772 | 0.764 | 0.795 (0.77) | 0.796 (0.74) | +0.103 |
| 10 | 0.421 | 0.636 | **0.780** | 0.650 | 0.717 | 0.717 | 0.707 | 0.781 (0.59) | 0.782 (0.57) | +0.002 |
| 11 | 0.398 | 0.637 | **0.623** | 0.433 | 0.785 | 0.786 | 0.789 | 0.788 (1.08) | 0.790 (1.04) | +0.164 |
| 12 | 0.453 | 0.664 | **0.767** | 0.674 | 0.715 | 0.716 | 0.706 | 0.768 (0.57) | 0.769 (0.55) | +0.001 |
| 13 | 0.398 | 0.620 | **0.730** | 0.610 | 0.751 | 0.752 | 0.748 | 0.767 (0.76) | 0.768 (0.74) | +0.037 |
| 14 | 0.419 | 0.661 | **0.758** | 0.674 | 0.720 | 0.721 | 0.716 | 0.755 (0.60) | 0.757 (0.59) | −0.003 |
| 15 | 0.475 | 0.639 | **0.724** | 0.668 | 0.641 | 0.641 | 0.632 | 0.705 (0.43) | 0.706 (0.42) | −0.019 |
clean − chain Dx: Δ ≤ 0.001 (fold moves Dx 1.9–2.8 %); exact Dx (GPU MC N = 2²⁵, column corr 0.996–0.998): Δ ≤ 0.002 ⇒ **input-fidelity loss ≈ 0**; recoverable = coefficients (+0.10 / +0.10 / +0.16 at L7 / L9 / L11) and coverage. Clean-Dx reader price (A4 export, FIT 0, same-meter 0.12.1): **+69,571,264,442 F = +24.1 % of bill (+3.16 % of B), +1,556 ops**; L ≥ 9 only +36.1 GF (+12.5 %); per-layer ΔF (GF) L3…L15: 2.11, 4.34, 5.54, 6.85, 7.37, 7.23, 6.54, 5.89, 5.23, 4.81, 4.25, 4.37, 5.04. Break-even raw ≈ −19 % (full) / −11 % (L ≥ 9).

## 5. Rerun-margin analysis (`experiments/2026-10-21_L1/RESULT.md` §2 (`margin.json`, `telemetry.json`), 2026-10-21; `DES:` A5s/B4 rows)

Data: 82 graded runs (84 by `LED:` M6/M7), Sep 14 – Oct 21, 23 UTC days, 100 MLPs each (14 reports fetched as `RR:L1fetch_*`). Constants: worst within-run tail residual max/median **1.19631** (#334487, Oct 6), wall **1.24193** (#336083, Oct 14); worst same-code day shift **1.02872** residual / **1.03132** wall (v50 ×3, Sep 22→23); worst per-op rate **15.92 µs/op** (#336147, Oct 15). Central = own median × tail; pess. = central × day shift; headroom = 1 − worst/cap (caps 0.4 s residual, 120 s wall).

Per-day summary rows (residual µs/op worst / p90 / median; kernel ps/FLOP w/p90/med; overhead share; worst residual tail; worst wall tail):

| day | runs | µs/op | ps/FLOP | overhead | res tail | wall tail |
|---|---:|---|---|---:|---:|---:|
| 2026-09-23 (noisiest early) | 5 | 11.86 / 11.26 / 10.22 | 219 / 191 / 164 | 0.105 | 1.142 | 1.234 |
| 2026-10-06 (worst residual tail) | 4 | 12.43 / 10.64 / 10.28 | 203 / 181 / 138 | 0.126 | **1.196** | 1.153 |
| 2026-10-14 (worst wall tail) | 5 | 14.12 / 13.68 / 12.24 | 273 / 230 / 195 | 0.131 | 1.090 | **1.242** |
| 2026-10-15 (worst per-op) | 7 | **15.92** / 14.33 / 13.77 | 244 / 223 / 196 | 0.143 | 1.141 | 1.183 |
| 2026-10-17 | 5 | 15.54 / 13.94 / 13.47 | 237 / 203 / 195 | 0.137 | 1.166 | 1.221 |
| 2026-10-21 (nominee day) | 2 (table) | 14.80 / 14.59 / 14.15 | 252 / 239 / 196 | 0.139 | 1.064 | 1.133 |
Era medians of µs/op: 10.1–10.6 (to Oct 12), 11.8–12.2 (Oct 13–14), 13.5–14.2 (Oct 15 on); each day within ±3 % of its era. Residual-tail deciles 1.02–1.12, wall 1.03–1.15; per-MLP corr(residual, wall) 0.87–0.98; residual tail inflates at 0.67× the wall tail. Worst-of-N tail (8,393 pooled per-MLP draws ÷ run median): residual ×1.095 (N = 100), ×1.140 (1,000), ×1.181 (10,000); wall ×1.127 / ×1.205 / ×1.242 (`DES:` § Provisional pair).

Nominee / candidate margin table (`DES:` tables; `L7:` §5; L1 §2 nominee table):

| build (#) | ops | graded residual med / max | worst-day residual central / pess. | residual headroom central / pess. | worst-day wall central / pess. | wall headroom | uniform slowdown to first fail (residual; wall) | on top of pess. worst day | worst MLP, N = 1,000 / 10,000 rerun |
|---|---:|---|---|---|---|---|---|---|---|
| **A5s #337601** | 20,991 | 0.2913 / 0.3017 | 0.349 / 0.359 | **+12.9 % / +10.4 %** | 88.5 / 91.3 (recomputed: 71.26 × 1.24193 × [1, 1.03132]) | +26 % / +24 % | **1.326×** (wall 1.528×) | 1.116× | — |
| **B4 #337562** | 19,684 | 0.2674 / 0.2819 | 0.320 / 0.329 | **+20.0 % / +17.7 %** | 68.0 / 70.2 | +43 % / +42 % | **1.419×** (wall 2.01×) | 1.215× | 0.305 / 0.316 s |
| A4s #337511 | 20,991 | 0.2900 / 0.3115 | 0.347 / 0.357 | +13.3 % / +10.8 % (L6 states +15.3/+12.1 as cap/worst − 1) | 88.7 / 91.4 | +26 % / +24 % | 1.284× (wall 1.496×) | 1.121× | 0.331 / 0.343 s |
| A1s #337481 | 20,630 | 0.2905 / 0.3023 | 0.348 / 0.358 | +13.1 % / +10.6 % | 87.2 / — | +27 % | 1.323× | — | 0.331 / 0.343 s |
| B-young #337462 | 18,943 | 0.2594 / 0.2760 | 0.310 / 0.319 (per-op 0.302) | +22.4 % / +20.2 % (per-op +24.6 %) | 66.4 / 68.5 | +45 % / +43 % | 1.449× (wall 1.979×) | 1.253× (L1: 1.289× central) | 0.296 / 0.306 s |
| A1 #337456 | 22,633 | 0.3261 / 0.3349 | 0.390 / 0.401 (per-op 0.360) | **+2.5 % / −0.3 %** (per-op +9.9 %) ✗ | 96.8 / 99.9 | +19 % / +17 % | 1.194× (wall 1.442×) | 1.025× | 0.372 / 0.385 s |
| λ 1e6 twin #337470 | 22,235 | 0.3171 / 0.3256 | 0.379 / 0.390 | +5.1 % / +2.4 % ✗ | 94.3 / 97.2 | +21 % / +19 % | 1.228× | — | — |
| A3 #337495 | 23,374 | 0.3331 / 0.3447 | 0.398 / 0.410 | +0.4 % / −2.4 % ✗ | — | — | 1.160× | — | — |
| 8be #336993 (ref) | 22,235 | 0.3081 / 0.3389 | 0.369 / 0.379 | +7.9 % / +5.2 % | 90.8 / 93.6 | +24 % / +22 % | 1.180× (wall 1.474×) | 1.085× | — |
Rule (L3): ≥ 10 % worst-day residual headroom, central and pessimistic: A5s, B4, A4s, A1s, B-young PASS; A1, twin, A3 FAIL. Ops line for ≥ 10 % central headroom: ≤ 20,889 (A1's rate) / ≤ 21,719 (8be's).

Leaders (board G §3; wall only; `experiments/2026-10-21_L1/RESULT.md` §2 leaders table):

| entry | adj | wall med / max | first fail × graded (120/max) | central worst-day wall (med × 1.242) | × worst day to fail |
|---|---|---|---:|---:|---:|
| marius_binner #337106 | 1.310e-9 | 87.8 / 98.2 | 1.222 | 109.0 | 1.100 |
| oliver_chen #337003 | 1.418e-9 | 103.3 / 110.6 | 1.085 | 128.3 | 0.935 |
| J2W #337185 | 1.418e-9 | 109.5 / 118.3 | **1.014** | 136.0 | 0.882 |
| mliston #337142 | 1.433e-9 | 100.1 / 112.0 | 1.071 | 124.3 | 0.965 |
| Luna #336449 / #337205 | 1.484e-9 / 1.503e-9 | 98.6 / 103.4 · 96.4 / 104.8 | 1.161 / 1.145 | 122.5 / 119.7 | 0.980 / 1.002 |
| AndreasHad04 #337136 | 1.614e-9 | 104.6 / 109.2 | 1.099 | 129.9 | 0.924 |
| thegamers #337296 | 1.830e-9 | 96.3 / 98.5 | 1.218 | 119.6 | 1.003 |
| andrei_bulzan #337327 | 1.833e-9 | 86.2 / 116.3 | 1.032 | 107.1 | 1.121 |
| megan_park #337303 | 1.930e-9 | 105.4 / 107.1 | 1.120 | 130.9 | 0.917 |
8 of 10 fail before A1 (1.194×); 7 of 10 exceed 120 s on a replay of our worst wall day; B4 (1.419×) and B-young (1.449×) outlast every entry; A5s (1.326×) outlasts all but marius / thegamers. Caveat: page maxima may understate by ≈ 1.6 % (ours 80.1 page vs 81.4 telemetry); their residual cliff is invisible.
Smoke shapes (L1 §1): A1 and B PASS 256×4, 512×16, 1024×16, 2048×4, 1024×32 (0.44–0.49 B F; ≈ 165–220 s served); weight scale ×0.9 / ×1.1 PASS; **×0.5 / ×2 at 1024×16 ⇒ IndexError in `_yp_contract` via `_dslice21`** (A1, B, 8be; `RESULTS.md:3331`). Rehearsals (`experiments/2026-10-21_L3/RESULT.md` §2): A1 300/300, B 300/300, A1s 100/100, A4s 100/100, B4 100/100, A5s 100/100 fresh seeds 9,250,000+, 0 errors; flopscope-server grows ≈ 0.55–0.57 KB/op (≈ 1.1 GB / 100 calls), same per op for every build.

## 6. Ledger rows not covered above (`docs/WRITEUP_LEDGER_2026-10-21.md` §1–§3, 2026-10-21; `B:` = `_claude_snap/handoff_2026-10-19/B_levers.md`, `F:` = `F_math.md`, `R:` = `RESULTS.md`)

| ledger # | mechanism / lever | number | evidence | date |
|---|---|---|---|---|
| 1 | hub-star κ3 lineages, sandbox n=128 L=8 share of κ3 effect | diag 58 % → +pair 90 % → +hub 99.8 % → +order-3 100 % | F:88-91 | 2026-10-19 |
| 1a | hub routes / compensation off (real chain) | +5.4 % raw (check) | A:209 | 2026-09-25 |
| 2a | capacity-cut elasticities | X2 Δraw/|ΔF| ≈ 4–5; age 5 −4.5 % F / +6.15 % raw; age 7 +3.75 % F / −1.74 % raw | X2; B:178; B:146 | 2026-10-13…20 |
| 3 | six exact ages (age 6 vs 5) | raw −6.15 % VAL, −5.84 % FRESH-A; F +5.55 %; graded −1.54 %; 53/100 | B:132; R:3113-3114, 3137 | 2026-10-12 |
| 4 | riding band r288 vs b636 join | F −2.586 %, raw −0.076 %, score −2.660 %, 92/100, t95 [4.92e-11, 6.59e-11]; join 20.59 → 7.35 GF | R:5723-5725; A:180 | 2026-10-18 |
| 5 | P2 7→4 leg identity, rank 4 | rank 2 +11 % raw (check); age6 + P2 r4 raw −4.51 %, F −0.23 % (check scope) | A:208; R:1324 | 2026-09-15…10-12 |
| 6 | Wick transport of κ3 legs (sandbox) | gate g³ vs pattern-aware = −0.4 % of κ3 effect | F:87 | 2026-10-19 |
| 7 | Edgeworth readout + θ-damped closure extras | extras off +18 % raw (check); sandbox direct readout 21 %, indirect 58 %, both 91 % of K2-only error removed | A:210; F:18, 84-85 | 2026-09-25 / 10-19 |
| 8, 8a, 8b | κ4 regeneration λ·C core (504aldo P1); λ law from first principles; λ vs exact labels | ≈ 90 % of AUGMENT gain at zero n³ (check); sandbox n=64 residual 8.16e-5 → 5.03e-5 (−38 %), v_eff(16) ≈ 0.006 vs fitted 0.006–0.0095; fitted v_l 0.00034 (L2) → 0.0016–0.0039 (L9–16) | CLAIMS_LEDGER:109; F:21, 262-264; H_x4_ceiling.md:12 | 2026-09-14 / 10-19 / 10-20 |
| 9 | kai's two-term Q31 law (GQ1), inside C1 cuts | C1: F −1.09 %, raw −0.19 % ± 0.07 FRESH-D, 18/32; STAR0/GQ1 alone −748,961,996 F / −346 ops | B:228; R:5539 | 2026-10-18 |
| 10 | K9.4b joint κ4 refit (Tier-1 D4 law, Tier-1.5/2) | raw −4.21 % VAL vs matched ctl; FRESH-A 1.0144× (69/100); graded −1.3 % | B:166 | 2026-10-16 |
| 11 | K9.2 fold births into D21 (|α| ≤ 2) | raw −1.89 % VAL (83/100); FRESH-A 1.0191× (87/100); F +0.26 %; graded −2.6 % | B:164; R:3435-3436 | 2026-10-16 |
| 12 | masked star-Q31 fold (M5.1) | graded raw −3.04 % (FRESH −2.84 %, 86/100); F +0.27 % | B:128 | 2026-10-11 |
| 13 | partial young-tree κ4 reader (tp2, forum 18249 physics) | raw −3.53 % VAL vs CONTROL (94/100); F +0.95 %; graded −3.70 % vs v97 | B:181; R:3648 | 2026-10-16 |
| 14 | dressed newborn legs; D21ᵀ dress gain 0.9 | dress raw −6.5 % FRESH-A (85/100); ×0.9: −0.38 % VAL, −0.37 % FRESH-A, 0 F | B:26, 105, 156; R:2741 | 2026-10-07 / 10-15 |
| 15 | per-neuron birth gates | birth gates −0.31 % (55/100); transport gates −0.42 % (70/100) (check active) | B:96 | 2026-09-27 |
| 16–16d | KAS pruning: v75 / v77 Hd; gain-mode fills; stack C; two-sided D21 + ρ; O704 | F −11.4 % raw +2.2 % (#334464); F −5.5 % raw −0.1 % (#334487); fills F −13.8 %, refit raw +1.77 %, graded −10.1 %; C F −2.09 %, raw +0.76 %, graded −1.63 %; V8b F −3.59 %, raw −0.16 %; O704 F −0.59 %, raw −0.076 %, FRESH score −0.70 % 100/100 | B:22-24, 104, 137, 149, 125, 228; R:2647, 2679, 5537 | 2026-10-06…18 |
| 17 | 7b covlive (α > −3.5) | F −0.49 %; raw −0.06 % DEV (90/100 score wins) | B:161 | 2026-10-16 |
| 18 | exact K2 chain, two-level symmetric sandwich | 29.19 GF = 10.2 %; sandwich F −0.23 %, raw 0; layer-0 WᵀW hygiene F −2.16 %, raw +0.02 % | A:177, 194; B:123, 109 | 2026-10-07…19 |
| 19–19e | batched Strassen–Winograd; sb2; sbtrim+depth16; sks:trim; λ-priced planner; padding; op diet + λ 1e6 | WP21 no accuracy cost; Winograd glue F −0.79 %; sb2 F −4.73 % (bit-identical); sbtrim+d16 F −0.97 %, raw +0.02 %; sbks:trim F −1.69 %, raw +0.073 % (39/100, CI incl. 0, #336099); depthw F −0.26 %, raw −0.03 %; pad:1024+opfix F −1.69 %, full pad F −1.04 %, raw 0; op diet F −5.97 % vs b636, raw −0.13 % (57/100), canonical VAL score −6.10 % (99/100) | archive LOG:34904; B:94, 133-134, 150, 151, 158, 160, 230; R:3148, 3210, 3291, 6412 | 2026-09-08…10-18 |
| 20, 20a, 20b, 20c | in-loop corrector (first version); hub features (hub84); terminal local-even; R1 faithful refit | raw −5.60 % DEV, −4.76 % HOLDOUT-600; hub84 graded −0.48 %, F +0.06 %; local-even FRESH +0.097 % (63/100), graded −0.16 %; R1 canonical VAL raw −1.518 %, 90/100, df99 95 % [1.263, 1.773] vs b636 | R:1966; B:92, 173, 211; R:4266, 6573-6582 | 2026-09-24…10-18 |
| N2, N3, N3a | age 5 @ r256 / age 6 @ r224 / r192; age 7 (O4); deep-only age 7 corrector off | +6.15 / +2.5 / +8 % raw for −4.5 / −1.3 / −3.7 % F; age 7 F +3.75 %, raw −1.74 %, score +1.94 %; +1.08 % (7/8 worse) vs +1.51 % break-even | B:178, 146, 203 | 2026-10-13…17 |
| N5 | U2b nested-Dx reader ages 2–3, L ≥ 9, matched refit 3444 | raw −1.242 % VAL, 100/100, [1.146, 1.339]; F ≈ +3.4 % ⇒ ≈ +2 % score | R:7452-7453; B:236 | 2026-10-18 |
| N6 | full retained-tree D4 half (3179) | F +68 %; raw −2.5 % / −5.5 % (1 net) | B:182 | 2026-10-16 |
| N9, N9a | X3 sphere formulation on Gaussian chain; sphere vs Gaussian sandbox | +380.7 % TRAIN900 (0/900), +361.7 % VAL (0/100) — spec error; sandbox MSE ratio 0.825 / 0.880 / 0.909 at n = 64/128/256 | R:8561-8562; F:272, 279 | 2026-10-19/20 |
| N10 | X4 SSC / scale-mixture κ4 faces (ceiling on 8be, 100 FIT nets) | var² face +2.9 % unweighted / +2.3 % weighted; (2,2) face v = 0.0000; expected −0.2 % raw | H_x4_ceiling.md:7-11, 86 | 2026-10-20 |
| N11, N11a | X6 rank-k covariance-error state (sandbox); zero-FLOP cov faces on real labels (J) | oracle rank 8–16 reproduces 88–99 % of variance error at n = 128; k = 16 ≈ 20–50 % of channel at 1024×16; ρ m mᵀ face R² 0.40–0.64 unweighted L9–15, weighted 0.4–0.6 % | I_x6_sandbox.md:14-20; J_cov_faces.md:7-9, 128-130 | 2026-10-20 |
| N14 | pooled-column κ3 carrier ("cap 3776", sandbox) | 5–10× worse weighted D3 error than exact-age + band at equal cost (n = 128, 256) | K_pooled_carrier.md:11-18 | 2026-10-21 |
| N16, N17, N18, N19 | star-D4 births (STAR0 removed); exact svd(k=256) join (removed, 0.12.3 +3.08 % F); O1 r224/r192/r128, O2 split legs, O3 no old tier; K4 drop old tier at last layer | −3.5 / −3.0 / −0.9 / −1.3 % raw; graded −0.67 %; r224 +2.46 %, r192 +8.0 %, r128 +41 %; O2 +1.07 %; O3 ×4.0…×16; +206…+226 % | B:110, 154, 70, 141-143, 139 | 2026-10-08…16 |
| N20–N26 | coherent-scale extraction; S5 learned closure; exact first-layer Gaussian pK; K5 adaptive counts; corrector h32+res64 / causal; adjoint; sketching | −27 / −33 % raw (2/32 nets, correctors off) at 11–14 TF; −0.75 % at F +1.68 %; +1.11 % (1/8); VAL −1.42 %, FRESH −0.52 %, F +0.89 %; −0.0002 % / +7.46 %; adjoint 27/21/12/7/4 % of remaining MSE at a = 1..5, ≈ 0 on 6 ages + band; sketch variance ≈ n/k | B:221, 202, 223, 100, 242, 140, 200-201; F:112, 20 | 2026-09-27…10-19 |
| M1–M5 | exact post-15 mean oracle; 3476 coupled-consumer anatomy; corrector-off exact κ3+κ4 slices; label floor; served D21 slice error | final MSE −74…−79 % (2 nets); C (exact pre-act cov) −37.7 / −39.3 %, ALL 0.065×; 0.072× (κ4 alone 0.564×, κ3 alone 0.596×), corrector on 1.012×; floor ≈ 7.7e-11 (N = 1e9 × 2), FRESH-A floor 7.48e-10 (N = 1e8); 3.6 % (l2) → 35 % (l3) → 68 % (l14) | B:279, 282, 277, 298; L4 | 2026-09-25…10-20 |
| M10 | 100-call native-Linux memory, 8 GiB RLIMIT_AS | VmPeak 4,543 (A1s) / 4,588 (A1) / 4,555 (8be) / 4,593–4,649 MB (A3s); A4s/A5s by identity 4,593 MB; B 3,668 MB; B4 3,848 MB; settled growth +1…+34 MiB after warm-up; WSL A1 #3626 4,613 MiB −27.0 MiB | L1 §6; L3b §A3; `DES:` | 2026-10-21 |
| E1 (external, forum 18264, abhimanyu_singh #337475, 2.966e-9, raw 1.986e-8, mult 0.1494) | σ² law 0.9σ² (D21) / 0.6σ² (D3), sketch k ≈ 280n; eval-layer attenuation ×36 early vs late; truncation bias transported at gain ≈ 1; full augment 1.6–2.6× better MSE for 3–4× cost | agrees with N26, M5, N7a (gain 1.00 ± 0.03), N6/N30 | `LED:` §2b | 2026-10-21 |
| Gaps (`LED:` §3) | no derivation of the hub-star floor theorem; no Edgeworth-readout ablation on the real chain; no `_LAM_GAIN` ablation on A1; no isolated GQ1 Δ; no corrector-off ablation of A1/8be; no batched-vs-unbatched F Δ; no riding-band "triangular variant" number; no single "A1 with vs without KAS" read | — | 2026-10-21 |

## 7. Provenance and public resources

### 7.1 504aldo mechanism table (`docs/PROVENANCE_504aldo.md:135-158,178-213,229-246`; repo `github.com/504aldo/whest-p2-cumulant-k3`, MIT, commit 18c17e2d (2026-09-10), forum post 2026-09-09; their #330093 V29 adj 5.40e-9, raw 2.13e-8, C/B 0.2526; our control then F21 #330950 adj 1.5957e-8, raw 3.7795e-8, C/B 0.4222)

| # | mechanism | their finding | our file / symbol | status | ours vs theirs |
|---|---|---|---|---|---|
| P1 | κ4 regeneration `G = diag(g) + λ·C_off`, consumers + κ4→κ3 feed | F68 | `estimator_k3_v30.py` → `_own_lambda`, `_LAM_GAIN` | RE-IMPLEMENTED (own code from F21 lineage); 1.2371× on 24/24 design nets, t95 [+3.36e-9, +4.39e-9]; their 1.97× | r = 1 harmonic core matched from `estimator_v17.py:679`; consumers/transport already ours; **own λ fit** (24 fresh P03 nets, label-free) — their `lam_prefit_perdump.npy` not taken |
| P2 | 7→4 leg identity `S21 = S_sep + R_res`, thin transported residual | F63 | `estimator_k3_v31.py` → `_p2_compress`, `_p2_recon`, `_P2_RANK = 4` | RE-IMPLEMENTED, **part theirs**; 0.7128× of v30 cost (−28.7 %), C/B 0.4222 → 0.3010 | **ours**: reconstruction identity `A_D = M·d(s) + 3·A_L·d(e) + Z·Lᵀ` (committed 1h47m before clone, cfb13bd 2026-09-14 22:24), 2-D Strassen-wide layout + shuttles, equivariant sketch (their `W[:, :r]` fails permutation-equivariance); **theirs, matched deliberately**: (2,1) regrouping (`LA/LP/MP/PPL`, e61b7d8 11h15m after clone), range finder (one line changed), rank 32 (we ship 4) |
| P3 | rank-16 D21→birth feedback | F69 | `_P3_ON = False`, `_P3_RANK` | RE-IMPLEMENTED, **OFF**: r16 −4.57 % raw / +6.63 % F = +1.76 % score; r8 −3.28 % / +3.64 % = +0.24 % | closed (PLAN rev 13 §5) |
| P4 | Strassen depth / pooling / block-symmetric `C_pre` / final-layer trim | F86 cost anatomy | `_STRASSEN_DEPTH`, `_p4_sym_sandwich`, `_P4_FINAL_VAR` | RE-IMPLEMENTED, IN | own code (batched flat-ops planner, sks, padding, λ-priced planner = ours, §6 rows 19–19f); idea list is F86's |
| P5 | age-gated shared basis 384@age4 → 224@age7 | F72/F73 | `_SHARED_ON`, `_SHARED_AGE = 6`, `_SHARED_RANK = 288`, `_shared_join_lean` | RE-IMPLEMENTED in own spelling, IN | single band (r256 → r288 riding, no re-rotation) vs their nested two-tier; age 4→5 D5 v53 #331880 1.1136×; riding band ours (`LED:` row 4) |
Not taken: their fitted λ table (fitted on 8 public MLPs), mutable-buffer/`out=` patterns, local-runner advice, range-finder sketch. Attribution statement: `docs/PROVENANCE_504aldo.md:268-276`.

### 7.2 Public resources used (paths where cited)

| resource | use | cited in |
|---|---|---|
| `hf://datasets/keenanpepper/arc-whestbench-p2-d8b-corpus-14k` (rev cc3405a9; 14,048 nets, N = 1e8 H200 MC, m1..m4) | TRAIN (d8b 0..2047) for R1/R3/A3; fresh epochs d8bext2/3/4 (A4, A5); FRESH-D (ext00 rows 0..99) | `experiments/2026-10-21_L6/RESULT.md` §1; `L7:` §1 |
| keenanpepper `whestbench-relu-mlp-moments-10k`, `whestbench-relu-mlp-means-500k`, `arc-whestbench-p2-higher-moments-2026`, `arc-whestbench-higher-moments-2026` | referenced datasets | `RESULTS.md`, `_claude_snap/handoff_2026-10-19/C_data_infra.md` |
| `hf://aicrowd/arc-whestbench-public-2026`, `github.com/AIcrowd/whest-starterkit`, `github.com/AIcrowd/flopscope` (v0.12.3; issue #321) | public panel, harness, meter | `RESULTS.md`, `docs/` |
| `github.com/504aldo/whest-p2-cumulant-k3` (MIT, 18c17e2d) | §7.1 | `docs/PROVENANCE_504aldo.md` |
| `github.com/alignment-research-center/mlp_cumulant_propagation` (ARC reference) | cumulant-propagation reference, power-cumulant trick | `E_literature.md`, `RESULTS.md` |
| `github.com/logxio/fourth-cumulant` (Susupro #336537, forum 18254) | width-exponent confirmation: K=3 MSE ∝ n^−3.08 [2.95, 3.17], K=4 ∝ n^−3.95 | `RESULTS.md` |
| `github.com/al07-alt/whest-p2-k31-two-term` (kai; forum 18246) | two-term Q31 law (`LED:` row 9); exact Q31 slice −18 % in V29, two-term law −3 % | `RESULTS.md`, `B_levers.md` |
| `github.com/AndreiBulzan/arc-whitebox-submission-and-documentation`, `github.com/acco-cyber/arc-whitebox-estimation-2026`, `github.com/agentaiventure-dot/arc-whest-estimator`, `github.com/jamesrahenry/arc-whitebox-replication`, `github.com/jgdumas/plinopt` | read for contrast / Strassen schedules; no build input recorded | `E_literature.md`, `G_board_forensics.md` |
| forum 18249 (Kuznetsov, write-up #333534) | young-tree κ4 reader physics (`LED:` row 13, −3.53 % VAL) | `RESULTS.md:3648`, `B:181` |
| forum 18248 (il.miscusi) | coherent-scale / scale-mode extraction (N20; −27…−33 % at 11–14 TF) | `B:221-223` |
| forum 18250 (DD_team) | exact compute identities (13.5 % saving claim; read only) | `RESULTS.md` |
| forum 18252 (Nurlan Abdrash #336225) | error budget; all-distinct κ3 = 0 costs 43× | `RESULTS.md` |
| forum 18254 (Susupro) / 18255 (hyojun #332177/#335413: S–V six-level schedule saved 335,364,096 F) / 18218 (ngmitam leaf 32→16→8: −0.5 % / −3.5 % F) | Strassen and width evidence | `RESULTS.md` |
| forum 18251 (organizer: flopscope 0.12.3, deadline Oct 21 23:59 UTC, nomination extended to Oct 22 23:59 UTC per 18251 as read 2026-10-21 night; the 18:00 figure is the pre-extension time) / 18118 (re-evaluation) / 18240 (derived-code rule) / 18241 (AndreasHad04 flopscope PRs #286–#303) / 18238 (long-lived-worker memory) / 18092 (`budget_summary` 157× residual) / 18109 (Phase 1 extension) / 18166, 18170 (P1 write-ups marius, mliston) | rules, grader behaviour | `docs/`, `D_aicrowd.md` |
| forum 18264 (abhimanyu_singh #337475, 2026-10-21) | external bias/variance budget, cited for contrast only (§6 E1) | `LED:` §2b |
