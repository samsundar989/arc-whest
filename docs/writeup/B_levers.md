# B — Lever ledger, ceilings, error budget, untried directions, process cost

Evidence audit of the read-only snapshot `/home/claude/whest` (state as of 2026-10-19 ~12:21 UTC).
Every number carries its source path. `[inferred]` marks anything derived by the auditor rather than stated in a document.
No recommendations are made here.

Conventions used below
- "raw" = final-layer MSE; "F/B" = FLOPs/2^41; score = raw × max(0.1, F/B).
- Paired gains quote the panel (DEV-100 mini, VAL-100 full 900–999, FRESH-A 100 own fresh nets, TRAIN0–7 = reused fitted nets, "2-net" = TRAIN0/1). "t95" = paired 95 % interval as reported.
- "u" = one dense 1024³ product = 2,146,435,072 FLOPs (audit convention, `docs/research/2026-10-15_exploration/audit/REPORT.md`).
- Gap not covered: the Project docs `claude/round4-reports-2026-10-17.md`, `claude/round4-attack-plan-2026-10-18.md`, `claude/round4-refit-runbook-2026-10-18.md`, `claude/handoff-2026-10-18-patches-and-memory-criterion.md`, `claude/checkin-2026-10-18-1430.md` are not in the snapshot and no project-read tool was available to this agent. Their content was reconstructed from what they left behind: `experiments/2026-10-18_p12/claude_handoff/README.md` (the Oct 18 patch archive), `docs/STRATEGIST_ADDENDUM_2026-10-17_b.md` §2 (rev-30 levers = round-4 attack list), `docs/STRATEGIST_ADDENDUM_2026-10-18_a.md` §0.3 (round-4 screen outcomes), and the Oct 17–18 rows of `RESULTS.md`.

---

## 1. GRADED TIMELINE (every graded submission since Oct 1)

Source: `experiments/submissions.jsonl` (94 rows total; 56 since Oct 1). F/B derived as score ÷ secondary (raw) `[inferred]`. "Δ vs prev" is the change relative to the previous A-line (or B-line for B twins) entry, as described in the ledger `description`/`note`. Standing is the ledger's `standing` field.

| id | UTC | source | score | raw | F/B | what changed vs previous | standing |
|---|---|---|---:|---:|---:|---|---|
| 334434 | 10-06 14:31 | v74_ws | 4.9761e-9 | 2.1516e-8 | 0.2313 | B twin of v73_gb at Strassen leaf 64 (insurance B, A/B slot) | clean |
| 334464 | 10-06 17:28 | v75_kas | 4.1871e-9 | 2.2014e-8 | 0.1902 | v73 + decided-neuron pruning (R rows + D21 rows, fixed K): F −11.4 %, raw +2.2 % | clean |
| 334474 | 10-06 18:50 | v76_kasws | 4.4952e-9 | 2.2018e-8 | 0.2042 | v75 at block 40 (B twin) | clean |
| 334487 | 10-06 19:48 | v77_kash | 3.9596e-9 | 2.2021e-8 | 0.1798 | v75 + dead birth columns dropped (Hd): F −5.5 % | clean |
| 334545 | 10-07 00:16 | v78_kashws | 4.2496e-9 | 2.2023e-8 | 0.1930 | v77 at block 40 (B twin) | clean |
| 334576 | 10-07 03:44 | v79_dress | 3.7025e-9 | 2.0580e-8 | 0.1799 | v77 + dressed legs (κ3 birth families folded into the light leg) + twin refit: raw −6.5 % | clean |
| 334659 | 10-07 14:35 | v80_hyg | 3.6224e-9 | 2.0579e-8 | 0.1760 | v79 + layer-0 identity sandwich, join re-rotation via Strassen, extra Strassen level: F −2.16 %, raw 0 | clean |
| 334781 | 10-08 05:26 | v80_hygws | 3.7056e-9 | 2.0582e-8 | 0.1800 | v80 at block 40 (B twin) | clean |
| 334848 | 10-08 15:19 | v81_star | 3.4984e-9 | 1.9859e-8 | 0.1762 | v80 + star-D4 fourth-order births (learned per-layer gain) + refit: raw −3.5 % | clean |
| 334855 | 10-08 16:53 | v81b_star2 | 3.3939e-9 | 1.9266e-8 | 0.1762 | longer star-D4 refit (gains ≈ 1): raw −3.0 % | clean |
| 334918 | 10-09 00:27 | v81b_star2ws | 3.4711e-9 | 1.9265e-8 | 0.1802 | v81b at block 40 (B twin) | clean |
| 334926 | 10-09 01:51 | v81c_starall | 3.3642e-9 | 1.9084e-8 | 0.1763 | star-D4 over every lineage incl. old tier: raw −0.9 % | clean |
| 334948 | 10-09 04:43 | v81d_starall2 | 3.3200e-9 | 1.8833e-8 | 0.1763 | longer starall refit: raw −1.3 % | clean |
| 335165 | 10-10 10:49 | v82a_block | 3.2795e-9 | 1.8833e-8 | 0.1741 | `fnp.block` reassembly + `nan_to_num` guards, bit-identical: F −1.22 % | clean |
| 335227 | 10-10 16:22 | v82b_bundle | 3.0717e-9 | 1.8607e-8 | 0.1651 | join-skip L14/15, fused advance, 2-level sandwich, two-sided D21 columns + ρ·d21e: F −5.2 %, raw −1.2 % | clean |
| 335250 | 10-10 19:58 | v82b_bundlews | 3.1433e-9 | 1.8605e-8 | 0.1690 | v82b at block 40 (B twin) | clean |
| 335253 | 10-10 21:03 | v82c_r1 | 3.0464e-9 | 1.8454e-8 | 0.1651 | R1 2-epoch twin refit on the pruned chain: raw −0.82 % | clean |
| 335298 | 10-11 02:28 | v82c_r2 | 3.0323e-9 | 1.8368e-8 | 0.1651 | R2 refit constants: raw −0.46 % | clean |
| 335319 | 10-11 03:20 | v82c_r2ws | 3.1018e-9 | 1.8359e-8 | 0.1690 | B twin of v82c_r2 | clean |
| 335365 | 10-11 12:31 | v83_fold | 2.9483e-9 | 1.7811e-8 | 0.1655 | masked star-Q31 fold + R3 refit: raw −3.0 %, F +0.27 % | clean |
| 335407 | 10-11 18:32 | v83_r4od16 | 2.9191e-9 | 1.7755e-8 | 0.1644 | R4 refit + per-shape Strassen depth leaf ≥ 16: F −0.68 %, raw −0.32 % | clean |
| 335455 | 10-12 01:43 | v83_r5od16 | 2.9082e-9 | 1.7688e-8 | 0.1644 | R5 refit constants: raw −0.37 % | clean |
| 335461 | 10-12 02:30 | v83_r5ws | 2.9952e-9 | 1.7681e-8 | 0.1694 | B twin (fold + R5, block 40) | clean |
| 335484 | 10-12 04:37 | v83_r7od16 | 2.8942e-9 | 1.7603e-8 | 0.1644 | R7 refit constants (refit recipe stopped after two < 0.3 % restarts): raw −0.48 % | clean |
| 335562 | 10-12 14:29 | v84_age6 | 2.8498e-9 | 1.6422e-8 | 0.1735 | shared age 6 (M6) + R8 refit, od16 dropped for wall: raw −6.7 %, F +5.6 % (53/100 per-net wins) | clean |
| 335688 | 10-13 03:33 | v83_r7ws | 2.9808e-9 | 1.7595e-8 | 0.1694 | B twin, R7 constants | clean |
| 335697 | 10-13 04:30 | v85_sb2 | 2.7149e-9 | 1.6422e-8 | 0.1653 | Strassen zero-copy glue `sb2` (bit-identical): F −4.73 % | clean |
| 335769 | 10-13 11:49 | v86_d16 | 2.6887e-9 | 1.6425e-8 | 0.1637 | `sbtrim` + λ-priced per-site depth leaf ≥ 16: F −0.97 % | clean |
| 335809 | 10-13 15:05 | v87_B | 2.4169e-9 | 1.6759e-8 | 0.1442 | d12c6 depth + Lane K pruning stack B (k0+k1+k2+k3) + R1 refit: F −11.9 %, raw +2.0 % | clean |
| 335891 | 10-14 00:00 | v87_Bws | 2.4299e-9 | 1.6765e-8 | 0.1449 | B twin of v87_B | clean |
| 336083 | 10-14 19:17 | v89_r2 | 2.3806e-9 | 1.6699e-8 | 0.1426 | + sbfree + depth8 (1.2e6/6/8) + J3 (`o4`) + R2 refit: F −1.1 %, raw −0.4 % | clean |
| 336093 | 10-14 20:29 | v89_sbkst | FAILED (smoke) | — | — | sbks:trim atom; client `RemoteArray.transpose(*axes)` defect | FAILED on server |
| 336099 | 10-14 21:10 | v89_sbkst2 | 2.3404e-9 | 1.6698e-8 | 0.1402 | re-canary with transpose fix; sbks:trim: F −1.69 %, raw 0 | clean |
| 336107 | 10-14 21:30 | est_C_o7 | 2.3418e-9 | 1.6777e-8 | 0.1396 | Lane O7 stack C (k1 \|α\| ≤ 2.5) refit: F −2.1 %, raw +0.5 % vs v89_r2 | clean |
| 336124 | 10-14 23:27 | v90a | 2.3001e-9 | 1.6774e-8 | 0.1371 | stack C + sbks:trim | clean |
| 336142 | 10-15 01:10 | v90b_dw | 2.2936e-9 | 1.6771e-8 | 0.1368 | wall-aware per-site depth `depthw`: F −0.26 % | clean |
| 336147 | 10-15 01:56 | v90a_Bws | 2.3464e-9 | 1.6782e-8 | 0.1398 | B twin of v90a | clean |
| 336198 | 10-15 07:34 | o83b | 3.1367e-9 | 1.6666e-8 | 0.1882 | exact `svd(k=256)` old-tier join; **client billed economy SVD** (signature feature-detect defect) | clean (billing defect) |
| 336217 | 10-15 08:59 | o83c | 2.2782e-9 | 1.6666e-8 | 0.1367 | same predictions, `__module__` detection: raw −0.63 % vs v90b_dw, F −0.05 % | clean |
| 336253 | 10-15 13:58 | v92 | 2.2704e-9 | 1.6608e-8 | 0.1367 | + K8.1 dress gain D21ᵀ×0.9: raw −0.35 % | clean |
| 336282 | 10-15 17:13 | v93 | 2.2349e-9 | 1.6618e-8 | 0.1345 | + `pad:1024` λ7e5 + `opfix`: F −1.6 %; served residual 0.337 s | clean |
| 336312 | 10-15 19:43 | v93r | 2.2427e-9 | 1.6614e-8 | 0.1350 | depthw λ 1.1e6 for residual headroom: F +0.37 % | clean |
| 336358 | 10-16 01:14 | bfc727bc (B) | 2.3133e-9 | 1.6615e-8 | 0.1392 | B twin of v92 line (svd join + dg90 + opfix) | clean |
| 336413 | 10-16 05:07 | v94 | 2.1841e-9 | 1.6217e-8 | 0.1347 | K9.2 κ4 (3,1)/(2,1,1) lineage births in the Q31 fold + S10 7b covlive: raw −2.4 %, F −0.23 % | clean |
| 336438 | 10-16 07:01 | B2b | 2.2736e-9 | 1.6308e-8 | 0.1394 | no-svd B twin + K9.2 + 7b; served residual 0.3085 s > 0.30 B gate | clean |
| 336453 | 10-16 08:40 | v96 | 2.1558e-9 | 1.5928e-8 | 0.1353 | K9.4b joint κ4 refit (step 84) + 7b: raw −1.8 % (graded on flopscope 0.12.1) | clean |
| 336470 | 10-16 10:35 | v96b | 2.2057e-9 | 1.5810e-8 | 0.1395 | K9.4b final params; **grader moved to 0.12.3**: `svd(k=)` re-priced +3.08 % F | clean |
| 336495 | 10-16 12:51 | B2e | 2.2741e-9 | 1.5899e-8 | 0.1430 | v96b block-40 twin, no svd, λ 4e6 (current gate-clean older-chain B) | clean |
| 336508 | 10-16 13:43 | v97 | 2.1763e-9 | 1.5839e-8 | 0.1374 | v96c (d8b refit) no-svd join + 7b at λ 3e6: raw −0.56 % vs v96, F +1.5 % (svd removal) | clean |
| 336617 | 10-17 00:48 | f853 (young H1d + cleanup) | 2.0957e-9 | 1.5115e-8 | 0.1387 | partial young-tree κ4 reader (forum 18249 physics; matched 112-update ARM 3175) + 159-call cleanup: raw −4.6 %, F +0.9 % | clean (memory exception) |
| 336626 | 10-17 01:36 | 9fad (owned SKS) | 2.0763e-9 | 1.4975e-8 | 0.1387 | fd97 (CONTROL112 continuation, −0.85 % raw vs 66ae) + owned-operand Strassen: raw −0.9 % | clean |
| 336665 | 10-17 06:10 | f2b7 (selected369) | 2.0657e-9 | 1.4898e-8 | 0.1387 | one 369-update pass over TRAIN900+d8b2048: raw −0.5 % | clean (timing exception) |
| 336756 | 10-17 16:07 | hub84 (474e) | 2.0558e-9 | 1.4819e-8 | 0.1387 | gated hub-feature corrector leaf (O9.4/corr) refit 113 updates: raw −0.5 %, +0.06 % F | clean (memory exception) |
| 336783 | 10-17 19:42 | b636 (hub + local-even) | 2.0526e-9 | 1.4795e-8 | 0.1387 | + fixed terminal corrector-even symmetrisation: raw −0.16 %, +0.003 % F; served residual max 0.3455 s | clean (memory exception) |
| 336993 | 10-18 21:16 | 8be (A4 + R1) | **1.8962e-9** | **1.4536e-8** | **0.1305** | Claude patch A4 (C1 cuts + riding band r288 + O704 + planner + op diet: F −5.97 %) + faithful R1 refit with corrector LR ×10 (raw −1.75 %); graded 21:55 UTC (`docs/CURRENT_STATE.md` "October18 21:55UTC") | clean; ledger row still says `pending` |

Notes on the timeline
- 336993's grade (1.8962023807611765e-9 / raw 1.4535553773953325e-8, served F 286,868,542,979 → F/B 0.13045) is recorded in `docs/CURRENT_STATE.md` and `RESULTS.md` § Calibration, not in the ledger row (`experiments/submissions.jsonl` last row `result: pending`).
- Every graded A-line entry from v82a (Oct 10) to v97 (Oct 16) landed within −0.5 %…+1.0 % of its FRESH-A projection (`RESULTS.md` § Calibration; `docs/STRATEGIST_RESPONSE_2026-10-16.md` §0.1). The one outlier is #336198 (billing defect, `docs/STRATEGIST_PACKET_2026-10-15_laneO.md`).
- Cumulative, Oct 10 → Oct 18: score 3.2795e-9 → 1.8962e-9 (−42.2 %); raw 1.8833e-8 → 1.4536e-8 (−22.8 %); F/B 0.1741 → 0.1305 (−25.1 %) `[inferred from the table]`.

---

## 2. LEVER LEDGER (Sep 25 → Oct 19; exhaustive for Oct 10 → Oct 19)

Categories: cost / κ3 young / κ3 old / κ4 / closure / corrector-refit / planner-Strassen / data / infra-gate. Verdicts: shipped / dead / parked-unrefit / blocked-by-price / blocked-by-gate.

### 2a. Sep 25 – Oct 9 (summary rows; details in `RESULTS.md` §§ "Step …", "PLAN rev 15/17/18/19/20")

| lever | cat | what | measured ΔF / Δraw / Δscore (n, paired) | verdict | evidence |
|---|---|---|---|---|---|
| in-loop mean corrector (v57–v60, sequential fit, 400 nets) | corrector-refit | per-layer MLP 33→32→32→1 on per-neuron features, teacher-forced then in-loop | raw −5.60 % DEV, −4.76 % HOLDOUT-600 (ridge version failed at 2.1 % predicted) | shipped | `RESULTS.md` "Step 2 VERDICT" (Sep 24) |
| differentiable twin S1–S3 end-to-end gains | corrector-refit | λ/θ/feed gains, per-term gains, multi-mode κ4_iiij | raw −0.61 / −0.48 / −1.95 / −0.71 / −0.82 % per stage (DEV 100, 63–95/100 wins) | shipped v62–v67 | ledger rows 332313–332440 |
| Strassen-Winograd glue | planner-Strassen | 15 adds/level | F −0.79 %, raw 0 (100/100) | shipped v68 | row 332478 |
| twin S2t3 steps (TRAIN900+d8b) | data | more steps vs d8b data | gain is steps, not d8b (equal-step control) | shipped v70/v71 | `RESULTS.md` "PLAN rev 15 Lane A" |
| transport gates (G1tg) / birth gates (G1bg) | κ3 young | state-dependent gates on Wick transport / births | raw −0.42 % / −0.31 % (70/100, 55/100) | shipped v72/v73 | rows 332766, 332893 |
| harmonic (radial) old tier | κ3 old | Sym(u⊗C) replaces aged lineages | leaves ×6.6 of tier error (radial carries ~70 % of budget) | dead | `RESULTS.md` "Step 1 (HANDOFF 09-27)" |
| old-tier shared rank 256 (both directions) | κ3 old | rank sweep | 256 optimum; 224 +2.5 %, 192 +8 %, 128 +41 % raw | shipped (kept 256) | "Step 4(c)" ; O1 (Oct 13) |
| birth-layer-aware fold age / per-lineage adaptive age | κ3 young | | KILLED −0.72 % vs 2 % kill; "fuzzy fixed age" | dead | "Step 4(b)", "Step 3 (HANDOFF 09-24)" |
| calibrated learned closure (S5) | closure | per-neuron 4→16→16→4 + per-pair 9→16→2 nets after closure, fine-tuned | raw −0.75 % at F +1.68 % (score-negative); per-neuron net alone = control to 1e-5 | dead | `RESULTS.md` "Step 3.3" (Sep 27); `experiments/2026-09-29_closure/` |
| refit ceilings (label oracles) | ceiling | exact κ3 / cov / κ4 / all-but-μ substituted + refit | 53.4 / 40.5 / 27.9 / 91.4 % of MSE (HOLDOUT-20) | measurement | "Step 2 (HANDOFF 09-27)"; `experiments/2026-09-28_ceil/` |
| all-distinct κ3 (k111) oracle + rank screen | κ3 young | label AD at every layer; rank-r of E | −33.1 % (−20…−46 %, 20 nets); r=32 −18.9 %, r=128 −30.4 %; scalar factor gates capture 40 % of energy but **+13.6…+15.8 % unrefit** | measurement; gates dead | "PLAN rev 15 B1/B3a" |
| compact-history carrier B5 | κ3 old | | closed at the E2 retention bound | dead | "PLAN rev17 B5" |
| decided-neuron pruning v75/v77 (R rows, D21 rows, dead birth columns Hd) | cost | KAS | F −11.4 % + −5.5 %; raw +2.2 % + −0.1 % | shipped | rows 334464, 334487 |
| dressed legs (`_dress_H`) | κ3 young | AUGMENT κ3 birth families (3,1,1) folded into light leg | raw −6.5 % FRESH-A (85/100) + refit | shipped v79 | row 334576; physics in `docs/research/2026-10-15_exploration/gen/REPORT.md` |
| gain-mode κ4 templates (P2.2) | κ4 | D22/Q31 ∝ c²σ²C | β 0.7 ×7.8 blow-up; refit +3.5 %/+4.3 %; blend +0.01 % | dead | `RESULTS.md` "P2.2" |
| exact spurious-term (−w2) correction (P2.1b) | κ3 young | Lane M2 family | +0.34…+3.57 % unrefit (hurts in our chain, −41 % in ARC's) | dead | "P2.1b" |
| always-on D21 rows / alive-block covariance (P2.4.1/4.2) | cost | | Da3.0 +34 %, 2.5 +87 %; C=1 ×13 | dead | "P2.4" |
| hygiene (layer-0 WᵀW, join via `_mm`, extra Strassen level) | cost | | F −2.16 %, raw +0.02 % (100/100) | shipped v80 | row 334659 |
| star-D4 births (recent → all lineages) + refits | κ4 | fourth-order star on κ3 legs, learned gain | raw −3.5 / −3.0 / −0.9 / −1.3 % (97–100/100) | shipped v81–v81d | rows 334848–334948 |
| κ4-core loop (P3.2) | κ4 | off-diagonal r=1 core births + K211, memory Core ← WᵀCoreW | unrefit ×1.38–×4.36; old tier dropped ×20 with or without loop | dead | `RESULTS.md` "P3.2"; `docs/research/2026-10-08_laneM3_report.md` |
| M4 neighbour-feature corrector | corrector-refit | | never beat init (< 2 % kill) | dead | `docs/research/2026-10-06_laneM_report.md` §7 |
| M7 coherent early-layer errors | closure | | early layers not the lever (inside budget by ~8×) | dead | `docs/research/2026-10-07_laneM2_report.md` §M7 |

### 2b. Oct 10 → Oct 19 (exhaustive)

| lever | cat | what (one line) | ΔF | Δraw | Δscore / wins (panel) | verdict | evidence |
|---|---|---|---:|---:|---|---|---|
| v82a `fnp.block` + `nan_to_num` | cost | bit-identical reassembly | −1.22 % | 0 | graded −1.22 % | shipped | `RESULTS.md` P5.0c |
| P5.1 item 1 join-skip L14/15 | cost | skip old-tier joins at last two layers | −0.90 % | −1.14 % (DEV 0–9, 10/10) | | shipped (v82b) | P5.1 |
| item 2 last-layer rows K16 = 800 | cost | drop dead last-layer rows | −0.78 % | **+6.77 %** (9/9 worse) | | dead | P5.1 |
| item 3 old rebuild on D-kept rows | cost | | −1.26 % | **+10.63 %** (10/10) | | dead | P5.1 |
| item 4 fused `_advance` / item 5 two-level sandwich | cost | reassociation | −0.39 % / −0.23 % | 0 | | shipped | P5.1 |
| item 6 covariance input pruning (C=3/4) | cost | | −1.24 % | +24.2 % / +11.3 % (twin VAL) | | dead | P5.2 |
| V8b two-sided D21 columns + ρ·d21e | cost | D21 columns on the D set with per-layer ρ scalar | −3.59 % | −0.16 % (VAL, 38/100) | | shipped (v82b) | P5.2 |
| V8 full two-sided incl. leg output rows (Dz+Dleg) | cost | zero dead rows of transported legs | −8.84 % | **+54 % / +66 %** (100/100) | | dead ("dead rows' D3 is not negligible") | P5.2 |
| R1/R2 refits on pruned chain | corrector-refit | 2-epoch twin refits | 0 | −0.82 %, −0.46 % FRESH-A (99/100, 96/100) | | shipped | rows 335253, 335298 |
| masked star-Q31 fold (M5.1) + R3 | κ4 | star lineage's Q31 folded into D21 readout left factors | +0.27 % | −3.04 % graded (FRESH −2.84 %, 86/100); R3 −2.20 % vs matched ctl | | shipped v83 | P6.1 |
| Habs / Hasym / Hd column-count variants | cost | drop hub columns by α count | −3…−12 % | **+4.9 … +140 %** (VAL unrefit) | | dead ("P2 residual rows of always-on columns carry the chain") | P6.1 step 2 |
| od12 / od16 per-shape Strassen depth | planner-Strassen | | −0.68 % (od16) | +0.034 % | od12 fails wall (112.5 s) | od16 shipped v83_r4od16, later dropped for age 6 wall | P6.3 |
| R4–R7 fold refits | corrector-refit | restarts | 0 | −0.43 / −0.33 / −0.26 / −0.22 % VAL | | shipped to R7, recipe stopped (two < 0.3 %) | P6 |
| age 6 (M6) | κ3 young | 6 exact ages instead of 5 | +5.55 % | −6.15 % VAL (matched ctl), −5.84 % FRESH-A | score −1.54 % graded, **53/100** per-net wins | shipped v84 (A only; B memory FAIL) | P6.2 |
| `sb2` zero-copy Strassen glue (S1) | planner-Strassen | pooled `out=` views, bit-identical | −4.73 % | 0 | graded −4.73 % | shipped v85 | P7.1; `docs/STRATEGIST_PACKET_2026-10-13_laneS.md` |
| `sbtrim` + `depth16` (S2) | planner-Strassen | ops −1,025; λ-priced per-site depth | −0.97 % | +0.02 % | graded −0.97 % | shipped v86 | P7.2 |
| `depth8` (leaf 8) on A | planner-Strassen | | −3.44 % vs v85 | | wall 1.31× sb2 ⇒ ~110 s FAIL; d12c6 (1.07×) shipped | partly shipped | P7.2 |
| K stack A (k0+k1 −2.5..3.0 + k2 τ3.0) | cost | gain-mode fill on dropped rows | −8.8 % | unrefit +0.96 %, refit −0.06 % vs ctl | | superseded by B | `docs/STRATEGIST_PACKET_2026-10-13_laneK.md` |
| **K stack B (A + k3 two-column hubs)** | cost | | **−13.8 %** (vs v85) | unrefit +3.02 %, refit **+1.77 %** vs ctl; graded raw +2.0 % | graded **−10.1 %** vs v86 | shipped v87 | same; P7.3 |
| K2 τ 2.5 | cost | | −6.19 % | +6.47 % | fails | dead | laneK packet |
| K4 (drop old tier at last layer) | κ3 old | | | **+206…+226 %** (old lineages carry 2/3 of v at L15) | | dead | laneK packet |
| K5 per-net adaptive counts | cost | 16 materialisations/net | mean F +0.89 % | −1.42 % VAL, −1.62 % DEV, **−0.52 % FRESH-A** | net ≈ +0.4 % worse | dead (did not transfer to fresh nets) | P7.4 |
| O1 exact SVD join r256 / r224 / r192 / r128 | κ3 old | | +0.88 / −1.07 / … | −0.24 / +2.46 / +8.0 / +41 % | net +0.64 / +1.37 % | dead | `docs/STRATEGIST_PACKET_2026-10-13_laneO.md` |
| O2 split legs (A 192 / M 256) | κ3 old | | −0.45 % | +1.07 % | +0.62 % | dead (M carries the old tier's value: M at 128 +47 %) | same |
| O3(i) no old tier at A = 9 / 8 / 7 / 6 | κ3 old | | −1 % | ×4.0 / ×6.4 / ×10 / ×16 | | dead | same |
| O3(ii) harmonic tier on v84 | κ3 old | | | not built (needs 98 % recovery, 71 % measured) | | dead by arithmetic | same |
| O4 J3 (`o4`) | cost | | +0.04 % | −0.27 % (75/100) | −0.22 %, −431 ops | shipped v88/v89 | same |
| O4 age 7 / J4 / SVD join on pruned stack | κ3 young | | +3.75 / +0.88 / +1.51 % | −1.74 / −0.54 / −0.20 % | +1.94 / +0.33 / +1.30 % | dead (elasticity ≥ 1) | same |
| K7.1 mean-coupled covariance corrector (a/b/d) | closure | K11 += x yᵀ + y xᵀ, y = MLP(33 feats + D3res), x = m or σ² | ≈ 0 | **−0.22 %** VAL refit vs ctl (63/100) all three | | dead (< 0.5 % kill) | `docs/STRATEGIST_PACKET_2026-10-14_laneK.md` |
| K7.2 gain-subtracted D3 in D21·D3 monomials | closure | | 0 | +0.13 % refit | | dead | same |
| stack C (k1 \|α\| ≤ 2.5) refit (O7.1) | cost | | −2.09 % | +0.76 % VAL refit vs B ctl; FRESH-A −1.75 % score (92/100) | graded −1.63 % | shipped v90a | `docs/STRATEGIST_PACKET_2026-10-14_laneO.md` |
| `sbks:trim` (S8.2) | planner-Strassen | operand passes via strided slot views | −1.69 % | +0.073 % (39/100, t95 incl 0) | graded −1.69 % | shipped #336099 | `docs/STRATEGIST_PACKET_2026-10-14_laneS.md` |
| `depthw` wall-aware depth (S8) | planner-Strassen | | −0.26 % | −0.03 % (48/100) | | shipped v90b_dw | same |
| O7.3 per-band fold/star gains | κ4 | | | closed by inspection (already per-layer, trained) | | dead | laneO packet Oct 14 |
| O8.1 observability-weighted old tier (balanced truncation) | κ3 old | basis from right s.v. of Y_l A | | keep rule holds **0/10 cells**; exact observability basis ×1.07–1.7 worse than energy | | dead | `docs/STRATEGIST_PACKET_2026-10-15_laneO.md`; `experiments/2026-10-15_o8/O81_VERDICT.md` (lane-m) |
| O8.3 / O8.3b exact `svd(k=256)` join | κ3 old | exact top-256 basis, FJ = S Vᵀ free | −0.05 % | −0.63 % graded (FRESH −0.22…−0.25 %, 58/98) | graded −0.67 % | shipped v92… then **removed** (0.12.3 re-priced svd(k=) +3 % F; no-svd costs +0.41 % F, +0.31 % raw) | same; S10 packet "URGENT 0.12.3" |
| K8.0 dress coefficient on v90a | κ3 young | regress E3(v90a) on label dress term | | coef **−0.32** at l=14 (chain over-delivers ≈ 25 %) | | measurement | `docs/STRATEGIST_PACKET_2026-10-15_laneK.md` |
| K8.1 dress gain D21ᵀ ×0.9 (flat) | κ3 young | 15 scalars, 0 FLOPs | 0 | −0.38 % VAL (72/100), −0.37 % FRESH-A (66/100); ramps worse; refit arm "not worth a pod" | | shipped v92 (unrefit) | same |
| pad6 (zero-pad dims to 2^d, exact `_sks` pricing) | planner-Strassen | audit item 1 | −2.70 % | +0.01 % | wall **1.30× sb2 ⇒ 114 s FAIL** on A | dead on A | `docs/STRATEGIST_PACKET_2026-10-15_laneS.md` |
| `pad:1024` + `opfix` | planner-Strassen | pad only when k ≤ 1024; −1,736 ops | −1.69 % | 0 (49/100) | graded −1.6 %; residual 0.337 s ⇒ v93r λ 1.1e6 (+0.37 % F) | shipped v93/v93r | same |
| pad7 (cap 7) / S4 (×64 tables) | planner-Strassen | | −3.30 % / −1.54 % | | wall 1.63× ⇒ 143 s | dead | same; laneS Oct 13 |
| full pad (`_SW_PAD_KMAX=0`) on A | planner-Strassen | | **−1.04 %** | 0 (exact) | quiet-box wall 1.192× v93r ⇒ served ≈ 97 s (91–102); fails 105/112 s rules | **parked (never canaried)** | `docs/STRATEGIST_PACKET_2026-10-16_laneS.md` S10.4; `docs/STRATEGIST_ADDENDUM_2026-10-18_a.md` §2.3 |
| 7b covlive (covariance sandwich over live rows) | cost | Λ = α > −3.5 | −0.49 % | −0.06 % DEV (90/100 score wins) | | shipped v94 | S10.2 |
| 7a H := A (drop M-only on-hub columns) | cost | | −3.17 % | **+47…+65 %** (3 VAL nets) | | dead | S10.3 |
| K9.1 Tier 1 (kai Q31 slot + D4 local law, star-D4 dropped) | κ4 | label-fitted c₀ on exact-input transport | ≈ −0.3 u | **+0.25…+27.7 %** VAL unrefit | | dead (spec defect: c₀ fitted on exact-input dG) | `docs/STRATEGIST_PACKET_2026-10-16_laneK.md` K9.1 |
| **K9.2 fold births (3,1)+(2,1,1) in the Q31 fold** | κ4 | pure addition on shipped slots, g = 1 | +0.26 % | **−1.89 %** VAL (83/100), FRESH-A 1.0191× (87/100) | graded −2.6 % (v94 vs v93r) | shipped v94 | K9.2 |
| K9.3 D4 births over all lineages | κ4 | | +0.7 u | +24 % unrefit | | → refit leaves | K9.3 |
| **K9.4b joint κ4 refit initialised at the full κ4 model** | κ4 | corrector free, κ4 leaves ×10 lr | ≈ 0 | **−4.21 %** VAL vs matched ctl (ctl −0.15 %); FRESH-A 1.0144× vs v94 (69/100) | graded −1.3 % (v96 vs v94, 0.12.1) | shipped v96/v96b | K9.4b |
| K9.4 identity-init refit | κ4 | | | trailed K9.4b at every read (−2.62 % vs −3.69…−4.0 %) | | dead | K9.4 |
| d8b data (2,948 nets) / lr 2e-4 / both | data | | 0 | −0.36 / −0.36 / **−0.73 %** VAL further | | shipped v96c/v96d → v97 | K9 closing |
| param soups (d8b+lr2, all three) | corrector-refit | | | +0.11 / +0.23 % worse | | dead | same |
| K10.0 orientation audit | κ4 | | | 27/27 clean, no wrong-orientation term | | measurement | `docs/STRATEGIST_ADDENDUM_2026-10-17_a.md` §0.3 |
| K10.1 dG consistency (γ22/γh/γp) + κ3² D22 law + hub leaf, matched refit (coupled 3178) | κ4 | | | fixed g22 = 0.5 **+8…+12.6 %**; refit **−0.10 %** (60/100) | | dead | same; `RESULTS.md` "coupled fixed-physics screen", "Coupled completed CONTROL and ARM 3178" |
| O9.3 corrector-off oracles on v93r | ceiling | exact Q31 / dG / κ4 / κ3 / κ3+κ4 substituted, rolled, unrefit | | corrector OFF: κ3+κ4 **0.072×**, κ4 0.564×, κ3 0.596×, dG/D4 0.797×, Q31 −10.5 %; corrector ON: κ3+κ4 1.012×, κ3 1.363×, D21 1.636× | | measurement | `docs/STRATEGIST_ADDENDUM_2026-10-16_a.md` §0.3; lane-m `o93_table_corroff.md` (not in snapshot) |
| O9.4 / corr gated hub features in corrector | corrector-refit | Σρ², Σρ⁴, (ρ∘ρ)Φ, (ρ∘ρ)φ × sensitivity profiles | +0.07 u | −0.71 % VAL screen, FRESH-A 1.0036× on v94; as "hub" refit on v98: VAL −0.76 %, FRESH-A +0.80 %/+0.84 % vs v97; hub84 on f2b7 +0.32 % (70/100) | graded −0.48 % (hub84 vs f2b7) | shipped #336756 | `docs/research/2026-10-16_round3/corr/REPORT.md`; `RESULTS.md` Oct 16 18:40–21:16, Oct 17 07:27–07:43 |
| round-3 `corr`: non-local corrector features (Puffi, cov mat-vecs, mliston messages, error-shaped transports) | corrector-refit | | ≈ 1.3 u | +0.016 R² pooled ⇒ ≈ −1.0 ± 0.35 % realised; variance-channel corrector R² ≤ 0.031 even with oracle input | | dead as a −10 % lever | corr/REPORT.md |
| round-3 `k43` transported κ4→κ3 feed | κ3 young | hub-double, (3,1), 3-star on κ3 legs | exact ≈ 115–170 u; surrogate ≈ 2 u/layer | explains 25 % of post-dress residual effect at L 6–8, 3 % at L 14; surrogate 0–2 % of real v90a residual | | dead for deadline; exact build post-deadline | `docs/research/2026-10-16_round3/k43/REPORT.md` |
| round-3 `cost2` low-rank spokes at birth (r 128/256/512) | cost | | −28 / −16 / 0 u | per-lineage readout error 27–41 % at r256; r512 **+154…+162 %** | | dead | `docs/research/2026-10-16_round3/cost2/REPORT.md` |
| cost2: −32 lowest-weight A-hubs/lineage | cost | | −1.2 % | +10…+15 % | | dead | same |
| cost2: age 5 @ r256; age 6 @ r224/192 | cost | | −4.5 % / −1.3 / −3.7 % | +6.15 / +2.5 / +8 % | | dead | same |
| cost2: Gram/`svd(k)`/rectangular ⟨3,3,6;40⟩ schemes | cost | | ≤ −3 u | wall ×1.5–2, +150–300 ops/product | | dead | same; `RESULTS.md` "October 17 — exact rectangular multiplication cost screen" (1.67–2.80 % F for +17k…+32k calls) |
| H1a / H1c / H1d exact contraction/closure cost cuts | cost | pK term fusion, fused old/recent contraction, shared Wick/closure | −0.0387 / −0.339 / 0 % ; −533 / −8 / −271 ops | H1a raw −0.38/−0.48 % (2 nets, reassociation); H1adc FRESH-A +0.008 % raw (46/100), score +0.39 % | H1a accepted +0.36 %; H1c inactive on b636 | H1a/H1d shipped (hub, f853 line) | `RESULTS.md` "Final push — exact contraction/closure cost cuts", Oct 16 21:00 |
| **partial young-tree κ4 reader (forum 18249) — H1d 66ae / f853** | κ4 | youngest-lineage fourth-order reader riding the κ3 legs (distinct from the later Claude U1 'tp2' reader); matched 112-update ARM | +0.95 % F (305.0 vs 302.1 GF) | **−3.53 %** VAL vs matched CONTROL (94/100); FRESH-A score **+3.15 %** vs v97 (92/100) | graded −3.70 % vs v97 (f853) | shipped #336617 | `RESULTS.md` Oct 16 21:00 "Partial young-tree selected ARM /3175", 22:00 checkpoint |
| full retained-tree D4 half (3179) | κ4 | all-ages young reader | **+68 %** F (509 GF / 636 GF paired) | −2.5 % / −5.5 % (1 net) | break-even 40–52 % | blocked-by-price | `RESULTS.md` Oct 16 21:00 |
| exact-J / covariance-X D4 residual half (3184/3187) | κ4 | | +7.1 % / +1.0 % F | **+7.7 % / +7.3 %** raw (DEV0) | | dead | Oct 16 21:16 |
| LP-only Q31 (3213) | κ4 | | +0.10 % F, +528 calls | score **+0.11 % worse** at g 0.05/0.1 (TRAIN8–15, 1/8) | | dead | Oct 16 22:34, 22:54 |
| covariance proxy on young (ARM 3210 / 3223) | closure | | ≈ +0.07 % F | −0.116 % VAL (71/100) / −0.11 % (66/100) | | dead | Oct 16 23:20; Oct 17 00:28 |
| terminal scalar replay (3194) / frozen terminal head (3140) | corrector-refit | | 0 | VAL −0.002 % / −0.26 % (72/100) | | dead / "small optional polish" | Oct 16 18:40, 21:16 |
| onepass128 P2 r4 (3168) / residual source sampling m64 (3132) / rank128 spoke repair (3128) / CP4 residual r32 (3108) / fourth-core cross-slice (3113) | κ3 young (compression) | fixed-producer compressions | −16.7 % F (onepass128) | D3 perturbation energy 9–39 % of MSE; all near-preservation checks fail | | dead | Oct 16 17:15, 18:30, 20:17 |
| CONTROL continuation 3201 (cloud) / 3215 (local) → fd97 | corrector-refit | 112 more updates, corrector free | −0.035 % F (cleanup) | −0.45 % / −0.48 % VAL (81/100); FRESH-A **+4.04 %** score vs v97 (94/100) | graded 9fad −0.93 % vs f853 | shipped #336626 | Oct 16 22:28, 23:36, 23:55 |
| selected369 one pass TRAIN900+d8b2048 (3263) → f2b7 | data | 369 updates | 0 | −0.28 % VAL (79/100); FRESH-A +0.336 % (71/100) | graded −0.51 % | shipped #336665 | Oct 17 05:00, 06:46 |
| broad mean-only continuation (3254) | corrector-refit | | 0 | −0.037 % (50/100) | | dead | Oct 17 03:12 |
| nonlinear shadow8 recurrent terminal (3276) | corrector-refit | 7,305-param recurrent vs 273 terminal | | −0.365 % raw (67/100), forecast score −0.17 % | | dead (< 1 %) | Oct 17 05:28 |
| coherent/covariance Gaussian k3/k4 completion (3234) | closure | | +189–441 MF | +0.18 % (6/8 TRAIN) | | parked | Oct 17 (3234) |
| covariance query heads from hidden33 (3242 / 3252 / 3253 / 3255 / 3256 / 3259 / 3314) | closure | learned next-variance query maps from the chain's hidden features | — | CHECK query error −0.34 % / −0.19 % / −0.92 % / −1.67 % / +0.0004 % / **+5.6 %** / +0.02 % (TRAIN0–3 fit, CHECK4–7) | all fail 10 % gate | dead/parked (proxy metric, never rolled out) | `RESULTS.md` Oct 17 01:29–03:20, 08:43 |
| shared-Q recent-M projection after six joins (3257) | cost | | | **×8.04** (0/8) | | dead | Oct 17 03:08 |
| youngest sb mixed-six / late first-four Q22 fixed units (3264/3265/3266) | κ4 | | forecast +7.7 % / +2.0 % F | raw −0.02 % / +0.67 % | score −8.3 % / −2.5 % (0/8) | dead | Oct 17 03:58–04:00 |
| query-protected compression (3283) / Dx64 (3286) / rank64 diag restoration (3284) | κ3 young (compression) | | | +3.2 % raw (1/8) / next-variance ×8.84 / restored 13× native | | dead | Oct 17 05:54–06:26 |
| sampled-center 6000 replacement (3293/3299) | sampling hybrid | antithetic Price2 centres + 512-update fit | — | clipped **2.57e-6** vs native 1.54e-8 (0/64) | | dead | Oct 17 07:34 |
| pair covariance 77/609 heads (3301 / 3331 / 3346) | closure | | | query +4.6 % held; terminal fits CHECK **worse** 0.046 % / 0.099 % (18/36, 11/36) | | dead | Oct 17 07:34, 13:53 |
| full sample mean/cov Price2 tail; K16 Gaussian tail; Hermite 3–16; post-ReLU exact C1 whitening (3310/3312/3313/3315) | sampling hybrid | | | variance 7.3e-7…2.4e-7 vs required 5.9e-9 (41–127× short) | | dead | Oct 17 08:11–08:49 |
| corrector capacity h32 + residual64 (3317) | corrector-refit | 33→64→64→1 residual on current X33 | | residual adds **−0.0002 %** (49/100); h32 continuation itself −0.46 % | | dead (capacity) | Oct 17 09:34 |
| h32 causal rollout (3318) | corrector-refit | | | **+7.46 %** VAL (14/100) | | dead | Oct 17 09:51 |
| local scale conjugation (3320) | κ4 (scale mode) | 18248 scale extraction, corrector off | | variance −0.16 %, mean +0.18 % (6/8) | | dead | Oct 17 10:00 |
| deep-only age 7 corrector-off (3330) | κ3 young | layers ≥ 10 | +1.51 % break-even | **+1.08 %** corrector-off (7/8 worse) | | dead (rank 320–448 / +32 hubs **never screened**) | Oct 17 10:56; `STRATEGIST_ADDENDUM_2026-10-18_a.md` §0.3 |
| local covariance forcing factorial (3326) / consumer factorial (3228) / partition (3232) / 3481 / 3476 | ceiling | oracle substitutions | — | see §3/§4 | | measurement | Oct 17 00:28, 10:37; Oct 19 02:48, 04:33 |
| h32/residual64 causal 226 (3327) | corrector-refit | | | +0.040 % / +0.040 % (71/68 of 100) | | parked | Oct 17 11:55 |
| output-step / output-unit causal 64 (3335/3339) | corrector-refit | | | CHECK +0.23 % / +0.30 % worse | | dead | Oct 17 11:55, 12:49 |
| terminal rank16 fixed/learned (3338) | corrector-refit | | | CHECK −0.04 % / −0.05 % (17/36, 14/36) | | dead | Oct 17 12:22 |
| fourth-core terminal diagnostic (3343) | κ4 | complete same-live off(B−λC) terminal dG residual | | **+1.42 %** ON, +3.86 % OFF (0/2) | | dead on 2 nets | Oct 17 13:09 |
| pooled incoming-mean heads (3348) / target-message 561 (3350) | corrector-refit | | | CHECK +0.045 % / +0.024 % / +0.007 % / −0.005 % (19–21/36) | | dead | Oct 17 14:20, 14:53 |
| literal final-weight reflection (3351) | closure | full terminal half-projection | +4.10 % | +0.09 % (1/2) | | parked | Oct 17 15:09 |
| **local corrector-even (3353 → fabb → b636)** | corrector-refit | final-only p₊ + ½(cd_R − cd₊) | +0.0029 % | +0.107 % clipped (67/100 TRAIN), FRESH-A +0.097 % (63/100); composed with hub +0.11 % (67/100) | graded −0.16 % (b636 vs hub84) | shipped #336783 | Oct 17 15:17–17:27 |
| all-16-layer local-even (3362) / terminal sensitivity O set (3373/3374) | corrector-refit | | +0.056 % / 0 | +0.16 % (1/2) / +0.074 % (63/100, t95 crosses 0) | | dead / parked | Oct 17 16:40, 19:17 |
| distinct double-third birth T² (3359) | κ3 young | | forecast +9.1 % | −0.00005 % | | dead | Oct 17 16:12 |
| scalar conditional-law centred defect (3364) | closure | | +0.41 % | **+16.3 %** (0/2) | | dead | Oct 17 17:12 |
| input-JVP energy features N128 (3369/3370) | corrector-refit | | +2.8 % F | CHECK +0.067 % worse | | dead | Oct 17 18:07 |
| consumer-aware rank128 / actual-signal RIGHT64 compression (3372/3375) | κ3 young (compression) | | −8.6 % / −11.3 % F | **×117.7 / ×13.2** | | dead | Oct 17 18:51, 19:33 |
| **A32 (late A-only retention) (3378/3379)** | cost | keep A-leg of old-age lineages | −0.155 % score at | +0.26 % raw (66/100) → cost gain 0.155 %, CI crosses 0 | | parked (NO_GO fixed recipe) | Oct 17 20:25 |
| shared-join diagonal repair (3380) | κ3 old | | +4 % F | +0.042 % (0/2) | | dead | Oct 17 20:52 |
| terminal antithetic CV (3381) | sampling hybrid | | ~2× work | variance 29.5× native MSE | | dead | Oct 17 20:52 |
| optimal rank64 Dx observer (3382) | κ3 young | | | Dx loss 14–59 %; D21 perturbation 11–29 % | | measurement (no cheap rank-64) | Oct 17 21:09 |
| coherent-scale extraction (18248) full-support f64 reference (3384/3386/3387) | κ4 (scale mode) | extraction 0.5 vs untreated, correctors off | 11–14 TF (not SDK bill) | **−27 % / −33 %** (2 / 32 nets); window0 8.6 % worse than full | | measurement: mechanism real, **cost prohibitive**; no cheap carrier found | Oct 17 21:38, 22:02, 22:15 |
| first-layer-centre ridge A35 (3385) / invariant centre feature (3383) | corrector-refit | | +33 MF | +0.053 % worse | | dead | Oct 17 21:25–21:38 |
| native scale projection hub (3389/3391) | κ4 (scale mode) | | | **×3.28** (0/32) | | dead | Oct 17 22:34, 23:19 |
| coherent teacher supervision (3390) | corrector-refit | teacher-final64 vs physical | | −1.34 % / −0.78 % CHECK (1/8, 2/8) | | dead | Oct 17 23:13 |
| known-input χ first-post inversion (3394) | κ4 (scale mode) | | | −10.2 % vs untreated, +34.6 % vs extracted | | dead | Oct 17 23:30 |
| native epsilon feature / FIT256 proof (3393/3395/3397) | κ4 (scale mode) | | +412 MF | checkpoint 366/365 mismatch; **Samuel stopped the route** | | dead (infra) | Oct 17 23:52; Oct 18 00:11 |
| signed post-K21 L14 message (3396 → 3493 → **3495**) | closure | learned 1024-weight message from K21@h1 into terminal readout; matched vs local H1 control | +0.025 % F | CHECK raw **+0.21 % / +0.08 % worse** (14/36, 18/36); FIT −0.33 / −0.36 % | | dead | `experiments/2026-10-19_push/REVIEW_AND_DECISION.txt`; `RESULTS.md` Oct 19 07:51 |
| **C1 cuts (STAR0 + GQ1 + SJG + LL704)** (Claude patch) | cost | star-D4 gain 0, Q31 gain 1, Sj Gram alias, last-layer 704 rows | −1.09 % | −0.19 % ± 0.07 FRESH-D (18/32) | | SJG **parity FAIL** (9.3e-6, not exact); O704 alone −0.59 % F, raw −0.076 % (FRESH 3412: score −0.70 %, 100/100) | partly shipped (8be) | `experiments/2026-10-18_p12/claude_handoff/README.md`; `RESULTS.md` Oct 18 00:33, 03:21 |
| **riding band r288 / r320** (`_RIDE`, FJ = R⁻¹QᵀJ onto carried Q) | κ3 old | old-tier join rides the carried basis, higher rank at lower cost | −2.59 % / −0.81 % | −0.076 % / **−1.29 %** VAL | score −2.66 % / −2.10 % (92/100, 86/100) | r288 shipped (8be); r320 parked ("do not reject after a matched refit") | `RESULTS.md` Oct 18 03:21; A6 canonical VAL −5.47 % score vs b636 but 0.66 % worse than A4 |
| op diet + OPF 1e6 planner (A4 pB288) | planner-Strassen | ≈ −1,250 ops spent on λ | −5.97 % total vs b636 | raw −0.13 % (57/100, CI incl 0) | canonical VAL score **−6.10 %** vs b636 (99/100) | shipped 8be | Oct 18 08:06 |
| alternate fixed planner (3448) | planner-Strassen | OP 2.5e6, fewer ops | +1.15 % vs A4 | +0.004 % | score **+1.15 % worse** vs A4 (0/100) | dead | Oct 18 12:36 |
| combined op diet SjGram (3407) | cost | | −0.0002 % F, −620 calls | parity FAIL 1.28e-5 | | dead (not exact) | Oct 18 00:53 |
| **R0 / R1 faithful b636 refits (corrector LR ×1 vs ×10)** | corrector-refit | 368 updates TRAIN900+d8b2048 | 0 | R0 −0.69 % (dev) / −0.71 % canonical (82/100); **R1 −1.53 % dev / −1.52 % canonical (90/100)**; R1 vs R0 −0.815 % (84/100) | | R1 shipped (8be) | Oct 18 05:11, 09:41, 13:45 |
| U1 tp2 reader + mean-field older ages + var² D4 face (Claude P1/P2), matched refit (3431) | κ4 | 9 NEW groups, NEW lr .001 | ≈ +0.15 % F | **−0.300 %** VAL (77/100, [0.22, 0.38]) | misses 1.5 % gate | blocked-by-gate (positive, unused) | Oct 18 10:03 |
| U1 NEW10 (NEW lr .01) (3472) | κ4 | | ≈ +0.15 % F | **−0.916 %** VAL (81/100, [0.69, 1.14]) | misses 1.5 % gate | blocked-by-gate (positive, unused) | Oct 19 00:25; `docs/CURRENT_STATE.md` |
| U2b nested-Dx reader ages 2–3, L ≥ 9 (Claude P3), matched refit (3444) | κ4 | | ≈ +3.4 % F (plan) | **−1.24 %** VAL (100/100) | misses 5 % gate; net score ≈ +2.1 % worse at +3.4 % F `[inferred]` | blocked-by-gate/price | Oct 18 21:15 |
| 3476 literal-8be coupled consumer anatomy | ceiling | C / T / F / ALL substitutions | — | see §3 | | measurement | `experiments/2026-10-19_research_lead/research_decision.txt` |
| 3477 operation census | cost | | `_advance` 36.3 %, `_yp_contract` 37.1 % of 286.9 GF; corrector 0.118 % | | | measurement | same |
| exact-cost reuse intake / common-left Winograd transform reuse | cost | | 0 proved / **−0.0078 %** (22.5 MF, 29 calls) | | | dead (negligible) | `research_lead/exact_cost_review/REVIEW.txt`; Oct 19 05:45 |
| damped all-consumer CQ transfer (3478) | closure | | +cost | **+0.59 %** raw (0/2) | | dead | Oct 19 04:18 |
| **K1 minus 32 source rows** (3479/3483) | cost | one table diff | **−1.139 %** F, −58 calls | +1.72 % raw (TRAIN0–7) | score **+0.57 % worse** (2/8, t95 crosses 0) | dead | Oct 19 04:19, 04:33 |
| exact first-layer Gaussian pK11/21/22 (3484) | closure | Cho–Saul identities replace fitted first-layer pair moments | +54.5 MF | **+1.11 %** raw (1/8) | score +1.15 % worse | dead | Oct 19 04:47 |
| primary-CDF production law (3485/3486) | numerics | | 0 | canonical-gap MSE −6.2 % / +2.2 % (2 nets); truth score +0.012 % worse | | dead | Oct 19 05:08 |
| frozen-objective auxiliary-ascent diagnosis (3491) | optimizer | total vs final-only Adam direction | — | both descend; T magnitude 0.0096 % weaker | | hypothesis closed | Oct 19 05:35 |
| one composite fresh-Adam step at scale 1 (3492) | optimizer | | 0 | **+9.9 % / +14.6 %** (0/8, 0/8) | | dead | Oct 19 06:15 |
| quadratic64 statistical control (3496–3498) | sampling hybrid | 64 exact-mean quadratic controls vs linear | +2.5 % F forecast | variance proxy **+8.42 % / +8.29 %** worse; both 2,367–2,537× native score | | dead | Oct 19 09:00 |
| directional subspace FIT64 candidate (3499/3500/3502) | optimizer | 8 saved per-network directions through the final Jacobian; SLSQP | — | 3500 solver status 8 (no endpoint); 3502 continuation mean −4.85e-12, **18/36 wins** | | dead | Oct 19 09:44–11:24 |
| two-mode input Krylov scalar probe (3504/3505) | closure | first-layer-exact mixed κ3/4/5 scalar | 2.33 GF forecast → 3.1 GF | runtime API failure (`fnp.all` absent); repaired probe RUNNING at 12:21 | needs 1.07 % raw gain at same cost | **no result** | `docs/CURRENT_STATE.md` 11:49–12:21 |
| conditional Gaussian line integration (deep) | closure | | — | source-only review: STOP (no affordable producer) | | not built | `REVIEW_AND_DECISION.txt` 09:06 |
| corpus-14k (12,000 extra labelled nets) | data | | — | never used ("No 14k corpus without Samuel's shards") | | **not started** | `docs/CURRENT_STATE.md`; `STRATEGIST_ADDENDUM_2026-10-18_a.md` §0.3 |
| long final refit R3 (736 updates, fresh Adam, cosine) | corrector-refit | | 0 | held; start Oct 20 06:00 UTC, step0 fallback | | pending | `docs/CURRENT_STATE.md` "R3 and cloud readiness" |
| B-young (block-40 twin of b636/8be) | infra-gate | diversified backup | | not built; B remains B2e #336495 / b636 | | not started | `STRATEGIST_ADDENDUM_2026-10-18_a.md` §2.1 |

---

## 3. CEILING / INFORMATION EXPERIMENTS

What was measured about what a learner could recover, with the inputs each used.

| date | experiment | inputs to the learner / oracle | result | path |
|---|---|---|---|---|
| Sep 24 | single-feature nonparametric ceiling on one-step residual | one per-neuron feature (α, D3/σ³, D4/σ⁴, ‖W‖) | 0.04–0.7 % of one-step residual; four axes ≈ 1.8 % | `RESULTS.md` "1e. The ceiling…"; `tools/corr_ceiling.py` |
| Sep 24 | ridge / MLP ceiling of a per-neuron corrector (`tools/corr_ceiling.py`, `tools/corr_mlp_ceiling.py`) | 33 per-neuron chain features (9 channels + squares/products), FIT-100, net-wise CV | ridge R² 0.02–0.31 by layer ⇒ 2.13 % predicted raw; MLP 32×32 R² 0.06–0.42 ⇒ **7.32 %** (realised in-loop −5.6 %; in-sample/held-out gap 0.0005 ⇒ feature-limited, not data-limited) | `RESULTS.md` "Step 2 priced BEFORE it is built" |
| Sep 24 | propagation identity | — | A_l = 0.60–0.88 (ReLU layer contracts incoming mean error); M_16 reconstructed to −0.00 % | same, §2a |
| Sep 27 | **refit ceilings** (label oracles + refit of all S3 leaves) | exact κ3 / cov / κ4 / all-but-μ substituted at every layer (N = 1e9 labels) | κ3 53.4 %, cov 40.5 %, κ4 27.9 % (7.6 % unrefit), all-but-μ 91.4 % of MSE (HOLDOUT-20); control refit never beat init | `RESULTS.md` "Step 2 (HANDOFF 09-27)"; `experiments/2026-09-28_ceil/` |
| Sep 27 | closure bench + regression | closure's own normalised per-neuron / per-pair inputs (exact labels) | MLP 16-16 held-out R² K1 0.63, K2 0.85, K3 0.87, K4 0.85, K11 0.68, K21 0.75 (ridge 0.03–0.61); embedded + fine-tuned: −0.75 % raw, score-negative | `RESULTS.md` "Step 3.1–3.3"; `tools/closure_regress.py`, `tools/closure_fit.py` |
| Sep 28 | k111 all-distinct κ3 oracle; rank screen; per-neuron factor gates | label AD tensor; rank-r truncation of E; LS gates on (1, α, α², γ3, Φ) | AD → label −33.1 %; r=32 −18.9 %, r=128 −30.4 %; gates capture 40 % of energy but unrefit **+13.6…+15.8 %** | `RESULTS.md` "PLAN rev 15 B1/B3a" |
| Oct 13 | O5 anatomy on v84 | exact-label single-object substitution, rolled | cov 42.7 %, κ3 22.0 %, κ4 8.7 %, old tier 1.9 % of chain MSE 1.618e-8 | `docs/STRATEGIST_PACKET_2026-10-13_laneO.md` |
| Oct 14 | κ4 forms / closure with exact inputs | exact pre-cumulants, N=1e9 post moments | chain's D22/Q31/D4 surrogates at the ceiling of O(n²) forms; first-order closure variance error 1.2e-5 (L1) → 1.07e-4 (L14), ≥ 97 % pair-sourced; one term (D21·D3, coef 1.00) removes 21–31 % and is already shipped at gain 0.92 | `docs/STRATEGIST_RESPONSE_2026-10-14.md` §3; `docs/research/2026-10-14_closure/` |
| Oct 15 | physics H1/H2 | injected errors into exact chain | (μ,C) closure truncation 3.2–4.9e-10 = 2–3 % of MSE; propagation gains 0.80–1.02; model's own D3 pattern 6.7–9.3e-9 ≈ 50 % of MSE; D4 +10 % ⇒ 14–22 %; Q31/D22 +10 % ⇒ ≤ 4 % | `docs/research/2026-10-15_exploration/physics/REPORT.md` |
| Oct 15 | learn (v55 dump, FIT-100) | per-neuron local features (21), + hub features (16 row sums), C-eigen subspace, rank-few, diagonal W³ transport | coherent scale a_l−1 = −0.0…−0.4 % (nothing to calibrate); sensitivity-weighted D3 error OOS R² 0.00/0.09/0.14 local, ≤ 0.21 with hubs; C-eigenbasis keeps 1–3 %; oracle rank-32 of R keeps 47 %; inheritance R² ≈ 0 | `…/learn/REPORT.md` |
| Oct 15 | basis | C eigvecs / legs / propagator / feature span / random; learned producer of n×64 coordinates from local features | two-sided ≤ 0.07 of E's effect at k=128; one-sided (c-index) C-top-64 keeps 0.53–0.68; producer OOS R² ≤ 0.34 | `…/basis/REPORT.md` |
| Oct 15 | gen | 18 closed-form newborn diagrams vs E3 | dress term (already in v79) R² 0.55–0.67; everything else ≤ 0.05 on residual | `…/gen/REPORT.md` |
| Oct 15 | K8.0/K8.2 on v90a | real chain E3; keenanpepper `pre_K3ad_PP` sketches; pair gates | exact AD oracle **−21.6 %** on v90a (−36.0 % on v62); rank-8/32 corrections +19/+20 % (worse), rank-128 −5.3 %; inherited scale 1.06, 85 % of error is shape; n⁴ newborn candidates ≤ 1.2 %; pair gates closed to noise by first-order κ3+κ4 with unit coefficients | `docs/STRATEGIST_PACKET_2026-10-15_laneK.md` |
| Oct 15/16 | round-3 k4 | exact κ4 slices injected; Gram fits on chain dump | shipped κ4 forms' isolated cost **1.48e-8** (≈ whole raw); Tier 1/1.5/2 would remove 34/42/50 % of it | `docs/research/2026-10-16_round3/k4/REPORT.md` |
| Oct 16 | round-3 corr | 33 local + Puffi neighbour (29) + cov mat-vecs + mliston messages + error-shaped transports; oracle diagonal transport of true previous variance error | pooled R² 0.081 → 0.097; 97.7–98.7 % of one-step mean error is first-order response to chain's own δσ/δD3/δD4; variance piece (49–51 % of energy at L12–16) R² ≤ 0.031 **even with oracle diagonal input** (≥ 97 % enters via off-diagonal sandwich) | `…/corr/REPORT.md` |
| Oct 16 | **O9.3 / O9.5 corrector-off oracles (v93r, v94)** | exact slices substituted in all uses, rolled, unrefit | corrector OFF: κ3+κ4 **0.072×** (1.42e-9 raw), κ4 0.564×, κ3 0.596×; corrector ON: κ3+κ4 1.012×, κ3 1.363×, D21 1.636× ⇒ the mean corrector (−16 %) is tuned to cancel the shipped slice errors | `docs/STRATEGIST_ADDENDUM_2026-10-16_a.md` §0.3 |
| Oct 17 | 3228 / 3232 (CONTROL 6a23, TRAIN0–7) | C / CF substitution; CV / CO / M / MC partitions | C −38.4 %, CF −43.1 % (8/8); CV −37.1 %, CO −15.7 %; M (premean reset) 3.25e-9 (−77 %) | `RESULTS.md` Oct 17 00:28 |
| Oct 17 | 3347 true incoming post15 mean | terminal-only oracle | 74–79 % of final MSE (2 nets) | Oct 17 14:09 |
| Oct 17 | 3330 saved post15 oracle variance response | linear response with label covariance | 9.7 % lower (100/100) | Oct 17 10:56 |
| Oct 17 | 3317 corrector capacity | current X33 + 33→64→64→1 residual | residual adds −0.0002 % | Oct 17 09:34 |
| Oct 19 | **3476 literal-8be coupled consumer anatomy** (TRAIN0/1) | C (pre-activation covariance), T (repeated-third consumers), F (repeated-fourth), ALL (+premean) | native 1.209e-8; C 7.42e-9 (−37.7/−39.3 %); T 2.40e-8 (worse); F 1.51e-8 (worse 14.8/33 %); CF 7.35e-9; CTF 2.21e-8 (worse 77/88 %); ALL 7.83e-10 (0.065×); ALL_off 3.47e-10; interactions CT +3.2e-9, CF −3.0e-9, TF −4.3e-9, CTF +3.9e-9 ⇒ "no additive oracle shares" | `experiments/2026-10-19_research_lead/research_decision.txt`; `anatomy/ANATOMY.txt` |
| Oct 19 | 3478 / 3481 local vs inherited covariance split | Gaussian-weighted next-variance query | inherited 8.45e-10 vs local 1.56e-11 of 8.66e-10 (inherited dominates) | `RESULTS.md` Oct 19 04:18, 04:33 |
| Oct 19 | 3491 objective diagnosis | frozen Adam direction | final-only and total objectives both descend on all 8 nets | Oct 19 05:35 |

**Twin / refit plateau (the "learned" ceiling of the fixed parameterisation):** D6 "more fitting adds NOTHING" (Sep 22); refit control never beat its init (Sep 27); R6/R7 < 0.3 % restarts (Oct 11–12); O9.2 matched control −0.15 % (Oct 16); K9.4b round 2 no gain; soups worse; mean-only continuation −0.037 %; d8b 369-update pass −0.28 %; CONTROL continuations −0.45 %/−0.48 %; U1 −0.30 %; NEW10 −0.92 %; U2b −1.24 %; R0 −0.69 %; R1 (corrector LR ×10) −1.52 %. Every refit that moved more than 1 % since Oct 10 changed the physics or the corrector's learning rate, not the training length (`RESULTS.md` P6 R3–R8; `docs/STRATEGIST_PACKET_2026-10-16_laneK.md` K9 closing; `RESULTS.md` Oct 18 05:11–13:45; `docs/CURRENT_STATE.md` Oct 19 00:35).

**Plain statement on the unconstrained regressor.** No experiment in the snapshot fits an unconstrained (FLOP-unlimited) regressor from the *full carried state at layer L−1* (μ, the full C matrix, every lineage leg and core, dG/λ, the old-tier basis) to the exact residual at layer L. Every learner that was fit took a reduced input: ≤ 33–48 per-neuron features (Sep 24 ridge/MLP, Oct 17 X33/h32/residual64), the closure's own per-neuron/per-pair scalars (Sep 27), 16 hub row-sums (Oct 15 learn), Puffi/mliston non-local summaries and cov mat-vecs (Oct 16 corr), fixed subspaces of C or Dx (rank 16–128; Oct 15 basis, Oct 17 3242–3259, 3283–3286), 77/609 per-pair features (3331/3346), or a single 1024-weight message from one K21 slice (3495). The largest-input fits were the 561-coefficient hidden-bilinear covariance head (3259: CHECK +5.6 %, NO-GO) and the 609-feature pair head (3346: worse). The nearest things to an unconstrained bound are label oracles, not regressors: exact-slice substitution with refit (91.4 % all-but-μ, Sep 27), corrector-off exact κ3+κ4 (0.072×, Oct 16), and the 3476 ALL arm (0.065×, Oct 19). The Oct 16 corr report explicitly states the variance half of the one-step error "is unreachable by any per-neuron corrector" and that it enters through the off-diagonal sandwich — the one input class no regressor was ever given in full. `[inferred: the gap is between "oracle labels" and "≤ 609 hand-chosen features"; nothing in between was tried.]`

---

## 4. ERROR BUDGET AS CURRENTLY KNOWN

Statements, by layer / order / index pattern, with measurement and date. Shares are of the *final-layer* MSE of the chain named.

| claim | number | chain / panel | measurement | date / path |
|---|---|---|---|---|
| Label floor at scored layer | ≈ 7.7e-11 organiser (N=1e9 ×2 draws), 4.9e-11–1.1e-10 over 20 FIT nets; **FRESH-A floor 7.48e-10** (N=1e8); same-layer TRAIN two-reference floors 8.0–8.5e-11 | — | MSE(aicrowd gt, keenanpepper mean)/2 | `AGENTS.md` Measurement; `RESULTS.md` Anatomy 2; `research_decision.txt` |
| Floor vs current raw | 0.53 % of 1.4536e-8 | 8be | arithmetic | `REVIEW_AND_DECISION.txt` |
| Layer-1 floor | 6.8e-10 (not 9e-11; "AGENTS was wrong by 7.6×" at the time) | — | | `RESULTS.md` 1a (Sep 24) |
| Propagation | A_l 0.60–0.88 (contracting) | v55 (Sep 24) vs per-layer gains δμ 1.25/1.09/1.03/1.02 then 0.98–1.02, δvar/δC_off 0.80→0.98 (Oct 15, exact chain) | identity decomposition vs injected-error rolls | `RESULTS.md` 2a; physics/REPORT.md |
| Refit ceilings (object shares) | κ3 53.4 %, cov 40.5 %, κ4 27.9 %, all-but-μ 91.4 % | v62+S3 (v67/v68 accuracy), HOLDOUT-20 | label substitution + refit | Sep 27 |
| Anatomy 2 shares | κ3 51 %, cov 42 %, κ4 15 %, all-but-μ 80 % | v59, FIT 0–19 | substitution, no refit | Sep 25 |
| O5 anatomy | cov 42.7 % (var L12–15 18.1 pts, C_off L7–11 12.2 pts), κ3 22.0 % (was 35.2 % before age 6), κ4 8.7 %, old tier 1.9 % (was 7.4 %) | v84 | substitution, rolled | Oct 13 |
| (μ,C) closure truncation | 2–3 % of MSE (3.2–4.9e-10); 79 % of it from the mean closure, 20 % from δK; mean-coupled mode −0.8 % of that | exact κ3/κ4, nets 0–2 | H1 | Oct 15 physics; Oct 14 closure (1e-4 relative variance per layer, ≥ 97 % pair-sourced) |
| Closure bias (earlier) | real, predictable offline (R² 0.6–0.87) but ≤ 0.75 % in the loop | v68 | S5 | Sep 27 |
| κ3 D3 error pattern | **≈ 50 %** of MSE (6.7–9.3e-9 on nets 0–2); 99 % from \|α\| < 2.5, 36–40 % from \|α\| < 1 | v55 D3 pattern injected into exact chain | pilot.py A | Oct 15 learn |
| κ3 coherent scale | a_l − 1 = −0.0…−0.4 % (nothing to calibrate) | v55, 100 nets | k3err.py | Oct 15 |
| κ3 locality | sensitivity-weighted error OOS R² ≤ 0.14 local, ≤ 0.21 with hub features; not inherited (R² ≈ 0); not in C's eigenbasis (1–3 %); rank-32 oracle of residual keeps < 47 % | v55 | | Oct 15 learn |
| κ3 all-distinct headroom | −33.1 % (v62, Sep 28) → **−21.6 %** (v90a, Oct 15); D3-only −8.3 %, D21-only −11.5 %; rank-128 oracle −5.3 % | k111 | | Sep 28; K8.0 |
| κ3 error anatomy at depth (pre 14) | dress over-delivered ≈ 25 % (15–25 % of E3 energy; recoverable −0.37 %); inherited scale 1.06 (≈ 15 %); n⁴ newborn diagrams ≤ 1 %; pair gates 0; **≈ 70–80 % high-rank shape error of the transported (inherited) tensor** | v90a | K8.2 | Oct 15 |
| κ3 error origin | births right in scale (newborn LS 1.014–1.021); error accumulates in transported content (inherited 1.014 → 1.064) | v90a vs `pre_K3ad_PP` | K8.2(1) | Oct 15 |
| Covariance channel | "42.7 % is a conduit (D21 → K11 → next variance), not a source" | | H2 | Oct 15 physics |
| Covariance channel (Codex) | C substitution −38.4 % (3228), −37.7/−39.3 % (3476); CV −37.1 %, CO −15.7 % (3232); inherited response 8.45e-10 vs local 1.56e-11 (3478); "coupled mean/covariance propagation + learned compensation, no exclusive share" | CONTROL 6a23 / literal 8be, TRAIN0–7 / 0–1 | oracle substitutions | Oct 17; Oct 19 |
| κ4 (sensitivity) | D4/dG +10 % ⇒ 14–22 %; Q31/D22 +10 % ⇒ ≤ 4 % | exact chain | H2 | Oct 15 physics |
| κ4 (shape) | shipped κ4 forms' isolated cost **1.48e-8 ≈ whole raw MSE** (dG/D4 55 %, Q31 29 %, D22 4 %, cross 12 %) | v93r forms in exact chain | k4 | Oct 16 |
| κ4 (corrector-off) | exact κ4 alone 0.564×; exact κ3 0.596×; both 0.072× (super-additive) | v93r | O9.3 | Oct 16 |
| κ4 realised | K9.2 births −1.9 %, K9.4b refit −4.2 %, young-tree reader −3.5 % VAL / +3.15 % score | | | Oct 16 |
| Terminal / mean propagation | true incoming post15 mean ⇒ −74…−79 % (3347); premean reset M ⇒ −77 % (3232); CTF vs ALL ⇒ "strong sensitivity to premean resetting" | 2 / 8 nets | | Oct 17; Oct 19 |
| Old tier | 1.9 % of error for 16.7 % of F (23.5 u) (v84/v90a); without it ×4–×16; M leg carries the value | | O5, O3, audit | Oct 13–15 |
| Where cost sits (8be) | `_advance` 104.2 GF (36.3 %) + `_yp_contract` 106.4 GF (37.1 %) = 73.4 %; matmul 84.4 %; corrector 0.118 %; join + newborn P2 compression 2.75 % | 8be | 3477 census | Oct 19 |
| Numerics | 3485/3486 CDF law explains ≤ ±6 % of a 1.5e-12 canonical gap; "exact observer/native equality removes instrumentation as a cause" | 8be | | Oct 19 |

**Contradictions / tensions between documents**
1. *Covariance share.* O5 (Oct 13) calls covariance "the dominant channel at 42.7 %"; physics (Oct 15) reclassifies it as a conduit for κ3 (D21) error; Codex 3228/3232/3476 (Oct 17–19) measure C-substitution at −37…−43 % and read it as "coupled propagation + compensation", explicitly rejecting additive shares (`research_decision.txt` "no unique additive split"). The documents agree on the number and disagree on its interpretation.
2. *κ4 size.* Sep 27 refit ceiling 27.9 %; Oct 13 O5 8.7 %; Oct 15 physics "D4 only; Q31/D22 irrelevant (≤ 4 %)"; Oct 16 k4 "isolated κ4 cost = 1.48e-8 = the whole raw MSE, Q31 29 %"; Oct 16 O9.3 corrector-off κ4 alone 0.564×. The Oct 15 figure used ±10 % scale perturbations, which (per kai's placebo result and the Oct 16 report) under-read shape errors; the Oct 14 "κ4 forms closed" verdict (which covered O(n²) forms only) was superseded two days later by the same strategist with O(n³)-on-legs births (`STRATEGIST_RESPONSE_2026-10-16.md` §0.3 vs `_2026-10-14.md` §3.1).
3. *κ3 headroom.* "−33 % ⇒ top 5" (`STRATEGIST_ADDENDUM_2026-10-15_k3.md` §1, from v62) vs K8.0 on the shipping chain −21.6 % the same day; the v55-dump "D3 pattern ≈ 50 %" (learn) is from a pre-corrector, pre-dress chain and was never re-measured on v90a+.
4. *Propagation.* "ReLU layer contracts error, A_l 0.60–0.88" (Sep 24) vs "neutral, 0.98–1.02 for δμ" (Oct 15). Both say no amplification; the magnitudes differ (different chains and injection conventions) `[inferred]`.
5. *Corrector-on vs corrector-off screens.* Every unrefit kill before Oct 16 (K4/K5/age 7/K9.1/K9.3 etc.) was with the corrector tuned to the shipped slices; O9.3 shows exact slices look neutral-to-harmful corrector-on. The Oct 16 addendum states "unrefit kills of slice upgrades and of capacity knobs are not evidence either way", but the Oct 17 round-4 age-7 screen was run corrector-OFF and used as a kill (`STRATEGIST_ADDENDUM_2026-10-18_a.md` §0.3) — consistent with the new rule, while old-tier rank/hubs were never re-screened either way.
6. *Label floor in paired tests.* Several Codex rows compare against "organizer floor 7.7e-11" while the FRESH-A panel's own floor is 7.48e-10 (N=1e8); `research_decision.txt` notes label-noise bias does not cancel when candidate costs differ.
7. *Residual of error attribution.* `REVIEW_AND_DECISION.txt` (Oct 19): "No defensible isolated oracle-percentage allocation identifies a single causal share", versus the percentage budgets in §2 of `STRATEGIST_RESPONSE_2026-10-15.md`.

---

## 5. UNTRIED OR UNDER-TRIED

Directions that appear in proposals/literature notes but were never measured, were killed on a 2-net (or 8-net) screen rather than a proper test, or were marked "stretch"/"if time"/"post-deadline".

**Never measured**
1. An unconstrained regressor from the full carried state (C, legs, cores) at L−1 to the exact residual at L (see §3). Closest: label oracles only.
2. corpus-14k data scaling (12,000 extra labelled d8b nets; `intel2` M8). Strategist §2.4 "not started"; Codex "No 14k corpus without Samuel's shards". d8b's 2,048 nets paid −0.28…−0.73 % (`K9 closing`; `3263`).
3. Deep-layer old-tier rank 320–448 and +32 hubs under refit with the corrector free (O10.3; rev-30 §2.1). Only age 7 was screened (corrector-off +1.08 %, 7/8, `3330`); the riding band's r320 result (raw −1.29 % at F −0.81 %, 86/100) is adjacent evidence that was explicitly left "not rejected after a matched refit" (`RESULTS.md` Oct 18 03:21) and never refit.
4. Full pad on A as a graded canary (−1.04 % F, 0 Δraw, served wall ≈ 97 s projected; `S10.4`; `STRATEGIST_ADDENDUM_2026-10-18_a.md` §2.3). Never submitted.
5. A B-young twin (block-40 twin of b636/8be) — recommended Oct 17, not built; B remains B2e (older chain) / b636.
6. Symmetric-tensor tagging of the n×n closure algebra (≤ 0.5–1 u, "optional, invasive"; audit item 7).
7. Exact κ4→κ3 feed term (iii) carried as one rank-n term per young lineage (≈ 3 u/lineage-layer, prior ≤ −2…−3 %; `k43/REPORT.md` §3) — post-deadline.
8. T3/T21/T111 decomposition of the D3 birth and the exact κ3 transport gain vs Φ_aΦ_bΦ_c (physics §5; K8 item 6 "if time").
9. Nudged-rollout (data-assimilation) training of the correctors, μ ∈ {0.25…1} (lit item 1) — never run; the physics budget it addresses was measured at 2–3 %.
10. Old-tier rank/age trade with an exact basis (O8 open question) — now moot after `svd(k=)` re-pricing.
11. Per-band (within-layer by α) fold/star gains (O7.3 / M6.5) — "only if time remains", not run.
12. 7c: old legs `Q@[FA|FM]` only on D21 rows (≈ 2–3 u, "open / risky", cost2 §2).
13. A cheap carrier for the 18248 coherent-scale extraction: the full-support f64 reference shows −27 % / −33 % (2 / 32 nets, correctors off, `3384/3386`) but the two cheap carriers tried (local scale conjugation `3320`, native scale projection `3389/3391`) failed; the "known-input χ inversion" (`3394`) helped vs untreated (−10 %) but not vs extracted. No third carrier was designed.
14. Residual headroom on A: 8be serves max residual 0.339 s against the 0.40 s cap and ≈ 22.2 k ops; the ≈ 1,500-op gap was never spent on an accuracy atom (`docs/CURRENT_STATE.md` 21:55; `[inferred]`).
15. Young-chain refits at the full-reader price were priced (+68 % F, `3179`) but a partial-reader/ages sweep between "youngest only" (shipped) and "all ages" (+68 %) was not run `[inferred from the table]`.

**Killed on 2-net (TRAIN0/1) or ≤ 8-net screens with sub-percent or mixed effects (SE ≈ 0.6 % per net pair `[inferred from the 100-net CIs]`)**
- Fourth-core terminal dG residual (`3343`: +1.42 % ON, 0/2).
- Shared-join diagonal repair (`3380`: +0.042 %, 0/2; +4 % F).
- Damped all-consumer CQ transfer (`3478`: +0.59 %, 0/2).
- Exact first-layer Gaussian pK (`3484`: +1.11 %, 1/8).
- K1 minus 32 rows (`3483`: score +0.57 % worse, 2/8, t95 crosses zero).
- Primary-CDF law (`3485/3486`: −6.2 % / +2.2 % on a 1.5e-12 gap; "fails both-network criterion").
- Distinct double-third birth (`3359`: −0.00005 %; forecast +9.1 % F — price kills it regardless).
- Covariance-only Gaussian k3/k4 completion (`3234`: −0.18 %, 6/8; parked, never priced whole-source).
- Coherent pK11/21/22 completion (`3234`: −0.17 %, 7/8; parked).
- Literal final-weight reflection (`3351`: +0.09 %, 1/2; priced +4.1 % F).
- Local-even composition (`3366`: +0.07 %, 1/2 → went to 100 nets, +0.11 %, shipped) and terminal sensitivity O set (`3373`: −0.027 %, 1/2 → 100 nets +0.074 %, CI crosses 0, parked) — the two cases where the 100-net follow-up changed the sign relative to the 2-net screen.
- Deep-only age 7 corrector-off (`3330`: +1.08 %, 7/8 vs 1.51 % break-even).
- L14 message / local H1 control (`3495`: +0.21 % / +0.08 % worse on CHECK36, 14/36 and 18/36).
- Quadratic64 statistical control (`3496`: +8.4 % on 2 nets, 8 replicas) — magnitude large enough that n is not the issue.

**Marked "stretch" by the strategist**
- "old tier −10 u or −10 % raw" (`docs/research/2026-10-12_deep_review.md` §6 stretch row; PLAN rev 18 §1 "Stretch ≤ 0.11 B at ≤ 1.9e-8 raw").
- Symmetric-state representation (κ3 as Σ Sym(q⊗G), κ4 as Sym(M⊗C), sandwich-priced) — "post-deadline research item" (`STRATEGIST_RESPONSE_2026-10-14.md` §5).
- Subspace cubature on top-k covariance directions; exact pair integrator via setup-time tables (same §5) — not built.

**Positive, measured, and left unused by a gate** (not "untried", but not shipped)
- U1 NEW10: −0.916 % VAL (81/100, [0.69, 1.14]) at ≈ +0.15 % F; gate 1.5 % (`docs/CURRENT_STATE.md` 00:35).
- U1 original: −0.300 % (77/100) same gate.
- U2b: −1.24 % VAL (100/100) at ≈ +3.4 % F; gate 5 % (net negative on score `[inferred]`).
- riding r320: raw −1.29 %, score −2.10 % vs b636 (86/100) but 0.57 % behind r288 unrefit; "do not reject after a future matched refit" — no refit done.
- K7.1 mean-coupled covariance corrector: −0.22 % refit (63/100), below the 0.5 % line.
- h32/residual64 causal 226: +0.040 % (71/100), parked.

---

## 6. PROCESS COST (CURRENT_STATE.md Oct 18 00:11 → Oct 19 12:21)

Source: `docs/CURRENT_STATE.md` timestamps, `RESULTS.md` Oct 18–19 rows, `experiments/2026-10-19_push/REVIEW_AND_DECISION.txt`. All hour figures are `[inferred]` from recorded start/complete times and queued-runtime fields; where only a bound was declared the bound is noted.

**Oct 18 (00:11 → 24:00), local RTX 5090 + CPU box + RunPod**

| bucket | jobs | GPU-hours | CPU/wall-hours | note |
|---|---|---:|---:|---|
| Science: matched refits | 3411 → 3413/3414 (R0/R1, 01:26–05:11), 3431 U1 (06:22–09:59), 3444 U2b (10:34–16:20, two 368-update arms), 3472 NEW10 (21:52–00:12) | ≈ 15 | — | the only large GPU consumers; each produced a VAL number (R0 −0.69 %, R1 −1.52 %, U1 −0.30 %, U2b −1.24 %, NEW10 −0.92 %) |
| Science: canonical VAL / FRESH / bills (CPU) | 3405 riding VAL (≈ 2 h), 3412 O704 FRESH (≈ 1.3 h), 3420 P-A FRESH, 3427 A4/A6 VAL (1.7 h), 3436 R1 VAL (0.8 h), 3445, 3447, 3448 (0.9 h), 3454 R0 (0.74 h), 3460 (0.75 h), 3462 FRESH (0.75 h), 3399/3419/3423/3456/3457 bills | 0 | ≈ 11 | 4-CPU jobs; several serialised behind the 20 GiB refit admission |
| Gates / certificates | 3428 P-A memory 100 (0.87 h), 3465 8be memory 100, 3429/3430/3439 quiet-wall **failures** (environment guards), 3466 quiet wall (0.5 h + OpenClaw stop/restore), 3437/3459 fidelity, 3440 U2b proof, 3446 generator proof (1000 rows), 3468 parser proof, protocol runs | 0 | ≈ 6 | three failed wall attempts before one pass; memory gate rewritten mid-day (M1–M4 v3.1) |
| Cloud (RunPod) | gjv6 proof pod (closed 13:04), ex9g481 (driver fail), 4090 pod 14:37–15:07 (admission window exhausted before binding), cu13 5090 pod (parity fail), container 2d249a1c (67 GB re-download) | ≈ 1.5 pod-h | — | ≈ USD 5 (posted Oct 18 total 4.95); zero scientific output; "ALL PAID COMPUTE STOPPED" ×6 |
| Review / freeze / pins (no compute) | ~75 CURRENT_STATE entries; each job carries manifest + N pins (e.g. 3462: 777 pins; 3465: 818; 3466: 872; R3 binding 276), peer review, root review, dry-run, saved-only audit, independent arithmetic audit, cleanup receipt | 0 | ≈ 24 wall-h of agent time `[inferred: continuous entries 00:11 → 00:35 next day]` | ≥ 2 audits per completed job |

**Oct 19 (00:11 → 12:21)** — the Codex "continued score push": 36 numbered jobs (3476–3505 plus audits), total queued numerical runtime ≈ **1.1 GPU-hours** `[inferred: 1184 s (3493+3495) + 1728 s (3500) + 46.7 s (3496–98) + 59.9 s (3499) + 282.6 s (3502) + 102 s (3485/86) + ≈ 450 s (3476–3492 combined, each 10–50 s) ≈ 3,850 s]`, against ≈ 12 wall-hours of root freezes, pin verification (e.g. 3495: 532 pins; 3502: 395 pins), three-worker source reviews with 20-minute ceilings, and saved audits (3501, 3503). Ratio of science compute to process wall time ≈ 1 : 11 `[inferred]`. Oct 19 also produced no change to the incumbent.

**Jobs 3495–3505, one line each (scientific outcome)**

| job | what | outcome |
|---|---|---|
| 3494 | l14_fit original packet | cancelled while queued; replaced by 3495 (never executed) |
| 3495 | matched L14 K21-message fit vs local-H1 control (FIT64/CHECK36, 109 full/1940 tail/1544 reverse/192 updates; 1152.9 s) | **NO_GO both arms**: CHECK raw +0.211 % / +0.083 % worse, 14/36 and 18/36 wins (need 27); precision-label note: scorer_f32 ≠ exact evaluator, audit reproduced ev.py — signs unchanged |
| 3496 | quadratic64 exact-mean statistical control (2 TRAIN nets, 65,600 trajectories, 17.3 s) | **NO_GO**: paired forecast-cost-weighted variance +8.42 % / +8.29 % vs linear control; all 8 replicas lose; proxies 2,367–2,537× native score |
| 3497 | saved-only audit of 3496 | producer/math stages PASS, **cost-parser failure** (preserved) |
| 3498 | saved-only repair of 3497 | PASS (math 9121983c / cost 96db6d5e); no model rerun |
| 3499 | directional response proof (23 forward / 4 VJP; 59.9 s) | **technical PASS only** (no fit, no score); enables 3500 |
| 3500 | fixed eight-direction FIT64/36 directional-subspace candidate (SLSQP, 636 calls/508 duals/64 VJPs, ≈ 29 min) | **FAILED**: SLSQP status 8 (positive directional derivative) after 13 iterations; no endpoint, accuracy unmeasured (HOLD_INTEGRITY) |
| 3501 | separate failure audit of 3500 (CPU, 49 s) | PASS: 567,146 checks; modelled loss change −1.75e-11 remains unmeasured |
| 3502 | finite continuation from 3500's certified point (241/133 plain calls, 282.6 s) | **NO_GO**: CHECK36 mean −4.85e-12, 18/36 wins (gate 22); interval crosses zero |
| 3503 | independent audit of 3502 (CPU, 13.7 s, 126,348 checks) | PASS; confirms NO_GO; no release/bill |
| 3504 | fixed two-mode input Krylov scalar probe (first-layer-exact mixed κ3/4/5) | **runtime API failure** before first full prediction (`fnp.all` absent; 5.07 s); no scientific result |
| 3505 | repaired probe (same science) | RUNNING at 12:21 UTC; cost forecast 3.1 GF (needs ≥ 1.07 % raw gain at same parent cost); **no result in snapshot** |

Summary of the Oct 18–19 window: ≈ 16 GPU-hours and ≈ 17 CPU-hours of measurement produced one shipped gain (8be: −7.6 % score, Oct 18) and five parked positives (U1/NEW10/U2b/r320/O704-class); the remaining ≈ 36 wall-hours went to freezes, pins, audits, failed wall/memory/cloud gates and ≈ 20 fixed NO-GO screens, none of which moved the incumbent `[inferred]`.
