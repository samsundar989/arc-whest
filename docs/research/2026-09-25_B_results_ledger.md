# B — Measured results ledger, arc-whest (neopok919), Phase 2

*Compiled 2026-09-25 from `RESULTS.md` (all 1,627 lines), `docs/CLAIMS_LEDGER.md`, `docs/DESIGNATION.md`, `docs/CURRENT_STATE.md`, the three strategist packets (09-22/23/25), the three strategist responses (09-22/23/24), the four handoffs (09-21..24), `docs/PLAN.md`, `docs/BENCHMARK.md`, `docs/VALIDATION.md` §2b-iii, `archive/2026-09-21/experiments/LOG.md` (for pre-Sep-21 submissions) and the `experiments/2026-09-2*` folders. Every number below is quoted from those sources; where two sources disagree I say so in §5. Scores are AIcrowd adjusted final-layer scores (lower is better); "raw" is unweighted final-layer MSE; F/B = FLOPs ÷ 2^41.*

---

## 1. Submission history

### 1a. Phase 1 (for reference only)

Phase 1 was a different task shape and the team ran a **sampling** method (Sobol RQMC + dead pruning + rank-32 rotation), not the cumulant chain. Best graded: **#327743 v2.4.1, 2.6314e-7 adjusted** (raw 1.3323e-6, multiplier 0.19896, residual 10.0 ms/MLP, 0/100 failed), 2026-08-10; rank #90 of 200 at Aug 6 (v2.4, 2.7194e-7). Earlier: #324094 v1.1 3.893e-7 (rank ~159), #324420 v2.4 2.7194e-7. Phase-1 best published honest method was Puffi at 9.10e-8 (`BENCHMARK.md` §2a). None of this transfers to Phase 2 except the measurement machinery.

### 1b. Phase 2 (Aug 22 onward) — every graded run

| # | id | date (UTC) | file / what | score | raw MSE | F/B | ops | wall worst (median) | residual worst (µs/op) | failed | rank at the time |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|
| 1 | 328662 | Aug 27 | canary (K2 sampling port) | 3.9330e-07 | — | — | — | — | — | 0/100 | — |
| 2 | 328664 | Aug 27 | v2.7 (K2p1) | 1.3738e-07 | 1.374e-6 | ~0.1 | — | — | — | 0/100 | 53 (Aug 31, 83 teams) |
| 3 | 329887/889/890/891/892/894 | Sep 5 | 7-slot instrument burst: 3 FAILED (`ESTIMATOR_EXCEPTION`, unguarded `setup`, `copy=False`), 329889 = v2.7 control 1.3738e-07, 329892 zeros-bisection, 329894 stage-diagnostic | — | — | — | — | — | — | 3 failed subs | — |
| 4 | **329896** | Sep 5 | canary4 — memoryless regenerating κ3 flow | **8.501e-08** (at the 0.1 floor) | 8.50e-7 | 0.077 | 2,690 | — | 30.1 ms med (11.19) | 0/100 | 47 of 109 |
| 5 | **330212** | Sep 7 | `estimator_k3_full.py` — exact K3 chain, no Strassen | **2.9467e-08** | 3.7793e-8 | 0.7797 | 3,249 | 9.54 s | 36.4 ms med (11.21) | 0/100 | — |
| 6 | **330290** | Sep 8 | `estimator_k3_strassen.py` — Strassen L=4 | **1.843e-08** (1.8360e-08 on matched rows) | 3.794e-8 | 0.48578 | 15,393 | **114.68 s** | — | 0/100 | 44 (Sep 10) |
| 7 | 330327 | Sep 8 | Strassen depth 3 | 2.031e-08 | — | ~0.53 | 12,357 | 61.93 s | — | 0/100 | — |
| 8 | 330566 | Sep 10 | Strassen depth 2 | 2.284e-08 | 3.779e-8 | 0.6042 | 9,321 | 33.0 s | — | 0/50 public | — |
| 9 | **330950** | Sep 14 | `estimator_k3_strassen_L4_f21.py` (F21, exact factored K3 tower, depth 4) | **1.5957e-08** | 3.779e-8 | 0.42221 | 14,091 | 93.16 s (89.5) | 0.1562 s (11.09) | 0/100 | 58 of 156 |
| 10 | **331317** | Sep 16 | `estimator_k3_v31.py` (F21 + P1 κ4 regen/label-free λ + P2 7→4 identity) | **8.7520e-09** | 2.9074e-8 | 0.30103 | 13,053 | 68.13 s (66.0) | 0.1402 s (10.74) | 0/100 | 58/165 → 71 of 184 by Sep 21 |
| — | — | Sep 16–21 | **no submissions** (5 days; C10 ready and unsent for 3 of them) | | | | | | | | |
| 11 | **331816** | Sep 22 | `estimator_k3_v43_complete_c10.py` (C10: rank-256/age-4 shared old tier, fitted (θ,s,t) controls, depth 5) | **6.4914e-09** | 2.9686e-8 | 0.21867 | 23,421 | 87.56 s (83.1) | 0.2363 s (10.09) | 0/100 | ≈59 expected |
| 12 | **331863** | Sep 22 | `estimator_k3_v50_safe.py` (C10 + D2 fitted spline profiles + D3 item-1 Strassen routing) | **6.192e-09** | 2.876e-8 | 0.21529 | 25,728 | 88.75 s (84.4) | 0.2655 s (10.32) | 0/100 | — |
| 13 | **331880** | Sep 22 | `estimator_k3_v53_age5.py` (v50 + old-tier compression age 4→5) | **5.5603e-09** | 2.434e-8 | 0.22848 | 25,140 | **96.49 s** (92.8) | 0.2685 s (10.68) | 0/100 | ~58 of 184 |
| 14 | 331939 | Sep 23 03:06 | v50 identical bytes (wall-variance rerun) | 6.192e-09 (identical to every digit) | 2.876e-8 | 0.21529 | 25,728 | 87.66 s (85.0) | 0.2629 s (10.22) | 0/100 | — |
| 15 | **331967** | Sep 23 08:27 | `estimator_k3_v55_p2r4.py` (v53 + `_P2_RANK` 32→4) | **5.3200e-09** | 2.461e-8 | 0.21618 | 23,645 | **114.49 s** (92.8), 27 nets >100 s | 0.2805 s (**11.86**) | 0/100 | 43 of 152 (Sep 23) |
| 16 | 331980 | Sep 23 09:18 | v55 identical bytes | 5.3200e-09 (identical) | 2.461e-8 | 0.21618 | 23,645 | 113.81 s (92.5), 15 nets >100 s | 0.2709 s (11.46) | 0/100 | — |
| 17 | 331989 | Sep 23 10:13 | v50 identical bytes as **server-state control** | 6.192e-09 (identical) | 2.876e-8 | 0.21529 | 25,728 | **104.1 s** (87.0), 15 nets >100 s | — | 0/100 | — |
| 18 | **332063** | Sep 23 19:56 | `estimator_k3_v56b_wallsafe.py` (v55 at Strassen depth 4 / leaf 64) — wall-safe twin | 5.7076e-09 | 2.4609e-8 (identical to v55) | 0.23193 | 19,804 | **60.7 s** (54.5) | 0.2188 s (11.05) | 0/100 | — |
| 19 | 332088 | Sep 24 00:11 | `…v56c_mix__tr64.py` (v55 + tower transports at leaf 64) — A/B wall reading | 5.4341e-09 | 2.4610e-8 | 0.22081 | 22,978 | 80.30 s (76.90) | 0.2413 s (10.50) | 0/100 | — |
| 20 | **332129** | Sep 24 08:30 | `estimator_k3_v57_corr.py` (v55 + in-loop learned mean corrector, 100-net sequential fit) | **5.0443e-09** | 2.3326e-8 | 0.21626 | 24,186 | 92.81 s (90.43) | 0.2507 s (10.37) | 0/100 | **~33** (5.00e-9 displayed) |
| 21 | **332223** | Sep 25 01:05 | `estimator_k3_v60_ws400.py` (400-net corrector, depth 4 / leaf 64) — **designation A** | 5.3387e-09 | **2.3011e-8** (best raw ever graded) | 0.23200 | 20,345 | **57.59 s** (55.73) | 0.2121 s (10.43) | 0/100 | — |

**Totals.** 24 Phase-2 submission ids consumed (3 failed at smoke on Sep 5; 21 graded, all 0/100 failed nets). Best graded **5.0443e-09** (#332129). Trajectory of the best: 1.374e-7 (Aug 27) → 8.50e-8 (Sep 5) → 2.95e-8 (Sep 7) → 1.84e-8 (Sep 8) → 1.60e-8 (Sep 14) → 8.75e-9 (Sep 16) → 6.49e-9 → 6.19e-9 → 5.56e-9 (Sep 22) → 5.32e-9 (Sep 23) → 5.04e-9 (Sep 24). Sep 21→24: 8.75e-9 → 5.04e-9 = **1.735×** in four days after the process reset.

**Board context** (from the packets): Sep 21: #1 J2W 2.10e-9 (raw 1.64e-8, 0.131 B, 303 entries), #2 marius 2.40e-9, #3 suliman/a_s6 2.50e-9, #10 3.50e-9; ranks 13–41 = trimmed 504aldo derivatives at 4.2–5.3e-9; ranks 42–57 = sixteen byte-identical 504aldo clones at 5.40e-9. Sep 23: J2W 2.1e-9 (3 sealed TIME_EXHAUSTED, walls 105–117 s), suliman 2.2e-9 (49 entries, 0 failures, walls 84–108 s), marius 2.3e-9; top-2 per-net FLOPs vary ±6%. Today (per brief): J2W 2.00e-9, suliman 2.10e-9, marius 2.30e-9; we are ~#33.

**Candidates evaluated but never sent:** `v46_d3` (6.3787e-9 proj, +1.74% only), `v58_ws` (5.3998e-9 proj, superseded by v60_ws400 before its slot), `v59_corr400` (4.9498e-9 proj DEV / **4.9763e-9 derived from A's graded raw** — the best score the project has, blocked by the 3% gate), `v61_rec` (4.9585e-9 proj, no gain over v59), `v56c tr64_wide64` (≈5.60e-9, ≈82 s proj, never built into a canary).

---

## 2. Lever ledger

Convention: Δraw < 0 is better; Δscore < 0 is better. Panels: DEV = 100 organizer mini nets; FIT = 100 disjoint full-split nets; HOLDOUT = 900 remaining (HOLDOUT-300 pinned for the 400-net fit; HOLDOUT-600 never fitted). Accuracy probes are cupy (panel mean agrees with numpy to +3.2e-5); costs are the f64 read-only flopscope twin (matches the server FLOP count to 0 / 0 / −491 FLOPs on every anchor).

### 2a. Structural levers on the chain

| lever | what it is | measured effect | panel | status |
|---|---|---|---|---|
| **Strassen depth** (2/3/4/5) | recursive Strassen–Winograd on the n³ products; leaf = 1024/2^depth | depth 0: 0.329085 B, 8,724 ops; depth 3: 0.240979, 20,017; depth 5: 0.218666, 23,421. Depth 5 vs 4: −7.3% FLOPs, +~3,800 ops, **+1.9× kernel ps/FLOP** (176 vs 93) → +54 s wall (v55 114 s vs v56b 60.7 s). Depth 6 (leaf 16): F/B **0.22014 > v55's 0.21618**, 26,773 ops — dearer | f64 twin; 10 graded runs | depth 5 = score minimum, depth 4 = designable; **depth 6 closed** |
| **63% of ops is Strassen glue** | depth 0→5 buys −33.6% FLOPs for +14,697 op calls; break-even ~61,000 FLOPs saved per extra op | measured | f64 twin | fact governing every cost lever |
| **Old-tier rank** (shared basis Q, age 4) | rank 64 / 128 / 192 / **256** / 320 / 384 | cost −23.2 / −15.6 / −6.8 / — / +9.3 / +19.4%; raw +545.6 / +170.4 / +43.9 / — / −14.9 / −19.9%; **score +395.8 / +128.1 / +34.1 / — / −6.95 / −4.42%** (vs v50) | DEV 100 | rank 256 kept; truncation **dead** ("r^-3 spectral tail"); rank 320 score-positive but dominated by age 5 |
| **Old-tier compression age** 4→5→6(→7→8) | layers a lineage stays exact before folding into Q | age 5: +6.13% FLOPs, −15.50% raw, **−10.32% score** (the largest single lever found); age 6: +11.59% / −20.17% / −10.91%; packet adds age 7 −8.13%, age 8 −5.28%. At age 5, rank above 256 is worth only 6.6% (384) / 6.9% (512) raw vs 19.9% at age 4 | DEV 100 | **age 5 shipped (v53)**; age 6 not shipped (VmPeak ~87% of 8 GiB, wall >100 s, only −0.66% more) |
| **P2 residual rank** (`_P2_RANK`, the rank-32 born-(2,1) residual) | 32 → 16 / 8 / **4** / 2 / 1 | Δraw +0.40 / +0.82 / **+0.94** / +11.17 / +1,175%; ΔFLOPs −3.00 / −4.56 / **−5.38** / −5.82 / −6.04%; Δscore −2.61 / −3.78 / **−4.49** / +4.70 / +1,098%. Rank 2 is a cliff (worst net 2.3×) | DEV 100, vs v53 | **rank 4 shipped (v55)**, graded 1.0452× vs v53 |
| **Elasticity table** (v53 components) | ablate each exact-tier component | no hubs: raw +5.41%, FLOPs −0.57%, score +4.81%; closure off: +18.14% raw / −1.03% / +16.92%; closure last-layer only: +18.03% / −1.03% / +16.81%; shared rank 224: +6.09% / −1.71% / +4.27%; age 4: +18.33% / −5.77% / +11.49%; age 6: −5.53% / +5.15% / −0.66%; age 6 + P2r8: −4.13%; age 6 + P2r4: **−4.73%** (23,057 ops, 91.4 s proj) | DEV 100 | only P2 rank has elasticity <1; age6+P2r4 not shipped (≈700 MiB memory vs D7's 7 GiB bound) |
| **Single-leg tower** (T0–T3; strategist's Sep 22 program) | keep only the diagonal-born block `A_D = M·d(sb)` for old lineages, regenerate the correlation-born block memoryless | T0 unshared exact chain 2.3642e-8 (= reference); v53 settings ×1.077 (+7.7%). **T1** (drop old (2,1) beyond age a): a=2 ×26.9, a=3 ×15.5, a=4 ×8.85. **T2** (drop L-block beyond a): a=0 ×73.5, **a=1 ×48.1**, a=2 ×32.0, a=3 ×21.2, a=6 ×5.77, a=10 ×1.276. T1∩T2 ×53.3. Diagnostics: drop whole lineage a=1 ×68.6, a=5 ×13.2; two-leg block only ×14.0; Z residual only ×6.41. **T3** fitted regeneration (CMA-ES on FIT, 2-param age-exponential: +0.59% in-sample; free 13-entry ρ: +2.99% in-sample, **+0.09% on DEV**, still ×48.1) | DEV 100 (cupy), 0/100 wins on every arm | **CLOSED.** Strategist predicted ≤8% (T1 a=3) and 10–40% (T2 a=1); measured +1,454% and +4,714%. T4/T5 never built |
| **Memoryless κ3 regeneration** | the whole tower regenerated each layer | #329896 graded 8.50e-8 at the 0.1 floor, raw 8.5e-7 ≈ 36× the exact tower | GRADED | closed (Sep 5, re-read Sep 23) |
| **N1 sensitivity-weighted old-tier basis** | choose Q by downstream sensitivity ‖Δμ_L‖² instead of energy | Spearman(energy, sensitivity) **0.994 / 0.958** on 2 nets; top-256 overlap 249/256, 239/256; sensitivity captured by energy-256 vs sensitivity-256: 98.83% vs 98.87%, 98.32% vs 98.48%. Ceiling of any 256-of-512 pick: energy-384 −6.56% raw (98/100), energy-512 −6.93% (99/100) | 2 DEV nets × 512 tails; DEV 100 for the ceiling | **KILLED** on its own criterion (Spearman > 0.9). Arms (c)/(d) not run. Earlier merged-core N1 (Sep 16, Tucker core) priced ≥0.143 B at r=96 only — parked then |
| **Tiered old-tier rank** (`_SHARED_RANK_SCHED`, v58_tier) | explicit re-compression of Q in the tail | 192 from layer 9: raw +12.33%, FLOPs −4.23%, **score +7.6%**; 192@11/128@13: +17.90% / −4.88% / **+12.1%**; 192 from 13: +2.46% / −1.44% / **+1.0%** | DEV 20 (0/20 wins each) | closed; loss 6–9× predicted |
| **Rank taper `_SHARED_RANK_TAIL`** (v51_h3) | request a smaller rank for the last h layers | byte-identical FLOPs/ops for h=2/3/4, rank 64/128; raw −0.01% | DEV 100 | inert (sketch width is the old Q's 256, not the requested rank) |
| **P3 D21 feedback** (`_P3_ON`) | rank-r feedback of D21 into the state | on v55: r16 raw −4.57% (72/100), FLOPs +6.63%, **score +1.76%**; r8 −3.28% (66/100), +3.64%, **+0.24%**. Earlier: C10+P3 r32 raw −6% for +12% FLOPs (0.953× score, Sep 18 factorial); uncompressed v42 −17% raw | DEV 100 | **closed** (score-negative at every rank). Note: P3 was **never on** in any shipped file C10→v61 (doc correction Sep 23) |
| **κ4 regeneration + label-free λ (P1)** | regenerate κ4 as exact diagonal + λ·C_off | 1.2371× on 24/24 (P01), cost +0.000044 B; λ-scale sweep 0.5/1.0/2.0 = 1.1964/1.2371/1.0344×; label-free λ within 1.1% of sweep optimum | P01 (24-net mint) | shipped in v31 (Sep 15) |
| **P2 7→4 identity** (separable (2,1) leg) | reconstruction identity `A_D = M·d(s)+3A_L·d(e)+Z·Lᵀ` | cost 0.7128× (C/B 0.42221→0.30097), raw 0.99994×; rank-32 M-residual +0.90% raw | P01 + twin | shipped in v31 |
| **Distinct-index κ4 birth feed (v32)** | first-order-complete κ4 at birth | raw 1.0363×, score 1.032×, FLOPs ×1.0043 | P11 (24) | "supported improvement", bundled; slice gate (≥1.5×) failed |
| **Rank-32 old-D21 query readout (v44)** | two-pass rank-32 approximation of the old-D21 readout | FLOPs −5.51% (480.85→454.34 G, ops 23,421→22,151); raw 3.18e-8→5.75e-8, score 0.568× | P12 (24) | closed Sep 18 |
| **Fitted dense P4 sentinel** | dense 4th-order | raw 2.596e-8 but 0.2894 B → 6.72e-9 official | P12 | closed |
| **K4 / unfactored AUGMENT / SIMPLE** (ARC's own repo, D4b) | richer/poorer term tables priced with ARC's counter | vs factored AUGMENT K3: factored SIMPLE K3 cost 0.125×, MSE 2.744×, **score 0.343×** (n=128) / 0.312× (n=256); unfactored AUGMENT 1.147× / 2.180× worse; **factored K4 cost 21.2–42.2×, MSE 0.294–0.154×, score 6.22–6.48× worse** | PROXY shapes (128,8), (256,8), 8 nets | K4 and D4b closed; "SIMPLE beats AUGMENT" never acted on |

### 2b. Fitting levers

| lever | what it is | measured effect | panel | status |
|---|---|---|---|---|
| **D2 fitted spline profiles** (v47/v48/v50) | six per-layer profiles as 4-knot splines, CMA-ES on FIT | raw −2.691% (DEV), −2.871% (HOLDOUT 100 of 900), 0.18 pp apart; with item-1 routing: score −3.7% vs C10 graded | FIT→DEV/HOLDOUT | shipped in v50; PLAN projected 1.15–1.75× |
| **D6 refit on v53 + old-tier readout gain** | more parameters (16 vs 12), CMA-ES seeded at shipped coefficients | identity 2.628931e-8; shipped 2.509809e-8 (−4.531%); best of **302 further evaluations: 2.509809e-8 (0.000%)**; 9 evals within +1%, closest non-baseline +0.196% | FIT | **saturated**. (First run reported 0.00% for the wrong reason — seeded at identity; retracted and re-run) |
| **Per-net adaptive λ** (v57_adlam) | `λ_l ← λ_l·clip(r_l/REF_l,0.5,2)^β`, r = mean(dG)/mean(var) | r_l CV across 100 FIT nets only 1.8–9.0%; β=0.5: −0.28% (71/100); **β=1: −0.39%** (62); β=2: −0.11%; β=−1: +1.01% (sign check). Same-layer (504aldo) form −0.35% | FIT ref, DEV 100 | closed as lever; kept as free component |
| **Joint refit on the P3 chain (4b)** | λ × θ × feed + adaptive-λ β | not run; D6 = 0.00% on v53 taken as the answer (≤1% inferred) | — | never measured |
| **Exact Gaussian bivariate tail** (v59_biv) | Price-theorem k≥3 Hermite tail, 4 GL nodes | 4 nodes −0.23% raw (±0.11, 65/100); 8 nodes −0.22%; ~+0.1% FLOPs | DEV 100 | closed; free component |
| **In-loop learned mean corrector** (v57_corr) | per-layer 33→32→32→1 tanh MLP on chain-state features, adds to μ_l before it feeds layer l+1; weights base64 in-file | ceiling (teacher-forced, FIT 100, net-wise 5-fold): ridge R² 0.019–0.309 → **2.13%** predicted; MLP R² 0.057–0.423 → **7.32%** (framing A) / 6.16% (framing B); half-R² haircut 3.66%. **Single-pass fit: +0.58% (worse), 46/100. Sequential teacher forcing: −5.60% DEV (88/100, ±0.87), −4.76% HOLDOUT-600 (93/100), −3.79% fresh-10 (9/10), −5.22% GRADED.** Cost +0.0326% FLOPs, +541 ops, 0 materialisations | FIT 100 → DEV/HOLDOUT/GRADED | **SHIPPED (#332129, best graded)** |
| **Corrector, 400 training nets** (v59_corr400 / v60_ws400) | FIT 100 + pinned HOLDOUT-300, same procedure | DEV **−7.17%** raw (99/100); layer-16 val R² 0.00946→0.01260 (+33%), layer 8 +27%, train/val gaps ~0.0015; **graded raw −1.35% vs v57** (2.301125e-8 vs 2.332585e-8) at leaf 64 | DEV, GRADED | leaf-64 twin graded (A); **leaf-32 version blocked by the 3% gate** (4.9763e-9 derived) |
| **Corrector recursive feature** (v61_rec) | add `(W_lᵀ c_{l−1})_i`, 10 raw → 35 columns | val R² layer 16 0.01244 vs 0.01260; layer 8 0.03542 vs 0.03637; end-to-end **−7.04% vs −7.17%** | DEV 100 | closed — redundant with `mu_pre = Wᵀμ` |
| **Corrector hidden width** | h64 vs h32 | h64 **uniformly worse** at all 15 layers (ΔR² −0.005 to −0.011), wider train/held-out gap (0.106 vs 0.061 at layer 15) | FIT 100 | closed; MLP is data-limited, ridge is not (gap ~0.0005) |
| **Per-lineage adaptive age** (v60_adage) | fold a lineage when its Gram energy < τ × youngest's (floor 3, ceiling 6) | e_old/e_new at 5 lineages: median 0.380, p5–p95 0.276–0.524 (×1.9 within an age vs ×1.7 between ages). τ 0.28: 1 schedule/20 nets (= fixed age 6), raw −5.40%, cost +5.45%, **score −0.25%**; τ 0.34: −4.16% / +4.23% / −0.11%; τ 0.40: −1.75% / +1.04% / **−0.72%**; τ 0.48: +8.61%; τ 0.58: +16.76%. Per-net FLOP spread 0.00–3.34% (leaders ±6%) | DEV 100 + 5-net cost | **KILLED** (<2% gate). First build never folded — caught by bit-parity |

### 2c. Cost / billing levers

| lever | what it is | measured effect | status |
|---|---|---|---|
| D3 item 1: Strassen for plain-`@` products | route join/P2 products through Strassen | blanket: −2.60% FLOPs, **+30.3% ops** (30,528 ops = 0.339 s proj); high-value subset: −1.55% FLOPs, +9.7% ops (25,698) | subset shipped in v50 |
| D3 item 2: 15-add Winograd | replace 18-add Strassen form | −0.69% FLOPs, −8.2% ops, **+29% RSS** (VmPeak 5,754.8→6,801.6 MiB); raw −0.0034% (51/100, rounding-order) | measured; **never shipped** (memory) |
| D3 items 1+2 (v46_d3) | combined | **−2.26% score** at 23,484 ops; proj 6.3787e-9 = +1.74% vs best graded | below 3% gate; base for D2 |
| D3 item 3: pooled `out=` buffers | preallocation | `fnp.add(a,b,out=buf)` bills exactly `a+b`; slice assignment raises (immutable); `fnp.empty` costs an op | **infeasible** |
| D3 item 4: remove recomputed `A_L*M`/`C_L` | dedupe | ~0.2–0.3% by inspection | not pursued |
| D3 item 5: `as_symmetric` aliasing | symmetry discount | matmul/einsum **no discount** (33,488,896 either way); `X+Y` 65,536→32,896; `as_symmetric` costs 458,751 (~7n²) | closed (~0.05%) |
| Wall-safe twin (`_STRASSEN_MIN_BLOCK=64`, v56b/v58_ws/v60_ws400) | depth 4 everywhere | FLOPs **+7.28%**, ops −3,841 (23,645→19,804), raw −0.00% (49/100), wall **114.5→60.7 s** graded | shipped as B, then A |
| `wide64` (only the two wide D21 contractions at leaf 64) | partial twin | FLOPs +3.13%, ops −529, raw +0.00% | too small to canary |
| `tr64` (tower transports at leaf 64; 29.4% of FLOPs) | partial twin | FLOPs +2.14%, ops 22,978, kernel 141.0 ps/FLOP, wall 80.3 s graded | superseded by v60_ws400 |
| `tr64_wide64` | both partials | ≈+5.27% FLOPs → ≈5.60e-9, ≈82 s proj | never built/sent |
| Op ceiling gate | §5 ops ≤ 24,500 → **26,000**; residual ≤ 0.33 s hard gate | changed after C10's 10.09 µs/op reading | in force |
| Block-permuted layout without `concatenate` | −3–5% FLOPs (strategist estimate) | not built | open, "only if idle" |

### 2d. Closed before the Sep 21 reset (from PLAN §9, CLAIMS_LEDGER, archive)

Sampling / QMC / control-variate sidecars (0.05 B sidecar needs 1600× variance reduction; best measured 1.5×); Gaussian-mixture closures; generic post-hoc correctors (1.01–1.02×, correlation −0.05 with local features); exact augmented κ4 (−9% for ≥0.5 B); standalone (3,1)/q31 consumer (×1.00 at 128×8); qEc/qEdc bases; CP/Tucker leg compression; f64 (2× trap); E14 Gaussian-tree λ (a different object, r = −0.01 with the λ that pays); dead/cold-column pruning (0.96×, not 0.80×); packed activation lane (Phase 1: 1.1216× vs 1.15× gate, residual +8%); `kind=AUGMENT` at 1024×16 (2.094× raw on 24/24 — attributed to κ4's (n,n) structure — but the (2,1,1) slice is a 4 GiB dense object).

---

## 3. Calibration facts

### 3a. DEV → graded

`projected = DEV adjusted ÷ PUBLIC_FACTOR`. Eight observations of DEV/graded: **1.0386 / 1.0385 / 1.0339 (v31) / 1.0426 (C10) / 1.0471 (v50) / 1.0458 (v53) / 1.0397 (v57) / 1.0361 (v60_ws400)**, mean 1.0403, range 1.0339–1.0471 (v55/v56b/tr64 all read ≈1.0439 but share one raw MSE and are not counted). Constant history: 1.037 (Sep 21) → 1.042 (Sep 22, after three rows landed 0.5–1.0% better than projected) → **1.038 (Sep 25)**, after two rows graded worse than projected. The factor is **entirely raw-MSE** (the graded 100 are slightly easier than the mini 100); C10's graded multiplier matched DEV to 2e-10.

Projected-vs-graded history: v31 +0.45% · C10 −0.53% · v50 −0.97% · v53 −0.84% · v55 −0.19% · v56b −0.19% · tr64 −0.19% · v57 **+0.22%** · v60_ws400 **+0.54%**. Never outside ±1%; PLAN's 2% recalibration trigger has never fired.

### 3b. FLOPs

The f64 read-only grader twin (`tools/_cost_worker.py`, one net) predicts the server exactly: F21 Δ0, v31 Δ0, C10 Δ−491 of 4.8e11 (1e-9 rel.), and −491 again on v53/v55/v57. `whest run --split mini` reads **2^25 = 33,554,432 low** (float32 dataset skips the opening cast). Per-net FLOP spread is 0.00% on every shipped file. The old "+2^24" rule and the counting shim (~18% low) are retired.

### 3c. Label noise floor

**6.8815e-10** measured (layer-1 closed form `‖W[:,j]‖/√(2π)` in f64 vs organizer labels, DEV 100) against analytic `Var(ReLU)/1e9 = 6.817e-10` (1%). AGENTS.md carried 9e-11 until Sep 24 (7.6× too low); BENCHMARK (Phase 1) carried ~2e-10 adjusted. At raw 2.3e-8 it is ~3% of our MSE, 4.2% of J2W's. Layers 1–2 of the chain sit at the floor (6.88e-10, 6.69e-10).

### 3d. Wall (four successive models in four days)

1. `local × 1.3097` (v31 anchor; predicted C10 78.9 s, graded 87.6 s).
2. `local × 1.5981` (23% low on v50).
3. `60.251 s/1e12 FLOPs + 2.478 s/1000 ops + 4.1 s` — said 90.6 s for v53, graded 96.5 s.
4. **The leaf-size law (Sep 23 refit, 9–10 graded runs):** `median = 0.371 ms × ops + FLOPs × rate(leaf, age)`; worst = median × 1.055 (quiet) or × 1.24 (noisy).

Kernel ps/FLOP by class (graded): leaf 64: **90.0 (F21), 91.8 (v31), 92.8 (v56b), 94.6 (v60_ws400)**; leaf 32 age ≤4: 154.8 (C10), 158.6 / 160.1 / 163.6 (v50 ×3); leaf 32 age 5: 166.2 (v53), 176.3 / 175.8 (v55), 171.3 (v57); mixed (tr64): 141.0. Overhead flat at **350–408 µs/op**. ev.py rates: leaf64/age≤4 92, leaf64/age≥5 94 (the inferred +10% did not exist), leaf32/age≤4 164, leaf32/age≥5 177. Every anchor projects at or above its graded worst; v60_ws400 projected 68.8 s noisy vs 69.1 implied.

### 3e. Residual (the 0.4 s cliff)

Per-op residual is per-operation and array-size-independent (canary4 11.19 vs k3_full 11.21 µs/op at 10.2× the FLOPs; 20 competitor rows fit `−0.023 s + 11.66 µs/op`, R² 0.939). Graded worst-row rates: F21 11.09, v31 10.74, C10 10.09, v50 10.32/10.22, v53 10.68, **v55 11.86 / 11.46 (noisy day)**, v56b 11.05, tr64 10.50, v57 10.37, v60_ws400 10.43 µs/op. ev.py uses 11.09 as "worst"; implied ceilings at 11.09: 36,069 ops at the cliff, 29,757 at the 0.33 s gate, 26,000 → 0.288 s. Local in-process `residual_wall_time_s` over-reads 12–40× and is never quoted.

### 3f. Server noise

Worst/median: quiet 1.026–1.055 (F21, v31, C10, v50 ×2, tr64, v57, v60_ws400), moderate 1.114 (v56b), **noisy 1.197–1.235** (v55 ×2, v50 #331989). Identical v50 bytes: worst 88.7 → 87.7 (Δ1.1 s, quiet) → 104.1 s (noisy) within one day; medians move only 84.4 → 87.0. Per-net wall correlation between identical v55 runs: **0.03** (slow nets are random). Scores are bit-deterministic across reruns. Noisy periods are not a time of day (Sep 23 noisy after ~08:00 UTC; Sep 24 08:30 quiet). **No wall-safe candidate has a measured noisy-day reading**; every "noisy" figure in DESIGNATION.md is a ×1.24 projection.

### 3g. Evaluator

numpy backend reproduces the official `whest run` raw MSE to every digit (3.094966e-8, C10); cupy differs by +3.2e-5 on the panel mean, up to 1.9e-3 per net (float32 associativity, not TF32). Pool bounded by RAM (~3.8 GiB/worker, 4 workers). A stderr-pipe deadlock cost three wrong diagnoses before it was found.

---

## 4. Residual anatomy (v55, DEV 100, organizer all-layer labels, `tools/anatomy.py`)

### 4a. MSE by layer and the one-step residual

`one-step` = one chain step from the labelled μ_{l−1} with the chain's own covariance/κ3/κ4 (an upper bound on local closure error).

| layer | MSE | MSE_l/MSE_{l−1} | one-step MSE | one-step / accumulated |
|---:|---:|---:|---:|---:|
| 1 | 6.8815e-10 | — | — | (closed form = noise floor) |
| 2 | 6.6882e-10 | 0.972 | 2.5453e-10 | 0.381 |
| 3 | 1.0107e-09 | 1.511 | 5.6486e-10 | 0.559 |
| 4 | 1.7219e-09 | 1.704 | 9.8144e-10 | 0.570 |
| 5 | 2.7002e-09 | 1.568 | 1.3928e-09 | 0.516 |
| 6 | 3.8990e-09 | 1.444 | 1.8073e-09 | 0.464 |
| 7 | 5.4538e-09 | 1.399 | 2.3002e-09 | 0.422 |
| 8 | 7.0950e-09 | 1.301 | 2.7205e-09 | 0.383 |
| 9 | 8.9375e-09 | 1.260 | 3.1377e-09 | 0.351 |
| 10 | 1.0973e-08 | 1.228 | 3.4865e-09 | 0.318 |
| 11 | 1.3266e-08 | 1.209 | 3.8480e-09 | 0.290 |
| 12 | 1.5351e-08 | 1.157 | 4.0925e-09 | 0.267 |
| 13 | 1.7628e-08 | 1.148 | 4.3515e-09 | 0.247 |
| 14 | 2.0113e-08 | 1.141 | 4.7224e-09 | 0.235 |
| 15 | 2.2797e-08 | 1.133 | 4.9517e-09 | 0.217 |
| 16 | 2.5691e-08 | 1.127 | 5.2837e-09 | **0.206** |

Growth is superlinear early (1.70× at layer 4), geometric 1.13–1.23× over layers 10–16. **79% of the final-layer error is propagated drift.** Reproduced on FIT 100 (0.205).

### 4b. Contraction factors (propagation model `M_l = A_l M_{l−1} + S_l`, FIT 100)

| l | M_l | S_l (one-step) | A_l | weight of S_l at layer 16 |
|---:|---:|---:|---:|---:|
| 5 | 2.6835e-09 | 1.3774e-09 | 0.767 | 0.155 |
| 8 | 7.1776e-09 | 2.7410e-09 | 0.814 | 0.298 |
| 12 | 1.5676e-08 | 4.1370e-09 | 0.870 | 0.584 |
| 16 | 2.5325e-08 | 5.1905e-09 | 0.876 | 1.000 |

A_l = 0.60–0.88 at every layer: **a ReLU layer contracts incoming mean error**; all growth is newly created error. The decomposition reconstructs M_16 to −0.00% (an identity, not a fit).

### 4c. Feature correlations (Pearson, pooled 100 × 1024, max |corr| over layers ≥ 8)

| feature | accumulated | one-step |
|---|---:|---:|
| μ_pre | 0.031 | 0.002 |
| σ | 0.062 | **0.097** |
| α = μ_pre/σ | 0.031 | 0.002 |
| D3/σ³ | 0.034 | 0.002 |
| D4/σ⁴ | 0.051 | 0.082 |
| ‖W[:,i]‖ | 0.023 | 0.022 |
| D21 row energy/σ⁴ | 0.045 | 0.073 |

Nothing reaches 0.10 (strategist predicted >0.5; worker 0.3–0.7). The "#330001: 0.975 correlation with Edgeworth features at one step" result does not reproduce as a per-feature correlation.

### 4d. Bin concentration and single-feature corrector ceilings

Share of squared error in the top-2 of 10 quantile bins (random = 0.20): accumulated, layer 16: α 0.363, D3/σ³ 0.364, D4/σ⁴ 0.216, ‖W‖ 0.212; one-step, layer 16: α **0.664**, D3/σ³ 0.478, D4 0.239, ‖W‖ 0.216 (layer 12: 0.576, layer 15: 0.631). The concentration sits in bins 4–5 (α ≈ 0, the ReLU kink) with per-bin bias/RMS of −0.001 / +0.024 there and +0.120 / +0.123 in the flanking bins — **same sign both sides of zero, i.e. symmetric spread, not an offset**; no closure term implicated. Single-feature nonparametric ceilings: 0.04%–0.7% of squared error (layer 16 one-step: α 0.459%, D3 0.711%, D4 0.513%, ‖W‖ 0.132%); ~1.8% if the four axes were independent. Yet the 33-feature MLP reaches R² 0.057–0.119 — the signal is in feature **interactions**.

### 4e. Corrector R² by layer (FIT 100, folds by net)

| layer | one-step MSE | ridge R² held-out | MLP R² held-out (h32) |
|---:|---:|---:|---:|
| 2 | 2.5383e-10 | 0.309 | 0.423 |
| 3 | 5.6345e-10 | 0.143 | 0.290 |
| 5 | 1.3774e-09 | 0.027 | 0.119 |
| 8 | 2.7410e-09 | 0.019 | 0.078 |
| 11 | 3.9118e-09 | 0.017 | 0.090 |
| 14 | 4.7635e-09 | 0.024 | 0.076 |
| 16 | 5.1905e-09 | 0.022 | 0.057 |

Late layers carry the gain (layers 14/15/16 contribute 1.10/1.24/1.18 pp of the 7.32%; layers 2–5 together 0.26 pp). Sequential-fit diagnostic (residual at l with correctors 0..l−1 active, FIT): layer 8 7.1776e-9 → 6.5962e-9 (−8.1%), layer 13 −8.1%, layer 16 2.5325e-8 → 2.3472e-8 (−7.3%). Sequential per-layer held-out R²: 0.166 (layer 2) → 0.0095 (layer 16) at 100 nets; 0.16352 / 0.03637 (layer 8) / 0.01260 at 400 nets.

---

## 5. Proposed but never measured; and results worth re-checking

### 5a. Never measured (and why)

1. **κ4-on-M for the exact tier (T4 / HANDOFF 09-23 item 4c).** Specified twice (Sep 22 on the T3 arm, Sep 23 "on the recent lineages only"); the Sep 24 response listed it as "beyond the one run already in flight" — **no run ever existed**. Recorded as NOT RUN, priced 0–3% by the strategist. Closed by decision, not by data.
2. **Joint refit on v55 (4b: λ table × θ × feed + β).** Never run on v55; D6's 0.00% on v53 was taken as sufficient.
3. **Shared corrector with an 8-dim layer embedding (`--shared`).** Exists in `tools/corr_mlp_ceiling.py`, never run; per-layer models shipped instead (190 KB base64).
4. **Young/old tier shares of D3 as corrector features** — need chain surgery; excluded from every ceiling.
5. **Corrector applied to the covariance/κ3 channels** — HANDOFF said "covariance channel left untouched"; never tested.
6. **Corrector on more than 400 nets** — 100→400 was worth 1.57 pp and the MLP is data-limited; HOLDOUT-600 was reserved, so 700+ nets was never tried.
7. **Corrector on top of an age-6 or rank-320 chain, or on tr64** — never built.
8. **H1 (regenerated old-tier aggregate) and H4 (linear-response hand-off)** — struck by inference from the rank table and then from T1–T3; "not measured; superseded by T1–T3".
9. **H3's second half** — switching off old readouts / births at layers ≥ L−2 keeping only the (3)-diagonal readout: "real surgery on the birth logic… not attempted yet".
10. **N1 arms (c)/(d)** (sensitivity-256/-192 on DEV 100) and **Step 5 adaptive rank per net by tolerance** — not run after the Spearman kill.
11. **Block-permuted layout without `concatenate`** (−3–5% FLOPs, strategist Q2) — not built.
12. **D3 item 4** (dedupe `A_L*M`/`C_L`, ~0.2–0.3%) — not pursued.
13. **Winograd 15-add + age 5** ("if memory allows a shallower Strassen depth") — item 2 measured on C10 but never combined with age 5 or with depth 4, where its +29% RSS would matter less (v60_ws400 sits at 4.9 GiB).
14. **A third hypothesis for the leaders' ±6% per-net cost** (an iterative solve with a conditioning-dependent iteration count) — proposed Sep 25, no mechanism specified.
15. **"SIMPLE beats AUGMENT on score by ~3×"** (D4b's surprise) — explicitly "not acted on".
16. **κ5 / (5,)-slice at the last layer** — only ever part of the dead single-leg spec.
17. **D2(d) finite-difference sensitivity anatomy** (∂μ_L/∂p_j span) — superseded by the all-layer anatomy; never done.
18. **A genuine noisy-day reading of any wall-safe candidate** (v56b, v60_ws400) — all six readings since Sep 23 19:56 are quiet/moderate.
19. **v59_corr400 graded row** (leaf 32, the best score the project has: 4.9763e-9 derived) — blocked by §5's 3% gate against our own best; needs Samuel.
20. **How two designated entries combine** in the private rerun (Rules v14 §6) — asked three times, never answered; drives the A/B choice.

### 5b. Measured results that look inconsistent or deserve a re-check

1. **Residual per-op constant is stale.** ev.py's "worst" rate is 11.09 µs/op (F21), but v55 #331967 graded **11.86 µs/op** and #331980 11.46 on the noisy day. At 11.86 the 26,000-op ceiling is 0.308 s (still < 0.33) and the cliff is 33,727 ops; gate-table headroom figures quoted at 11.09 are ~7% optimistic for a noisy day.
2. **The DEV→graded factor section in RESULTS.md is layered and self-contradictory** ("stays at 1.042" and "LOWERED to 1.038" in the same bullet; "six observations" in CLAIMS_LEDGER #30 vs eight in RESULTS). AGENTS.md still says 1.037, a 1.3097 wall factor and a "+2^24" FLOP offset — all superseded.
3. **v50 #331863 has no GRADED row in the Rows table**; its DEV row still says "READY, NOT SENT" though it was graded the same day (6.192e-9) and reappears in the calibration tables and rerun rows.
4. **The D5 age table differs between sources**: RESULTS.md stops at age 6; the Sep 22 packet adds age 7 (−8.13%) and age 8 (−5.28%) — those two rows have no artifact pointer in RESULTS.
5. **Age-5 wall penalty exists only at leaf 32** (v55 176 vs v50 160 ps/FLOP, ~+10%), not at leaf 64 (v56b 92.8 vs v31 91.8). ev.py carries 177 vs 164 for leaf 32; the mechanism (memory traffic, VmPeak 6.5 vs 5.8 GB) is a hypothesis.
6. **Corrector fresh-10 generalisation is thin**: −3.79% on 10 nets vs −5.60% DEV (1.81 pp against a 2 pp kill); the graded −5.22% and HOLDOUT −4.76% sit in between. The 400-net corrector's DEV −7.17% translated to graded raw −1.35% relative to v57 (DEV predicted −1.66%) — consistent, but its own absolute DEV→graded factor (1.0361) is the lowest of the eight.
7. **Two mechanism explanations were corrected after publication**: the H3 taper ("no compression events in the tail" → "joins at every layer 5–14; the sketch width fixes the rank") and P3's status ("D2's base turns it on at rank 16" → never on in any shipped file). Both are now consistent, but any downstream reasoning written between Sep 22 and Sep 23 inherited the wrong version.
8. **T0's unshared reference (2.3642e-8) vs v55's graded raw (2.461e-8)** implies the entire shared-basis compression costs only ~4% on the graded suite (vs +7.7% DEV cupy vs v53) — the exact-tower ceiling for this chain family without the corrector is ≈2.27e-8 board-scale, i.e. still 1.4× J2W's 1.64e-8.
9. **The `whest run` local wall figures** (52.02 s v31, 60.26 s C10, 45.07 s v50, 54.39 s v55, 90.55 s v56b) do not order the same way as graded walls (v56b local 90.5 s → graded 60.7 s) — local wall is not a server predictor and should be dropped from the row schema.
10. **The 3% §5 gate now measures against the team's own fresh best**, so a −7.17% DEV result produced a 1.87% "projected" margin and was blocked; the gate's intent (stop noise-chasing) and its effect (block the best score ever measured, 99/100 wins, ±0.81) have diverged.

---

## 6. Throughput

**Submissions per UTC day (graded):** Aug 27: 2 · Sep 5: 7 slots (4 graded, 3 failed) · Sep 7: 1 · Sep 8: 2 · Sep 10: 1 · Sep 14: 1 · Sep 16: 1 · Sep 17–21: **0** · Sep 22: **3** (C10, v50, v53) · Sep 23: **5** (v50 rerun, v55, v55 rerun, v50 control, v56b) · Sep 24: **2** (tr64, v57) · Sep 25: 1 so far (v60_ws400). Standing approval is ≤5 canaries/day plus one A/B safety slot/day; AIcrowd allows 10/day. Sep 22–25 average 2.75/day against 15 in the previous 25 days.

**Candidate evaluations per day (DEV-100-equivalent arms, from RESULTS):** Sep 21: 2 (D1 acceptance on both backends) · Sep 22: ~17 DEV arms (D5 rank/age ×8, H3 ×3, v46, v50 DEV+HOLDOUT, v53) + 302 CMA-ES evaluations (D6) + ~8 f64 cost runs + 12 ARC-repo arms (D4b) · Sep 23: ~50 DEV arms (T0–T2 ×15, T3, elasticity ×14, P3 ×2, adlam ×8, biv ×2, wallsafe ×3, N1 ceiling ×2) + 205 CMA-ES evaluations (T3) + 1,024 N1 tail continuations + tier ×3 (DEV 20) + depth-6 cost ×2 · Sep 24–25: ~15 DEV arms (anatomy, corrector single/sequential, HOLDOUT-600, fresh-10, adage ×5, v58_ws, v59, v60_ws400 on 40, v61) + ~200 ridge/MLP fold fits + adage 5-net cost ×3 + three multi-hour corrector fits.

**Time per probe:** cupy accuracy **2.03 s/net → 3.4 min per 100-net arm** (8.0 GiB); numpy (instrument of record) ~50 s/net → 38 min/100 with 4 workers (3.8 GiB each); one f64-twin cost run = one net, minutes; WSL memory gate 23 checks; `whest validate` 96–143 ms; N1 tail continuation 0.78 s; CMA-ES fits 302 evals (D6) / 205 evals (T3) on 10–25-net batches; sequential corrector fit **3h15m** (100 nets, worker-q 2095), **3h07m** (400 nets, worker-q 2112), **2h04m** (recursive variant, worker-q 2124); server grading ≈ 100 nets × ~60–95 s + queue. The measured DEV→graded error (±1%) means a 3.4-minute cupy arm predicts a graded score to ~1%; the binding delay is the daily slot budget and the numpy row for §5, not measurement.
