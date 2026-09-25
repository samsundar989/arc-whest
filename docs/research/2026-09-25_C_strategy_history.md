# C — Strategic history, competitive intelligence, and the ideas graveyard

*Reconstructed 2026-09-24/25 from the read-only snapshot at `/home/claude/arc-whest`: `docs/`, `docs/research/` (58 files), `docs/refs/algorithmic_contribution_puffi.pdf`, `archive/2026-09-21/` (61 docs + the 55k-line `experiments/LOG.md`), `experiments/board_snapshots/`, `RESULTS.md`, and the public `external/504aldo/` and `external/mlp_cumulant_propagation/` clones. Where a claim's only source is a project doc on claude.ai (not on disk) it is marked `[project doc]`.*

---

## 1. Timeline — Aug 4 → Sep 25

Scores are graded public adjusted scores unless marked local/DEV. Submission ids in `#`.

### 1.1 Phase 1 (256×32, λ=1e11): the sampling era (Aug 4–10)

| date | what | score | source |
|---|---|---|---|
| Aug 4 | v0: covariance propagation + offline per-layer calibration (`#323872`) | 2.49e-6 | `archive/.../experiments/LOG.md:3-105`, `docs/BENCHMARK.md` §2b |
| Aug 5 | v1.0 antithetic MC; v1.1 Sobol RQMC + exact layer 1 (`#324094`) | 3.893e-7 | LOG.md:148-235 |
| Aug 5 | v2.4: Sobol RQMC + dead-neuron pruning + rank-32 input rotation (`#324420`), rank ~88 | 2.7194e-7 | LOG.md:542 |
| Aug 5–6 | "score identity" discovered (adjusted = V·c/B, N cancels); B1 fusion, B3 learnability probe (R²=0.012), G1/G2, Track T shrinkage tail, Track P rank-k, Sprint 3 structure-aware allocation all fail their gates | — | LOG.md:235-933 |
| Aug 6 | Sprint 4/5: packed activation propagation (dead/on/kink, gathers) reproduces competitor 319341's counted arithmetic to 0.4% but **D14 gate FAIL 1.1216× vs 1.15×** — killed because gathers are cheap in the price table and expensive in wall time billed at λ=1e11 | — | `docs/BENCHMARK.md` §6–7; LOG.md:1509-2157 |
| Aug 6 | MOONSHOT infrastructure (low-rank K3/K4 adapter) built, "no gate opened" | — | LOG.md:1488, 2157 |
| Aug 10 | v2.4.1 graded (`#327743`), a draw with v2.4; Phase-1 final rank ~104/312 | 2.6314e-7 | BENCHMARK §1 |
| Aug 19 | carrier audit: input-space variance lane closed; degree-4/5 angular design worth 1.48× but needs 131,072 points (C/B 1.35) — unaffordable; mid-layer CV "dies by identity" | — | LOG.md:3426-6398 |

**Decision points.** The Phase-1 record is a chain of pre-registered gates that failed: every sampling refinement (fusion, control variates, shrinkage, packing, designs) bought ≤1.2× and the score identity (`adjusted = V·c/B`) proved N is not a lever. The board forensics (LOG.md:736; `RESEARCH_NOTES.md` §2g) found the public score exposes 50 fixed MLPs with per-MLP feedback (257 submissions over-determine the references) — recorded privately, never used, never posted.

### 1.2 Phase 2 launch: K2, then the Kerdock carrier detour (Aug 24 – Sep 2)

| date | what | score | source |
|---|---|---|---|
| Aug 24 | Live audit: shipped file routes width>256 to first-order K2 closure; leaders Puffi/pranav/Activation-Regions at 1.8–3.1e-8 spending 0.5–0.8 B; Puffi's exact FLOP count `1,099,802,385,000` is within 0.0264% of 32,768 dense forward directions = 16 Kerdock/MUB bases × 1024 × antipodes → **"16-basis structured angular carrier" declared highest-EV reproduction** | — | `archive/.../PHASE2_LIVE_AUDIT_2026-08-24.md` |
| Aug 21–26 | P2-CP1 baseline ladder on block P01; P2-CP3R: 94.6% of K2 error is accumulated (μ,σ) state drift, readout ceiling 4.5%; P2-CAL: per-layer scalar calibration exhausted at 2.96× | — | LOG.md:8916-15606 |
| ~Aug 26–28 | v2.6.1 K2 canary (`#328662`) 3.933e-7; v2.7 calibrated K2 (`#328664`) **1.374e-7**, rank 53/83 on Aug 31 | 1.374e-7 | `PHASE2_FRONTIER_AUDIT_2026-08-31.md` |
| Aug 25 | Phase 1A: width-1024 real-MUB Kerdock carrier built and verified (`KERDOCK_MUB_SPEC.md`); K6/K8 predictions corrected (no independent reference generator exists) | — | `archive/.../KERDOCK_MUB_SPEC.md` |
| Aug 27 | **P-A: factorized K3 carries the information — raw 4.0047e-8 on 24 nets, 3.9249e-8 on organiser labels** (K3 confirmed as the accuracy oracle) | — | LOG.md:16802-17683 |
| Aug 28 | W-EFF (exact sparse-aware factorized K3) prices at **1.0732 B** vs a ≤0.35 B screen → **killed**; late-window K3 killed (2.1–4.7× worse than v2.7); rank-capping K3 "priced out" | — | LOG.md:19431-20725 |
| Aug 31 | GAME_PLAN rev 9: two nominees — A = structured angular carrier (Kerdock/MUB + exact radial), B = compact analytic (AUGMENT-K2 + query-directed/adjoint K3). WP1 (legal fnp carrier) and WP2 (truth-free kernel objective) built; WP3 AUGMENT priced +3.7%; WP4 adjoint K3 killed then kill retracted by Skeptic | — | `docs/archive/GAME_PLAN_before_2026-09-10_audit.md`; LOG.md:21368-23760 |
| Sep 2 | **SCREEN WP12: the carrier family is FLAT** — adjusted score flat to 0.62% over a 4× cost range; structured advantage of Kerdock over random orthogonal at width 1024 = 0.155%. Carrier lane closed by the score identity. WP9/WP10: per-neuron correctors on K2 state — readout 0.99×, in-flight variance 17× worse, in-flight mean needs R² ≥ 99.85% | — | `STRATEGIST_PACKET_2026-09-02.md` §3 |

**Decision points.** Aug 24's Puffi FLOP-fingerprint inference drove a week of Kerdock work that the Sep 2 identity screen closed ("a pure sampler cannot climb this leaderboard"). Aug 28's W-EFF kill (priced against a 0.35 B floor screen, not the score) was later called "the right measurement and the wrong decision" (`PHASE2_STRATEGY_2026-09-05.md` §0).

### 1.3 The regenerating-K3 flow and its compression program (Sep 2–5)

WP13–WP20: a 0.077 B "regenerating" K3 flow (κ₃ rebuilt each layer from its (3,) and (2,1) read slices, the all-distinct block discarded). Five routes to recover the discarded object, each with a measured ceiling: selectors 1.45–2.10× *worse* than nothing; Σ's range = chance; learned E3 vector ceiling 0.258, E21 matrix ceiling 0.167; carried three-generation sketch **3.33×** end-to-end against a family ceiling 3.874× and a 10× kill line; oracle joint injection 11.1× (rank 128) / 334× (rank 512). Canary4 (`#329896`, Sep 5) graded **8.50e-8** at the 0.1 floor, rank 47/109 (`STRATEGIST_PACKET_2026-09-03/04.md`; `writeup-p2-draft.md` §7).

### 1.4 The exact K3 tower and the AUGMENT discovery (Sep 5–10)

| date | what | score | source |
|---|---|---|---|
| Sep 5 | Strategy rev 10: **reverse the W-EFF kill**; telemetry says the frontier is a deterministic K3-class chain at ~0.2 B + a ~2× accuracy layer | — | `PHASE2_STRATEGY_2026-09-05.md` |
| Sep 7 | `estimator_k3_full.py` (`#330212`), 0.78 B, raw 3.93e-8 | 2.9467e-8 | addendum Sep 7 |
| Sep 8 | batched Strassen–Winograd L=3 (`#330327`, 0.537 B, 61.9 s) and L=4 (`#330290`, 0.486 B, **114.7 s worst wall**) | 2.031e-8 / 1.836e-8 (~#27) | addendum Sep 8 |
| Sep 8 | WP22/O1: output renormalisation is worth 0.75% — "the residual is structure, not scale" | — | LOG.md:34810 |
| Sep 9 | **D1: ARC `kind=AUGMENT` in torch = 1.912e-8 vs SIMPLE 4.005e-8 (2.09×, 24/24)** — reproduces the leaders' raw floor; ablation: all of the gain is κ₄'s (n,n) structure, κ₄ zeroed is 9.2× worse | — | LOG.md:36353-36843 |
| Sep 10 | Factored AUGMENT port is 3.6–5.8× over the hard budget; its gain vanishes under truncation ("κ₄ pays only in an exact tower" — later retracted); symmetric packing 1.0067×; exact metric 1.07×; T1 AUGMENT-lite 74–109× worse; T2 learned corrector on exact-K3 chain 1.01× | — | LOG.md:38441-40404 |
| Sep 10 | Nominee A moved to Strassen depth 2 (`#330566`, 0.604 B, 33 s) under the ≥2× wall-margin rule | 2.284e-8 | `STRATEGIST_RESPONSE_2026-09-10.md` §4 |

### 1.5 The Codex "toy fixture" detour and the 504aldo pivot (Sep 10–16)

Sep 10–13: a worker built exact signed-law K4 fixtures (axis-Gauss, full-polar bank, causal signed K4 increments, nonlinear old-history, integrated first-source cost meter — §5 below). The Sep 12 strategist ruled **"NO-GO as the next submission project: the integrated causal signed-law K4 architecture"** and redirected to one exact contraction fusion in the working estimator (`STRATEGIST_RESPONSE_2026-09-12.md`). That fusion became **F21** (`#330950`, Sep 14): **1.5957e-8**, rank 58–62 — the last pre-504aldo file.

Sep 9–10: team **504aldo** published an MIT-licensed factorized-K3 estimator at 5.4e-9 with a 25-item dead-end catalogue (§2.16). Sep 14 evening strategist verdict: *"The family is right. The product is wrong."* E14 (the K4 "tree" with a label-free Gaussian-tree λ) was stopped by a same-night experiment: its λ has the opposite sign from layer 4 on, R² ≤ 0.2 (`STRATEGIST_RESPONSE_2026-09-14.md` §3). Samuel's decisions: re-implement 504aldo's mechanisms in own code, clone as parity oracle only (Sep 15), and **not** ask the organizers the eligibility question (Sep 16; `docs/PROVENANCE_504aldo.md`).

| date | what | score | source |
|---|---|---|---|
| Sep 15 | P1 κ₄ regeneration (own λ) 1.2371× on 24/24; P2 7→4 contraction identity 0.7128× cost | — | `CLAIMS_LEDGER.md` rows 3–14 |
| Sep 16 | **v31** (`#331317`) raw 2.907e-8, 0.301 B, 68 s | **8.752e-9** (rank 58→71) | RESULTS.md rows |
| Sep 16–18 | closure-equivalence audit (v32 has 25 terms vs v17's 87), PAIR/Q closures, P3 feedback, qEc/qEd bases, N1 merged Tucker core at n=96, six-arm P12 factorial → **C10** selected (local 6.49e-9); rank-32 old-query approximation 0.57× (catastrophic); 143 commits and 0 submissions on Sep 18 | — | `STRATEGIST_PACKET_2026-09-18_COMPLETE_SLICE.md`; `PLAN.md` §2 |

### 1.6 The Sep 21 reset and the fast loop (Sep 21–25)

PLAN rev 12 (Sep 21) archived the pre-registration process, unlocked the organizer public labels (DEV 100 / FIT 100 / HOLDOUT 800–900), gave standing approval for ≤5 canaries/day at a ≥3% DEV gate, and built `tools/ev.py` (100 nets in minutes on cupy; every graded row since within 1% of projection).

| date | file | graded | key change | source |
|---|---|---:|---|---|
| Sep 22 | C10 `#331816` | 6.4914e-9 | age-4/rank-256 shared old tier, P4 depth 5 | RESULTS.md |
| Sep 22 | v50 `#331863` | 6.192e-9 | D2 fitted profiles (+2.7%) + D3 billing (−2.26%) | RESULTS.md §D3/D6 |
| Sep 22 | v53 `#331880` | 5.5603e-9 | **age 4→5: −10.32% score from one integer** ("structure beats fitting") | RESULTS.md §D5 |
| Sep 23 | v55 `#331967/#331980` | 5.3200e-9 | P2 residual rank 32→4 (elasticity table); worst wall 114 s on a noisy day | RESULTS.md |
| Sep 23 | v56b `#332063` | 5.7076e-9 | Strassen leaf 64: **wall law** — kernel ps/FLOP set by leaf size, 60.7 s | CLAIMS_LEDGER 46–52 |
| Sep 24 | tr64 `#332088` | 5.4341e-9 | mixed leaf, 80.3 s | |
| Sep 24 | **v57 `#332129`** | **5.0443e-9** | **in-loop learned mean corrector (33→32→32→1 tanh MLP per layer, sequential teacher forcing on organiser all-layer labels): −5.22% raw graded** | RESULTS.md §Step 2 |
| Sep 25 | v60_ws400 `#332223` | 5.3387e-9 | 400-net corrector at leaf 64; 57.6 s; designation A | `DESIGNATION.md` |
| Sep 25 | v59_corr400 | 4.9763e-9 (derived) | same corrector at leaf 32 — **blocked by PLAN §5's own 3% gate** | `STRATEGIST_PACKET_2026-09-25.md` §3 |

Closed by measurement in this window: single-leg tower (T0–T3), N1, tiered ranks, P3 at any rank, per-net λ, closure tail, joint refit, Strassen depth 6, K4/unfactored AUGMENT (proxy), adaptive age, recursive corrector feature, hidden-64.

---

## 2. Competitive intelligence dossier

Telemetry fields: `wall` = participant_predict_wall_s (min/median/max over 100 MLPs), `kern`/`ovh` = flopscope server kernel/overhead wall, `res` = residual (cap 0.4 s). Primary telemetry table: `experiments/board_snapshots/board_2026-09-05_telemetry.md`. Board snapshots: `experiments/board_snapshots/2026-09-16T1611Z_*.md`, `docs/research/2026-09-1{0,2,4,6,8}_competition_*.md`, `docs/PLAN.md` §1 (Sep 21), `docs/STRATEGIST_RESPONSE_2026-09-24.md` §0 (Sep 23).

**Board trajectory of the top (adjusted):** Aug 24 Puffi 1.80e-8 (0.50 B) → Aug 31 J2W 5.60e-9 (raw 2.25e-8) → Sep 5 marius/suliman 3.80e-9 → Sep 10 marius 2.70e-9 (raw 1.71e-8) → Sep 14 suliman 2.54e-9 → Sep 16 marius 2.448e-9 (raw 1.678e-8, 0.1459 B) → Sep 18 J2W 2.158e-9 (raw 1.645e-8, 0.1312 B) → Sep 21 J2W 2.10e-9. Best raw ever posted (Insights): 1.50e-8 (Sep 14–16). All-layers/final error ratio for the leader 0.41–0.45, same as ours (0.41) — depth-accumulated truncation error.

### 2.1 J2W (joe_wanza) — #1
- Sep 5 (`#329787`): 3.93e-9, raw 2.02e-8, all-layers 1.07e-8, FLOPs **4.21–4.36e11 variable**, util 0.195, wall **99/104/119 s** ("at both caps"), res 0.337/0.354/**0.389**.
- Sep 18 (`#331502`): 2.158e-9, raw 1.645e-8, 13.12% (mean F 2.88e11), per-net costs vary ±6%; `#331435` had a **TIME_EXHAUSTED** sealed net and contradictory failure labels (`2026-09-18_competition_snapshot.md`).
- Sep 21–23: 2.10e-9, raw 1.64e-8, 0.131 B, **303–313 entries**, walls 105–117 s, **three sealed-net TIME_EXHAUSTED failures** on the board entry (`PLAN.md` §1, §6-D7; `STRATEGIST_RESPONSE_2026-09-24.md` §0).
- Phase 1: #2 at 5.39e-8 (`#319049`), 942 entries, flagged "probing" in the Aug 6 census (8.40× late jump) — `docs/BENCHMARK.md` §3.
- Inference (`RESP 09-24` §1): kernel rate ≈340 ps/FLOP (vs our 92–176) ⇒ leaf-16-class Strassen or einsum-heavy contractions; nominal un-Strassened chain ~20% leaner than ours; raw 1.87→1.69→1.64e-8 in small steps over ~300 submissions "is what constant-fitting looks like" [I]. Would lose ~3% of nets to timeouts on a noisy private-rerun day.

### 2.2 suliman_tadros — #2/#3
- Appeared Sep 5 "NEW, 13 entries", tied #1 at 3.80e-9 (`#329895`): raw 2.03e-8, all-layers 8.13e-9, FLOPs **3.89–4.31e11 variable**, wall **41/44/52 s**, res 0.119–0.136 — the fastest and lowest-residual leader.
- Sep 14 (`#330876`): 2.540e-9, raw 1.891e-8, 13.43%, per-net FLOPs 2.77–3.11e11 (±6%), walls 71–94 s, all-layers 7.82e-9, 50/50 clean.
- Sep 23: 2.2e-9, raw 1.63e-8, 0.138, **49 entries**, walls 84–108 s, 0 failures. "suliman's count says it is not brute-force server tuning" (`RESP 09-24` §1). Reaching #1 in 13 entries was read (Sep 5) as "the method is reproducible from public material".
- Phase 1: rank 65, 76 entries, plausible-honest.

### 2.3 marius_binner — #1–#3
- Sep 5 (`#329857`): 3.83e-9, raw **1.87e-8** (best raw of the top), all-layers 7.38e-9, FLOPs **4.504e11 constant**, wall 70/74/79, res 0.286–0.306, setup 1.1 s, ~27k ops, 7 GFLOP/s.
- Sep 10: new #1 at 2.70e-9, raw 1.71e-8, ~0.158 B — **below 504aldo's f64 "K3 + memoryless κ₄" ceiling (1.94e-8)**, the datum that says leaders carry κ₄ content beyond a memoryless core (`2026-09-14_public_frontier_504aldo_mechanism.md` §5).
- Sep 16 (`#331308`): 2.448e-9, raw 1.678e-8, 0.1459; public walls reach 111.2 s. Sep 21: 2.40e-9, 0.146 B, 151 entries; Sep 23: 2.3e-9, raw 1.56e-8, 0.147. "marius is resubmitting identical bytes to find a safe pair" (`PLAN.md` D7).
- **Phase-1 write-up (forum 18166 PDF)**: Kerdock tight-design sampler, raw 2.6e-7 at 0.37 B, var·k 2.27× below iid MC; §8.1 "weights arrive as float64; NumPy silently promotes… 2.00× loss"; **discloses a 5-level Strassen–Winograd recursion billed "close to the analytical count" (−12.5/−23.4/−33% per 1/2/3 levels), 79% of FLOPs through Strassen leaves in 10,402 calls**; his §6.4 explains kmax=3 was 3.04 B at Phase-1 shape; §8 obs. 4: best-of-N selection over hundreds of entries flatters apparent MSE by 33–39% (`PHASE2_STRATEGY_2026-09-05.md` §1–2; 504aldo `findings_log.md:1390-1398`).

### 2.4 Puffi (andrew_epstein / Cipo) — #2–#5
- Aug 24 (`#328223`): #1 at 1.80e-8, raw 3.603e-8, 1.10e12 FLOPs (50.01%), all-layers 1.41e-8 — the "32,768 directions" fingerprint (`PHASE2_LIVE_AUDIT_2026-08-24.md`). Aug 31 (`#329181`): 5.892e-9, raw 2.262e-8, 5.73e11.
- Sep 5 (`#329816`): 3.88e-9, raw 2.13e-8, FLOPs 3.87–4.31e11 variable, wall 86/91/98, res 0.18–0.35. Board-wide Sep 4–5 drops (Puffi 0.209→0.157 at flat raw) attributed by 504aldo to 1–3 Strassen levels (`findings_log.md:1310-1398`).
- Sep 14–21: 2.8e-9, raw 1.94e-8, 0.146 B, 78 entries — the worst raw of the top six; "Phase-1 SSC author sits last of the six ⇒ low-rank traceless carriers saturate at the same floor" (`RESP 09-09` §2).
- Public: Phase-1 write-ups thread 18175 (score system = exact radial integration + Kerdock/Walsh MUB angular design + compact analytic prefix "A716" with marginal κ₃/κ₄ and a **10-channel learned in-flight adjustment per layer, 90 coefficients fitted to layer-8 residuals** + mid-network control variate; raw 1.89e-7, adj 9.09e-8) and the SSC PDF (§3 below). 504aldo rebuilt A716 at 1024×16: 9.8e-7 (dead end #3).

### 2.5 a_s6 — #3–#5
- Aug 26–28: 2.719e-7 at the 0.1 floor (10 entries). Sep 16: #5 at 2.9e-9, raw 2.06e-8, 0.1408, 73 entries. Sep 21: #3 at 2.50e-9, **0.126 B** (lowest util of the top), implied raw 1.98e-8; "flat per-layer error profile — not paying for intermediate-layer accuracy" (`PLAN.md` §3). No public statement found.

### 2.6 oqaris — #5–#6
- Sep 5 (`#329737`): 4.30e-9, raw 2.06e-8, all-layers 7.69e-9, **4.60e11 constant**, wall 86/91/109, res 0.315–0.363. Sep 16: 3.1e-9, raw 2.15e-8, 0.1442, 93 entries. Deterministic chain profile.

### 2.7 reds, ben3, thegamers, Camaro, mliston, hyojun_kwon, fklassen
- **reds**: Sep 16 #11 at 4.2e-9, raw 2.14e-8, 0.196, 38 entries; Sep 21 tied #5 at 2.80e-9. Listed with J2W/marius/ben3 as below the 1.94e-8 memoryless-κ₄ ceiling (`PLAN.md` §3).
- **ben3**: Phase 1 rank 39 (1.451e-7, 0.709 util); Sep 16 #8 at 3.5e-9, raw **1.81e-8**, 0.1934, 16 entries.
- **thegamers**: Sep 10 #5 at 3.60e-9, raw 1.79e-8 at 0.212 B ("our cost tier" per 504aldo); Sep 16 #9 3.6e-9, 37 entries.
- **Camaro**: Sep 5 (`#329762`) 8.70e-9, raw 2.73e-8, 6.78–7.23e11, wall 54–63, res 0.27–0.31; "+25 places in 1h" on Sep 1; Sep 16 #7 3.4e-9, raw 1.89e-8, 0.18. Server per-op residual 12.17 µs (LOG.md:34318).
- **mliston**: Phase-1 write-up 18170 — cumulant closure to L18 + layer-22 control variate (coef 0.6) + **frozen 52k-param GRU centering corrector (+17–19% held-out)**; forum 18108 (Jul 28) proposed fixing the wall-time channel (cap residual / λ→1e12 / analytic-only). Phase 2: Sep 5 1.44e-8 (0.455 B, 985–1040e9 variable); Sep 16 #11 4.2e-9, raw 2.13e-8; Sep 21 #10 3.50e-9.
- **hyojun_kwon**: Phase-1 write-up 18154 — Gram–Charlier skew-augmented closure with Mehler co-skewness + **mid-network (layer-6) control variate, 2.39× held-out (110× oracle at penultimate)**; adj 2.90e-7 despite the in-pipeline win. Phase 2 (`#329812`): 6.41e-9, raw 2.83e-8, 4.99e11 constant, setup 1.3 s; Sep 16 #10 3.8e-9, raw 2.7e-8, 0.1407.
- **fklassen**: Sep 5 5.44e-9, raw 2.50e-8, 4.7–4.8e11; Sep 16 #13 4.5e-9.

### 2.8 jamespayor, lode_dockx, MeatProxy, Activation-Regions, ArtemVoronov, AndreasHad04
- **jamespayor**: the dense-matmul datum. Sep 5 (`#329375`) 9.41e-9, raw **1.96e-8**, 1.06e12 constant, util 0.482, wall **16–17 s**, 85 GFLOP/s, res 0.10 — "a straightforward dense factorized K3 at c″₃≈3.9" (`PHASE2_STRATEGY_2026-09-05.md` §2.5). Sep 16 #47 6.4e-9, raw 1.97e-8, 0.325.
- **lode_dockx**: **final-row-only** (all-layers MSE 0.70): Sep 5 7.06e-9, raw 4.78e-8, 0.148, wall 32–36 s; Sep 16 5.1e-9, raw 4.25e-8, **0.120 B**. Cited in `PLAN.md` §3 as proof that intermediate-layer accuracy need not be paid for.
- **MeatProxy**: Aug 26 #5 at 1.71e-8 (raw 3.76e-8, 0.4548); Aug 31 #10 1.06e-8 (`#328595`); Sep 16 #33 5.4e-9, raw 2.99e-8, 0.1806 — the raw says it is not a 504aldo clone.
- **Activation-Regions (emil_partow)**: Aug 24 #3 at 3.08e-8 (81% B); Sep 5 (`#329501`) 5.39e-9, raw 2.80e-8, all-layers 3.4e-4, **8 non-completed rows, hits the 120 s cap**; Sep 16 #33 5.4e-9. Name suggests kink-region (SOX) lineage.
- **ArtemVoronov**: 1.83e-8, raw 2.33e-8 at **0.787 B** — evidence that the raw floor does not move with budget. **AndreasHad04**: 2.8e11 FLOPs at 103 s wall (2.9 GFLOP/s), raw 9.57e-8 at 0.127 — FLOP-cheap/wall-expensive ops.

### 2.9 504aldo (public, MIT)
- Trajectory (`external/504aldo/docs/community_post.md` §1): V16b 2.22e-8 (raw 4.46e-8, 0.4986) → V17 +κ₄ regeneration 1.11e-8 (raw 2.26e-8) → V18 +D21 feedback rank 16 1.06e-8 → V19 +final trim 9.95e-9 → V22/V24 shared basis 8.10/7.90e-9 → V25 adaptive λ 7.80e-9 → V26–V29 Strassen 7.37 → 5.70 → **5.40e-9 (raw 2.13e-8, 0.2526 B)**. f64 ceiling of its closure ~1.94e-8 (LB scale). 14–16 accounts submitted byte-identical clones (rank 42–57 at 5.40e-9); derivatives trimmed 7–22% sit at 4.2–5.3e-9 (ranks 13–41 on Sep 21).
- Telemetry Sep 5 (`#329644`): 8.06e11 constant, wall 31–37 s, 27 GFLOP/s, res 0.059–0.072 — "few big ops".
- Statements: "the leaders' bill equals ours minus the old tier, to three digits" (0.150 B); "top methods carry fourth-cumulant content beyond a memoryless core AND old-source content at roughly the cost of our young tier alone… not a per-source hub-tensor expansion"; "asked the sponsor on 2026-09-05 whether Strassen expressed this way is within the fair-accounting rule"; Sep 13 reply: **sponsor confirmed on Sep 11 that metered Strassen–Winograd has no recursion-depth limit** (participant-reported); "others have reduced cost another 7–22%". Grader notes §7: 5 s setup cap fails the whole submission; smoke test uses non-suite shapes (depth 32); f64 weights billed 2×; residual dominated by buffer allocation, `out=` gives ~0.02 ms/op.

### 2.10 ednacob (Phase-1 winner) [project doc `claude/phase2-intel-2026-09-01.md`]
- Post-rescore #1 `#327208`: adjusted 1.845e-8 = raw 3.63e-8 × 0.51, 111–169 entries, plausible-honest. jamesrahenry's census (forum 18157): final-layer-only, cubature-class, raw 5.2× below the saturated Kerdock band, "almost exactly matches the signed reweighting oracle ceiling" (bulzan 18183: per-network reweighting of 129 frame endpoints removes ~96% of error, 82% survives cross-fitting). No write-up or code ever appeared; the "~Sep 30 harvest" watch was withdrawn (`PHASE2_STRATEGY_2026-09-05.md` §1). Phase-1 #2 J2W 5.39e-8, #3 dstepanov 6.22e-8 (411 entries, flagged probing).

### 2.11 Cross-cutting telemetry facts
- Top-8 cluster (Sep 5): util 0.18–0.23, wall 70–120 s, res 0.29–0.39, overhead 10–13 s ⇒ ~25–35k ops, kernel 5–10 GFLOP/s; all-layers MSE 7–11e-9 < final 1.9–2.8e-8; raw floor 1.87–2.33e-8 across util 0.18→0.79 ("a bias floor, not a variance floor").
- Fixed per-MLP FLOPs: marius, oqaris, hyojun, jamespayor, 504aldo. Variable (±6%): suliman, J2W, Puffi. Two adaptive hypotheses tested on our chain (per-net λ, per-lineage age) both died because our per-net statistics are too uniform (CV 1.8–9%) (`STRATEGIST_PACKET_2026-09-25.md` §4).
- Server per-op residual is flat at 10–13 µs/op; per-op overhead ~0.36 ms; kernel ps/FLOP set by Strassen leaf size (leaf 32: 155–176, leaf 64: 90–95, mixed 141); wall tail is the day (worst/median 1.03–1.055 quiet, 1.20–1.235 noisy), not the net (per-net wall correlation 0.03 between identical-bytes runs).
- **Never verified about anyone:** point family, carrier, closure, or corrector. Every audit repeats "telemetry is an envelope, not a fingerprint" after the Aug 24 Kerdock inference was falsified by the Sep 2 identity.

---

## 3. What Puffi's write-up says

`docs/refs/algorithmic_contribution_puffi.pdf` — "Separate-Universe Response Closure for Deep ReLU Networks: submission #327725", Cipo on behalf of Team Puffi (Phase 1, n=256, L=32, B=2.72e11). Extracted with `pdftotext`; summary also in `archive/2026-09-21/docs/SSC_PUFFI.md`.

**Method.** A deduction–projection estimator: carry mean, covariance and structured κ₃ (CP columns, `Sym(v,M)` atoms, shared-middle and paired groups — ARC's S.4.3 carrier algebra), and **regenerate κ₄ after every ReLU from p ≤ 4 covariance-response modes** coupled by a small fitted matrix Λ:

`κ₄,ijkl = Σ_ab Λ_ab (B_a,ij B_b,kl + B_a,ik B_b,jl + B_a,il B_b,jk)` (Eq. 2)

Modes (Eq. C2): B₁ = C, B₂ = Q/v̄ (Q a frozen six-coefficient quadratic form of the live state, Eq. C3), B₃ = v̄I, B₄ = ½(g_i+g_j)C_ij with g the standardised centred gate probabilities. Readout is a per-neuron scalar `κ₄,j = 3 q_jᵀ Λ q_j`, `q_aj = w_jᵀ B_a w_j` (Eq. 3), consumed as the fourth-order Edgeworth contribution to the next ReLU mean. Λ may be indefinite ("signed ansatz"); the dense n⁴ tensor is never formed. Deployed extras: up to 64 fresh cross-family κ₃ atoms capped to 4; one-layer response memory (β = −0.3, Eq. D8); pair-series degrees 4 for (1,1) and 2 for (2,1); a **recurrent drift corrector 31→32→32→1 tanh** on 31 deterministic features (Eq. D9–D11: α, gate probability, κ_iii/σ³, log-variance ratios, C⁺a, (C⁺)²a, R⁺g, K⁺1, top-8 neighbour sums, softmax-weighted neighbour averages…), active at post-ReLU layers 9–31.

**Fitting (§3.3).** Q's six coefficients by per-layer ridge to κ_iiij+κ_jjji; Λ by ridge to the measured marginal fourth cumulant of the next preactivation (ridge value chosen on dev nets by the MSE of the resulting Edgeworth mean term); response-memory coefficient frozen at −0.3; drift corrector fitted on 160 public Phase-1 nets, checked on 16 held out, **teacher-forced on one-step mean drift, not through the score**. Nothing fitted at inference.

**Numbers.** On a matched 30-net panel: layer-32 MSE **−77.2% vs ARC factorized K=3** (95% CI 68.9–83.2), −48.2% vs factorized K=4 (30.3–59.4); named-op cost 0.501 B vs 0.373 B for the parity-checked K=3 port (**+34.3% ops**); K=4 ~10² B. Depth exponent (variance-normalised, L=8–32): SSC 2.21 [1.97,2.51] vs K3 2.69 vs K4 4.13. Response-branch ablation (Fig. D1): disabling it raises layer-32 MSE 2.728e-6 → 7.698e-6 (−64.6%, lower on 30/30). Graded `#327725`: raw 2.6597e-6, adjusted 1.6533e-6 at 0.622 B — **7.94× worse than full-budget Kerdock QMC and 2.94× worse than iid MC at layer 32**; "#327950 scored better" (their score entry). Reference noise Var/1e8 = 6.82e-9 matches the analytic layer-1 MSE 6.80e-9.

**Self-stated caveats.** No small expansion parameter, no truncation bound; the −77.2% is the whole deployed stack, not Eq. (2) alone; one width tests depth only.

**The unshipped improvement (§4).** Refitting Λ to *the mean correction consumed after the next weight and ReLU update* instead of marginal κ₄ cut layer-32 MSE a further **13.6%** (26/30 nets); "the measured reduction came from changing the target"; never embedded in #327725. "In Phase 2, we plan to investigate other deduction–projection estimators."

**How the repo used it.** Sep 5: "the most evidence-backed accuracy layer". Sep 8: reframed as one route to "κ₄-class information in flight". Sep 9: the board (Puffi worst of six) says low-rank traceless carriers saturate at the AUGMENT floor. 504aldo's F82: A716 at 1024×16 = 9.8e-7. 504aldo's F68 explains B₁ = C: the κ₄ core is, to R² 0.66–0.97, λ·C_off. The drift corrector's teacher-forced design is the direct ancestor of the Sep 24 corrector that finally worked.

---

## 4. Organizer statements about what works (with sources)

1. **Jacob Hilton, LessWrong announcement comments** (`claude/phase2-intel-2026-09-01.md` §2 [project doc]; quoted in `PHASE2_STRATEGY_2026-09-05.md` §2, `STRATEGIST_RESPONSE_2026-09-10.md` §1, `PLAN.md` §3, `STRATEGIST_RESPONSE_2026-09-24.md` §1): warm-up leaders ran *"variants on the factorized 3rd cumulant propagation algorithm … combined with learned networks consuming the cumulant estimates as features"*; depth was raised specifically to advantage "approaches going beyond basic cumulant propagation."
2. **Phase-2 launch post (forum 18197, Aug 23)**: networks made 4× wider / 2× shallower "specifically to make cumulant-propagation-style methods more competitive while retaining difficult depth dependence"; Phase-1 winners were sampling variants; clean-compute policy (no raw arrays, FFI, threads) (`2026-09-10_competition_sources.md`; 504aldo F41).
3. **Townhall (forum 18078)**: higher-K cumulant propagation "improves results initially but eventually diverges"; Borel resummation explored, unsolved; ARC expects cumulant propagation to beat QMC long-term; **"low-rank refinement (exploiting decreasing effective rank at depth)" named the most promising direction**; offline training on external MC data is allowed and unmetered; algorithmic prize favours "meaningful mechanistic analysis that measurably improved the score" over "opaque LLM-optimized code" (`RESEARCH_NOTES.md` §3, §5; intel 09-01 §2).
4. **ARC blog "random products" post**: deduction–projection estimators + "mechanistic L² sketching" — project onto leading eigenfunctions of E_f[f fᵀ] so the readout still sees the same expectation (the theoretical blessing cited for query-directed compression, N1) (intel 09-01 §2).
5. **ARC repo `TODO.md`** (`external/mlp_cumulant_propagation/TODO.md`): factorized AUGMENT "is now defined as the augmented term set minus the terms it cannot afford (`factored_keeps_term`: non-hypertree top-slice diagrams, e.g. covariance cycles, and products of the factored top-degree cumulant's all-distinct block with other blocks). It is therefore NOT equivalent to unfactored AUGMENT; the dropped terms have the same Θ(n^-k_max) squared size as the extra augmented terms that are kept… The paper needs updating." Also: the augmented Edgeworth cap is K+1, contra S.4.2. Repo priced (D4b, proxy): restoring those diagrams is 1.15×/2.18× worse on score (`CLAIMS_LEDGER` 38).
6. **Residual-billing rulings (forum 18122/18132)** (`GOTCHAS.md:1770`): eval servers ship `flopscope-client`, arrays are `RemoteArray`s so raw-array bypass "is impossible there"; "we will examine this during the private re-evaluation and prize due diligence"; "using residual pricing as an alternative arithmetic budget is not [legitimate]".
7. **Rules v14** (`2026-09-10_competition_sources.md`): §5.5/§5.7 honest counted computation, "genuine algorithm/precision/symmetry savings encouraged"; §7 **"only that Participant's best-scoring such submission is ranked"** (see §6 below); §6 write-up must correspond to one successfully evaluated submission.
8. **Sponsor on Strassen (participant-reported by 504aldo, Sep 13)**: metered Strassen–Winograd permitted with no recursion-depth limit; Phase-1 #1 "disclosed the same technique".
9. **AIcrowd (Mohanty)**: Phase-1 extension (18109) because of flopscope repricing; "every submission… will be re-evaluated on the updated cost model"; error-message leak acknowledged and to be fixed (18217, Sep 9).
10. **"Ridge regression and gradient boosting"** — `PLAN.md` §3 and `RESP 09-24` §1 cite "a leaked grader message from a top team". **No primary source exists in the repository**; the Sep 10/14/16/18 audits state the 18217 leak screenshot was deliberately not opened. Treat as unsourced hearsay.

---

## 5. Ideas graveyard

Format: description · claimed potential · why dropped · does the reason still hold now (corrector landed −7% raw; chain 0.216 B; wall-bound; PLAN §5 gate binding).

**5.1 MOONSHOT — low-rank K3/K4 cumulant flow (Aug 6; `archive/.../MOONSHOT.md`, LOG.md:1488, 2157).** Phase-1 plan for a compact joint-state flow. Adapter built, "no gate opened"; superseded by the Phase-2 program. Its surviving idea (compact joint state) became the K3 chain — effectively executed.

**5.2 FRONTIER_ARCHITECTURE_PROPOSAL (Phase 1, 256×32).** Stub only; recoverable in git history not on disk. Dead by shape change.

**5.3 Kerdock/MUB angular carrier (`KERDOCK_MUB_SPEC.md`, WP1/WP2, Phase 1A).** 16 real MUBs at width 1024 with antipodes and exact radial factor; motivated by Puffi's 32,768-direction FLOP fingerprint; claimed potential: match the Aug leaders at ~0.5 B. Killed by measurement: MMD² = C1/k + C0 exactly (score flat to 0.62% over 4× cost), structured advantage 0.155% over random orthogonal; sampling at 1024×16 sits ~1.2e-6 (504aldo F42). **Reason holds** — the identity is algebraic and λ=0 does not change it.

**5.4 Packed activation propagation / dead-on-kink (Phase 1 Sprints 4–5).** Reproduced 319341's arithmetic to 0.4%; killed because gathers bill 4 FLOPs/elt but cost wall time at λ=1e11. **That reason no longer holds (λ=0)**, but the family is dead by 5.3 anyway.

**5.5 Query-directed / adjoint K3 (WP4, Aug 31; H4 "linear-response hand-off", PLAN D5; 504aldo #20).** Back-propagate final-output adjoints and contract each layer-local κ₃ source directly: `δm_L ≈ Σ_l J_{L←l} S_l^(3)`. Potential: full-K3 accuracy without the recurrent tower. WP4 was killed on a preregistered falsifier, the kill retracted (the statistic kills a provably linear system), and the instrument never re-run. Sep 14: "dead for cost at this shape" — D21 is consumed by the closure at every layer so the backward object is rank-n per readout neuron; 504aldo #20 found no cost drop. H4 (after age 4 read each source out once through J, keep intermediate effect only via rank-16 feedback) was struck by inference on Sep 22 and then "superseded by T1–T3": T1 shows dropping old sources' *intermediate* D21 contributions costs 8.9–27× raw, which bounds H4's premise from above. **Never measured as specified; the T1 bound makes it very unlikely to survive.**

**5.6 Sensitivity-weighted old-tier basis / merged core (N1, Sep 14–23).** Choose the shared old-source basis by output sensitivity (balanced truncation) rather than leg energy, then merge all old sources into one Tucker core (cost flat in source count) with per-net adaptive rank — the one hypothesis that fit the leaders' 0.13 B and ±6% cost spread. Sep 16: mean-path relevance proxy rejected; merged core metered at n=96 with joins dominating; Sep 17 8-net screen registration-deficient. **Sep 23 kill by measurement**: energy and sensitivity orderings agree (Spearman 0.994/0.958; top-256 overlap 249/256); rank-512 ceiling only −6.9% raw. The *adaptive-rank-per-net* half was never run (N1 died first). **Reason holds** for the basis; the merged Tucker core was never built at 1024 and 504aldo priced it at r³/n² (10.7 u at 224, 54 u at 384) — dead by arithmetic unless r ≲ 150, which the rank cliff forbids.

**5.7 The single-leg diagonal-born tower (T0–T5, Sep 22–23).** Carry one transport map M per source with diagonal births (κ₃ sb, κ₄ q, κ₅), regenerate correlation-born κ₃ memoryless with fitted ρ. Potential: 0.11–0.15 B at unshared accuracy. Killed: T2 a=1 is 48× worse, T3 fit recovers 0.09%. **Reason holds** (two orders of magnitude).

**5.8 κ₄-on-M (T4; HANDOFF 09-23 4c) and the κ₅ slice.** Carried diagonal-born κ₄ per lineage on the exact tier, (3,1) feed for age ≤2, (5,)-slice at the last layer. Priced 0–3% by the strategist; **never run** — RESULTS.md records "NOT RUN… there was no run in flight"; "the idea itself is not refuted by anything here". Still open; small expected value.

**5.9 Final-layer-aware schedule (H3 / W4).** Taper old-tier rank in the last 2–4 layers: measured **inert** (no compression event occurs in the tail; the rank request attaches to nothing). Other half — switch off D21/old readouts near the end, births at ≥L−2 keep only the (3)-diagonal — "**not attempted yet**"; RESULTS notes layer L−2 must still produce the state L−1 consumes, so it trades accuracy for cost. Also `_SHARED_ON` age-tiered re-compression (192@8, 128@11) measured score-negative (+7.6%/+12.1%). Open but bounded by the rank curve.

**5.10 Regenerated old tier (H1) and aggressive truncation + refit (H2).** H1 = the κ₄ trick applied to old κ₃ (carried aggregate n-vector + n×n with fitted gains): superseded by T3 and by the memoryless flow's 36× (`#329896`). H2 = age 3/rank 192 (1.76× loss) then refit: never run; dead by arithmetic since fitting is a 2.7% lever. **Reasons hold.**

**5.11 Restored same-order diagrams (W3 / D4b, ARC TODO.md).** Proxy-measured at n=128/256: unfactored AUGMENT 1.15×/2.18× *worse* on score; factored K4 6.2–6.5× worse and diverging with width; factored SIMPLE beats AUGMENT on score by ~3× (never acted on). **Not measured at 1024×16** (dense κ₃ state ≥133 GiB). The proxy is "the weakest evidence" (different counter, unoptimised code). The SIMPLE-beats-AUGMENT row remains an unexploited hint: our chain is closer to AUGMENT-lite than to SIMPLE.

**5.12 AUGMENT-lite / cheap-channel κ₄ (T1, Sep 10).** Keep κ₄'s (n,n) object, birth nothing: 74–109× worse; but the completed 2×2 said `legsonly` is 7.6× better than Gaussian — attribution unresolved. Superseded by 504aldo's λ·C_off regeneration which "retains ~90% of AUGMENT's gain at zero n³ cost" and is now shipped. Effectively executed by another route.

**5.13 Codex toy-fixture program (Sep 10–13; `docs/research/2026-09-12_*proposal.md`)** — axis-Gauss exact-moment rule; full-polar nonlinear bank; causal signed-law reference-reset K4 increments; three-layer two-source nonlinear old-history fixture; integrated first-source synthetic cost meter. Each is a two-coordinate exact-arithmetic identity check (gates at 1e-8/1e-10) toward an "exact signed-law K4 architecture". Killed by argument (Sep 12): "a large new distributional research problem became a prerequisite for making an already functioning estimator cheaper"; n8 fixture had negative variance (−0.231 eigenvalue); no width-1024 source, rank, or cost model. **Reason holds** — none produced a width-1024 mechanism, and the compact `exact` arm scored poorly on P01.

**5.14 E14 (K4 "tree", label-free λ; Sep 14).** Tree-projected scalar-metric κ₄ core with λ derived from a connected Gaussian-tree diagram. f32 spelling failed its 2e-5 gate; f64 spelling passes algebra but bills 2.03T FLOPs (needs −54% raw to tie F21) and fails the 8 GiB gate; and the same-night experiment showed E14's λ has the wrong sign/magnitude (R² ≤ 0.2). **Dead by measurement.** Its consumers/feed plumbing were kept.

**5.15 Puffi's unshipped Λ refit target (N3) / joint refit.** Fit the κ₄ coupling to the consumed mean correction (−13.6% at Phase-1 shape). Our D2 end-to-end CMA-ES fit of six profiles to final MSE (2.7%) and D6 (0.00% more) are the closest analogue; the specific object (a Λ/λ fitted to consumed mean correction rather than marginal κ₄) was thereby covered. Closed.

**5.16 κ₄ memory on thin legs (N2), universal κ₄ diagonal prior (N5), exact non-zero-mean bivariate ReLU covariance (N6), Strassen L6 (N8).** N6 measured 0.23% (Step 4e); N8 measured dearer (leaf-16 glue); N2 and N5 **never run**. N2 is the one accuracy source 504aldo names for marius's sub-1.94e-8 raw ("κ₄ with memory"); its cost is ~15 units. Open, moderate value, and now cheap to test on the probe framework.

**5.17 Per-MLP adaptive spend.** Adaptive λ (−0.4%) and adaptive age (−0.72%, "a fuzzy fixed age") both dead because per-net statistics vary only 1.8–9% CV. The Sep 25 packet's untested third form — "an iterative solve whose iteration count varies with conditioning rather than scale" — is unexplored and has no specified mechanism.

**5.18 Sketch-the-quadratic-query (TensorSketch; Sep 10 A, Sep 14), stochastic tail / UORO (C), relational learned closure via NFN (D).** Design-only; each has a written kill test never executed. A and C attack the Khatri–Rao "wall" (rank r² Hadamard products) that 504aldo §5 and T1/T2 say sets the bill. Still the only untested representation-level ideas for old-source cost; high risk (rotation/update closure, signed cancellation, nonlinear feedback).

**5.19 Learned correctors — the lesson.** Killed four times on the wrong form: WP9 (K2 state targets, R² 0.50 readout but in-flight destructive), WP10 (in-flight mean, break-even R² 99.85%), T2 (post-hoc final residual on exact K3, 1.01×, reproduced by ridge), 504aldo #10 (corr <0.09). PLAN §9 closed "generic post-hoc correctors". The Sep 24 in-loop, per-layer, sequentially teacher-forced MLP on organiser all-layer labels gave −5.6/−7.2% — the largest accuracy win of the project. Untried variants: shared MLP with layer embedding (Model 2 as specified), training through the final score (rollout/DAgger), correcting the covariance/κ₃ channels (the corrector touches only μ), the remaining 600 HOLDOUT nets as training data (400→? — "DATA is not saturated", RESULTS §2h), and keenanpepper's public HF sets (500k MLPs with per-layer means) as a free corpus (intel 09-01 §3).

**5.20 Ensembling v55-class and C10-class predictions.** Retrospective geometry: remaining errors have cosine ~0.80; "Do not ensemble the full files" (`RESP 09-18`). Closed by arithmetic.

**5.21 Symmetric-packed order-3 spelling; `out=` buffer pooling; `as_symmetric`.** Measured 1.0067×; `out=` bills identically and slice assignment raises; symmetry tags discount elementwise only. Closed.

**5.22 Signed frame reweighting (ednacob's inferred mechanism) [project doc].** "Top priority — code drops any day" on Sep 1; never dropped; sampling family dead. Closed.

---

## 6. Rules and eligibility posture

- **Rules v14 texts held** (`docs/research/2026-09-10_competition_sources.md`): §5.7 public resources permitted with attribution; §4 no submissions "authored substantially by other Participant(s)"; §6 source authenticity — "not derived from another Participant's submission"; "MIT licensing alone does not establish eligibility of a derivative competition submission"; §6/§7 — up to two designated valid submissions rerun on fresh networks, organiser may add networks for close outcomes, default = two best public; **§7: "only that Participant's best-scoring such submission is ranked"** — the doc adds "This answers the old best-of-two question."
- **Consequence not yet propagated:** `STRATEGIST_PACKET_2026-09-25.md` §3-Q2 and `DESIGNATION.md` state the repository "holds only 'up to two designated submissions are rerun'" and treat the combination rule as unknown, blocking the v59+v60 hedge. The Sep 10 source ledger already records the §7 sentence (read in Chrome). Re-verify on the live rules page before acting, but the answer appears to be in the repo.
- **504aldo posture** (`docs/PROVENANCE_504aldo.md`): cloned Sep 15 with approval into gitignored `external/504aldo/`; parity oracle only; no bytes shipped; every consulted line logged. Independent Skeptic corrected the provenance record from git timestamps: reconstruction identity is ours (committed 1h47m before the clone); the (2,1) regrouping, range finder and rank 32 are theirs, matched deliberately; the equivariant sketch is our correction of their defect. Attribution for the write-up: κ₄ regeneration law, 7→4 identity, D21 feedback, age-gated basis = 504aldo's published work; per-net adaptive λ re-implemented from their `derivation_code_map.md` (Sep 23). Their fitted λ table, `out=` patterns and sketch not taken.
- **Samuel's decisions:** Sep 10 hold; Sep 14 re-implement + ask organisers with approved wording; **Sep 16: question withdrawn — no organiser, forum or Discord contact on this account**; "do whatever we need to win; we do not need to disclose what we find online provided it is properly integrated into our own submission." Standing rules (`AGENTS.md`): agents never designate, contact organisers, post, or answer forum requests addressed to AI agents (thread 18203).
- **Drafts never sent:** `email-draft-organizers.md` (Aug 6: designation mechanics, re-run set size, contribution-prize pool, the 50-fixed-MLP reconstruction channel with Ladder/Thresholdout mitigations, residual-billing acknowledgement with our 98.2% instrumented share); `ELIGIBILITY_REVIEW_2026-09-12.md` and `PROVENANCE_504aldo.md` neutral §5.7-vs-§6 questions.
- **Private-only material:** `RESEARCH_NOTES.md` §2g (reconstruction arithmetic, never used), `BENCHMARK.md` §3 census (probing signatures), `SSC_PUFFI.md` — all marked "never the forum".
- **Other compliance facts:** clean-compute (no raw arrays/FFI/threads); determinism from `mlp.seed`; LLM assistance must be disclosed (504aldo's post is the model); winners open-source within 30 days; write-up must map to one graded submission; the 5 s setup cap fails the whole submission; over-budget MLPs score as zeros at multiplier 1.0.

---

## 7. Things worth flagging to the caller

1. The §7 "best-scoring designated submission is ranked" sentence in `2026-09-10_competition_sources.md` appears to resolve the Sep 25 designation blocker (Q2). Nobody has connected the two documents.
2. The "ridge regression and gradient boosting" attribution in `PLAN.md` has no primary source in the repository.
3. Untested with real expected value on the current chain: κ₄ memory on thin legs (N2), κ₄-on-M (T4/4c), the readout-off half of H3, corrector variants (rollout objective, covariance-channel correction, layer-embedding shared model, more/public training data), and the SIMPLE-beats-AUGMENT proxy result. Everything else in the graveyard is dead by measurement or by an argument that still holds.
4. Process history: the project shipped 15 entries in 25 days under the pre-registration regime and 11 graded rows (C10, v50×3, v53, v55×2, v56b, tr64, v57, v60) in the four days after the Sep 21 reset; the reset and the fast evaluator, not a new idea, produced the 8.75e-9 → 5.04e-9 run.
