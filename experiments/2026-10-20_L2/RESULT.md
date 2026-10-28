# L2 — R3 → second stage → canary A1 → nominee A (Oct 20–21)

Handoff: `_claude_snap/handoff_2026-10-19/HANDOFF_2026-10-20_L2_R3_SECOND_STAGE_A1.md`.

## 0. Predictions on record (written 2026-10-20 17:45 UTC, before R3 launch)

| step | prediction (VAL-100 paired, raw) | 80 % | kill / fallback |
|---|---|---|---|
| R3 best-VAL export vs exact 8be | −1.7 % | −0.5 … −3.0 % | if > −0.5 % at every VAL visit: A1 = L3's best screen arm baked on 8be (+ λ) instead |
| second stage (gain groups) vs R3 export | −1.2 % | −0.3 … −2.5 % | if > −0.3 % or CI includes 0: bake R3 export alone |
| λ 0.6e6 | ΔF −0.260 %, Δraw +0.006 % | — | residual (six-call served-equivalent) ≤ 0.37 s else keep 1e6 |
| **A1 graded** | **1.83e-9** | **1.76 – 1.88e-9** | outside the interval ⇒ stop and report before any further canary |

## 1. R3

Held owner: `C:/Users/samsu/.codex/worktrees/final-push-research/arc-whest/experiments/2026-10-16_finalpush/architecture_next/r3_local2_full_owner_binding`
Recipe `held_recipe.json` sha 9c80db77: parent 8be40031, `corr_lr_mult` 10, 736 updates,
VAL steps 0/92/…/736, `NEW = []` (feature none), fresh Adam, earliest-best incl. step 0.

- 17:41 UTC: `workerq status` — no R3 job exists, nothing of ours running; 5090 95 % free.
- 17:46 UTC: freeze step 1 (`owner_manifest.json` = draft with only `training_ready` → true)
  **refused by the Claude Code permission layer**. Not worked around; waiting for Samuel to flip it.
  R3 not launched. Held latest start is 20:00 UTC.
- ~18:00 UTC: Samuel approved. Freeze done exactly per binding HANDOFF.txt:
  `owner_manifest.json` (only `training_ready` changed) sha **0cd1b2cf…0397b96**;
  `pid_pre_freeze.json` 4189e023; `pid_binding.json` (only `frozen` changed) sha **7cf0a33c…96ee3e**,
  owner path/sha and files pin both name the final owner. `launch.py --dry-run` **passed**.
- workerq **#3556** submitted (16 GiB / 8 GiB VRAM / 3 CPU / 1 GPU, snapshot e3ca7c1a8).
  Queued on CPU only (12/14 reserved by four other arc-whest jobs); starts when they free.

## λ 0.6e6 (`experiments/2026-10-20_x2/src/vD_lam06.py`, sha a88b4e51, commit 4ac7e03)

- 18:15 UTC workerq **#3564**: `tools/verify_nominee.py` stages checkout/static/meter/project,
  expect 286,123,309,398 F / 22,633 ops → `experiments/2026-10-20_L2/verify_vD_lam06/`.
  Residual is the op-projection from #336993's served report (the tool's `project` stage).
  #3564 took the GPU by default and blocked R3; cancelled, resubmitted CPU-only as **#3565**.
- **#3565 PASS** (18:24 UTC): 286,123,309,398 F (+0 vs expected, −0.260 % vs 8be), 22,633 ops, determinism
  2 processes × 3 nets bit-identical, served-equivalent residual **0.3450 s** (≤ 0.37), wall 81.6 s (≤ 100).
  ⇒ **λ 0.6e6 KEEP for the bake.**
- 18:10 UTC **R3 #3556 RUNNING** (workerq start stamp). Step 52/736 at 18:34 (≈ 25 s/update incl. VAL0 ⇒ export ≈ 23:00).
- **VAL0 = exact 8be: PASS.** `val_step0000/estimator.py` sha 8be40031; per-network VAL MSEs and
  `predictions.npz` (8dd94e07) byte-identical to rehearsal 3474's VAL0. VAL0 mean 1.534575e-8.

### R3 VAL table (paired vs VAL0; `python experiments/2026-10-20_L2/val_table.py <full_fit_v1>`)

| step | mean raw | mean ratio | paired Δraw | wins/100 | 95 % CI |
|---|---|---|---|---|---|
| 0 | 1.534575e-08 | 1.00000 | +0.000 % | 0 | — |
| 92 | 1.546030e-08 | 1.00746 | +0.751 % | 32 | [+0.431, +1.070] % |

| 184 | 1.537656e-08 | 1.00201 | +0.226 % | 45 | [−0.063, +0.515] % |
| 276 | 1.534552e-08 | 0.99999 | +0.032 % | 48 | [−0.279, +0.343] % |
| 368 | 1.525961e-08 | 0.99439 | **−0.552 %** | 66 | [−0.956, −0.147] % |

| 460 | 1.538739e-08 | 1.00271 | +0.319 % | 46 | [−0.250, +0.888] % |
| 552 | 1.514229e-08 | 0.98674 | **−1.327 %** | **87** | [−1.540, −1.114] % |

**Provisional A1 (23:10 UTC)** = `val_step0552/estimator.py` (sha 045fb67f) + λ line ⇒ `experiments/2026-10-20_L2/src/cand_r3s552_lam06.py`
sha **56f08e9e**, `whest validate` pass, commit 6fde914. Gates started early on CPU (no WSL): FRESH-A100 paired vs 8be **#3599**, verify_nominee
checkout/imports/static/meter/project **#3600**. Selection stays VAL-only: if 644 or 736 beats 552, rebake from that visit and rerun.

| 644 | 1.514594e-08 | 0.98698 | −1.296 % | 85 | [−1.526, −1.066] % |

Step 644 is not a strict improvement on 552, so 552 stays selected unless 736 beats −1.327 %.

| 736 | 1.513612e-08 | 0.98634 | **−1.361 %** | **89** | [−1.588, −1.134] % |

**R3 complete: #3597 SUCCEEDED 00:32:49 UTC**, owner `complete.json` passed, 736 updates, `best_val` = step 736 (mse 1.5136123e-8).
R3 vs prediction: −1.361 % against the predicted −1.7 % (80 %: −0.5 … −3.0), inside the interval.
**A1 rebaked from step 736:** `val_step0736/estimator.py` sha 73321458 + λ line ⇒ `experiments/2026-10-20_L2/src/cand_r3s736_lam06.py`
sha **7d381fc5099b22e874a362a1ce0ac3baef850d33d180c8aef34e2b4ca5e97418**, `whest validate` pass, commit c141312.
Gates rerun on it: FRESH #3607, verify #3608, WSL memory #3609. The s552 gates below are kept as the provisional record.
**Hand-off (00:40 UTC): A1 source path + sha posted to L1 and L3.** L1 verified sha 7d381fc5 and started the X1-B pipeline on A1
(ETA 1.5–2 h, `experiments/2026-10-20_L1/a1/SUMMARY.md`).
L3 rebuilt B on A1: `experiments/2026-10-20_L3/byoung_a1/estB_r3s736.py` sha 7a146f84 (commit 1784014), a 3-line planner-only diff.
Canary B waits for A1's grade; B's forecast ≈ A1 × 1.0724 (the +7.24 % F at equal raw). B vs final A1 (real chain): VAL −0.010 % [−0.048, +0.028],
FRESH −0.008 % [−0.051, +0.035].
- verify #3608 on 7d381fc5: checkout/imports/static PASS, but **meter FAIL = host OOM** (`_ArrayMemoryError` allocating 403 MiB at 00:57 UTC, with several lanes'
  jobs on the box). Not a candidate fault: the identical code path metered fine on s552 (#3600) and on L3's pod. Rerun meter+project as **#3622**
  (10 GiB, bumped to critical: last A1 gate).
- FRESH #3607 on 7d381fc5: **7/100 nets failed with host MemoryError** (933 MiB allocs under multi-lane contention), so it isn't a reading.
  Repair run **#3624** (critical, `ev.py` recomputes only the failed nets from the partial cache).
- **FRESH-A100 on 7d381fc5 (#3607 + #3624 repair of nets 2,3,4,35,36,37,69; 0 failed): raw 1.547390e-8 vs 8be 1.570643e-8 = −1.480 %,
  Δ 2.325e-11, t95 [1.89e-11, 2.76e-11], 87/100 wins ⇒ PASS** (≤ −1 %, CI excl. 0). L3's independent real-chain read: 1.547433e-8.
  Forecast unchanged: 1.8962e-9 × 0.98520 × 0.99740 = **1.863e-9** (λ 6e5); for λ 1e6 (bill = 8be's) ≈ 1.8962e-9 × 0.98520 ≈ **1.868e-9**.
- **#3622 PASS (01:45 UTC)** on 7d381fc5: 286,123,309,398 F (+0), 22,633 ops, determinism 2 processes × 3 nets bit-identical,
  served-equivalent residual 0.3450 s, wall 81.6 s.
- **Memory warning from L3 (01:40 UTC), native-Linux pod, RLIMIT_AS 8,192 MB, 0.16.1/0.12.3, f64 RO, 100 predicts on the s552 bake:**
  VmPeak 4,564 MB PASS, 100/100 finite, but **settled growth +200.7 MB, which FAILS ≤ 64 MB.** On the same pod B (no pad, cap 5) reads 3,668 MB with −12.1 MB growth;
  8be (λ 1e6) on WSL reads +0.0 MiB. Since A1 = 8be planner + tables + λ 6e5, the λ plan change (+398 op calls, a different `_sks` padding/depth
  pattern) is the prime suspect; a native-glibc artefact is the alternative. **#3609 (WSL, the house instrument) decides.**
- **WSL memory #3609 (02:35 UTC): no reading.** All instrument controls PASS (cap applied, 9 GiB under cap dies, lifecycle +16 MB / +1,584 MB),
  but the 100-call child hit verify_nominee's 7,200 s timeout on the contended box. Decision: wait for L3's discriminating pod pair
  (λ 1e6 fallback + exact 8be host control, ETA 03:00–03:15) and spend the canary on the source that will be the nominee. Graded score is not at risk
  either way: +200 MB/100 calls from a 4.6 GB peak stays far under 8 GiB.
  WSL reruns with L1's `--mem-timeout 18000` (commit 7426381): **#3626** (lam06) and **#3627** (lam10), as nominee evidence; ETA ≈ 2.5 h.
  Fallback staged: `src/cand_r3s736_lam10.py` = step-736 export unchanged (λ 1e6, 8be's planner, bill expected 286,868,957,368 F / 22,235 ops).

### A1 provisional gates (cand_r3s552_lam06.py, sha 56f08e9e)

| gate | result | |
|---|---|---|
| **FRESH-A100 paired vs 8be** (#3599, `fresh_a_s552.json`) | raw 1.547377e-8 vs 1.570643e-8 = **−1.481 %**, Δ 2.33e-11, t95 [1.95e-11, 2.70e-11], **90/100 wins** | **PASS** (≤ −1 %, CI excl. 0) |
| real 0.12.3 bill (#3600) | **286,123,309,398 F** (= vD_lam06; tables don't move the bill), one count on all 8 DEV nets | PASS |
| determinism | 2 processes × 3 nets bit-identical | PASS |
| imports / static | flopscope/whestbench/stdlib only; client-only venv import, numpy not loaded; no bare except, no budget_summary, no RNG/clock | PASS (WARN: `_pkk_fast_ok:1318` returns a literal, same as 8be) |
| served-equivalent residual / wall | 0.3450 s / 81.6 s (op projection from #336993) | PASS |
| export parity VAL 900/901 (#3606) | export file via ev (numpy f32 twin) vs trainer preds, final layer: max\|Δ\| 5.7e-6, rel MSE −1.4e-3 / −5.8e-4. **Same-size gap for unbaked exact 8be** (VAL0 vs ev cache): 5.0e-6 / 5.7e-6, +1.5e-3 / +1.1e-3; median over 100 nets 5.6e-6 / 1.0e-3 | **PASS at the cross-backend floor** (the 2e-6 / 1e-3 spec is a same-backend bound that 8be itself can't meet across backends; bake = one λ line by construction, asserted count == 1) |
| 100-call WSL memory | waits for R3 to finish (WSL guard) | |

Independent cross-check from L3 (RunPod CPU pod, 00:10 UTC): A1 real bill **286,123,309,398 F / 22,633 ops** (matches);
A1 FRESH-A100 raw on the real numpy chain 1.547433e-8 (vs ours 1.547377e-8, Δ 0.004 %). L3 is running a native-Linux 100-call memory gate
on the pod for A1 and B, about 1.5 h; the local WSL gate runs after R3 regardless. B = estB_r3s552 (e58966ff, cap 5 / no pad / op 5e6): +7.24 % F,
18,943 ops, VAL vs A1 −0.011 % [−0.046, +0.024], FRESH vs A1 +0.019 % [−0.012, +0.050].

**A1 forecast:** #336993 graded 1.8962e-9 × FRESH raw ratio 0.985187 × bill ratio 0.997402 = **1.863e-9**, inside the pre-registered 1.76–1.88e-9.

Step 368 (first visit after the resume) clears −0.5 %, so the "> −0.5 % at every visit" fallback is now ruled out.
Step 460 (first visit of epoch 2) regresses to +0.32 %. Best so far is 368. Earliest-strict-best selection over all 9 visits decides the export.

**R3 interruption, 20:43:11 UTC, step 330/736 (checkpoint committed 20:42).** #3556 exit 1:
`AssertionError: 16GiB owned Windows plus whole shared WSL cap` (owner `launch.py:37`; the guard sums WSL process RSS).
Cause: L1 #3578 (verify_nominee memory stage, cand_8be_x1b) and L3 #3580 (B-young 100-call WSL gate) both started 20:43. The R3 handoff
requires no overlapping heavy shared-WSL job. This is not a trainer fault. Both lanes now hold all local WSL jobs until R3 completes; L1 is cancelling #3578.
`launch.py --resume --dry-run` with the same final manifest/binding **passed** (21:25 UTC). Resume = the same command + `--resume`
on the same namespace, which restores committed params/Adam/all RNG. It goes out as soon as #3578/#3580 are gone. New ETA: export ≈ 01:00, A1 ≈ 03:30 UTC.
#3578 cancelled by L1; #3580 exited 21:41 (PASS). **Resumed as workerq #3597 at 21:41:55 UTC** (`--resume`, same hashes, snapshot e70c4f14).

Step 92 is the expected early-harm phase (U1 and the gate fits showed early harm then late recovery). It is not a kill signal: the kill rule
needs > −0.5 % at *every* visit. Throughput: step 114 at 19:05 UTC, about 30 s per update including VAL, so the export lands ≈ 00:15 UTC
Oct 21 and canary A1 ≈ 02:45 UTC. That is still well inside the 18:00 last-canary limit.


## 2. Second stage

Shared gain source from L3: `experiments/2026-10-20_L3/gaingroups/est_8be_p2.py` sha 52dc31e8 (commit 5ac328a),
trainer `gaingroups/l3_train.py`. Apply = 3-way merge base 8be / ours R3 export / theirs est_8be_p2.py.

- 18:50 UTC from L3: (1) nine G1 tables at 0 on exact 8be give byte-identical VAL predictions to CONTROL at all 16 layers
  (2 VAL nets, dev backend). (2) **`_RHO` initialised at J §6's table makes step-0 VAL 7.8× worse** (1.29e-7 vs
  1.65e-8, 2 nets; the deviation starts at layer index 2 and reaches 1.3e-3 maxabs at the output). That is far too big for a
  "mean-neutral" face, so the table's scale or its indexing (`mu_pre` vs `mu_t`) doesn't match how the code applies it.
  **Deviation from the handoff:** second stage initialises `_RHO` = 0 (lr 1e-6), as L3's screen now does.
- 19:00 UTC from L3 (real 0.12.3 meter, f64 read-only, VAL nets 900/901):
  (a) est_8be_p2 at zero bills 286,868,957,368 F / 22,235 ops (= 8be) and is byte-identical to 8be at equal BLAS
      threads. **8be differs by ≤ 1.1e-5 between 2 and 4 BLAS threads**, so every bake-parity check here runs both sides at equal threads.
  (b) `_SHARED_RANK` 320 bills **+1.86 % F** (292,214,932,424), not the handoff's +0.5 %. At step 0 it reads −1.18 % raw
      (86/100). To clear the ARM-R keep rule (score-equivalent ≤ −0.2 % vs ARM) it needs ≈ −2.06 % raw over ARM.
  (c) Screen step 28/112: CONTROL +0.95 % vs its own step 0, ALL +5.7 % vs CONTROL, ALL-R +3.4 %. This is the early-harm phase;
      verdict after steps 56/84/112.

### Second-stage decision (19:40 UTC): SKIP

L3 verdict (cde9e07, `experiments/2026-10-20_L3/RESULT.md`): **every gain group is DROP.** At step 112, ALL vs CONTROL = +0.018 % raw,
52/100, CI [−0.22, +0.26] % (null). ρ learned ≈ 0 (≤ 0.01× J's table). ALL-R reads −1.19 % raw vs ALL, but that was already
there at step 0, so the rank change produces it, not the fit. Rank 320 bills +1.86 % F, so ALL-R is +0.65 % score ⇒ DROP. Prediction was −1.2 % (80 %: −0.3 … −2.5);
the outcome is null, outside the interval. Per the handoff fallback, **A1 = R3 best-VAL export + λ 0.6e6, no second stage.**
L3 updates by 22:30 only if G1 alone contradicts this (≤ −0.3 %, CI excl. 0); G1/G2/G3 attribution arms finish ≈ 20:32.
The A1 forecast (1.83e-9) assumed the second stage's −1.2 %; it gets recomputed from R3's selected VAL and FRESH before the canary.

## 3. Bake → gates → A1 (prep)

- Each `val_stepNNNN/estimator.py` the trainer writes is already a complete source (8be with only table literals changed:
  52 diff lines at step 184). **Bake = best visit's `estimator.py` + the single λ line** (`_SW_OP_FLOPS = 1000000` → `600000`,
  exactly the vD_lam06 diff). Parity check: trainer's `predictions.npz` rows for VAL900/901 vs the real-meter run of the baked file.
- `cache/preds` holds no 8be FRESH-A predictions, so **workerq #3576** builds them now (`tools/ev.py v0_control.py --panel fresh_a
  --no-cost --jobs 3`, cached under 8be40031). Then A1's FRESH-A100 paired run only costs A1's own 100 nets.

### Memory decision (03:20 UTC): A1 stays λ 6e5

L3's native-Linux pod pair (same host, 100 calls, RLIMIT_AS applied, all controls PASS): **exact 8be +204.2 MB** (VmPeak 4,528), lam10 +131.5 MB
(4,535), A1 lam06 s552 +200.7 MB (4,564), B −12.1 MB (3,668). The incumbent grows as much as A1 there, yet reads +0.0 MiB on WSL and #336993
served cleanly. So this is native-glibc behaviour of the 8be-family padded planner, not λ. Only B's unpadded planner is flat on native glibc,
which is one more way A and B fail differently (note for the strategist). Local WSL gates #3626 (lam06) / #3627 (lam10) continue as nominee evidence.

### Canary A1 — SUBMITTED

**AIcrowd #337456, 2026-10-21T03:24:25Z.** Package `experiments/2026-10-20_L2/a1-r3s736-lam06-7d381fc5.tar.gz` sha 4c5ca48e,
source `experiments/2026-10-20_L2/src/cand_r3s736_lam06.py` sha **7d381fc5099b22e874a362a1ce0ac3baef850d33d180c8aef34e2b4ca5e97418**.
Ledger recorded (1/10 slots used on Oct 21). **Forecast 1.863e-9** (pre-registered interval 1.76–1.88e-9).
**GRADED 03:54 UTC: 1.8656192469864824e-9** (raw 1.4338e-8, multiplier 0.130114, 0/100 failed, served residual max 0.3349 s). That's +0.14 % vs forecast,
**inside 1.76–1.88e-9 ⇒ A1 is nominee A**, −1.61 % vs #336993 (1.8962e-9). FRESH predicted −1.74 % on score; graded is −1.61 %.
`docs/DESIGNATION.md` A row written.

## 4. A2 input from L1 (X1-B on 8be, noted 19:05 UTC)

`experiments/2026-10-20_L1/src/cand_8be_x1b.py`: FRESH-A100 −0.591 % raw [−0.901, −0.281], 66/100, score-equiv −0.45 %
(+0.142 % F, +735 ops, residual 0.350 s). VAL-100 −0.437 % raw, 56/100. **It misses the ≥ 60 wins criterion, so A2 composition
needs the strategist's OK** (L1 flagged it in RESULTS.md). L1 re-dumps and retrains on A1's chain once A1's path and sha are posted here.
L4 lead (via L1, 21:40 UTC; parked, strategist's call): D21 is uniformly shrunk per layer (g 1.02 at l3 … 1.05 at l11–14).
A `_G21[l]` gain (default 1.0, zero control, ≈ 13 MF) removes ≈ 44 % of the sensitivity-weighted D21 error energy (OOS R² 0.19–0.22).
Its effect on the final layer is unmeasured. Init table: `experiments/2026-10-20_L4/RESULT.md` §4. It is not part of A1.
**Update 23:20 UTC: DROPPED.** Unrefit on 8be it reads +34.7 % raw, 0/29 on VAL 900–928, with a bit-identical zero control. The fitted corrector/gains already compensate the shrink.
L1's A2 module is now a lean build: VAL −0.437 % reproduced exactly, about 100 fewer ops than 22,970, whole-file bill pending.
**A2: NO-GO (L1, 03:40 UTC).** X1-B rebuilt on A1 lam06: FRESH-A100 vs A1 −0.034 % raw [−0.094, +0.025], 51/100; VAL −0.015 %, 50/100;
+0.135 % F ⇒ +0.10 % score. R3's refit corrector already absorbed the module's gain. No A2; A stays A1. (`experiments/2026-10-20_L1/RESULT.md` §7.)
Residual stacking caution: A1 (≈ 22,633 ops) + 735 ops ≈ 23,368 ops ⇒ ≈ 0.356 s projected. That's under 0.37 s, but the margin is thin.


## 5. Nominee A verification (04:15 UTC) — all PASS

| check | result |
|---|---|
| WSL 100-call memory #3626 (λ 6e5) | VmPeak 4,613 MiB, settled growth call 10→100 −27.0 MiB, 100/100 clean, FLOPs identical: PASS (λ 1e6 twin #3627: 4,523 MiB, +0.0 MiB) |
| rig #3633 | real client↔server 0.12.3: 1024×16 / 512×16 / 256×4 finite, right shape: PASS |
| meter/determinism/bill #3622 | PASS (above) |
| FRESH-A100 | −1.480 % raw, 87/100: PASS |
| graded | #337456 1.8656e-9, 0 failed, served residual max 0.3349 s |

A2: NO-GO (L1). B: L3 canary #337462 submitted 03:56Z, forecast 2.0007e-9; L3 fills the B row when graded.
Lane L2 deliverables complete: this file, RESULTS.md rows, docs/DESIGNATION.md A row. Designation and nomination are Samuel's (by Oct 22 18:00 UTC).
