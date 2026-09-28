# RESULTS — ARC White-Box Estimation Challenge, Phase 2

One row per candidate evaluation or graded run. This file replaces
`experiments/LOG.md` (archived 2026-09-21). Append; never rewrite history.
The plan is `docs/PLAN.md`; the submission gates are its §5.

## Column meaning

| column | meaning |
|---|---|
| **date** | UTC date of the evaluation or of the submission |
| **file** | candidate source in the repo root |
| **sha8** | first 8 hex of sha256 of the *estimator source bytes* (not the package) |
| **panel** | `DEV` = the 100 organizer mini nets · `FIT` = 100 full-split nets disjoint from mini · `HOLDOUT` = the remaining **900** (≤2 reads/week, never fitted) · `GRADED` = AIcrowd server, 100 nets (50 public + 50 sealed) |
| **raw MSE** | final-layer MSE, unweighted |
| **F/B** | FLOPs per MLP ÷ 2^41, i.e. the score multiplier before the 0.1 floor |
| **ops** | flopscope operation calls per MLP (drives residual wall time) |
| **wall** | worst per-MLP wall seconds. Local rows are marked `(local)` and come from the one-net grader twin; the server figure comes from the wall law (median = 0.36 ms × ops + FLOPs × kernel rate; worst = median × 1.055 quiet / 1.24 noisy; § Calibration) |
| **RSS** | VmPeak under the WSL `RLIMIT_AS` 8,192 MB gate |
| **projected** | predicted public score = DEV score ÷ 1.038 (§ Calibration) |
| **graded** | the AIcrowd adjusted score actually returned |
| **paired gain** | paired arithmetic gain vs the stated baseline, common random numbers |
| **wins/N** | nets on which the candidate beat that baseline |

## Calibration constants (recheck after every graded row)

All four are implemented in `tools/ev.py`; change them there and here together.

- **DEV → graded**: divide the 100-net DEV adjusted score by **1.038** — the
  current constant, in `tools/ev.py` `PUBLIC_FACTOR`. **Eight** observations:
  1.0386 / 1.0385 / 1.0339 / 1.0426 / 1.0471 / 1.0458 / 1.0397 / 1.0361, mean
  1.0403 (the last two are v57 #332129 and v60_ws400 #332223).
  History: 1.037 until 2026-09-22 (three rows in a row graded 0.53–0.97% *better*
  than projected) → 1.042 → **1.038 on 2026-09-25**, because the next two rows
  graded *worse* than projected (+0.22%, +0.54%), i.e. 1.042 was ~0.4% optimistic.
  1.038 sits just under the mean, so projections err slightly pessimistic — the
  side this constant exists to err on. If a graded row ever deviates by >2%,
  stop submitting and recalibrate before the next slot.

  The factor is entirely a raw-MSE effect: C10's graded multiplier matched its
  DEV multiplier to 2e-10, while raw MSE moved 3.094966e-08 → 2.968645e-08. The
  graded 100 nets are simply a little easier than the mini 100; nothing about
  cost changes between the two.

- **FLOPs: there is no offset to add.** Take the number from the **f64 read-only
  grader twin** (`tools/_cost_worker.py`, one net) and use it as-is. Verified
  2026-09-21 against both graded rows we hold:

  | candidate | twin FLOPs/MLP | twin multiplier | graded multiplier | delta |
  |---|---:|---:|---:|---|
  | F21 #330950 | 928,457,637,478 | 0.42221365105 | 0.4222136511 | **0** |
  | v31 #331317 | 661,961,618,994 | 0.30102529263 | 0.3010252926 | **0** |
  | C10 #331816 | 480,850,933,121 | 0.21866568801 | 0.21866568778 | **491 FLOPs of 4.8e11**, i.e. 1e-9 relative |

  (A third confirmation is on record: the twin predicted #330212's `flops_used`
  to delta 0 out of sample, `docs/GOTCHAS.md` 2026-09-07.) What *does* read low
  is `whest run --split mini`, by exactly **2^25 = 33,554,432** — its local
  dataset hands out float32 weights so the candidate's opening cast never runs.
  That is the "+2^25" PLAN §4 mentions; taking the number from the twin makes
  the correction unnecessary rather than approximate. C10 measured: twin
  480,850,933,121 vs `whest run` 480,817,378,689, difference 2^25 exactly.

- **server wall — REFIT 2026-09-23 (supersedes the two entries below).** Nine
  graded runs, six candidates, using the per-net telemetry split the server
  reports (`flopscope_server_overhead_wall_s` vs `..._kernel_wall_s`):

      median wall = 0.371 ms × ops  +  FLOPs × kernel rate
      worst-of-100 = median × 1.055 (quiet day)   or   × 1.24 (noisy day)

  | run | FLOPs | ops | median | worst | kernel ps/FLOP | overhead µs/op |
  |---|---:|---:|---:|---:|---:|---:|
  | F21 #330950 (depth 4, leaf 64) | 928.5 G | 14,091 | 89.5 | 93.2 | **90.0** | 408 |
  | v31 #331317 (depth 4, leaf 64) | 662.0 G | 13,053 | 66.0 | 68.1 | **91.8** | 391 |
  | C10 #331816 | 480.9 G | 23,421 | 83.1 | 87.6 | 154.8 | 361 |
  | v50 #331863 / #331939 | 473.4 G | 25,728 | 84.4 / 85.0 | 88.7 / 87.7 | 158.6 / 160.1 | 351 / 350 |
  | v53 #331880 (age 5) | 502.4 G | 25,140 | 92.8 | 96.5 | 166.2 | 360 |
  | v55 #331967 / #331980 (age 5), noisy | 475.4 G | 23,645 | 92.8 / 92.5 | 114.5 / 113.8 | 176.3 / 175.8 | 371 / 366 |
  | v50 #331989, noisy | 473.4 G | 25,728 | 87.0 | 104.1 | 163.6 | 360 |

  Three findings. (1) **Overhead is flat** at ~0.36 ms per op call. (2) **Kernel
  time per FLOP is not a constant — it depends on Strassen leaf size**: ~91 ps
  with 64-wide leaves (depth 4) vs 155–177 ps with 32-wide leaves (every depth-5
  chain). The server's BLAS runs 32×32 leaves at ~55% of the efficiency of 64×64
  ones, so depth 5's 12% FLOP saving *costs* wall. Age-5 chains are a further
  ~10% slower per FLOP. The old FLOPs+ops fit could see neither effect. (3) **The
  tail is the day, not the candidate**: worst/median is 1.03–1.055 on a quiet
  server and 1.20–1.235 on a noisy one; medians barely move (v50 85.0 → 87.0).

  Rates used (`tools/ev.py` `KERNEL_PS_PER_FLOP`, worst observed per class):
  leaf 64 / age ≤4 **92**, leaf 64 / age ≥5 **101** (inferred, +10%), leaf 32 /
  age ≤4 **164**, leaf 32 / age ≥5 **177** ps/FLOP. Every anchor projects at or
  above its graded worst:

  | candidate | projected quiet | graded quiet | projected noisy | graded noisy |
  |---|---:|---:|---:|---:|
  | F21 | 95.6 | 93.2 | 112.4 | — |
  | v31 | 69.4 | 68.1 | 81.5 | — |
  | C10 | 92.4 | 87.6 | 108.6 | — |
  | v50 | 92.0 | 88.7 / 87.7 | 108.1 | 104.1 |
  | v53 | 103.7 | 96.5 | 121.8 | — |
  | **v55** | **98.0** | — | **115.2** | **114.5 / 113.8** |

  Against the Sep 23 designation policy (A ≤ 105 s noisy, B ≤ 95 s noisy):
  **v55 is neither** (noisy 114.5 s measured); **v50 is marginal A** (104.1 s
  measured, 108.1 s projected); **nothing we hold is B**. B needs a projected
  noisy median ≤ 76.6 s, which on this model means 64-wide leaves.
  §5's "projected server wall < 100 s" gate is read as the **quiet-day**
  projection (a canary's job is to measure wall); the noisy projection is the
  designation number, reported alongside.

- **server worst-of-100 wall** *(superseded above, kept as history)*: fitted on **FLOPs and op calls**, plus the
  model's own error band:

      wall = 60.251 s per 1e12 FLOPs + 2.478 s per 1000 op calls + 4.1 s

  Refit on five graded rows. The 4.1 s is the fit's own worst residual and it is
  **added** rather than ignored, because the wall cap is a cliff: a net over
  120 s is TIME_EXHAUSTED and scores as zeros.

  | candidate | FLOPs | ops | graded WORST | projected |
  |---|---:|---:|---:|---:|
  | F21 #330950 | 928.5 G | 14,091 | 93.160 s | 95.0 s |
  | v31 #331317 | 662.0 G | 13,053 | 68.132 s | 76.3 s |
  | C10 #331816 | 480.9 G | 23,421 | 87.556 s | 91.1 s |
  | v50 #331863 | 473.4 G | 25,728 | 88.747 s | 96.4 s |
  | v53 #331880 | 502.4 G | 25,140 | **96.488 s** | 96.7 s |

  Conservative on every anchor now. **Two successive corrections were needed
  here.** The original model (`twin wall × 1.5981`) was 23% low on v50. Its
  replacement, fitted on four rows without an error band, said 90.6 s for v53
  which graded 96.488 s — inside the residual band, but on the wrong side of
  PLAN D7's 95 s finals bound, so a candidate could look final-ready and not be.
  Refit again when a sixth row grades.

- **WALL MODEL WARNING (2026-09-23, five v50/v55 graded runs).** Two effects
  the FLOPs+ops model does not capture, both measured today:

  | run | median | p90 | worst | nets >100 s |
  |---|---:|---:|---:|---:|
  | v50 #331863 (Sep 22) | 84.4 | 87.0 | 88.7 | 0 |
  | v50 #331939 (Sep 23 03:00 UTC) | 85.0 | 86.7 | 87.7 | 0 |
  | v50 #331989 (Sep 23 ~08:00 UTC) | 87.0 | 100.8 | **104.1** | 15 |
  | v55 #331967 | 92.8 | 104.4 | **114.5** | 27 |
  | v55 #331980 | 92.5 | 105.6 | **113.8** | 15 |

  1. **Server state drifts.** Identical v50 bytes grew a ~16 s tail within one
     day. A "worst of 100" from one run is a sample, not a property.
  2. **Age-5 chains are ~5–8 s slower at the median than the model says.** v55
     does about the same work as v50 (FLOPs +0.4%, ops −2,083) yet its median is
     ~5.5 s above v50's under the same server state, and equals v53's (92.7 s).
     The age-5 old tier costs wall that neither FLOPs nor op calls meter —
     probably memory traffic (VmPeak 6.5 vs 5.8 GB). The model projected v55 at
     91.3 s worst; its median alone is 92.5 s.

  Until refit with a memory/age term: **treat an age-5 chain's projected wall as
  ≥ v53's graded worst (96.5 s) on a quiet server, and ≥ 114 s on a noisy one.**
  For the private re-run, v50 (worst 104 s even on today's noisy server) keeps
  more margin to the 120 s cliff than v55 (114 s).

- **ops → server residual**: use **11.09 µs/op**, the worst of three graded
  anchors, because the cliff FAILS the MLP outright and over-projecting is free:

  | candidate | twin ops | graded residual | µs/op | headroom vs 0.4 s |
  |---|---:|---:|---:|---:|
  | F21 #330950 | 14,091 | 0.156232 s | 11.087 | 2.56× |
  | v31 #331317 | 13,053 | 0.140244 s | 10.744 | 2.85× |
  | **C10 #331816** | **23,421** | **0.236255 s** | **10.087** | **1.69×** |

  The rate is **not** constant — it falls as ops rise (11.09 → 10.09), so a flat
  µs/op is a crude model of a mix-dependent cost. It is not even monotone in
  ops between F21 and v31, so do not fit a line to three points; take the worst
  rate and keep margin. Implied ceilings at 11.09 µs/op: the 0.4 s cliff is
  36,069 ops, **1.5× headroom is 24,046 ops**, and PLAN D7's 0.33 s finals bound
  is 29,757 ops.

  The in-process local `residual_wall_time_s` is an instrument artefact (C10
  reads 5.9–9.1 s locally against 0.236 s on the server) and must never be
  quoted — project from ops.

## Rows

| date | file | sha8 | panel | raw MSE | F/B | ops | wall | RSS | projected | graded | paired gain | wins/N | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-14 | `estimator_k3_strassen_L4_f21.py` | `d77dab0e` | GRADED | 3.779e-08 | 0.42221 | 13,936 (local meter) | 93.160 s | — | — | **1.5957e-08** | — | — | sub #330950, 100/100, 0 failed. Worst residual 0.156232 s (2.56× headroom). The designated fallback and the anchor for `predict_public.py`. |
| 2026-09-16 | `estimator_k3_v31.py` | `6dde5fd4` | GRADED | 2.9074e-08 | 0.30103 | 13,037 | 68.132 s | 5,434.6 MB | 8.713e-09 | **8.7520e-09** | 1.8233× vs F21 | 50/50 | sub #331317, 100/100, 0 failed, smoke 8/8. Worst residual 0.140244 s (2.85× headroom, 10.76 µs/op). Projection landed +0.45% — the best graded score to date and the §5 baseline. |
| 2026-09-18 | `estimator_k3_v31.py` | `6dde5fd4` | DEV | 3.006065e-08 | 0.30101 | 13,037 | 52.02 s (local) | 5,434.6 MB | 8.713e-09 | see above | baseline | — | The DEV control. `experiments/2026-09-16_slice7_canary/`. Reproduce this row in D1: DEV adjusted 9.0486e-09 ± 1%. |
| 2026-09-18 | `estimator_k3_v43_complete_c10.py` | `fd8d497d` | DEV | 3.094966e-08 | 0.21865 | 23,405 | 60.259 s (local) → 78.9 s proj | 5,754.8 MiB | **6.526e-09** | PENDING | 1.3371× vs v31 DEV (score ratio, unpaired) | — | C10. DEV adjusted 6.767156e-09, 100/100, 0 failed, official `v2-phase2` mini, subprocess runner (`experiments/2026-09-18_c10_diagnostic/c10_mini.json`). Raw MSE is 3.0% WORSE than v31; the whole gain is cost (0.30103 → 0.21865 F/B). All §5 gates pass — see below. |
| 2026-09-21 | `estimator_k3_v43_complete_c10.py` | `fd8d497d` | DEV | 3.094966e-08 | 0.21867 | 23,421 | 49.881 s (local) → 79.7 s proj | 3842 MiB | **6.5262e-09** | PENDING | — | — | D1 acceptance on the instrument of record: raw MSE reproduces the Sep-18 official whest run EXACTLY (3.094966e-08). Multiplier is the f64 twin's, hence 2^25 above whest run's. |
| 2026-09-22 | `estimator_k3_v43_complete_c10.py` | `fd8d497d` | GRADED | 2.968645e-08 | 0.21867 | 23,421 | — | — | 6.526e-09 | **6.4914e-09** | 1.3482× vs v31 (graded/graded) | — | **sub #331816, THE NEW BEST. 100/100, 0 failed.** Server residual **0.236255 s** of the 0.4 s cliff (1.69× headroom, 10.09 µs/op) — the first reading at 23k ops, and the reason D0 existed. Graded landed 0.53% BETTER than the 6.5262e-09 projection. Server multiplier matched the f64 twin to 2e-10. best net 5.029e-09, worst 8.43e-09. Report: `experiments/run_reports/c10_canary_331816_graded_report.json`. |
| 2026-09-22 | `estimator_k3_v50_safe.py` | `f9cb9439` | DEV | 3.011684e-08 | 0.21529 | 25,728 | 45.065 s (local) → 72.0 s proj | 5,784.9 MiB | **6.2524e-09** | READY, NOT SENT | raw −2.691% / cost −1.544% vs C10 | — | **D2 fitted coefficients + D3 item-1 Strassen routing.** All seven §5 gates pass, 0 failed nets, 3.7% better than the 6.4914e-09 graded. HOLDOUT (100 of 900) agrees: +2.871% raw against DEV's +2.687%, 0.18 pp apart. Package `experiments/2026-09-22_v50_canary/v50-canary.tar.gz`. |
| 2026-09-22 | `estimator_k3_v53_age5.py` | `8257c055` | GRADED | 2.434000e-08 | 0.22848 | 25,140 | **96.488 s** | 6,457.1 MiB | 5.607e-09 | **5.5603e-09** | 1.1136× vs v50 (graded/graded) | — | **sub #331880, THE NEW BEST. 100/100, 0 failed.** D5's result: old-tier compression age 4→5. Graded 0.84% better than projected — the third row running to beat its projection. Residual 0.268490 s (1.49× headroom, 10.68 µs/op). Multiplier matched the twin exactly; FLOPs delta −491. **WALL 96.488 s EXCEEDS PLAN D7's 95 s finals bound** — a good canary, not a designable entry. |
| 2026-09-23 | `estimator_k3_v50_safe.py` | `f9cb9439` | GRADED | 2.876e-08 | 0.21529 | 25,728 | **87.663 s** | — | — | **6.192e-09** | identical to #331863 | — | **sub #331939, identical bytes rerun (HANDOFF 09-22 Bank). 100/100, 0 failed.** Same score to every digit (the chain is deterministic). Worst wall 87.663 s vs 88.747 s first run (−1.2%); worst residual 0.26291 s vs 0.26550 s. Run-to-run wall spread on identical bytes ≈1.1 s — inside the wall model's ±4.1 s band. v50 stays the safe designate-B candidate. |
| 2026-09-23 | `estimator_k3_v55_p2r4.py` | `aac641b4` | DEV | 2.569027e-08 | 0.21618 | 23,645 | 54.385 s (local) → 91.3 s proj | VmPeak 6,470.5 MB (WSL gate PASS, 23/23) | **5.3300e-09** | PENDING | 1.0470× vs v53 (Δ +2.610e-10, t95 [+2.432e-10, +2.788e-10]) | 100/100 | **v55 = v53 + `_P2_RANK` 32→4** (elasticity table). Raw +0.94% vs v53, cost −5.38%, ops −1,495. All seven §5 gates pass; projected wall 91.3 s is under D7's 95 s bound (v53: 96.7 s). An earlier run of this row was INVALID — 5 MemoryError nets while the box was loaded (`v55_ev_INVALID_memoryerror.log`); rerun clean, 0 failed. Package `experiments/2026-09-23_v55_canary/v55-canary.tar.gz`. |
| 2026-09-23 | `estimator_k3_v55_p2r4.py` | `aac641b4` | GRADED | 2.461e-08 | 0.21618 | 23,645 | **114.487 s** (median 92.8) | — | 5.330e-09 | **5.3200e-09** | 1.0452× vs v53 (graded/graded) | — | **sub #331967, THE NEW BEST. 100/100, 0 failed.** Graded 0.19% better than projected (DEV/graded factor 1.0439). Residual worst 0.28054 s (1.43× headroom). Multiplier matched the twin (FLOPs Δ −491). **WALL ANOMALY: worst 114.487 s, 27 nets > 100 s**, against a 91.3 s projection. v55 does ~5% less work than v53 yet its median (92.8 s) equals v53's (92.7 s), and the slow nets cluster late in the run (index-decile medians 92–95 s for the first 80, 100.8 / 99.6 s for the last 20). v50 — slightly MORE work — graded 85 s median twice. Reads as server-side slowdown, not the candidate; **not designable until an identical-bytes rerun shows a normal wall.** |
| 2026-09-23 | `estimator_k3_v55_p2r4.py` | `aac641b4` | GRADED | 2.461e-08 | 0.21618 | 23,645 | **113.807 s** (median 92.5) | — | 5.330e-09 | **5.3200e-09** | identical to #331967 | — | **sub #331980, identical-bytes rerun of v55. 100/100, 0 failed.** Same score to every digit. Worst wall 113.807 s again, 15 nets > 100 s; per-net wall correlation with #331967 is 0.03 — the slow nets are random, not net-specific. Worst residual 0.27087 s. |
| 2026-09-23 | `estimator_k3_v50_safe.py` | `f9cb9439` | GRADED | 2.876e-08 | 0.21529 | 25,728 | **104.1 s** (median 87.0) | — | — | 6.192e-09 | identical to #331863/#331939 | — | **sub #331989, v50 identical bytes as a SERVER-STATE CONTROL for v55's wall.** 0 failed. The same bytes that graded worst 87.7 s at 03:00 UTC now grade worst 104.1 s with 15 nets > 100 s: **the server got noisier today (+~16 s tail).** See the wall note under Calibration. |
| 2026-09-23 | `estimator_k3_v56b_wallsafe.py` | `b401ad75` | DEV | 2.569219e-08 | 0.23193 | 19,804 | 90.549 s (local) → 62.1 s proj | 3018 MiB | **5.7187e-09** | PENDING | 0.9320× vs estimator_k3_v55_p2r4 (Δ -4.050e-10, t95 [-4.159e-10, -3.941e-10]) | 0/100 | v56b wall-safe twin = v55 at Strassen depth 4 (min leaf 64). Designate-B candidate; fails the 3% score gate by design. |

| 2026-09-23 | `estimator_k3_v56b_wallsafe.py` | `b401ad75` | GRADED | 2.460897e-08 | 0.23193 | 19,804 | **60.7 s** (median 54.5) | 4,908 MiB | 5.7187e-09 | **5.7076e-09** | 1.0729× vs v55 (graded/graded; raw identical) | — | **sub #332063, the WALL-SAFE TWIN. 100/100, 0 failed, 0 nets > 100 s.** Graded 0.19% better than projected. Raw MSE identical to v55's graded 2.461e-08; the whole difference is the +7.28% FLOP multiplier. **Worst wall 60.7 s vs v55's 114 s**: kernel 92.8 ps/FLOP (v55: 176), confirming the leaf-size mechanism of the wall refit; the inferred age-5 penalty for 64-wide leaves does not exist (rate set to 94 in ev.py). Worst/median 1.114. Worst residual 0.2188 s (1.83× headroom). Submitted under HANDOFF 09-23 Step 2's explicit instruction, not the §5 standing approval (fails the 3% score gate by design). **Meets both A (≤105 s) and B (≤95 s) wall policies, and dominates v50 (6.192e-09, 104.1 s).** |
| 2026-09-24 | `experiments/2026-09-24_wallsafe/mix/estimator_k3_v56c_mix__tr64.py` | `52d83de7` | GRADED | 2.461002e-08 | 0.22081 | 22,978 | **80.30 s** (median 76.90) | VmPeak 6,468.4 MB (WSL 23/23) | 5.4444e-09 | **5.4341e-09** | raw identical to v55 | — | **sub #332088, the A/B DESIGNATION WALL READING. 100/100 completed, 0 failed, 0 nets > 100 s.** tr64 = v55 + `_STRASSEN_TRANSPORT_MIN_BLOCK = 64`: 139.6 G (29.4% of the chain) moves to 64-wide leaves. Graded **0.19% better than projected** — the sixth row running to beat its projection. Raw MSE 2.461002e-08 vs v55's graded 2.461e-08: **accuracy is v55's, the whole difference is the +2.14% multiplier**, as designed. **Wall: median 76.90 s, p90 79.68, worst 80.30, worst/median 1.044 (a QUIET day).** Kernel **141.0 ps/FLOP** — between leaf-32's 176 and leaf-64's 92.5, which is what a mixed-leaf chain should read and the first direct confirmation of the leaf-size law on a partial twin. Overhead 357 µs/op (flat, as in every row). Residual worst 0.24128 s (1.66× headroom). The projection (median 81.6, quiet worst 86.1) was **6% conservative**, i.e. wrong in the safe direction. Implied noisy-day worst = 76.90 × 1.24 ≈ **95.4 s**: clears the A bound (105 s) by ~10 s and misses the B bound (95 s) by 0.4 s. **tr64 is the A candidate**; it cuts the designation gap from 7.3% to 2.1%. First slot of UTC day 2026-09-24. |
| 2026-09-24 | κ4-on-M (HANDOFF 09-23 item 4c) | — | — | — | — | — | — | — | — | **NOT RUN** | — | — | **Recorded as not-run, not as a result.** The Sep 23 packet §3.4 listed 4c as "the worker's next task unless you redirect"; the Sep 24 response §5 then redirected to §3/§4 and listed κ4-on-M under "not doing … beyond the one run already in flight". **There was no run in flight** — no experiment directory, no arm file, no log exists for it. HANDOFF 09-24 Step 0 asked for it as a RESULTS row if finished, capped at 2 further hours if not; it is recorded here as never started rather than given 2 hours, because the response prices it at 0–3% and closes it, and the slice's time went to Step 1/Step 2 instead. Reopening it needs a decision, not a rerun. |
| 2026-09-24 | `estimator_k3_v57_corr.py` | `253c7b19` | GRADED | 2.332585e-08 | 0.21626 | 24,186 | **92.81 s** (median 90.43) | VmPeak 6,433.6 MB (WSL 23/23) | 5.0332e-09 | **5.0443e-09** | 1.0547x vs v55 (graded/graded) | 88/100 DEV | **sub #332129, THE NEW BEST (prev 5.3200e-09). 100/100 completed, 0 failed, 0 nets > 100 s.** The in-loop learned mean corrector, weights from SEQUENTIAL teacher forcing on FIT 100. Raw 2.461e-08 -> 2.332585e-08 = **-5.22% raw graded**, against -5.60% on DEV and -4.76% on HOLDOUT-600: the panels agreed. Multiplier matched the f64 twin exactly (0.21625525 vs 0.2162552464). **Wall: median 90.43, p90 91.94, worst 92.81, worst/median 1.026 -- a QUIET day.** Kernel **171.3 ps/FLOP** (v55: 176.3), overhead 359 us/op. Residual worst 0.25071 s (1.60x headroom). **FIRST ROW TO GRADE WORSE THAN PROJECTED**: 5.0443e-09 against 5.0332e-09 = +0.22%, implying a DEV/graded factor of 1.0397 against the 1.042 in use. Inside the 2% recalibration trigger, and the seventh observation now spans 1.0339-1.0471. **NOT DESIGNABLE**: implied noisy worst 90.43 x 1.24 = 112.1 s, over the 105 s A bound -- as projected (115.1 s). The wall-safe twin (v57 at Strassen depth 4) is the designable follow-up. Second slot of UTC day 2026-09-24. |
| 2026-09-24 | `estimator_k3_v58_ws.py` | `ws64` | DEV | 2.425243e-08 | 0.23200 | 20,345 | 58.6 s proj quiet (median 55.5) | VmPeak 4,919.6 MB (WSL 23/23, **1.67x**) | **5.3998e-09** | **READY, NOT SENT** | -5.60% raw vs its own off-arm (identical to v57) | 88/100 | **The v57 WALL-SAFE TWIN** = v57 + `_STRASSEN_MIN_BLOCK = 64`, the same transform v56b applied to v55. Accuracy is v57's: on-arm raw 2.425243e-08 vs v57's 2.425178e-08 = **+0.0027%**, i.e. rounding order. Cost +7.28% multiplier (exactly v55->v56b's inflation), ops 24,186 -> **20,345**. Residual 0.2256 s (**1.77x** headroom). Projected noisy worst **68.8 s** -- meets BOTH the A (105 s) and B (95 s) bounds with 51 s of margin to the cliff. **It retires tr64**: better on score (5.3998e-09 vs 5.4341e-09) AND on wall (68.8 vs 95.4 s noisy). Fails §5's 3% score gate by design (+7.05% vs the 5.0443e-09 best graded), so it needs the A/B designation-safety allowance, not standing approval. **HELD DELIBERATELY, not blocked.** Two reasons: the A/B allowance is one slot per UTC day and `tr64` #332088 spent today's; and the 400-net sequential refit (worker-q 2112, ~7 h) may produce a better v57, whose twin would supersede this file. Spending a slot on a candidate about to be superseded is the waste the slot budget exists to prevent. Submit on Sep 25 UTC, built from whichever v57 is then best. |
| 2026-09-24 | `estimator_k3_v59_corr400.py` | `seq400` | DEV | 2.384984e-08 | 0.21626 | 24,186 | 97.9 s proj quiet | (v57's, 6,434 MB) | **4.9498e-09** | **HELD — FAILS §5** | **-7.17%** raw vs v55 | **99/100** | **The 400-net sequential refit** (FIT 100 + pinned HOLDOUT-300, worker-q 2112, 3h07m). DEV -7.17% against the 100-net fit's -5.60%: **more data was worth 1.57 pp**, as the h64-vs-h32 capacity result predicted. Per-layer val R^2 rose where it matters — layer 16 0.00946 -> **0.01260** (+33%), layer 8 +27% — and the train/val gaps closed to ~0.0015, so 400 nets largely removes the data limitation. **But the BAR MOVED**: v57 #332129 is now the best graded at 5.0443e-09, so v59's projected 4.9498e-09 is only **1.87% better** and **fails PLAN §5's 3% gate**. Not submittable under standing approval. It needs ~1.2 pp more, and the untried lever is the handoff's recursive feature (the transported previous correction), which could not be measured before a corrector existed. |
| 2026-09-24 | `estimator_k3_v60_ws400.py` | `e9344107` | DEV | 2.384984e-08 | 0.23200 | 20,345 | 58.6 s proj quiet (median 55.5) | VmPeak 4,941.3 MB (WSL 23/23, **1.66x**) | **5.3102e-09** | **READY — submit Sep 25 UTC** | -7.41% raw on DEV 40 vs its own off-arm | 39/40 | **The v59 WALL-SAFE TWIN, and the best DESIGNABLE candidate we have.** v59 + `_STRASSEN_MIN_BLOCK = 64`. Cost 510,179,282,809 FLOPs = 0.232003 x 2^41, ops 20,345, 0 materializations, residual 0.2256 s (**1.77x**). Projected noisy worst **~68.8 s** — meets both the A (105 s) and B (95 s) bounds with **51 s of margin** to the cliff. validate 119 ms; validate-package OK. **It retires BOTH** `tr64` (5.4341e-09 @ ~95.4 s) **and** `v58_ws` (5.3998e-09 @ ~68.8 s), beating each on score at equal-or-better wall. Fails §5's 3% score gate by design (+5.27% vs best graded), so it goes under the A/B designation-safety allowance — whose Sep 24 slot `tr64` already spent, hence Sep 25. |
| 2026-09-25 | `estimator_k3_v61_rec.py` | `rec400` | DEV | 2.388179e-08 | 0.21634 | 24,202 | — | — | 4.9585e-09 | **CLOSED — no gain** | -7.04% raw vs v55 | 96/100 | **The RECURSIVE feature is worth NOTHING, and that closes the last untried lever in the spec.** `(W_l^T c_{l-1})_i`, the previous layer's applied correction transported into this layer's coordinates — HANDOFF 09-24 Step 2 lists it, and it is the one feature that could not be measured before a corrector existed, so every ceiling number on record excluded it. Same 400-net sequential procedure, 10 raw channels -> 35 expanded columns instead of 9 -> 33. Per-layer val R^2 is identical to within noise and mostly a hair WORSE (layer 16 **0.01244** vs 0.01260; layer 8 0.03542 vs 0.03637; layer 2 0.16351 vs 0.16352), and end to end it is **-7.04% against -7.17%**. **Interpretation:** the channel is redundant. `mu_pre = W^T mu` already carries the previous correction — the corrector was never missing that information, only unable to separate it, and separating it turns out not to matter. Fit in 2h04m (worker-q 2124). |
| 2026-09-25 | `estimator_k3_v60_ws400.py` | `e9344107` | GRADED | 2.301125e-08 | 0.23200 | 20,345 | **57.59 s** (median 55.73) | VmPeak 4,941.3 MB (WSL 23/23) | 5.3102e-09 | **5.3387e-09** | raw **-1.35%** vs v57 GRADED | 99/100 DEV | **sub #332223, THE DESIGNATION CANDIDATE. 100/100 completed, 0 failed, 0 nets > 100 s.** The 400-net corrector at Strassen depth 4. **Wall: median 55.73, p90 56.91, worst 57.59, worst/median 1.033.** Kernel **94.6 ps/FLOP** against the 94 used in the model — the leaf-64 rate is now confirmed on a third candidate. Residual worst 0.21211 s (**1.89x** headroom). Implied noisy worst **69.1 s** against a 68.8 s projection: **meets A (105 s) and B (95 s) with ~51 s of margin to the cliff.** **THE RAW MSE IS THE OTHER RESULT:** 2.301125e-08 against v57's graded 2.332585e-08 = **-1.35% raw, GRADED** — the 400-net refit's DEV gain transferred to the server. It follows that the same corrector at leaf 32 (`estimator_k3_v59_corr400.py`) would grade **4.9763e-09**, i.e. 1.35% better than the best graded and still short of §5's 3% gate. |
| 2026-09-25 | `estimator_k3_v59_corr400.py` | `f419f765` | GRADED | 2.300780e-08 | 0.21626 | 24,186 | **94.42 s** (median 89.84) | (v57's code path, 6,434 MB) | 4.9763e-09 derived | **4.9756e-09** | **1.0138x vs v57 #332129** (graded/graded) | 99/100 DEV | **sub #332232, THE NEW BEST (prev 5.0443e-09). 100/100 completed, 0 failed, 0 nets > 100 s.** First canary under PLAN rev 13 decision 1 (any projected gain over the best graded; the 3% gate is retired). The 400-net corrector at leaf 32. **Raw 2.300780e-08 vs v60_ws400's 2.301125e-08: the same accuracy to 0.015%**, as designed -- the two files differ only in `_STRASSEN_MIN_BLOCK`. The score was DERIVED before submission from #332223's graded raw x this multiplier (4.9763e-09) and graded 0.01% better. Against the DEV projection at the current 1.038 (5.1577e-09 / 1.038 = 4.9689e-09) it graded +0.13% worse: implied factor **1.0366**, the NINTH observation (range now 1.0339-1.0471, mean 1.0399), no change to the constant. **Wall: median 89.84, p90 92.66, worst 94.42, worst/median 1.051 -- a quiet day.** Kernel 170.3 ps/FLOP (v57: 171.3), overhead 8.65 s. Residual worst 0.25114 s (1.59x). Multiplier = twin's 0.21625525 exactly. **Implied noisy worst 89.84 x 1.24 = 111.4 s: meets the rev-13 A policy (<= ~115 s)** by ~3.6 s and the 120 s cliff by ~8.6 s. `docs/DESIGNATION.md`: A = v59, B = v60_ws400. First slot of UTC day 2026-09-25. |
| 2026-09-25 | `estimator_k3_v62_s1.py` | `d1d722db` | DEV | 2.370347e-08 | 0.21626 | 24,186 | 62.291 s (local) → 98.3 s proj (= v59's code) | VmPeak 6,445 MiB (WSL PASS) | **4.9383e-09** | PENDING #332313 | 1.0061× vs v59 (Δ +3.147e-11, t95 [+1.698e-11, +4.597e-11]) | 63/100 | **Twin S1** (corrector pages fitted end to end through the chain, 900 TRAIN nets). Official numpy DEV agrees with the twin's 2.370264e-08 to 3.5e-5. All hard gates pass; submitted as the second canary of Sep 25 under decision 1. Data lines only vs v59, so its wall is v59's measured one. |
| 2026-09-25 | `estimator_k3_v62_s1.py` | `d1d722db` | GRADED | 2.281909e-08 | 0.21626 | 24,186 | **102.61 s** (median 90.27) | (v59's code path) | 4.9383e-09 | **4.9347e-09** | **1.0083× vs v59 #332232** (graded/graded) | 63/100 DEV | **sub #332313, THE NEW BEST (prev 4.9756e-09). 100/100, 0 failed.** The twin's S1 corrector pages -- the first number the end-to-end fit has put on the board. Raw −0.82% graded vs −0.61% DEV / −0.76% HOLDOUT-100: the panels transferred. Graded 0.07% better than projected (implied factor 1.0387; tenth observation, constant stays 1.038). **Wall: median 90.27, p90 97.03, worst 102.61, worst/median 1.137, 5 nets > 100 s** -- the SAME code as v59 (median 89.84, worst 94.42 this morning), so this is a moderately noisy server, and it is the first leaf-32 reading of the day at that tail. Implied noisy worst 90.27 × 1.24 = 111.9 s: inside the rev-13 A policy (≤ ~115 s), 8 s from the cliff. Residual worst 0.27131 s. Second canary of UTC day 2026-09-25. |
| 2026-09-25 | `estimator_k3_v64_s2.py` | `e92f2e7e` | DEV | 2.359028e-08 | 0.21626 | 24,186 | 61.439 s (local) → 98.3 s proj (= v59's code) | VmPeak 6,458 MiB (WSL PASS) | **4.9148e-09** | PENDING #332350 | 1.0048× vs v62_s1 (Δ +2.448e-11, t95 [+1.930e-11, +2.966e-11]) | 83/100 | **Twin S2**: per-layer λ/θ/feed gains (moved up to ±19%) trained jointly with the corrector pages through the chain, 900 TRAIN nets; HOLDOUT-100 −0.42%, twin DEV −0.475% (official numpy agrees to 1.4e-5). Gains are pure-Python tables: FLOPs and ops identical to v62_s1. Third canary of Sep 25. |
| 2026-09-25 | `estimator_k3_v64_s2.py` | `e92f2e7e` | GRADED | 2.271571e-08 | 0.21626 | 24,186 | **94.47 s** (median 91.11) | (v59's code path) | 4.9148e-09 | **4.9124e-09** | **1.0045× vs v62_s1 #332313** (graded/graded) | 83/100 DEV | **sub #332350, THE NEW BEST (prev 4.9347e-09). 100/100, 0 failed, 0 nets > 100 s.** Twin S2 (end-to-end λ/θ/feed gains + corrector). Raw −0.45% graded vs −0.48% DEV / −0.42% HOLDOUT-100. Graded 0.05% better than projected (implied factor 1.0385). **Wall: median 91.11, p90 93.36, worst 94.47, worst/median 1.037 -- quiet.** Implied noisy worst 91.11 × 1.24 = 113.0 s: inside the A policy (≤ ~115 s). Residual worst 0.2531 s. Cumulative twin gain vs v59 graded: −1.27% raw. Third canary of UTC day 2026-09-25. |
| 2026-09-26 | `estimator_k3_v64_ws.py` | `(v64_s2+leaf64)` | GRADED | 2.271497e-08 | 0.23200 | 20,345 | **63.55 s** (median 55.27) | VmPeak 4,962 MiB (WSL PASS) | 5.2701e-09 derived | **5.2699e-09** | raw = v64_s2's to 3e-5 | — | **sub #332364, THE NEW B (prev v60_ws400 5.3387e-09). 100/100, 0 failed, 0 nets > 100 s.** v64_s2 at Strassen leaf 64, on the Sep 26 A/B slot. **Wall: median 55.27, p90 61.00, worst 63.55, worst/median 1.150 -- a NOISY-day reading (≥ 1.15)**, the first for a wall-safe twin (strategist response 09-25 §4): 63.6 s against B's 95 s policy, 56 s to the cliff. Residual worst 0.2252 s (1.78x). |
| 2026-09-26 | `estimator_k3_v65_s2t.py` | `29a65bd2` | DEV | 2.312891e-08 | 0.21634 | 24,438 | 53.269 s (local) → 98.4 s proj | VmPeak 6,451 MiB (WSL PASS) | **4.8206e-09** | PENDING #332393 | 1.0195× vs v64_s2 (Δ +9.770e-11, t95 [+8.299e-11, +1.124e-10]) | 92/100 | **Twin S2t**: per-term closure gains (13 table + 8 one-D + 12 uni + 27 biv per layer) with profiles and corrector, re-fitted at lr 3e-4 (2e-3 diverged). HOLDOUT-100 −2.39% vs S2b; twin DEV −1.96% vs S2 (official agrees to 3.5e-5); −3.0% vs v59. First canary of Sep 26. |
| 2026-09-26 | `estimator_k3_v65_s2t.py` | `29a65bd2` | GRADED | 2.224876e-08 | 0.21634 | 24,438 | **94.34 s** (median 89.27) | (v59's code path) | 4.8206e-09 | **4.8134e-09** | **1.0206× vs v64_s2 #332350** (graded/graded) | 92/100 DEV | **sub #332393, THE NEW BEST (prev 4.9124e-09). 100/100, 0 failed, 0 nets > 100 s.** Twin S2t. Raw −2.06% graded vs −1.95% DEV / −2.39% HOLDOUT-100. Graded 0.15% better than projected (implied factor 1.0396). **Wall: median 89.27, p90 91.70, worst 94.34, worst/median 1.057 -- quiet.** Implied noisy worst 110.7 s (A policy ≤ ~115 s). Residual worst 0.2606 s. **Cumulative twin gain vs v59 graded (4.9756e-09): −3.26% score.** First canary of UTC day 2026-09-26. |
| 2026-09-26 | `estimator_k3_v66_s2t2.py` | `5766053c` | DEV | 2.296551e-08 | 0.21634 | 24,438 | 98.4 s proj (= v59's code) | VmPeak 6,462 MiB (WSL PASS) | **4.7866e-09** | PENDING #332409 | 1.0071× vs v65_s2t (Δ +3.535e-11, t95 [+3.046e-11, +4.024e-11]) | 95/100 | **Twin S2t2** (S2t continued 2 epochs, lr 3e-4); HOLDOUT-100 −0.63%. Official agrees with the twin to 2e-5. Second canary of Sep 26. |
| 2026-09-26 | `estimator_k3_v66_s2t2.py` | `5766053c` | GRADED | 2.211922e-08 | 0.21634 | 24,438 | **101.10 s** (median 89.24) | (v59's code path) | 4.7866e-09 | **4.7854e-09** | **1.0059× vs v65_s2t #332393** (graded/graded) | 95/100 DEV | **sub #332409, THE NEW BEST (prev 4.8134e-09). 100/100, 0 failed, 1 net > 100 s.** Twin S2t2. Raw −0.58% graded vs −0.71% DEV / −0.63% HOLDOUT-100. Graded 0.03% better than projected. **Wall: median 89.24, p90 91.51, worst 101.10, worst/median 1.133 -- moderately noisy.** Implied noisy worst 110.7 s (A policy ≤ ~115 s). Residual worst 0.2635 s. **Twin total vs v59 graded: −3.82% score.** Second canary of UTC day 2026-09-26. |
| 2026-09-26 | `estimator_k3_v67_s3.py` | — | DEV | 2.276868e-08 | 0.21643 | 24,648 | 98.5 s proj | VmPeak 6,461 MiB (WSL PASS) | **4.7475e-09** | PENDING #332440 | 1.0082× vs v66 (Δ +4.061e-11, t95 [+3.306e-11, +4.817e-11]) | 85/100 | **Twin S3** (multi-mode κ4_iiij). Third canary of Sep 26. |
| 2026-09-26 | `estimator_k3_v67_s3.py` | — | GRADED | 2.190255e-08 | 0.21643 | 24,648 | **94.44 s** (median 90.74) | (v59's code path) | 4.7475e-09 | **4.7404e-09** | **1.0095× vs v66 #332409** (graded/graded) | 85/100 DEV | **sub #332440, THE NEW BEST (prev 4.7854e-09). 100/100, 0 failed, 0 nets > 100 s.** Twin S3 (multi-mode κ4_iiij). Raw −0.98% graded vs −0.82% DEV / −0.88% HOLDOUT-100. Graded 0.15% better than projected. **Wall: median 90.74, p90 93.85, worst 94.44, worst/median 1.041 -- quiet.** Implied noisy worst 112.5 s (A ≤ ~115 s). Residual worst 0.2566 s. **Twin total vs v59 graded: −4.73% score.** Third canary of UTC day 2026-09-26. |
| 2026-09-26 | `estimator_k3_v68_wino.py` | — | DEV | 2.276931e-08 | 0.21473 | 22,701 | 97.1 s proj quiet / 114.1 noisy | VmPeak 6,736 MiB (WSL PASS) | **4.7103e-09** | PENDING #332478 | 1.0079× vs v67_s3 (adjusted; raw equal to rounding) | 100/100 | **Step 4(a) Strassen-Winograd** on v67. The first build held the eight Winograd block temporaries across the recursive call and peaked at 7,727 MiB (465 MiB under the limit) -- rebuilt with them released before recursing, 6,736 MiB, identical arithmetic. Fourth canary of Sep 26. |
| 2026-09-26 | `estimator_k3_v68_wino.py` | — | GRADED | 2.190577e-08 | 0.21473 | 22,701 | **93.86 s** (median **84.00**) | VmPeak 6,736 MiB | 4.7103e-09 | **4.7039e-09** | **1.0078× vs v67 #332440** (graded/graded) | 100/100 DEV | **sub #332478, THE NEW BEST (prev 4.7404e-09). 100/100, 0 failed, 0 nets > 100 s.** Step 4(a) Strassen-Winograd. Raw = v67's to 0.015%; the gain is the multiplier. **Wall: median 84.00 (v67: 90.74, −6.7 s), p90 87.08, worst 93.86, worst/median 1.117.** The wall fell ~4× more than ops × 0.36 ms + ΔFLOPs × kernel rate predicts (~1.5 s): the Winograd glue is memory-bound work the law under-prices. Implied noisy worst 84.00 × 1.24 = **104.2 s** — A's margin to the ~115 s policy grows from 2.5 s to ~11 s. Residual worst 0.2521 s. **Total vs v59 graded: −5.46% score.** Fourth canary of UTC day 2026-09-26. |
| 2026-09-27 | `estimator_k3_v68_ws.py` | `0db9afba` | GRADED | 2.190628e-08 | 0.23127 | 19,361 | **55.67 s** (median 51.89) | VmPeak 5,085 MiB (WSL PASS) | 5.08e-09 derived | **5.0664e-09** | raw = v68_wino's graded to 2e-5 | — | **sub #332544, THE NEW B (prev v64_ws 5.2699e-09). 100/100 completed, 0 failed.** v68_wino at Strassen leaf 64, on the Sep 27 A/B slot (HANDOFF 09-27 Step 0.1). **Wall: median 51.89, p90 53.00, worst 55.67, worst/median 1.073 -- quiet.** Kernel **88.1 ps/FLOP** (v64_ws: ~92), overhead 6.84 s median. Implied noisy worst 51.89 × 1.24 = **64.3 s**: B policy (≤ 95 s) with ~31 s margin, 64 s to the cliff. Residual worst 0.2063 s (**1.94x**). Report `experiments/run_reports/v68ws_canary_332544_graded_report.json`. First slot of UTC day 2026-09-27. |
| 2026-09-27 | `estimator_k3_v70_s2t3.py` | `45f26986` | DEV | 2.263189e-08 | 0.21473 | 22,701 | 73.4 s (local); code path = v68_wino → inherits its MEASURED median 84.0 s, implied noisy 104.2 s | VmPeak 6,712 MiB (WSL PASS) | **4.6819e-09** | PENDING | 1.0061× vs v68_wino (Δ +2.951e-11, t95 [+1.728e-11, +4.173e-11]) | 74/100 | **S2t3 step-368 parameters** (S3 set continued on TRAIN 900 + d8b 2,048, lr 1.5e-4; worker-q 2229, still running) baked by `build_s2.py` + `build_winograd.py` (pipeline reproduces v67/v68 byte-exact); data-only diff vs v68_wino. VAL-100 (full 900–999, checkpoint-selection set) 2.246946e-08 = −0.79% vs S3; twin DEV 2.263058e-08, official DEV 2.263189e-08 (parity 5.8e-5); raw −0.60% vs v68 DEV at identical FLOPs/ops. Baseline v68 DEV preds reused from its identical-arithmetic build 56b92735 (only a `del` of temporaries differs; bit-identical on 3 overlapping nets). validate OK; package sha 073b58f4, inner estimator.py = 45f26986. Equal-step control (worker-q 2236) queued, attribution only. |
| 2026-09-28 | `estimator_k3_v70_s2t3.py` | `45f26986` | GRADED | 2.172776e-08 | 0.21473 | 22,701 | **84.73 s** (median 80.65) | VmPeak 6,712 MiB (WSL PASS) | 4.6819e-09 | **4.6656e-09** | **1.0082× vs v68_wino #332478** (graded/graded; raw −0.81%) | 74/100 DEV | **sub #332737, THE NEW BEST (prev 4.7039e-09). 100/100 completed, 0 failed, 0 nets > 100 s.** S2t3 step-368 parameters, data-only on v68's code path. Graded 0.35% better than projected: implied DEV→graded factor **1.0416** (eleventh observation; range 1.0339–1.0471), constant stays 1.038. Multiplier = meter exactly (472,199,314,830 FLOPs). **Wall: median 80.65, p90 82.58, worst 84.73, worst/median 1.051 — quiet.** Implied noisy worst 80.65 × 1.24 = **100.0 s** (A policy ≤ ~115 s). Residual worst 0.2372 s (1.69×). First submission of UTC day 2026-09-28. Report `experiments/run_reports/v70_canary_332737_graded_report.json`. |
| 2026-09-28 | `estimator_k3_v71_s2t3f.py` | `dde9a863` | DEV | 2.244217e-08 | 0.21473 | 22,701 | code path = v68/v70 → inherits v70's MEASURED median 80.65 s, implied noisy 100.0 s | VmPeak 6,714 MiB (WSL PASS) | **4.6426e-09** | PENDING | 1.0146× vs v68_wino (t95 [+6.20e-11, +7.84e-11], 94/100); **1.0085× vs v70_s2t3** (Δ +4.074e-11, t95 [+3.10e-11, +5.05e-11], 79/100) | 79/100 | **S2t3 FINAL (step 736) parameters** (VAL 2.232939e-08 = −1.41% vs S3), same build path; data-only diff vs v68/v70. Twin DEV 2.243796e-08 vs official 2.244217e-08: parity **1.9e-4** (inside the 1e-3 twin bar; PLAN §5.2 corrected). Derived from v70's graded raw: 2.172776e-08 × (2.244217/2.263189) × 0.214731 ≈ **4.6265e-09**. |
<!-- ev.py --append inserts new rows directly above this marker. Keep it here. -->

## PLAN rev 15 Lane A — S2t3 and its equal-step control: the gain is steps, not d8b (2026-09-28)

S2t3 = S3 parameter set continued (lr 1.5e-4, accum 8, 736-step cosine) on TRAIN 900 + d8b 2,048
(worker-q 2229, finished). Control = identical flags on TRAIN 900 only, 784-step cosine (the twin takes
whole epochs; worker-q 2236, died of the Sep 27–28 full disk (ENOSPC) at step ~530 — not resumed, the
question is answered). VAL-100 = full rows 900–999 (the twin logs call it HOLDOUT-100; repeatedly read
for checkpoint selection, so a validation set). S3 = 2.264823e-08.

| step | S2t3 (TRAIN + d8b) | control (TRAIN only) |
|---:|---:|---:|
| 230 | 2.255358e-08 | 2.250329e-08 |
| 368 | 2.246946e-08 (−0.79%) | **2.240008e-08 (−1.10%)** |
| 506 | 2.236140e-08 | **2.234042e-08** |
| 736 (end) | **2.232939e-08 (−1.41%)** | — |

**The d8b corpus adds nothing measurable**; the gain over S3b (224 steps, −0.17…−0.28%) is longer
optimisation at small lr. Predictions: strategist −1 to −2% (✓ −1.41% at the end, but not from d8b);
mine −0.3 to −0.8% at S2t3's end ✗ (too low), "most of the gain is steps" ✓ (control ≥ S2t3 at equal
steps). Exports: step 368 → `estimator_k3_v70_s2t3.py` (canary #332737); step 736 →
`estimator_k3_v71_s2t3f.py` (twin DEV 2.243796e-08 = −1.44% vs v68's twin read; official DEV below).

## PLAN rev 15 B1 — all-distinct κ3 projected diagnostic: the error is real, score-relevant, and outside the gates' span (2026-09-28)

`tools/k111_diag.py`, `experiments/2026-09-27_k111/` (worker-q 2245/2250/2252; FIT 0–19 — nets in
TRAIN; twin = v62_s1 + S3 set = v67/v68 accuracy; no refit anywhere). With t = post K3, S = post K21
(T_aab, diagonal 0), W next weights: repeated D3 = Σ W³t + 3 Σ W²(SW), AD = D3_next − repeated; D21
analogue. **Step 0** (float64 dense symmetric tensors, n = 6–8): both identities exact to 1.7e-15;
Sᵀ / multiplicity 2 / un-zeroed diagonal fail at 0.1–10; K21 orientation untransposed (≤ 2% vs ≥ 43%).
In-run: transported D3 = chain tap to 1.1e-7; model AD = inherited + newborn to 2.2e-5.

| source layer l | ‖AD‖/‖D3_next‖ | model AD rel. error D3 / D21 (free) | exact-input mode | noise floor (K21 rep1 + N=1e9 est.) | label/model AD | AD share of D3 error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.22 | 0.043 / 0.042 | same | 0.042 | 1.00 | 0.61 |
| 4 | 0.30 | 0.100 / 0.100 | 0.100 | 0.008 | 0.99 | 0.57 |
| 8 | 0.29 | 0.165 / 0.166 | 0.170 | 0.005 | 0.945 | 0.84 |
| 14 | 0.26 | 0.252 / 0.254 | 0.257 | 0.003 | 0.877 | 0.91 |

Layer 0 (model exact) reads at the floor, which calibrates it. From l = 8 the AD error is 30–75× noise;
exact-input mode gives the same error (it lives in the carried tensor, not in input drift — this does
not separate birth from inheritance). The model's AD is ~12% too large at depth.

Downstream (free-running, every layer, final-layer MSE, median over 20 nets; base 2.15e-08):
AD part → label **−33.1%** (−20.5…−46.4%; rep1 labels −35.5% vs s00 −35.4% on nets 0–7, so not noise);
full exact D3+D21 −46.9%; repeated part only +4.8%; compression mismatch only −3.4%; AD on D3 only
−11.6%, on D21 only −9.6%.

**Span of per-neuron factor gates** (features 1, α, α², γ3, Φ(α) on w2b, cA/cB, inherited row scale;
least squares, leave-one-net-out): AD error at l = 14 0.252 → 0.195 at best; three scalar gains alone
0.201; two-leg only 0.232; hubs only 0.250. The captured ~40% of squared error is a uniform shrink of
inherited transport; **applying it without refit worsens final MSE +13.6…+15.8%**, while adding only the
component orthogonal to the model's AD gives −15.9%. Reading: the end-to-end-trained gains already
compensate the scalar overshoot; the value is in an out-of-span pattern. Predictions (agent): AD ≥ 70% of
the D3/D21 error ✓ at depth (84–96%), ✗ early (57–78%); gates capture < 30% ≈ (40% squared, harmful).
PLAN consequence: B2 gets ONE end-to-end falsification run (transport, gate leaves only); the birth arm
is not run unless it moves.

## PLAN rev 15 B3a — rank screen of the all-distinct error: NOT rejected, compact at depth (2026-09-28)

`tools/k111_diag.py rank`, `experiments/2026-09-27_k111/rank/` (worker-q 2262, 11 min; FIT 0–7, free-running,
v62_s1 + S3 set, no refit). E = label AD − model AD in full D21 form, D3 AD error restored on the diagonal
(it carries only 0.3% of ‖E‖²). Fraction of ‖·‖²_F in the top-r singular directions (median, 8 nets):

| l | E r=8 / 16 / 32 / 64 / 128 | E ⊥ model AD, r = 64 | ‖noise‖²/‖E‖² (rep1) |
|---:|---|---:|---:|
| 2 | 0.15 / 0.23 / 0.34 / 0.50 / 0.70 | 0.50 | 3.5e-3 |
| 5 | 0.31 / 0.42 / 0.55 / 0.69 / 0.83 | 0.67 | 6e-4 |
| 8 | 0.48 / 0.59 / 0.70 / **0.81** / 0.91 | 0.78 | 2e-4 |
| 11 | 0.59 / 0.71 / 0.80 / **0.88** / 0.94 | 0.83 | 1e-4 |
| 14 | 0.67 / 0.78 / 0.85 / **0.92** / 0.96 | 0.86 | 4e-5 |

Truncated oracle (every layer; D3 full AD fix + D21 model + rank-r(E) off-diagonal), final-MSE change, median:
r = 0 (D3 only) −9.5%; r = 8 −13.6%; **r = 32 −18.9%**; r = 128 −30.4%; full −35.3%. Stop rules (rank 64 < 50%
at l ≥ 8; rank-32 oracle < 1/3 of full) do not fire (0.54 of full; 0.36 of the D21-only increment). Energy and
score value do not scale together (rank 32 = 70–85% of energy, ~half the value). **Label-oracle evidence only:
it says a compact extra third-order component COULD carry most of the missing AD content, not that an online
update can produce it.** Prediction (agent): rejection ✗; rank-32 oracle −17…−20% ✓.

## PLAN rev 15 B2 prep — identity-initialised factor gates (2026-09-28)

`tools/build_gate.py`, `experiments/2026-09-27_gates/DESIGN.md`; twin stages G1b/G1t/G1bg/G1tg
(additive). g = 1 + S·tanh(MLP_8(α, γ3, γ4)), W2 = b2 = 0 ⇒ g ≡ 1. Birth arm gates fac1's ends and
w2b's middle with p111_21/S_sep/hub compensation from the same vectors; transport arm gates
`_wick_tower`'s w1 with p111_3/p111_21 recomputed. Identity files bit-identical to v68 (twin raw, hooked,
and real meter 128×6); gradients finite and non-zero. Full-rank P2 control (n = 64, f64): K3/K21 of the
gated tower = table to ≤ 2e-15 while the consumed D3 moves 5–26%. Shipped rank 4: 5–30% of a transport
gate's effect leaks into K3/K21 (birth arm: small, except layers 1 and 6). Real meter at 1024×16: birth
+1.9e-6 of 2^41, +306 ops; transport +1.6e-6, +291 ops (worker-q 2249).

## Step 2 (HANDOFF 09-27) — refit ceilings: κ3 53%, cov 41%, κ4 28%, all-but-μ ≈ 91% (2026-09-27)

Twin `--subst X` (v62_s1 + the S3 parameter set = v67/v68 accuracy): X replaced by its exact label
(full1000, N = 1e9) at every layer ≥ 1 in train AND holdout passes, then the S3 leaves (corrector
pages, λ/θ/feed, term gains, q31) refit on FIT rows 0–79 (220 steps × 8, lr 3e-4), read on FIT
rows 80–99 (HOLDOUT-20). `experiments/2026-09-28_ceil/<X>/train.log`. The control's refit never
beat its init (2.42 → 2.25 → 2.1587e-08, early stop): 80 nets do not move a converged chain.

| X exact | HOLDOUT-20 final MSE (init → best refit) | unrefit share | **ceiling** | Anatomy 2 share (v59, FIT 0–19) |
|---|---|---:|---:|---:|
| control | 2.158156e-08 → (no gain) | — | — | — |
| cov = var + C_off | 1.376388e-08 → 1.283910e-08 | 36.2% | **40.5%** | 42% |
| κ3 = D3 + D21 | 1.102086e-08 → 1.005448e-08 | 48.9% | **53.4%** | 51% |
| κ4 = D4 + D22 + Q31 | 1.993065e-08 → 1.557046e-08 | 7.6% | **27.9%** | 15% |
| all but μ | → 1.845817e-09 (resumed from step 120 after a host-RAM kill) | — | **91.4%** | 80% |

κ3 is the largest ceiling → Step 3's target. κ4 roughly quadruples under refit (the λ/θ/q31
compensation undone), confirming the strategist's point, but stays **below the 30% bar: Step 4(b)
stays closed.** Predictions: strategist cov ≥ 45% ✗, κ3 ≥ 45% ✓, κ4 30–45% ✗ (27.9); mine cov
45–55% ✗, κ3 50–62% ✓, κ4 12–25% ✗ (low), all 82–92% ✓. Two jobs died of host-RAM exhaustion from
another project's 35 GiB process; `all` resumed from its step-120 checkpoint (twin ckpt.pt).

## Step 3.3 (HANDOFF 09-27) — the calibrated closure, embedded and fine-tuned: ≤ 0.75% raw, score-negative (2026-09-27)

`tools/closure_fit.py` (offline fit on exact-input pairs, FIT 0–79; held-out R² uni K1/K2/K3/K4
0.61/0.85/0.87/0.85, pair K11/K21 0.19/0.61) → `tools/build_clos.py` (per-neuron net 4→16→16→4 on
K1..K4, per-pair net 9→16→2 on K11/K21, applied to K right after the closure) → twin stage S5
(S3 leaves + closure nets, TRAIN 900, lr 3e-4, 224 steps, select on HOLDOUT-100).
`experiments/2026-09-29_closure/`. HOLDOUT-100, S3 parameter set = 2.264823e-08:

| arm | HOLDOUT-100 | vs S3 | F/B (real meter) | ops | verdict |
|---|---:|---:|---:|---:|---|
| zero-weight build (identity check) | 2.264823e-08 | exact | — | — | ✓ |
| S3b control (S3 continued, no nets, lr 1.5e-4) | 2.260955e-08 | −0.17% | 0.21473 | 22,701 | — |
| offline init, untuned | 2.379587e-08 | +5.1% | | | double counting |
| **S5** (both nets, fine-tuned) | **2.247849e-08** | **−0.75%** | **0.21833 (+1.68%)** | 23,670 | score-negative |
| S5z (identity start: zero output layers) | 4.61e-08 at step 28 | ×2.04 | | | diverged, cancelled |
| S5 weights, pair net removed | 2.387342e-08 | +5.4% | | | co-adapted |
| **S5u** (per-neuron net only, fine-tuned) | **2.260988e-08** | **−0.17%** | 0.21474 (+0.004%) | 23,145 | = the S3b control |

`estimator_k3_v69_clos.py` (S5 baked, Winograd) reproduces the twin (HOLDOUT-100 2.247738e-08 vs
2.247849e-08; DEV twin 2.259824e-08 = −0.74% vs v68) but its FLOP multiplier is +1.68%: **not
submitted**. The per-neuron correction is free but worth exactly as much as more training of the
existing parameters (S5u = S3b to 1e-5 relative). **Conclusion:** the closure bias is real (the
transported var residual equals the observed per-layer drift) and predictable offline (R² 0.6–0.87),
but in the loop the chain's existing trainable compensations (λ/θ/q31 profiles, term gains, the μ
corrector) have already absorbed almost all of it; what remains is worth ≤ 0.75% raw and lives in
the O(n²) pair correction, whose cost exceeds it. Handoff prediction buckets — strategist: ≥ 20%
(35%), 5–20% (35%), < 5% (30%) → **< 5%**. Mine: −3% to −8% after S5 ✗ (−0.75%).

## Step 3.1–3.2 (HANDOFF 09-27) — closure bench and regression: the lane is OPEN (2026-09-27)

**Bench** (`tools/closure_bench.py`, FIT 0–99, twin on GPU): every pre-activation object the closure
reads — μ, var, C11, D3, D21, dG, D22, Q31 — replaced by its exact label at every layer, the
closure's own K tapped and compared with exact post-ReLU moments. Median over 100 nets of the
relative Frobenius residual; **layer 0** (exact chain, labels only) is the label-noise ceiling —
an upper bound, since at l ≥ 1 inputs and targets share Monte-Carlo samples. `var_next`/`Coff_next`
= the covariance residual as the next layer receives it (Wᵀ ΔC W vs exact pre var / C_off at l+1),
directly comparable with Anatomy 2's per-layer state-error increments (var ~1.3e-4, C_off ~6e-4).

| layer | K1 | K2 | K11 | K3 | K4 | K21 | K22 | var_next | Coff_next |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (noise) | 2.4e-5 | 5.5e-5 | 1.4e-3 | 1.2e-4 | 3.0e-4 | 2.0e-2 | 1.5e-2 | 4.6e-5 | 8.5e-4 |
| 1 | 4.8e-6 | 1.8e-5 | 5.5e-4 | 5.2e-5 | 5.7e-4 | 6.6e-3 | 8.8e-2 | 2.5e-5 | 3.6e-4 |
| 4 | 5.1e-6 | 2.4e-5 | 6.6e-4 | 1.3e-4 | 1.8e-3 | 7.1e-3 | 6.6e-2 | 5.4e-5 | 5.4e-4 |
| 8 | 7.9e-6 | 5.1e-5 | 7.4e-4 | 4.2e-4 | 6.6e-3 | 1.2e-2 | 5.5e-2 | 8.9e-5 | 6.6e-4 |
| 12 | 9.1e-6 | 1.0e-4 | 8.0e-4 | 9.9e-4 | 1.7e-2 | 1.4e-2 | 6.7e-2 | 1.3e-4 | 7.4e-4 |
| 14 | 9.1e-6 | 1.4e-4 | 9.3e-4 | 1.4e-3 | 2.1e-2 | 1.2e-2 | 8.2e-2 | 1.7e-4 | 8.9e-4 |

(K21 orientation checked: the chain's K[(2,1)] matches bench `post_K21` untransposed — transposed
reads 0.45–0.85.) **Kill rule does not fire**: the transported variance residual grows with depth
to 0.9–1.7e-4 at layers 10–14 — the SAME size as the observed per-layer var drift — so the
covariance drift is closure bias, not only amplification. K3's marginal residual is 1–30% of the D3
increment and K21's (0.6–1.4%) is under its N = 1e8 label floor: the κ3 drift (53% ceiling) is not
visible in the marginal / (2,1) closure outputs — it lives in the all-distinct κ3 births the bench
does not read. K4 grows to 2% at depth (κ4 ceiling 28%).

**Regression** (`tools/closure_regress.py`; features = the closure's own normalised inputs; fit on
FIT 0–59, R² on held-out nets 60–99; `experiments/2026-09-29_closure/regress.json`):

| residual | ridge (deg-2 poly) | **mlp 16-16** | mlp + exact pre κ5, κ6 (diagnostic) | per-layer bias share |
|---|---:|---:|---:|---:|
| K1 (layers 1–14) | 0.03 | **0.63** | 0.92 | 1% |
| K2 | 0.33 | **0.85** | 0.95 | 2% |
| K3 | 0.35 | **0.87** | 0.95 | 23% |
| K4 | 0.61 | **0.85** | 0.90 | 6% |
| K1 at layer 15 (scored) | 0.01 | **0.52** | — | 1% |
| K11 (8,192 pairs/layer) | 0.06 | **0.68** | — | 1% |
| K21 / K12 | 0.41 | **0.75** | — | 5% |

The residual is a smooth NONLINEAR state function (ridge fails; no clean analytic term), not a
per-layer constant; the unobservable κ5/κ6 truncation adds 5–30 pp. Handoff 3.3 route: elementwise
nets, initialised from this fit, then fine-tuned through the twin. Predictions: mine (kill rule does
not fire ✓; univariate R² 0.3–0.6 with ridge ✓, higher with mlp ✓; κ5/κ6 add < 0.1 ✗ — they add
0.1–0.3 for K1/K4).

## Step 1 (HANDOFF 09-27) — the harmonic old tier: CLOSED, the radial part carries ~70% of the tier's error budget and leaves ×6.6 (2026-09-26)

`tools/build_harm.py` on v68_wino: at the end of each layer the lineage that ages out (5 kept
exact, = v68's exact set) is reduced to its radial part Sym(u ⊗ C) in the post-activation
metric; u joins one carried n-vector U (U ← W^T U, U ← w11·U), consumed as D3 += U∘var,
D21[a,i] += (2U_a C_ai + U_i var_a)/3; the lineage and the shared rank-256 basis are dropped.
`tools/harm_check.py`: ARC's `proj_geq_r` constant reproduced (1.5e-8 = ARC's f32 coefficient),
factored trace vs dense 1e-15, consumer slices vs dense transported radial tensor 7e-15.
(The handoff's written D21 consumer is the transpose of the file's `D21[a,i] = T_aai`.)

**Metric C is ill-posed.** v68's own `cov_out` is PD but its condition runs 1e3 (layer 1) → 1e11
(layer 6) → 1e15 (layer 12), from near-dead neurons; C⁻¹ weights those directions, u is garbage,
the H3c chain diverged (LinAlgError at layer 14). So u was chosen to make the remainder traceless
under a well-posed weight K: u = 3(τI + 2CK)⁻¹(T:K), first order (≤ 0.8% error for diag, 2–63%
for euc at depth): **d** K = diag(1/var), **r** K = (C + mean(var) I)⁻¹, **e** K = C (Euclidean
least squares), and the control **z** = drop the lineage, carry nothing.

DEV net 0, real f32 chain; FLOPs from the real f64 meter (`experiments/2026-09-27_harm/cost.json`):

| arm | final MSE | × v68 raw | F/B | ΔFLOPs | ops | score × v68 |
|---|---:|---:|---:|---:|---:|---:|
| v68_wino | 2.109e-08 | 1 | 0.21473 | — | 22,701 | 1 |
| H5z (drop, U ≡ 0) | 4.085e-07 | ×19.4 | 0.15450 | −28.0% | 17,673 | ×13.9 |
| H5e (K = C) | 1.662e-07 | ×7.9 | 0.16995 | −20.9% | 18,964 | ×6.2 |
| H5d (K = diag) | 1.451e-07 | ×6.9 | 0.15475 | −27.9% | 18,614 | ×5.0 |
| **H5r (ridge)** | **1.392e-07** | **×6.6** | 0.17972 | −16.3% | 19,034 | ×5.5 |

The old tier is **28% of v68's FLOPs** (H5z) and is worth a factor 19 in raw MSE. The radial
(r = 1) part recovers (4.09 − 1.39)/(4.09 − 0.21) ≈ **70% of the tier's excess error**; the
traceless remainder carries the other 30%, which alone is ≈ 6× v68's whole error. Handoff
decision: > +50% raw ⇒ **closed**; no gain fit (16 gains cannot move ×6.6 to ×1.1) and no
rank-16 remainder day (that lane opened at +10–50%). Write-up result: *the old third cumulant is
not isotropic about the covariance — its value to the final mean lies in the traceless part.*
Predictions: strategist H5c ≤ +10% at −20% FLOPs [35%] — wrong (H5c unrunnable; best well-posed
arm +560%); strategist H5d ≥ 2× worse than the metric arm — wrong (d ≈ r ≈ e). Mine: +40–150%,
P(≤ +10%) = 10% — right in direction, 4× too small in magnitude; H5z ≈ ×8–11 — too small (×19).
**DEV-100 confirmation (twin, float32 torch):** v68 2.2766e-08; H5e 1.3507e-07 (**×5.93**, per-net
×4.3–7.9); H5r 1.4274e-07 (×6.27); H5z 4.0580e-07 (×17.8); H5d 4.0467e-07 (×17.8 — i.e. the twin's
H5d carries ~nothing, unlike the f32 numpy chain's ×6.9 on net 0; its 1/var weight reaches 1e12 on
near-dead neurons, a float32 casualty on the torch backend — not chased, it cannot change the
verdict). Closed stands: the best radial arm is ×5.9 over 100 nets.

## Step 4(a) — Strassen-Winograd (15 block adds per level instead of 18) (2026-09-26)

`tools/build_winograd.py` replaces `_sb`'s classic Strassen block (10 pre + 8 post additions) with
Winograd's variant (8 + 7) — the same seven products, still one batched matmul per level; identity
self-tested exact to 2e-15 in float64. On v67_s3 → `estimator_k3_v68_wino.py`:

| | F/B | FLOPs | ops | twin DEV raw vs v67 |
|---|---:|---:|---:|---:|
| v67_s3 | 0.21643 | 475,937,828,857 | 24,648 | — |
| **v68_wino** | **0.21473** | 472,199,315,449 (**−0.79%**) | **22,701 (−1,947)** | −0.014% ± 0.039%, 49/100 (rounding only) |

**v68_ws (leaf-64 B twin): F/B 0.23127, 19,361 ops, residual 0.215 s, projected noisy worst 68.2 s.**
Its first WSL gate run FAILED the two-sided check "settled VmSize did not change across repeats"
at −404 MB (the allocator returning memory after the first predict: 2.58 → 2.17 → 2.18 GB); a
re-run PASSED at +20 MB (2,309 → 2,329 MiB; peak 5,085 MiB). Allocator noise, not a leak — both
runs recorded (`experiments/2026-09-26_wino/v68ws_gates*.json`).

A free ~0.8% on score and ~1,950 fewer op calls (≈ −0.7 s median wall at 0.36 ms/op, −0.02 s
residual) — the first wall-engineering step toward PLAN Step 4's "v59-class FLOPs at ≤ 95 s".

## Step 4(c) / Step 3 rank item — shared old-tier rank, both directions: 256 is the optimum (2026-09-26)

v67_s3 with `_SHARED_RANK` changed (age 5 unchanged); real-meter cost (net 0), twin DEV 100 unrefit,
paired against v67_s3 raw-twin (2.276898e-08). `experiments/2026-09-26_rank/`.

| rank | F/B | ΔFLOPs | ops | Δraw | wins | **Δscore** |
|---|---:|---:|---:|---:|---:|---:|
| 192 | 0.20398 | −5.75% | 22,854 | +21.0% | 0/100 | **+14.0%** |
| 224 | 0.21252 | −1.81% | 22,854 | +6.77% | 5/100 | **+4.84%** |
| **256 (v67)** | 0.21643 | — | 24,648 | — | — | — |
| 288 | 0.22481 | +3.87% | 24,648 | −3.45% ± 0.49 | 91/100 | **+0.29%** |
| 320 | 0.23351 | +7.89% | 24,648 | −5.47% ± 0.62 | 96/100 | **+1.99%** |

The old tier is valuable (every 32 of rank moves raw 3.5–14%) but rank 256 sits at the SCORE optimum
in both directions: accuracy per FLOP is almost exactly the multiplier's price. Closed both ways. (The
op count falls at 192/224 because non-power-of-two thin shapes take fewer Strassen levels.)
Predictions (mine): downward raw 3× larger than predicted; upward raw right, FLOPs ~2× larger.

## Step 4(b) (PLAN rev 13) — birth-layer-aware fold age: KILLED (2026-09-26)

`tools/build_fold.py` on v65_s2t: lineages born at layer ≤ U fold into the shared rank-256 basis at
age A instead of 5 (a function of layer index only — not PLAN §5's energy-adaptive age). Cost from the
real f64 meter (`experiments/2026-09-26_fold/cost.json`); accuracy from the twin, DEV 100, paired
against the `off` arm (which reproduces v65_s2t's FLOPs to the FLOP and its twin DEV read to 1.3e-6).

| arm | F/B | ΔFLOPs | ops | Δraw | wins | **Δscore** |
|---|---:|---:|---:|---:|---:|---:|
| off (= v65_s2t) | 0.21634 | — | 24,438 | — | — | — |
| A4U3 | 0.21051 | −2.69% | 24,725 | +7.65% | 6/100 | **+4.75%** |
| A3U3 | 0.20468 | −5.39% | 25,012 | +31.6% | 0/100 | **+24.5%** |
| A3U5 | 0.19848 | −8.26% | 25,012 | +50.9% | 0/100 | **+38.4%** |

**Killed** (kill line: < 2% score gain). PLAN's premise — early lineages are contracted ×0.6–0.88 per
layer, so their old-tier content matters little by layer 16 — is wrong by a wide margin: compressing
the layer 0–3 births two layers early costs 8–51% raw. This agrees with D5 (the old tier is
under-funded, not over-funded). Prediction (mine: neutral, no arm clears) right in sign, 10–20×
too small in magnitude. Caveat: the corrector and gains were fitted at age 5; a refit cannot plausibly
recover +31%.

## Step 2 (HANDOFF 09-25) — the differentiable twin, and stage S1 (2026-09-25)

**The twin runs the shipped file itself.** `tools/twin/twin.py` imports the estimator with
`flopscope` replaced by a torch shim (`tools/twin/shim`), so autograd flows through exactly the
arithmetic that ships (Strassen, tower, closure, κ4 regeneration, corrector). Acceptance: **DEV
100 raw 2.385123e-08 vs v59 2.384984e-08 (+0.006%, bar 0.1%)**. Per-net parity on FIT 0–2:
+7.5e-4 / +6.3e-5 / −8.4e-5 relative. The handoff's per-net 1e-4 bar is below what any two float32
backends reach (cupy vs the official numpy chain: worst net 1.9e-3, § "Two accuracy backends"), so
the panel criterion binds. 0.6 s/net forward; 3.4 s/net/step with per-layer checkpointing, 5.45 GiB
VRAM — but 18 s/net/step in the S1 run while other projects held all 16 CPU threads.

Two traps, both fixed before any number was trusted: (1) Adam at the handoff's absolute lr 1e-3
took train MSE 2.4e-8 → 3.5e-3 in one step, because the corrector's output page carries the target
scale (~1e-4) — steps are now relative, lr_k = lr × RMS(tensor_k); (2) a parameter-file key bug made
the first S1 DEV read silently evaluate v59 — `install` now refuses unknown keys.

**S1 — corrector pages end to end** (900 TRAIN nets, 2 epochs, relative lr 2e-3, cosine, early
stop on HOLDOUT-100; worker-q 2144, 9h02m):

| panel | v59 | S1 | paired | wins |
|---|---:|---:|---:|---:|
| HOLDOUT-100 (rows 900–999) | 2.386798e-08 | **2.368646e-08** | −0.76% | — |
| DEV 100 (twin) | 2.385123e-08 | **2.370264e-08** | **−0.62% ± 0.28%** | 65/100 |

HOLDOUT−DEV gap 0.14 pp (alarm at 1.5 pp). Predictions: strategist −3 to −6%, mine −1.5 to −3.5% —
**both too optimistic**; end-to-end credit assignment adds little on top of the sequential
teacher-forced fit. Under the handoff's rule (S1 < 1%) no further S1 work; its pages are kept as
the base for S4, and the file `estimator_k3_v62_s1.py` (v59 + S1 pages, data lines only) goes
through the §5 gates as a canary candidate under decision 1 (projected ≈ 4.945e-09 < 4.9756e-09).

**S2 — λ/θ/feed gains + corrector pages, on S1** (`tools/build_s2.py` → `estimator_k3_v64_s2.py`;
900 TRAIN, 2 epochs; worker-q 2154, 2h41m): HOLDOUT-100 2.368646e-08 → **2.358810e-08 (−0.42%)**;
DEV (twin) **−0.475% ± 0.10%, 83/100** vs S1; official numpy −0.48%. Learned gains: θ +10–16% over
layers 3–8 and +10% at layer 15, feed +8–19% over layers 4–8 and 14–15, λ within ±8%. Prediction
(mine −0.5 to −2%) in range. Cumulative twin gain vs v59: −1.1% raw DEV.

**S2b / S2t — convergence, then per-term gains** (all on 900 TRAIN; `tools/build_s2.py` bakes every
gain into Python tables/constants). S2b (S2 continued, lr 2e-3): HOLDOUT-100 −0.13% more. S2t
(adds 60 gains per layer: the 13 bivariate table terms, the 1-D κ3/κ4 rows per order, the 3×4
univariate and 27 bivariate closure extras): at lr 2e-3 it **diverged** (+53% at step 28, early
stop); at **lr 3e-4** HOLDOUT-100 2.355795e-08 → **2.299547e-08 (−2.39%)**, DEV (twin) **−1.96% ±
0.29% vs S2, 90/100**; −3.0% vs v59. The learned term gains are small (table ±1%, closure extras
±2.5%), so a good part of the gain is the corrector and profiles settling at the smaller step — the
2e-3 schedule was too coarse for this landscape all along. Prediction (mine −0.3 to −1.0%) too low.
Cost: +0.04% FLOPs, +252 op calls.

**S2t2 — S2t continued 2 epochs at lr 3e-4**: HOLDOUT-100 2.299547e-08 → **2.285008e-08 (−0.63%)**;
DEV (twin) **−0.70% ± 0.10% vs S2t, 95/100** → `estimator_k3_v66_s2t2.py`.

**S3 — multi-mode κ4_iiij on S2t2** (5 per-layer coefficients on {ρ, ρ², Φ(α_i)ρ, α_iα_j, 1} added to
the normalised Q31 the closure reads; zero-initialised; `tools/build_s2.py` bakes them; worker-q 2176):
HOLDOUT-100 2.285008e-08 → **2.264823e-08 (−0.88%)**; DEV (twin) **−0.86% ± 0.15% vs S2t2, 89/100** →
`estimator_k3_v67_s3.py` (export reproduces the twin to 6.3e-5). The learned correction SHRINKS the
slice (ρ and Φρ coefficients −0.001 → −0.006 with depth), although Anatomy 2's truth-fit λ was larger
than the chain's: end to end, the κ4_iiij channel is compensating for errors elsewhere, not being
made more accurate. Anatomy 2 capped S3 at ~3% of the 15% κ4 share; it delivered 0.9% of score-relevant
error because the fit is to the output, not to the slice.

**S4 — sigma page on var_pre** (`estimator_k3_v63_sig.py` = v62_s1 + a 27→32→1 page rescaling the
variance consumed at each layer; standardisers from 20 TRAIN nets; 900 TRAIN, 2 epochs; worker-q
2153, 2h29m): HOLDOUT-100 2.368646e-08 → **2.366172e-08 (−0.10%)** — noise. **Closed.** Anatomy 2
says exact var is worth ~47% of each layer's one-step error, but the learned output page moved only
~2 Adam steps' worth in 448 (rms 4e-4, i.e. ~1e-4 relative on var against a ~1e-3 error): the
gradient has no consistent sign, so the variance error is not predictable from the 6 local channels
(μ_pre, σ, α, D3/σ³, ‖W_i‖, centred μ). Prediction (mine −3 to −8%) wrong. No canary.

**Step 3 "test first" — cov_out from the corrected pK1** (diag(cov_out) −= 2δ·pK1 + δ², the
variance made consistent with the corrector's mean shift δ; twin arm `pk1cov`, DEV 100 paired):
**−0.15% ± 0.14% raw, 57/100 wins. KILLED** (Step 3 kill line 2%).

## Anatomy 2 — channel drift against EXACT pre-activation labels (HANDOFF 09-25 Step 1, 2026-09-25)

**Instrument.** `tools/anatomy2.py` on `estimator_k3_v59_corr400.py`, cupy, 20 **FIT** nets (full rows
0–19), worker-q 2132. Labels: keenanpepper `arc-whestbench-p2-full1000-N1e9` (raw moments to 4th order,
dense pairwise, N = 1e9) converted to cumulants by `tools/labels/cumulants.py`; `gate_p` from
`whest-p2-bakev2-bench-supp`. **FIT, not DEV**: the dense labels exist only for the `full` split, which is
disjoint from `mini` (DEV). Checks: layer 0 (exactly Gaussian) reads every normalised κ3/κ4 slice at
4–8e-5 = the 1/√N floor; all 20 names match `data/panels/fit.json`; the patched chain driven by the
tool's own loop is **bit-identical** to `predict`, recording is inert, and substituting every object
with the chain's own values changes nothing (max abs 0.0). Every MSE below is the **cross estimate**
mean((p−gt)(p−kp)) over two independent N = 1e9 label sets — the noise-free MSE.

**Label noise is layer-dependent (correction of AGENTS.md).** MSE(gt, kp)/2, mean over the 20 nets:
6.83e-10 at layer 1, falling monotonically to **7.71e-11 at the scored layer 16** (4.9e-11 – 1.1e-10). The
Sep 24 note applied layer 1's 6.8e-10 to the final layer; at layer 16 the floor is ~0.3% of our raw MSE.

**How to read §3.** *One-step*: from the chain's own state after layer l−1, step layer l with μ_pre
labelled (base) and additionally one object labelled; the layer-l mean reads only the marginals (μ,
var, D3, D4), so the off-diagonal objects are tested by *two-step* (label X and μ at l−1, μ at l, read
the mean at l). *Rolled*: X labelled at every layer 2..P+1, chain free after — "how much of the final
error does X own". The chain's corrector stays on throughout (it is part of v59).

## 1. Object errors (relative Frobenius, mean over nets)

| layer (1-based) | mu | var | C11 | D3 | D21 | dG | D22 | Q31 | cov_out | cov err: diag / trace / traceless share |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.65e-05 | 4.49e-05 | 0.000817 | 0.00923 | 0.0118 | 0.0942 | 0.103 | 0.733 | 0.00099 | 0.00 / 0.000 / 1.00 |
| 3 | 3.38e-05 | 6.85e-05 | 0.000991 | 0.0227 | 0.0322 | 0.108 | 0.112 | 0.744 | 0.00172 | 0.00 / 0.000 / 1.00 |
| 4 | 3.55e-05 | 0.000139 | 0.00172 | 0.0265 | 0.039 | 0.115 | 0.118 | 0.715 | 0.00239 | 0.00 / 0.000 / 1.00 |
| 5 | 4.49e-05 | 0.000217 | 0.00239 | 0.0293 | 0.0439 | 0.126 | 0.129 | 0.709 | 0.00304 | 0.00 / 0.000 / 1.00 |
| 6 | 5.58e-05 | 0.000307 | 0.00304 | 0.0316 | 0.0487 | 0.139 | 0.143 | 0.709 | 0.00369 | 0.00 / 0.000 / 1.00 |
| 7 | 6.6e-05 | 0.000407 | 0.00369 | 0.0365 | 0.0556 | 0.154 | 0.158 | 0.714 | 0.00436 | 0.00 / 0.000 / 1.00 |
| 8 | 7.73e-05 | 0.000512 | 0.00436 | 0.04 | 0.062 | 0.175 | 0.176 | 0.721 | 0.00504 | 0.00 / 0.000 / 1.00 |
| 9 | 8.87e-05 | 0.000629 | 0.00504 | 0.0442 | 0.0686 | 0.192 | 0.194 | 0.73 | 0.00574 | 0.00 / 0.000 / 1.00 |
| 10 | 9.84e-05 | 0.00077 | 0.00575 | 0.0479 | 0.0753 | 0.216 | 0.216 | 0.748 | 0.00647 | 0.00 / 0.000 / 1.00 |
| 11 | 0.000109 | 0.000918 | 0.00645 | 0.0527 | 0.082 | 0.237 | 0.237 | 0.766 | 0.00719 | 0.00 / 0.000 / 1.00 |
| 12 | 0.000118 | 0.00108 | 0.00717 | 0.0565 | 0.0879 | 0.255 | 0.253 | 0.736 | 0.00785 | 0.00 / 0.000 / 1.00 |
| 13 | 0.000126 | 0.00125 | 0.00781 | 0.0601 | 0.0943 | 0.274 | 0.273 | 0.733 | 0.0085 | 0.00 / 0.000 / 1.00 |
| 14 | 0.000134 | 0.00143 | 0.00849 | 0.0637 | 0.101 | 0.302 | 0.294 | 0.74 | 0.00917 | 0.00 / 0.000 / 1.00 |
| 15 | 0.000142 | 0.00164 | 0.00914 | 0.0669 | 0.106 | 0.333 | 0.321 | 0.755 | 0.00975 | 0.00 / 0.000 / 1.00 |
| 16 | 0.000147 | 0.00186 | — | 0.0703 | — | 0.354 | — | — | — | — |

### T0

| layer (1-based) | mu | var | C11 | D3 | D21 | dG | D22 | Q31 | cov_out | cov err: diag / trace / traceless share |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.65e-05 | 4.49e-05 | 0.000817 | 0.00923 | 0.0118 | 0.0942 | 0.103 | 0.733 | 0.00099 | 0.00 / 0.000 / 1.00 |
| 3 | 3.38e-05 | 6.85e-05 | 0.000991 | 0.0227 | 0.0322 | 0.108 | 0.112 | 0.744 | 0.00172 | 0.00 / 0.000 / 1.00 |
| 4 | 3.55e-05 | 0.000139 | 0.00172 | 0.0265 | 0.039 | 0.115 | 0.118 | 0.715 | 0.00239 | 0.00 / 0.000 / 1.00 |
| 5 | 4.49e-05 | 0.000217 | 0.00239 | 0.0293 | 0.0439 | 0.126 | 0.129 | 0.709 | 0.00304 | 0.00 / 0.000 / 1.00 |
| 6 | 5.58e-05 | 0.000307 | 0.00304 | 0.0316 | 0.0487 | 0.139 | 0.143 | 0.709 | 0.00369 | 0.00 / 0.000 / 1.00 |
| 7 | 6.6e-05 | 0.000407 | 0.00369 | 0.0351 | 0.0539 | 0.154 | 0.158 | 0.714 | 0.00431 | 0.00 / 0.000 / 1.00 |
| 8 | 7.68e-05 | 0.000505 | 0.00431 | 0.0385 | 0.0601 | 0.175 | 0.176 | 0.721 | 0.00494 | 0.00 / 0.000 / 1.00 |
| 9 | 8.74e-05 | 0.000619 | 0.00494 | 0.0427 | 0.0668 | 0.192 | 0.194 | 0.73 | 0.00562 | 0.00 / 0.000 / 1.00 |
| 10 | 9.62e-05 | 0.000755 | 0.00562 | 0.0463 | 0.0734 | 0.216 | 0.216 | 0.748 | 0.00632 | 0.00 / 0.000 / 1.00 |
| 11 | 0.000106 | 0.000899 | 0.00631 | 0.0511 | 0.0802 | 0.237 | 0.236 | 0.766 | 0.00702 | 0.00 / 0.000 / 1.00 |
| 12 | 0.000115 | 0.00106 | 0.00701 | 0.055 | 0.0861 | 0.254 | 0.253 | 0.736 | 0.00767 | 0.00 / 0.000 / 1.00 |
| 13 | 0.000122 | 0.00122 | 0.00763 | 0.0584 | 0.0926 | 0.274 | 0.272 | 0.733 | 0.00831 | 0.00 / 0.000 / 1.00 |
| 14 | 0.00013 | 0.00141 | 0.00831 | 0.0624 | 0.099 | 0.301 | 0.294 | 0.74 | 0.00898 | 0.00 / 0.000 / 1.00 |
| 15 | 0.000137 | 0.00161 | 0.00896 | 0.0657 | 0.105 | 0.332 | 0.321 | 0.755 | 0.00956 | 0.00 / 0.000 / 1.00 |
| 16 | 0.000143 | 0.00182 | — | 0.0689 | — | 0.353 | — | — | — | — |

T0 final MSE (x-est): 2.1479e-08

### T0x

| layer (1-based) | mu | var | C11 | D3 | D21 | dG | D22 | Q31 | cov_out | cov err: diag / trace / traceless share |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4.65e-05 | 4.49e-05 | 0.000817 | 0.00855 | 0.011 | 0.0942 | 0.103 | 0.733 | 0.00098 | 0.00 / 0.000 / 1.00 |
| 3 | 3.38e-05 | 6.79e-05 | 0.000982 | 0.0199 | 0.0281 | 0.108 | 0.112 | 0.744 | 0.00162 | 0.00 / 0.000 / 1.00 |
| 4 | 3.48e-05 | 0.000131 | 0.00162 | 0.0245 | 0.0359 | 0.115 | 0.118 | 0.715 | 0.00227 | 0.00 / 0.000 / 1.00 |
| 5 | 4.33e-05 | 0.000206 | 0.00227 | 0.0278 | 0.0415 | 0.126 | 0.129 | 0.709 | 0.00291 | 0.00 / 0.000 / 1.00 |
| 6 | 5.37e-05 | 0.000293 | 0.00292 | 0.0304 | 0.0468 | 0.139 | 0.143 | 0.709 | 0.00356 | 0.00 / 0.000 / 1.00 |
| 7 | 6.38e-05 | 0.000392 | 0.00356 | 0.0341 | 0.0523 | 0.154 | 0.158 | 0.714 | 0.00418 | 0.00 / 0.000 / 1.00 |
| 8 | 7.45e-05 | 0.00049 | 0.00418 | 0.0378 | 0.0587 | 0.175 | 0.176 | 0.721 | 0.00482 | 0.00 / 0.000 / 1.00 |
| 9 | 8.48e-05 | 0.000604 | 0.00482 | 0.042 | 0.0656 | 0.192 | 0.194 | 0.73 | 0.0055 | 0.00 / 0.000 / 1.00 |
| 10 | 9.38e-05 | 0.000739 | 0.00551 | 0.0456 | 0.0724 | 0.216 | 0.215 | 0.748 | 0.00621 | 0.00 / 0.000 / 1.00 |
| 11 | 0.000104 | 0.000883 | 0.00619 | 0.0505 | 0.0792 | 0.237 | 0.236 | 0.766 | 0.00691 | 0.00 / 0.000 / 1.00 |
| 12 | 0.000112 | 0.00104 | 0.0069 | 0.0544 | 0.0852 | 0.254 | 0.253 | 0.736 | 0.00756 | 0.00 / 0.000 / 1.00 |
| 13 | 0.000119 | 0.0012 | 0.00751 | 0.0578 | 0.0918 | 0.274 | 0.272 | 0.733 | 0.0082 | 0.00 / 0.000 / 1.00 |
| 14 | 0.000128 | 0.00139 | 0.0082 | 0.062 | 0.0982 | 0.301 | 0.293 | 0.74 | 0.00887 | 0.00 / 0.000 / 1.00 |
| 15 | 0.000135 | 0.00159 | 0.00886 | 0.0652 | 0.104 | 0.332 | 0.32 | 0.755 | 0.00946 | 0.00 / 0.000 / 1.00 |
| 16 | 0.00014 | 0.0018 | — | 0.0685 | — | 0.353 | — | — | — | — |

T0x final MSE (x-est): 2.0654e-08

## 2. κ4 model check

Fitted on nets 0..9, R² on nets 10..19 (normalised slices, off-diagonal entries; R² is about the eval-set mean, and every fitted model has an intercept unless marked).

κ4_iijj / (σ_i²σ_j²). B = (κ4_iiii + κ4_jjjj)/6 from TRUE marginals = the chain's form (m/6)(dG_i+dG_j) with exact dG:

| layer | mean target | B, coef 1 | λ*·ρ (no intercept): λ*, R² | const + λ*ρ | const + B + λ*ρ | const + 4-mode {ρ, ρ², ḡρ, α_iα_j} | const + B + 4-mode | T (transported diagonal), coef 1 | const + T | const + B + 4-mode + T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3.22e-03 | -0.183 | 0.0001, -95.537 | -0.002 | 0.031 | 0.000 | 0.033 | -6.779 | 0.105 | 0.145 |
| 3 | 5.38e-03 | -0.052 | -0.0001, -78.213 | -0.010 | 0.093 | -0.008 | 0.095 | -49.852 | 0.061 | 0.131 |
| 4 | 7.21e-03 | -0.008 | -0.0000, -69.038 | -0.003 | 0.117 | -0.002 | 0.119 | -57.214 | 0.065 | 0.147 |
| 5 | 8.85e-03 | 0.003 | -0.0001, -58.579 | -0.000 | 0.129 | 0.001 | 0.131 | -52.646 | 0.053 | 0.150 |
| 6 | 1.03e-02 | 0.008 | 0.0000, -48.621 | -0.000 | 0.134 | 0.001 | 0.136 | -45.455 | 0.039 | 0.150 |
| 7 | 1.16e-02 | 0.024 | 0.0001, -40.217 | -0.002 | 0.142 | 0.000 | 0.145 | -38.421 | 0.037 | 0.156 |
| 8 | 1.28e-02 | 0.019 | -0.0001, -33.643 | -0.008 | 0.140 | -0.005 | 0.143 | -32.560 | 0.012 | 0.148 |
| 9 | 1.38e-02 | 0.021 | 0.0000, -28.326 | -0.012 | 0.139 | -0.009 | 0.143 | -27.592 | 0.005 | 0.148 |
| 10 | 1.49e-02 | 0.008 | 0.0000, -24.578 | -0.012 | 0.128 | -0.006 | 0.135 | -24.002 | -0.009 | 0.136 |
| 11 | 1.60e-02 | 0.017 | -0.0001, -21.256 | -0.007 | 0.136 | 0.001 | 0.143 | -20.759 | 0.006 | 0.146 |
| 12 | 1.70e-02 | 0.029 | 0.0000, -18.468 | -0.011 | 0.147 | 0.002 | 0.157 | -18.119 | 0.002 | 0.160 |
| 13 | 1.81e-02 | 0.022 | 0.0001, -16.634 | -0.006 | 0.142 | 0.012 | 0.156 | -16.348 | -0.002 | 0.157 |
| 14 | 1.89e-02 | 0.049 | 0.0002, -14.268 | -0.009 | 0.163 | 0.014 | 0.180 | -14.037 | -0.002 | 0.181 |
| 15 | 2.00e-02 | 0.056 | 0.0001, -12.971 | -0.004 | 0.170 | 0.030 | 0.193 | -12.775 | 0.002 | 0.194 |

κ4_iiij / (σ_i³σ_j). The chain's form is (m/2)·λ·C_ij (no intercept); "fitted coef" is the best (m/2)λ:

| layer | chain form: fitted (m/2)λ, R² | chain (m/2)λ used | λ*·ρ (no intercept) R² | const + 4-mode {ρ, ρ², g_iρ, α_iα_j} | const + C-form + 4-mode | T, coef 1 | const + T | const + C-form + 4-mode + T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.0137, 0.471 | 0.0117 | 0.473 | 0.473 | 0.473 | 0.364 | 0.376 | 0.528 |
| 3 | 0.0147, 0.467 | 0.0116 | 0.468 | 0.470 | 0.470 | 0.086 | 0.112 | 0.498 |
| 4 | 0.0146, 0.495 | 0.0131 | 0.496 | 0.500 | 0.501 | 0.024 | 0.029 | 0.517 |
| 5 | 0.0139, 0.504 | 0.0129 | 0.505 | 0.514 | 0.515 | 0.008 | 0.008 | 0.524 |
| 6 | 0.0132, 0.498 | 0.0135 | 0.501 | 0.515 | 0.516 | 0.003 | 0.003 | 0.522 |
| 7 | 0.0126, 0.492 | 0.0135 | 0.494 | 0.518 | 0.520 | 0.001 | 0.001 | 0.523 |
| 8 | 0.0122, 0.479 | 0.0116 | 0.482 | 0.516 | 0.518 | -0.000 | 0.000 | 0.520 |
| 9 | 0.0116, 0.465 | 0.0109 | 0.470 | 0.514 | 0.516 | -0.000 | 0.000 | 0.518 |
| 10 | 0.0111, 0.462 | 0.0088 | 0.471 | 0.524 | 0.525 | 0.000 | 0.000 | 0.527 |
| 11 | 0.0106, 0.458 | 0.0074 | 0.471 | 0.530 | 0.532 | 0.001 | 0.001 | 0.533 |
| 12 | 0.0102, 0.456 | 0.0092 | 0.472 | 0.538 | 0.541 | 0.000 | 0.001 | 0.541 |
| 13 | 0.0100, 0.458 | 0.0087 | 0.480 | 0.545 | 0.547 | 0.000 | 0.001 | 0.547 |
| 14 | 0.0097, 0.456 | 0.0080 | 0.481 | 0.555 | 0.557 | 0.000 | 0.001 | 0.558 |
| 15 | 0.0093, 0.466 | 0.0067 | 0.501 | 0.576 | 0.578 | 0.000 | 0.001 | 0.578 |

## 3a. One-step (substituted at l, mean read at l)

| layer | chain MSE | base MSE | var rec. | D3 rec. | D4 rec. | marg rec. |
|---|---|---|---|---|---|---|
| 2 | 4.583e-11 | 5.038e-11 | -6% | +3% | +18% | +18% |
| 3 | 3.698e-10 | 3.362e-10 | +17% | +31% | +24% | +71% |
| 4 | 1.021e-09 | 7.589e-10 | +33% | +28% | +22% | +85% |
| 5 | 1.935e-09 | 1.165e-09 | +39% | +25% | +18% | +85% |
| 6 | 3.137e-09 | 1.633e-09 | +43% | +24% | +22% | +91% |
| 7 | 4.489e-09 | 2.007e-09 | +43% | +25% | +16% | +89% |
| 8 | 6.187e-09 | 2.470e-09 | +46% | +24% | +17% | +88% |
| 9 | 7.987e-09 | 2.761e-09 | +46% | +21% | +14% | +83% |
| 10 | 9.861e-09 | 3.256e-09 | +48% | +23% | +16% | +86% |
| 11 | 1.193e-08 | 3.472e-09 | +46% | +22% | +15% | +86% |
| 12 | 1.398e-08 | 3.667e-09 | +46% | +19% | -2% | +75% |
| 13 | 1.630e-08 | 3.928e-09 | +46% | +20% | +10% | +86% |
| 14 | 1.863e-08 | 4.319e-09 | +46% | +20% | +14% | +85% |
| 15 | 2.065e-08 | 4.575e-09 | +48% | +19% | +12% | +85% |
| 16 | 2.294e-08 | 4.823e-09 | +48% | +18% | +10% | +85% |

## 3b. Two-step (substituted at l−1, mean read at l)

| layer | chain MSE | base2 MSE | var rec. | C11 rec. | cov rec. | D3 rec. | D21 rec. | D4 rec. | D22 rec. | Q31 rec. | offdiag rec. | all rec. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 3.698e-10 | 3.361e-10 | +0% | -0% | +0% | +0% | +1% | -1% | +8% | +8% | +17% | +18% |
| 4 | 1.021e-09 | 7.588e-10 | -0% | +4% | +4% | -0% | +15% | -1% | +5% | +9% | +33% | +33% |
| 5 | 1.935e-09 | 1.164e-09 | +0% | +11% | +11% | +0% | +13% | -0% | +3% | +8% | +37% | +37% |
| 6 | 3.137e-09 | 1.632e-09 | +0% | +18% | +18% | -0% | +12% | -1% | +2% | +7% | +39% | +39% |
| 7 | 4.489e-09 | 2.007e-09 | +0% | +21% | +22% | -0% | +10% | -2% | +1% | +5% | +39% | +39% |
| 8 | 6.187e-09 | 2.470e-09 | +0% | +25% | +25% | -0% | +10% | -2% | -0% | +6% | +41% | +41% |
| 9 | 7.987e-09 | 2.760e-09 | +0% | +27% | +27% | -0% | +8% | -2% | -0% | +6% | +42% | +42% |
| 10 | 9.861e-09 | 3.256e-09 | +0% | +30% | +31% | -0% | +8% | -3% | -1% | +6% | +45% | +44% |
| 11 | 1.193e-08 | 3.473e-09 | +0% | +30% | +31% | -0% | +7% | -3% | -1% | +5% | +43% | +42% |
| 12 | 1.398e-08 | 3.667e-09 | -0% | +30% | +31% | -0% | +7% | -7% | -4% | +5% | +39% | +39% |
| 13 | 1.630e-08 | 3.928e-09 | -0% | +31% | +32% | -0% | +6% | -6% | -4% | +5% | +40% | +39% |
| 14 | 1.863e-08 | 4.316e-09 | -1% | +34% | +34% | -0% | +5% | -8% | -5% | +5% | +40% | +39% |
| 15 | 2.065e-08 | 4.575e-09 | -1% | +35% | +36% | +0% | +6% | -8% | -6% | +4% | +41% | +41% |
| 16 | 2.294e-08 | 4.822e-09 | -0% | +35% | +37% | -0% | +6% | -12% | -11% | +3% | +36% | +35% |

## 3c. Rolled

Chain final-layer MSE (x-est, mean of 20 nets): **2.2943e-08**

| substituted at layers 2..P+1 (1-based) | final MSE | share of chain final MSE removed | MSE at layer 9 | at layer 13 |
|---|---:|---:|---:|---:|
| mu @ 2..16 | 4.8176e-09 | +79.0% | 2.760e-09 | 3.924e-09 |
| var @ 2..16 | 1.3360e-08 | +41.8% | 4.735e-09 | 9.353e-09 |
| C11 @ 2..16 | 1.5678e-08 | +31.7% | 6.207e-09 | 1.175e-08 |
| cov @ 2..16 | 1.3209e-08 | +42.4% | 4.709e-09 | 9.275e-09 |
| D3 @ 2..16 | 1.7977e-08 | +21.6% | 5.880e-09 | 1.248e-08 |
| D21 @ 2..16 | 2.0076e-08 | +12.5% | 6.484e-09 | 1.361e-08 |
| k3 @ 2..16 | 1.1366e-08 | +50.5% | 3.860e-09 | 7.843e-09 |
| D4 @ 2..16 | 2.3233e-08 | -1.3% | 6.888e-09 | 1.567e-08 |
| D22 @ 2..16 | 2.4452e-08 | -6.6% | 7.687e-09 | 1.622e-08 |
| Q31 @ 2..16 | 1.9390e-08 | +15.5% | 7.020e-09 | 1.406e-08 |
| k4 @ 2..16 | 1.9404e-08 | +15.4% | 5.677e-09 | 1.298e-08 |
| nomu @ 2..16 | 4.5234e-09 | +80.3% | 1.029e-09 | 2.860e-09 |
| all @ 2..16 | 7.3363e-10 | +96.8% | 4.624e-10 | 5.450e-10 |
| cov @ 2..5 | 2.2419e-08 | +2.3% | 7.255e-09 | 1.558e-08 |
| cov @ 2..9 | 1.9743e-08 | +14.0% | 4.709e-09 | 1.247e-08 |
| cov @ 2..13 | 1.5657e-08 | +31.8% | 4.709e-09 | 9.275e-09 |
| D21 @ 2..5 | 2.2502e-08 | +1.9% | 7.386e-09 | 1.575e-08 |
| D21 @ 2..9 | 2.1364e-08 | +6.9% | 6.484e-09 | 1.441e-08 |
| D21 @ 2..13 | 2.0458e-08 | +10.8% | 6.484e-09 | 1.361e-08 |
| k3 @ 2..5 | 2.2335e-08 | +2.6% | 7.052e-09 | 1.558e-08 |
| k3 @ 2..9 | 2.0357e-08 | +11.3% | 3.860e-09 | 1.282e-08 |
| k3 @ 2..13 | 1.6120e-08 | +29.7% | 3.860e-09 | 7.843e-09 |
| k4 @ 2..5 | 2.2647e-08 | +1.3% | 7.323e-09 | 1.582e-08 |
| k4 @ 2..9 | 2.2052e-08 | +3.9% | 5.677e-09 | 1.452e-08 |
| k4 @ 2..13 | 2.0692e-08 | +9.8% | 5.677e-09 | 1.298e-08 |
| nomu @ 2..5 | 2.1920e-08 | +4.5% | 6.262e-09 | 1.498e-08 |
| nomu @ 2..9 | 1.8067e-08 | +21.3% | 1.029e-09 | 9.834e-09 |
| nomu @ 2..13 | 1.0899e-08 | +52.5% | 1.029e-09 | 2.860e-09 |


### Decision (PLAN §3 Step 3 mapping) and what the tables say

1. **Covariance and κ3 are the two channels; κ4 is minor.** Rolled at every layer: cov −42% of final
   MSE (var alone −42%, C_off −32%), κ3 −51% (D3 −22%, D21 −12.5%), κ4 −15% — and all of κ4's share is
   Q31; exact D4 (−1%) and exact D22 (−7%) make the chain *worse*, because λ/θ/feed were fitted to
   compensate for the model κ4. The error is late: substituting cov only over layers 2..9 recovers
   14% of what 2..16 does 42%; the same shape holds for every channel.
2. **The variance consumed at a layer is the largest single marginal source**: one-step, exact var
   recovers **46–48%** of the per-layer error at every layer ≥ 6, D3 ~20%, D4 ~12%, all four ~85%. Its
   relative error is tiny (0.19% at layer 16) but ∂E[ReLU]/∂σ = φ(α) is the largest sensitivity. The
   off-diagonal covariance, one layer later, recovers 35% (two-step C11) — the largest off-diagonal.
3. **D21 error is birth/closure, not compression.** The unshared T0 chain's D21 error is 99% of v59's
   (0.105 vs 0.106 at layer 15); exact P2 births (T0x) 98%. The old-tier compression owns ~1–2% of the
   D21 object error. (It owns more of the final MSE — T0 2.148e-08, T0x 2.065e-08 vs v59 2.294e-08,
   −6.4% / −10% — through channels other than D21's Frobenius error.)
4. **κ4 models.** κ4_iijj: the chain's form (m/6)(dG_i+dG_j) with *exact* dG has R² ≈ 0.0–0.06; no
   tested family — λρ, ρ², ḡρ, α_iα_j, or the transported post-κ4 diagonal (W∘W)ᵀdiag(κ4(a))(W∘W) —
   explains more than **0.19** of its off-diagonal variance (sd 5e-3 around a mean of 2e-2 at layer 15,
   ≫ the 1e-4 label noise). κ4_iiij: the chain's λ·C form has R² ≈ **0.46–0.50, flat with depth**, and
   its fitted (m/2)λ is within 5–40% of the chain's (e.g. 0.0093 vs 0.0067 at layer 16); the 4-mode
   model adds only +0.00–0.11 (0.47 → 0.58).

**Mapping.** Covariance dominant ⇒ **S4 (σ corrector, `estimator_k3_v62_sig.py`) is in, promoted
ahead of S3**; and the cov_out-from-corrected-pK1 one-line test. κ3 co-dominant, D21 birth-side ⇒
S2's birth gains; harmonic projection-at-fold is **not** implicated (compression ~1% of D21 error).
κ4 ⇒ S3's multi-mode core has at most ~15% × (0.58−0.47)/0.5 ≈ 3% headroom; demoted to last.

### Predictions scored

| claim | strategist | mine | measured |
|---|---|---|---|
| cov share of one-step error | < 15% | var 25–40% | var **46–48%** (one-step); C_off 35% (two-step) — both wrong, strategist further |
| D21 share | 30–50% | largest off-diagonal | **6%** two-step, 12.5% rolled — both wrong |
| κ4 slices share | 20–40% | 10–25% rolled | **15%** rolled (all Q31) — mine right |
| λ·C_off R² on κ4_iijj by L10 | < 0.5 | < 0.3 | ≤ 0 — both right |
| 4-mode R² on κ4_iijj | > 0.8 | 0.5–0.8 | **≤ 0.19** — both wrong |
| λ·C R² on κ4_iiij | — | 0.3–0.7, falling | 0.46–0.50, **flat** — range right, trend wrong |
| rolled cov / κ3 / all-but-μ | — | 30–50 / 30–50 / 70–85% | 42 / **51** / 80% — κ3 just above |
| D21 compression share | unknown | 20–40% | **~1–2%** — wrong |
| object errors at L16: var / C_off / D3 / D21 / D4 / Q31 | — | ≲1 / 3–8 / 10–25 / 25–45 / 30–60 / >50% | 0.19 / **0.9** / **7** / **10.6** / 35 / 75% — the carried objects are 3–4× more accurate than I expected; their *effect* is what is large |

Artifacts: `experiments/2026-09-25_anatomy2/{fit20.json,fit20.md,PREDICTIONS.md,verify.json}`.

## The evaluator (D1, 2026-09-21)

`python tools/ev.py <candidate.py> [--panel dev|fit|holdout] [--n 100]
[--baseline <file>] [--mem] [--append]` produces every column above in one
command. `tools/panels.py` pins the panels; `tools/check_backends.py` reports
what the GPU backend costs in accuracy.

### Panels — HOLDOUT is 900 nets, not 800

PLAN §4 assumed the 100 `mini` nets were a subset of the 1000-net `full` split.
They are not: **the overlap is zero**. So DEV = 100 mini, FIT = the first 100
full nets, HOLDOUT = the remaining **900**. DEV and FIT were verified disjoint
by sha256 of the weight tensors across all 200 nets, not just by seed.

### Two accuracy backends

| backend | s/net | 100 nets | host RSS | agreement with the official 100-net mini run |
|---|---:|---:|---:|---|
| `numpy` (default) | ~50 | **38 min** (4 workers) | 3.8 GiB | **exact** — 100-net mean 3.094966e-08, the official figure to every digit it reports |
| `cupy` on the 5090 | **2.03** | **3.4 min** | 8.0 GiB | panel mean **+3.2e-05**; worst single net 1.9e-03 |

PLAN §4 asked the GPU to match numpy to 1e-6 relative. **It does not, and it
cannot.** Both run the chain in float32, the GPU reduces in a different order,
and sixteen layers amplify that to ~1e-3 on a *per-net* MSE. It is not TF32 —
on a 1024³ f32 product against a float64 reference cupy is *more* accurate than
the CPU BLAS (3.5e-7 vs 7.2e-7) — it is float32 associativity, and nothing
turns it off without changing the candidate.

But the per-net errors are signed and roughly independent, so **the panel mean
is 30× tighter than the tolerance that would matter and 1000× inside the 3%
decision gate**. Measured on C10 over all 100 DEV nets:

    cupy  100-net mean raw MSE   3.095064010e-08     adjusted 6.767843e-09
    official whest run           3.094965995e-08     adjusted 6.767156e-09
    relative                     +3.2e-05

So: **the GPU is sound for panel-level decisions and is what the D2 fit will
run on**; `ev.py` still uses numpy for anything that becomes a RESULTS.md row or
a §5 gate, because a submission is worth 40 minutes. Do not run the two
backends concurrently — 8.0 + 15.2 GiB puts this box into paging.

### Acceptance — met on both backends

C10 on DEV, 100 nets, against the Sep-18 official `whest run`
(`experiments/2026-09-18_c10_diagnostic/c10_mini.json`):

| | raw MSE | adjusted | vs official raw |
|---|---|---|---|
| official `whest run` | 3.094965995e-08 | 6.767156e-09 | — |
| **numpy (instrument of record)** | **3.094966e-08** | 6.767629e-09 | **exact** |
| cupy (screening) | 3.095064010e-08 | 6.767843e-09 | +3.2e-05 |

PLAN D1 asked for ±1%; numpy reproduces the official raw MSE to every digit the
official report carries. The small spread in *adjusted* is not accuracy — it is
the multiplier: ev.py takes FLOPs from the f64 twin, which is 2^25 above what
`whest run` bills, and that is the correct, graded-verified number.

Evidence: `experiments/2026-09-21_d1/c10_dev_numpy.json`,
`experiments/2026-09-21_d1/c10_dev100_cupy.json`.

### Operational notes that cost time

- **A pipe deadlock made the pool look unparallelisable, and it was our bug.**
  `ev.py` gave each worker `stderr=subprocess.PIPE` and drained it only at
  `communicate()`, which does not run until the process exits. The candidate
  emits Python warnings with source context, so a worker that wrote ~64 KiB
  filled the OS pipe buffer and blocked **forever**, at ~0% CPU, a few hundred
  MiB resident, having just entered `predict`. Workers now write to per-worker
  `.out`/`.err` files. The `.err` files are kept: a candidate that starts
  emitting new warnings is telling you something.

  The measurement that settled it — and that no earlier hypothesis was checked
  against — was per-process CPU time: a "slow" worker had accumulated **3 CPU
  seconds in three minutes**. Three CPU-seconds is not slow, it is stopped. A
  control confirmed it: the panel index a pooled worker had sat on for 144 s ran
  standalone in 62 s *while that worker was still stuck*.

  **Three wrong diagnoses preceded this**, each plausible, each acted on, each
  costing a cancelled run: the HF hub's rate limiter, BLAS thread
  oversubscription, and the Windows memory manager trimming working sets. They
  are recorded so the next session does not requeue them.

- **The pool is still bounded by free RAM, not cores** — ~3.8 GiB per worker
  against ~30 GiB genuinely free on a 16-logical / 8-physical-core box. Four is
  the default. That ceiling is real on its own and does not depend on the
  deadlock it was briefly confused with.
- Workers write a per-net progress sidecar, because the driver reads stdout only
  at the end and a slow worker is otherwise indistinguishable from a hung one.
- `HF_HUB_OFFLINE=1` in the worker env: otherwise every worker resolves the
  dataset revision against an unauthenticated, rate-limited endpoint.

## D3 — billing polish: measured at −2.26%, not PLAN's −6–10% (2026-09-22)

**PLAN's D3 target is not reachable from the five items it names.** Every item was
measured under the real meter. Two are infeasible, one is worth ~0.05%, and the
two that work only work *together*. The deliverable is
`estimator_k3_v46_d3.py`: **−2.26% score at flat op count**, which is real but
**below §5's 3% gate**, so it cannot ship alone.

| item | measured | verdict |
|---|---|---|
| 1. Strassen for plain-`@` products | blanket: −2.60% FLOPs, **+30.3% ops**; high-value subset: −1.55% FLOPs, +9.7% ops | only viable once item 2 funds the ops |
| 2. 15-add Winograd instead of 18-add Strassen | −0.69% FLOPs, **−8.2% ops**, +29% RSS | small alone; its op saving is what pays for item 1 |
| 3. pooled preallocated `out=` buffers | **INFEASIBLE** | see below |
| 4. remove recomputed `A_L*M` / `C_L` | ~0.2–0.3% by inspection | marginal, not pursued |
| 5. aliasing / `as_symmetric` | ~0.05% | **closed**, see below |
| **1 + 2 combined** | **−2.26% FLOPs, ops 23,421 → 23,484** | the deliverable |

### Why the op budget, not the FLOP budget, governs this work

Measured by forcing `_STRASSEN_DEPTH`:

| depth | FLOPs | multiplier | ops |
|---|---:|---:|---:|
| 0 (off) | 723,666,642,305 | 0.329085 | **8,724** |
| 3 | 529,917,341,057 | 0.240979 | 20,017 |
| 5 (C10) | 480,850,933,121 | 0.218666 | **23,421** |

**63% of C10's op budget is Strassen glue.** Strassen buys a 33.6% FLOP
reduction with 14,697 op calls. So routing *more* products through it spends the
residual budget fast: the blanket version of item 1 reached 30,528 ops = 0.339 s
projected, which busts both §5's 24,500 ceiling and PLAN D7's 0.33 s finals
bound. `tools/d3_strassen_table.py` prices the trade per shape — break-even is
~61,000 FLOPs saved per extra op call, and every site clears it, so the
constraint is the residual cliff, not efficiency.

### Item 3 is infeasible, and the reason is a hard flopscope property

- `fnp.add(a, b, out=buf)` bills **exactly** what `a + b` bills: 1 op call, same
  FLOPs. `out=` saves nothing.
- Slice assignment raises: *"flopscope arrays are immutable, so item assignment
  (arr[i] = ...) and indexed in-place updates are not supported."*
- `fnp.empty(...)` itself costs an op call.

So there is no way to write into a pooled buffer, and nothing to gain if there
were. PLAN's note that "504aldo ships `out=` and is graded" may be true, but it
buys no FLOPs and no ops here.

### Item 5 is closed: symmetry discounts elementwise ops, not matmuls

| operation | untagged | symmetric-tagged |
|---|---:|---:|
| `X @ Y` (n=256) | 33,488,896 | **33,488,896** |
| `einsum('ij,jk->ik')` | 33,488,896 | **33,488,896** |
| `X + Y` | 65,536 | **32,896** |
| `X * 2` | 65,536 | **32,896** |
| `as_symmetric(X)` itself | — | **458,751** (≈7n², it verifies) |

Matmuls — which are essentially the whole bill — get **no discount at all**.
Elementwise ops get a genuine 2×, but `as_symmetric` costs ~14 elementwise ops
to pay for itself, and n×n elementwise work is a negligible share of 481 G.

### The deliverable, on DEV (100 nets)

    C10   raw 3.095064010e-08   multiplier 0.218666   adjusted 6.767843e-09
    v46   raw 3.094959965e-08   multiplier 0.213724   adjusted 6.614663e-09
          raw change  -0.0034%      score change  -2.2634%

**Accuracy is unharmed.** Winograd perturbs individual activations by up to
4.3e-3 relative — it is algebraically exact (verified to 2.7e-15 in float64) but
numerically noisier than classical Strassen in float32 — yet the panel raw MSE
moves −0.0034% and 51/100 nets improve. That is a coin flip, i.e. pure
rounding-order noise with no systematic direction. It is **not** the ≤1e-6
bit-parity PLAN assumed the item would have, so it is validated by panel
measurement rather than by parity.

Projected public **6.378653e-09** = only **+1.74%** better than the 6.4914e-09
best graded. Below the 3% gate. `estimator_k3_v46_d3.py` is a **base to stack
D2's accuracy gain on**, not a candidate.

**The memory margin is the real price.** WSL 8 GiB gate PASSES, but VmPeak rises
**5,754.8 → 6,801.6 MiB** — 83% of the 8,192 MiB cap against C10's 70%, leaving
1,390 MiB. PLAN D7's finals bound is 7 GiB, so this clears it by 200 MiB.
Winograd is the cause: it holds eight S/T temporaries simultaneously where the
classical form folds its sums into the `concatenate` call.

### Two bases, and the choice is deferred to D2

Neither ships alone, and which one D2 should stack on depends on D2's own
memory footprint (it re-enables P3 feedback), which does not exist yet:

| base | FLOPs | score vs C10 | ops | VmPeak |
|---|---:|---:|---:|---:|
| C10 (unchanged) | 480,850,933,121 | — | 23,421 | 5,754.8 MiB |
| item 1 subset only | 473,416,794,497 | −1.55% | 25,698 | ~5,800 MiB |
| **v46 = items 1 + 2** | 469,983,381,889 | **−2.26%** | 23,484 | 6,801.6 MiB |

The middle row exceeds §5's 24,500-op ceiling, but that ceiling was set *before*
any graded residual existed and PLAN §5 says to revisit it "after the first
graded residual reading from C10". At the worst observed rate 25,698 ops is
0.285 s (1.40× headroom, inside D7's 0.33 s); at C10's own measured rate it is
0.259 s. Raising the ceiling to ~26,000 is defensible on that evidence — but it
is a gate change, so it waits for Samuel rather than being taken silently.

## D4b and K4 — both CLOSED on cost (2026-09-22)

D4 measured accuracy only. This competition scores **MSE × FLOPs**, so the same
arms were priced with ARC's own FLOP counter (`tools/d4b_price_k4.py`) and the
two sides combined. Against factored AUGMENT K3, the arm our chain is closest to:

| arm | n=128 L=8 cost× | MSE× | **score×** | n=256 L=8 cost× | MSE× | **score×** |
|---|---:|---:|---:|---:|---:|---:|
| factored SIMPLE K3 | 0.125 | 2.744 | **0.343** | 0.124 | 2.509 | **0.312** |
| unfactored AUGMENT K3 | 1.361 | 0.843 | 1.147 | 2.359 | 0.924 | 2.180 |
| **factored K4** | **21.19** | 0.294 | **6.222** | **42.15** | 0.154 | **6.483** |

**K4 is closed.** Its 3.4–6.5× accuracy costs 21–42× the FLOPs — a 6.2–6.5×
*worse* score — and the cost ratio doubles from n=128 to n=256, so it degrades
further toward our width of 1024. The accuracy lever D4 found is real and
unaffordable.

**D4b is closed**, now on cost rather than on the shrinking-gain argument alone:
restoring the dropped diagrams is score-negative at both shapes (1.15×, 2.18×
worse) and worsens with width.

**The surprise, and it points at D2.** Factored *SIMPLE* K3 beats factored
AUGMENT on score by ~3× at both shapes: the extra Edgeworth diagrams cost 8×
and buy only 2.5–2.7×. If that transfers, the frontier is not reached by richer
term sets at all — it is reached by a cheap term set with well-fitted
coefficients, which is exactly D2's bet and exactly what the leaders' 0.13 B
cost profile looks like.

*Limits, stated because the numbers are seductive:* torch's counter uses its own
conventions and counts a different program from the one flopscope bills for our
chain, so only ratios between arms under the same counter mean anything; ARC's
implementation is not optimised the way ours is (Strassen, f32); the shapes are
(128,8) and (256,8), not (1024,16); and our chain is not exactly any of these
arms. This is a screening result that closes two lanes, not a prediction of our
score.

## PLAN §5 gate change — op ceiling 24,500 → 26,000, residual now a hard gate

PLAN §5 provides for this explicitly: "ops ≤ 24,500 (tighten or loosen after the
first graded residual)". C10 #331816 supplied that reading — 23,421 ops billed
0.236255 s, 10.087 µs/op, 1.69× headroom, better than the 0.252–0.260 s
projected.

Two changes, made together:

- **Op ceiling 26,000.** At the worst of three graded rates (11.09 µs/op) that is
  0.288 s: 1.39× headroom against the cliff and inside D7's bound. At C10's own
  measured rate it is 0.262 s.
- **Projected residual is now a real gate at ≤ 0.33 s**, not advisory. It is
  gated at PLAN D7's *finals* bound rather than at the 0.4 s cliff, so no canary
  is ever built on margin a designated entry could not keep. This is the binding
  constraint; the op ceiling is the cheap proxy that catches a regression early.

The cliff FAILS the MLP, and a failed net scores ~0.9 against ~6e-9. It is the
one number where being wrong is unrecoverable, which is why the gate sits well
inside it.

## D5-H3 first cut — the old-tier RANK TAPER is worth nothing (2026-09-22)

PLAN D5-H3 proposes a final-layer-aware schedule: "taper old-tier rank and
switch off D21/old readouts over the last 2–4 layers", expecting −5–10% FLOPs.
The taper half is measured and **dead**.

`estimator_k3_v51_h3.py` adds `_SHARED_RANK_TAIL = (h, rank)`: the last `h`
layers compress the old tier at `rank` instead of `_SHARED_RANK`. With it set to
`None` the file is bit-identical to v50 in both FLOPs and ops. Swept on DEV:

| taper | multiplier | ops | raw MSE | score |
|---|---:|---:|---:|---:|
| off (v50) | 0.215285 | 25,728 | 3.011892e-08 | — |
| last 2 layers @ 128 | 0.215285 | 25,728 | 3.011560e-08 | −0.01% |
| last 3 layers @ 128 | 0.215285 | 25,728 | 3.011560e-08 | −0.01% |
| last 4 layers @ 128 | 0.215285 | 25,728 | 3.011560e-08 | −0.01% |

Not a rounding artefact — a structural fact about the implementation, and the
mechanism was **verified by probe rather than inferred**, because the first
explanation written here was imprecise.

The taper *does* fire: instrumenting `_shared_rank` over one net shows 28 calls,
two per layer at layers 1–14, of which **6 take the taper branch and return 64
instead of 256** (layers 12, 13, 14 at h=4). So the constant is reaching the
code.

It is inert for a narrower reason. `_shared_rank` is consulted only at *guard*
sites, each of the form `if _SHARED_ON and _shared_rank(n) and (tw.k - tw.ko) >=
_SHARED_AGE`, and in the tail the second condition is false: **no compression
event occurs in those layers at all.** The old-tier basis `Q` is built once,
earlier, when enough lineages have accumulated; the tail layers only transport
it. A smaller rank requested where nothing is being compressed is never used to
build anything, which is why the multiplier does not move and why h = 2/3/4 and
rank 64 vs 128 are all byte-identical.

**That points at the real lever.** To cut tail cost the existing basis has to be
*re-compressed* smaller on entry to the tail — an explicit rank reduction of `Q`
at layer `L-h`, which then makes every remaining transport cheaper — rather than
a request that silently finds no compression to attach to.

To cut tail cost the basis itself would have to be *re-compressed* smaller, or
the readouts switched off. That is the other half of H3 ("births at layers
≥ L−2 keep only the (3)-diagonal readout"), and it is real surgery on the birth
logic rather than a constant: the last layer already prunes to
`{(1,), (3,), (4,)}` and skips every term with `ip != (1,)`, but layer L−2 still
has to produce the full state that L−1 consumes, so pruning backwards trades
accuracy for cost rather than removing waste outright. Not attempted yet.

## Single-leg kill tests T0–T3 (HANDOFF 2026-09-22, started 2026-09-23 UTC)

Probe file `estimator_k3_p60_probe.py` = v53 verbatim plus knobs `_D21_AGE`
(T1), `_LBLOCK_AGE` (T2), `_RHO` (T3), and two diagnostics (`_KEEP_AGE`: drop
old lineages from both readouts; `_LB_PART`: which part of the L-block T2
drops). Runner `tools/probe.py`: every arm on every net in one persistent cupy
process, so arms are paired by construction. **Probe arms are accuracy-only and
run on cupy** (panel-mean agreement with numpy 3.2e-05); nothing here is a §5
row. Age = `l-1-b` for lineage `b` at layer `l`; the newborn is age 0.

Parity, 2 DEV nets: the probe at v53's settings reproduces v53 to every digit
(2.521407e-08 both); `_LBLOCK_AGE=99`, `_D21_AGE=99` and `_RHO=0` reproduce
their bases exactly.

**Predictions, recorded before the 100-net run.** Strategist: T1 ≤8% raw loss
at a=3; T2 10–40% at a=1 ("I don't know"). Mine, and **made after** a 2-net
smoke test that already showed T2 a=1 at +5,700% and T1 a=3 at +1,500%: T2 a=1
loses >40× raw on DEV 100; T1 a=3 >10×; no a ∈ {0..3} of either gate comes
within 40% of T0; the loss falls roughly geometrically with a (a=10 ≈ +30%).
Diagnostic prediction: the pure two-leg block (`w2b`) carries most of T2's loss,
the rank-32 `Z` residual little.

### T0–T2 on DEV 100 — the single-leg hypothesis is falsified (2026-09-23)

`experiments/2026-09-23_single_leg/t0t2_dev100.json` (per-net MSE, every arm),
cupy, 100/100 nets, all arms paired against T0 on the same nets.

| arm | what is removed for lineages older than `a` | raw MSE | × T0 | ±95% | wins/100 |
|---|---|---:|---:|---:|---:|
| **T0** | nothing: unshared exact chain (`_SHARED_ON=False`) | **2.3642e-08** | 1 | — | — |
| v53 settings | old tier compressed at age 5, rank 256 | 2.5451e-08 | 1.077 | ±0.008 | 1 |
| T1 a=2 | old lineages' (2,1) slice | 6.356e-07 | 26.9 | ±0.8 | 0 |
| T1 a=3 | ″ | 3.675e-07 | 15.5 | ±0.4 | 0 |
| T1 a=4 | ″ | 2.092e-07 | 8.85 | ±0.24 | 0 |
| T2 a=0 | L-block (`A_L`, `eb`, `Z`) → `A_D = M d(sb)` | 1.737e-06 | 73.5 | ±2.5 | 0 |
| **T2 a=1** | ″ (the decisive arm) | **1.138e-06** | **48.1** | ±1.6 | 0 |
| T2 a=2 | ″ | 7.561e-07 | 32.0 | ±0.9 | 0 |
| T2 a=3 | ″ | 5.002e-07 | 21.2 | ±0.6 | 0 |
| T2 a=6 | ″ | 1.363e-07 | 5.77 | ±0.15 | 0 |
| T2 a=10 | ″ | 3.017e-08 | 1.276 | ±0.015 | 0 |
| T1 a=3 ∩ T2 a=1 | both | 1.259e-06 | 53.3 | ±1.7 | 0 |
| diag: K a=1 | the whole lineage, from both readouts | 1.623e-06 | 68.6 | ±2.3 | 0 |
| diag: K a=5 | ″ | 3.122e-07 | 13.2 | ±0.3 | 0 |
| diag: T2 a=1, two-leg only | `w2b` (the `A_L ⊗ 3M ⊗ C_L` block) | 3.304e-07 | 14.0 | ±0.4 | 0 |
| diag: T2 a=1, `Z` only | the rank-32 born-(2,1) residual | 1.516e-07 | 6.41 | ±0.21 | 0 |

**Reading.** Every prediction the strategist wrote is wrong by one to two orders
of magnitude (T1 a=3: predicted ≤8%, measured +1,454%; T2 a=1: predicted 10–40%,
measured +4,714%). My post-smoke prediction (>40× at a=1, geometric decay, a=10
≈ +30%) held. Four facts:

1. **The old D-block alone is nearly worthless.** Keeping `M d(sb)` for old
   sources (T2 a=1, 48×) is barely better than dropping them outright (K a=1,
   69×). The single-leg object the design rests on does not carry the accuracy.
2. **Every part of the L-block matters on its own**: the pure correlation-born
   two-leg block 14×, even the rank-32 `Z` residual 6.4×. None is cheap to drop.
3. **Old (2,1) content matters as much as old (3,) content** (T1 a=3 15.5×).
   Leaving D21 at age ≤3 is not the 60%-of-cost free saving §1 of the response
   assumed.
4. **It is consistent with a number we already had.** v53's own docstring and
   `experiments/submissions.jsonl` record the fully memoryless (regenerating) κ3
   flow, #329896: graded 8.50e-08 at the 0.1 floor, i.e. **raw ≈ 8.5e-07 ≈ 36×
   the exact tower**. "Regenerate the correlation-born κ3 memoryless" was in
   effect measured in September. v53's shared basis, by contrast, costs only
   +7.7%: the old content can be *compressed* cheaply but cannot be *dropped or
   reduced to its diagonal*.

**T3 — fitted regeneration, CLOSED.** `tools/probe_fit.py`, CMA-ES on FIT, base
T2 a=1. The 2-parameter age-exponential form (50 evaluations, 25-net batch)
recovered 0.59% in-sample; the free 13-entry ρ vector (155 evaluations, 10-net
batch) recovered 2.99% in-sample and **0.09% on DEV 100** (T3 1.1370e-06 vs T2
1.1380e-06, still 48.1× T0). Gate was "within 15% of T0". No HOLDOUT read was
spent on a 48× arm. Fitted ρ: `experiments/2026-09-23_single_leg/t3_fit_vec16_best.json`.

**Step 6 decision: the third branch.** T2 loses >40% and T3 does not bring it
under 40%, so the single-leg chain (T5/v60) is not built. T4 (carried κ4 on M)
was specified *on the T3 arm*; on an arm 48× worse than T0 it cannot change this
decision and was not run. The κ4-on-M idea itself is not refuted by anything
here and could be tested on the v53 chain's exact tier later.

### Fallback (response §6): the elasticity table on v53 (2026-09-23)

Accuracy: `tools/probe.py` on v53, DEV 100, cupy, paired vs v53
(`experiments/2026-09-23_single_leg/elastic_dev100.json`, `p2_dev100.json`).
Cost: `tools/variants.py --cost`, the real flopscope f64 read-only twin, one net
(`.../elastic/cost.json`, `.../p2/cost.json`). Projected wall uses the 5-anchor
model + 4.1 s band. Negative Δscore is better.

| arm | Δraw | ±95% | wins | ΔFLOPs | ops | Δscore | proj. wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| v53 | — | — | — | — | 25,140 | — | 96.7 s |
| no hubs (`_HUB_ROUTES=()`) | +5.41% | 0.59 | 4 | −0.57% | 24,312 | +4.81% | |
| closure last layer only | +18.03% | 1.22 | 0 | −1.03% | 21,690 | +16.81% | |
| closure off | +18.14% | 1.23 | 0 | −1.03% | 21,634 | +16.92% | |
| shared rank 224 | +6.09% | 0.66 | 3 | −1.71% | 23,346 | +4.27% | |
| age 4 | +18.33% | 1.26 | 0 | −5.77% | 25,728 | +11.49% | |
| age 6 | −5.53% | 0.67 | 94 | +5.15% | 24,552 | −0.66% | |
| P2 rank 16 | +0.40% | 0.18 | 29 | −3.00% | 24,519 | −2.61% | |
| P2 rank 8 | +0.82% | 0.25 | 25 | −4.56% | 23,990 | −3.78% | 92.4 s |
| **P2 rank 4** | **+0.94%** | 0.26 | 24 | **−5.38%** | **23,645** | **−4.49%** | **91.3 s** |
| P2 rank 2 | +11.17% | 4.60 | 10 | −5.82% | 23,645 | +4.70% | |
| P2 rank 1 | +1,175% | 39.6 | 0 | −6.04% | 23,645 | +1,098% | |
| age 6 + P2 rank 8 | −4.69% | 0.74 | 92 | +0.59% | 23,402 | −4.13% | 92.5 s |
| age 6 + P2 rank 4 | −4.51% | 0.75 | 91 | −0.23% | 23,057 | −4.73% | 91.4 s |
| Strassen depth 6 | 0 | — | — | 0 | 25,140 | 0 | (no-op: `_STRASSEN_MIN_BLOCK=32` caps 1024 at depth 5) |

**Reading.** Of the exact tier's components only one has score elasticity < 1:
the **P2 residual rank**. Everything else (hubs, closure, shared rank, age) pays
for itself. Rank 4 keeps rank 8's per-net tail (worst net +4.2% vs +3.8%);
rank 2 is a cliff (worst net 2.3×). Re-spending the saving on age 6 buys only
0.25% more for ~700 MiB of memory, which D7's 7 GiB bound forbids. **v55 =
v53 + `_P2_RANK = 4`** is the fallback canary: −4.49% score, projected wall
91.3 s (v53: 96.7 s), i.e. it also moves under D7's 95 s bound. The T0–T3
single-leg work is closed; H1/H4-type ideas are "measured, superseded by T1–T3".

## Step 1 (HANDOFF 09-23) — P3 D21-feedback on v55 (started 2026-09-23)

P3 is **off** in v50, v53 and v55 (see the D2-prep correction below), so the
arms are `_P3_ON=True` at `_P3_RANK` 16 and 8 on v55, paired against v55 on
DEV 100 (cupy probe), then one f64-twin cost run per arm and the WSL memory gate
on the shipping arm.

**Predictions, recorded before any arm ran.** Strategist: r16 −4–8% score at
+3–6% FLOPs. Mine: prior evidence is C10 + P3 r32 = −6% raw for +12% FLOPs (a
net *loss* on score), and 504aldo's "r16 lossless vs r32". The P3 thin cost
scales with the number of exact lineages `k`, which age 5 raises, so I expect
r16 at **−4 to −7% raw, +5 to +8% FLOPs, Δscore −2% to +2%**, and r8 at −3 to
−6% raw, +3 to +5% FLOPs, Δscore −3% to +1%. I expect **neither arm to clear
the 3% gate**, r8 being the better of the two.

**Result (DEV 100, cupy, paired vs v55; cost from the f64 twin).**
`experiments/2026-09-24_p3/p3_dev100.json`, `.../arms/cost.json`.

| arm | Δraw | ±95% | wins | FLOPs (F/B) | ΔFLOPs | ops | **Δscore** |
|---|---:|---:|---:|---:|---:|---:|---:|
| v55 (P3 off) | — | — | — | 0.216185 | — | 23,645 | — |
| P3 r16 | −4.57% | 1.48 | 72/100 | 0.230512 | +6.63% | 25,950 | **+1.76%** |
| P3 r8 | −3.28% | 1.48 | 66/100 | 0.224062 | +3.64% | 25,329 | **+0.24%** |

**P3 is CLOSED on this chain.** The accuracy gain is real (72/100 wins at r16)
but smaller than its FLOP bill at both ranks, so both arms are score-*negative*.
No canary. My prediction held (Δscore −2..+2%, neither clears 3%); the
strategist's (−4–8% score) did not — the uncompressed −17% and C10's −6% do not
survive on a chain that already carries age-5 exact content. r16 would also
have sat 50 ops under the 26,000 ceiling. Not re-fitted with a D21 gain: at
best that closes part of a 1.8% gap on a lever that is at best break-even.

## Step 3 (HANDOFF 09-23) — P-N1 sensitivity-weighted old-tier basis

**Predictions, recorded before any N1 run (2026-09-23).** Strategist:
sensitivity-256 within 3% raw of energy-384 (≥15% raw better than energy-256);
Spearman(energy, sensitivity) < 0.7. **Mine:** Spearman between 0.6 and 0.85 on
the first-join layer — the top ~30 energy directions will also be the most
sensitive, the tail will be weakly ordered — so the Spearman kill does not fire.
But the old tier is **re-based at every join** (one lineage per layer from the
age gate on; `_shared_join_lean` rebuilds `Q` from `span{Q, J}` by energy every
time), so a sensitivity choice made at one layer is re-truncated by energy at
the next. I therefore expect a first-join-only sensitivity-256 arm to gain
**< 5% raw** over energy-256 (kill), and an every-join version to gain 5–12%,
short of energy-384's gain on v55 (which I expect at −12 to −18% raw).
Harness: `tools/n1_probe.py` (eigen-rotates the rank-512 basis at the first
join by its carried Gram `Sg`, checks rotation parity, then zeroes one
direction at a time and continues the real chain).

### Result — N1 KILLED on its own criterion (2026-09-23)

Two DEV nets, v55 at `_SHARED_RANK = 512`, first join at layer 5, all 512
eigen-directions zeroed one at a time and the real chain continued (cupy,
0.78 s per tail, rotation parity 5.2e-07). `experiments/2026-09-25_n1/net{0,1}_r512.json`,
summarised by `tools/n1_spectra.py`.

| rank | energy, net 0 | sensitivity ‖Δμ_L‖², net 0 | energy, net 1 | sensitivity, net 1 |
|---:|---:|---:|---:|---:|
| 1 | 4.39e-03 | 1.82e-06 | 5.04e-03 | 2.00e-06 |
| 16 | 2.28e-03 | 2.68e-07 | 2.63e-03 | 3.41e-07 |
| 64 | 8.31e-04 | 9.89e-08 | 9.19e-04 | 9.90e-08 |
| 128 | 2.64e-04 | 2.82e-08 | 2.91e-04 | 3.56e-08 |
| 256 | 3.14e-05 | 3.51e-09 | 3.35e-05 | 5.35e-09 |
| 384 | 3.03e-06 | 6.18e-10 | 3.05e-06 | 8.46e-10 |
| 512 | 3.03e-08 | 2.61e-10 | 3.12e-08 | 3.31e-10 |

| | net 0 | net 1 |
|---|---:|---:|
| **Spearman(energy, sensitivity)** | **0.994** | **0.958** |
| top-256 overlap, energy vs sensitivity order | 249/256 | 239/256 |
| sensitivity captured by energy-256 / sensitivity-256 | 98.83% / 98.87% | 98.32% / 98.48% |

The ceiling for any 256-of-512 selection, measured on DEV 100 (cupy, paired
vs energy-256 = v55, `rank_dev100.json`):

| arm | Δraw | ±95% | wins |
|---|---:|---:|---:|
| energy-384 | −6.56% | 0.71 | 98/100 |
| energy-512 | −6.93% | 0.73 | 99/100 |

**Decision: N1 is dead; the compression lane is closed.** Kill criterion
"Spearman > 0.9" fires on both nets. The energy basis already *is* the
sensitivity basis to within 0.04–0.16 pp of captured sensitivity, and even a
perfect selection is bounded by energy-512's −6.9% — against the ≥15% predicted.
Arms (c)/(d) were not run: they cannot reach the 5% pass bar when the whole
512-dimensional candidate space is worth 6.9% and the reordering captures
~nothing of it. Every prediction was wrong: the strategist's (Spearman < 0.7,
sensitivity-256 ≥15% better) and mine (Spearman 0.6–0.85). Mine that
energy-384 would be −12 to −18% was also wrong: at age 5 the rank above 256 is
worth only 6.6% (v50/age 4: 19.9%), because the extra exact layer already holds
what the higher ranks used to carry. At rank 384's ~+19% FLOPs (D5 table) that
is score-negative, so no rank change ships either.

Reading for the write-up: the old tier's spectrum decays ~10⁵ from rank 1 to
512 in energy and ~10⁴ in sensitivity, *in the same order*. The remaining
transport is not selectively contracting old-tier directions, so observability
and controllability are aligned here and balanced truncation has nothing to add.
The third negative on "old content at zero cost" becomes a fourth.

## Step 4a (HANDOFF 09-23) — per-network adaptive λ

`estimator_k3_v57_adlam.py` = v55 + `λ_{l} ← λ_{l}·clip(r_l/REF_l, 0.5, 2)^β`,
`r_l = mean(dG_l)/mean(var_l)` read at layer l and applied to the λ layer l+1
consumes; own code, mechanism from v29 (provenance entry added). REF from FIT
100 at β=0 (`experiments/2026-09-26_adlam/ref_fit100.json`): r_l falls from
0.0085 (layer 3) to 0.0047 (layer 14); **its coefficient of variation across
the 100 FIT nets is only 1.8–9.0%**, so the clip never binds.

**Prediction, recorded before the DEV run.** With per-net r within ±9% of REF,
the factor is ≈ 1 ± 0.09β, i.e. a per-net λ perturbation far smaller than D2's
fitted profile moved λ (0.5–1.35×) for a 2.7% total. I expect |Δraw| < 1% at
β ∈ {0.5, 1} — no canary. Strategist: 0–8% [?].

**Result (DEV 100, cupy, paired vs v55 = β 0).** `experiments/2026-09-26_adlam/adlam{,_same}_dev100.json`.

| β | lagged: Δraw | ±95% | wins | same-layer (504aldo's form): Δraw | ±95% | wins |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | −0.28% | 0.10 | 71 | −0.25% | 0.10 | 71 |
| **1** | **−0.39%** | 0.18 | 62 | −0.35% | 0.19 | 65 |
| 2 | −0.11% | 0.35 | 53 | −0.06% | 0.37 | 55 |
| −1 (sign check) | +1.01% | 0.30 | 17 | +0.95% | 0.29 | 20 |

**Closed as a lever: −0.4% raw, as predicted.** The effect is real and signed
(β = −1 loses 1%) but per-net r varies too little on this panel for per-net λ to
matter. FLOP cost is two n-reductions/layer (lagged) — ~0 — so β = 1 lagged is
kept in `estimator_k3_v57_adlam.py` as a free component for a later bundle,
not a canary.

## Step 4d (HANDOFF 09-23) — tiered old-tier ranks: CLOSED, score-negative

`estimator_k3_v58_tier.py` = v55 + `_SHARED_RANK_SCHED`. **First, a correction
to the H3 note (D5-H3, 2026-09-22):** instrumenting v55 shows a join at *every*
layer 5–14, so compression events do happen in the tail. The taper was
byte-inert for a different reason: the range finder's sketch is the old `Q`,
and its width (256), not `_SHARED_RANK`, sets the new basis's rank. v58 narrows
the sketch to the top-r columns of `Q·Sg` by norm when the scheduled rank drops.

Prediction (strategist): −3–5% cost at ≤2% loss. Measured, DEV 20 (cupy,
paired vs v55; the 2-net smoke of five arms had already shown +13 to +54%) and
f64-twin cost (`experiments/2026-09-26_tier/`):

| schedule | Δraw | ±95% | wins | ΔFLOPs | ops | **Δscore** |
|---|---:|---:|---:|---:|---:|---:|
| 192 from layer 9 | +12.33% | 1.97 | 0/20 | −4.23% | 22,661 | **+7.6%** |
| 192 from 11, 128 from 13 | +17.90% | 1.79 | 0/20 | −4.88% | 23,080 | **+12.1%** |
| 192 from layer 13 | +2.46% | 0.63 | 0/20 | −1.44% | 23,397 | **+1.0%** |

The cost side landed as predicted; the loss is 6–9× the predicted ceiling.
Consistent with N1 and D5: the old tier's rank is not over-provisioned at any
age — cutting it is always worse than its bill. The compression lane stays closed.

## Step 4e (HANDOFF 09-23) — exact Gaussian bivariate tail in the covariance

`estimator_k3_v59_biv.py` = v55 + `_biv_tail`: the table's Gaussian post-ReLU
covariance is the Hermite series Σ C^k/k! w(k,1)_i w(k,1)_j **truncated at k=2**;
by Price's theorem the k≥3 tail is exactly
`C²/(σᵢσⱼ) ∫₀¹ (1−u)(φ₂(αᵢ,αⱼ;ρu) − φᵢφⱼ) du`, done with 4 Gauss–Legendre nodes.
Checked offline: kept + tail = the exact Price integral to 1e-7 at ρ = 0.3 and
0.5; Monte Carlo agrees within its noise. Motivation (one DEV net, v55):
off-diagonal |ρ| rms rises 0.031 → 0.123 from layer 0 to 14 (max 0.60), and the
dropped tail is 0.06% → 1.0% of the kept Gaussian covariance.

**Prediction, recorded before the DEV run.** −0.5% to −2% raw at ~+0.1% FLOPs;
the correction is Gaussian-only (the κ3/κ4 terms keep their own truncation) and
at most ~1% of the late-layer covariance. Strategist: 0–5% [?].

**Result (DEV 100, cupy, paired vs v55).** 4 nodes −0.23% raw (±0.11, 65/100);
8 nodes −0.22% (±0.11, 65/100) — the quadrature is converged at 4. **Real, and
below my predicted floor. Closed as a lever**; kept as a free bundle component
(~0.1% FLOPs). The Gaussian closure's truncation was not where the error is.

## Strassen depth 6 (leaf 16) — dearer, closed (2026-09-23)

v56c with `_STRASSEN_DEPTH = 6, _STRASSEN_MIN_BLOCK = 16`: F/B **0.22014**,
26,773 ops (v55: 0.21618, 23,645). Transports only at leaf 16: 0.22205. At
16-wide leaves the Winograd glue outruns the multiply saving, so leaf 32 is the
FLOP minimum in our spelling — and leaf 16 would also be the slowest kernel.
`experiments/2026-09-24_depth6/cost.json`.

## Step 2 (HANDOFF 09-23) — wall-safe twin (started 2026-09-23)

`estimator_k3_v56b_wallsafe.py` = v55 + `_STRASSEN_WIDE_MIN_BLOCK = 64`: the two
wide D21 contractions (`rec = [LA|LP][A_L|M]ᵀ`, inner dim `2kn`, and the old-tier
`X_old`) use 64-wide Strassen leaves; everything else keeps depth 5 / leaf 32.
Arms: `parity_v55` (knob `None`, must equal v55's FLOPs and ops exactly),
`wide64` (the twin), `all64` (`_STRASSEN_MIN_BLOCK = 64`, depth 4 everywhere).

**Predictions, recorded before the cost runs.** wide64: FLOPs +3 to +5%, ops
−1,000 to −2,000. all64: FLOPs +12 to +15%, ops −3,000 to −4,000. Accuracy of
both within ±0.5% raw of v55 (Strassen depth is a rounding-order change). On the
refit wall model, only all64 is cleanly classable (leaf 64, age 5: 101 ps/FLOP
inferred) and should project a noisy worst ≤ 80 s; wide64's real rate has to be
read from a canary.

**Result.** Cost: f64 twin (`experiments/2026-09-24_wallsafe/arms/cost.json`);
accuracy: DEV 100 cupy, paired vs v55 (`.../twin_dev100.json`).

| arm | F/B | ΔFLOPs | ops | Δraw | ±95% | wins | proj. median | proj. worst quiet / noisy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v55 (= `parity_v55`, exact) | 0.216185 | — | 23,645 | — | — | — | 92.9 | 98.0 / 115.2 |
| wide64 | 0.222947 | +3.13% | 23,116 | +0.00% | 0.01 | 47 | ? (mixed leaves) | — |
| **all64** | **0.231932** | **+7.28%** | **19,804** | **−0.00%** | 0.01 | 49 | **58.9** | **62.1 / 73.0** |

Both predictions on FLOPs were wrong in the good direction for all64 (+7.3%,
not +12–15%) and wrong in the bad direction for wide64 (ops −529, not −1,000 to
−2,000). wide64 buys too little to be worth a server reading. Accuracy is
untouched, as predicted. all64's projection rests on an **inferred** leaf-64 /
age-5 kernel rate (101 ps/FLOP = the measured 92 ps + the age-5 10%); even at
130 ps it projects a 91 s noisy worst, inside B's 95 s bound.

**all64 is shipped as `estimator_k3_v56b_wallsafe.py`** (`_STRASSEN_MIN_BLOCK =
64`; the `_STRASSEN_WIDE_MIN_BLOCK` knob is then inert). Its score is v55 ×
1.0728, so it **fails §5's 3% score gate by design** — it is the designate-B
candidate, and HANDOFF 09-23 Step 2 asks for one canary to read its real wall.

## Step 1 (HANDOFF 09-24) — all-layer residual anatomy: the error is PROPAGATED, and it is not locally predictable (2026-09-24)

`tools/anatomy.py` on v55, DEV 100, cupy, against the organizer's
`all_layer_means` (every layer, N = 1e9). Predictions were written first
(`experiments/2026-09-24_anatomy/PREDICTIONS.md`). The instrument patches a COPY
of v55 at two anchored points and proves itself inert: with the hook off AND
with hook + one-step on, the patched prediction is **bit-identical** to the
unpatched one on 2 nets (max|diff| = 0.000e+00), so the extra mid-chain `_layer`
call perturbs nothing. Per-net final MSEs reproduce the v55 arm of
`experiments/2026-09-24_wallsafe/mix_dev100.json` exactly.

### 1a. The label noise floor is 6.8e-10, NOT 9e-11 — AGENTS.md is wrong by 7.6x

Layer 1 is the one row v55 does not compute with the chain: it is the exact
closed form `||W[:, j]|| / sqrt(2 pi)`, evaluated in float64. Its MSE is
therefore a **direct measurement of the organizer labels' own Monte-Carlo
noise**, and it reads **6.8815e-10**. The analytic value for N = 1e9 samples is
`Var(ReLU(z))/N = s^2 (1/2 - 1/(2 pi)) / 1e9 = 6.817e-10` at He init
(`||W[:, j]||^2 -> 2`) — agreement to **1%**.

AGENTS.md "Measurement" says "(organizer labels: ~9e-11)". That is too low by
**7.6x** and should be corrected there. It changes no decision we have taken —
at raw 2.46e-8 the floor is 2.8% of our MSE and 4.2% of J2W's 1.64e-8, so nobody
on the board is near it — but every "MSE quoted with its noise floor" from here
on should quote 6.8e-10.

### 1b. MSE by layer, and the one-step residual

`one-step` = one chain step taken from the **labelled** mu_{l-1}, with this
layer's covariance, kappa3 tower and kappa4 core left exactly as the chain has
them. It substitutes the mean channel only, so the cumulants entering that step
have still drifted for l-1 layers: the one-step number is an **upper bound** on
local closure error, which is the conservative direction for the question asked.

| layer | MSE | MSE_l/MSE_{l-1} | one-step MSE | one-step / accumulated |
|---:|---:|---:|---:|---:|
| 1 | 6.8815e-10 | — | — | — (closed form = the noise floor, 1a) |
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
| **16** | **2.5691e-08** | **1.127** | **5.2837e-09** | **0.206** |

Three readings.

1. **Layers 1-2 are at the noise floor** (6.88e-10, 6.69e-10) — the chain's own
   error there is below this instrument's resolution, and the ratio < 1 at
   layer 2 is the floor varying slightly by layer, not an improvement. Real
   error starts at layer 3.
2. **The error is propagated, and increasingly so with depth.** one-step /
   accumulated falls monotonically from 0.570 at layer 4 to **0.206** at
   layer 16. Even taking the one-step figure at face value (it is an upper
   bound), **79% of the final-layer error is drift carried in from earlier
   layers, not error made at the last step.**
3. Growth is superlinear early (1.70x at layer 4) and settles to a geometric
   **1.13-1.23x** over layers 10-16.

**Predictions scored.** Strategist: "one-step < 20% of accumulated at layer 16"
— **0.206, just outside, i.e. right in substance.** Worker: "10-30%, at or below
his 20% bound, near its top end" — **right, and the tighter call.** Worker on
the growth ratio ("declining, geometric 1.15-1.35x by layers 10-16") — right in
shape, slightly high in level (measured 1.13-1.23).

### 1c. The residual does not correlate with ANY local feature — both predictions wrong

Pearson correlation of the residual with each per-neuron feature, pooled over
100 nets x 1024 neurons. Largest |corr| over layers >= 8:

| feature | accumulated | one-step |
|---|---:|---:|
| mu_pre | 0.031 | 0.002 |
| sigma | 0.062 | **0.097** |
| alpha = mu_pre/sigma | 0.031 | 0.002 |
| D3/sigma^3 | 0.034 | 0.002 |
| D4/sigma^4 | 0.051 | 0.082 |
| norm of W[:, i] | 0.023 | 0.022 |
| D21 row energy/sigma^4 | 0.045 | 0.073 |

**Nothing reaches 0.10.** Strategist predicted "> 0.5 with (alpha, D3/sigma^3)
at layers >= 8": measured **0.002 and 0.002** — wrong by two orders of
magnitude. Worker predicted 0.3-0.7 and 0.2-0.6: **also wrong, in the same
direction, just less so.** Both were far too optimistic.

The #330001 result quoted in the response §3 ("0.975 correlation with Edgeworth
features at one step") does **not** reproduce as a per-feature correlation on
this chain at this depth. Whatever it measured, it was not this.

### 1d. The bin concentration is real but has NO consistent sign — no formula fix

Share of squared error in the top-2 of 10 quantile bins (random = 0.20):

| layer | kind | alpha | D3/s^3 | D4/s^4 | norm W |
|---:|---|---:|---:|---:|---:|
| 8 | accumulated | 0.320 | 0.318 | 0.214 | 0.214 |
| 12 | accumulated | 0.357 | 0.355 | 0.217 | 0.217 |
| 15 | accumulated | 0.360 | 0.362 | 0.214 | 0.216 |
| 16 | accumulated | 0.363 | 0.364 | 0.216 | 0.212 |
| 8 | one-step | 0.476 | 0.365 | 0.208 | 0.217 |
| 12 | one-step | **0.576** | 0.428 | 0.217 | 0.210 |
| 15 | one-step | **0.631** | 0.461 | 0.251 | 0.210 |
| 16 | one-step | **0.664** | 0.478 | 0.239 | 0.216 |

The handoff's trigger is ">50% of the squared error **with a consistent sign**
at >= 3 layers". The one-step residual clears the 50% half at layers 12, 15 and
16 on alpha — **and fails the sign half outright.** The concentration sits in
bins 4-5, i.e. **alpha ~ 0, the ReLU kink**, where the residual is simply
largest; the per-bin bias there is 0.1-2.4% of the per-bin RMS:

    layer 16, one-step, binned by alpha
    bin  alpha range        bias        rms     bias/rms   se_share
      3  [-1.80, -0.88]   1.023e-05  8.547e-05     0.120      0.138
      4  [-0.88, -0.01]  -1.288e-07  1.322e-04    -0.001      0.331
      5  [-0.01,  0.87]   3.208e-06  1.327e-04     0.024      0.333
      6  [ 0.87,  1.81]   1.065e-05  8.632e-05     0.123      0.141

The error is **symmetric spread, not offset**. The largest signed effect
anywhere is bias/rms = 0.12 at moderate |alpha|, and it has the SAME (positive)
sign on both sides of zero — an even-in-alpha bias, not the odd-in-alpha
signature a mis-scaled closure term would leave.

**Decision: no formula fix is implicated; Step 1 hands over to Step 2.** Worker
prediction 4 ("top-2 bins 25-45%, below the trigger, hands to Step 2") — right
on the accumulated residual (32-36%) and right on the decision, though it did
not anticipate the one-step residual crossing 50% on concentration alone.

### 1e. The ceiling this puts on a per-neuron corrector

Subtracting the per-bin mean is the best any function of a single feature can
do. That ceiling, as a fraction of squared error removed:

| layer | kind | alpha | D3/s^3 | D4/s^4 | norm W |
|---:|---|---:|---:|---:|---:|
| 8 | accumulated | 0.00211 | 0.00418 | 0.00153 | 0.00043 |
| 16 | accumulated | 0.00129 | 0.00221 | 0.00135 | 0.00056 |
| 15 | one-step | 0.00522 | 0.00596 | 0.00664 | 0.00107 |
| 16 | one-step | 0.00459 | 0.00711 | 0.00513 | 0.00132 |

**0.04% to 0.7%.** A perfect nonparametric corrector on the single best feature
removes at most 0.7% of the one-step residual. Even adding the four axes as if
they were independent reaches ~1.8%, below Step 2's 3% gate and at the edge of
its 1% ridge kill.

This bounds single features, not the ~50-feature ridge Step 2 specifies, and the
in-loop correction compounds (a fraction rho removed from each layer's increment
removes ~rho of the accumulated total, not rho x 0.206). So it does **not** close
the lane by itself. It does mean the lane is priced before it is built: the
ridge's real held-out R^2 on the full Step-2 feature set is measured directly on
FIT 100 first (`tools/anatomy.py --dump`, then `tools/corr_ceiling.py`).


## Step 2 priced BEFORE it is built — the corrector lane is OPEN, and it is the MLP that opens it (2026-09-24)

HANDOFF 09-24 Step 2 is a two-day build (teacher-forced in-loop corrector, ridge
and MLP, trained on FIT + 300 HOLDOUT). Its kill criteria — "ridge < 1% on DEV",
"MLP < 3%" — are, to first order, one number: **the fraction of each layer's
one-step residual variance a function of the per-neuron features can explain.**
That number was measured directly first, in an hour, from
`tools/anatomy.py --dump` on FIT 100. Nothing was fitted on DEV or on HOLDOUT.

Panels: features and targets from **FIT 100 only**; 5-fold cross-validation with
folds **split by NET, not by neuron** (1024 neurons of one MLP share its weights,
so a neuron-level split leaks and inflates R^2). HOLDOUT is untouched.

### 2a. Why R^2 on the one-step residual is the gain, and the propagation model

The corrector runs in the loop, so it shrinks each layer's *increment* to the
accumulated error rather than the accumulated error itself. Writing
`M_l = A_l M_{l-1} + S_l` — accumulated MSE at l = amplified incoming error plus
newly created error — Step 1's one-step MSE **is** `S_l`, so `A_l` is measurable:

| l | M_l (accumulated) | S_l (new) | A_l | weight of S_l at layer 16 |
|---:|---:|---:|---:|---:|
| 5 | 2.6835e-09 | 1.3774e-09 | 0.767 | 0.155 |
| 8 | 7.1776e-09 | 2.7410e-09 | 0.814 | 0.298 |
| 12 | 1.5676e-08 | 4.1370e-09 | 0.870 | 0.584 |
| 16 | 2.5325e-08 | 5.1905e-09 | 0.876 | 1.000 |

**A_l is 0.60-0.88, i.e. < 1 at every layer: a ReLU layer CONTRACTS incoming mean
error.** All of the growth in M_l is newly created error, not amplification. That
is a genuinely useful structural fact and it is new here.

`M_16 = M_1 w_1 + sum_l S_l w_l` with `w_l = prod_{k>l} A_k` reconstructs the
measured 2.5325e-08 **to -0.00%**, so the decomposition is not a fit — it is an
identity, and it converts a per-layer R^2 into a predicted final-layer gain.

### 2b. Ridge: 2.13% — would have FAILED the 3% gate

Per-layer ridge, the handoff's feature expansion (9 raw chain channels — mu_pre,
sigma, alpha, D3/s^3, D4/s^4, ||W[:,i]||, D21 row energy/s^4, the chain's mu_l,
dmu — plus squares and pairwise products of the top 6 = 33 columns), lambda by
inner CV:

| layer | one-step MSE | R^2 held-out | R^2 in-sample | gap |
|---:|---:|---:|---:|---:|
| 2 | 2.5383e-10 | 0.30900 | 0.30932 | 0.0003 |
| 5 | 1.3774e-09 | 0.02675 | 0.02695 | 0.0002 |
| 8 | 2.7410e-09 | 0.01887 | 0.01968 | 0.0008 |
| 12 | 4.1370e-09 | 0.01918 | 0.01968 | 0.0005 |
| 16 | 5.1905e-09 | 0.02229 | 0.02275 | 0.0005 |

Through the propagation model: **predicted raw gain 2.13%.** Below §5's 3% ship
gate, above Step 2's own 1% ridge kill.

**The in-sample/held-out gap is ~0.0005 everywhere.** The ridge is not
data-limited and not overfitting — 102,400 rows for 33 columns. So the ceiling it
reports is a statement about the FEATURES and the model CLASS, not about sample
size, and more training nets (the 300 HOLDOUT nets Step 2 proposes to draw) would
buy the ridge **nothing**. That is worth knowing before spending a HOLDOUT read.

### 2c. MLP: 7.32% — the lane is open

Same rows, same net-wise folds, same 33 features; 2 hidden layers x 32, tanh,
Adam, 2,000 steps, targets standardized per layer (`tools/corr_mlp_ceiling.py`):

| layer | one-step MSE | R^2 held-out ridge | R^2 held-out MLP | ratio |
|---:|---:|---:|---:|---:|
| 2 | 2.5383e-10 | 0.309 | **0.423** | 1.4x |
| 3 | 5.6345e-10 | 0.143 | **0.290** | 2.0x |
| 5 | 1.3774e-09 | 0.027 | **0.119** | 4.4x |
| 8 | 2.7410e-09 | 0.019 | **0.078** | 4.1x |
| 11 | 3.9118e-09 | 0.017 | **0.090** | 5.3x |
| 14 | 4.7635e-09 | 0.024 | **0.076** | 3.2x |
| 16 | 5.1905e-09 | 0.022 | **0.057** | 2.6x |

**Predicted raw gain 7.32%**, and the layers that carry it are the LATE ones
(layer 14 contributes 1.10 pp, 15 contributes 1.24 pp, 16 contributes 1.18 pp;
layers 2-5 together contribute 0.26 pp), because `w_l` discounts early error.

Robustness of the 7.32%:

| variant | predicted raw gain |
|---|---:|
| MLP at the measured R^2 | **7.32%** |
| MLP, only layers >= 5 corrected | 7.17% |
| MLP at HALF the measured R^2 (a compounding haircut) | **3.66%** |
| ridge at the measured R^2 | 2.13% |

**Even halving every R^2 clears the 3% gate.** That is the margin that makes this
worth building.

### 2d. What this number is NOT

It is an **optimistic** reading of the real thing, on four counts, and all four
push the same way:

1. **Teacher-forced, one layer at a time.** Each layer's corrector is scored on
   the residual it sees given the chain's own state. The real in-loop corrector
   compounds its own errors across 16 layers.
2. **The corrector feeds the covariance and kappa3 channels.** Step 2 applies the
   correction to the state, not just the output row, and this measurement says
   nothing about whether that helps or hurts those channels.
3. **Cost is not in it.** It is small but not zero — see 2e.
4. **Two features in the handoff's list are absent** and cannot be measured this
   way: the *transported previous correction* `(W_l c_{l-1})_i` is recursive (it
   does not exist until a corrector does), and the young/old tier shares of D3
   need further chain surgery. Both could only raise the number.

### 2e. Cost estimate for the shipped corrector (not yet metered)

Per layer, per neuron: 33 features -> 32 -> 32 -> 1. FLOPs ~ `1024 x (33x32 +
32x32 + 32) x 2` = 4.3 MFLOP/layer, **69 MFLOP over 16 layers = 0.015% of v55's
475 G** — irrelevant. The feature construction is elementwise or already computed
by the chain, except `D21` row energy at `n^2` = 1.0 MFLOP/layer, also
irrelevant.

**Op calls are the constraint, as always.** ~15-25 flopscope calls per layer for
the matmuls, activations and feature assembly = **240-400 ops** on top of 23,645
(+1.0 to +1.7%), i.e. +0.003 to +0.004 s of residual against 0.146 s of headroom.
Comfortable.

**Weight storage.** Per-layer models are 16 x ~2,144 = ~34k float literals
(~400 KB of source on top of the current 101 KB). The handoff's shared model with
an 8-dim layer embedding is ~2k weights instead — and since 2b shows the models
are not data-limited, sharing costs nothing statistically and is the form to
ship. **Not yet verified: the shared+embedding variant's R^2 has not been
measured** (`--shared` exists in the tool and has not been run).

### 2f. Predictions scored

| | ridge | MLP |
|---|---|---|
| strategist | 3-6% | 8-15% |
| worker (PREDICTIONS.md, Step 1) | lane likely dead; single-axis ceilings 0.04-0.7% | same |
| **measured ceiling** | **2.13%** | **7.32%** |

The strategist was optimistic on both but **close on the MLP**, and his §3 bet —
that the all-layer labels are exact supervision for a channel where the error is
locally observable — is **vindicated for a nonlinear model**, having looked dead
after Step 1's correlations. The worker's Step 1 inference ("the lane is very
likely dead because nothing correlates") was **wrong, and wrong for an
instructive reason**: every *linear* correlation is < 0.10 and every *single-axis*
nonparametric ceiling is < 0.7%, yet a 33-feature MLP reaches R^2 = 0.057-0.119.
The signal is entirely in **interactions between features**, invisible to both
diagnostics Step 1 ran. Single-feature correlations were the wrong instrument for
this question and should not be used to close a learned-model lane again.


### 2g. Which target, and why the choice barely matters

There are two ways to read "the corrector's R^2", and the first draft of this
analysis conflated them:

* **framing A** — the corrector removes a fraction of each layer's NEWLY created
  error `S_l` (target = the one-step residual);
* **framing B** — it removes a fraction of the ACCUMULATED residual
  `label - chain_l` at each layer, which is what HANDOFF Step 2's training
  procedure actually produces, since at inference the chain's input at layer l is
  its own imperfect mean.

**Framing B is the correct description of the shipped thing.** But the two agree,
because the predictable signal is the same signal seen against two denominators.
Comparing the ABSOLUTE variance each explains per layer:

| model | framing A gain | framing B gain | per-layer absolute ratio B/A |
|---|---:|---:|---|
| ridge | 2.13% | 2.22% | 1.01–1.48 |
| **MLP** | **7.32%** | **6.16%** | 0.62–1.01 |

So the end-to-end gain brackets at **6.2%–7.3%**, and the corrector is trained on
framing B's target (the `target` array in the dump) because that is the quantity
that exists at inference. Both clear §5's 3% gate with margin.

Two caveats that only the end-to-end DEV number can settle, and both push down:
a greedy per-layer corrector may re-predict signal an earlier layer already
removed (the additive assumption in framing B), and the correction feeds the
covariance and kappa3 channels, which this measurement does not touch.

### 2h. Capacity is already saturated at hidden = 32 — but DATA is not

Same rows, same folds, hidden 64 instead of 32:

| layer | R^2 h32 | R^2 h64 | delta | h64 train R^2 |
|---:|---:|---:|---:|---:|
| 5 | 0.11861 | 0.11164 | −0.00697 | — |
| 8 | 0.07800 | 0.07017 | −0.00783 | — |
| 11 | 0.09043 | 0.08017 | −0.01026 | — |
| 15 | 0.07117 | 0.06068 | −0.01049 | 0.10649 |
| 16 | 0.05748 | 0.05233 | −0.00516 | 0.07995 |

**h64 is uniformly WORSE, at every one of the 15 layers**, with a wider
train/held-out gap (0.106 vs 0.061 at layer 15). So 32 hidden units is at or past
the capacity limit for 100 training nets, and going bigger is not the lever.

But note what the gap says: the **MLP is data-limited** (train 0.080 vs held-out
0.057 at layer 16) where the **ridge is not** (gap ~0.0005). That is the argument
for HANDOFF Step 2's "FIT 100 + 300 from HOLDOUT": more nets should help the MLP
specifically. The 300 are pinned in `data/panels/holdout_train_split.json` — the
first 300 of the pinned holdout order, chosen that way rather than by a random
seed so the membership is checkable by eye and does not depend on a PRNG version.
**HOLDOUT-600 (indices 400–999) stays untouched and unfitted.**

### 2i. The shipped corrector exists and is inert until fitted

`estimator_k3_v57_corr.py` is GENERATED by `tools/build_v57.py` from v55 — three
anchored insertions, ~36k float32 weights carried as base64 (190 KB, decodes with
`base64` + `array` from the stdlib to max error 0.0, against ~700 KB of decimal
literals that would also round). Weights are rebuilt inside `_constants` every
`predict`, following v55's own rule that per-MLP cost must not depend on MLP
order.

**Parity is asserted, not assumed:** with `_CORR_ON = False` the generated file is
**bit-identical to v55** on real nets (`tools/build_v57.py --verify` →
`max|diff| = 0.000e+00` on 2 DEV nets, whole (16, 1024) array). An all-zero layer
is also exactly inert (`tanh(0) = 0` through both hidden layers, then `+ b3 = 0`),
so a partial fit degrades to the base chain layer by layer rather than
misbehaving.

**Cost, MEASURED on the f64 grader twin** (not projected). Built with ZERO
weights so the corrector's whole arithmetic path runs and returns exactly 0 --
which is also a strong test of the injection: v57 with `_CORR_ON = True` and zero
weights is **bit-identical to v55** (`tools/_parity.py`, max|diff| = 0.000e+00),
so base64 decode, the stacked `_constants` arrays, the feature assembly and the
MLP forward are all live and provably inert.

| | v55 | v57 (corrector live) | delta |
|---|---:|---:|---:|
| FLOPs / MLP | 475,395,428,585 | 475,550,316,409 | **+0.0326%** |
| multiplier | 0.21618481 | 0.21625525 | +0.0326% |
| op calls | 23,645 | **24,186** | +541 (+2.29%) |
| scalar materializations | 0 | **0** | — |
| projected residual @ 11.09 us/op | 0.2622 s | **0.2682 s** | 1.49x headroom |

The projection in the paragraph above (+0.027% FLOPs, ~500 ops) was accurate.
Ops clear the 26,000 gate; the residual headroom equals shipped v53's 1.49x.
**Cost is a non-issue: essentially the entire raw gain flows to the score.**

**So the gate is a single number.** Against the 5.3200e-09 best graded, at
multiplier 0.21625525 and the 1.042 DEV->graded factor, the corrector must deliver

    raw gain >= 3.22% on DEV 100

to clear PLAN §5's "3% better than the best graded". The measured teacher-forced
ceiling is 6.16-7.32%, so there is ~2x of margin for the compounding losses §2d
lists. What it buys if it lands:

| DEV raw gain | projected public | vs best graded |
|---:|---:|---:|
| 3.00% | 5.1719e-09 | −2.78% (fails the gate) |
| **3.22%** | **5.1604e-09** | **−3.00% (the gate)** |
| 5.00% | 5.0652e-09 | −4.79% |
| 6.16% | 5.0034e-09 | −5.95% |
| 7.32% | 4.9415e-09 | −7.11% |



## Step 2 VERDICT — the in-loop corrector WORKS: −5.60% raw on DEV, −4.76% on HOLDOUT-600 (2026-09-24)

`estimator_k3_v57_corr.py`, generated by `tools/build_v57.py` from v55, weights
from `tools/fit_corr_seq.py` on FIT 100 (worker-q 2095, 3h15m).

| panel | raw MSE | Δraw vs v55 | ±95% | wins | status |
|---|---:|---:|---:|---:|---|
| DEV 100 | 2.425178e-08 | **−5.60%** | 0.87 | **88/100** | the decision panel |
| HOLDOUT-600 (100 of 600) | 2.402558e-08 | **−4.76%** | 0.67 | **93/100** | one read, never fitted |
| fresh 10 (datasets 400–409) | — | −3.79% | — | 9/10 | never in ANY fit |

**Both generalisation kills pass.** DEV vs HOLDOUT-600 is **0.84 pp** against a
1 pp kill; DEV vs the fresh 10 is **1.81 pp** against a 2 pp kill (10 nets, so
that figure is noisy by construction). The arms are the SAME module with
`_CORR_ON` false/true, so the control is bit-identical to v55 rather than merely
equivalent to it.

### 2j. The single-pass fit failed and the sequential fit fixed it — the difference is the whole result

| fit | procedure | DEV Δraw | wins |
|---|---|---:|---:|
| `tools/fit_corr.py` | all 16 correctors fitted in ONE pass on the UNCORRECTED chain | **+0.58%** (worse) | 46/100 |
| `tools/fit_corr_seq.py` | sequential: corrector l fitted with 0..l−1 ACTIVE and frozen | **−5.60%** | 88/100 |

Same architecture, same features, same 33 columns, same cost, same panel. The
only difference is the distribution the weights were fitted against, and it is
worth **6.2 percentage points**. That is exposure bias, measured: a corrector
fitted on the uncorrected chain predicts the right MAGNITUDE (corrections are
~1e-4 against a residual RMS of 1.6e-4) in the wrong direction once earlier
layers have moved the state.

**This was the first caveat listed in §2d before any of it was built**, and it is
the one that mattered. The teacher-forced ceiling (6.16–7.32%) was not wrong — the
sequential fit lands at −5.60%, inside it and near the middle — but the naive
route to it returned zero. Anyone reading the ceiling as "the corrector is worth
6%" would have been right about the number and wrong about the engineering.

The sequential fit's own diagnostic shows the mechanism directly. Its MSE column
is the residual at layer l **with correctors 0..l−1 already active**:

| layer | uncorrected (FIT) | with 0..l−1 corrected | |
|---:|---:|---:|---:|
| 8 | 7.1776e-09 | 6.5962e-09 | −8.1% |
| 13 | 1.8029e-08 | 1.6567e-08 | −8.1% |
| 16 | 2.5325e-08 | **2.3472e-08** | **−7.3%** |

i.e. the corrections genuinely propagate, which is the entire premise of applying
them to the state rather than to the output row. Per-layer held-out R² runs 0.166
(layer 2) down to 0.0095 (layer 16); only layer 1 was zeroed (R² −0.00003), and
zeroing is exactly inert by construction.

### 2k. Score, and the §5 gate table

Cost is unchanged from the zero-weight build to the fitted one — **identical to
every digit** (475,550,316,409 FLOPs, 24,186 ops, 0 materializations), which
confirms the corrector's cost is value-independent.

    DEV adjusted   5.244575e-09
    projected      5.033181e-09   =  5.39% better than the 5.3200e-09 best graded

| §5 gate | threshold | v57 | verdict |
|---|---|---|---|
| DEV projected vs best graded | ≥3% better than 5.3200e-09 | **5.0332e-09 = 5.39% better** | **PASS** |
| `whest validate` | passes | 143 ms, setup 0.00 s | PASS |
| WSL `RLIMIT_AS` 8 GiB | PASS | 23/23, VmPeak 6,433.6 MB (1.27×) | PASS |
| projected server wall | < 100 s (quiet) | **97.9 s** (median 92.8) | PASS, thin |
| op calls | ≤ 26,000 | 24,186 | PASS |
| residual | < 0.4 s hard | 0.2682 s (1.49× headroom) | PASS |
| setup | < 2 s | 0.00 s | PASS |
| no bare `except` returning a constant | — | none added by the corrector | PASS |

**Not designable, for the same reason v55 is not:** the projected noisy-day worst
is **115.1 s**, 5 s from the 120 s cliff. The wall-safe route is the one v56b
already proved — v57 at Strassen depth 4 — which would cost +7.28% of multiplier
and land ≈5.40e-09 at ≈68 s. That is the natural follow-up and it would be a new
**A** at a better score than tr64's 5.4341e-09.

### 2l. One dispatch bug the WSL gate caught, and it is why that check exists

The first fitted build failed the memory gate on **one** of 23 checks — "the
candidate child wrote no stderr" — because the corrector emitted a NEW warning on
every net:

    UserWarning: np.divide(FlopscopeArray) auto-routed to fnp.divide;
                 call fnp.divide directly to avoid this warning.
      inv = dtype(1.0) / sigma

With the scalar on the left, numpy's own `__truediv__` wins the dispatch and
flopscope auto-routes — **the operation was nearly billed by the wrong meter**,
which is the rule this file's own `dG` block already records for products ("the
array goes on the LEFT of every scalar product"). Fixed to `fnp.divide(dtype(1.0),
sigma)`, which is the operation the auto-route performed, called directly:
FLOPs, ops and materializations are **identical to every digit**, and the
per-net predictions over 20 DEV nets are **bit-identical** (max diff 0.000e+00),
so no result above needed re-measuring.

RESULTS.md's own note on the `.err` files — "a candidate that starts emitting new
warnings is telling you something" — earned its keep here. A pure-accuracy screen
would never have surfaced it.


## Step 3 (HANDOFF 09-24) — per-lineage adaptive age: a FUZZY FIXED AGE, not an allocation (2026-09-24)

`estimator_k3_v60_adage.py`, generated by `tools/build_v60.py`: v55 with the
fixed `_SHARED_AGE = 5` replaced by an energy tolerance — fold the oldest exact
lineage into `Q` when its carried Gram energy falls below `tau` x the youngest
exact lineage's, floor 3, ceiling 6. Bit-identical to v55 at `_ADAPT_TAU = None`
(`tools/_parity.py`, max|diff| = 0.000e+00 on 2 DEV nets).

Predictions recorded first (`experiments/2026-09-24_adage/PREDICTIONS.md`):
strategist −2 to −5% score; worker **−0% to −2%, i.e. expected to fail the 2%
kill**, because D5 measured age 5 → 6 at only a further −0.66% of score, so the
curvature an adaptive schedule could exploit is small.

### 3a. The parity gate caught a silent failure, and it is worth recording

The first build **never folded at all**, and every arm would have reported
age-6 behaviour under an "adaptive" label. `_wick_tower` constructs a NEW
`_Tower` between `_shared_contract_lean` (which sets the fold decision) and
`_shared_join_lean` (which reads it), and carries `_wts` across by hand; the new
`_dojoin` field had to travel the same way or the join saw `None` and declined
forever. Bit-parity at `_ADAPT_TAU = None` is what exposed it, in one run, before
any measurement was taken. A sweep without that gate would have produced a full
table of plausible, meaningless numbers.

### 3b. tau was calibrated before it was swept

`tools/adage_calib.py` recorded `e_old / e_new` at every layer where the gate is
live, with the tolerance OFF, so the ratios are the ones v55 itself sees:

| exact lineages carried | median | p5 | p95 |
|---:|---:|---:|---:|
| 3 | 0.648 | 0.606 | 0.657 |
| 4 | 0.442 | 0.423 | 0.459 |
| **5 (the operating point)** | **0.380** | **0.276** | **0.524** |

Energy decays by a factor of ~0.38 per age step, and **the spread at a given age
is narrow** — a factor of 1.9 from p5 to p95, against a factor of 1.7 between
consecutive ages. The two distributions barely separate. That predicts, before
any arm runs, that a `tau` threshold does not allocate age per net: it moves
almost every net across the same boundary at once.

### 3c. The schedules confirm it — the adaptive arms are fixed ages in disguise

`tools/adage_schedule.py` records the fold schedule (integers only, no FLOP
claim) for 20 DEV nets per arm:

| arm | distinct schedules / 20 | folds min | mean | max |
|---|---:|---:|---:|---:|
| v55 (fixed age 5) | **1** | 10 | 10.00 | 10 |
| tau 0.28 | **1** | 9 | 9.00 | 9 |
| tau 0.34 | 5 | 9 | 9.00 | 9 |
| tau 0.40 | 7 | 9 | 9.05 | 10 |
| tau 0.48 | 13 | 9 | 9.55 | 10 |
| tau 0.58 | 6 | 10 | 10.35 | 11 |

**`tau = 0.28` produces ONE schedule across every net — it is fixed age 6 wearing
a tolerance.** v55 likewise produces one, which independently re-confirms the
property every graded row reports ("FLOPs spread 0 across all 100 nets"). The
arms that do vary per net (0.40, 0.48) vary by at most one fold in sixteen
layers, and they are the arms whose accuracy is worse.

### 3d. Accuracy: the tau axis retraces the fixed-age curve

DEV 100, cupy, paired against v55 on common random numbers:

| arm | raw MSE | Δraw vs v55 | ±95% | wins/100 | folds |
|---|---:|---:|---:|---:|---:|
| v55 (age 5) | 2.569077e-08 | — | — | — | 10 |
| **tau 0.28** | 2.430295e-08 | **−5.40%** | 0.67 | 94 | 9 |
| tau 0.34 | 2.462154e-08 | −4.16% | 0.67 | 92 | 9 |
| tau 0.40 | 2.524226e-08 | −1.75% | 0.33 | 88 | 9.05 |
| tau 0.48 | 2.790240e-08 | +8.61% | 1.36 | 11 | 9.55 |
| tau 0.58 | 2.999709e-08 | +16.76% | 1.29 | 0 | 10.35 |

tau 0.28's **−5.40% raw is fixed age 6's −5.53%** (derived from D5: age 6 vs
age 5 = 0.7983/0.845 − 1). The same lever, reached a different way. And the
ordering is monotone in folds, not in adaptivity: **every arm that improves raw
does so by folding LESS often**, which is what age 6 already does.

### 3e. Verdict — KILLED at −0.72%, against a 2% kill

Per-net cost on the f64 grader twin, 5 DEV nets per arm (`tools/cost_sweep.py`,
worker-q 2090). v55 reads **475,395,428,585 FLOPs with 0.00% spread** — exactly
its known value, which is the control passing:

| arm | FLOPs mean | spread | multiplier | Δraw | Δcost | **Δscore** | projected public |
|---|---:|---:|---:|---:|---:|---:|---:|
| v55 (age 5) | 475,395,428,585 | **0.00%** | 0.21618 | — | — | — | 5.3301e-09 |
| tau 0.28 | 501,295,522,365 | **0.00%** | 0.22796 | −5.40% | +5.45% | **−0.25%** | 5.3169e-09 |
| tau 0.34 | 495,499,113,431 | 3.34% | 0.22533 | −4.16% | +4.23% | **−0.11%** | 5.3243e-09 |
| tau 0.40 | 480,349,166,551 | 2.13% | 0.21844 | −1.75% | +1.04% | **−0.72%** | 5.2916e-09 |

**Best arm −0.72%. The kill is < 2%. Step 3 is CLOSED.**

Two independent confirmations inside the table. `tau 0.28`'s FLOP spread is
**0.00%** — it is a fixed schedule, i.e. fixed age 6, as §3c said — and its
+5.45% cost against −5.40% raw reproduces D5's age-6 measurement (+5.14% for
−5.53%) from a completely different mechanism. And the arms that DO vary per net
reach only **2.13–3.34%** of spread, against the ±6% (a 12% range) the leaders'
telemetry shows.

**Predictions scored.** Strategist: −2 to −5% score. Worker: **−0% to −2%, "I
expect this to FAIL its 2% kill"**, with the specific sub-predictions that per-net
FLOP spread would be **< 4%** (measured 0.00–3.34%) and that any gain would come
from the low-age end rather than smarter allocation (it did: every arm that
improved raw folded less often). The worker's reasoning — that D5's age curve is
nearly flat past 5, so the curvature available to an adaptive schedule is small —
is what the numbers show. That is now three slices running where the strategist's
predictions were optimistic and the worker's were right; the priors should be set
accordingly.

**And the materialisation question never arose.** It was to be bought with a
canary only if an arm cleared 2%. None did, so the standing zero-materialisations
invariant is untouched and no slot was spent on it.

**What this closes, and what it leaves open.** Response §4 named adaptive age as
"the only candidate that survives the evidence" for the leaders' per-net cost
variation. It is not that mechanism in this chain — the lineage energies are too
tightly distributed for a tolerance to allocate age per net. The leaders' ±6%
remains **unexplained**, and this is the second hypothesis for it to fail (after
per-net λ, whose driving ratio varies only 1.8–9.0% CV). Both failures have the
same shape: *this chain's per-net statistics are too uniform to key an adaptive
decision on.* If a third hypothesis is wanted, it should look for variation that
is NOT built from the weight energies — e.g. an iterative solve with a
convergence test, whose iteration count varies for reasons of conditioning rather
than scale.


## D5 — PLAN's premise is BACKWARDS: the old tier is under-funded (2026-09-22)

PLAN D5 calls the old tier "the cost bet" and proposes cheapening it — H1
replaces it with a carried aggregate, H2 refits under "aggressive truncation",
H3 tapers its rank in the tail. **Every one of those directions is measured
wrong.** The chain is not over-spending on the old tier; it is compressing too
early and too hard.

Cost and accuracy on 100 DEV nets, against v50 (rank 256, age 4):

| setting | cost | raw MSE | **score** | projected public |
|---|---:|---:|---:|---:|
| rank 64 | −23.21% | +545.6% | +395.8% | 3.1000e-08 |
| rank 128 | −15.64% | +170.4% | +128.1% | 1.4262e-08 |
| rank 192 | −6.81% | +43.9% | +34.1% | 8.3849e-09 |
| **v50: rank 256, age 4** | — | — | — | 6.2528e-09 |
| rank 320 | +9.34% | −14.89% | **−6.95%** | 5.8185e-09 |
| rank 384 | +19.37% | −19.94% | −4.42% | 5.9762e-09 |
| **age 5** | +6.13% | −15.50% | **−10.32%** | 5.6075e-09 |
| **age 6** | +11.59% | −20.17% | **−10.91%** | **5.5704e-09** |

Two things follow, and both are larger than anything D2 or D3 produced.

**Truncation is dead, and not recoverable by fitting.** Even the mildest cut —
rank 192, saving 6.8% of cost — costs 43.9% of raw MSE and is 34% worse on
score. PLAN H2 proposes rerunning D2's fit on exactly this configuration; D2's
entire lever was 2.7%, against a 44% loss. H1's "carried aggregate" would have
to reproduce content worth 44%+ of raw MSE at 2 units/layer instead of 7, which
the rank curve says is not a small ask but a very large one.

**Spending more wins.** Raising the compression age from 4 to 6 — keeping
sources exact two layers longer — costs 11.6% more FLOPs and returns 20.2% of
raw MSE, a net **−10.91% on score**. Age is a better axis than rank: at
comparable accuracy (≈−20% raw) age 6 costs 11.6% where rank 384 costs 19.4%.

This is the largest single lever found so far. For context, D2's fitted
coefficients were worth −2.7% and D3's billing polish −1.5%, and each took a day.

## D6 — more fitting adds NOTHING on this chain (2026-09-22)

PLAN D6 asks for "a second fitting round on the final structure with more
parameters (per-term-class × layer, ~130) and per-net adaptation". The first
step of that — refitting the existing profiles on v53's structure, plus a newly
wired **old-tier readout gain** (the one profile never fitted, and the one D5
showed carries the chain's accuracy) — returns **0.00%**.

    identity (all profiles 1.0)      2.628931e-08
    D2's shipped coefficients        2.509809e-08    -4.531%
    best of 302 further evaluations  2.509809e-08     0.000%

The search was genuinely exploring, not mis-scaled: of 302 evaluations, 9 landed
within +1% of the incumbent and the closest non-baseline point was +0.196%. None
was better. D2's coefficients sit at a local optimum for this structure, and 16
parameters do no better than 12.

**A first run of this reported the same 0.00% for the wrong reason** and nearly
went on the record as a result about the old-tier gain. `tools/fit.py` seeded
CMA-ES at identity, discarding whatever coefficients the candidate already
carried, so it restarted 4.5% behind and never recovered the point D2 had
already found. Fixed: the search now seeds from the shipped coefficients and
prints both references. The second run above is the honest one.

**What this means, taken with D5.** On this chain, *structure* beats *fitting* by
a wide margin: one integer (compression age 4 → 5) was worth −10.32% on score,
while the entire fitted-coefficient apparatus is worth −2.7% and will not go
further. PLAN's ordering — fitting as "the main accuracy bet", structure as a
cost afterthought — is exactly inverted.

## D2 prep — the six profiles already exist as constants (read 2026-09-21)

PLAN §6 D2(a) asks for six per-layer profiles parametrised as 4-knot splines,
"defaults = current constants exactly". They do not need to be invented: five of
the six are already module-level constants in `estimator_k3_v43_complete_c10.py`,
so the refactor is *read these from an embedded `coeffs.json`*, not a rewrite,
and bit-parity at defaults is close to free.

| PLAN's profile | where it lives in C10 | current form |
|---|---|---|
| λ (κ4 regen) | `_K4_LAM` (line 130) via `_own_lambda()` (1487) | **already per-layer**: a 16-entry table, times `_OWN_S_LAM = 1.225`, times a linear depth tilt `(1 + _OWN_T_LAM·phase)`, `_OWN_T_LAM = -0.1` |
| closure scale θ | `_OWN_THETA_FAM = 0.625` (122), applied at 432 | global scalar |
| κ4→κ3 feed gain | `_OWN_S_FEED = 1.0` (125), applied at 1771–1773 | global scalar |
| D21-feedback gain | `_P3_ON = False`, `_P3_RANK = 32` (117–118) | **off**; D2's base turns it on at rank 16 — **CORRECTION 2026-09-23: it never did.** Every file from C10 through v47–v55 ships `_P3_ON = False`, `_P3_RANK = 32` (grep). HANDOFF 09-21 *planned* P3 r16 on the D2 base; it was not turned on. v50, v53, v55 all have **P3 off**. |
| (2,1)-consumer gain | `_P2_SEPARABLE = True` (79) | boolean, no gain exposed yet |
| old-tier readout gain | `_SHARED_*` (89–95), age 4 / rank 256 | no gain exposed yet |

So four of the six are a scalar or a table to generalise, one is a switch to
turn on, and two ((2,1)-consumer, old-tier readout) need a gain introduced
before they can be fitted. `_own_lambda`'s existing "table × scale × linear
tilt" is already a crude 3-parameter profile — the spline generalises it rather
than replacing a constant, which is a useful sanity check that the
parametrisation is in the right place.

Note `_K4_LAM[15] = 0.0`: the last layer's κ4 channel is already off, and
`_K4_LAM` is documented as "an INITIALISER measured on our own fresh design
networks, not a fitted value". Fitting it on FIT's 100 noise-free organizer nets
is exactly what D2 is for.

## Open items

### C10 canary — PACKAGED AND GATED, NOT YET SUBMITTED (2026-09-21)

Package `experiments/2026-09-21_c10_canary/c10-canary.tar.gz`
(sha256 `1d476ab2d3527eae4ec4729ce012f8e2399bdee7c9d54ccbdc751cca24d3ec00`;
inner `estimator.py` = `fd8d497d…6152d1bf`, byte-identical to
`estimator_k3_v43_complete_c10.py`; LICENSE = the ARC MIT notice shipped with
#330950 and #331317). `whest validate-package` passes.

PLAN §5 gate readings, all green:

| gate | threshold | C10 | verdict |
|---|---|---|---|
| DEV projected vs best graded | ≥3% better than 8.7520e-09 | 6.526e-09 = **25.4% better** | PASS |
| `whest validate` | passes | passes, 96 ms | PASS |
| WSL `linux_memory_gate.py`, RLIMIT_AS 8 GiB | PASS | PASS, VmPeak 5,754.8 MiB, peak RSS 3,890.2 MiB | PASS |
| projected server wall | < 100 s | 60.259 × 1.3097 = **78.9 s** | PASS |
| op calls | ≤ 24,500 | **23,405** | PASS |
| setup | < 2 s | 0.00 s | PASS |
| no bare `except` returning a constant | — | one `except Exception` in `setup`, best-effort, returns nothing; `predict` has no broad handler | PASS |

Advisory (not a §5 gate): projected server residual = 23,405 ops × 10.8–11.2 µs
= **0.252–0.262 s** against the 0.4 s cliff, i.e. 1.53–1.59× headroom. Thinner
than v31's 2.85× and F21's 2.56×. Measuring it is half the point of this canary
(PLAN §6 D0: "the first server residual measurement at 23k ops").

**Blocked:** the submit call was refused by the agent session's own permission
classifier, not by any project gate. Samuel runs it, or grants the permission:

```powershell
.\.venv\Scripts\whest.exe submit experiments\2026-09-21_c10_canary\c10-canary.tar.gz `
  --description "C10 canary" --watch --format plain
```

Slot ledger at the time of packaging: 0/10 used on 2026-09-21 UTC.
