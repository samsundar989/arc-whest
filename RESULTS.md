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
| **wall** | worst per-MLP wall seconds. Local rows are marked `(local)` and come from the one-net grader twin; the server figure is `local × 1.5981` (§ Calibration) |
| **RSS** | VmPeak under the WSL `RLIMIT_AS` 8,192 MB gate |
| **projected** | predicted public score = DEV score ÷ 1.037 |
| **graded** | the AIcrowd adjusted score actually returned |
| **paired gain** | paired arithmetic gain vs the stated baseline, common random numbers |
| **wins/N** | nets on which the candidate beat that baseline |

## Calibration constants (recheck after every graded row)

All four are implemented in `tools/ev.py`; change them there and here together.

- **DEV → graded**: divide the 100-net DEV adjusted score by **1.042**
  (raised from 1.037 on 2026-09-22). Six observations: **1.0386 / 1.0385 /
  1.0339 / 1.0426 / 1.0471 / 1.0458**, mean 1.0411, last three averaging 1.0452.
  At 1.037 the last three graded rows all came in *better* than projected — by
  0.53%, 0.97% and 0.84% — which is a consistent bias, not noise. 1.042 sits
  under the recent mean, so projections stay slightly pessimistic and the gate
  slightly strict. The constant is deliberately left at the *low* end of that
  range: projected = DEV/1.037 comes out slightly pessimistic, which makes §5's
  "≥3% better than the best graded" gate harder rather than easier. C10 #331816
  landed 0.53% *better* than projected. If a graded row ever deviates by >2%,
  stop submitting and recalibrate before the next slot (PLAN §5).

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
<!-- ev.py --append inserts new rows directly above this marker. Keep it here. -->

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
