# DESIGNATION — current A and B (maintained by the worker; designating is Samuel's)

*Policy (STRATEGIST_RESPONSE_2026-09-23 §5, pending Samuel's acceptance):*
**A** = the best score whose worst wall is ≤ 105 s on a **noisy-day** run.
**B** = a wall-safe twin whose worst wall is ≤ 95 s on a noisy-day run.
Re-run each designate on one quiet and one noisy day before Oct 15. Never spend
B's wall margin for score. Agents never designate; this file only keeps the
evidence current.

"Noisy day" means a run whose worst/median ratio is ≥ 1.15 (Sep 23 after ~08:00
UTC: 1.197–1.235). Quiet runs sit at 1.03–1.055. See RESULTS.md, wall refit.

## Standing — 2026-09-24 02:30 UTC

**A is now `tr64`, measured.** Submission #332088 read a QUIET-day wall of 80.30 s
worst / 76.90 s median (worst/median 1.044), 0 nets > 100 s, 0 failed. Implied
noisy-day worst = 76.90 x 1.24 = **95.4 s**, which clears A's 105 s bound by ~10 s
and misses B's 95 s bound by 0.4 s. Its raw MSE is v55's to six digits, so the
only cost of choosing it over the (undesignable) v55 is the +2.14% multiplier.

| role | file | sha8 | graded rows | score | worst wall per run (median) | residual worst | VmPeak | meets policy? |
|---|---|---|---|---:|---|---:|---:|---|
| **A (recommended)** | `experiments/2026-09-24_wallsafe/mix/estimator_k3_v56c_mix__tr64.py` | `52d83de7` | #332088 | **5.4341e-09** | **80.30 (76.90)** quiet, ratio 1.044 | 0.24128 s | 6,468 MiB | **yes, A** by ~10 s on the implied noisy day; needs one genuine noisy-day re-run |
| **B (recommended)** | `estimator_k3_v56b_wallsafe.py` | `b401ad75` | #332063 | 5.7076e-09 | 60.7 (54.5), ratio 1.114 | 0.2188 s | 4,908 MiB | **yes, both** by 44 s / 34 s; needs a quiet and a noisy re-run |
| dominated | `estimator_k3_v50_safe.py` | `f9cb9439` | #331863, #331939, #331989 | 6.192e-09 | 88.7 (84.4) q, 87.7 (85.0) q, 104.1 (87.0) noisy | 0.2655 s | 5,785 MiB | yes, but **dominated by tr64 and v56b** |
| best score, NOT designable | `estimator_k3_v55_p2r4.py` | `aac641b4` | #331967, #331980 | **5.320e-09** | 114.5 (92.8) noisy, 113.8 (92.5) noisy | 0.2805 s | 6,471 MiB | **no** (6 s from the 120 s cliff) |

**The designation gap is now 2.1% of score**, not 7.3%: v55 5.3200e-09 (undesignable)
vs tr64 5.4341e-09 (A). Choosing A costs 2.1% and buys ~19 s of margin to the cliff.

The mixed-leaf projections below are now superseded for `tr64` by its own reading;
`tr64_wide64` is still a projection and there is no reason to spend a slot on it,
since tr64 already clears A.

| arm | dFLOPs vs v55 | score | median | worst noisy |
|---|---:|---:|---:|---:|
| v55 | — | 5.320e-09 graded | 92.6 graded | 114.5 graded |
| **`tr64`** | +2.14% | **5.4341e-09 graded** | **76.90 graded** | **~95.4 implied** |
| `tr64_wide64` (projection only) | +5.27% | ~5.600e-09 | ~66 | ~82 |
| v56b | +7.28% | 5.7076e-09 graded | 54.5 graded | ~68 |

**Kernel rate, measured on three leaf configurations** — the wall law now has a
mixed-leaf anchor, not just its two endpoints:

| config | leaves | kernel ps/FLOP |
|---|---|---:|
| v55 | all 32 | 176.3 |
| **tr64** | **transports 64, rest 32** | **141.0** |
| v56b | all 64 | 92.8 |

## Re-run log

| date (UTC) | file | submission | day | worst | median |
|---|---|---|---|---:|---:|
| 2026-09-22 | v50 | #331863 | quiet | 88.7 | 84.4 |
| 2026-09-23 03:06 | v50 | #331939 | quiet | 87.7 | 85.0 |
| 2026-09-23 08:27 | v55 | #331967 | noisy | 114.5 | 92.8 |
| 2026-09-23 09:18 | v55 | #331980 | noisy | 113.8 | 92.5 |
| 2026-09-23 10:13 | v50 | #331989 | noisy | 104.1 | 87.0 |
| 2026-09-23 19:56 | v56b | #332063 | moderate (1.114) | 60.7 | 54.5 |
| 2026-09-24 00:11 | tr64 | #332088 | quiet (1.044) | 80.30 | 76.90 |
