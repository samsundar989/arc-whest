# DESIGNATION — current A and B (maintained by the worker; designating is Samuel's)

*Policy (STRATEGIST_RESPONSE_2026-09-23 §5, pending Samuel's acceptance):*
**A** = the best score whose worst wall is ≤ 105 s on a **noisy-day** run.
**B** = a wall-safe twin whose worst wall is ≤ 95 s on a noisy-day run.
Re-run each designate on one quiet and one noisy day before Oct 15. Never spend
B's wall margin for score. Agents never designate; this file only keeps the
evidence current.

"Noisy day" means a run whose worst/median ratio is ≥ 1.15 (Sep 23 after ~08:00
UTC: 1.197–1.235). Quiet runs sit at 1.03–1.055. See RESULTS.md, wall refit.

## Standing — 2026-09-24 16:30 UTC

**The best score moved, and it is NOT designable.** v57 (#332129) grades
**5.0443e-09** with an implied noisy-day worst of 90.43 x 1.24 = **112.1 s**,
over A's 105 s bound and 8 s from the cliff. A therefore stays `tr64` until the
wall-safe twin of v57 exists.

**The designable follow-up is already proven in principle:** v57 at Strassen
depth 4 is the same transform v56b applied to v55 (+7.28% multiplier, raw
IDENTICAL, kernel 176 -> 93 ps/FLOP). Applied to v57 it projects
**~5.41e-09 at ~68 s**, which would be a better A than tr64 on BOTH axes --
score (5.41 vs 5.4341e-09) and wall (~68 vs 80.3 s). Building and canarying it
is the highest-value remaining designation action.

| role | file | sha8 | graded rows | score | worst wall per run (median) | residual worst | VmPeak | meets policy? |
|---|---|---|---|---:|---|---:|---:|---|
| **best score, NOT designable** | `estimator_k3_v57_corr.py` | `253c7b19` | #332129 | **5.0443e-09** | 92.81 (90.43) quiet, ratio 1.026 | 0.2507 s | 6,434 MiB | **no** — implied noisy 112.1 s > 105 |
| **A (current)** | `...v56c_mix__tr64.py` | `52d83de7` | #332088 | 5.4341e-09 | 80.30 (76.90) quiet, ratio 1.044 | 0.2413 s | 6,468 MiB | **yes, A** (implied noisy ~95.4 s) |
| **B (current)** | `estimator_k3_v56b_wallsafe.py` | `b401ad75` | #332063 | 5.7076e-09 | 60.7 (54.5), ratio 1.114 | 0.2188 s | 4,908 MiB | **yes, both** |
| dominated | `estimator_k3_v50_safe.py` | `f9cb9439` | #331863/939/989 | 6.192e-09 | 88.7 q / 87.7 q / 104.1 noisy | 0.2655 s | 5,785 MiB | dominated |
| superseded | `estimator_k3_v55_p2r4.py` | `aac641b4` | #331967/980 | 5.320e-09 | 114.5 / 113.8 noisy | 0.2805 s | 6,471 MiB | no |

**Still outstanding for BOTH designates: a genuine noisy-day reading.** Every row
above was measured on a quiet or moderate server (ratios 1.026-1.114); the noisy
figures are the x1.24 projection, not readings. Sep 23's noise appeared after
~08:00 UTC, and today's 08:30 UTC v57 run came back quiet (1.026), so the tail is
not a fixed time of day. Spend a slot on an A/B re-run whenever a same-day row
shows a ratio >= 1.15.

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
| 2026-09-24 08:30 | v57 | #332129 | quiet (1.026) | 92.81 | 90.43 |
