# DESIGNATION — current A and B (maintained by the worker; designating is Samuel's)

*Policy — PLAN rev 13, Samuel's decision 2 (2026-09-25):*
**A** = the best-scoring entry. Strassen leaf 32 is allowed; implied **noisy-day worst ≤ ~115 s**.
**B** = its wall-safe twin (leaf 64), implied **noisy-day worst ≤ 95 s**.
Re-run each designate on one quiet and one noisy day before Oct 10. Never spend
B's wall margin for score. **Agents never designate**; this file only keeps the
evidence current.

"Noisy day" = a run whose worst/median ratio is ≥ 1.15 (Sep 23 after ~08:00 UTC:
1.197–1.235). Quiet runs sit at 1.03–1.055. Implied noisy worst = graded median ×
1.24 (RESULTS.md § Calibration, wall refit).

## Why the pair is a free option — the rule, quoted

Live Rules v14 §7, read in Chrome 2026-09-25 (PLAN rev 13 decision 2):

> *"Where a Participant has more than one submission eligible for score-based prize
> consideration in a Phase, only that Participant's best-scoring such submission is
> ranked, so that each Participant occupies at most one position in that Phase's
> score-based ranking."*

**Fallback reading.** A net that exceeds the 120 s wall (or the 0.4 s residual) is
not a disqualification: it gets the zero-prediction fallback (§5.5), which scores
that net at roughly **0.9** instead of ~5e-9 — ruinous for *that entry's* mean,
but it touches only that entry. The other designate is ranked on its own, and
only the better of the two counts. So A may carry wall risk that B does not;
B exists so that a bad private-run day on A costs us A's edge, not the entry.

## Standing — 2026-09-25 06:40 UTC (v59 graded)

| role | file | sha8 | graded | score | worst wall (median) | implied noisy worst | residual | VmPeak | meets policy |
|---|---|---|---|---:|---|---:|---:|---:|---|
| **A (recommended)** | `estimator_k3_v59_corr400.py` | `f419f765` | #332232 | **4.9756e-09** | **94.42 (89.84)**, 1.051 | 111.4 s | 0.2511 s (1.59x) | 6,434 MiB (v57 path) | **yes, A (≤ 115 s)** — by 3.6 s |
| A fallback | `estimator_k3_v57_corr.py` | `253c7b19` | #332129 | **5.0443e-09** | 92.81 (90.43), 1.026 | 112.1 s | 0.2507 s | 6,434 MiB | yes, A |
| **B (recommended)** | `estimator_k3_v60_ws400.py` | `e9344107` | #332223 | **5.3387e-09** | 57.59 (55.73), 1.033 | 69.1 s | 0.2121 s (1.89x) | 4,941 MiB | yes, A and B, by 46 / 26 s |
| retired | `estimator_k3_v56b_wallsafe.py` | `b401ad75` | #332063 | 5.7076e-09 | 60.7 (54.5), 1.114 | ~68 s | 0.2188 s | 4,908 MiB | dominated by v60_ws400 |
| retired | `...v56c_mix__tr64.py` | `52d83de7` | #332088 | 5.4341e-09 | 80.30 (76.90) | 95.4 s | 0.2413 s | 6,468 MiB | superseded |

v59 is v60_ws400 with `_STRASSEN_MIN_BLOCK = 32` instead of 64 — same corrector
weights, same accuracy (v60's graded raw 2.301125e-08), v57's cost and code path.
Its derived score was 2.301125e-08 × 0.21625525 = 4.9763e-09; **#332232 graded
4.9756e-09** (raw 2.300780e-08), so A is now MEASURED. v57 #332129 stays the
fallback A (same code path, 1.4% worse).

**A's thin margin.** v57/v59's implied noisy worst (~112 s) clears the ~115 s A
bound by about 3 s and the 120 s cliff by about 8 s. That is the accepted risk
of decision 2, and it is exactly why B is v60_ws400 (51 s to the cliff).
The anchor is real: v55 (leaf 32, median 92.8 s, 2.5 s above v57's) read
114.5 / 113.8 s worst on the Sep 23 noisy day, with 0 failures.

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
| 2026-09-25 01:05 | v60_ws400 | #332223 | quiet (1.033) | 57.59 | 55.73 |
| 2026-09-25 04:05 | v59 | #332232 | quiet (1.051) | 94.42 | 89.84 |
