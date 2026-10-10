# DESIGNATION — current A and B (maintained by the worker; designating is Samuel's)

## Recommendation — 2026-10-10 evening (Lane P5; no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (recommended)** | `estimator_k3_v82b_bundle.py` | `a358fb45` | #335227 | **3.0716825837413753e-9** | 1.8607e-8 | 0.16508 | local A/B 0.953 vs v82a (0.934 vs v81d) → below v81d's ~101 s noisy-implied; law 76.5 / 90.0 s | 6,026 MiB |
| **B (recommended)** | `estimator_k3_v82b_bundlews.py` | `7148e22a` | #335250 | **3.1432860023588976e-9** | 1.8605e-8 | 0.16895 | A/B 0.916 vs v81b_ws → ≈ 78 s noisy-implied | 6,017 MiB |

**Update: v82c_r1 graded 3.0464210089351763e-9 (#335253) and is now the recommended A** (same code as v82b, refit
constants only; VmPeak 6,024 MiB; wall = v82b's). Its ws twin `estimator_k3_v82c_r1ws.py` is built but not gated.
**Samuel designates.**

## Recommendation — 2026-10-10 (Lane P5, plan rev 22; no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (recommended)** | `estimator_k3_v81d_starall2.py` | `b7efb0f5` | #334948 | **3.3200317352723394e-9** | 1.883284e-8 | 0.17629 | same code as v81c (A/B 0.95 vs v81b; v81 ≈ v80's ~101 s noisy-implied); server wall max 90.0 s (Oct 10 board read) | 6,144 MiB |
| **B (recommended)** | `estimator_k3_v81d_starall2ws.py` once graded (Oct 10 A/B slot); until then `estimator_k3_v81b_star2ws.py` | `1a586019` / `512c33ec` | — / #334918 | — / 3.4711270541038904e-9 | — / 1.9265e-8 | 0.18030 / 0.18018 | block-40 twins (~85 s noisy-implied) | 6,166 MiB (v81b_ws) |

A = best graded score; B = its wall- and memory-safe twin (Strassen block 40). v82a (#335165,
bit-identical to v81d at −1.22 % F) replaces A here if it grades as projected (3.2795e-9).
Final A/B are decided on FRESH-LOCK (≤ 3 reads) plus wall/residual margins on Oct 16–17.
**Samuel designates.**

## Evidence update — 2026-10-06 night (no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (candidate)** | `estimator_k3_v77_kash.py` | `4ff8018f` | #334487 | **3.959571663492616e-9** | 2.202053984490249e-8 | 0.17981 | local A/B vs v73 = 0.897 → ≈ 74.7 s median, **≈ 92.7 s noisy-implied** (A ≤ 105 s); law 83.5 / 98.1 s | 6,116 MiB |
| **B (graded)** | `estimator_k3_v76_kasws.py` | `fe05c0a4` | #334474 | 4.4952209455824486e-9 | 2.2018e-8 | 0.20416 | ≈ 62.6 s noisy-implied | 4,885 MiB |
| B (built, gated, not submitted) | `estimator_k3_v78_kashws.py` | `6b3db2a5` | — | projected 4.2260e-9 | DEV 2.2733e-8 | 0.19296 | local A/B vs v74_ws = 0.890 → ≈ 57.4 s noisy-implied | 4,687 MiB |

v78 = v77 at Strassen block 40, the matching wall-safe B; queued for the Oct 7 A/B slot. All
walls are measured-parent projections, not server reads. Read the server walls of v77 and
v76/v78 from the result pages before designation.

## Evidence update — 2026-10-06 evening (Lane P P2; no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (candidate)** | `estimator_k3_v75_kas.py` | `d4d7885c` | #334464 | **4.1871242928610786e-9** | 2.201440164384394e-8 | 0.19020 | NEW code path: local A/B vs v73 = 0.951 → ≈ 79.2 s median, **≈ 98.2 s noisy-implied** (A ≤ 105 s); law 87.3 / 102.6 s | 6,388 MiB |
| **B (candidate)** | `estimator_k3_v76_kasws.py` | `fe05c0a4` | #334474 | **4.4952209455824486e-9** | 2.201782972832689e-8 | 0.20416 | local A/B vs v74_ws = 0.971 → ≈ 50.5 s median, **≈ 62.6 s noisy-implied** (B ≤ 95 s) | 4,885 MiB |

v75 = v73 + decided-neuron pruning (`tools/build_kas.py`); v76 = v75 at Strassen block 40. Both
walls are measured-parent projections (local ratio × a parent's measured server median), not
server reads. **v75's A margin is ~7 s on a noisy day**, so read its server wall from the
result page before it is designated. Previous pair (v73 #332893 / v74 #334434) below.

## Evidence update — 2026-10-06 (PLAN rev 18, Lane P P0; no nomination/designation made)

| role | file | sha8 | submission | score | raw | F/B | wall evidence | VmPeak |
|---|---|---|---|---:|---:|---:|---|---|
| **A (candidate)** | `estimator_k3_v73_gb.py` | `8fc42e77` | #332893 | **4.6198973531323855e-9** | 2.1514587515980564e-8 | 0.21473 | projected noisy 103.3 s (v72 measured median 83.30 s + 0.005 s); A policy ≤ 105 s | 6,724 MiB |
| **B (candidate)** | `estimator_k3_v74_ws.py` | `5aa60322` | #334434 | **4.976093977527745e-9** | 2.1515782080427925e-8 | 0.23128 | projected quiet 58.1 s / noisy 68.3 s from the wall law; B policy ≤ 95 s | 5,073 MiB |

v74_ws is v73_gb with `_STRASSEN_MIN_BLOCK = 64` and nothing else. DEV raw 2.227566e-8 vs v73's
2.227946e-8 (−0.017 %); 19,667 ops; residual 0.218 s; packaged estimator = tested source;
validate + validate-package OK. It replaces v68_ws #332544 (5.0664e-9) as the wall-safe B once
graded. Under PLAN rev 18 §2, designation is decided on FRESH-LOCK plus wall/residual margins
(Oct 14–16); this table is evidence only. **Samuel designates.**

## Evidence update — September 28 review (no nomination/designation made)

The best graded source is now **v72_gt**, SHA8 `3b6cc397`, submission **#332766**,
score **4.620737758949154e-9**. Live Chrome confirms100 completed/0failed; measured
wall median83.30295s, worst86.2848s, implied noisy worst103.295658s. Exact source/package
and WSL memory hashes match; VmPeak6702.67MiB. Meter472,202,879,721FLOPs,22,992ops;
projected residual from ops is approximately0.255s, not a new measured server residual.
v70 #332737 and v71 #332753 preceded it at4.665633e-9 and4.636794e-9.

The last graded wall-safe fallback is still **v68_ws #332544**, score5.0664e-9;
it has older accuracy and is not a v72 twin. No refreshed B or final nomination was
performed by this review. PLAN rev16 governs further work and all acceptance gates,
including105s noisy for a new A code path and95s for B. The older standing table below
is retained as historical evidence and is superseded by this update.

Full review and source attribution: `docs/research/2026-09-28_worker_evidence_review_codex.md`
and the September28 `RESULTS.md` addendum.

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

## Standing — 2026-09-27 00:40 UTC (v68_ws graded)

| role | file | sha8 | graded | score | worst wall (median) | implied noisy worst | residual | VmPeak | meets policy |
|---|---|---|---|---:|---|---:|---:|---:|---|
| **A (recommended)** | `estimator_k3_v68_wino.py` | — | #332478 | **4.7039e-09** | **93.86 (84.00)**, 1.117 | 104.2 s | 0.2521 s (1.59x) | 6,736 MiB | **yes, A (≤ 115 s)** — by ~11 s |
| A fallback | `estimator_k3_v67_s3.py` | — | #332440 | **4.7404e-09** | **94.44 (90.74)**, 1.041 | 112.5 s | 0.2566 s (1.56x) | v59 path | **yes, A (≤ 115 s)** — by 2.5 s |
| retired | `estimator_k3_v66_s2t2.py` | `5766053c` | #332409 | **4.7854e-09** | **101.10 (89.24)**, 1.133 | 110.7 s | 0.2635 s (1.52x) | v59 path | **yes, A (≤ 115 s)** — by 4.3 s |
| retired | `estimator_k3_v65_s2t.py` | `29a65bd2` | #332393 | **4.8134e-09** | **94.34 (89.27)**, 1.057 | 110.7 s | 0.2606 s (1.53x) | v59 path | **yes, A (≤ 115 s)** — by 4.3 s |
| retired | `estimator_k3_v64_s2.py` | `e92f2e7e` | #332350 | **4.9124e-09** | **94.47 (91.11)**, 1.037 | 113.0 s | 0.2531 s (1.58x) | v59 path | **yes, A (≤ 115 s)** — by 2 s |
| retired | `estimator_k3_v62_s1.py` | `d1d722db` | #332313 | **4.9347e-09** | **102.61 (90.27)**, 1.137 | 111.9 s | 0.2713 s (1.47x) | v59 path | **yes, A (≤ 115 s)** — by 3.1 s |
| retired | `estimator_k3_v59_corr400.py` | `f419f765` | #332232 | **4.9756e-09** | **94.42 (89.84)**, 1.051 | 111.4 s | 0.2511 s (1.59x) | 6,434 MiB (v57 path) | **yes, A (≤ 115 s)** — by 3.6 s |
| retired | `estimator_k3_v57_corr.py` | `253c7b19` | #332129 | **5.0443e-09** | 92.81 (90.43), 1.026 | 112.1 s | 0.2507 s | 6,434 MiB | yes, A |
| **B (recommended)** | `estimator_k3_v68_ws.py` | `0db9afba` | #332544 | **5.0664e-09** | **55.67 (51.89)**, 1.073 quiet | 64.3 s | 0.2063 s (1.94x) | 5,085 MiB | **yes, A and B** — by 31 s (B) |
| B fallback | `estimator_k3_v64_ws.py` | leaf64 of e92f2e7e | #332364 | **5.2699e-09** | **63.55 (55.27)**, 1.150 noisy | 68.5 s | 0.2252 s (1.78x) | 4,962 MiB | **yes, A and B** — measured on a noisy day |
| retired | `estimator_k3_v60_ws400.py` | `e9344107` | #332223 | **5.3387e-09** | 57.59 (55.73), 1.033 | 69.1 s | 0.2121 s (1.89x) | 4,941 MiB | yes, A and B, by 46 / 26 s |
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
| 2026-09-25 | v62_s1 | #332313 | moderate (1.137) | 102.61 | 90.27 |
| 2026-09-25 22:30 | v64_s2 | #332350 | quiet (1.037) | 94.47 | 91.11 |
| 2026-09-26 00:03 | v64_ws | #332364 | noisy (1.150) | 63.55 | 55.27 |
| 2026-09-26 04:50 | v65_s2t | #332393 | quiet (1.057) | 94.34 | 89.27 |
| 2026-09-26 07:10 | v66_s2t2 | #332409 | moderate (1.133) | 101.10 | 89.24 |
| 2026-09-26 09:20 | v67_s3 | #332440 | quiet (1.041) | 94.44 | 90.74 |
| 2026-09-26 13:20 | v68_wino | #332478 | moderate (1.117) | 93.86 | 84.00 |
