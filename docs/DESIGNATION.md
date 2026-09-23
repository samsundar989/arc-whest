# DESIGNATION — current A and B (maintained by the worker; designating is Samuel's)

*Policy (STRATEGIST_RESPONSE_2026-09-23 §5, pending Samuel's acceptance):*
**A** = the best score whose worst wall is ≤ 105 s on a **noisy-day** run.
**B** = a wall-safe twin whose worst wall is ≤ 95 s on a noisy-day run.
Re-run each designate on one quiet and one noisy day before Oct 15. Never spend
B's wall margin for score. Agents never designate; this file only keeps the
evidence current.

"Noisy day" means a run whose worst/median ratio is ≥ 1.15 (Sep 23 after ~08:00
UTC: 1.197–1.235). Quiet runs sit at 1.03–1.055. See RESULTS.md, wall refit.

## Standing — 2026-09-23 21:00 UTC

| role | file | sha8 | graded rows | score | worst wall per run (median) | residual worst | VmPeak | meets policy? |
|---|---|---|---|---:|---|---:|---:|---|
| **A and B (recommended)** | `estimator_k3_v56b_wallsafe.py` | `b401ad75` | #332063 | **5.7076e-09** | **60.7 (54.5)**, worst/median 1.114 | 0.2188 s | 4,908 MiB | **yes, both**, by 44 s / 34 s; one run only — needs a quiet and a noisy re-run |
| former A | `estimator_k3_v50_safe.py` | `f9cb9439` | #331863, #331939, #331989 | 6.192e-09 | 88.7 (84.4) q · 87.7 (85.0) q · 104.1 (87.0) noisy | 0.2655 s | 5,785 MiB | yes, by 0.9 s — **dominated by v56b** on score and wall |
| best score, not designable | `estimator_k3_v55_p2r4.py` | `aac641b4` | #331967, #331980 | **5.320e-09** | 114.5 (92.8) noisy · 113.8 (92.5) noisy | 0.2805 s | 6,471 MiB | **no** (A bound 105 s; 6 s from the 120 s cliff) |

**The designation gap is now 7.3% of score** (v55 5.320e-09, not designable, vs
v56b 5.708e-09). It can be narrowed by moving only *some* products to 64-wide
leaves. `estimator_k3_v56c_mix.py`, all arms at v55 accuracy on DEV 100
(±0.01%), costs from the f64 twin, wall projected from the measured
v55 → v56b kernel saving (36.4 s per 242 G of depth-5 Strassen FLOPs moved,
≈0.15 s/G) — **projections, not server readings**:

| arm | ΔFLOPs vs v55 | projected score | projected median | projected worst noisy (×1.24) |
|---|---:|---:|---:|---:|
| v55 | — | 5.320e-09 (graded) | 92.6 (graded) | 114.5 (graded) |
| `tr64` (tower transports at leaf 64) | +2.14% | ≈5.434e-09 | ≈82 | ≈101 |
| `tr64_wide64` (+ the wide D21 products) | +5.27% | ≈5.600e-09 | ≈66 | ≈82 |
| v56b (everything at leaf 64) | +7.28% | 5.708e-09 (graded) | 54.5 (graded) | ≈68 |

`tr64` would be an A at ~2% from v55's score with ≈4 s of margin to the 105 s
bound on a noisy day; its canary would fail §5's 3% score gate (it is not a
new best), so it needs Samuel's go-ahead.

## Re-run log

| date (UTC) | file | submission | day | worst | median |
|---|---|---|---|---:|---:|
| 2026-09-22 | v50 | #331863 | quiet | 88.7 | 84.4 |
| 2026-09-23 03:06 | v50 | #331939 | quiet | 87.7 | 85.0 |
| 2026-09-23 08:27 | v55 | #331967 | noisy | 114.5 | 92.8 |
| 2026-09-23 09:18 | v55 | #331980 | noisy | 113.8 | 92.5 |
| 2026-09-23 10:13 | v50 | #331989 | noisy | 104.1 | 87.0 |
| 2026-09-23 19:56 | v56b | #332063 | moderate (1.114) | 60.7 | 54.5 |
