# DESIGNATION — current A and B (maintained by the worker; designating is Samuel's)

*Policy (STRATEGIST_RESPONSE_2026-09-23 §5, pending Samuel's acceptance):*
**A** = the best score whose worst wall is ≤ 105 s on a **noisy-day** run.
**B** = a wall-safe twin whose worst wall is ≤ 95 s on a noisy-day run.
Re-run each designate on one quiet and one noisy day before Oct 15. Never spend
B's wall margin for score. Agents never designate; this file only keeps the
evidence current.

"Noisy day" means a run whose worst/median ratio is ≥ 1.15 (Sep 23 after ~08:00
UTC: 1.197–1.235). Quiet runs sit at 1.03–1.055. See RESULTS.md, wall refit.

## Standing — 2026-09-25 01:30 UTC

**A is now MEASURED, not projected.** `v60_ws400` (#332223) graded **5.3387e-09**
with worst wall **57.59 s** (median 55.73, ratio 1.033), 0 nets over 100 s and 0
failed. Implied noisy worst **69.1 s** against a 68.8 s projection — it meets the
A bound (105 s) by 36 s and the B bound (95 s) by 26 s, with **~51 s of margin to
the 120 s cliff**. It retires `tr64` outright, beating it on BOTH axes
(5.3387 vs 5.4341e-09, and 69.1 vs ~95.4 s).

| role | file | sha8 | graded | score | worst wall (median) | residual | VmPeak | policy |
|---|---|---|---|---:|---|---:|---:|---|
| **A (recommended)** | `estimator_k3_v60_ws400.py` | `e9344107` | #332223 | **5.3387e-09** | **57.59 (55.73)**, 1.033 | 0.2121 s (1.89x) | 4,941 MiB | **yes, A and B**, by 36 / 26 s |
| best score, NOT designable | `estimator_k3_v57_corr.py` | `253c7b19` | #332129 | **5.0443e-09** | 92.81 (90.43), 1.026 | 0.2507 s | 6,434 MiB | **no** — implied noisy 112.1 s |
| **B (fallback)** | `estimator_k3_v56b_wallsafe.py` | `b401ad75` | #332063 | 5.7076e-09 | 60.7 (54.5), 1.114 | 0.2188 s | 4,908 MiB | yes — but **dominated by A** |
| retired | `...v56c_mix__tr64.py` | `52d83de7` | #332088 | 5.4341e-09 | 80.30 (76.90) | 0.2413 s | 6,468 MiB | superseded by A |
| retired | `estimator_k3_v58_ws.py` | — | never sent | 5.3998e-09 proj | ~68.8 s proj | 0.2256 s | 4,920 MiB | superseded by A |

### The one open decision, and it is Samuel's

`estimator_k3_v59_corr400.py` — the SAME corrector as A, at Strassen depth 4's
leaf 32 — would grade **4.9763e-09**, derived from A's own graded raw MSE
(2.301125e-08 x 0.21625525). That is **1.35% better than the best graded** and the
**best score this project has ever produced**. It is blocked twice over:

1. **PLAN §5 needs 3%.** At 1.35% it cannot be canaried under standing approval,
   so it can never acquire the graded row designation requires.
2. **Its wall is v57's.** Implied noisy worst ~112 s, over the A bound.

So the pair on the table today is **A = v60_ws400 (measured, safe, 5.3387e-09)**
and **B = v56b (measured, safest, 5.7076e-09)** — with A dominating B, which means
the second slot is currently buying nothing.

The alternative pair, **A = v59 (4.9763e-09, ~112 s) + B = v60_ws400 (5.3387e-09,
69 s)**, is a genuine hedge: the best score we have, floored by a safe twin. It
needs two things that are not an agent's to give — a graded row for v59 (i.e. a
§5 exception or a lowered gate), and an answer to whether the better of two
designated entries counts or a failure in either taints the result. The Rules v14
text this repository holds says only that "up to two designated submissions are
rerun on fresh MLPs".

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
