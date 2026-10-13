# CURRENT_STATE — 2026-10-13 (plan rev 24, Lane P7)

Plan rev 24 = `docs/research/2026-10-12_deep_review.md` §6 (on PLAN rev 18 §5 gates, with Samuel's Oct 12 decisions:
same-shape `out=` allowed; ops gate becomes "projected residual ≤ 0.33 s at v85's measured per-op rate" once v85 is graded,
≤ 26,000 until then). **Lane P6 and Lane M6 are COMPLETE.** Active lanes:
- **P7** (main, this file's owner): `docs/HANDOFF_2026-10-12_LANE_P7.md` — integrate, refit, canary, harden. Builder `tools/build_v85.py`.
- **S** (branch `lane-s`): `docs/HANDOFF_2026-10-12_LANE_S.md` — Strassen v2 (`sb2` → depth → variant C → tables on 64).
- **K** (branch `lane-k`): `docs/HANDOFF_2026-10-12_LANE_K.md` — pruning v2 with analytic gain replacement (K0–K4, K5 optional).
- **O** (branch `lane-m`, follows M6): `docs/HANDOFF_2026-10-12_LANE_O.md` — old tier and spend items.

Best graded A: **v84_age6 2.849755454236018e-9 (#335562)**. B: v83_r5ws 2.9952e-9 (#335461); v83_r7ws → Oct 13 A/B slot.

# (history) CURRENT_STATE — 2026-10-11 (plan rev 23, Lane P6)

Plan rev 23 = `docs/research/2026-10-10_evening_review.md` (§0, §2, §5, §6), on PLAN rev 18 §5 gates.
**Lane P5 and Lane M5 are COMPLETE.** Active: Lane P6 (main) per `docs/HANDOFF_2026-10-11_LANE_P6.md`;
Lane M6 (branch `lane-m`) per `docs/HANDOFF_2026-10-11_LANE_M6.md`. Exact pricing tool:
`docs/research/2026-10-11_p6price/`. **Oct 11 ~03:30 UTC:** lane-m merged (480dc74; ρ reproduces exactly). v82c_r2 (R2 refit, FRESH-A −0.46 % vs r1,
96/100) submitted **#335298**, projected 3.0275e-9; v82c_r2ws (A/B slot) on FRESH-A. **P6.1 kink-active H CLOSED**
(Hfull +65 % unrefit: always-on residual rows + S3 carry the chain; RESULTS § P6) → **v83 = v82c_r2 + fold**,
R3 refit (`fold` vs `ctl`, from R2) on pod `acc7entl4kttqk` in `/workspace/p6/r3`, ETA ≈ 07:00 UTC; build with
`tools/build_v83.py --variant fold`. Next: P6.3 per-shape depth (od12 −1.21 % F on fold) wall A/B
(`tools/wall_ab.py`) + DEV raw check; P6.2 spend items as M6 evidence arrives.

**Oct 12 15:15 UTC:** best graded **v84_age6 2.849755454236018e-9 (#335562)** (fold + age 6, R8 refit; no od16 — age6+od16
fails A wall). B v83_r5ws 2.9952e-9 (#335461); v83_r7ws (1.0052× vs r5ws, all gates) packaged-ready for the Oct 13 A/B slot
(build `experiments/2026-10-12_*`-style release from `estimator_k3_v83_r7ws.py`). Fold refit recipe exhausted. Pod stopped.
Waiting on strategist: two-column H, use of the ~10 s A wall margin, P6.5. Packet: `docs/STRATEGIST_PACKET_2026-10-12.md`.

**Oct 12 ~02:50 UTC:** best graded **v83_r5od16 2.908247724225528e-9 (#335455)**; B v82c_r2ws 3.1018e-9 (#335319),
B candidate v83_r5ws #335461 grading. v83_r7od16 (final fold refit, R7) on FRESH-A vs r5od16. Fold refit recipe
stopped (R6 −0.26 %, R7 −0.22 %). P6.1 H closed; P6.3 od16 shipped (A wall ≈ 101.9 s, 3 % margin). Pod stopped.
Open: P6.2 spend items need M6 twin evidence; two-column-set H needs a strategist go/no-go.

# (history) CURRENT_STATE — 2026-10-10 (plan rev 22, Lane P5)

Plan rev 22 = `docs/research/2026-10-10_pivot_rev22.md` (§3–§6), on top of PLAN rev 18 §5 gates.
Lane P5 (main) per `docs/HANDOFF_2026-10-10_LANE_P5.md`; Lane M5 (branch `lane-m`) per
`docs/HANDOFF_2026-10-10_LANE_M5.md`. **`HANDOFF_2026-10-10_LANE_P4.md` and `..._M4.md` are
SUPERSEDED (never started); do not execute them.** Exact cost model: `docs/research/2026-10-10_flopmodel/`.

**Oct 10 end of day:** A = v82c_r1 #335253 3.0464e-9; B = v82b_bundlews #335250 3.1433e-9. Builder
`tools/build_v82.py` (items 0,1,4,5,7 live; 2,3,6 closed). Two-sided D ships as V8b (columns + ρ d21e);
leg-row / dead-row D3 pruning is closed. Pod `acc7entl4kttqk` RUNNING with R2 (v82b-arm, 2 more epochs) in
`/workspace/p5/r2`; twin on the pod: `cd /workspace/p5/arc && python3 -u tools/twin/twin.py ...`. Next: P5.3.

# (history) CURRENT_STATE — 2026-10-08 (PLAN rev 20, Lane P3)

Rev 20 (`docs/PLAN_rev20_2026-10-08.md`) supersedes rev 19 §1–§4; Lane P3 per
`docs/HANDOFF_2026-10-08_LANE_P3.md`. Oct 10: GO/no-GO on v90 (κ4-core loop). Rev 19 notes below are history.

# (history) CURRENT_STATE — 2026-10-07 (PLAN rev 19, Lane P2)

The plan is `docs/PLAN.md` rev 18 §2/§5/§6 plus `docs/PLAN_rev19_2026-10-07.md` (§1, §3, §4).
Lane P2 (main) per `docs/HANDOFF_2026-10-07_LANE_P2.md`; Lane M2 (branch `lane-m`, worktree
`../arc-whest-lane-m`) per `docs/HANDOFF_2026-10-07_LANE_M2.md`. **Acceptance = FRESH-A paired
vs the incumbent** (`tools/ev.py --panel fresh_a --baseline <incumbent>`); lane-m merged.

## Graded

| role | source / SHA8 | submission | score | F/B |
|---|---|---|---:|---:|
| **best A** | `estimator_k3_v82c_r1.py` / `66f3ee4a` | #335253 | **3.0464210089351763e-9** (raw 1.8454e-8) | 0.16508 |
| A (prev.) | `estimator_k3_v82b_bundle.py` / `a358fb45` | #335227 | 3.0716825837413753e-9 (raw 1.8607e-8) | 0.16508 |
| A (prev.) | `estimator_k3_v82a_block.py` / `b65a70a6` | #335165 | 3.2794636795053282e-9 (raw 1.8833e-8) | 0.17414 |
| A (prev.) | `estimator_k3_v81d_starall2.py` / `b7efb0f5` | #334948 | 3.3200317352723394e-9 (raw 1.8833e-8) | 0.17629 |
| A (prev.) | `estimator_k3_v81c_starall.py` / `71b5e991` | #334926 | 3.364228068774433e-9 (raw 1.9084e-8) | 0.17629 |
| A (prev.) | `estimator_k3_v81b_star2.py` / `09ecbf95` | #334855 | 3.393934944659043e-9 (raw 1.9266e-8) | 0.17616 |
| A (prev.) | `estimator_k3_v81_star.py` / `376ae1c5` | #334848 | 3.4983524381277624e-9 (raw 1.9859e-8) | 0.17616 |
| A (prev.) | `estimator_k3_v80_hyg.py` / `73b2c19c` | #334659 | 3.6223815250878947e-9 (raw 2.0579e-8) | 0.17602 |
| A (prev.) | `estimator_k3_v79_dress.py` / `2cf3acf4` | #334576 | 3.7025275784494673e-9 (raw 2.0580e-8) | 0.17991 |
| **B** (wall-safe) | `estimator_k3_v82b_bundlews.py` / `7148e22a` | #335250 | **3.1432860023588976e-9** (raw 1.8605e-8) | 0.16895 |
| B (prev.) | `estimator_k3_v81b_star2ws.py` / `512c33ec` | #334918 | 3.4711270541038904e-9 (raw 1.9265e-8) | 0.18018 |
| B (prev.) | `estimator_k3_v80_hygws.py` / `dd82721c` | #334781 | 3.705600144458426e-9 (raw 2.0582e-8) | 0.18004 |
| B (prev.) | `estimator_k3_v78_kashws.py` / `6b3db2a5` | #334545 | 4.249585835999417e-9 (raw 2.2023e-8) | 0.19296 |
| A (prev.) | `estimator_k3_v77_kash.py` / `4ff8018f` | #334487 | 3.959571663492616e-9 | 0.17981 |
| B (prev.) | `estimator_k3_v76_kasws.py` / `fe05c0a4` | #334474 | 4.4952209455824486e-9 | 0.20416 |

No final entry designated (Samuel designates; evidence in `docs/DESIGNATION.md`).
v79's B twin `estimator_k3_v79_dressws.py` (`16379e50`) passes FRESH-A (1.078× vs v78) and WSL; v80's B twin is next.

## What v79 is (plain English)

When a layer creates new third-order structure ("births"), our chain used only the simplest
recipe (one covariance edge on each side of a hub neuron). ARC's AUGMENT adds four heavier
corrections. "Dressed legs" folds those corrections into the existing leg itself, so no new
matrix products are needed (+0.06 % FLOPs). After a short refit of the learned gains in the
twin, raw error fell 6.8 % on VAL, 7.4 % on FRESH-A, and 6.5 % on the graded run.

## Lane P2 status

- P2.0 done: rev 19 committed, lane-m merged, v78 graded (new B), v77 FRESH-A 4.032e-9, v77
  census (recent tier 58.1 %), flopscope #275 exposure none.
- **P2.1 dressed legs: shipped as v79** (RESULTS § "P2.1").
- P2.2 gain-mode κ4: KILLED (refit arms +3.5 / +4.3 / +0.01 % vs control).
- P2.1b closed (exact correction hurts in our chain). P2.4: always-on D21 rows and alive-block
  covariance closed (accuracy cliffs); layer-0 + join-via-Strassen + big-product level shipped as
  **v80_hyg (new best 3.6224e-9)**. v80's local wall is +8.6 % vs v79 (noisy-implied ≈ 101 s, A ≤ 105 s).
- Open for the Oct 10 decision: Lane M2's gain-mode E21b (−11.6 % in ARC's chain; in v77 it needs an
  off-diagonal κ4-core channel, ≈ +16–24 % F). Lane M2 is measuring a diagonal-only route.
- B for the Oct 8 A/B slot: v79_dressws gated except wall A/B; a v80 B twin (block 40) is the
  better candidate — build and gate it.

## Traps found today

- A twin job runs from a git snapshot: untracked params/K tables must be passed by absolute
  main-checkout path.
- Extra birth columns must keep lineage widths at multiples of 32, or Strassen loses levels
  (K_H + 153 cost +53 % FLOPs).
- The WSL gate's "settled address space did not grow" check is allocator noise at this margin:
  v77 itself shows +406 MiB over 6 repeats; judge VmPeak and repeat a FAIL.
