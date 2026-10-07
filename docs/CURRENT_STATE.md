# CURRENT_STATE — 2026-10-07 (PLAN rev 19, Lane P2)

The plan is `docs/PLAN.md` rev 18 §2/§5/§6 plus `docs/PLAN_rev19_2026-10-07.md` (§1, §3, §4).
Lane P2 (main) per `docs/HANDOFF_2026-10-07_LANE_P2.md`; Lane M2 (branch `lane-m`, worktree
`../arc-whest-lane-m`) per `docs/HANDOFF_2026-10-07_LANE_M2.md`. **Acceptance = FRESH-A paired
vs the incumbent** (`tools/ev.py --panel fresh_a --baseline <incumbent>`); lane-m merged.

## Graded

| role | source / SHA8 | submission | score | F/B |
|---|---|---|---:|---:|
| **best A** | `estimator_k3_v79_dress.py` / `2cf3acf4` | #334576 | **3.7025275784494673e-9** (raw 2.0580e-8) | 0.17991 |
| **B** (wall-safe) | `estimator_k3_v78_kashws.py` / `6b3db2a5` | #334545 | **4.249585835999417e-9** (raw 2.2023e-8) | 0.19296 |
| A (prev.) | `estimator_k3_v77_kash.py` / `4ff8018f` | #334487 | 3.959571663492616e-9 | 0.17981 |
| B (prev.) | `estimator_k3_v76_kasws.py` / `fe05c0a4` | #334474 | 4.4952209455824486e-9 | 0.20416 |

No final entry designated (Samuel designates; evidence in `docs/DESIGNATION.md`).
v79's B twin `estimator_k3_v79_dressws.py` (`16379e50`) is built, not yet gated or submitted.

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
- P2.1b (Lane M2's spurious-term correction as −w2 birth columns): built (`build_dress.py
  --corr`), f = 0.15 costs +14.2 % FLOPs; unrefit twin screen pending.
- P2.2 gain-mode κ4: unrefit β 0.7 blew up (κ4 doubled); chain-matched β refits arm a +3.5 %,
  arm b ≈ +4 % vs control; arm c (blend) pending. Heading for the kill line.
- Next: P2.3 combine whatever survives; P2.4 cost hygiene on the v79 chain.

## Traps found today

- A twin job runs from a git snapshot: untracked params/K tables must be passed by absolute
  main-checkout path.
- Extra birth columns must keep lineage widths at multiples of 32, or Strassen loses levels
  (K_H + 153 cost +53 % FLOPs).
- The WSL gate's "settled address space did not grow" check is allocator noise at this margin:
  v77 itself shows +406 MiB over 6 repeats; judge VmPeak and repeat a FAIL.
