# CURRENT_STATE — 2026-10-06 evening (PLAN rev 18, Lane P)

The only plan is `docs/PLAN.md` **rev 18** (Oct 6). Lane P (main; decided-neuron pruning) per
`docs/HANDOFF_2026-10-06_LANE_P.md`; Lane V+M (branch `lane-m`, worktree `../arc-whest-lane-m`)
per `docs/HANDOFF_2026-10-06_LANE_VM.md`. FRESH-A/FRESH-LOCK not delivered yet. Until they are,
acceptance uses DEV-100 + VAL-100, as the handoff allows.

## Graded

| role | source / SHA8 | submission | score | F/B |
|---|---|---|---:|---:|
| **best A** | `estimator_k3_v77_kash.py` / `4ff8018f` | #334487 | **3.959571663492616e-9** (raw 2.2021e-8) | 0.17981 |
| A (prev.) | `estimator_k3_v75_kas.py` / `d4d7885c` | #334464 | 4.1871242928610786e-9 (raw 2.2014e-8) | 0.19020 |
| B | `estimator_k3_v76_kasws.py` / `fe05c0a4` | #334474 | **4.4952209455824486e-9** (raw 2.2018e-8) | 0.20416 |
| previous A | `estimator_k3_v73_gb.py` / `8fc42e77` | #332893 | 4.6198973531323855e-9 | 0.21473 |
| previous B | `estimator_k3_v74_ws.py` / `5aa60322` | #334434 | 4.976093977527745e-9 | 0.23128 |

No final entry designated (Samuel designates; evidence in `docs/DESIGNATION.md`). v75's A wall
margin is a measured-parent projection (≈ 98 s noisy vs 105 s), not a server read.

## What v75 is (plain English)

Neurons whose input is almost always negative ("dead", α < −2) pass almost nothing forward,
so the third-order bookkeeping can skip their rows. v75 keeps a fixed number of the
most-alive rows per layer (R: 1024 → 736 by layer 15) in the expensive transports. It forms
only the kept rows of the D21 readout (D: → 800), and gives no mean correction on the
dropped rows (their corrector feature would be wrong). Built by `tools/build_kas.py` from
v73. Cost ×0.886, raw +2.3 %, score −9.4 % graded.

## Lane P status

- P0 done (v74_ws B). P1 done: `RESULTS.md` § "PLAN rev 18 — Lane P" (mask table, H
  decomposition, corrected D table, frontier, K quantiles).
- **P2 shipped: v75 (R + D rows), v76 (its B), v77 (+ Hd dead birth columns, new best 3.9596e-9).**
  v78 (v77's B, block 40) is gated and waits for the Oct 7 A/B slot. Every role sits at its
  threshold optimum (R 2.0, Dz 2.5, Hd 2.5: each has a cliff just below). Refitting is closed.
- (superseded) Next: v77 = v75 + Hd (drop dead birth columns).
  The twin with fixed K reads raw +2.58 % at modeled F 0.839, about −5 % score vs v75. It
  needs variable lineage widths: P2 pages and hub coefficients kept in groups of
  consecutive equal-width lineages.
- The prescribed S3 refit at lr 3e-4 cannot beat its control on converged params (spike +7 % at
  step 28, back to init by 84). A corrector-only, low-lr refit on the compact chain is
  still untested.

## Traps found today (also in RESULTS)

- Strassen: at `_STRASSEN_MIN_BLOCK = 32` any compact K < 1024 loses the 5th level (K = 992
  costs MORE than 1024). Compact chains need block 20 (A) or 40 (B).
- A gather bills 8 FLOPs per output element. Selection and gathers bill 0 materializations.
- The corrector's `d21e` feature makes two-sided D21 masks look 7–10 % bad. Row-only D plus
  no correction on the dropped rows (Dz) is nearly free.
