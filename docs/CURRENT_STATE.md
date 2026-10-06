# CURRENT_STATE — 2026-10-06 (PLAN rev 18, Lane P)

The only plan is `docs/PLAN.md` **rev 18** (Oct 6). Two lanes run in parallel:
Lane P (this file's owner; main, decided-neuron pruning, cost) per
`docs/HANDOFF_2026-10-06_LANE_P.md`, and Lane V+M (branch `lane-m`, worktree
`../arc-whest-lane-m`; fresh-network panels, then the dropped-diagram measurement) per
`docs/HANDOFF_2026-10-06_LANE_VM.md`. Everything before rev 18 is history.

## Graded incumbent and insurance

| role | source / SHA8 | submission | score |
|---|---|---|---:|
| best A | `estimator_k3_v73_gb.py` / `8fc42e77` | #332893 | **4.6198973531323855e-9** (raw 2.1515e-8, F/B 0.21473) |
| new B (P0) | `estimator_k3_v74_ws.py` / `5aa60322` | #334434 | **4.976093977527745e-9** (raw 2.1516e-8, F/B 0.23128) |
| old B | `estimator_k3_v68_ws.py` / `0db9afba` | #332544 | 5.0664e-9 |

No final entry designated (Samuel designates; `docs/DESIGNATION.md` keeps the evidence).

## Lane P progress

- **P0 done** — v74_ws built, every PLAN §5 hard gate evidenced, submitted on the A/B slot.
- **P1 mask table done, refit running.** `RESULTS.md` § "PLAN rev 18 — Lane P". Key facts:
  dead birth columns can be dropped whole for +0.03 % (τ 3.0) / +0.15 % (τ 2.5) raw — the
  literal H rule's +2 % was the κ4 hubs on *always-on* columns; D21's cost is the mean
  corrector's `d21e` feature (+0.03 % with it unmasked), which a refit must absorb; R2.0
  costs +0.9 %, R2.5 +0.06 %. Modeled cost of the full sets: F/B ≈ 0.17–0.18.
  Refits (workerq 2404 safe-d R2.0/Hd3.0/D3.0, 2405 aggr-d R2.0/Hd2.5/D2.5, 2406 unmasked
  control) write to `experiments/2026-10-06_mask/refit_*`. R decomposition (Ri/Rn) on CPU,
  job 2407.
- **P2 prerequisite checked**: real flopscope selects and gathers with a device-side index at
  zero materializations; a gather bills 8 FLOPs per output element (not 4); scatter-back is
  a pad + double `take` (`experiments/2026-10-06_mask/gather_probe.json`).

## Tools added for Lane P

- `tools/twin/twin.py --arm mask --mask R=..,H=..|Hd=..|Hk=..,D=..|Dc=..` (eval and train),
  `--mask-stats` (per-layer α kept fractions). Inert without `--mask`; bit-identical control.
- `tools/twin/mask_grid.py` — resumable VAL grid driver (`--cpu` for a CPU job).
- `tools/mask_flops.py` — modeled FLOP factor from the job2309 per-site census.

## Next

P1 kill/go after the refits (safe-d > +3 % raw vs the refit control ⇒ close the lane, go to
P3). If go: P2 compact `estimator_k3_v75_kas.py` with dead-only H (Hd), per-layer fixed keep
counts, gathers via `fnp.take`, and the corrector refit on the compact chain.
