# Strategist packet — 2026-10-13 (Lane P7, main)

Plan rev 24 (`docs/research/2026-10-12_deep_review.md` §6), handoff `docs/HANDOFF_2026-10-12_LANE_P7.md`. Lanes S (arc-whest-65),
K (arc-whest-52), O (arc-whest-43) ran in parallel all day; this packet is the integration ledger.

## Ledger vs v84 (graded rows exact; projections from FRESH-A paired ratios on the graded predecessor)

| build | content | F/B | ΔF vs v84 | ops | raw (graded / FRESH-A) | wall (server median / noisy-implied) | residual max | score | strategist pred. | worker pred. |
|---|---|---:|---:|---:|---|---|---:|---:|---|---|
| v84_age6 #335562 | — | 0.17354 | — | 22,113 | 1.642152e-8 | ≈ 82 s / 94.9 s | 0.240 s | **2.8498e-9** | — | — |
| v85_sb2 #335697 | + Strassen v2 glue (bit-identical) | 0.16533 | −4.73 % | 24,576 | = v84 | 67.69 s / ≈ 84 s | 0.311 s | **2.7149e-9** | 2.713e-9 ±0.2 % ✓ | 2.7150e-9 ✓ |
| v86_d16 #335769 | + sbtrim (−1,025 ops) + λ-priced depth, leaf ≥ 16 | 0.16370 | −5.67 % | 23,891 | 1.642486e-8 (+0.02 %) | 75.71 s / 93.9 s | 0.304 s | **2.6887e-9** | 2.57–2.64e-9 (assumed leaf 8) ✗ | 2.689e-9 ✓ |
| v87_B #335809 | + d12c6 depth + K stack B (k0, k1 asym, k2 3.0, k3) + R1 refit | 0.14422 | −16.9 % | 24,371 | FRESH-A +1.01 % vs v86 | proj. ≈ 75 s / 93.6 s | proj. 0.308 s | **≈ 2.393e-9** (grading) | 2.25–2.40e-9 | 2.40e-9 |
| B: v83_r7ws #335688 | (A/B slot) | 0.16941 | | 20,512 | 1.75955e-8 | ≈ 78 s | | **2.9808e-9** | — | 2.9754e-9 |
| B cand.: v85_trimws | age6 + sb2 + sbtrim, block 40 | 0.16943 | | 21,361 | FRESH-A 1.0614× vs r7ws, 100/100 | wall A/B pending | 0.270 s proj. | proj. 2.796e-9 | — | 2.77–2.80e-9 ✓ |

## What was learned (measured)
1. **Pooled `out=` glue raises per-call residual +16 %** (12.64 µs/op worst on v85 vs 10.85 on v84). The ops cap is therefore
   ≈ 26,100 at the 0.33 s gate, not the higher number the review hoped for. Lane S's probe: every call costs 12–13 µs
   locally regardless of type (`empty` = a matmul) ⇒ call count is the lever ⇒ `sbtrim` (−1,025 ops, bit-identical).
2. **Server wall per local A/B is ×1.042 worse for deeper recursion** (v86 server/v85 = 1.118 vs local 1.073; all kernel
   time). Rule now: noisy-implied = 67.69 s × A/B(vs sb2) × 1.042 × 1.24. **Wall is the binding constraint**, not FLOPs or ops:
   depth8 on stack B projects 105.5 s (fails), d12c6 93.6 s.
3. **Lane K pruning refits well:** stack B +1.77 % raw vs matched ctl for −13.8 % F; stack A −0.06 % for −11.2 %.
4. **Lane O closed O1–O3** (SVD join r224 +2.5 %; split legs ≤ +0.62 % net; no old tier ×4–×16). The old-tier stretch item
   is gone for this week.
5. **S4 (dims on ×64) does not pay in wall**: ≤ −1.54 % F for ≈ 13 s noisy-implied (≈ 8.2 s per 1 % F at depth 6), against
   v87's ≈ 11 s headroom. Lane S stopped its build loop; the remaining lever (overhead of ~117k tiny batched GEMMs at depth 6)
   is research.
6. Infrastructure: both RunPod pods vanished overnight (re-created, provisioned, stopped); workerq now carries every heavy job
   (Samuel); two queue traps recorded in `docs/GOTCHAS.md` (head-of-line blocking, 250-repeat WSL control).

## Decisions needed (strategist / Samuel)
1. **How to spend v87's ≈ 11 s of A wall headroom** (93.6 of 105 s): (a) keep as rerun safety (the private rerun runs each
   MLP once; leaders hug caps); (b) depth8 on stack B (−1.2 % F, 105.5 s — fails as is); (c) more K pruning (K4 last-layer
   transport, K5 adaptive counts — wall-cheap, FLOP-saving); (d) a spend item (age 7) — costs wall. Worker's default: (a)+(c).
2. **B designation candidate** for the Oct 14 A/B slot: v85_trimws (proj. 2.796e-9, gates done except wall A/B) vs waiting for
   the v87 B twin (Lane S building: depth λ 1.5e6 ⇒ 23,241 ops, + sbfree; VmPeak@250 in progress). Worker's default: submit
   whichever has all B gates first on Oct 14; the v87 twin should land ≈ 2.4e-9.
3. FRESH-LOCK reads (≤ 3, designation only) — plan for Oct 16 on the A and B finalists.

## Next (Lane P)
v87 grading → telemetry (wall/residual) → recalibrate the ×1.042 factor. B: v87 twin gates + FRESH-A, Oct 14 A/B slot.
K: K4/K5 screens (Lane K). v88 = whichever of K4/K5 passes + any refit polish.

## Addendum (late Oct 13) — memory grows per prediction inside flopscope (Lane S, lane-s `experiments/2026-10-13_s/probe_opgrowth.*`)
- Settled VmSize grows **≈ +14 MB per predict in one process**, every build incl. pre-sb2 (v84 +11.5, r7ws +10.5, r7od16 +13.5).
  Cause: flopscope's module-level `_accumulator` (`_budget.py:2266`) keeps every exited BudgetContext's op log (~404 B/op call,
  ≈ 9.4 MB of the 14.2 at ~23k ops). The estimator must not call `budget_reset()` (it also rebases the grader's live meter).
- v87's B twin hit the 8 GiB cap at predict 244/250 ⇒ PLAN's B gate "VmPeak ≤ 7,800 MiB at 250 predicts" is unmeetable at this size.
- **Rerun risk (decision-relevant):** whestbench's subprocess_worker loops over stdin and never resets the accumulator, so a worker
  that serves N MLPs carries ≈ N × 14 MB. 100/100 graded runs pass today; if the private rerun pushes many more MLPs through one
  worker, every ~23k-op entry (ours and the leaders') nears the cap around N ≈ 180–200. Fewer ops per predict is our only lever.
- Interim B memory gate (worker, conservative): VmPeak at 100 predicts ≤ 7,200 MiB (12 % margin) **and** ≈ 0 MB/predict
  candidate retention measured with the accumulator cleared in the harness only.
**Decision for Samuel:** (1) adopt the interim gate (or another); (2) whether to ask the organizers how many MLPs a rerun worker
serves — agents may not contact them (AGENTS rule 3); Rules §5.5 lets a participant rely on a written answer.
