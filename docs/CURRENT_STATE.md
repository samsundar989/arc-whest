# CURRENT_STATE — 2026-09-28 review

The only plan is `docs/PLAN.md` rev17. Start there, then read the September 28 review
addendum in `RESULTS.md`. The original worker completed S2t3 and both factor-gate fits.
The current execution handoff is `docs/HANDOFF_2026-09-28_COMPACT_HISTORY.md`.
Rev17 specifies implementation, algebra/cost screens, a 60-step controlled pilot,
conditional TRAIN900 expansion, failure diagnostics and the complete canary loop.
This is a plan update only: the compact student remains unbuilt/unpriced/untrained;
no new candidate, training result or submission was produced by this handoff turn.

## Graded incumbent and fallback

| role | source / SHA8 | submission | reported score | measured server median / worst |
|---|---|---|---:|---|
| best A | `estimator_k3_v72_gt.py` / `3b6cc397` | #332766 | **4.620737758949154e-9** | 83.303 / 86.285s |
| older wall-safe fallback | `estimator_k3_v68_ws.py` / `0db9afba` | #332544 | 5.0664e-9 | 51.9 / 55.7s |

v72 raw final MSE 2.1518525734620654e-8; reference noise floor approximately 7.7e-11.
100 completed / 0 failed in Chrome; current board rank 36. No final entry designated.
The B fallback is older accuracy; a v72-or-later B has not been graded.

J2W #332700: public score 1.748e-9, raw 1.616e-8, cost 0.1082B. Matched 50 public nets:
leader wins 50/50; mean paired leader/ours score 0.37975, raw 0.75376, FLOPs 0.50424.
Use the detailed result, not the coarse 1.70e-9 board display. Both accuracy and cost need work.

## Worker review

- Actual v70/v71/v72 improvement and packaged hashes/WSL evidence checked. G1bg best
  step112 VAL 2.213009e-8 is available but has no DEV/graded evidence yet.
- Fixed evaluator acceptance of stale or wrong-hash memory PASS evidence; 13 mocked
  regression tests pass, portability 102 checks pass. No old submission invalidated.
- Wall still requires manual measured-parent/code-path evidence under PLAN §5.8;
  evaluator's advisory wall row alone cannot authorize a submission.
- Oracle substitution had inconsistent repeated-slice compensation. Shipped gates
  do not have that defect. Corrected FIT2 diagnostic retains a large AD signal.
- Prior statements proving d8b useless or excluding all gate functions were too strong.

## New research and next action

- `tools/k111_consistent_probe.py`: tiny exact CPU regression and workerq2293 FIT2 done.
  AD intervention median final-MSE reduction 34.86% versus legacy 37.63%; full-input
  replacement 47.75% both. Oracle/TRAIN-net evidence only.
- `tools/k111_feature_probe.py`: workerq2292 smoke and2300 FIT8 done. LONO state-feature
  fits capture only 6–8% of orthogonal late error. Covariance top32 right modes capture
  58–68%, but matching linear left modes only ~3%. This motivates a symmetric quadratic
  core test, not a larger generic gate network.
- Symmetric-core oracle (job2305, FIT4) makes final MSE 4.71–5.81% worse despite
  useful moment-energy fit; unrestricted orthogonal AD oracle helps 23.24%. The
  selected q4/8 Gaussian triangle (job2304, FIT2) explains under 0.01% of deep D21
  orthogonal error. **Both tested correction designs are closed.** No long fits.
- Current exact cost census (job2309) reconciles every FLOP: recent history61.373%,
  old resolved work27.086%, covariance7.690%, mean corrector0.033%. See cost_sites/report.md.
- Next is PLAN B5: one replacement history state
  with analytic symmetric-core folding, age3/q32 and q64 as two cost/capacity points.
  Only a cheap, valid implementation goes to shared final-output training and a
  controlled score comparison. Execute PLAN E0–E6; the earlier DISTILL_NEXT note is
  supporting proposal history, not the execution queue.
- Cost research must include recent dense history: the entire old tier costs only 28.05%
  of FLOPs, so even lossless removal cannot close the present leader gap alone.

Evidence: `docs/research/2026-09-28_worker_evidence_review_codex.md`,
`docs/research/2026-09-28_frontier_options_codex.md`,
`experiments/2026-09-28_review_arch/REVIEW.md`,
`experiments/2026-09-28_ad_features/report.md`, and the paired public CSV in docs/research.

Keep existing checkpoints/logs/tar archives. Other projects share the machine; heavy
research/training uses workerq with a written optimization review. No cleanup or
priority change is part of this review.
