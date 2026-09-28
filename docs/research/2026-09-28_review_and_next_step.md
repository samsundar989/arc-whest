# Worker review and next architecture decision — September 28, 2026

The worker's improvements are real, but the current architecture is still substantially
behind the leader. This review used three agents for implementation, evidence, and
literature analysis, checked the live competition through Samuel's Chrome, and ran new
bounded experiments. The next experiment is specified in **PLAN rev16**, which remains
the only authoritative plan. No newly tested method in this review improved the best
graded submission.

## What is verified

Our best is [v72, submission332766](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/332766):
**4.620737758949154e-9**, rank36 at inspection, 100 completed and zero failed. The current
[J2W result332700](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/332700)
shows **1.748e-9** on its public result page. The leaderboard's1.70e-9 is a coarser display.

On the same50 public networks the leader wins every comparison. Mean paired leader/ours
ratios are0.37975 for score,0.75376 for prediction error,0.50424 for charged computation.
In everyday terms, it is about25% more accurate while using half the charged arithmetic.
See `2026-09-28_public_pairs_332766_vs_332700.csv` for the rounded source rows and RESULTS
for paired differences/intervals. This is public-board evidence, not the final private rerun.

The v70/v71/v72 package hashes, saved predictions, real-meter figures and Linux memory
evidence agree. v72's server wall median/worst is83.303/86.285seconds. The remaining raw
final MSE is2.15185e-8 against a reference-label noise floor about7.7e-11; label noise is
not the obstacle.

Two review findings needed action. The evaluator could mistakenly accept an old memory
PASS after a failed new run; that is fixed, with13 regression tests passing. The oracle
diagnostic changed a carried-statistic subtraction inconsistently; a corrected experiment
preserves useful missing-information signal but qualifies the previous claimed ceilings.
The deployed v72 gates do not have that inconsistency. All102 portability checks pass.

## What the new experiments establish

| experiment | observation | implication |
|---|---|---|
| Consistent oracle correction, two diagnostic networks | Supplying the missing three-neuron contribution still reduces final error substantially; median34.86% in this small check. | There is meaningful missing information. Reference-label access is not a deployable method. |
| Predicting that error from16 current-state matrix features, eight networks | Cross-network coefficient fits explain only6–8% of deeper error beyond a single rescaling direction. | A larger generic gate/closure fit is not justified by these features. |
| Coherent small tensor representation | It can explain some internal error, but its actual four-network rollout makes final MSE4.71–5.81% worse. The unrestricted orthogonal oracle improves23.24% on those same networks. | Fitting internal tensor error is a poor substitute for optimizing the scored output. The tested direct correction is closed. |
| One explicitly derived omitted covariance interaction | Its small-rank form explains under0.01% of deep D21 error even with an oracle coefficient. | Close this particular Gaussian-triangle candidate before spending time training it. This is not a proof against every omitted interaction. |

The corrected rollouts passed exact baseline/consumer checks and finite-value checks.
The synthetic tensor identities were tested against explicit dense tensors. These are
small diagnostic TRAIN panels, already familiar to the baseline; they do not establish
unseen competition performance. The negative results are reasons to change the design,
not to conceal the result or widen the same model until something looks positive.

## Why the next bet targets cost and the final output together

The new exact cost measurement makes the bottleneck concrete. Recent-history transport
and readout consume **61.4%** of charged computation; older-history operations consume
another **27.1%**. Covariance transport costs7.69%, and the small learned mean corrector
only0.033%. The profile matches the official total exactly, with no missing or double
counted FLOPs. Improving the efficiency of history storage/readout has far more potential
than micro-optimizing the small neural corrector. Full details are in
`experiments/2026-09-28_cost_sites/report.md`.

The score multiplies prediction error by charged arithmetic, until the0.1 floor. At
unchanged accuracy, reaching that floor would give us roughly2.152e-9—still behind the
leader. At unchanged cost, we would need raw error near8.14e-9. We need a better joint
accuracy/cost tradeoff.

Deleting the whole old-history tier saves only28.05% of FLOPs and destroys accuracy.
Even a hypothetical free, perfectly accurate replacement of that tier cannot close the
gap by itself. Recent dense history also needs to become cheaper. The replacement must
learn to preserve the final prediction, rather than treating every internal tensor
entry as equally valuable.

The next proposed student retains three recent exact source records and replaces the
remaining accumulated history with one compact symmetric tensor and its basis. Start
with exactly two capacities,32 and64. Construct its inference computation and verify
the tensor bookkeeping first. Price the whole implementation before long training,
aiming near0.12 of the budget instead of the current0.21473. No labels, expensive
teacher, or per-network fitted coefficients are allowed at inference.

If that cost target is credible, train shared update rules through the actual smaller
estimator. Use the current estimator as an offline teacher to stabilize the replacement,
then optimize against organizer TRAIN final means. Compare with direct projection and
an equal-training incumbent control. Select on validation **score**, allowing a raw-error
increase only when the compute saving more than pays for it. Require evidence for at
least15% score improvement before a longer run; target20–30% for the next substantial
submission. These are thresholds and objectives, not forecasts.

This proposal is different from changing the old model's rank/age knobs or adding the
failed correction. It changes the stored representation and trains it for the actual
output. Its risk is real: a small tensor may not preserve enough information, or a cheap
causal update may not be learnable. The early algebra, cost and short controlled-fit
checks are designed to answer those questions before a long run.

## Research and durable evidence

The [ARC cumulant-propagation paper](https://arxiv.org/html/2605.05179) and its
[implementation TODO](https://github.com/alignment-research-center/mlp_cumulant_propagation/blob/main/TODO.md)
motivate examining omitted connected terms. [Goal-oriented tensor decomposition](https://arxiv.org/abs/2508.11139)
supports including downstream quantities in compression objectives, but does not establish
that this competition admits our proposed cheap representation. No adjacent paper checked
here reports a directly comparable Phase2 competition score. The leaders' internals remain
unknown; their observed accuracy/cost combination establishes what we need to beat.

- `docs/PLAN.md`, rev16: authoritative ordered experiment and submission gates.
- `RESULTS.md`, September28 addendum: graded closeout, review corrections and every new result.
- `docs/research/2026-09-28_worker_evidence_review_codex.md`: detailed submission/code audit.
- `docs/research/2026-09-28_frontier_options_codex.md`: primary-source research and derivations.
- `experiments/2026-09-28_review_arch/REVIEW.md`: tensor identities and corrected oracle experiments.
- `experiments/2026-09-28_review_arch/DISTILL_NEXT.md`: implementation detail subordinate to PLAN.
- `experiments/2026-09-28_ad_features/` and `experiments/2026-09-28_triangle/`: reproducible probe outputs.
