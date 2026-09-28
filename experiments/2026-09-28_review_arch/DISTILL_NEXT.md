# Next representation experiment — September 28, 2026

> **Proposal history.** PLAN rev17 §4 B5 E0–E6 is now the executable queue, including
> corrected post-fold compensation, one fixed history-factor basis, controlled pilot,
> cost-aware admission and submission gates. Where this earlier proposal differs,
> follow PLAN. No experiment result is added by this notice.

This is a proposed bounded experiment, not a validated architecture or a replacement
for `docs/PLAN.md`. The tested covariance-core **AD correction** and selected covariance
triangle are closed on current evidence. The next hypothesis is **history distillation**:
replace old carried state to reduce cost, while fitting the final prediction objective.

## Why this must change the representation

Current v72 bills472.203G FLOPs (0.214733B). Removing all existing old-tier machinery
saves132.442G,28.05%; even a free lossless old tier cannot close the leaderboard gap.
The recent tier must also become cheaper. Existing age/rank/gain sweeps do not test the
proposed compact representation and should not be repeated.

Use exactly two initial configurations: **age3/q32** and **age3/q64**. Keep other
algorithm choices fixed. Do not widen this into a rank/age/basis sweep. The first cost
target is a real whole-chain multiplier **at most0.12B**, with residual/wall/memory gates
retained. This would be substantial progress, not a guarantee of beating the leader.

## Exact sequence

1. **Implement an analytic shipping-state update first.** Keep three recent source
   lineages and one symmetric old core G(q,q,q) with basis U(n,q). Under linear and Wick
   transport, update U by the same linear map/row scaling; G stays fixed. At a fold,
   choose a basis from known current state only, project the existing old core into it,
   and add the joining lineage's core formed directly from its available A_L/M/Z/hub
   factors. The baseline contains no per-network fitted coefficients and no teacher.
   A bounded covariance range finder is one implementable basis choice; its algorithm
   and iteration count must be fixed for both q values and charged to the meter.

2. **Verify the representation, then price it.** On tiny full-rank cases, compare the
   full dense symmetric tensor after transport, Wick scaling, joining, and both slices.
   Preserve repeated-slice compensation using the tensor actually carried. Price each
   complete age3/q32 and age3/q64 candidate with the real grader twin, including joins,
   basis finding, core rotations, readouts, arrays, and operation calls. No cost claim
   comes from a torch timing or a leading-order formula. Drop any configuration that
   cannot plausibly meet the0.12B target before doing substantial fitting.

3. **Measure direct projection before a fitted oracle.** On a small common TRAIN panel,
   compare unchanged v72, zero old history, and the two affordable analytic students.
   Record final-mean drift from the v72 teacher, labeled MSE, and old-tier D3/D21 errors.
   This asks whether a causal state update is already near a useful tradeoff. The target
   is the old **model** tensor, not the missing label-model AD correction from today's
   failed core experiment.

4. **Train a small shared update through the real student.** If the analytic students
   retain useful signal, apply shared gates to their joining-factor contractions using
   available features such as column norms of A/M, A dot M, w2, sb/eb, age, and compressed
   state summaries. Gate parameters are common across networks, with identity start.
   Optimize `mean ||mu_student,16 - mu_v72,16||^2` through the actual compact recurrence;
   add an intermediate-mean stability term only if necessary. Then fine-tune to TRAIN
   ground-truth means and use VAL for selection. Include an equal-training unchanged
   v72 control. Do not optimize individual networks' cores or use teacher suffixes at
   inference. The teacher is offline supervision only.

5. **Use an adjoint core oracle only to diagnose failure.** If the analytic/shared update
   fails, fit a core at one selected late boundary through the actual teacher suffix,
   holding the same known basis fixed and bounding optimization work. This distinguishes
   an inadequate represented space from a poor update rule. A symmetric q32 core has
   5,984 free coefficients, already more than the1,024 final outputs; q64 has45,760.
   Therefore successful per-network final-output fitting can overfit that teacher
   output and is weak evidence of learnability. It cannot promote a candidate. Only a
   cross-network shared update and an actual low-cost free-running student can do so.

6. **Stop or promote based on paired final score and full cost.** Keep only a student
   whose actual accuracy/cost tradeoff improves, with clean contract and resource gates.
   If both q values fail the cost screen, or the affordable analytic/shared learners
   fail accuracy, record the failure and stop this specific representation. An oracle
   result alone must not trigger a broader sweep or a submission.

The final-layer reference noise floor is approximately7.7e-11. TRAIN and VAL have both
been used in the project's development history; neither should be described as untouched
competition evidence. Heavy fitting and substantive diagnostics need a fresh resource
optimization review and workerq submission. No job for this design has been launched.

Current-cost confirmation (job2309): exclusive real-meter recent transport/readout is
61.3728% ofv72, resolved old work27.0861%, covariance7.6904%, mean corrector0.0328%.
The full profile reconciles to472,202,879,721 FLOPs. This replaces the historical recent
percentage and supports targeting both recent and old history. See
`experiments/2026-09-28_cost_sites/report.md`; the student itself is still unbuilt/unpriced.
