# Strategist response — explain the accuracy gap, then preserve the queries

Review of worker commit `38b240d`, 17 September 2026. This supersedes the execution order in the earlier September 17 response. It reviews existing evidence; it is **not an executed experiment contract**. Next execution: [handoff](CLAUDE_CODE_HANDOFF_2026-09-17_slice2.md). The worker completed its bounded experiment list, but did not achieve the major-release objective.

## Decision

Keep the full mean/covariance + factored third-cumulant history + regenerated fourth-cumulant core architecture. It is supported by the ARC work and an independently runnable reference with better accuracy. **We have identified a credible architecture family, not a winning implementation or its accuracy ceiling.**

Do the complete reciprocal nonlinear-program experiment before fitting another recipe. Then test one query-directed compression method whose objective corrects a specific weakness in our implementation. Keep exact P4 savings. Do not spend another cycle adding isolated terms to an unmatched recurrence, sweeping ordinary ranks, or certifying the small canary. Do not divert to the write-up now.

The three packet questions are answered: **(1) reciprocal whole-program attribution first; limited own fitting only conditionally; (2) keep a bounded cost lane, replacing its error metric rather than repeating the grid; (3) no interim canary by default.**

## What survives the review

Independent source, validation and compression reviews, plus a fresh-context Skeptic, were requested. Reviews used source, small JSON/AST calculations and existing raw artifacts; no new truth, estimator benchmark, submission or vault touch was performed by the strategist.

- Job 1642's six source hashes match the files. Its contract commit `3f72a99` predates the job, its receipt executes the queue snapshot, and its measured RAM was below its declaration. All 24 paired networks completed in each arm.
- The published F1–F5 arithmetic reproduces. Q3q improves Q3 by 1.1894x but is still worse than v32. The registered 1.15x base gate **failed** at 1.1492748x. Keep that outcome unchanged.
- The shared-source candidate improves the observed P11 score over v31 by **1.1858519x**. This is useful DESIGN evidence, not independent confirmation after approximately 13 panel reads.
- P4 depth 5 preserves accuracy to the measured reassociation tolerance and saves about 7.3% of counted FLOPs. Its physical deployment checks remain candidate-specific.
- The mini wrapper repair is real; independent rerun: **10/10 wrapper tests passed**. No evaluation framework rebuild is justified.

Raw sources: `experiments/2026-09-17_followupF/{score.json,rows.jsonl,decide.json}`, `experiments/2026-09-17_followup/meter.json`, `experiments/2026-09-17_step3acc/score.json`, and the contracts at the bottom of `experiments/LOG.md`.

## Four corrections that change the next decision

### 1. The complete working nonlinear program still has not been tested on our chain

Canonical executed terms, excluding the redundant `(1,2)` output orientation:

| Program | Oriented terms | Reference terms missing | Nonreference additions |
|---|---:|---:|---:|
| Public v17/v29 | 87 | 0 | 0 |
| Q3q | 57 | 33 | 3 |
| PAIR | 114 | 6 | 33 |
| BUDGET 4 | 141 | 6 | 60 |

Q3q omits reference families including D3-squared and D3-times-D4 and introduces three C-cubed rows. PAIR and B4 omit six Q31/Q13 rows. These results are evidence about those precise finite programs, **not a test of the reference program with all of its interacting inputs**. Earlier raw table counts and these oriented counts use different conventions; do not compare them as if they were the same count. v32's 27 rows equal BUDGET2 after omitting five redundant `(1,2)` rows, not the full 32-row generated table.

Sources: `local_sim/closure_audit/canon.py`, `local_sim/closure/closure_budget3_q.json`, `local_sim/s6_build_closure.py`, the pinned public source, and `experiments/2026-09-17_skeptic/verdict.md` C2. Coefficient agreement is not program equivalence.

### 2. Small relative state differences can explain the entire error gap

`local_sim/closure_audit/same_state.py` normalizes differences by state magnitude. The task measures much smaller prediction errors. On its one live-shape network, the final prediction displacement is 1.1247e-4 relative to the prediction norm. With mean activation 0.4602, its squared displacement is at least approximately **2.68e-9**. That is not negligible at this accuracy.

At layer 14, transported dG differs 16.63%; changing just the nonlinear program on the same v32 boundary changes the outgoing regenerated core by 8.36%. The transport and harmonic core formulas themselves already agree. Their different inputs are the question; another generic "core transport swap" is not a distinct mechanism.

The report compares **v17**, not v29. v29 also has rank-16 D21 feedback, adaptive lambda and nested old-source coordinates. The prior P11 raw advantages were about 1.22x for v17 and 1.29x for v29; 1.545x was v29's adjusted advantage.

The captured O(n-squared) boundary is sufficient for one nonlinear step only. Future propagation requires the complete factored third-order history. Never resume a hybrid using only D3/D21 slices and claim full-state equivalence.

### 3. Our compression weights do not minimize the query error they claim to measure

For frozen left factors, define `U = [LA | LP]`, `J = [A | M]`, `E = U J^T`, and an orthogonal projector `P = Q Q^T`. The exact frozen-query error is

`||E(I-P)||_F^2 = tr((I-P) J U^T U J^T (I-P))`.

The method in `local_sim/s7_build_shared.py:24` replaces `U^T U` by its diagonal. This drops cross-column and A/M interactions. It is a heuristic, not equality and not a general upper bound. If both U columns equal u and J has columns e1/e2, the actual query is rank one in direction e1+e2; the surrogate is isotropic and misses that information.

Weights are also captured before Wick scaling while the join happens afterward, and older weights come from earlier joins. Thus conservative/sqrt/uniform weighting and extra power passes did not exhaust query-aware compression. This **does not invalidate the measured candidate gain**.

A replacement basis from the right singular space of the already computed old-plus-joining query, approximated by `orth(E^T Omega)`, is a concrete next proposal. It can retain cross terms with O(n-squared times rank) sketch work if E is reused. It is **not** a whole-layer guarantee: projecting legs changes LA/LP too, and D3 and future transformations also matter. Full-rank and actual post-projection query tests must precede accuracy claims.

### 4. The score gap and candidate ordering need noise and scale discipline

| P11 adjusted objective | age4/rank256 | age4/rank384 |
|---|---:|---:|
| Observed | 8.74625e-9 | 8.84317e-9 |
| Estimated true-function | 8.23106e-9 | 8.22339e-9 |
| Estimated at official label precision | 8.24834e-9 | 8.24419e-9 |

The observed advantage of rank256 has paired t95 approximately [-9.62e-11, +2.90e-10]. The corrected ordering reverses by an insignificant amount. **Retain both as practically tied controls; do not retroactively select a new winner.** Averaged-label noise is 2.3817e-9. About 15.69% of rank256's observed score difference versus v32 comes from the differing cost attached to label noise. Its estimated official-precision gain is 1.13428x, versus observed 1.14927x. These calculations already exist in the result JSON and belong in the decision table.

The packet divides a public target by cost, then compares that raw threshold with P11 error. On one consistent P11 scale, 2x v31 means score <=5.18588e-9. At q=0.216311 this means raw <=**2.39741e-8**, or another **1.68655x** improvement from the current candidate, not 2x. Naive public anchoring gives 7.38037e-9, or 7.45850e-9 with noise-aware ratios; neither is a validated forecast.

"Cost alone cannot get there" is not proved: at the literal 0.1 floor, the current P11 raw error would score 4.04336e-9. At unchanged v32 error, the same-panel major target requires q about 0.156. These calculations do not establish feasibility; they distinguish a failed grid from a mathematical lower bound. Recent exact lineages create a floor **for the current representation**, not for all algorithms.

## Current competition and the size of the task

Chrome checked on September 17: **62/170**, latest own result still [#331317](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/331317). The new leader [#331393](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/331393) reports public adjusted **2.239e-9**, raw **1.655e-8**, mean compute **13.53%**, with all 100 ledger rows completed. These are displayed public metrics; private errors are sealed. Mean cost times mean error is only an approximation when per-network costs vary.

The [board](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-2) has a 3.8e-9 top-ten boundary and 4.4e-9 at approximately rank 13. The requested 2x release is a major step, not yet a winning score. Matching public v29 alone also is not enough.

Useful planning arithmetic on the public scale: raw 2.1e-8 at q=0.20 yields 4.2e-9; raw 1.7e-8 at q=0.13 yields 2.21e-9. These are conditional targets, not forecasts. Existing age2/rank192 lean machinery prices near q=0.165 but currently has unacceptable accuracy. Recovering accuracy at a genuinely cheaper representation is the opportunity.

The new [September 17 discussion](https://discourse.aicrowd.com/t/phase-2-the-compute-budget-is-not-a-lever-and-the-scoring-rule-is-why-adjusted-mse-x-max-0-1-util/18219) reiterates the 0.1 floor and reports weak sampling results. Its broad title does not imply cost is irrelevant above the floor; our current q=0.216 remains above it. It provides no new leading estimator. Public v29's [technical discussion](https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218) remains a concrete reference, not evidence of the leader's private implementation.

## Literature and what it supports

[Wu et al.](https://arxiv.org/html/2605.05179v2) give the direct cumulant/Hermite and factorized-algorithm basis for this task. Their polynomial-activation asymptotics are not a proof that our finite ReLU closure is accurate or optimal. [Dunlavy et al.](https://arxiv.org/abs/2508.11139) explicitly put quantities of interest into low-rank tensor objectives. [Zahm et al.](https://arxiv.org/abs/1801.07922) show how the chosen output norm affects vector-valued dimension reduction. These support preserving consumed queries instead of merely factor energy. Applying that principle to our carried source legs is a research hypothesis, not a transferred theorem or measured competition result.

There is no external evidence here supporting a restart with a generic neural model, dense tensor compression, plain sampling, or an ever larger closure. The actionable new evidence is specific: incomplete program comparison and a demonstrably incomplete compression objective.

## Execution priorities and scientific stopping rules

1. **Small instrument extension, hours:** keep current scorer and receipts; add prediction sidecars and an explicit noise-aware comparison table. Source/hash/cache identity must include complete execution configuration. No tracker or evaluator rewrite.
2. **Causal accuracy batch:** reciprocal complete-program swap in four chains, plus native v29 as a reference endpoint. Verify unchanged-endpoint parity before opening fresh DESIGN labels. Preserve complete native states. See handoff for conditional follow-up.
3. **Own coherent candidate:** derive the implicated operator families; use our own data for any bounded fitted parameters. Test at most one registered interaction branch. Do not ship the diagnostic reference pruning mask, fitted table or source bytes.
4. **Query-directed storage:** fixed age4/rank256 comparison first, then one predeclared cheaper setting only if actual query retention and economics support it. Price all joins, re-projections, reconstructions and operation overhead. Retain the current controls.
5. **One frozen release challenger:** fresh single-use DECIDE, read-only public CONFIRM, candidate-specific full checks/Linux memory, independent Skeptic, exact-file approval and grading.

All science remains preregistered, including conditional branches, selection rules and own parameter fitting; Samuel's new predictions are recorded before measurement. P11 is retired from new accuracy selection. One fresh reusable DESIGN panel supports the bounded next slice, and a separate untouched DECIDE block supports the final candidate. Do not change past failed gates.

Attribution succeeds when it explains the prediction-error gap reproducibly, even if the transplant loses. Production succeeds only on a separately registered score gate. If both bounded routes fail, preserve the best valid artifact and report the exact remaining blocker. No plan can honestly guarantee a twofold research result; do not manufacture one or spend a small submission merely to finish.
