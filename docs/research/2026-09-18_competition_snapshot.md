# Competition review — September 18, 2026 UTC

Read in Samuel's Chrome during the September 17 evening US Central session. This is a dated observation, not an automatic monitor. All navigation was read-only. No posts, uploads, submissions or designations were made. Scores below are displayed public scores; the server does not expose their Monte Carlo noise floors. No private accuracy was accessed.

## Live standings and latest submission

The refreshed [Phase 2 board](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-2) showed **neopok919 rank 62 of 170**, 15 Phase-2 entries. The authenticated team submission history confirms **#331317 remains our latest submission**, September 16. It shows 21 entries across phases.

| Entry | Displayed adjusted | Displayed raw final MSE | Compute / budget | State |
|---|---:|---:|---:|---|
| [Our v31 #331317](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/331317) | 8.752e-9 | 2.907e-8 | 30.10% | 50/50 public scored; 100 completed in ledger |
| [J2W #331393, board-leading clean reference](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/331393) | 2.239e-9 | 1.655e-8 | 13.53% | 50/50 public, 100 completed |
| [J2W #331435, newer displayed result](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/331435) | 2.220e-9 | 1.648e-8 | 13.47% | 99 completed; one TIME_EXHAUSTED row; inconsistent summary labels |

The last row is **not a clean reliability benchmark**. Its page displayed 50/51 public scored, 1 failed, a separate contradictory “Failures 0 of 51”, and a TIME_EXHAUSTED row for kathleen-carlson. Insights displayed best adjusted 2.22e-9 while the refreshed leaderboard's link still pointed to #331393. This review uses 2.239e-9 as the clean comparison and records the newer visible score separately. We cannot resolve the site's failure/partition presentation from the UI and do not infer new scoring semantics from it.

Other refreshed board rows: marius_binner 2.4e-9, suliman_tadros 2.5e-9, Puffi 2.8e-9. Two tied rank-9 rows at 3.6e-9 fill the top ten; rank11 is 3.8e-9, rank12 4.2e-9, rank13 4.4e-9. Thus halving v31 would reach about rank13 at this snapshot, not first. Rounding and movement prevent a precise promised rank.

#331393 Configuration reports whestbench **0.16.1**, flopscope client/server **0.12.1**, NumPy **2.2.6**, Python **3.10.21**, width1024, depth16. Its ledger's costs vary across MLPs. Do not replace mean(MSE*multiplier) by mean(MSE)*mean(multiplier) in an actual evaluation. Ratios of these displayed means are useful gap descriptions only, not paired promotion evidence.

No source-code tab or download for the leading entry was exposed; Overview and Configuration were available. We can inspect the public implementations released by their authors, but cannot identify the leader's architecture from its score and cost alone.

The requested [Phase 1 board](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-1) shows ednacob first at rounded1.84e-8, J2W second5.39e-8, dstepanov third6.22e-8; our rank103. This is historical context. Phase1 used different dimensions and cost rules and does not define the Phase2 frontier or reveal the private prize ranking.

## Competition definition and operational implications

Reviewed the [overview](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026), [rules](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules), [resources](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/dynamic_contents), and [insights](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/insights).

- Fixed-network means under Gaussian inputs, He-Gaussian weights, 1024x16; all-layer output required, final-layer squared error is the primary score.
- Per-network score is MSE*max(0.1,F/2^41), then averaged over networks. Phase2 lambda is zero; residual time is a failure cliff, not a cost term.
- Limits:120s wall/MLP,400ms residual/MLP,5s import/setup; participant process8GB; CPU-only, participant one physical core (2vCPU), backend/harness seven physical cores (14vCPU). The local RTX5090 accelerates research/truth creation, not submitted inference.
- All numerical work through flopscope; pure-Python bookkeeping allowed. Data artifacts allowed. Live resources distinguish participant memory from whole-instance64GB.
- Ten submissions per team per UTC day. This is a maximum, not a scientific target.
- Registration/team freeze October2, submissions close October17 23:59UTC, contribution write-up October24 23:59UTC.
- Final prize ranking uses fresh private reruns of up to two designated entries, with additional networks if required to establish ranking. The two best public valid entries are the default if no designation is made. Explicit designation remains Samuel's action under this repo's contract.
- Public HF Mini100/Full1000 are separate development networks. They are not the leaderboard's visible50 or the fresh final rerun. Similar distribution does not make their realized mean errors equal.

The overview explicitly states fresh private reruns even though submission-detail pages also describe a sealed50-network partition. Keep these distinct. No local forecast can know the exact fresh private draw or server-tail failures.

## Latest and decision-relevant discussions

Read the current [discussion index](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/discussion) and these threads in Chrome:

1. [September17 compute-floor/QMC report](https://discourse.aicrowd.com/t/phase-2-the-compute-budget-is-not-a-lever-and-the-scoring-rule-is-why-adjusted-mse-x-max-0-1-util/18219). Useful evidence that their randomized Sobol sampler's accuracy-cost product stayed nearly flat and that the floor prevents further savings below10%. Its title's broad claim does **not** apply to our30.1% multiplier or the leader's13.53%. Its sampler sweep is above the floor; that flatness is the sampled convergence rate, a different mechanism from the analytic baseline's floor. It provides no reason to abandon our cost work.
2. [504aldo's complete method and dead-end catalogue](https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218). Most relevant public frontier implementation: factored K3 history, cheap regenerated K4, thin D21 feedback, shared older-source coordinates, batched Strassen. September13 reply reports sponsor approval of instrumented Strassen without a recursion-depth limit. Author's method-wide impossibility and guesses about leaders remain interpretations, not proofs. Their own later reply acknowledges other entrants trimmed7–22% beyond their cost anatomy.
3. [Zero fitted constants and accumulated drift](https://discourse.aicrowd.com/t/phase-2-write-up-zero-fitted-constants-measured-what-the-64-calibration-numbers-in-a-relu-mlp-mean-propagator-are-actually-absorbing-submission-330001/18216). Their main lesson is the distinction between one-step error with exact incoming moments and accumulated error under recurrent propagation. Their score is much weaker than ours; this is a diagnostic lesson, not a replacement architecture.
4. [Exact bivariate covariance plus calibration](https://discourse.aicrowd.com/t/phase-2-write-up-exact-bivariate-relu-covariance-plus-per-layer-calibration-3x-over-the-kit-baseline-with-negative-results-submission-329951/18214). Better pairwise Gaussian integration alone gave a small gain; closure dominated. Smoke tests include non-target shapes. No evidence for replacing our higher-order history with this simpler family.
5. [Residual-cap thread](https://discourse.aicrowd.com/t/shipped-covariance-propagation-example-trips-the-phase-2-residual-cap-locally-0-16-0-but-grades-fine-what-happens-when-the-evaluator-upgrades/18202). Later corrections supersede early explanations: materializing scalars with bool/float/int can flush the pending graph and consume residual. Count and payload alone are not sufficient runtime models. An unresolved later batched-array failure shows why official-package tests and realistic lifecycle/memory matter.
6. [Error-message exposure](https://discourse.aicrowd.com/t/errors-are-leaking-information/18217). Organizer acknowledged global-message exposure and said it would be fixed. It is an infrastructure clarification, not research evidence about another entrant's method.
7. [Organizer talks](https://discourse.aicrowd.com/t/interesting-talks-by-the-organizers/18206). Links to conceptual talks; the thread provides no measured new estimator or winning implementation.
8. [Official Phase2 announcement](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197). Organizers explicitly widened/shortened the networks to make cumulant methods more competitive while retaining difficult depth dependence. This supports the architecture family's relevance, not a guarantee that our truncation/compression is optimal.

Claims above summarize observations. Public participant experiments have not been rerun in this review. The literature companion report records newly checked primary source/code findings and counterevidence.
