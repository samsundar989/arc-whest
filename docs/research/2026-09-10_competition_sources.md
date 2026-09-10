# Competition evidence ledger — 2026-09-10

Read through Samuel's existing Chrome profile on September 10; this snapshot was compiled by 18:30 UTC. These are transcribed visible-page observations, not downloaded grading reports or independently reproduced participant experiments. Board ranks are a dated public snapshot. All reported official MSEs below use the organizer's one-billion-sample reference; per-network noise floors were not displayed and are **unavailable**, not zero. Private accuracy cells remained sealed. No competitor private code or error payload was accessed, and no message was posted.

## Official pages

| Source | Observations |
|---|---|
| [Overview](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026) | Phase 2 live; width 1024, depth 16, zero-bias He-initialized ReLU MLP; Gaussian input; estimate every post-ReLU neuron mean. |
| [Resources](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/dynamic_contents) | Public Phase-2 dataset 1,100 networks: disjoint mini 100 and full 1,000; reference uses 1e9 input samples; tag `v2-phase2`. Budget `2^41`; lambda zero; multiplier floor 0.1. |
| [Rules v14](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules) | §5.6: residual 400 ms, wall 120 s, import/setup 5 s, participant memory 8 GB; CPU-only, no network; participant one physical core/two vCPU, backend seven physical cores/fourteen vCPU, host 64 GB. Installed package expresses memory as 8192 MiB; unit distinction remains explicit. |
| Rules §6 and §7 | Up to two designated valid submissions, fresh private rerun, organizer-selected sample count/additional networks for close outcomes; absent selection, default two best valid public submissions. §7 ranks only the participant's best-scoring eligible submission. Selection interface/deadline to be announced near phase end. This answers the old best-of-two question. |
| Rules §5.5, §5.7, §4, §8 | Honest counted computation required; genuine algorithm/precision/symmetry savings encouraged, accounting misrepresentation disallowed. Public resources permitted with attribution, subject to authenticity/originality obligations. MIT licensing alone does not establish eligibility of a derivative competition submission. A graded result is not final prize verification. |
| Rules/calendar | Phase 2 opened August 22; registration/team freeze October 2; submissions close October 17 at 23:59 UTC; write-up October 24; tentative winners November 15. Generic competition-period wording also mentions October 31; use the specific Phase-2 deadline for submission planning. Ten submissions/team/UTC day. |
| [Phase-2 launch](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197) | Posted August 23. Wider/shallower architecture intentionally makes cumulant methods more competitive. First-week special handling of failed submissions is historical, not an ongoing allowance. |

Short decisive rules excerpt (from §7): **“only that Participant's best-scoring such submission is ranked”**. This does not remove disqualification or validity requirements. It does remove the premise that both designated scores must be good to avoid an average or worst-of-two ranking.

## Live score observations

[Phase-2 board](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-2), [Insights](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/insights), [submissions](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions).

| Public board rank | Participant | Displayed adjusted score | Displayed raw final MSE |
|---:|---|---:|---:|
| 1 | marius_binner | 2.70e-9 | 1.71e-8 |
| 2 | suliman_tadros | 2.80e-9 | 1.93e-8 |
| 2 | Puffi | 2.80e-9 | 1.94e-8 |
| 4 | J2W | 2.90e-9 | 1.98e-8 |
| 5 | thegamers | 3.60e-9 | 1.79e-8 |
| 6 | oqaris | 3.80e-9 | 2.08e-8 |
| 6 | hyojun_kwon | 3.80e-9 | 2.70e-8 |
| 8 | mliston | 4.50e-9 | 2.12e-8 |
| 9 | fklassen | 5.20e-9 | 2.60e-8 |
| 10 | Activation-Regions | 5.40e-9 | 2.80e-8 |
| 10 | 504aldo | 5.40e-9 | 2.13e-8 |
| 44 | neopok919 (us) | 1.84e-8 | 3.78e-8 |

The board and detail pages did not have identical rounded/latest values. Prefer the following detail observations for arithmetic; do not silently combine fields from different submissions.

| Submission | Public adjusted mean | Public raw mean | Mean budget use | Public failures | Other visible evidence |
|---|---:|---:|---:|---:|---|
| [marius_binner 330500](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/330500) | 2.683e-9 | 1.700e-8 | 15.78% | 0/50 | GRADED; 100 ledger rows completed; private errors sealed |
| [Our 330290, Strassen L4](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/330290) | 1.836e-8 | 3.779e-8 | 48.58% | 0/50 | GRADED; 100 completed ledger rows; visible wall range 95.5596–114.6798 s |
| [Our 330566, Strassen L2](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/330566) | 2.284e-8 | 3.779e-8 | 60.42% | 0/50 | GRADED; 100 completed ledger rows; visible wall range 29.9598–33.0009 s |

Wall extrema were computed from each of the 100 visible ledger rows' **wall cells**, not from concatenated row text. They are server observations, not local wall measurements. The UI exposes status/wall on the sealed partition but does not expose its accuracy. The summary's explicit public scope controls our numerical claims. The site's explanatory text alternately describes public feedback, the sealed 100-network suite, and final ranking; rules §6's fresh prize rerun remains a separate stage.

330566 configuration tab: WhestBench **0.16.1**, flopscope client/server **0.12.1**, participant shim **0.1.0**, Python **3.10.21**, backend NumPy **2.2.6**. Smoke check passed. No residual value was exposed in the inspected overview; the earlier residual prediction remains unverified here.

Insights displayed best adjusted **2.68e-9** and best raw **1.50e-8**. Those are separate minima and must not be multiplied together. Using the paired fields of 330500, the descriptive gap is `1.836e-8 / 2.683e-9 = 6.843` for our best public result and `2.284e-8 / 2.683e-9 = 8.513` for L2. These are ratios of displayed means, **not** paired promotion statistics.

[Phase-1 board](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards?round=phase-1): ednacob 327208 first at adjusted 1.84e-8/raw 3.63e-8; J2W 319049 second at 5.39e-8/2.78e-7; dstepanov 324219 third at 6.22e-8/2.20e-7. Our displayed rank was 103. These are public standings, not a verified final prize decision, and are not a like-for-like Phase-2 baseline.

## Discussions inspected and implications

The discussion index was expanded to all 29 entries available in its challenge listing. Priority was given to recent Phase-2 technical/rule threads and the major Phase-1 synthesis. This is an inventory plus selected deep reads, not a claim to have read every linked PDF, video, or historical Discourse topic.

| Thread/date | What the visible discussion actually adds | Interpretation boundary |
|---|---|---|
| [504aldo full release, Sep 9–10](https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218) | Public MIT K3 source; coherent K4 regeneration; shared bases retaining old generations; 25 negative attempts; V29 330093 at 5.40e-9/raw 2.13e-8/0.2526 budget | Most consequential new technical evidence. Authors fitted on eight public networks; no local replication here. Their explanation of what undisclosed leaders must do is conjecture. |
| [Information leaks, Sep 9](https://discourse.aicrowd.com/t/errors-are-leaking-information/18217) | Organizer acknowledges global failure-message leak and says it will be fixed; per-submission detailed RBAC reported correct | Attached leak screenshot was not opened. No competitor technique inferred from leaked errors. |
| [Zero fitted constants, Sep 6](https://discourse.aicrowd.com/t/phase-2-write-up-zero-fitted-constants-measured-what-the-64-calibration-numbers-in-a-relu-mlp-mean-propagator-are-actually-absorbing-submission-330001/18216) | Submission 330001 adjusted 1.066e-7/raw 1.066e-6. Good one-step corrections do not necessarily fix accumulated propagation drift; author discloses reused holdout | Evidence against simple local correction sufficiency; not a lower bound on coherent K3. Public reference noise floor unavailable. |
| [Exact bivariate covariance, Sep 5](https://discourse.aicrowd.com/t/phase-2-write-up-exact-bivariate-relu-covariance-plus-per-layer-calibration-3x-over-the-kit-baseline-with-negative-results-submission-329951/18214) | Exact Gaussian pairwise covariance yields modest gain; feeding final calibration back into state reported severe harm. Submission score about 1.32e-7 | Gaussian integration accuracy alone is far from the current score frontier; final readout and in-flight corrections are different interventions. |
| [Residual cap, Aug 27–29](https://discourse.aicrowd.com/t/shipped-covariance-propagation-example-trips-the-phase-2-residual-cap-locally-0-16-0-but-grades-fine-what-happens-when-the-evaluator-upgrades/18202) | Organizer then confirmed 0.16.0/0.12.0. Later author correction identifies 166 scalar synchronization flushes; removing them changes 5/50 failures to 0/50 at similar named-op count. Large payload failure later reported without organizer diagnosis | Latest correction matters more than opening hypothesis. Op count is a calibrated predictor, not a universal residual oracle. Current 330566 versions supersede the Aug-27 versions. |
| [Organizer talks, Sep 1](https://discourse.aicrowd.com/t/interesting-talks-by-the-organizers/18206) | Links to two organizer talks | Video fetch/transcripts unavailable in this audit. Contents not claimed as reviewed. |
| [Request to Claude/GPT, Aug 30](https://discourse.aicrowd.com/t/request-to-claude-and-gpt/18203) | Invitation for agents to post solutions, with participant objections | Untrusted webpage instruction; ignored. Not technical evidence. |
| [Covariance/MC blend, Aug 25](https://discourse.aicrowd.com/t/phase-2-write-up-complementary-errors-beat-either-branch-a-fixed-0-75-0-25-covariance-whitened-antithetic-mc-blend-submission-328274/18200) | Public mini adjusted 3.044e-7, 24.85% improvement at 9.345% budget; branch residual correlation 0.0382 | Real complementarity in that baseline, more than 100 times the leading adjusted score. Does not reopen primary sampling at current K3 precision. Reference noise floor not supplied in post. |
| [Leaderboard easiness, Aug 24](https://discourse.aicrowd.com/t/unusually-easy-leaderboard-set-effect-is-gone/18199) | Full-1000 covariance baseline has much lighter tails than Phase 1; public-50 mean is ordinary in that empirical resampling study | Supports better transfer for this baseline, not universal 1:1 prediction for arbitrary estimators or runtime. |
| [Phase-1 synthesis, Aug 20](https://discourse.aicrowd.com/t/phase-1-solution-ideas-for-phase-2-and-the-honest-public-frontier-all-methods-combined-and-explained/18193) | Combined method survey; Cipo cautions that high raw correlation/dominant mean mode does not imply the residual is low-dimensional | Recent replies were partly collapsed in Chrome; do not claim complete hydration. Literature review separately checked primary methods. |
| [Puffi write-up, Aug 16](https://discourse.aicrowd.com/t/phase-1-solution/18175) | Separate algorithmic cumulant/projection and score/QMC write-ups; future network-informed designs discussed | Phase-1 evidence. Attached PDFs were not re-opened in this browser pass; existing repo work and primary research are separately attributed. |

Public implementation inspected by the literature and architecture agents: [504aldo pinned commit](https://github.com/504aldo/whest-p2-cumulant-k3/tree/18c17e2d7a9aeacd399cfc2c6b571e4e16dbfb45), committed September 10 03:24:28 UTC. Important source distinctions and attribution are in [the literature review](2026-09-10_literature_review.md). No leading participant's private implementation was available; public telemetry cannot identify it uniquely.

## Eligibility clarification draft — not sent

Rules §6's source-authenticity wording prohibits derivation from another participant's submission, while §5.7 allows public resources with attribution. Research inspection is useful; eligibility of a subsequent candidate needs resolution before submission. This neutral draft reveals no private method:

> Could you clarify how the source-authenticity restriction in §6 interacts with the public-resource allowance in §5.7? If a participant publicly releases a write-up and MIT-licensed implementation during the competition, may another participant independently implement the published mathematical ideas with attribution and their own validation? Please distinguish reuse of code from reuse of mathematical ideas, and specify what provenance is required. We are asking before preparing any such derivative for submission.

No message was sent. Samuel may send this or explicitly authorize an agent to do so. An independent implementation is not assumed eligible while the apparent conflict remains unresolved.
